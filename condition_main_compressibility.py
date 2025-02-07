import torch
from PIL import Image
import sys
import os
import copy
cwd = os.getcwd()
sys.path.append(cwd)

from compressibility_scorer import CompressibilityScorer,condition_CompressibilityScorerDiff_4class
from compressibility_scorer import classify_compressibility_scores_4class
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime

from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags

from diffusers_patch.ddim_with_kl import ddim_step_KL
from diffusers_patch.utils import compute_classification_metrics


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/condition.py:compressibility", "Training configuration.")

from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.state import AcceleratorState

logger = get_logger(__name__)

def condition_compressibility_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     config=None,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 512
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = condition_CompressibilityScorerDiff_4class(dtype=torch_dtype,config=config).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    
    for param in scorer.parameters():
        assert not param.requires_grad, "Scorer should not have any trainable parameters"

    def loss_fn(im_pix_un, class_labels):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        
        assert im_pix.shape[2] == 512 and im_pix.shape[3] == 512, "Image size should be 512x512"
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        
        normalized_im_pix = normalize(im_pix).to(im_pix_un.dtype)
        probabilities, _ = scorer(normalized_im_pix, config)
        selected_probs = probabilities[torch.arange(probabilities.size(0)), class_labels]
        rewards = torch.log(selected_probs + 1e-6)
        nll_loss = -1 * rewards # Computing negative log likelihood loss

        return nll_loss * grad_scale, rewards, im_pix
    return loss_fn

def evaluate_compressibility_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 512
    scorer = CompressibilityScorer(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        assert im_pix.shape[2] == 512 and im_pix.shape[3] == 512, "Image size should be 512x512"
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        
        rewards, images = scorer(im_pix)
        loss = -1 * rewards
        return loss * grad_scale, rewards, images
    return loss_fn


def evaluate(latent, train_neg_prompt_embeds, prompts, train_neg_labels, class_labels, pipeline, accelerator, inference_dtype, config, loss_fn):
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)       
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]        
    
    all_rgbs_t = []
    for i, t in tqdm(
        enumerate(pipeline.scheduler.timesteps), 
        total=len(pipeline.scheduler.timesteps),
        disable=not accelerator.is_local_main_process
        ):
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = pipeline.unet(latent, t, train_neg_prompt_embeds, class_labels=train_neg_labels).sample
        noise_pred_cond = pipeline.unet(latent, t, prompt_embeds, class_labels=class_labels).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(
            noise_pred, 
            t[0].long(), 
            latent, 
            eta=config.sample_eta
            ).prev_sample
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
    
    if "hps" in config.reward_fn:
        loss, rewards = loss_fn(ims, prompts)
    else:    
        _, rewards, embeds = loss_fn(ims)
    return ims, rewards, embeds

    
    

def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    
    if accelerator.is_main_process:
        wandb_args = {
            "name": config.run_name,
        }
        if config.debug:
            wandb_args.update({'mode':"disabled"})        
        accelerator.init_trackers(
            project_name="CTRL-compressibility", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, config.run_name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

    
    logger.info(f"\n{config}")

    # Original set_seed function (device_specific is very important to get different prompts on different devices)
    # set_seed(config.seed, device_specific=True)
    def torch_set_seed(seed: int = 42, device_specific = False) -> None: # Custom set_seed to improve reproducibility
        if device_specific:
            seed += AcceleratorState().process_index
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, the following are needed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    torch_set_seed(config.seed, device_specific=True)
    
    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    unet_pretrained = copy.deepcopy(pipeline.unet)
    for param in unet_pretrained.parameters():
        param.requires_grad = False
    
    # disable safety checker
    pipeline.safety_checker = None    
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps)

    # Set up class embeddings for additional conditioning
    num_classes = len(list(config.class_label_dist))  # Example: Number of distinct classes, 2
    pipeline.unet.config.class_embed_type = None  # Use naive embeddings or timestep embeddings
    pipeline.unet.class_embedding = torch.nn.Embedding(num_classes+1, embedding_dim=1280) # Set up the class embedding layer
    
    # Initialized to zero for keeping KL divergence low at the beginning
    torch.nn.init.zeros_(pipeline.unet.class_embedding.weight)
    
    # Initialize valid class embeddings with Gaussian
    # torch.nn.init.normal_(pipeline.unet.class_embedding.weight[:-1], mean=0, std=1 / torch.sqrt(torch.tensor(1280).float()))
    # torch.nn.init.zeros_(pipeline.unet.class_embedding.weight[-1])
    
    assert pipeline.unet.class_embedding.weight.requires_grad, "Class embeddings not added correctly"
   

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    inference_dtype = torch.float32
    
    pipeline.unet.class_embedding.to(accelerator.device)
    
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    unet_pretrained.to(accelerator.device, dtype=inference_dtype)
        
    # Set correct lora layers
    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    pipeline.unet.set_attn_processor(lora_attn_procs)

    # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
    # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
    # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipeline.unet(*args, **kwargs)

    unet = _Wrapper(pipeline.unet.attn_processors)        

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        # assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
            torch.save(pipeline.unet.class_embedding.state_dict(), os.path.join(output_dir, "class_embedding.pth"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.soup_inference:
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            if config.resume_from_2 != "stablediffusion":
                tmp_unet_2 = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
                )
                tmp_unet_2.load_attn_procs(config.resume_from_2)
                
                attn_state_dict_2 = AttnProcsLayers(tmp_unet_2.attn_processors).state_dict()
                
            attn_state_dict = AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            if config.resume_from_2 == "stablediffusion":
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1
            else:
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1 + attn_state_dict_2[attn_state_key]*(1.0 - config.mixing_coef_1)
            
            models[0].load_state_dict(attn_state_dict)
                    
            del tmp_unet                
            
            if config.resume_from_2 != "stablediffusion":
                del tmp_unet_2
                
        elif isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            
            class_embedding_path = os.path.join(input_dir, "class_embedding.pth")
            if os.path.exists(class_embedding_path):
                class_embedding_state_dict = torch.load(class_embedding_path)
                pipeline.unet.class_embedding.load_state_dict(class_embedding_state_dict)
                print("Loaded class embeddings")
                
            del tmp_unet
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)    

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer
    assert pipeline.unet.class_embedding.weight.requires_grad, "Class embeddings not added correctly"
    
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [
            {'params': unet.parameters(), 'lr':config.train.learning_rate},
            {'params': pipeline.unet.class_embedding.parameters(), 'lr':config.train.embed_learning_rate}
        ],
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_fn = getattr(prompts_file, config.prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)
    
    train_neg_labels = torch.ones(config.train.batch_size_per_gpu_available, dtype=torch.long).to(accelerator.device)
    train_neg_labels = train_neg_labels * num_classes # Set to the NULL embedding (last class)

    autocast = contextlib.nullcontext
    
    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)
    
    if config.reward_fn=='compressibility':
        loss_fn = condition_compressibility_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    config=config,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
        eval_loss_fn = evaluate_compressibility_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
    else:
        raise NotImplementedError

    keep_input = True
    timesteps = pipeline.scheduler.timesteps #[981, 961, 941, 921,]
    
    eval_prompts, eval_prompt_metadata = zip(
        *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
    )    
    
    with open('assets/simple_animals.txt', "r") as f:
        lines = [line.strip() for line in f.readlines()]
    Div_prompts = lines[:config.num_samples_Div]
    # print(Div_prompts)
    
    config.eval_div_freq = 99999 # Disable CLIP similairity calculation

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0 
       
    global_step = 0

    if config.only_eval:
        #################### EVALUATION ONLY ####################                

        all_eval_images = []
        all_eval_rewards = []
        info = defaultdict(list)
        
        total_samples = config.train.batch_size_per_gpu_available * config.max_vis_images
                       
        samples_per_class = [int(total_samples * dist) for dist in list(config.class_label_dist)]
        samples_per_class[-1] = total_samples - sum(samples_per_class[:-1])
        eval_labels = torch.cat([torch.full((count,), i, device=accelerator.device, dtype=torch.long) for i, count in enumerate(samples_per_class)])
        
        if config.same_evaluation:
            generator = torch.cuda.manual_seed(config.seed)
            latent = torch.randn((total_samples, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)
        else:
            latent = torch.randn((total_samples, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)
                            
        with torch.no_grad():
            for index in range(config.max_vis_images):
                ims, rewards, _ = evaluate(
                    latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],
                    train_neg_prompt_embeds, 
                    eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], 
                    train_neg_labels,
                    eval_labels[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],
                    pipeline, 
                    accelerator, 
                    inference_dtype,
                    config, 
                    eval_loss_fn)
                all_eval_images.append(ims)
                all_eval_rewards.append(rewards)
        
        eval_rewards = torch.cat(all_eval_rewards)
        eval_reward_mean = eval_rewards.mean()
        print("Evaluation results", eval_reward_mean)
        eval_images = torch.cat(all_eval_images)
        
        # Calculate class-wise mean and std
        class_0_rewards = eval_rewards[eval_labels == 0]
        class_1_rewards = eval_rewards[eval_labels == 1]
        class_2_rewards = eval_rewards[eval_labels == 2]
        class_3_rewards = eval_rewards[eval_labels == 3]
        
        class_0_mean = class_0_rewards.mean() if len(class_0_rewards) > 0 else 0
        class_0_std = class_0_rewards.std() if len(class_0_rewards) > 0 else 0
        class_1_mean = class_1_rewards.mean() if len(class_1_rewards) > 0 else 0
        class_1_std = class_1_rewards.std() if len(class_1_rewards) > 0 else 0
        
        class_2_mean = class_2_rewards.mean() if len(class_2_rewards) > 0 else 0
        class_2_std = class_2_rewards.std() if len(class_2_rewards) > 0 else 0
        class_3_mean = class_3_rewards.mean() if len(class_3_rewards) > 0 else 0
        class_3_std = class_3_rewards.std() if len(class_3_rewards) > 0 else 0
        
        # Calculate classification metrics
        predicted_classes = classify_compressibility_scores_4class(eval_rewards)
        metrics = compute_classification_metrics(predicted_classes, eval_labels)
        
        
        eval_image_vis = []
        if accelerator.is_main_process:

            if config.run_name != "":
                name_val = config.run_name
            else:
                name_val = wandb.run.name            
            log_dir = f"logs/{name_val}/eval_vis"
            os.makedirs(log_dir, exist_ok=True)
            
            np.save(f"logs/{name_val}/comp_predictions.npy", predicted_classes.cpu().numpy())
            np.save(f"logs/{name_val}/comp_labels.npy", eval_labels.cpu().numpy())
            
            for i, eval_image in enumerate(eval_images):
                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                
                prompt = eval_prompts[i]
                label = eval_labels[i]
                reward = eval_rewards[i]
                predicted_class = predicted_classes[i]
                
                pil.save(f"{log_dir}/{i:03d}_{prompt}_{reward:.2f}_class={predicted_class}_label={label}.png")
                
                pil = pil.resize((256, 256))
                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f} (class:{predicted_class}) | true class:{label}"))          
            
            accelerator.log({"eval_images": eval_image_vis},step=global_step)   
            
            info.update({"eval_rewards_mean":eval_reward_mean,
                        "eval_class_0_rewards_mean":class_0_mean,
                        "eval_class_1_rewards_mean":class_1_mean,
                        "eval_class_2_rewards_mean":class_2_mean,
                        "eval_class_3_rewards_mean":class_3_mean,   
                        "eval_accuracy":metrics['accuracy'],
                        "eval_macro_F1": metrics['macro_F1'],
                        "eval_macro_precision": metrics['macro_precision'],
                        "eval_macro_recall": metrics['macro_recall'],
                    })
                    
            accelerator.log(info, step=global_step)     
    else:
        #################### TRAINING ####################        
        for epoch in list(range(first_epoch, config.num_epochs)):
            unet.train()
            info = defaultdict(list)
            info_vis = defaultdict(list)
            image_vis_list = []
            
            for inner_iters in tqdm(
                    list(range(config.train.data_loader_iterations)),
                    position=0,
                    disable=not accelerator.is_local_main_process
                ):
                latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64),
                    device=accelerator.device, dtype=inference_dtype)    

                # Probabilities for each class
                probabilities = torch.tensor(list(config.class_label_dist)).repeat(config.train.batch_size_per_gpu_available, 1).to(accelerator.device)
                class_labels = torch.multinomial(probabilities, num_samples=1, replacement=True).squeeze().to(accelerator.device, dtype=torch.long)
                
                if accelerator.is_main_process:
                    logger.info(f"{config.run_name.rsplit('/', 1)[0]} Epoch {epoch}.{inner_iters}: training")

                prompts, prompt_metadata = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )

                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)   

                pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
                
            
                with accelerator.accumulate(unet):
                    with autocast():
                        with torch.enable_grad(): # important b/c don't have on by default in module                        
                            keep_input = True
                            
                            kl_loss = 0
                            
                            for i, t in tqdm(
                                enumerate(timesteps), 
                                total=len(timesteps),
                                disable=not accelerator.is_local_main_process,
                            ):
                                t = torch.tensor([t],
                                        dtype=inference_dtype,
                                        device=latent.device
                                    )
                                t = t.repeat(config.train.batch_size_per_gpu_available)
                                
                                if config.grad_checkpoint:
                                    noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, train_neg_labels, use_reentrant=False).sample
                                    noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, class_labels, use_reentrant=False).sample
                                    
                                    old_noise_pred_uncond = checkpoint.checkpoint(unet_pretrained, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                    old_noise_pred_cond = checkpoint.checkpoint(unet_pretrained, latent, t, prompt_embeds, use_reentrant=False).sample
                                    
                                else:
                                    noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds,class_labels=train_neg_labels).sample
                                    noise_pred_cond = unet(latent, t, prompt_embeds, class_labels=class_labels).sample
                                
                                    old_noise_pred_uncond = unet_pretrained(latent, t, train_neg_prompt_embeds).sample
                                    old_noise_pred_cond = unet_pretrained(latent, t, prompt_embeds).sample    
                                        
                                if config.truncated_backprop:
                                    if config.truncated_backprop_rand:
                                        timestep = random.randint(
                                            config.truncated_backprop_minmax[0],
                                            config.truncated_backprop_minmax[1]
                                        )
                                        if i < timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()
                                    else:
                                        if i < config.trunc_backprop_timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()

                                grad = (noise_pred_cond - noise_pred_uncond)
                                old_grad = (old_noise_pred_cond - old_noise_pred_uncond)
                                
                                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                                old_noise_pred = old_noise_pred_uncond + config.sd_guidance_scale * old_grad 
                                               
                                # latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                
                                latent, kl_terms = ddim_step_KL(
                                    pipeline.scheduler,
                                    noise_pred,   # (2,4,64,64),
                                    old_noise_pred, # (2,4,64,64),
                                    t[0].long(),
                                    latent,
                                    eta=config.sample_eta,  # 1.0
                                )
                                kl_loss += torch.mean(kl_terms).to(inference_dtype)
                                          
                            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample  # latent entries around -5 - +7
                            # ims tensor shape [B,3,512,512], max: 1.22, min -1.39

                            loss, rewards, _ = loss_fn(ims, class_labels)
                            loss = loss.mean() * config.train.loss_coeff
                            
                            total_loss = loss + config.train.kl_weight*kl_loss
                            
                            rewards_mean = rewards.mean()
                            rewards_std = rewards.std()
                            
                            if len(info_vis["image"]) < config.max_vis_images:
                                info_vis["image"].append(ims.clone().detach())
                                info_vis["rewards_img"].append(rewards.clone().detach())
                                info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                            
                            info["loss"].append(total_loss)
                            info["KL-entropy"].append(kl_loss)
                            
                            info["rewards"].append(rewards_mean)
                            info["rewards_std"].append(rewards_std)
                            
                            info["embedding_norm"].append(torch.norm(pipeline.unet.class_embedding.weight[:-1]))
                            info["null_embedding_norm"].append(torch.norm(pipeline.unet.class_embedding.weight[-1]))
                            
                            # backward pass
                            accelerator.backward(total_loss)
                            
                            # Manually zero out gradients for the last class embedding
                            if pipeline.unet.class_embedding.weight.grad is not None:
                                pipeline.unet.class_embedding.weight.grad[-1] = 0
                                    
                            if accelerator.sync_gradients:                                
                                all_params = list(unet.parameters()) + list(pipeline.unet.class_embedding.parameters())
                                accelerator.clip_grad_norm_(all_params, config.train.max_grad_norm) # For LoRA layers + embedding layers
                                
                                # accelerator.clip_grad_norm_(pipeline.unet.class_embedding.parameters(), config.train.max_grad_norm_embeddings)  # For embeddings                      
                                # # Clip gradients for each parameter group separately
                                # accelerator.unscale_gradients(optimizer)
                                # torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                                # torch.nn.utils.clip_grad_norm_(pipeline.unet.class_embedding.parameters(), config.train.max_grad_norm_embeddings)
                                
                            optimizer.step()
                            optimizer.zero_grad()                   

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0
                    # log training and evaluation 
                    if config.visualize_eval and (global_step % config.vis_freq ==0):

                        all_eval_images = []
                        all_eval_rewards = []
                        
                        total_samples = config.train.batch_size_per_gpu_available * config.max_vis_images
                       
                        samples_per_class = [int(total_samples * dist) for dist in list(config.class_label_dist)]
                        samples_per_class[-1] = total_samples - sum(samples_per_class[:-1])
                        eval_labels = torch.cat([torch.full((count,), i, device=accelerator.device, dtype=torch.long) for i, count in enumerate(samples_per_class)])
                        
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                            latent = torch.randn((total_samples, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)
                        else:
                            latent = torch.randn((total_samples, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)
                        
                        with torch.no_grad():
                            for index in range(config.max_vis_images):
                                ims, rewards, _ = evaluate(
                                    latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available*(index+1)],
                                    train_neg_prompt_embeds,
                                    eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available*(index+1)], 
                                    train_neg_labels,
                                    eval_labels[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available*(index+1)],
                                    pipeline, 
                                    accelerator, 
                                    inference_dtype,
                                    config, 
                                    eval_loss_fn)
                                all_eval_images.append(ims)
                                all_eval_rewards.append(rewards)
                        
                        eval_rewards = torch.cat(all_eval_rewards)
                        eval_reward_mean = eval_rewards.mean()
                        eval_reward_std = eval_rewards.std()
                        eval_images = torch.cat(all_eval_images)
                        
                        # Calculate class-wise mean and std
                        class_0_rewards = eval_rewards[eval_labels == 0]
                        class_1_rewards = eval_rewards[eval_labels == 1]
                        class_2_rewards = eval_rewards[eval_labels == 2]
                        class_3_rewards = eval_rewards[eval_labels == 3]
                        
                        class_0_mean = class_0_rewards.mean() if len(class_0_rewards) > 0 else 0
                        class_0_std = class_0_rewards.std() if len(class_0_rewards) > 0 else 0
                        class_1_mean = class_1_rewards.mean() if len(class_1_rewards) > 0 else 0
                        class_1_std = class_1_rewards.std() if len(class_1_rewards) > 0 else 0
                        
                        class_2_mean = class_2_rewards.mean() if len(class_2_rewards) > 0 else 0
                        class_2_std = class_2_rewards.std() if len(class_2_rewards) > 0 else 0
                        class_3_mean = class_3_rewards.mean() if len(class_3_rewards) > 0 else 0
                        class_3_std = class_3_rewards.std() if len(class_3_rewards) > 0 else 0
                        
                        # Calculate classification metrics
                        predicted_classes = classify_compressibility_scores_4class(eval_rewards)
                        metrics = compute_classification_metrics(predicted_classes, eval_labels)
                        
                        eval_image_vis = []
                        if accelerator.is_main_process:
                            name_val = config.run_name
                            log_dir = f"logs/{name_val}/eval_vis"
                            os.makedirs(log_dir, exist_ok=True)
                            
                            for i, eval_image in enumerate(eval_images):
                                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                                pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                                prompt = eval_prompts[i]
                                label = eval_labels[i]
                                
                                pil.save(f"{log_dir}/{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                                pil = pil.resize((256, 256))
                                
                                reward = eval_rewards[i]
                                predicted_class = predicted_classes[i]
                                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f} (class:{predicted_class}) | true class:{label}"))                    
                                
                            accelerator.log({"eval_images": eval_image_vis},step=global_step)
                    
                    logger.info("Logging")
                    
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                    info.update({"epoch": epoch, 
                                 "inner_epoch": inner_iters, 
                                 "eval_rewards_mean":eval_reward_mean,
                                #  "eval_rewards_std":eval_reward_std,
                                 "eval_class_0_rewards_mean":class_0_mean,
                                #  "eval_class_0_rewards_std":class_0_std,
                                 "eval_class_1_rewards_mean":class_1_mean,
                                #  "eval_class_1_rewards_std":class_1_std, 
                                "eval_class_2_rewards_mean":class_2_mean,
                                "eval_class_3_rewards_mean":class_3_mean,   
                                 "eval_accuracy":metrics['accuracy'],
                                 "eval_macro_F1": metrics['macro_F1'],
                                 "eval_macro_precision": metrics['macro_precision'],
                                 "eval_macro_recall": metrics['macro_recall'],
                    })
                    
                    accelerator.log(info, step=global_step)

                    global_step += 1
                    info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            
            if epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()

if __name__ == "__main__":
    app.run(main)
