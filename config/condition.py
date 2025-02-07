import ml_collections
import os

def general():
    config = ml_collections.ConfigDict()

    ###### General ######    
    config.eval_prompt_fn = ''
    config.soup_inference = False
    config.save_freq = 4
    config.resume_from = ""
    config.resume_from_2 = ""
    config.vis_freq = 1
    config.max_vis_images = 2
    config.only_eval = False
    
    # prompting
    config.prompt_fn = "simple_animals"
    config.reward_fn = "aesthetic"
    config.debug =False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 10
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = "test"
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 42    
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100    

    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = False
    config.visualize_eval = True

    config.truncated_backprop = False
    config.truncated_backprop_rand = False
    config.truncated_backprop_minmax = (35,45)
    config.trunc_backprop_timestep = 100
    
    config.grad_checkpoint = True
    config.same_evaluation = True
    
    
    ###### Training ######    
    config.train = train = ml_collections.ConfigDict()
    config.train.loss_coeff = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8 
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0    
    
    train.kl_weight = 0.0

    config.aesthetic_target = 10
    config.grad_scale = 1
    config.sd_guidance_scale = 7.5
    config.steps = 50 
    
    config.sample_eta = 1.0

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"
    return config



def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):
    #  Samples per epoch
    config.train.total_samples_per_epoch = total_samples_per_epoch  #256
    # config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus # 64
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #128
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus  # 32
    config.train.batch_size_per_gpu_available = per_gpu_capacity    # 4
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available # 8
    
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available  # 16
    return config

def aesthetic():
    config = general()
    config.num_epochs = 60
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "eval_aesthetic_animals"

    config.reward_fn = 'aesthetic' # CLIP or imagenet or .... or .. 
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 0.1
    
    config.train.learning_rate = 1e-4
    config.train.embed_learning_rate = 5e-3
    
    config.debug = False
    config.sample_eta = 1.0
    config.train.kl_weight = 0.01
    
    config.eval_div_freq = 6
    config.num_samples_Div = 32
    
    config.class_label_dist = (0.5, 0.5)
    config.class_embed_type = None  # None or 'timestep'
    
    config.max_vis_images = 5
    config.train.adam_weight_decay = 0.1
    
    config.save_freq = 1
    # config.num_epochs = 7
    config.num_checkpoint_limit = 100
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0,50)
    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    
    config.train.batch_size_per_gpu_available = 4
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    config.train.num_gpus = num_gpus
    
    config = set_config_batch(
                config,
                total_samples_per_epoch=128*4,
                total_batch_size=128*4, 
                per_gpu_capacity=config.train.batch_size_per_gpu_available
            )
    return config

def compressibility():
    config = general()
    config.num_epochs = 60
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "eval_aesthetic_animals"

    config.reward_fn = 'compressibility' # CLIP or imagenet or .... or .. 
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 0.1
    
    config.train.learning_rate = 1e-3
    config.train.embed_learning_rate = 1e-2
    
    config.debug = False
    config.sample_eta = 1.0
    config.train.kl_weight = 0.01
    
    config.eval_div_freq = 6
    config.num_samples_Div = 32
    
    # config.class_label_dist = (0.5, 0.5)
    config.class_label_dist = (0.25, 0.25, 0.25, 0.25)
    config.class_embed_type = None
    
    config.max_vis_images = 10
    config.train.adam_weight_decay = 0.1
    
    config.save_freq = 1
    # config.num_epochs = 7
    config.num_checkpoint_limit = 100
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0,50)
    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    
    config.train.batch_size_per_gpu_available = 4
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    config.train.num_gpus = num_gpus
    
    config = set_config_batch(
                config,
                total_samples_per_epoch=128*4,
                total_batch_size=64*4,  
                per_gpu_capacity=config.train.batch_size_per_gpu_available
            )
    return config

def multitask():
    config = general()
    config.num_epochs = 60
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "eval_aesthetic_animals"

    config.reward_fn = 'multitask' # CLIP or imagenet or .... or .. 
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 0.1
    
    config.train.learning_rate = 3e-4
    config.train.embed_learning_rate = 1e-2
    
    config.train.compress_weight = 2.0
    
    config.debug = False
    config.sample_eta = 1.0
    config.train.kl_weight = 0.01
    
    config.eval_div_freq = 6
    config.num_samples_Div = 32
    
    # (aes:0 and comp:0, aes:0 and comp:1, aes:1 and comp:0, aes:1 and comp:1)
    config.compressibility_class_label_dist = (0.5, 0.5)
    config.aesthetic_class_label_dist = (0.5, 0.5)
    
    config.class_label_dist = (
        config.aesthetic_class_label_dist[0] * config.compressibility_class_label_dist[0],  # aes_class_0 and comp_class_0
        config.aesthetic_class_label_dist[0] * config.compressibility_class_label_dist[1],  # aes_class_0 and comp_class_1
        config.aesthetic_class_label_dist[1] * config.compressibility_class_label_dist[0],  # aes_class_1 and comp_class_0
        config.aesthetic_class_label_dist[1] * config.compressibility_class_label_dist[1]   # aes_class_1 and comp_class_1
    )
    
    config.class_embed_type = None  # None or 'timestep'
    
    config.max_vis_images = 12
    config.train.adam_weight_decay = 0.1
    
    config.save_freq = 1
    # config.num_epochs = 7
    config.num_checkpoint_limit = 100
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0,50)
    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    
    config.train.batch_size_per_gpu_available = 4
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    config.train.num_gpus = num_gpus
    
    config = set_config_batch(
                config,
                total_samples_per_epoch=256*4,
                total_batch_size=128*4, 
                per_gpu_capacity=config.train.batch_size_per_gpu_available
            )
    return config


def evaluate_multitask():
    config = multitask()
    config.prompt_fn = "eval_aesthetic_animals"
    config.only_eval = True
    config.same_evaluation = True
    config.max_vis_images = 32
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    config.train.num_gpus = num_gpus
    
    config = set_config_batch(
                config,
                total_samples_per_epoch=64*num_gpus,
                total_batch_size=32*num_gpus, 
                per_gpu_capacity=4
            )
    return config

def evaluate_comp():
    config = compressibility()
    config.prompt_fn = "eval_aesthetic_animals"
    config.only_eval = True
    config.same_evaluation = True
    config.max_vis_images = 32
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    config.train.num_gpus = num_gpus
    
    config = set_config_batch(
                config,
                total_samples_per_epoch=64*num_gpus,
                total_batch_size=32*num_gpus, 
                per_gpu_capacity=4
            )
    return config


def get_config(name):
    return globals()[name]()