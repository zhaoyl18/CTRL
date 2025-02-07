<!-- TITLE -->
# **Adding Conditional Control to Diffusion Models with Reinforcement Learning**

<!-- ## Code -->
## Installation  

Create a conda environment with the following command:

```bash
conda create -n CTRL python=3.10
conda activate CTRL
pip install -r requirements.txt
```

Please use accelerate==0.17.0, other library dependancies might be flexible.

## Inference with Fine-tuned Weights

#### Single-Task: Compressibility

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch condition_main_compressibility.py --config config/condition.py:evaluate_comp --config.resume_from logs/compressibility_4class_bs=256/checkpoints/checkpoint_8 --config.run_name Eval_ckpt8_128 --config.max_vis_images 32
```  

#### Multi-Task: Compressibility and Aesthetic Scores

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch condition_main.py --config config/condition.py:evaluate_multitask --config.resume_from logs/multitask_bs=512/checkpoints/checkpoint_47 --config.run_name Eval_multitask_ckpt47_512 --config.max_vis_images 128
```  

## Training

HuggingFace Accelerate will automatically handle parallel training.  
We conduct our experiments on image tasks using 4 A100 GPUs. Please adjust ``config.train.batch_size_per_gpu_available`` variable in config files according to your GPU memory.  

#### Single-Task: Compressibility  

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch condition_main_compressibility.py --config config/condition.py:compressibility --config.run_name singletask_train
```

#### Multi-Task: Compressibility and Aesthetic Scores  

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch condition_main.py --config config/condition.py:multitask --config.resume_from logs_multitask/checkpoints/checkpoint_9 --config.run_name multitask_train
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{
zhao2025adding,
title={Adding Conditional Control to Diffusion Models with Reinforcement Learning},
author={Yulai Zhao and Masatoshi Uehara and Gabriele Scalia and Sunyuan Kung and Tommaso Biancalani and Sergey Levine and Ehsan Hajiramezanali},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=svp1EBA6hA}
}
```