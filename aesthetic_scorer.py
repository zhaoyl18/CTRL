# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import os,sys
cwd = os.getcwd()
sys.path.append(cwd)

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from transformers import CLIPModel
from PIL import Image
from torch.utils.checkpoint import checkpoint
from diffusers_patch.utils import TemperatureScaler

ASSETS_PATH = resources.files("assets")

def classify_aesthetic_scores(y):
    # Applying thresholds to map scores to classes
    class_labels = torch.zeros_like(y, dtype=torch.long)  # Ensure it's integer type for class labels
    class_labels[y >= 5.7] = 1
    class_labels[y < 5.7] = 0
    if class_labels.dim() > 1:
        return class_labels.squeeze(1)
    return class_labels

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class MLPDiff_class(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, out_channels),
        )

    def forward(self, embed):
        return self.layers(embed)
class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1), embed
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed


class condition_AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.dtype = dtype
        
        state_dict = torch.load('reward_aesthetic/models/MLP_3class_easy_v1_final_calibrated.pth')

        self.scaler = TemperatureScaler()
        self.scaler.load_state_dict(state_dict['scaler'])
        
        self.mlp = MLPDiff_class(out_channels=3)
        self.mlp.load_state_dict(state_dict['model_state_dict'])
        
        self.eval()

    def __call__(self, images, config):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        
        logits = self.mlp(embed)
        calibrated_logits = self.scaler(logits)
        probabilities = F.softmax(calibrated_logits, dim=1)
        
        return probabilities, embed


if __name__ == "__main__":
    scorer = AestheticScorerDiff(dtype=torch.float32).cuda()
    scorer.requires_grad_(False)
    for param in scorer.clip.parameters():
        assert not param.requires_grad
    for param in scorer.mlp.parameters():
        assert not param.requires_grad