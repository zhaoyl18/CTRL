import os, pdb
import sys
import os
import copy
cwd = os.getcwd()
sys.path.append(cwd)

from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import wandb
import numpy as np
import argparse
import datetime
from PIL import Image

import torchvision
from torchvision import transforms

from compressibility_scorer import ThreeLayerConvNet, classify_compressibility_scores, classify_compressibility_scores_4class, jpeg_compressibility
from diffusers_patch.utils import TemperatureScaler

class CompressibilityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        self.transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
        
        if self.transform:
            image_tensor = self.transform(image)
            image_normalized = self.normalize(image_tensor)
        
        compressibility = jpeg_compressibility(image_tensor.unsqueeze(0))
        
        return image_normalized, torch.tensor(compressibility[0], dtype=torch.float32)
    
def predict_with_calibration(model, scaler, inputs, device=None):
    if device is None:
        device = model.device
    
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)
        calibrated_logits = scaler(logits)
        probabilities = F.softmax(calibrated_logits, dim=1)
    return probabilities


def train():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--val_bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--accumulation_steps', type=int, default=4)

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id

    wandb.init(project="conditioning_reward_comp", name=args.run_name,
        config={
        'lr': args.lr,
        'num_epochs':args.num_epochs,
        'train_batch_size':args.train_bs,
        'val_batch_size':args.val_bs,
    })
    
    torch.manual_seed(42)
    dataset = CompressibilityDataset("./reward_compressibility/data/images")

    # Splitting dataset
    val_size = int(len(dataset) * 0.20)  # 20% for validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=16, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs, num_workers=16, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ThreeLayerConvNet(num_channels=3, num_classes=5).to(device)

    optimizer = torch.optim.Adam(model.parameters()) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training loop
    criterion = nn.CrossEntropyLoss()

    best_loss = 999
    
    accumulation_steps = args.accumulation_steps

    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        losses = []
        save_name = f'./reward_compressibility/models/{args.run_name}_{epoch+1}.pth'
        
        model.train()
        optimizer.zero_grad()
        for batch_num, (inputs,targets) in enumerate(tqdm(train_loader,
                                desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            class_labels = classify_compressibility_scores_4class(targets)

            outputs = model(inputs)
            
            loss = criterion(outputs, class_labels.detach())
            loss = loss / accumulation_steps
            loss.backward()
            losses.append(loss.item() * accumulation_steps)
            wandb.log({"batch_loss": loss.item() * accumulation_steps})
            
            if (batch_num + 1) % accumulation_steps == 0:
                optimizer.step()  # Update parameters
                optimizer.zero_grad()  # Clear gradients for the next accumulation
        
        print('Epoch %d | CEL Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        wandb.log({"epoch": epoch, "CEL loss": sum(losses)/len(losses)})
        
        model.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            val_loss = 0
            for batch_num, (inputs,targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                class_labels = classify_compressibility_scores_4class(targets)

                outputs = model(inputs)
                val_loss += criterion(outputs, class_labels.detach()).item()
            
            val_loss /= len(val_loader)
            print('\tValidation - Epoch %d | CEL Loss %6.2f' % (epoch, val_loss))
            wandb.log({"Val loss": val_loss})

        if val_loss < best_loss:
            
            best_loss = val_loss
            print(f"Best Val loss so far: {best_loss}... Saving model")
            torch.save(model.state_dict(), save_name)
        
        scheduler.step(val_loss)
    
    torch.save(model.state_dict(), save_name)

    print("Best CEL loss:", best_loss) 

    print("Training done")
    
    

if __name__   == "__main__":
    train()
    
    # Calibrate model
    model_ckpt = 'reward_compressibility/models/CNN_5class_v1_64.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_ckpt)
    model = ThreeLayerConvNet(num_channels=3, num_classes=5).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    scaler = TemperatureScaler().to(device)
    
    
    val_logits = []
    val_labels = []
    
    torch.manual_seed(42)
    dataset = CompressibilityDataset("./reward_compressibility/data/images")

    # Splitting dataset
    val_size = int(len(dataset) * 0.20)  # 50% for validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, drop_last=True)
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader, desc="Processing batches for calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            class_labels = classify_compressibility_scores_4class(targets)

            outputs = model(inputs)
            val_logits.append(outputs)
            val_labels.append(class_labels)

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        scaler.calibrate(val_logits, val_labels)
        
    base_name, extension = model_ckpt.rsplit('.', 1)
    new_model_ckpt = f"{base_name}_final_calibrated.{extension}"
    
    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler.state_dict()}, 
            new_model_ckpt)
    print("Training and calibration completed.")
    print("Calibrated temperature:", scaler.temperature.item())

    # Inference test with calibrated model
    print("Inference test with dummy samples from the val set for sanity check")
    
    inputs, _ = next(iter(val_dataloader))
    probabilities = predict_with_calibration(model, scaler, inputs[:5], device)
    print(probabilities.size())
    print(probabilities)