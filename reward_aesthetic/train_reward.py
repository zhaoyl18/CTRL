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
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
import argparse
import datetime
from aesthetic_scorer import MLPDiff, MLPDiff_class, classify_aesthetic_scores
from diffusers_patch.utils import TemperatureScaler

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
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--train_bs', type=int, default=256)
    parser.add_argument('--val_bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--noise', type=float, default=0.1, help='label noise level')
    parser.add_argument('--run_name', type=str, default='calibrated_classifier_v2')
    
    parser.add_argument('--SGLD', type = bool, default = False)
    parser.add_argument('--SGLD_base_noise', type = float, default = 0.1)

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id

    wandb.init(project="conditioning_reward_aesthetic", name=args.run_name,
        config={
        'lr': args.lr,
        # 'num_data':args.num_data,
        'num_epochs':args.num_epochs,
        'train_batch_size':args.train_bs,
        'val_batch_size':args.val_bs,
    })
    # load the training data 

    x = np.load("./reward_aesthetic/data/ava_x_openclip_l14.npy")
    y = np.load("./reward_aesthetic/data/ava_y_openclip_l14.npy")

    val_percentage = 0.20 # 20% of the trainingdata will be used for validation

    train_border = int(x.shape[0] * (1 - val_percentage) )

    train_tensor_x = torch.Tensor(x[:train_border]) # transform to torch tensor
    train_tensor_y = torch.Tensor(y[:train_border])

    train_dataset = TensorDataset(train_tensor_x,train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=16)


    val_tensor_x = torch.Tensor(x[train_border:])
    val_tensor_y = torch.Tensor(y[train_border:])

    val_dataset = TensorDataset(val_tensor_x,val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, num_workers=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLPDiff_class(out_channels=3).to(device)   # CLIP embedding dim is 768 for CLIP ViT L 14

    optimizer = torch.optim.Adam(model.parameters()) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)



    # choose the loss you want to optimze for
    criterion = nn.CrossEntropyLoss()

    model.train()
    best_loss = 999
    best_model = None
    
    eval_model = MLPDiff().to(device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    s = torch.load("./assets/sac+logos+ava1-l14-linearMSE.pth")
    eval_model.load_state_dict(s)

    def adjust_noise(learning_rate, batch_size):
        return args.SGLD_base_noise * (learning_rate ** 0.5) / (batch_size ** 0.5)   

    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        if args.SGLD:
            noise_level = adjust_noise(optimizer.param_groups[0]['lr'], args.train_bs)
        
        losses = []
        save_name = f'./reward_aesthetic/models/{args.run_name}_{epoch+1}.pth'
        
        for batch_num, (x,_) in enumerate(tqdm(train_loader,
                                desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            optimizer.zero_grad()
            x = x.to(device).float()
            y_real = eval_model(x).to(device)
            # noisy_y = y_real + torch.randn_like(y_real,device=device) * args.noise

            class_labels = classify_aesthetic_scores(y_real)
            outputs = model(x)
            
            loss = criterion(outputs, class_labels.detach())
            loss.backward()
            losses.append(loss.item())
            
            if args.SGLD:
                for param in model.parameters(): # add Gaussian noise to gradients
                    param.grad += noise_level * torch.randn_like(param.grad)

            optimizer.step()
            wandb.log({"batch_loss": loss.item()})

        print('Epoch %d | CEL Loss %6.4f' % (epoch, sum(losses)/len(losses)))
        wandb.log({"epoch": epoch, "CEL loss": sum(losses)/len(losses)})
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_num, input_data in enumerate(val_loader):
                optimizer.zero_grad()
                x, _ = input_data
                x = x.to(device).float()
                y_real = eval_model(x).to(device)
                class_labels = classify_aesthetic_scores(y_real)

                outputs = model(x)
                val_loss += criterion(outputs, class_labels.detach()).item()
            
            val_loss /= len(val_loader)
            print('\tValidation - Epoch %d | CEL Loss %6.4f' % (epoch, val_loss))
            wandb.log({"Val loss": val_loss})

        if val_loss < best_loss:
            
            best_loss = val_loss
            print(f"Best Val loss so far: {best_loss}... Saving model")
            torch.save(model.state_dict(), save_name)
            best_model = model.state_dict()
        
        scheduler.step(val_loss)
    
    # torch.save(model.state_dict(), save_name)

    print("Best CEL loss:", best_loss) 

    print("Training done")
    
    # After training, perform calibration using the validation set
    scaler = TemperatureScaler().to(device)
    
    model.load_state_dict(best_model)
    model.eval()
    model.requires_grad_(False)
    
    val_logits = []
    val_labels = []
    
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device).float()
            y_real = eval_model(x).to(device)
            class_labels = classify_aesthetic_scores(y_real)
            outputs = model(x)
            val_logits.append(outputs)
            val_labels.append(class_labels)

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        scaler.calibrate(val_logits, val_labels)

    # Saving the final model state with calibration
    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler.state_dict()}, f'./reward_aesthetic/models/{args.run_name}_final_calibrated.pth')
    print("Training and calibration completed.")
    print("Calibrated temperature:", scaler.temperature.item())

    # Inference test with calibrated model
    print("Inference test with dummy samples from the val set for sanity check")
    
    probabilities = predict_with_calibration(model, scaler, x[:10], device)
    print(probabilities.size())
    print(probabilities)

if __name__   == "__main__":
    train()