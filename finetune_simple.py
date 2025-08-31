#!/usr/bin/env python3
"""
StarNet Fine-tuning SIMPLE
Versione minimal che funziona SUBITO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# =================== ARCHITETTURA STARNET MINIMAL ===================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.003):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class StarNetNoReduce512(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # NoReduce512: mantiene 512 canali
        # up1: input 512, output 256 (dopo upsample, concat con x4: 512+512=1024)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)
        self.outc = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

# =================== DATASET MINIMAL ===================

class SimpleDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.*")))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, "*.*")))
        
        if len(self.input_files) != len(self.target_files):
            raise ValueError(f"Mismatch: {len(self.input_files)} input vs {len(self.target_files)} target")
        
        print(f"Dataset: {len(self.input_files)} pairs")
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Carica immagini
        input_img = Image.open(self.input_files[idx]).convert('RGB')
        target_img = Image.open(self.target_files[idx]).convert('RGB')
        
        # Converti in tensor e normalizza
        input_tensor = torch.tensor(np.array(input_img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        target_tensor = torch.tensor(np.array(target_img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Normalizza [-1, 1]
        input_tensor = (input_tensor - 0.5) / 0.5
        target_tensor = (target_tensor - 0.5) / 0.5
        
        return input_tensor, target_tensor

# =================== TRAINING ===================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# =================== MAIN ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./finetune.pth')
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    print("🚀 StarNet Fine-tuning SIMPLE (NoReduce512)")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Model
    print("Loading model...")
    model = StarNetNoReduce512().to(device)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Model loaded")
    else:
        print("⚠️ Model not found, starting from scratch")

    # Dataset
    print("Loading dataset...")
    train_input = os.path.join(args.data_dir, 'train', 'input')
    train_target = os.path.join(args.data_dir, 'train', 'target')
    val_input = os.path.join(args.data_dir, 'val', 'input')
    val_target = os.path.join(args.data_dir, 'val', 'target')

    train_dataset = SimpleDataset(train_input, train_target)
    val_dataset = SimpleDataset(val_input, val_target)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🔥 Starting training for {args.epochs} epochs...")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_path = os.path.join(args.output_dir, 'best_finetuned_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_path)
            print(f"🎯 New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"⏹️ Early stopping after {args.patience} epochs without improvement")
            break

    print(f"\n✅ Training completed!")
    print(f"🏆 Best Val Loss: {best_val_loss:.6f}")
    print(f"📁 Model saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
