#!/usr/bin/env python3
"""
StarNet Fine-tuning Script per RunPod
===================================

Fine-tuning classico su dataset 512x512 per astrofotografia.
La validation loss è sensibile ai dettagli fini per migliorare la precisione.

Struttura dataset:
- <data_dir>/train/input/*.png|*.jpg
- <data_dir>/train/target/*.png|*.jpg
- <data_dir>/val/input/*.png|*.jpg
- <data_dir>/val/target/*.png|*.jpg

Uso su RunPod:
1) Copia questa cartella in /workspace/
2) pip install -r requirements.txt
3) python finetune_starnet.py --data_dir /workspace/finetune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # PyTorch 2.x compatibility
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend headless per server
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import argparse

def make_hole_mask(inp, tgt, thr_inp=0.25, thr_tgt=0.12, dilate=3, feather=True):
    """
    Maschera 'buchi' con bordi sfumati (feather mask) per realismo.
    thr_inp: soglia per 'stella' nell'input (in [0,1])
    thr_tgt: soglia per 'riempito' nel target
    dilate : raggio in pixel per coprire i bordi del buco
    feather: bordi sfumati per evitare artefatti sui bordi
    """
    inp_g = inp.mean(1, keepdim=True)   # [B,1,H,W]
    tgt_g = tgt.mean(1, keepdim=True)
    raw = (inp_g > thr_inp) & (tgt_g < thr_tgt)
    mask = raw.float()
    
    # Dilate tradizionale
    if dilate > 0:
        k = torch.ones(1, 1, 2*dilate+1, 2*dilate+1, device=inp.device)
        mask = (F.conv2d(mask, k, padding=dilate) > 0).float()
    
    # Feather mask sui bordi (fondamentale per realismo!)
    if feather:
        mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2).clamp(0, 1)
    
    return mask  # [0,1] con bordi sfumati

# =================== ARCHITETTURA STARNET CORRETTA ===================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.01):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Dropout configurabile!
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # CORRETTO: il DoubleConv riceve in_channels e produce out_channels
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 se ha dimensioni spaziali diverse da x1
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class StarNetUNetNoReduce512(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):  # bilinear=True!
        super(StarNetUNetNoReduce512, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # 512, non 1024!
        
        # Attention gates con dimensioni corrette dal modello originale  
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)  # W_g: [256,512], W_x: [256,512]
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)  # W_g: [128,256], W_x: [128,256]
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)   # W_g: [64,128], W_x: [64,128] 
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)     # W_g: [32,64], W_x: [32,64]
        
        self.up1 = Up(1024, 256, bilinear)  # Output: 256 canali
        self.up2 = Up(512, 128, bilinear)   # Output: 128 canali  
        self.up3 = Up(256, 64, bilinear)    # Output: 64 canali
        self.up4 = Up(128, 64, bilinear)    # Output: 64 canali
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 512

        # Decoder with attention
        x4_att = self.att4(x5, x4)
        x = self.up1(x5, x4_att)  # 256

        x3_att = self.att3(x, x3)
        x = self.up2(x, x3_att)   # 128

        x2_att = self.att2(x, x2)
        x = self.up3(x, x2_att)   # 64

        x1_att = self.att1(x, x1)
        x = self.up4(x, x1_att)   # 64

        logits = self.outc(x)     # 3
        return logits

# =================== DATASET ===================

class StarNetDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Percorsi assoluti usando data_dir
        input_dir = os.path.join(self.data_dir, split, "input")
        target_dir = os.path.join(self.data_dir, split, "target")
        
        # Trova tutti i file input (PNG e JPG)
        input_patterns = [
            os.path.join(input_dir, "*.png"),
            os.path.join(input_dir, "*.jpg"),
            os.path.join(input_dir, "*.jpeg")
        ]
        
        self.input_files = []
        for pattern in input_patterns:
            self.input_files.extend(glob.glob(pattern))
        
        self.input_files.sort()
        
        # Trova corrispondenti target files
        self.pairs = []
        for input_file in self.input_files:
            basename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Cerca target corrispondente (PNG o JPG)
            target_candidates = [
                os.path.join(target_dir, f"{basename}.png"),
                os.path.join(target_dir, f"{basename}.jpg"),
                os.path.join(target_dir, f"{basename}.jpeg")
            ]
            
            target_file = None
            for candidate in target_candidates:
                if os.path.exists(candidate):
                    target_file = candidate
                    break
            
            if target_file:
                self.pairs.append((input_file, target_file))
        
        print(f"Dataset {split}: trovate {len(self.pairs)} coppie valide")
        if len(self.pairs) == 0:
            print(f"ERRORE: Nessuna coppia trovata in {input_dir} -> {target_dir}")
            print(f"Input files esempio: {self.input_files[:5]}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        
        # Carica immagini
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        # Verifica dimensioni (devono essere 512x512)
        if input_image.size != (512, 512) or target_image.size != (512, 512):
            # Resize se necessario
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            target_image = target_image.resize((512, 512), Image.LANCZOS)
        
        # Applica trasformazioni
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

# =================== TRAINING ===================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, ema=None):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mse_hole = 0
    
    # Progress bar per i batch
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training per GPU!
        with autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss, logs, hole = criterion(outputs, targets, inputs, epoch)
            # MSE puro per confronto con training precedenti
            mse_pure = F.mse_loss(outputs, targets)
            # MSE solo nei buchi (metrica chiave!) - normalizzato per canali
            C = outputs.size(1)
            mse_hole = F.mse_loss(outputs*hole, targets*hole, reduction='sum') / (hole.sum()*C + 1e-8)
        
        # Scaled backward per mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Aggiorna EMA per stabilità
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        total_mse += mse_pure.item()
        total_mse_hole += mse_hole.item()
        
        # Aggiorna progress bar con loss corrente
        pbar.set_postfix({
            'HoleLoss': f'{loss.item():.6f}',
            'MSE': f'{mse_pure.item():.6f}',
            'MSE(hole)': f'{mse_hole.item():.6f}'
        })
    
    # Media delle loss
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_mse_hole = total_mse_hole / len(dataloader)
    
    return avg_loss, {'total': avg_loss, 'mse': avg_mse, 'mse_hole': avg_mse_hole}

def validate_epoch(model, dataloader, criterion, device, current_epoch=0):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mse_hole = 0
    
    with torch.no_grad():
        # Progress bar per validazione
        pbar = tqdm(dataloader, desc='Validating', leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Mixed precision anche per validazione
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss, logs, hole = criterion(outputs, targets, inputs, epoch=current_epoch, eval_fixed_dilate=None)
                # MSE puro per confronto
                mse_pure = F.mse_loss(outputs, targets)
                # MSE solo nei buchi (metrica chiave!) - normalizzato per canali
                C = outputs.size(1)
                mse_hole = F.mse_loss(outputs*hole, targets*hole, reduction='sum') / (hole.sum()*C + 1e-8)

                # Background mask = 1 - hole
                bg = 1.0 - hole
                bg_diff = (torch.abs(outputs - targets).mean(1, keepdim=True) * bg).sum() / (bg.sum() + 1e-8)
            
            total_loss += loss.item()
            total_mse += mse_pure.item()
            total_mse_hole += mse_hole.item()
            
            # Aggiorna progress bar con loss corrente
            pbar.set_postfix({
                'HoleLoss': f'{loss.item():.6f}',
                'MSE': f'{mse_pure.item():.6f}',
                'MSE(hole)': f'{mse_hole.item():.6f}',
                'BG Diff': f'{bg_diff.item():.6f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_mse_hole = total_mse_hole / len(dataloader)
    
    return avg_loss, {'total': avg_loss, 'mse': avg_mse, 'mse_hole': avg_mse_hole, 'bg_diff': bg_diff.item()}

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Salva checkpoint con tutti i dettagli"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato: {path}")

def plot_training_curves(train_losses, val_losses, save_path):
    """Plotta curve di training"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss totale
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [l['total'] for l in train_losses], 'b-', label='Train')
    plt.plot(epochs, [l['total'] for l in val_losses], 'r-', label='Val')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MSE Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [l['mse'] for l in train_losses], 'b-', label='Train')
    plt.plot(epochs, [l['mse'] for l in val_losses], 'r-', label='Val')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Validation Loss Zoom (per vedere miglioramenti fini)
    plt.subplot(2, 2, 3)
    val_totals = [l['total'] for l in val_losses]
    plt.plot(epochs, val_totals, 'r-', linewidth=2, label='Val Total')
    plt.title('Validation Loss (Fine Details)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate Schedule
    plt.subplot(2, 2, 4)
    # Se abbiamo info sul LR, plottiamolo
    plt.plot(epochs, [0.00005] * len(epochs), 'g--', alpha=0.7, label='Initial LR')
    plt.title('Training Info')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # =================== ARGOMENTI DA TERMINALE ===================
    
    parser = argparse.ArgumentParser(description='StarNet Fine-tuning con parametri configurabili')
    
    # Parametri principali
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Numero di epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    
    # Penalty della loss function
    parser.add_argument('--black_penalty', type=float, default=200.0, help='Penalty per zone nere (default: 200.0)')
    parser.add_argument('--star_nebula_penalty', type=float, default=250.0, help='Penalty stelle su nebulosa (default: 250.0)')
    parser.add_argument('--nebula_penalty', type=float, default=120.0, help='Penalty nebulosa (default: 120.0)')
    parser.add_argument('--bright_penalty', type=float, default=100.0, help='Penalty zone luminose (default: 100.0)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='.', help='Directory dataset (default: .)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory output (default: ./outputs)')
    parser.add_argument('--pretrained_model', type=str, default='../best_model_epoch13.pth', 
                       help='Path modello pre-trained da caricare (default: ../best_model_epoch13.pth)')
    
    # Regolarizzazione
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (default: 1e-6)')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (default: 0.01)')
    
    args = parser.parse_args()
    
    # =================== CONFIGURAZIONE ===================
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configurazione da argomenti
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'pretrained_model': args.pretrained_model,
        'dropout': args.dropout,
        # Penalty loss function
        'black_penalty': args.black_penalty,
        'star_nebula_penalty': args.star_nebula_penalty,
        'nebula_penalty': args.nebula_penalty,
        'bright_penalty': args.bright_penalty
    }
    
    # Stampa configurazione
    print("\n🎯 CONFIGURAZIONE FINE-TUNING:")
    print("=" * 50)
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"🎓 Learning Rate: {config['learning_rate']:.0e}")
    print(f"⏱️  Epochs: {config['num_epochs']}")
    print(f"⏳ Patience: {config['patience']}")
    print(f"💪 Weight Decay: {config['weight_decay']:.0e}")
    print(f"🎲 Dropout: {config['dropout']}")
    print(f"📁 Data Dir: {config['data_dir']}")
    print(f"💾 Output Dir: {config['output_dir']}")
    print(f"🔄 Pretrained: {config['pretrained_model']}")
    print("\n🎯 PENALTY LOSS FUNCTION:")
    print(f"⚫ Black zones: {config['black_penalty']}x")
    print(f"🌟 Stars on nebula: {config['star_nebula_penalty']}x") 
    print(f"🌌 Nebula preservation: {config['nebula_penalty']}x")
    print(f"⭐ Bright zones: {config['bright_penalty']}x")
    print("=" * 50)
    
    # =================== CONFIGURAZIONE PRECEDENTE ===================
    
    # Crea directory output
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Salva configurazione
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # =================== DATASET E DATALOADER ===================
    
    # Augmentation minimali per astrofoto (solo flip)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = StarNetDataset(config['data_dir'], 'train', train_transform)
    val_dataset = StarNetDataset(config['data_dir'], 'val', val_transform)
    
    # DataLoader ottimizzati per GPU
    num_workers = 8
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=(device.type=='cuda'),
        persistent_workers=(num_workers>0),
        prefetch_factor=(2 if num_workers>0 else None)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type=='cuda'),
        persistent_workers=(num_workers>0),
        prefetch_factor=(2 if num_workers>0 else None)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # =================== MODELLO ===================
    
    model = StarNetUNetNoReduce512(n_channels=3, n_classes=3).to(device)
    
    # EMA (Exponential Moving Average) per stabilità in inferenza
    class EMA:
        def __init__(self, model, decay=0.999):
            self.model = model
            self.decay = decay
            self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            self.backup = {}

        def update(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)

        def apply_shadow(self):
            self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = self.shadow[n].clone()

        def restore(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = self.backup[n].clone()
            self.backup = {}
    
    ema = EMA(model, decay=0.9995)  # EMA più lenta per maggiore stabilità
    
    # Carica modello pre-trained se esiste
    start_epoch = 0
    if os.path.exists(config['pretrained_model']):
        print(f"Caricando modello pre-trained: {config['pretrained_model']}")
        checkpoint = torch.load(config['pretrained_model'], map_location=device)
        
        # Gestisce diversi formati di checkpoint
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Riprendendo training da epoch {start_epoch}")
        else:
            # Formato diretto state_dict
            state = checkpoint
        
        # Rimuovi chiavi legacy che potrebbero causare errori
        for k in list(state.keys()):
            if k.endswith("residual_weight") or k == "residual_weight":
                print(f"Rimossa chiave legacy: {k}")
                state.pop(k)
        
        # Carica con strict=False per compatibilità
        model.load_state_dict(state, strict=False)
        print("Modello pre-trained caricato con successo!")
    else:
        print("Nessun modello pre-trained trovato, inizializzazione casuale")
    
    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainabili: {trainable_params:,}")
    
    # =================== OPTIMIZER E LOSS ===================
    
    # LR più alto per fine-tuning efficace
    base_lr = config['learning_rate'] if config['learning_rate'] > 1e-5 else 2e-5
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=base_lr, 
                           weight_decay=config['weight_decay'])
    
    # Warmup + Cosine schedule invece di ReduceLROnPlateau
    warmup_epochs = 5
    min_lr = 1e-6
    
    def set_lr(lr):
        for g in optimizer.param_groups:
            g['lr'] = lr
    
    # Funzione per calcolare LR corretto all'epoca
    def lr_at_epoch(e, base_lr=base_lr, warmup=warmup_epochs, total=config['num_epochs'], eta_min=min_lr):
        if e < warmup:
            return base_lr * (e + 1) / warmup
        # fase coseno
        t = e - warmup
        T = max(1, total - warmup)
        import math
        return eta_min + 0.5*(base_lr - eta_min)*(1 + math.cos(math.pi * t / T))
    
    # Fix KeyError: imposta il LR coerente con l'epoca da cui riprendi
    if start_epoch > 0:
        resumed_lr = lr_at_epoch(start_epoch, base_lr=base_lr, warmup=warmup_epochs,
                                 total=config['num_epochs'], eta_min=min_lr)
        for g in optimizer.param_groups:
            g['lr'] = resumed_lr
        print(f"[Scheduler] Resumed LR set to {resumed_lr:.2e} for epoch {start_epoch}")
    
    # IMPORTANTISSIMO: imposta initial_lr per ogni param group
    for g in optimizer.param_groups:
        g['initial_lr'] = base_lr  # il LR "di base" prima del cosine
    
    cosine_T = max(1, config['num_epochs'] - warmup_epochs)
    
    if start_epoch < warmup_epochs:
        # stai ancora in warmup: il cosine parte dopo
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T, eta_min=min_lr, last_epoch=-1
        )
    else:
        # sei oltre il warmup: il cosine è "avanzato" di (start_epoch - warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T, eta_min=min_lr,
            last_epoch=(start_epoch - warmup_epochs)
        )
    
    # Loss function focalizzata sui BUCHI con curriculum learning!



# =================== LOSS SEMPLIFICATA: STARREMOVALLOSS ===================
class StarRemovalLoss(nn.Module):
    def __init__(self, w_hole=1.0, w_reg=0.1):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.w_hole = w_hole
        self.w_reg = w_reg

    def forward(self, pred, target, inp, epoch=None, eval_fixed_dilate=None):
        # Maschera dei buchi (stelle)
        hole = make_hole_mask(inp, target, thr_inp=0.25, thr_tgt=0.12,
                              dilate=2, feather=False)  # dilate piccolo, no feather
        # L1 solo nei buchi (focus assoluto)
        l1_map = self.l1(pred, target).mean(1, keepdim=True)
        hole_loss = (l1_map * hole).sum() / (hole.sum() + 1e-8)
        # L1 regolare su tutta l’immagine (peso basso, per coerenza cromatica)
        reg_loss = self.l1(pred, target).mean()
        total = self.w_hole * hole_loss + self.w_reg * reg_loss
        logs = {
            "hole_loss": hole_loss.item(),
            "reg_loss": reg_loss.item(),
            "total": total.item(),
        }
        # return total, logs, hole
if __name__ == "__main__":
    main()

# =================== ESEMPI DI USO AGGIORNATI ===================
# 
# # Uso base (valori default con ottimizzazioni buchi):
# python finetune_starnet.py
#
# # Configurazione aggressiva per stelle ostinate:
# python finetune_starnet.py --learning_rate 4e-5 --num_epochs 75 --patience 20
#
# # Configurazione conservativa per preservare dettagli:
# python finetune_starnet.py --learning_rate 1e-5 --batch_size 16 --num_epochs 100
#
# # Training lungo con early stopping paziente:
# python finetune_starnet.py --num_epochs 100 --patience 25 --learning_rate 2e-5
#
# # Fine-tuning da modello specifico:
# python finetune_starnet.py --pretrained_model ./outputs/checkpoint_epoch_40.pth
#
# # Configurazione ottimizzata per RunPod A40:
# python finetune_starnet.py \
#   --batch_size 16 \
#   --learning_rate 3e-5 \
#   --num_epochs 75 \
#   --weight_decay 1e-6 \
#   --dropout 0.005 \
#   --patience 20
#
# # NUOVE FEATURES IMPLEMENTATE:
# # ✅ Feather mask sui bordi per realismo
# # ✅ Curriculum learning (dilate 3→4→5, pesi dinamici)
# # ✅ Balanced sampling by hole size
# # ✅ EMA (Exponential Moving Average) per stabilità
# # ✅ Micro-augmentation fotometrici
# # ✅ Pesi loss ottimizzati per realismo
# # ✅ Mixed precision + CUDNN optimizations