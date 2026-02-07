import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm

from nano_sora.models.dit import DiT
from nano_sora.training.flow_matching import RectifiedFlow
from nano_sora.config import Config

def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = DiT(
        input_size=Config.input_size,
        patch_size=Config.patch_size,
        in_channels=Config.in_channels,
        hidden_size=Config.hidden_size,
        depth=Config.depth,
        num_heads=Config.num_heads,
        mlp_ratio=Config.mlp_ratio,
        class_dropout_prob=Config.class_dropout_prob,
        num_classes=Config.num_classes
    ).to(device)
    
    rf = RectifiedFlow(model)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)
    scaler = GradScaler()
    criterion = nn.MSELoss()
    
    model.train()
    
    print(f"Starting training on {device}...")
    
    for epoch in range(Config.epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Flow Matching training tuple
            xt, t, target = rf.get_train_tuple(x)
            
            with autocast(enabled=(Config.mixed_precision == "fp16")):
                v_pred = model(xt, t, y)
                loss = criterion(v_pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{Config.epochs} | Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Checkpointing
        if (epoch + 1) % Config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(Config.checkpoint_dir, f"dit_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train()
