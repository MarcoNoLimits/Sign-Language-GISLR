import math
import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import GISLRTransformer

class GISLRKaggleDataset(Dataset):
    def __init__(self, df, root_dir, sign_map, max_frames=100, augment=True):
        self.data_df = df
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.augment = augment
        self.sign_to_index = sign_map
        
        # Hard offsets to strictly align missing coordinate rows
        self.offsets = {'face': 0, 'left_hand': 468, 'pose': 489, 'right_hand': 522}

    def __len__(self):
        return len(self.data_df)

    def rotate_landmarks(self, landmarks):
        angle = np.random.uniform(-0.15, 0.15)
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
        landmarks[:, :2] = np.dot(landmarks[:, :2], rotation_matrix)
        return landmarks

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        parquet_path = os.path.join(self.root_dir, row['path'])
        label = self.sign_to_index[row['sign']]
        
        try:
            df = pd.read_parquet(parquet_path)
            frames = df['frame'].unique()
            num_frames = min(len(frames), self.max_frames)
            
            tensor_data = np.zeros((self.max_frames, 543, 3), dtype=np.float32)
            
            for i, frame in enumerate(frames[:num_frames]):
                frame_df = df[df['frame'] == frame]
                frame_coords = np.zeros((543, 3), dtype=np.float32)
                
                # Rigid mapping to prevent coordinate shifting!
                for _, row_data in frame_df.iterrows():
                    l_type = row_data['type']
                    l_idx = int(row_data['landmark_index'])
                    abs_idx = self.offsets[l_type] + l_idx
                    if abs_idx < 543:
                        frame_coords[abs_idx] = [row_data['x'], row_data['y'], row_data['z']]
                
                # Normalization is now safely handled natively by PyTorch's LayerNorm in model.py!
                # Doing it manually with Numpy masks caused a divide-by-zero that triggered the NaN explosion. 
                
                if self.augment:
                    # Only augment frames that actually have data (non-zero)
                    mask = (frame_coords != 0.0)
                    frame_coords += np.random.normal(0, 0.005, frame_coords.shape) * mask
                    if np.random.rand() > 0.5:
                        frame_coords = self.rotate_landmarks(frame_coords)

                tensor_data[i] = frame_coords
            
            # CRITICAL FIX: The Parquet files contain literal `NaN` floats for missing coordinates!
            # If a single NaN hits the LSTM, the entire sequence and the Loss instantly become NaN.
            tensor_data = np.nan_to_num(tensor_data, nan=0.0)
            
            return torch.tensor(tensor_data), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return torch.zeros((self.max_frames, 543, 3)), torch.tensor(label, dtype=torch.long)

def train_kaggle_model():
    print("Initializing GISLR Transformer (10-Class Fixed Tracker)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Mini 10-Class Problem
    POPULAR_10 = ['hello', 'thankyou', 'please', 'yes', 'no', 'mom', 'dad', 'eat', 'water', 'drink']
    sign_map_10 = {sign: i for i, sign in enumerate(POPULAR_10)}
    print(f"Testing on 10 Classes: {POPULAR_10}")

    model = GISLRTransformer(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    KAGGLE_ROOT = "/kaggle/input/competitions/asl-signs"
    TRAIN_CSV = "/kaggle/input/competitions/asl-signs/train.csv"
    
    if not os.path.exists(TRAIN_CSV):
        print("Error: Dataset not found. Check your Input sidebar.")
        return

    full_df = pd.read_csv(TRAIN_CSV)
    subset_df = full_df[full_df['sign'].isin(POPULAR_10)]
    subset_df = subset_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Subsetting complete: Using {len(subset_df)} samples.")

    dataset = GISLRKaggleDataset(subset_df, KAGGLE_ROOT, sign_map_10, max_frames=100, augment=True)
    # Set num_workers=0 to prevent Kaggle notebook deadlocks when reading Parquets!
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    
    epochs = 20 
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, leave=True)
        
        for i, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=avg_loss)
            
        current_epoch_loss = total_loss / len(dataloader)
        scheduler.step(current_epoch_loss)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {current_epoch_loss:.4f}")
        
        # Save a checkpoint immediately so Kaggle timeouts don't destroy the run!
        checkpoint_path = f"/kaggle/working/checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved {checkpoint_path}")

    print("Training Completed! Saving final weights...")
    torch.save(model.state_dict(), "/kaggle/working/weights_10class.pth")
    
    with open('/kaggle/working/sign_map_10.json', 'w') as f:
        json.dump(sign_map_10, f)

if __name__ == "__main__":
    train_kaggle_model()