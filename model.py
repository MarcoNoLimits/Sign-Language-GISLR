import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



class GISLRTransformer(nn.Module):
    def __init__(self, num_classes=250, num_landmarks=543, spatial_dims=3, d_model=256, dropout=0.3):
        super(GISLRTransformer, self).__init__()
        self.input_dim = num_landmarks * spatial_dims
        
        # 1. Coordinate Embedding
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Sequence Modeling (LSTM is vastly superior to Unmasked Transformers for zero-padded coordinates)
        # Bidirectional gets forward and backward motion
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model // 2, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        
        # 3. Dense Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x shape: [Batch, Frames, Landmarks=543, Spatial=3]
        """
        batch_size, frames = x.size(0), x.size(1)
        
        # [Batch, Frames, 1629]
        x = x.view(batch_size, frames, -1)
        
        # Project coordinates
        x = self.embedding(x)
        
        # Temporal context
        x, _ = self.lstm(x)
        
        # GLOBAL MAX POOLING: This mathematically ignores the 0.0 padded frames! 
        # (Average pooling includes the zeros, which destroys the gradients)
        x_pooled, _ = torch.max(x, dim=1)
        
        # Classification
        logits = self.classifier(x_pooled)
        return logits
