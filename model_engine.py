import torch
import torch.nn.functional as F
import time
import os
import json
from model import GISLRTransformer

class ModelEngine:
    def __init__(self, weights_path="checkpoint_epoch_15.pth", vocab_path="sign_to_prediction_index_map.json"):
        """
        Initializes the model engine and loads PyTorch model weights via GISLRTransformer.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dynamically load the vocabulary map provided by Kaggle dataset
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                raw_map = json.load(f)
                # Invert the dictionary so we can lookup word by its integer index
                self.vocab = {v: k for k, v in raw_map.items()}
            num_classes = len(self.vocab)
        else:
            print(f"WARNING: {vocab_path} not found! Falling back to 250 classes.")
            self.vocab = {i: f"[Sign_{i}]" for i in range(250)}
            num_classes = 250
        
        self.model = GISLRTransformer(num_classes=num_classes)
        self.model.to(self.device)
        
        if os.path.exists(weights_path):
            print(f"Loading PyTorch weights from {weights_path}...")
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print("Weights loaded successfully!")
            except Exception as e:
                print(f"Architecture Mismatch: Old weights format detected. Cannot load!")
                print("Running with randomly initialized network until you retrain on Kaggle.")
        else:
            print(f"WARNING: {weights_path} not found. Running with randomly initialized network!")
            
        self.model.eval()
        self.last_inference_time = time.time()
        
    def predict(self, tensor: torch.Tensor):
        """
        Input: torch.Tensor of shape (1, N, 543, 3)
        Returns: translated string
        """
        # Ensure tensor matches model architecture device
        tensor = tensor.to(self.device)
        
        # Rate limit to emulate word boundaries (don't translate every single frame overlap)
        current_time = time.time()
        if current_time - self.last_inference_time < 2.0:
            return None 
            
        self.last_inference_time = current_time
        
        with torch.no_grad():
            # forward pass: [Batch, num_classes]
            logits = self.model(tensor)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # get argmax for label
            top_prob, top_idx = torch.max(probs, dim=1)
            
            top_idx = top_idx.item()
            confidence = top_prob.item()
            
            word = self.vocab.get(top_idx, f"[Sign_{top_idx}]")
            
            # Provide an uppercase clean string for the HoloLens Subtitle UI
            return f"{word.upper()} ({confidence:.2f})"
