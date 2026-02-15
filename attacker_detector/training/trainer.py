"""Training loop and utilities."""

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..data import AttackerDataset


class Trainer:
    """
    Trainer class for attacker detection models.
    
    Args:
        model: PyTorch model to train
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Optimizer learning rate
        model_type: Type of model ('mlp', 'gan', 'attention') - affects optimizer config
        epochs: Number of epochs (needed for attention scheduler)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        model_type: str = 'mlp',
        epochs: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.scheduler = None
        
        if model_type == 'gan':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        elif model_type == 'attention':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.criterion = None  
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256
    ) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Calculate class imbalance
        num_benign = (y_train == 0).sum()
        num_attackers = (y_train == 1).sum()
        ratio = num_benign / max(num_attackers, 1)
        print(f"Class Imbalance Ratio: 1 Attacker : {ratio:.2f} Benign")
        
        # Set up weighted loss
        pos_weight = torch.tensor([ratio]).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Create data loader
        train_dataset = AttackerDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training loop
        print(f"\nStarting Training ({epochs} epochs)...")
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for attention model
                if self.model_type == 'attention':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Step scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            else:
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
        
        print("Training complete!")
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 4096,
        label: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate the model and return aggregate metrics.

        Args:
            X: Scaled feature array.
            y: Ground-truth labels.
            batch_size: Batch size for inference.
            label: Label for print output (e.g. 'Test', 'Eval').

        Returns:
            Dict with Accuracy, Precision, Recall, F1_Score.
        """
        self.model.eval()
        all_probs = []

        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch))
                all_probs.append(probs.cpu().numpy().flatten())
            del batch

        probs = np.concatenate(all_probs)
        preds = (probs > 0.5).astype(int)

        metrics = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1_Score': f1_score(y, preds, zero_division=0),
        }

        print(f"\n{label} Metrics ({len(y)} samples):")
        for name, val in metrics.items():
            print(f"  {name.replace('_', ' ')}: {val:.4f}")

        return metrics

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from: {path}")
