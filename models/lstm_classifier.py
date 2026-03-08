"""
DAVID ORACLE — LSTM Sequence Classifier
=========================================
Uses PyTorch LSTM to predict market direction from 10-day feature sequences.
Unlike tree-based models that see each day independently, the LSTM sees
patterns across time (e.g., 5 consecutive red days → bounce likely).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, SIDEWAYS, C

TARGET_MAP = {0: UP, 1: DOWN, 2: SIDEWAYS}


class LSTMNet(nn.Module):
    """2-layer LSTM with dropout for market direction prediction."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMClassifier:
    """
    LSTM-based market direction classifier.
    
    Key difference from tree models:
    - Input is a SEQUENCE of N past days (default: 10)
    - The model learns temporal patterns like momentum shifts, reversal patterns, etc.
    """
    
    def __init__(self, seq_len=10, hidden_size=64, num_layers=2, lr=0.001, epochs=50, batch_size=64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.device = torch.device('cpu')
    
    def _create_sequences(self, X, y=None):
        """Convert flat feature matrix to overlapping sequences."""
        sequences = []
        targets = []
        
        for i in range(self.seq_len, len(X)):
            seq = X[i - self.seq_len : i]
            sequences.append(seq)
            if y is not None:
                targets.append(y[i])
        
        X_seq = np.array(sequences, dtype=np.float32)
        if y is not None:
            y_seq = np.array(targets, dtype=np.int64)
            return X_seq, y_seq
        return X_seq
    
    def train(self, df, feature_cols, verbose=True):
        """Train the LSTM on the full dataset."""
        self.feature_cols = feature_cols
        
        X = df[feature_cols].values
        y = df["target"].values.astype(int)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if verbose:
            print(f"\n  LSTM Training:")
            print(f"    Sequences: {len(X_seq)} | Seq Length: {self.seq_len} | Features: {len(feature_cols)}")
        
        # Build model
        self.model = LSTMNet(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Train/val split (last 20% for validation)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            total_loss = 0
            n_batches = 0
            
            for start in range(0, len(X_train_t), self.batch_size):
                end = min(start + self.batch_size, len(X_train_t))
                batch_idx = indices[start:end]
                
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_pred = torch.argmax(val_out, dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                val_loss = criterion(val_out, y_val_t).item()
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}: loss={total_loss/n_batches:.4f} val_acc={val_acc:.1%}")
            
            # Early stopping
            if patience_counter >= 10:
                if verbose:
                    print(f"    Early stopped at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.is_trained = True
        
        if verbose:
            print(f"    Best validation accuracy: {best_val_acc:.1%}")
        
        return best_val_acc
    
    def predict(self, X_row_or_df):
        """
        Predict direction for a single row (uses last seq_len rows from context)
        or for a full dataframe.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        self.model.eval()
        
        if isinstance(X_row_or_df, pd.DataFrame):
            X = X_row_or_df[self.feature_cols].values
            X_scaled = self.scaler.transform(X)
            X_seq = self._create_sequences(X_scaled)
            
            X_t = torch.FloatTensor(X_seq).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_t)
                probs = torch.softmax(outputs, dim=1).numpy()
                preds = np.argmax(probs, axis=1)
            
            return preds, probs
        else:
            # Single row — need context from training data
            raise ValueError("LSTM requires a DataFrame, not a single row")
    
    def predict_today(self, df):
        """Predict direction for the latest day using the last seq_len days as context."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        self.model.eval()
        
        # Get last seq_len rows
        X = df[self.feature_cols].iloc[-self.seq_len:].values
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        with torch.no_grad():
            output = self.model(X_t)
            probs = torch.softmax(output, dim=1).numpy()[0]
        
        pred_class = int(np.argmax(probs))
        direction = TARGET_MAP[pred_class]
        confidence = float(probs[pred_class])
        
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_up": float(probs[0]),
            "prob_down": float(probs[1]),
            "prob_sideways": float(probs[2]),
            "model": "LSTM"
        }
    
    def save(self, path=None):
        if not self.is_trained:
            return
        path = path or os.path.join(MODEL_DIR, "lstm_classifier.pkl")
        data = {
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
        joblib.dump(data, path)
        print(f"  LSTM saved to {path}")
    
    def load(self, path=None):
        path = path or os.path.join(MODEL_DIR, "lstm_classifier.pkl")
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.scaler = data["scaler"]
            self.feature_cols = data["feature_cols"]
            self.seq_len = data["seq_len"]
            self.hidden_size = data["hidden_size"]
            self.num_layers = data["num_layers"]
            
            self.model = LSTMNet(
                input_size=len(self.feature_cols),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to(self.device)
            self.model.load_state_dict(data["model_state"])
            self.model.eval()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"  Failed to load LSTM: {e}")
            return False
