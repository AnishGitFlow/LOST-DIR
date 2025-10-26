"""
Advanced Features for Urban Sound Classification
- Mixed Precision Training
- Gradient Accumulation
- Model Quantization
- TensorBoard Integration
- Early Stopping
- Learning Rate Finder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from collections import defaultdict
import json
import os


# =============================================================================
# MIXED PRECISION TRAINING
# =============================================================================

class MixedPrecisionTrainer:
    """Training with automatic mixed precision for faster training"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        self.best_val_acc = 0.0
        self.history = defaultdict(list)
    
    def train_epoch(self):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.train_loader), 100. * correct / total


# =============================================================================
# GRADIENT ACCUMULATION
# =============================================================================

class GradientAccumulationTrainer:
    """Training with gradient accumulation for large effective batch sizes"""
    
    def __init__(self, model, train_loader, val_loader, config, accumulation_steps=4):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.accumulation_steps = accumulation_steps
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        print(f"Gradient Accumulation: {accumulation_steps} steps")
        print(f"Effective Batch Size: {config.BATCH_SIZE * accumulation_steps}")
    
    def train_epoch(self):
        """Train with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target) / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.train_loader), 100. * correct / total


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# =============================================================================
# LEARNING RATE FINDER
# =============================================================================

class LearningRateFinder:
    """Find optimal learning rate using Leslie Smith's method"""
    
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
    
    def find(self, start_lr=1e-7, end_lr=1, num_iter=100):
        """Find optimal learning rate"""
        print("\n" + "="*60)
        print("LEARNING RATE FINDER")
        print("="*60)
        
        lrs = []
        losses = []
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        
        # Learning rate schedule
        lr_lambda = lambda x: np.exp(x * np.log(end_lr / start_lr) / num_iter)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        self.model.train()
        iterator = iter(self.train_loader)
        
        for iteration in range(num_iter):
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                data, target = next(iterator)
            
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record
            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            
            # Stop if loss explodes
            if iteration > 10 and losses[-1] > 4 * min(losses):
                break
        
        # Find best learning rate
        best_idx = np.argmin(losses)
        best_lr = lrs[best_idx]
        
        # Recommend learning rate (typically 1/10th of best)
        recommended_lr = best_lr / 10
        
        print(f"\nResults:")
        print(f"  Best LR: {best_lr:.6f}")
        print(f"  Recommended LR: {recommended_lr:.6f}")
        print(f"  Current LR: {self.config.LEARNING_RATE:.6f}")
        
        # Plot (optional)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(lrs, losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.axvline(best_lr, color='r', linestyle='--', label=f'Best LR: {best_lr:.6f}')
            plt.axvline(recommended_lr, color='g', linestyle='--', label=f'Recommended: {recommended_lr:.6f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('lr_finder.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: lr_finder.png")
        except:
            pass
        
        return recommended_lr, lrs, losses


# =============================================================================
# MODEL QUANTIZATION
# =============================================================================

class ModelQuantizer:
    """Quantize model for edge deployment"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def quantize_dynamic(self):
        """Dynamic quantization (post-training)"""
        print("\n" + "="*60)
        print("DYNAMIC QUANTIZATION")
        print("="*60)
        
        # Quantize
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Compare sizes
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(quantized_model)
        
        print(f"\nResults:")
        print(f"  Original Size: {original_size:.2f} MB")
        print(f"  Quantized Size: {quantized_size:.2f} MB")
        print(f"  Compression: {original_size/quantized_size:.2f}x")
        
        return quantized_model
    
    def quantize_static(self, calibration_loader):
        """Static quantization (requires calibration)"""
        print("\n" + "="*60)
        print("STATIC QUANTIZATION")
        print("="*60)
        
        # Prepare model
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate
        print("Calibrating...")
        with torch.no_grad():
            for data, _ in calibration_loader:
                data = data.to(self.config.DEVICE)
                self.model(data)
        
        # Convert
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(quantized_model)
        
        print(f"\nResults:")
        print(f"  Original Size: {original_size:.2f} MB")
        print(f"  Quantized Size: {quantized_size:.2f} MB")
        print(f"  Compression: {original_size/quantized_size:.2f}x")
        
        return quantized_model
    
    def _get_model_size(self, model):
        """Calculate model size in MB"""
        torch.save(model.state_dict(), "temp_model.pth")
        size = os.path.getsize("temp_model.pth") / 1e6
        os.remove("temp_model.pth")
        return size


# =============================================================================
# TENSORBOARD INTEGRATION
# =============================================================================

class TensorBoardLogger:
    """Log training metrics to TensorBoard"""
    
    def __init__(self, log_dir='runs'):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            print(f"✓ TensorBoard logging enabled: {log_dir}")
        except ImportError:
            print("⚠ TensorBoard not available (install with: pip install tensorboard)")
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, values, step):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_model_graph(self, model, input_size):
        """Log model architecture"""
        if self.enabled:
            dummy_input = torch.randn(input_size)
            self.writer.add_graph(model, dummy_input)
    
    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()


# =============================================================================
# ADVANCED TRAINER WITH ALL FEATURES
# =============================================================================

class AdvancedTrainer:
    """Full-featured trainer with all optimizations"""
    
    def __init__(self, model, train_loader, val_loader, config, 
                 use_amp=True, accumulation_steps=1, patience=15):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.use_amp = use_amp and config.DEVICE.type == 'cuda'
        self.accumulation_steps = accumulation_steps
        
        # Optimizer and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
        
        # TensorBoard
        self.logger = TensorBoardLogger()
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = defaultdict(list)
        
        print(f"\nAdvanced Trainer Configuration:")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Accumulation: {accumulation_steps}x")
        print(f"  Early Stopping: {patience} epochs")
        print(f"  Effective Batch Size: {config.BATCH_SIZE * accumulation_steps}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.accumulation_steps
                loss.backward()
            
            # Update weights with gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Log to TensorBoard
        self.logger.log_scalar('Loss/train', avg_loss, epoch)
        self.logger.log_scalar('Accuracy/train', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Log to TensorBoard
        self.logger.log_scalar('Loss/val', avg_loss, epoch)
        self.logger.log_scalar('Accuracy/val', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop with all features"""
        print(f"\n{'='*80}")
        print("STARTING ADVANCED TRAINING")
        print(f"{'='*80}\n")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.logger.log_scalar('Learning_Rate', current_lr, epoch)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping
            if self.early_stopping(val_acc):
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
            
            print()
        
        self.logger.close()
        print(f"\n{'='*80}")
        print(f"Training Complete! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, 
                                 self.config.BEST_MODEL_NAME)
        torch.save(checkpoint, save_path)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Advanced features module for Urban Sound Classification")
    print("\nAvailable features:")
    print("  - MixedPrecisionTrainer")
    print("  - GradientAccumulationTrainer")
    print("  - EarlyStopping")
    print("  - LearningRateFinder")
    print("  - ModelQuantizer")
    print("  - TensorBoardLogger")
    print("  - AdvancedTrainer (all features combined)")
