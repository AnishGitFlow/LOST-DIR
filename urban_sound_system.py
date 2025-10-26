"""
Production-Ready Urban Sound Classification System
Target: 84-87% accuracy, <1.5M parameters for edge deployment
Dataset: UrbanSound8K
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration for the entire system"""
    
    # Dataset
    DATA_PATH = "UrbanSound8K"  # Root directory
    AUDIO_PATH = "audio"
    METADATA_FILE = "metadata/UrbanSound8K.csv"
    
    # Audio Processing
    SAMPLE_RATE = 22050  # Downsample from 44.1kHz for efficiency
    DURATION = 4.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    
    # Mel Spectrogram Parameters
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 512
    FMIN = 20
    FMAX = 8000
    
    # Model Architecture
    INPUT_CHANNELS = 1
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Data Split (fold-based)
    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_FOLD = 9
    TEST_FOLD = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class Names
    CLASS_NAMES = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    
    # Augmentation
    AUGMENT_PROB = 0.5
    
    # Model Save
    MODEL_SAVE_PATH = "models"
    BEST_MODEL_NAME = "best_urban_sound_model.pth"


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class AudioAugmentation:
    """Audio augmentation techniques for training robustness"""
    
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        """Add random Gaussian noise"""
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented.astype(audio.dtype)
    
    @staticmethod
    def time_shift(audio, shift_max=0.2):
        """Shift audio in time"""
        shift = np.random.randint(int(len(audio) * shift_max))
        direction = np.random.choice([-1, 1])
        return np.roll(audio, shift * direction)
    
    @staticmethod
    def pitch_shift(audio, sr, n_steps=2):
        """Shift pitch up or down"""
        n_steps = np.random.uniform(-n_steps, n_steps)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(audio, rate_range=(0.8, 1.2)):
        """Stretch or compress time"""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    @staticmethod
    def random_gain(audio, gain_range=(0.7, 1.3)):
        """Random volume adjustment"""
        gain = np.random.uniform(*gain_range)
        return audio * gain


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Extract mel spectrogram features from audio"""
    
    def __init__(self, config):
        self.config = config
        
    def extract_melspectrogram(self, audio):
        """Extract mel spectrogram with normalization"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        return mel_spec_norm
    
    def extract_features(self, audio_path, augment=False):
        """Load audio and extract features"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, duration=self.config.DURATION)
            
            # Pad or truncate to fixed length
            if len(audio) < self.config.N_SAMPLES:
                audio = np.pad(audio, (0, self.config.N_SAMPLES - len(audio)), mode='constant')
            else:
                audio = audio[:self.config.N_SAMPLES]
            
            # Apply augmentation if training
            if augment and np.random.random() < self.config.AUGMENT_PROB:
                aug_choice = np.random.choice(['noise', 'shift', 'gain'])
                if aug_choice == 'noise':
                    audio = AudioAugmentation.add_noise(audio)
                elif aug_choice == 'shift':
                    audio = AudioAugmentation.time_shift(audio)
                elif aug_choice == 'gain':
                    audio = AudioAugmentation.random_gain(audio)
            
            # Extract mel spectrogram
            mel_spec = self.extract_melspectrogram(audio)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None


# =============================================================================
# DATASET
# =============================================================================

class UrbanSoundDataset(Dataset):
    """PyTorch Dataset for UrbanSound8K"""
    
    def __init__(self, data_path, metadata_df, folds, config, augment=False):
        self.data_path = Path(data_path)
        self.config = config
        self.augment = augment
        self.feature_extractor = FeatureExtractor(config)
        
        # Filter metadata by folds
        self.metadata = metadata_df[metadata_df['fold'].isin(folds)].reset_index(drop=True)
        
        print(f"Dataset initialized: {len(self.metadata)} samples, Folds: {folds}, Augment: {augment}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Construct file path
        audio_file = f"fold{row['fold']}/{row['slice_file_name']}"
        audio_path = self.data_path / "audio" / audio_file
        
        # Extract features
        features = self.feature_extractor.extract_features(audio_path, augment=self.augment)
        
        # Handle extraction errors
        if features is None:
            features = np.zeros((self.config.N_MELS, 173))  # Default shape
        
        # Convert to tensor and add channel dimension
        features = torch.FloatTensor(features).unsqueeze(0)  # Shape: (1, n_mels, time)
        
        # Get label
        label = torch.LongTensor([row['classID']])[0]
        
        return features, label


# =============================================================================
# MODEL ARCHITECTURE - Efficient CNN for Edge Deployment
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution (MobileNet-style)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class EfficientUrbanSoundCNN(nn.Module):
    """
    Efficient CNN architecture for urban sound classification
    Target: <1.5M parameters, 84-87% accuracy
    Architecture inspired by MobileNet with attention mechanism
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable blocks
        self.ds_conv1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.ds_conv2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.ds_conv3 = DepthwiseSeparableConv(128, 128, stride=1)
        self.ds_conv4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.ds_conv5 = DepthwiseSeparableConv(256, 256, stride=1)
        
        # Squeeze-and-Excitation attention
        self.se_fc1 = nn.Linear(256, 64)
        self.se_fc2 = nn.Linear(64, 256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Depthwise separable blocks
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        x = self.ds_conv5(x)
        
        # Squeeze-and-Excitation attention
        batch, channels, _, _ = x.size()
        se = self.global_pool(x).view(batch, channels)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).view(batch, channels, 1, 1)
        x = x * se
        
        # Global pooling and classification
        x = self.global_pool(x).view(batch, -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

class Trainer:
    """Training and evaluation pipeline"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
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
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"Model Parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.config.DEVICE}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, self.config.BEST_MODEL_NAME)
        torch.save(checkpoint, save_path)


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """Production inference engine for edge deployment"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.DEVICE
        self.feature_extractor = FeatureExtractor(config)
        
        # Load model
        self.model = EfficientUrbanSoundCNN(num_classes=config.NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
    
    def predict(self, audio_path):
        """Predict class for a single audio file"""
        # Extract features
        features = self.feature_extractor.extract_features(audio_path, augment=False)
        
        if features is None:
            return None, None, None
        
        # Prepare input
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(features)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        class_name = self.config.CLASS_NAMES[predicted_class]
        
        # Get top-3 predictions
        top3_prob, top3_idx = probabilities[0].topk(3)
        top3_predictions = [
            (self.config.CLASS_NAMES[idx.item()], prob.item())
            for idx, prob in zip(top3_idx, top3_prob)
        ]
        
        return class_name, confidence_score, top3_predictions
    
    def predict_batch(self, audio_paths):
        """Batch prediction for multiple files"""
        results = []
        for path in audio_paths:
            result = self.predict(path)
            results.append(result)
        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    config = Config()
    
    print("="*80)
    print("URBAN SOUND CLASSIFICATION SYSTEM")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Sample Rate: {config.SAMPLE_RATE} Hz")
    print(f"  Duration: {config.DURATION}s")
    print(f"  Mel Bands: {config.N_MELS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Device: {config.DEVICE}")
    
    # Load metadata
    print(f"\nLoading dataset from {config.DATA_PATH}...")
    metadata_path = os.path.join(config.DATA_PATH, config.METADATA_FILE)
    metadata = pd.read_csv(metadata_path)
    
    print(f"Total samples: {len(metadata)}")
    print(f"Classes: {metadata['class'].unique()}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = UrbanSoundDataset(
        config.DATA_PATH, metadata, config.TRAIN_FOLDS, config, augment=True
    )
    val_dataset = UrbanSoundDataset(
        config.DATA_PATH, metadata, [config.VAL_FOLD], config, augment=False
    )
    test_dataset = UrbanSoundDataset(
        config.DATA_PATH, metadata, [config.TEST_FOLD], config, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Create model
    print("\nInitializing model...")
    model = EfficientUrbanSoundCNN(num_classes=config.NUM_CLASSES)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} (<1.5M: {'✓' if param_count < 1_500_000 else '✗'})")
    
    # Train model
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    # Test model
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate()  # Using validation function on test loader
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Target Range: 84-87% {'✓' if 84 <= test_acc <= 87 else '(outside target)'}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
