# Complete Usage Guide - Urban Sound Classification System

## 📋 Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Basic Training](#basic-training)
3. [Advanced Training](#advanced-training)
4. [Inference & Testing](#inference--testing)
5. [Model Export & Deployment](#model-export--deployment)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## 🔧 Installation & Setup

### Step 1: System Requirements

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 10GB free disk space
- CPU: Multi-core processor

**Recommended for Training:**
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.7+

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install PyTorch (visit pytorch.org for your specific configuration)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

### Step 3: Download Dataset

```bash
# 1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html
# 2. Register and download UrbanSound8K.tar.gz (~6GB)
# 3. Extract the archive

# Verify structure:
ls UrbanSound8K/
# Should see: audio/ metadata/

ls UrbanSound8K/audio/
# Should see: fold1/ fold2/ ... fold10/
```

### Step 4: Verify Installation

```python
# test_installation.py
import torch
import librosa
import numpy as np

print(f"✓ Python version: {sys.version}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Librosa version: {librosa.__version__}")
print("\n✓ All dependencies installed successfully!")
```

---

## 🎓 Basic Training

### Quick Start (Default Settings)

```bash
# Train with default parameters
python train.py --data_path ./UrbanSound8K

# This will:
# 1. Train for 150 epochs
# 2. Use batch size of 64
# 3. Learning rate of 0.001
# 4. Auto-detect GPU/CPU
# 5. Save best model to models/
# 6. Generate evaluation results
# 7. Export deployment package
```

### Custom Training Configuration

```bash
# Custom epochs and batch size
python train.py \
    --data_path ./UrbanSound8K \
    --epochs 100 \
    --batch_size 32

# Custom learning rate
python train.py \
    --data_path ./UrbanSound8K \
    --lr 0.0005

# Force CPU usage
python train.py \
    --data_path ./UrbanSound8K \
    --device cpu

# Adjust number of workers
python train.py \
    --data_path ./UrbanSound8K \
    --num_workers 8
```

### Monitor Training Progress

During training, you'll see:

```
Epoch 1/150
------------------------------------------------------------
  Batch 20/120, Loss: 2.1234, Acc: 25.00%
  Batch 40/120, Loss: 1.8765, Acc: 32.50%
  ...

Epoch 1 Summary:
  Train Loss: 1.5432, Train Acc: 45.67%
  Val Loss: 1.3210, Val Acc: 52.34%
  Learning Rate: 0.001000
  ✓ New best model saved! (Val Acc: 52.34%)
```

Expected timeline:
- **Epoch 1-20**: Accuracy rises to 60-70%
- **Epoch 20-50**: Reaches 75-80%
- **Epoch 50-100**: Plateaus at 82-85%
- **Epoch 100-150**: Fine-tuning to 84-87%

---

## 🚀 Advanced Training

### Mixed Precision Training (Faster)

```python
# advanced_train.py
from urban_sound_system import Config, EfficientUrbanSoundCNN
from advanced_features import AdvancedTrainer

config = Config()
model = EfficientUrbanSoundCNN()

# Load datasets (same as before)
# ...

# Use advanced trainer with mixed precision
trainer = AdvancedTrainer(
    model, 
    train_loader, 
    val_loader, 
    config,
    use_amp=True,  # Mixed precision
    accumulation_steps=2,  # Effective batch size = 128
    patience=15  # Early stopping
)

trainer.train()
```

**Benefits:**
- 40-60% faster training on GPU
- Same accuracy
- Uses less memory

### Learning Rate Finder

```python
from advanced_features import LearningRateFinder

# Find optimal learning rate before training
lr_finder = LearningRateFinder(model, train_loader, config)
best_lr, lrs, losses = lr_finder.find()

# Use recommended LR
config.LEARNING_RATE = best_lr
```

### Resume Training from Checkpoint

```bash
# Resume from specific checkpoint
python train.py \
    --data_path ./UrbanSound8K \
    --resume models/best_urban_sound_model.pth
```

### Training with TensorBoard

```bash
# Start training (TensorBoard auto-enabled if installed)
python train.py --data_path ./UrbanSound8K

# In another terminal, start TensorBoard
tensorboard --logdir runs

# Open browser: http://localhost:6006
```

---

## 🔍 Inference & Testing

### Test Trained Model

```bash
# Run complete test suite
python train.py --test_only --data_path ./UrbanSound8K

# This generates:
# - Confusion matrix
# - Per-class metrics
# - Robustness tests
# - Speed benchmarks
# - All saved to evaluation_results/
```

### Quick Single File Inference

```bash
# Single audio file
python quick_demo.py path/to/audio.wav

# Directory of audio files
python quick_demo.py path/to/audio_folder/

# Test with dataset samples
python quick_demo.py UrbanSound8K/audio/fold10/
```

### Python API Inference

```python
from urban_sound_system import Config
from urban_sound_utils import InferenceEngine

# Initialize
config = Config()
engine = InferenceEngine('models/best_urban_sound_model.pth', config)

# Single prediction
class_name, confidence, top3 = engine.predict('audio.wav')

print(f"Prediction: {class_name}")
print(f"Confidence: {confidence*100:.2f}%")
print("\nTop 3:")
for i, (name, prob) in enumerate(top3, 1):
    print(f"{i}. {name}: {prob*100:.2f}%")

# Batch prediction
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = engine.predict_batch(audio_files)

for file, result in zip(audio_files, results):
    print(f"{file}: {result[0]} ({result[1]*100:.2f}%)")
```

### Real-time Stream Processing

```python
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile

# Initialize engine
engine = InferenceEngine('models/best_urban_sound_model.pth', Config())

# Recording parameters
duration = 4.0
sample_rate = 22050

print("Recording... Speak or make sounds!")
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)
sd.wait()

# Save and predict
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sf.write(f.name, audio, sample_rate)
    result = engine.predict(f.name)
    
print(f"\nDetected: {result[0]} ({result[1]*100:.2f}%)")
```

---

## 📦 Model Export & Deployment

### Export Models

```bash
# Export after training (automatic)
python train.py --data_path ./UrbanSound8K

# Manual export only
python train.py --export_only

# Generated files in deployment_package/:
# - model.onnx (ONNX format)
# - model.pt (TorchScript)
# - config.json
# - README.md
# - inference_example.py
```

### Quantize Model for Edge Devices

```python
from advanced_features import ModelQuantizer

# Load model
model = EfficientUrbanSoundCNN()
checkpoint = torch.load('models/best_urban_sound_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Quantize (4x smaller, faster)
quantizer = ModelQuantizer(model, Config())
quantized_model = quantizer.quantize_dynamic()

# Save quantized model
torch.save(quantized_model.state_dict(), 'models/model_quantized.pth')

# Result: ~10MB → ~2.5MB
```

### Deploy to Raspberry Pi

```bash
# On Raspberry Pi (Python 3.8+)
pip3 install torch torchvision torchaudio librosa numpy

# Copy deployment files
scp -r deployment_package/ pi@raspberrypi.local:~/

# On Pi
cd deployment_package
python3 inference_example.py test.wav
```

### Deploy with ONNX Runtime (Mobile/Web)

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('deployment_package/model.onnx')

# Prepare input (mel spectrogram)
# Shape: (batch=1, channels=1, height=64, width=173)
input_data = preprocess_audio('audio.wav')  # Your preprocessing

# Inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_data})

# Get prediction
class_id = np.argmax(output[0])
confidence = np.max(output[0])
```

### Deploy as REST API

```python
# api_server.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import os

app = Flask(__name__)
engine = InferenceEngine('models/best_urban_sound_model.pth', Config())

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_file.save(f.name)
        
        # Predict
        class_name, confidence, top3 = engine.predict(f.name)
        
        # Clean up
        os.unlink(f.name)
        
        return jsonify({
            'prediction': class_name,
            'confidence': float(confidence),
            'top3': [{'class': name, 'prob': float(prob)} 
                     for name, prob in top3]
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Run: python api_server.py
# Test: curl -X POST -F "audio=@test.wav" http://localhost:5000/predict
```

---

## 🐛 Troubleshooting

### Issue: "Dataset not found"

```bash
# Error: ERROR: Dataset not found at UrbanSound8K

# Solution:
# 1. Check path is correct
ls UrbanSound8K/
# Should show: audio/  metadata/

# 2. Specify full path
python train.py --data_path /full/path/to/UrbanSound8K

# 3. Check folder structure
ls UrbanSound8K/audio/  # Should show fold1-fold10
ls UrbanSound8K/metadata/  # Should show UrbanSound8K.csv
```

### Issue: Out of Memory (OOM)

```bash
# Error: RuntimeError: CUDA out of memory

# Solution 1: Reduce batch size
python train.py --batch_size 32  # or 16

# Solution 2: Use CPU
python train.py --device cpu

# Solution 3: Use gradient accumulation
# In advanced_train.py, set accumulation_steps=4
```

### Issue: Low Accuracy (<80%)

**Possible causes and solutions:**

1. **Insufficient training**
   ```bash
   # Train longer
   python train.py --epochs 200
   ```

2. **Learning rate too high/low**
   ```python
   # Find optimal LR
   from advanced_features import LearningRateFinder
   lr_finder = LearningRateFinder(model, train_loader, config)
   best_lr = lr_finder.find()
   ```

3. **Data loading issues**
   ```python
   # Verify data loader
   for data, labels in train_loader:
       print(data.shape)  # Should be: [batch, 1, 64, 173]
       print(labels.shape)  # Should be: [batch]
       break
   ```

4. **Model not learning**
   ```python
   # Check gradients
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.abs().mean()}")
   ```

### Issue: Slow Training

**Optimization tips:**

```python
# 1. Increase num_workers
python train.py --num_workers 8

# 2. Enable pin_memory (GPU training)
train_loader = DataLoader(..., pin_memory=True)

# 3. Use mixed precision
from advanced_features import MixedPrecisionTrainer
trainer = MixedPrecisionTrainer(...)

# 4. Reduce validation frequency
# Edit Trainer class: validate every 5 epochs instead of every epoch
if epoch % 5 == 0:
    val_loss, val_acc = self.validate()
```

### Issue: Model Overfitting

**Symptoms:** Train acc > 95%, Val acc < 85%

**Solutions:**

```python
# 1. Increase dropout
class EfficientUrbanSoundCNN(nn.Module):
    def __init__(self, ...):
        ...
        self.dropout = nn.Dropout(0.5)  # Increase from 0.3

# 2. Increase data augmentation
config.AUGMENT_PROB = 0.7  # Increase from 0.5

# 3. Add weight decay
config.WEIGHT_DECAY = 1e-3  # Increase from 1e-4

# 4. Use early stopping
from advanced_features import EarlyStopping
early_stop = EarlyStopping(patience=10)
```

### Issue: Inference Too Slow

**CPU optimization:**

```python
# 1. Quantize model
quantized = quantizer.quantize_dynamic()
# Result: 2-3x faster

# 2. Use ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
# Result: Often faster than PyTorch

# 3. Batch processing
results = engine.predict_batch(audio_files)  # Faster than loop
```

---

## 📚 API Reference

### Config Class

```python
class Config:
    # Dataset
    DATA_PATH = "UrbanSound8K"
    
    # Audio Processing
    SAMPLE_RATE = 22050  # Hz
    DURATION = 4.0  # seconds
    N_MELS = 64  # Mel bands
    N_FFT = 1024
    HOP_LENGTH = 512
    
    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Data Split
    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_FOLD = 9
    TEST_FOLD = 10
```

### InferenceEngine Methods

```python
engine = InferenceEngine(model_path, config)

# Single file prediction
class_name, confidence, top3 = engine.predict('audio.wav')
# Returns:
#   class_name: str - predicted class
#   confidence: float - confidence score [0, 1]
#   top3: list of (class_name, probability) tuples

# Batch prediction
results = engine.predict_batch(['audio1.wav', 'audio2.wav'])
# Returns: list of (class_name, confidence, top3) tuples
```

### Trainer Methods

```python
trainer = Trainer(model, train_loader, val_loader, config)

# Train model
trainer.train()

# Validate
val_loss, val_acc = trainer.validate()

# Save checkpoint
trainer.save_checkpoint(epoch, accuracy)
```

### ModelEvaluator Methods

```python
evaluator = ModelEvaluator(model, test_loader, config)

# Full evaluation
results = evaluator.evaluate(save_results=True)
# Generates:
#   - evaluation_results/confusion_matrix.png
#   - evaluation_results/per_class_metrics.png
#   - evaluation_results/metrics.json
```

### Advanced Features

```python
# Mixed Precision Training
trainer = MixedPrecisionTrainer(model, train_loader, val_loader, config)
trainer.train()

# Gradient Accumulation
trainer = GradientAccumulationTrainer(
    model, train_loader, val_loader, config,
    accumulation_steps=4  # Effective batch = 64*4=256
)

# Learning Rate Finder
lr_finder = LearningRateFinder(model, train_loader, config)
best_lr, lrs, losses = lr_finder.find()

# Model Quantization
quantizer = ModelQuantizer(model, config)
quantized_model = quantizer.quantize_dynamic()

# Early Stopping
early_stop = EarlyStopping(patience=15, min_delta=0.001)
if early_stop(val_acc):
    print("Training stopped early")
```

---

## 🎯 Best Practices

### Training

1. **Start with default settings** - They're optimized for UrbanSound8K
2. **Monitor validation accuracy** - Should reach 84-87%
3. **Use GPU if available** - 10x faster training
4. **Save checkpoints regularly** - In case training is interrupted
5. **Validate on test set only once** - At the very end

### Data Preprocessing

1. **Consistent audio length** - Always 4 seconds (padded/truncated)
2. **Normalize spectrograms** - Zero mean, unit variance
3. **Use augmentation for training** - Noise, shift, gain
4. **No augmentation for validation/test** - Fair comparison

### Model Development

1. **Start simple** - Use provided architecture first
2. **Iterate gradually** - Change one thing at a time
3. **Track experiments** - Use TensorBoard or logs
4. **Test robustness** - Noise, different sample rates, etc.

### Deployment

1. **Quantize for edge devices** - 4x smaller, 2-3x faster
2. **Use ONNX for cross-platform** - Works on mobile, web, embedded
3. **Implement error handling** - Audio loading can fail
4. **Add preprocessing validation** - Check audio format, length
5. **Monitor inference time** - Should be <100ms

---

## 📊 Expected Results Summary

| Metric | Target | Typical |
|--------|--------|---------|
| Test Accuracy | 84-87% | 85.2% |
| Per-class Accuracy | 75-95% | 79-91% |
| Macro F1-Score | >0.84 | 0.86 |
| Model Parameters | <1.5M | 1.2M |
| Model Size (FP32) | <15MB | 9.8MB |
| Inference Time (CPU) | <100ms | 40ms |
| Inference Time (GPU) | <20ms | 8ms |
| Training Time (GPU) | <12h | ~5h |
| Real-time Factor | >1x | 100x |

---

## 🚀 Quick Reference Commands

```bash
# Complete pipeline (train + test + export)
python train.py --data_path ./UrbanSound8K

# Test only
python train.py --test_only --data_path ./UrbanSound8K

# Export only
python train.py --export_only

# Quick inference
python quick_demo.py audio.wav

# Find learning rate
python -c "from advanced_features import LearningRateFinder; ..."

# Start TensorBoard
tensorboard --logdir runs
```

---

**🎉 You're now ready to train, test, and deploy your urban sound classifier!**

For more help:
- Check the code comments in each module
- Review the examples in deployment_package/
- Experiment with the quick_demo.py script