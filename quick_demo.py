#!/usr/bin/env python3
"""
Quick Demo: Urban Sound Classification
Simple inference example for testing the trained model
"""

import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path

# Import custom modules
try:
    from urban_sound_system import Config, EfficientUrbanSoundCNN, FeatureExtractor
except ImportError:
    print("ERROR: Cannot find urban_sound_system.py")
    print("Make sure all required files are in the same directory.")
    sys.exit(1)


class SimpleInference:
    """Simple inference class for quick testing"""
    
    def __init__(self, model_path='models/best_urban_sound_model.pth'):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = EfficientUrbanSoundCNN(num_classes=self.config.NUM_CLASSES)
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Please train the model first using: python train.py")
            sys.exit(1)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor(self.config)
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Validation Accuracy: {checkpoint['accuracy']:.2f}%")
    
    def predict(self, audio_path):
        """Predict class for audio file"""
        print(f"\nProcessing: {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: File not found: {audio_path}")
            return None
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_path)
            
            if features is None:
                print("ERROR: Failed to extract features")
                return None
            
            # Prepare input
            input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            class_name = self.config.CLASS_NAMES[predicted_class]
            
            # Get top-3 predictions
            top3_prob, top3_idx = probabilities[0].topk(3)
            
            # Print results
            print(f"\n{'='*60}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"Top Prediction: {class_name}")
            print(f"Confidence: {confidence_score*100:.2f}%")
            print(f"\nTop 3 Predictions:")
            for i, (idx, prob) in enumerate(zip(top3_idx, top3_prob), 1):
                name = self.config.CLASS_NAMES[idx.item()]
                print(f"  {i}. {name:<20} {prob.item()*100:>6.2f}%")
            print(f"{'='*60}")
            
            return {
                'class_name': class_name,
                'confidence': confidence_score,
                'top3': [(self.config.CLASS_NAMES[idx.item()], prob.item()) 
                         for idx, prob in zip(top3_idx, top3_prob)]
            }
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_directory(self, directory_path, extension='.wav'):
        """Predict all audio files in a directory"""
        print(f"\nSearching for {extension} files in: {directory_path}")
        
        audio_files = list(Path(directory_path).glob(f'*{extension}'))
        
        if not audio_files:
            print(f"No {extension} files found in {directory_path}")
            return
        
        print(f"Found {len(audio_files)} files\n")
        
        results = []
        for audio_file in audio_files:
            result = self.predict(str(audio_file))
            if result:
                results.append((audio_file.name, result))
        
        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"{'File':<30} {'Prediction':<20} {'Confidence':>10}")
        print(f"{'-'*60}")
        for filename, result in results:
            print(f"{filename:<30} {result['class_name']:<20} {result['confidence']*100:>9.2f}%")
        print(f"{'='*60}")


def demo_with_test_audio():
    """Demo with test audio from UrbanSound8K dataset"""
    print("="*60)
    print("URBAN SOUND CLASSIFICATION - QUICK DEMO")
    print("="*60)
    
    # Initialize inference
    inference = SimpleInference()
    
    # Example: Test with files from UrbanSound8K test fold
    test_fold = "UrbanSound8K/audio/fold10"
    
    if os.path.exists(test_fold):
        print(f"\n✓ Found test data: {test_fold}")
        
        # Predict first few files
        test_files = list(Path(test_fold).glob('*.wav'))[:5]
        
        for test_file in test_files:
            inference.predict(str(test_file))
    else:
        print(f"\n⚠ Test fold not found at: {test_fold}")
        print("Please provide path to audio files:")
        print("  python quick_demo.py <audio_file.wav>")
        print("  python quick_demo.py <directory_with_wav_files>")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        # Run demo with test data
        demo_with_test_audio()
    else:
        # User provided path
        path = sys.argv[1]
        
        inference = SimpleInference()
        
        if os.path.isfile(path):
            # Single file
            inference.predict(path)
        elif os.path.isdir(path):
            # Directory
            inference.predict_directory(path)
        else:
            print(f"ERROR: Invalid path: {path}")
            print("Please provide a valid audio file or directory.")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)