"""
Test script untuk memverifikasi model yang sudah di-train.
"""

import joblib
import numpy as np
from pathlib import Path

def test_trained_model():
    """Test model yang sudah di-train."""
    print("Testing trained alphabetic recognition model...")
    
    model_path = "models/alphabetic_classifier_model.pkl"
    config_path = "models/feature_extractor_config.pkl"
    
    # Check if model files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        # Load model
        print("Loading model...")
        model_data = joblib.load(model_path)
        feature_config = joblib.load(config_path)
        
        print("✓ Model loaded successfully!")
        print(f"  Model type: {model_data['model_type']}")
        print(f"  Accuracy: {model_data['accuracy']:.4f}")
        print(f"  Feature size: {model_data['feature_size']}")
        print(f"  Classes: {model_data['classes']}")
        print(f"  Total classes: {len(model_data['classes'])}")
        
        # Test prediction with random features
        print("\nTesting prediction with random features...")
        classifier = model_data['model']
        feature_size = model_data['feature_size']
        
        # Create random feature vector
        random_features = np.random.rand(1, feature_size)
        
        # Make prediction
        prediction = classifier.predict(random_features)[0]
        
        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(random_features)[0]
            confidence = np.max(probabilities)
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.4f}")
        else:
            print(f"  Prediction: {prediction}")
        
        print("\n✓ Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_trained_model()
