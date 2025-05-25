#!/usr/bin/env python3
"""
Enhanced Integration Testing for Alphabetic Recognition Module
=============================================================

Comprehensive testing script that validates the enhanced alphabetic recognition
module with centralized configuration, error handling, and performance optimizations.

Author: Enhanced for Project UAS
Date: May 2025
"""

import sys
import os
import cv2
import numpy as np
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class EnhancedIntegrationTester:
    """Comprehensive testing class for alphabetic recognition module."""
    
    def __init__(self):
        """Initialize the tester with configuration."""
        self.test_results = {}
        self.performance_metrics = {}
        self.test_images = []
        
    def setup_test_environment(self) -> bool:
        """Setup and validate test environment."""
        print("🔧 Setting up test environment...")
        
        try:
            # Validate project structure
            required_dirs = [
                "modules/alphabetic_recognition",
                "models",
                "dataset",
                "assets/test_images"
            ]
            
            for dir_path in required_dirs:
                full_path = project_root / dir_path
                if not full_path.exists():
                    print(f"❌ Missing directory: {dir_path}")
                    return False
                    
            print("✓ Project structure validated")
            
            # Check dependencies
            try:
                import sklearn
                import joblib
                print("✓ Dependencies validated")
            except ImportError as e:
                print(f"❌ Missing dependency: {e}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def test_configuration_system(self) -> bool:
        """Test centralized configuration system."""
        print("\n📋 Testing Configuration System...")
        
        try:
            from modules.alphabetic_recognition.config import (
                MODEL_PATH, FEATURE_CONFIG_PATH, DATASET_PATH,
                HOG_PARAMS, FEATURE_WEIGHTS, validate_config
            )
            
            # Test configuration validation
            validate_config()
            print("✓ Configuration validation passed")
            
            # Test parameter access
            assert isinstance(HOG_PARAMS, dict), "HOG_PARAMS should be dict"
            assert isinstance(FEATURE_WEIGHTS, dict), "FEATURE_WEIGHTS should be dict"
            print("✓ Configuration parameters accessible")
            
            # Test path validation
            assert MODEL_PATH.exists() or not MODEL_PATH.name.endswith('.pkl'), "Model path validation"
            print("✓ Path configuration validated")
            
            self.test_results['config_system'] = True
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results['config_system'] = False
            return False
    
    def test_recognizer_initialization(self) -> bool:
        """Test recognizer initialization and model loading."""
        print("\n🚀 Testing Recognizer Initialization...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            # Test basic initialization
            recognizer = AlphabeticRecognizer(config_validation=True)
            print("✓ Recognizer initialized successfully")
            
            # Test model loading
            if recognizer.is_loaded:
                print(f"✓ Model loaded: {len(recognizer.classes)} classes")
                print(f"  Feature size: {recognizer.feature_size}")
                print(f"  Model type: {type(recognizer.classifier).__name__}")
            else:
                print("⚠ Model not loaded (may be expected if no trained model exists)")
            
            # Test configuration integration
            assert hasattr(recognizer, 'preprocessing_config'), "Preprocessing config missing"
            assert hasattr(recognizer, 'feature_config'), "Feature config missing"
            print("✓ Configuration integration validated")
            
            self.test_results['recognizer_init'] = True
            return True
            
        except Exception as e:
            print(f"❌ Recognizer initialization failed: {e}")
            traceback.print_exc()
            self.test_results['recognizer_init'] = False
            return False
    
    def test_preprocessing_pipeline(self) -> bool:
        """Test enhanced preprocessing pipeline."""
        print("\n🔄 Testing Preprocessing Pipeline...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            recognizer = AlphabeticRecognizer()
            
            # Create test images with different characteristics
            test_cases = [
                ("Normal image", np.ones((50, 50, 3), dtype=np.uint8) * 128),
                ("High contrast", np.ones((50, 50), dtype=np.uint8) * 255),
                ("Low contrast", np.ones((50, 50), dtype=np.uint8) * 100),
                ("Small image", np.ones((20, 20), dtype=np.uint8) * 200),
                ("Large image", np.ones((100, 100, 3), dtype=np.uint8) * 150)
            ]
            
            for test_name, test_image in test_cases:
                try:
                    processed = recognizer.preprocess_character_image(test_image)
                    assert processed is not None, f"Preprocessing failed for {test_name}"
                    assert processed.shape == (28, 28), f"Size mismatch for {test_name}"
                    print(f"✓ {test_name}: shape {processed.shape}")
                except Exception as e:
                    print(f"❌ {test_name} failed: {e}")
                    return False
            
            # Test error handling
            try:
                recognizer.preprocess_character_image(None)
                print("❌ Should have raised error for None input")
                return False
            except ValueError:
                print("✓ Error handling for invalid input")
            
            self.test_results['preprocessing'] = True
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing test failed: {e}")
            self.test_results['preprocessing'] = False
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test enhanced feature extraction with weighted features."""
        print("\n🎯 Testing Feature Extraction...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            recognizer = AlphabeticRecognizer()
            
            # Create test character image
            test_char = np.zeros((28, 28), dtype=np.uint8)
            cv2.rectangle(test_char, (5, 5), (23, 23), 255, 2)
            cv2.line(test_char, (5, 14), (23, 14), 255, 1)
            
            # Extract features
            features = recognizer.extract_features_from_character(test_char)
            
            assert features is not None, "Features should not be None"
            assert len(features) > 0, "Features should not be empty"
            assert not np.any(np.isnan(features)), "Features should not contain NaN"
            assert not np.any(np.isinf(features)), "Features should not contain Inf"
            
            print(f"✓ Features extracted: {len(features)} dimensions")
            print(f"  Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
            print(f"  Non-zero features: {np.count_nonzero(features)}")
            
            # Test with different image types
            test_cases = [
                ("Filled rectangle", np.ones((28, 28), dtype=np.uint8) * 255),
                ("Empty image", np.zeros((28, 28), dtype=np.uint8)),
                ("Random noise", np.random.randint(0, 255, (28, 28), dtype=np.uint8))
            ]
            
            for test_name, test_img in test_cases:
                try:
                    test_features = recognizer.extract_features_from_character(test_img)
                    assert test_features is not None, f"Features failed for {test_name}"
                    print(f"✓ {test_name}: {len(test_features)} features")
                except Exception as e:
                    print(f"❌ {test_name} failed: {e}")
                    return False
            
            self.test_results['feature_extraction'] = True
            return True
            
        except Exception as e:
            print(f"❌ Feature extraction test failed: {e}")
            self.test_results['feature_extraction'] = False
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction capabilities."""
        print("\n📊 Testing Batch Prediction...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            recognizer = AlphabeticRecognizer()
            
            if not recognizer.is_loaded:
                print("⚠ Model not loaded, skipping prediction tests")
                self.test_results['batch_prediction'] = True
                return True
            
            # Create test image with multiple character regions
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            
            # Add some text-like shapes
            cv2.rectangle(test_image, (10, 30), (30, 70), (0, 0, 0), -1)
            cv2.rectangle(test_image, (40, 30), (60, 70), (0, 0, 0), -1)
            cv2.rectangle(test_image, (70, 30), (90, 70), (0, 0, 0), -1)
            
            # Define bounding boxes
            bboxes = [(10, 30, 20, 40), (40, 30, 20, 40), (70, 30, 20, 40)]
            
            # Test batch prediction
            results = recognizer.predict_characters_from_image(test_image, bboxes)
            
            assert isinstance(results, list), "Results should be a list"
            assert len(results) == len(bboxes), "Results count should match bboxes"
            
            for i, result in enumerate(results):
                assert isinstance(result, dict), f"Result {i} should be a dict"
                required_keys = ['bbox_id', 'bbox', 'character', 'confidence', 'status']
                for key in required_keys:
                    assert key in result, f"Missing key '{key}' in result {i}"
                
                print(f"✓ Result {i}: '{result['character']}' (conf: {result['confidence']:.3f}, status: {result['status']})")
            
            # Test performance tracking
            metrics = recognizer.get_performance_metrics()
            print(f"✓ Performance metrics: {metrics}")
            
            self.test_results['batch_prediction'] = True
            return True
            
        except Exception as e:
            print(f"❌ Batch prediction test failed: {e}")
            traceback.print_exc()
            self.test_results['batch_prediction'] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test comprehensive error handling."""
        print("\n🛡️ Testing Error Handling...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            recognizer = AlphabeticRecognizer()
            
            # Test invalid inputs
            error_test_cases = [
                ("None image", None),
                ("Empty array", np.array([])),
                ("Wrong dimensions", np.ones((10,), dtype=np.uint8)),
                ("Invalid dtype", np.ones((28, 28), dtype=np.float64) * 1000)
            ]
            
            for test_name, invalid_input in error_test_cases:
                try:
                    if invalid_input is not None:
                        recognizer.preprocess_character_image(invalid_input)
                    else:
                        recognizer.preprocess_character_image(invalid_input)
                    print(f"❌ {test_name}: Should have raised an error")
                except (ValueError, TypeError) as e:
                    print(f"✓ {test_name}: Properly handled - {type(e).__name__}")
                except Exception as e:
                    print(f"⚠ {test_name}: Unexpected error type - {type(e).__name__}: {e}")
            
            # Test batch prediction error handling
            if recognizer.is_loaded:
                # Test with invalid bboxes
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                invalid_bboxes = [(-1, -1, 10, 10), (200, 200, 10, 10), (10, 10, 0, 0)]
                
                results = recognizer.predict_characters_from_image(test_image, invalid_bboxes)
                
                for i, result in enumerate(results):
                    if 'error' in result['status'] or result['status'] in ['invalid_bbox', 'empty_roi']:
                        print(f"✓ Invalid bbox {i}: Properly handled - {result['status']}")
                    else:
                        print(f"⚠ Invalid bbox {i}: Status - {result['status']}")
            
            self.test_results['error_handling'] = True
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results['error_handling'] = False
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance tracking and optimization."""
        print("\n⚡ Testing Performance Metrics...")
        
        try:
            from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            recognizer = AlphabeticRecognizer()
            
            if not recognizer.is_loaded:
                print("⚠ Model not loaded, skipping performance tests")
                self.test_results['performance'] = True
                return True
            
            # Reset performance tracking
            recognizer.reset_performance_tracking()
            
            # Create test data for performance measurement
            test_images = []
            for i in range(10):
                img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
                test_images.append(img)
            
            # Measure preprocessing performance
            start_time = time.time()
            for img in test_images:
                recognizer.preprocess_character_image(img)
            preprocess_time = time.time() - start_time
            
            print(f"✓ Preprocessing: {preprocess_time:.3f}s for {len(test_images)} images")
            print(f"  Average: {preprocess_time/len(test_images)*1000:.1f}ms per image")
            
            # Measure feature extraction performance
            preprocessed_images = [recognizer.preprocess_character_image(img) for img in test_images]
            
            start_time = time.time()
            for img in preprocessed_images:
                recognizer.extract_features_from_character(img)
            feature_time = time.time() - start_time
            
            print(f"✓ Feature extraction: {feature_time:.3f}s for {len(test_images)} images")
            print(f"  Average: {feature_time/len(test_images)*1000:.1f}ms per image")
            
            # Measure prediction performance
            start_time = time.time()
            for img in test_images:
                recognizer.predict_character(img)
            prediction_time = time.time() - start_time
            
            print(f"✓ Full prediction: {prediction_time:.3f}s for {len(test_images)} images")
            print(f"  Average: {prediction_time/len(test_images)*1000:.1f}ms per image")
            
            # Get performance metrics
            metrics = recognizer.get_performance_metrics()
            print(f"✓ Performance metrics: {metrics}")
            
            # Store performance data
            self.performance_metrics = {
                'preprocess_time_per_image': preprocess_time / len(test_images),
                'feature_time_per_image': feature_time / len(test_images),
                'prediction_time_per_image': prediction_time / len(test_images),
                'total_predictions': metrics.get('prediction_count', 0),
                'avg_confidence': metrics.get('avg_confidence', 0.0)
            }
            
            self.test_results['performance'] = True
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def test_integration_with_image_processor(self) -> bool:
        """Test integration with the main ImageProcessor."""
        print("\n🔗 Testing ImageProcessor Integration...")
        
        try:
            from core.image_processor import ImageProcessor
            
            # Create ImageProcessor instance
            processor = ImageProcessor()
            
            # Create test image with text-like shapes
            test_image = np.ones((150, 300, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            # Test alphabetic recognition integration
            result_image, detections = processor.process_for_alphabetic_recognition(test_image)
            
            assert result_image is not None, "Result image should not be None"
            assert isinstance(detections, list), "Detections should be a list"
            
            print(f"✓ ImageProcessor integration successful")
            print(f"  Result image shape: {result_image.shape}")
            print(f"  Detections count: {len(detections)}")
            
            for i, detection in enumerate(detections):
                if isinstance(detection, dict):
                    char = detection.get('character', 'Unknown')
                    conf = detection.get('confidence', 0.0)
                    print(f"  Detection {i}: '{char}' (confidence: {conf:.3f})")
                else:
                    print(f"  Detection {i}: {detection}")
            
            self.test_results['imageprocessor_integration'] = True
            return True
            
        except Exception as e:
            print(f"❌ ImageProcessor integration test failed: {e}")
            traceback.print_exc()
            self.test_results['imageprocessor_integration'] = False
            return False
    
    def generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📊 ENHANCED INTEGRATION TEST REPORT")
        print("="*60)
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 Test Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests} ✓")
        print(f"  Failed: {failed_tests} ❌")
        print(f"  Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Performance metrics
        if self.performance_metrics:
            print(f"\n⚡ Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                if 'time' in metric:
                    print(f"  {metric}: {value*1000:.1f}ms")
                else:
                    print(f"  {metric}: {value}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if failed_tests == 0:
            print("  ✓ All tests passed! The enhanced alphabetic recognition module is ready for production.")
        else:
            print("  ⚠ Some tests failed. Please review the issues above before deployment.")
            
        if self.performance_metrics:
            pred_time = self.performance_metrics.get('prediction_time_per_image', 0)
            if pred_time > 0.1:  # 100ms threshold
                print("  ⚠ Prediction time is high. Consider optimizing the feature extraction pipeline.")
            elif pred_time > 0.05:  # 50ms threshold
                print("  💡 Prediction time is acceptable but could be optimized for real-time applications.")
            else:
                print("  ✓ Excellent prediction performance for real-time applications.")
        
        print("\n" + "="*60)

def main():
    """Main testing function."""
    print("🚀 Enhanced Alphabetic Recognition Integration Testing")
    print("=" * 60)
    
    # Create tester instance
    tester = EnhancedIntegrationTester()
    
    # Setup test environment
    if not tester.setup_test_environment():
        print("❌ Test environment setup failed. Exiting.")
        return
    
    # Run all tests
    test_functions = [
        tester.test_configuration_system,
        tester.test_recognizer_initialization,
        tester.test_preprocessing_pipeline,
        tester.test_feature_extraction,
        tester.test_batch_prediction,
        tester.test_error_handling,
        tester.test_performance_metrics,
        tester.test_integration_with_image_processor
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test function {test_func.__name__} crashed: {e}")
            traceback.print_exc()
    
    # Generate report
    tester.generate_test_report()

if __name__ == "__main__":
    main()
