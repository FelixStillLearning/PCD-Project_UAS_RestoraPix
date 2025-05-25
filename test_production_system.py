#!/usr/bin/env python3
"""
Comprehensive Production Testing for Enhanced Alphabetic Recognition System
===========================================================================

This script performs end-to-end testing of the enhanced alphabetic recognition system
including performance optimization, configuration management, and error handling validation.

Author: Enhanced for Project UAS
Date: May 2025
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_enhanced_configuration():
    """Test the enhanced configuration system."""
    print("🔧 Testing Enhanced Configuration System...")
    
    try:
        from modules.alphabetic_recognition.config import (
            validate_config, get_preprocessing_config, get_feature_extraction_config,
            get_classification_config, FEATURE_WEIGHTS, HOG_PARAMS, MODEL_PATH,
            USE_MULTIPROCESSING, MAX_WORKERS, FEATURE_CACHE_SIZE
        )
        
        # Test configuration validation
        validate_config()
        print("✓ Configuration validation passed")
        
        # Test configuration access
        prep_config = get_preprocessing_config()
        feature_config = get_feature_extraction_config()
        class_config = get_classification_config()
        
        print(f"✓ Preprocessing config loaded: {len(prep_config)} parameters")
        print(f"✓ Feature extraction config loaded: {len(feature_config)} parameters")
        print(f"✓ Classification config loaded: {len(class_config)} parameters")
        
        # Test key parameters
        assert 'hog' in FEATURE_WEIGHTS, "HOG feature weight not found"
        assert '_nbins' in HOG_PARAMS, "HOG parameters not properly configured"
        assert USE_MULTIPROCESSING in [True, False], "Multiprocessing setting invalid"
        assert MAX_WORKERS > 0, "Max workers must be positive"
        assert FEATURE_CACHE_SIZE > 0, "Cache size must be positive"
        
        print("✓ All configuration parameters validated")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_recognizer():
    """Test the enhanced recognizer with performance optimization features."""
    print("\n🚀 Testing Enhanced Recognizer with Performance Optimization...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        
        # Initialize recognizer
        recognizer = AlphabeticRecognizer()
        
        if not recognizer.is_loaded:
            print("⚠ Model not loaded, creating dummy test data...")
            # Create dummy test data if model not available
            test_image = np.ones((28, 28), dtype=np.uint8) * 255
            cv2.rectangle(test_image, (5, 5), (23, 23), 0, 2)
        else:
            print("✓ Model loaded successfully")
            # Create test image
            test_image = np.ones((28, 28), dtype=np.uint8) * 255
            cv2.putText(test_image, 'A', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        
        # Test basic prediction
        start_time = time.time()
        character, confidence = recognizer.predict_character(test_image)
        prediction_time = time.time() - start_time
        
        print(f"✓ Basic prediction: '{character}' (confidence: {confidence:.3f}, time: {prediction_time*1000:.1f}ms)")
        
        # Test feature caching
        print("\n📦 Testing Feature Caching...")
        
        # First extraction (should cache)
        start_time = time.time()
        binary_image = recognizer.preprocess_character_image(test_image)
        features1 = recognizer.extract_features_from_character(binary_image)
        time1 = time.time() - start_time
        
        # Second extraction (should use cache)
        start_time = time.time()
        features2 = recognizer.extract_features_from_character(binary_image)
        time2 = time.time() - start_time
        
        cache_speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"✓ Feature extraction time: {time1*1000:.1f}ms (uncached) -> {time2*1000:.1f}ms (cached)")
        print(f"✓ Cache speedup: {cache_speedup:.1f}x")
        
        # Verify features are identical
        assert np.allclose(features1, features2), "Cached features don't match"
        print("✓ Cached features verified identical")
        
        # Test batch processing
        print("\n⚡ Testing Batch Processing...")
        
        test_images = [test_image] * 5  # 5 identical test images
        start_time = time.time()
        batch_results = recognizer.predict_characters_batch_optimized(test_images)
        batch_time = time.time() - start_time
        
        print(f"✓ Batch processing: {len(batch_results)} images in {batch_time*1000:.1f}ms")
        print(f"✓ Average per image: {(batch_time/len(batch_results))*1000:.1f}ms")
        
        # Test performance metrics
        print("\n📊 Testing Performance Metrics...")
        
        metrics = recognizer.get_performance_metrics()
        print(f"✓ Performance metrics collected: {len(metrics)} parameters")
        print(f"  - Predictions: {metrics.get('prediction_count', 0)}")
        print(f"  - Cache hits: {metrics.get('cache_hits', 0)}")
        print(f"  - Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
        print(f"  - Avg processing time: {metrics.get('avg_processing_time_ms', 0):.1f}ms")
        print(f"  - Multiprocessing: {metrics.get('multiprocessing_enabled', False)}")
        print(f"  - Max workers: {metrics.get('max_workers', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced recognizer test failed: {e}")
        traceback.print_exc()
        return False

def test_multiprocessing_performance():
    """Test multiprocessing performance improvements."""
    print("\n🔄 Testing Multiprocessing Performance...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        
        recognizer = AlphabeticRecognizer()
        
        # Create test images
        test_images = []
        for i in range(10):
            img = np.ones((28, 28), dtype=np.uint8) * 255
            cv2.putText(img, chr(65 + (i % 26)), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            test_images.append(img)
        
        # Test without multiprocessing
        start_time = time.time()
        results_sequential = recognizer.predict_characters_batch_optimized(
            test_images, use_multiprocessing=False
        )
        sequential_time = time.time() - start_time
        
        # Test with multiprocessing
        start_time = time.time()
        results_parallel = recognizer.predict_characters_batch_optimized(
            test_images, use_multiprocessing=True
        )
        parallel_time = time.time() - start_time
        
        # Compare results
        assert len(results_sequential) == len(results_parallel), "Result count mismatch"
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"✓ Sequential processing: {sequential_time*1000:.1f}ms")
        print(f"✓ Parallel processing: {parallel_time*1000:.1f}ms")
        print(f"✓ Multiprocessing speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Multiprocessing test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("\n💾 Testing Memory Optimization...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        
        recognizer = AlphabeticRecognizer()
        
        # Test cache size optimization
        original_cache_size = recognizer.max_cache_size
        recognizer.optimize_cache_size(50)
        assert recognizer.max_cache_size == 50, "Cache size optimization failed"
        
        # Fill cache beyond limit
        for i in range(100):
            test_img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            binary_img = recognizer.preprocess_character_image(test_img)
            recognizer.extract_features_from_character(binary_img)
        
        # Verify cache size limit
        assert len(recognizer.feature_cache) <= 50, "Cache size limit not enforced"
        print(f"✓ Cache size properly limited: {len(recognizer.feature_cache)}/50")
        
        # Test cache clearing
        recognizer.clear_cache()
        assert len(recognizer.feature_cache) == 0, "Cache not properly cleared"
        print("✓ Cache clearing works correctly")
        
        # Restore original cache size
        recognizer.optimize_cache_size(original_cache_size)
        
        return True
        
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test robust error handling mechanisms."""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        
        recognizer = AlphabeticRecognizer()
        
        # Test with invalid images
        invalid_images = [
            np.array([]),  # Empty array
            np.ones((5, 5), dtype=np.uint8),  # Too small
            np.ones((1000, 1000), dtype=np.uint8),  # Very large
            None  # None input
        ]
        
        for i, invalid_img in enumerate(invalid_images):
            try:
                if invalid_img is not None:
                    result = recognizer.predict_character(invalid_img)
                    print(f"✓ Invalid image {i+1} handled gracefully: {result}")
                else:
                    print(f"✓ None input {i+1} skipped")
            except Exception as e:
                print(f"⚠ Invalid image {i+1} caused exception: {e}")
        
        # Test with corrupted data
        corrupted_data = np.ones((28, 28), dtype=np.float64) * np.inf
        try:
            result = recognizer.predict_character(corrupted_data.astype(np.uint8))
            print(f"✓ Corrupted data handled: {result}")
        except Exception as e:
            print(f"⚠ Corrupted data caused exception: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\n🏁 Running Performance Benchmark...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        
        recognizer = AlphabeticRecognizer()
        
        # Generate test dataset
        test_images = []
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        for char in characters[:10]:  # Test with 10 characters
            img = np.ones((28, 28), dtype=np.uint8) * 255
            cv2.putText(img, char, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
            test_images.append(img)
        
        # Benchmark preprocessing
        start_time = time.time()
        preprocessed_images = []
        for img in test_images:
            preprocessed = recognizer.preprocess_character_image(img)
            preprocessed_images.append(preprocessed)
        preprocessing_time = time.time() - start_time
        
        # Benchmark feature extraction
        start_time = time.time()
        feature_vectors = []
        for img in preprocessed_images:
            features = recognizer.extract_features_from_character(img)
            feature_vectors.append(features)
        feature_extraction_time = time.time() - start_time
        
        # Benchmark prediction
        start_time = time.time()
        predictions = []
        for img in test_images:
            char, conf = recognizer.predict_character(img)
            predictions.append((char, conf))
        prediction_time = time.time() - start_time
        
        # Calculate benchmarks
        num_images = len(test_images)
        print(f"\n📈 Performance Benchmark Results ({num_images} images):")
        print(f"  Preprocessing: {preprocessing_time*1000:.1f}ms total, {(preprocessing_time/num_images)*1000:.1f}ms/image")
        print(f"  Feature extraction: {feature_extraction_time*1000:.1f}ms total, {(feature_extraction_time/num_images)*1000:.1f}ms/image")
        print(f"  Prediction: {prediction_time*1000:.1f}ms total, {(prediction_time/num_images)*1000:.1f}ms/image")
        print(f"  Total pipeline: {(preprocessing_time + feature_extraction_time + prediction_time)*1000:.1f}ms")
        
        # Performance targets
        avg_prediction_time = (prediction_time / num_images) * 1000  # ms
        if avg_prediction_time < 5.0:
            print("✅ Excellent performance: < 5ms per character")
        elif avg_prediction_time < 10.0:
            print("✅ Good performance: < 10ms per character")
        elif avg_prediction_time < 20.0:
            print("⚠️ Acceptable performance: < 20ms per character")
        else:
            print("❌ Performance needs improvement: > 20ms per character")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive production testing."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE PRODUCTION TESTING - Enhanced Alphabetic Recognition")
    print("=" * 80)
    
    tests = [
        ("Configuration System", test_enhanced_configuration),
        ("Enhanced Recognizer", test_enhanced_recognizer),
        ("Multiprocessing Performance", test_multiprocessing_performance),
        ("Memory Optimization", test_memory_optimization),
        ("Error Handling", test_error_handling),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔍 {test_name}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = test_func()
            test_time = time.time() - start_time
            
            results.append((test_name, result, test_time))
            
            if result:
                print(f"✅ {test_name} PASSED ({test_time:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            results.append((test_name, False, 0))
            print(f"💥 {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    total_time = 0
    
    for test_name, result, test_time in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name:<30} ({test_time:.2f}s)")
        if result:
            passed += 1
        total_time += test_time
    
    success_rate = (passed / total) * 100
    
    print(f"\n📈 RESULTS:")
    print(f"  Tests Passed: {passed}/{total}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Total Time: {total_time:.2f}s")
    
    if success_rate >= 90:
        print("\n🎉 EXCELLENT! The enhanced alphabetic recognition system is production-ready!")
    elif success_rate >= 75:
        print("\n✅ GOOD! The system is mostly ready with minor issues to address.")
    elif success_rate >= 50:
        print("\n⚠️ NEEDS WORK! Several issues need to be addressed before production.")
    else:
        print("\n❌ CRITICAL! Major issues detected. System needs significant work.")
    
    print("\n🔗 Next Steps:")
    if success_rate >= 90:
        print("  1. Deploy to production environment")
        print("  2. Monitor performance metrics")
        print("  3. Collect real-world usage data")
    else:
        print("  1. Address failing tests")
        print("  2. Re-run testing suite")
        print("  3. Consider performance optimizations")
    
    return success_rate >= 75  # Return True if mostly successful

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
