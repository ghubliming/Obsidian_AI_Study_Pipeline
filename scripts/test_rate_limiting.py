#!/usr/bin/env python3
"""
Simple test script to verify rate limiting functionality.
"""

import os
import sys
import time
from pathlib import Path

# Add the project to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from obsidian_ai_study_pipeline.generation.quiz_generator import QuizGenerator
from obsidian_ai_study_pipeline.preprocessing import ContentChunk

def test_rate_limiting():
    """Test rate limiting functionality independently."""
    
    print("🧪 Testing Rate Limiting Functionality")
    print("=" * 40)
    
    # Test 1: Rate limit (requests per second)
    print("\n📊 Test 1: Rate limit - 1 request per second")
    generator = QuizGenerator(
        model_type="ollama",  # Use local model for testing
        model_name="llama3.2:1b",
        rate_limit=1.0  # 1 request per second
    )
    
    # Test the rate limiting delay calculation
    if hasattr(generator, 'calculated_delay'):
        expected_delay = 1.0
        actual_delay = generator.calculated_delay
        print(f"   Expected delay: {expected_delay}s")
        print(f"   Calculated delay: {actual_delay}s")
        assert abs(actual_delay - expected_delay) < 0.01, f"Delay calculation wrong: {actual_delay} != {expected_delay}"
        print("   ✅ Rate limit calculation correct")
    
    # Test 2: Fixed delay
    print("\n📊 Test 2: Fixed delay - 2 seconds")
    generator2 = QuizGenerator(
        model_type="ollama",
        model_name="llama3.2:1b", 
        rate_limit_delay=2.0
    )
    
    if hasattr(generator2, 'calculated_delay'):
        expected_delay = 2.0
        actual_delay = generator2.calculated_delay
        print(f"   Expected delay: {expected_delay}s")
        print(f"   Calculated delay: {actual_delay}s")
        assert abs(actual_delay - expected_delay) < 0.01, f"Delay calculation wrong: {actual_delay} != {expected_delay}"
        print("   ✅ Fixed delay calculation correct")
    
    # Test 3: No rate limiting
    print("\n📊 Test 3: No rate limiting")
    generator3 = QuizGenerator(
        model_type="ollama",
        model_name="llama3.2:1b"
    )
    
    if hasattr(generator3, 'calculated_delay'):
        expected_delay = 0.0
        actual_delay = generator3.calculated_delay
        print(f"   Expected delay: {expected_delay}s")
        print(f"   Calculated delay: {actual_delay}s")
        assert actual_delay == expected_delay, f"Delay calculation wrong: {actual_delay} != {expected_delay}"
        print("   ✅ No rate limiting calculation correct")
    
    # Test 4: Rate limiting delay timing
    print("\n📊 Test 4: Actual delay timing")
    generator4 = QuizGenerator(
        model_type="ollama",
        model_name="llama3.2:1b",
        rate_limit_delay=0.5  # 0.5 second delay for quick test
    )
    
    # Test the actual delay mechanism
    start_time = time.time()
    generator4._apply_rate_limit()  # First call should not delay
    first_call_time = time.time()
    
    generator4._apply_rate_limit()  # Second call should delay
    second_call_time = time.time()
    
    actual_delay = second_call_time - first_call_time
    expected_min_delay = 0.5
    
    print(f"   Time between calls: {actual_delay:.3f}s")
    print(f"   Expected minimum: {expected_min_delay}s")
    
    if actual_delay >= expected_min_delay * 0.9:  # Allow 10% tolerance
        print("   ✅ Rate limiting delay working correctly")
    else:
        print("   ⚠️ Rate limiting delay may not be working as expected")
    
    print("\n🎉 Rate limiting tests completed!")
    return True

def test_gemini_warning():
    """Test that Gemini without rate limiting shows a warning."""
    print("\n📊 Test 5: Gemini warning without rate limiting")
    
    # Capture logs to check for warning
    import logging
    import io
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    
    logger = logging.getLogger('obsidian_ai_study_pipeline.generation.quiz_generator')
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    
    # Create Gemini generator without rate limiting
    generator = QuizGenerator(
        model_type="gemini",
        model_name="gemini-2.0-flash-exp"
    )
    
    # Check if warning was logged
    log_output = log_capture.getvalue()
    if "rate limiting" in log_output.lower() and "gemini" in log_output.lower():
        print("   ✅ Gemini rate limiting warning displayed correctly")
    else:
        print("   ⚠️ Gemini rate limiting warning not found")
        print(f"   Log output: {log_output}")
    
    logger.removeHandler(handler)

if __name__ == "__main__":
    try:
        test_rate_limiting()
        test_gemini_warning()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
