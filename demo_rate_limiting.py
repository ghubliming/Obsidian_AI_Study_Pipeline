#!/usr/bin/env python3
"""
Demo script showcasing rate limiting functionality for Google AI Studio.
"""

import os
import sys
import time
from pathlib import Path

# Add the project to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from obsidian_ai_study_pipeline import ObsidianStudyPipeline
from obsidian_ai_study_pipeline.utils import ConfigManager, PipelineConfig, GenerationConfig

def demo_rate_limiting():
    """Demonstrate rate limiting functionality with Gemini API."""
    
    print("🚀 Obsidian AI Study Pipeline - Rate Limiting Demo")
    print("=" * 50)
    
    # Check if Google API key is available
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ No GOOGLE_API_KEY found in environment variables.")
        print("   Please set your Google AI Studio API key to test rate limiting.")
        print("   Example: export GOOGLE_API_KEY='your_key_here'")
        return
    
    print("✅ Google API key found")
    
    # Create a test configuration with rate limiting
    config = PipelineConfig()
    config.generation = GenerationConfig(
        model_type="gemini",
        model_name="gemini-2.0-flash-exp",  # Use a fast model for demo
        api_key=api_key,
        max_questions=5,  # Small number for demo
        questions_per_chunk=1,
        rate_limit=0.3  # 0.3 requests per second (safe for 30 RPM limit)
    )
    
    print(f"🤖 Model: {config.generation.model_type}:{config.generation.model_name}")
    print(f"⏱️ Rate limit: {config.generation.rate_limit} requests/second")
    print(f"   (This means 1 request every {1/config.generation.rate_limit:.1f} seconds)")
    print()
    
    # Use sample vault if available
    sample_vault = project_root / "examples" / "sample_vault"
    if not sample_vault.exists():
        print(f"❌ Sample vault not found at {sample_vault}")
        print("   Please ensure the sample vault exists or modify the vault path.")
        return
    
    config.vault.vault_path = str(sample_vault)
    config.output.output_dir = str(project_root / "output_rate_limit_demo")
    
    print(f"📂 Using vault: {config.vault.vault_path}")
    print(f"📁 Output directory: {config.output.output_dir}")
    print()
    
    try:
        # Initialize pipeline
        pipeline = ObsidianStudyPipeline(config=config)
        
        print("📖 Step 1: Parsing vault...")
        start_time = time.time()
        pipeline.parse_vault(config.vault.vault_path)
        vault_stats = pipeline.get_vault_stats()
        print(f"   ✅ Parsed {vault_stats.get('total_notes', 0)} notes")
        
        print("⚙️ Step 2: Preprocessing content...")
        pipeline.preprocess_content()
        preprocessing_stats = pipeline.get_preprocessing_stats()
        print(f"   ✅ Created {preprocessing_stats.get('total_chunks', 0)} content chunks")
        
        print("🔍 Step 3: Building semantic index...")
        pipeline.build_semantic_index()
        print("   ✅ Semantic index ready")
        
        print("❓ Step 4: Generating questions with rate limiting...")
        question_start_time = time.time()
        print("   📊 Watch the timestamps to see rate limiting in action:")
        print()
        
        # This will apply rate limiting
        questions = pipeline.generate_questions()
        
        question_end_time = time.time()
        question_duration = question_end_time - question_start_time
        
        print()
        print(f"   ✅ Generated {len(questions)} questions in {question_duration:.1f} seconds")
        
        if len(questions) > 1:
            expected_min_time = (len(questions) - 1) * (1 / config.generation.rate_limit)
            print(f"   📈 Expected minimum time with rate limiting: {expected_min_time:.1f} seconds")
            if question_duration >= expected_min_time * 0.8:  # Allow some tolerance
                print("   ✅ Rate limiting appears to be working correctly!")
            else:
                print("   ⚠️ Rate limiting may not be working as expected.")
        
        print("📤 Step 5: Exporting questions...")
        exported_files = pipeline.export_questions()
        
        total_time = time.time() - start_time
        
        print()
        print("🎉 Demo completed successfully!")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"📊 Questions generated: {len(questions)}")
        print("📁 Exported files:")
        for file_path in exported_files:
            print(f"   • {file_path}")
        
        print()
        print("💡 Rate Limiting Tips:")
        print("   • For Google AI Studio free tier (30 RPM, 1500/day):")
        print("   • Use --rate-limit 0.3 (recommended, ~18 RPM)")
        print("   • Use --rate-limit 0.2 (conservative, ~12 RPM)")  
        print("   • Use --rate-limit-delay 3.5 (alternative to 0.3)")
        print("   • Adjust based on your daily usage patterns")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Ensure your GOOGLE_API_KEY is valid")
        print("   • Check your Google AI Studio quota")
        print("   • Try increasing the rate limit delay")
        
        # If it's a quota error, provide specific guidance
        if "quota" in str(e).lower():
            print("   • This appears to be a quota error - try --rate-limit 0.2 for slower requests")

def demo_cli_usage():
    """Show CLI usage examples for rate limiting."""
    print()
    print("📚 CLI Usage Examples:")
    print("=" * 30)
    print()
    print("# Basic usage with rate limiting (0.3 requests per second, safe for 30 RPM):")
    print("obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit 0.3")
    print()
    print("# Using fixed delay (3.5 seconds between requests):")
    print("obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit-delay 3.5")
    print()
    print("# Very conservative rate limiting for heavy daily usage:")
    print("obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit 0.1")
    print()
    print("# Combined with other options:")
    print("obsidian-ai-pipeline run ./vault --questions 20 --model-type gemini --rate-limit 0.3 --output ./my_output")

if __name__ == "__main__":
    demo_rate_limiting()
    demo_cli_usage()
