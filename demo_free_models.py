#!/usr/bin/env python3
"""
Demo script showing how to use the new free online LLM models.
"""

import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

def demo_new_models():
    """Demonstrate the new model configuration options."""
    
    print("🚀 Obsidian AI Study Pipeline - Free Online Models Demo")
    print("=" * 60)
    print()
    
    print("✨ NEW: Support for Free Online LLMs")
    print()
    
    print("1️⃣ Openrouter (Multiple Free Models)")
    print("   • Website: https://openrouter.ai/")
    print("   • Get free API key and access to many models")
    print("   • Example command:")
    print("     python run_pipeline.py examples/sample_vault \\")
    print("       --model-type openrouter \\")
    print("       --model 'microsoft/phi-3-mini-128k-instruct:free' \\")
    print("       --api-key 'your-openrouter-api-key'")
    print()
    
    print("2️⃣ Google Gemini AI Studio (Free Tier)")
    print("   • Website: https://aistudio.google.com/")
    print("   • Free tier with generous limits")
    print("   • Example command:")
    print("     python run_pipeline.py examples/sample_vault \\")
    print("       --model-type gemini \\")
    print("       --model 'gemini-1.5-flash' \\")
    print("       --api-key 'your-google-api-key'")
    print()
    
    print("📋 Popular Free Models:")
    print()
    print("Openrouter Free Models:")
    print("  • microsoft/phi-3-mini-128k-instruct:free")
    print("  • meta-llama/llama-3.2-3b-instruct:free")
    print("  • meta-llama/llama-3.2-1b-instruct:free")
    print("  • mistralai/mistral-7b-instruct:free")
    print()
    print("Google Gemini Models (Free Tier):")
    print("  • gemini-1.5-flash (Recommended)")
    print("  • gemini-1.5-pro (Higher quality)")
    print("  • gemini-pro (Previous generation)")
    print()
    
    print("⚙️ Configuration Examples:")
    print()
    print("YAML Configuration for Openrouter:")
    print("```yaml")
    print("generation:")
    print("  model_type: 'openrouter'")
    print("  model_name: 'microsoft/phi-3-mini-128k-instruct:free'")
    print("  api_key: 'your-openrouter-api-key'")
    print("```")
    print()
    print("YAML Configuration for Gemini:")
    print("```yaml")
    print("generation:")
    print("  model_type: 'gemini'")
    print("  model_name: 'gemini-1.5-flash'")
    print("  api_key: 'your-google-api-key'")
    print("```")
    print()
    
    print("🔧 Environment Variables (Recommended):")
    print("export OPENROUTER_API_KEY='your-openrouter-api-key'")
    print("export GOOGLE_API_KEY='your-google-api-key'")
    print()
    
    print("Then use with:")
    print("python run_pipeline.py vault/ --model-type openrouter --api-key $OPENROUTER_API_KEY")
    print("python run_pipeline.py vault/ --model-type gemini --api-key $GOOGLE_API_KEY")
    print()
    
    print("✅ Benefits of Free Online Models:")
    print("  • No local setup required")
    print("  • Fast inference")
    print("  • No GPU needed")
    print("  • Always up-to-date models")
    print("  • Great for getting started")
    print()
    
    print("🎯 Quick Start Recommendation:")
    print("1. Sign up for Google AI Studio (fastest setup)")
    print("2. Get your free API key")
    print("3. Run: python run_pipeline.py examples/sample_vault --model-type gemini --model gemini-1.5-flash --api-key YOUR_KEY")
    print()
    
    print("💡 For more examples, see the updated README.md")

if __name__ == "__main__":
    demo_new_models()