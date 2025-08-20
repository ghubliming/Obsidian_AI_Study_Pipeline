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
    print("       --model 'microsoft/phi-3-mini-128k-instruct:free'")
    print("     # API key loaded from .env file automatically")
    print()
    
    print("2️⃣ Google Gemini AI Studio (Free Tier)")
    print("   • Website: https://aistudio.google.com/")
    print("   • Free tier with generous limits")
    print("   • Example command:")
    print("     python run_pipeline.py examples/sample_vault \\")
    print("       --model-type gemini \\")
    print("       --model 'gemini-1.5-flash'")
    print("     # API key loaded from .env file automatically")
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
    print("1. Using .env file (Recommended):")
    print("```bash")
    print("# Copy config/.env.example to .env")
    print("cp config/.env.example .env")
    print()
    print("# Edit .env file:")
    print("OPENROUTER_API_KEY=your_actual_openrouter_key")
    print("GOOGLE_API_KEY=your_actual_google_key") 
    print("```")
    print()
    print("Then run without API key in command:")
    print("python run_pipeline.py examples/sample_vault --model-type openrouter --model 'microsoft/phi-3-mini-128k-instruct:free'")
    print("python run_pipeline.py examples/sample_vault --model-type gemini --model 'gemini-1.5-flash'")
    print()
    print("2. YAML Configuration:")
    print("```yaml")
    print("generation:")
    print("  model_type: 'openrouter'")
    print("  model_name: 'microsoft/phi-3-mini-128k-instruct:free'")
    print("  # api_key will be loaded from environment variables")
    print("```")
    print()
    
    print("🔧 Getting Started (3 Easy Steps):")
    print()
    print("1. Copy the example environment file:")
    print("    print("   cp config/.env.example .env")")
    print()
    print("2. Edit .env and add your API keys:")
    print("   OPENROUTER_API_KEY=your_key_here")
    print("   GOOGLE_API_KEY=your_key_here")
    print()
    print("3. Run the pipeline:")
    print("   python run_pipeline.py examples/sample_vault --model-type gemini --model gemini-1.5-flash")
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
    print("2. Copy config/.env.example to .env and add your Google API key")
    print("3. Run: python run_pipeline.py examples/sample_vault --model-type gemini --model gemini-1.5-flash")
    print()
    
    print("💡 For more examples, see the updated README.md")

if __name__ == "__main__":
    demo_new_models()