# Scripts

This folder contains demo scripts, test utilities, and other executable scripts for the Obsidian AI Study Pipeline.

## Demo Scripts

- **[demo.py](demo.py)** - Basic demo of vault parsing functionality
- **[demo_free_models.py](demo_free_models.py)** - Demo showcasing free online LLM models (Openrouter, Google Gemini)
- **[demo_rate_limiting.py](demo_rate_limiting.py)** - Demo of rate limiting functionality for Google AI Studio

## Test Scripts

- **[test_rate_limiting.py](test_rate_limiting.py)** - Test script to verify rate limiting functionality

## Running Scripts

All scripts should be run from the project root directory:

```bash
# Run basic demo
python scripts/demo.py

# Run free models demo
python scripts/demo_free_models.py

# Run rate limiting demo
python scripts/demo_rate_limiting.py

# Test rate limiting functionality
python scripts/test_rate_limiting.py
```

## Requirements

Make sure you have:
1. Installed the project dependencies: `pip install -r requirements.txt`
2. Set up your API keys in `.env` file (copy from `config/.env.example`)
3. Have Ollama installed and running (for local models)
