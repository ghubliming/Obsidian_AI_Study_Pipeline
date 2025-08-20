# Configuration

This folder contains configuration examples and templates for the Obsidian AI Study Pipeline.

## Files

- **[.env.example](.env.example)** - Environment variables template

## Setup

1. **Copy the environment template:**
   ```bash
   cp config/.env.example .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   # Google AI Studio (Gemini)
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Openrouter (Multiple free models)
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # OpenAI (if using)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Anthropic (if using)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

3. **Get API Keys:**
   - **Google AI Studio**: https://makersuite.google.com/app/apikey
   - **Openrouter**: https://openrouter.ai/keys
   - **OpenAI**: https://platform.openai.com/api-keys
   - **Anthropic**: https://console.anthropic.com/

## Security Note

- Never commit your `.env` file to version control
- The `.env` file is already in `.gitignore`
- Keep your API keys secure and don't share them
