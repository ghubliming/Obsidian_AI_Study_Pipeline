# Rate Limiting for Google AI Studio

This document explains how to use the new rate limiting features to handle Google AI Studio API quotas and avoid rate limit errors.

## Problem

Google AI Studio has rate limits that can cause errors like:
- "Learn more about Gemini API quotas"
- Rate limit exceeded errors
- Quota exhausted messages

## Solution

The Obsidian AI Study Pipeline now includes built-in rate limiting functionality with two approaches:

### 1. Rate Limit (Requests Per Second)

Control the maximum number of API requests per second:

```bash
# Allow 0.5 requests per second (1 request every 2 seconds)
obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit 0.5

# Very conservative: 0.2 requests per second (1 request every 5 seconds)
obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit 0.2
```

### 2. Fixed Delay Between Requests

Set a fixed delay between API calls:

```bash
# Wait 2 seconds between each API call
obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit-delay 2.0

# Wait 5 seconds between each API call (very conservative)
obsidian-ai-pipeline run ./vault --model-type gemini --model gemini-2.0-flash-exp --rate-limit-delay 5.0
```

## Configuration File

You can also set rate limiting in your configuration file:

```yaml
# config.yaml
generation:
  model_type: "gemini"
  model_name: "gemini-2.0-flash-exp"
  api_key: null  # Will use GOOGLE_API_KEY environment variable
  rate_limit: 0.5  # 0.5 requests per second
  # OR
  rate_limit_delay: 2.0  # 2 seconds between requests
  max_questions: 50
  quiz_types:
    - "flashcard"
    - "multiple_choice"
    - "short_answer"
```

Then run with:
```bash
obsidian-ai-pipeline run ./vault --config config.yaml
```

## Recommended Settings

### For Google AI Studio Free Tier (30 RPM, 1500/day)
```bash
# Recommended: Safe under both limits
--rate-limit 0.3
# OR
--rate-limit-delay 3.5

# Conservative: Well under limits
--rate-limit 0.2
# OR  
--rate-limit-delay 5.0

# Very conservative: Prioritizes daily limit
--rate-limit 0.1
# OR
--rate-limit-delay 10.0
```

### For Google AI Studio Free Tier (Legacy)
```bash
--rate-limit 0.5
# OR
--rate-limit-delay 2.0
```

### For Strict Quotas or High Usage
```bash
--rate-limit 0.2
# OR  
--rate-limit-delay 5.0
```

### For Testing/Development
```bash
--rate-limit 0.1
# OR
--rate-limit-delay 10.0
```

## CLI Examples

### Basic Usage with Rate Limiting
```bash
# Generate 20 questions with rate limiting
obsidian-ai-pipeline run ./vault --questions 20 --model-type gemini --rate-limit 0.5

# Use specific output directory
obsidian-ai-pipeline run ./vault --model-type gemini --rate-limit-delay 2.0 --output ./my_output

# Generate only flashcards with rate limiting
obsidian-ai-pipeline run ./vault --model-type gemini --rate-limit 0.5 --formats markdown
```

### Combined with Other Options
```bash
# Full example with all options
obsidian-ai-pipeline run ./vault \
  --questions 30 \
  --model-type gemini \
  --model gemini-2.0-flash-exp \
  --rate-limit 0.5 \
  --output ./quiz_output \
  --formats markdown quizlet_csv \
  --api-key your_api_key_here
```

## How It Works

**Google AI Studio Free Tier Limits:**
- **30 RPM** (requests per minute) = 0.5 requests per second
- **1500 requests per day** = ~1 request per minute for sustained usage

**Rate Limiting Logic:**
1. **Rate Limit (`--rate-limit`)**: Calculates delay as `1 / rate_limit` seconds
   - `--rate-limit 0.3` = 3.33 seconds between requests (18 RPM)
   - `--rate-limit 0.5` = 2 seconds between requests (30 RPM - at limit)
   - `--rate-limit 1.0` = 1 second between requests (60 RPM - exceeds limit!)

2. **Fixed Delay (`--rate-limit-delay`)**: Uses exact delay value
   - `--rate-limit-delay 3.5` = exactly 3.5 seconds between requests

3. **Automatic Warnings**: The system will warn if you use Gemini without rate limiting

## Monitoring

The system provides feedback about rate limiting:

```
🤖 Model: gemini:gemini-2.0-flash-exp
⏱️ Rate limit: 0.5 requests/second
   (This means 1 request every 2.0 seconds)
```

Watch the timestamps during question generation to see rate limiting in action.

## Troubleshooting

### Still Getting Rate Limit Errors?
- Increase the delay: try `--rate-limit 0.2` or `--rate-limit-delay 5.0`
- Check your Google AI Studio quota usage
- Ensure you're using the correct API key

### Generation Taking Too Long?
- Reduce the delay: try `--rate-limit 1.0` or `--rate-limit-delay 1.0`
- Reduce the number of questions: `--questions 10`
- Monitor for quota errors and adjust accordingly

### Testing Rate Limiting
Run the test script to verify functionality:
```bash
python scripts/test_rate_limiting.py
```

## Environment Variables

Set your Google API key:
```bash
# Windows
set GOOGLE_API_KEY=your_api_key_here

# macOS/Linux
export GOOGLE_API_KEY=your_api_key_here

# Or create a .env file in your project root:
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

## Best Practices

1. **Start Conservative**: Begin with `--rate-limit 0.5` or `--rate-limit-delay 2.0`
2. **Monitor Usage**: Watch your Google AI Studio quota dashboard
3. **Adjust as Needed**: Increase or decrease delays based on your quota
4. **Use Configuration Files**: For consistent settings across runs
5. **Test First**: Use small numbers of questions (`--questions 5`) to test settings
