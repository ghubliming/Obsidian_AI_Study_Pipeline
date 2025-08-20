It works in Anki now, but it is not yet in Quizlet.

Only tried sample vault, already take 7k token on gemini.

And Gemini 2.0 flash is encountering rate issues.
- At least it has some output, but not full lists.
- ` python run_pipeline.py examples/sample_vault --model-type gemini --model "gemini-2.0-flash" `
- default is already quizlet `--formats quizlet_csv`
- Find a way to limit the rate locally
  - ` python run_pipeline.py examples/sample_vault --model-type gemini --model "gemini-2.0-flash-lite" --rate-limit 0.3`
  - 
- Try Gemini 2.5 flash lite -> still rate limited. ` "Learn more about Gemini API quotas"`

Same with Openrouter too slow and rate limited. Almost not usable!
- tried 0324 DeepSeek
- `python run_pipeline.py examples/sample_vault --model-type openrouter --model "deepseek/deepseek-chat-v3-0324:free"`

Try Ollama local on OLLAMA
- Use Ollama Deepseek r1 7b.
- ` python run_pipeline.py examples/sample_vault --model-type ollama --model deepseek-r1:7b --formats quizlet_csv`
-  Has bug with reasoning model like deepseek r1. 
- Tried Mistral 7b, it works. Take a lot of time. has bugs

```mistral 7b bugs
❓ Generating quiz questions...
2025-08-20 09:24:32,428 - obsidian_ai_study_pipeline.pipeline - INFO - Generating quiz questions from 25 chunks
2025-08-20 09:24:33,691 - obsidian_ai_study_pipeline.generation.quiz_generator - INFO - Initialized QuizGenerator with ollama:mistral:latest
2025-08-20 09:24:56,242 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-08-20 09:25:11,964 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-08-20 09:25:40,800 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-08-20 09:25:58,956 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-08-20 09:25:58,957 - obsidian_ai_study_pipeline.generation.quiz_generator - ERROR - Error parsing AI response: 'list' object has no attribute 'strip'
2025-08-20 09:30:14,487 - obsidian_ai_study_pipeline.generation.quiz_generator - ERROR - Error parsing AI response: Invalid \escape: line 3 column 23 (char 116)
2025-08-20 09:30:24,728 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-08-20 09:30:24,728 - obsidian_ai_study_pipeline.generation.quiz_generator - ERROR - Error parsing AI response: Invalid \escape: line 3 column 23 (char 116)
2025-08-20 09:31:07,746 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
```