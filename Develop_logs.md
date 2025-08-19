It works in Anki now, but it is not yet in Quizlet.

Only tried sample vault, already take 7k token on gemini.

And Gemini 2.0 flash is encountering rate issues.
- At least it has some output, but not full lists.
- ` python run_pipeline.py examples/sample_vault --model-type gemini --model "gemini-2.0-flash" `
- default is already quizlet `--formats quizlet_csv`
- Find a way to limit the rate locally!!!

Same with Openrouter too slow and rate limited. Almost not usable!
- tried 0324 DeepSeek
- `python run_pipeline.py examples/sample_vault --model-type openrouter --model "deepseek/deepseek-chat-v3-0324:free"`

Try Ollama local on OLLAMA
- Use Ollama Deepseek r1 7b.
- ` python run_pipeline.py examples/sample_vault --model-type ollama --model deepseek-r1:7b --formats quizlet_csv`
-  Has bug with reasoning model like deepseek r1. Try another one!!!
