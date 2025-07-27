# Obsidian_AI_Study_Pipeline

## Idea

This project aims to build an AI-powered pipeline for generating exam preparation tools from an Obsidian vault containing many Markdown notes, images, and math content. The goal is to automatically create quizzes (with answers), export them to platforms like Quizlet, and generate Markdown files with quiz questions and answers, all referencing the exact note and path in the vault.

The pipeline will prioritize free, open-source models and SOTA (state-of-the-art) retrieval and generation techniques, such as RAG (Retrieval-Augmented Generation).

---

## Planned Features

- **Automated Quiz Generation**: Use AI to generate quiz questions and answers from the vault’s content.
- **Source Reference**: Each quiz item will reference its source note and path within the vault.
- **Export Options**: Output to Quizlet (via CSV or API), and Markdown files for easy review in Obsidian.
- **Support for Math & Images**: Handle Obsidian's math grammar and image links.
- **Retrieval-Augmented Generation**: Use RAG or similar methods for contextual and accurate question generation.
- **Free/Open Source AI Models**: Prioritize models like Llama, Mistral, OpenAI GPT-4-free equivalents, HuggingFace models, etc.
- **Customizable Quiz Types**: Flashcards, multiple choice, cloze deletion, etc.

---

## Pipeline Plan (Step-by-Step)

1. **Vault Parsing**
   - Traverse the Obsidian vault directory.
   - Index all Markdown notes and images.
   - Extract note content, metadata, math blocks, and image references.

2. **Preprocessing**
   - Clean and segment notes into manageable chunks (e.g., sections, paragraphs).
   - Normalize math and image syntax for model input.
   - Build a local semantic search index (e.g., using FAISS or Haystack).

3. **Retrieval**
   - Implement semantic search (using free models, e.g., MiniLM, BGE, InstructorXL).
   - Retrieve relevant context for each quiz topic or question seed.

4. **Generation**
   - Use free LLMs (e.g., Llama.cpp, OpenChat, Mistral, Ollama) to generate quiz questions and answers from retrieved context.
   - Support various quiz formats (Flashcards, MCQ, Cloze, etc.).
   - Ensure answers reference the exact note and note path.

5. **Output Formatting**
   - Generate Markdown files with quizzes and answers, including source references.
   - Export quizzes to Quizlet-compatible formats (CSV, TSV, or direct API if available).
   - Optionally, create a dashboard or script for reviewing and editing generated quizzes.

6. **Integration**
   - (Optional) Build a UI for management and review, or use existing Obsidian plugins.
   - (Optional) Set up scheduled updates for new notes.

---

## Possible Tools & Technologies

- **Parsing & Indexing**: Python (os, pathlib, frontmatter), Node.js (obsidian-api), Rust (for performance).
- **Semantic Search**: FAISS, Haystack, Elasticsearch, Milvus.
- **AI Models**: HuggingFace Transformers, Llama.cpp, Mistral, OpenChat, Ollama, SentenceTransformers.
- **Quiz Generation**: Open-source LLMs via API or local inference.
- **Export**: CSV/TSV generator, Quizlet API, Markdown writer.
- **Math Support**: KaTeX/MathJax parsing, LaTeX normalization.
- **Images**: Copy or reference image files in output, preserve Obsidian image syntax.

---

## Example Workflow

1. **Run the pipeline script**: `python run_pipeline.py --vault ./ObsidianVault`
2. **Generate semantic index**: All notes and images are indexed for semantic retrieval.
3. **Select quiz topics or auto-generate based on vault content**.
4. **AI generates quiz questions + answers, with reference links**.
5. **Save as Markdown and/or export to Quizlet**.
6. **Review and edit quizzes in Obsidian or Quizlet as needed**.

---

## Next Steps

1. Define data structures for notes, quiz items, and references.
2. Prototype the parser and semantic indexer.
3. Test free LLMs for question generation.
4. Implement markdown and Quizlet exporters.
5. Document setup and usage.

---

## References

- [RAG (Retrieval Augmented Generation)](https://huggingface.co/docs/haystack/v1.21.2/guides/retrieval_augmented_generation)
- [Quizlet API](https://quizlet.com/api/2.0/docs/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Haystack](https://github.com/deepset-ai/haystack)
- [Open Source LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

**Feel free to open issues or pull requests to contribute ideas, tools, or improvements!**
