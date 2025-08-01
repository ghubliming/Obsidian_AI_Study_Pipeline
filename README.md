# Obsidian AI Study Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project is an AI-powered pipeline for generating exam preparation tools from an Obsidian vault containing many Markdown notes, images, and math content. The goal is to automatically create quizzes (with answers), export them to platforms like Quizlet, and generate Markdown files with quiz questions and answers, all referencing the exact note and path in the vault.

The pipeline prioritizes free, open-source models and SOTA (state-of-the-art) retrieval and generation techniques, such as RAG (Retrieval-Augmented Generation).

## ✨ Features

- **Automated Quiz Generation**: Use AI to generate quiz questions and answers from the vault's content
- **Source Reference**: Each quiz item references its source note and path within the vault
- **Multiple Export Options**: Output to Quizlet (CSV), Anki (CSV), Markdown files, JSON, and study guides
- **Support for Math & Images**: Handle Obsidian's math grammar and image links
- **Retrieval-Augmented Generation**: Use RAG or similar methods for contextual and accurate question generation
- **Free/Open Source AI Models**: Prioritize models like Llama, Mistral, OpenAI GPT-4-free equivalents, HuggingFace models, etc.
- **Customizable Quiz Types**: Flashcards, multiple choice, cloze deletion, short answer, true/false
- **Semantic Search**: Find relevant content using advanced similarity search
- **Modular Design**: Each component can be used independently

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ghubliming/Obsidian_AI_Study_Pipeline.git
cd Obsidian_AI_Study_Pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Ollama (recommended for free local AI):**
```bash
# On macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (in a new terminal)
ollama pull llama3.2:1b
```

### Basic Usage

**Run the complete pipeline on your Obsidian vault:**
```bash
python run_pipeline.py /path/to/your/obsidian/vault
```

**Use the example vault:**
```bash
python run_pipeline.py examples/sample_vault
```

**Generate questions about a specific topic:**
```bash
python run_pipeline.py examples/sample_vault --topic "machine learning" --questions 15
```

**Customize output and model:**
```bash
python run_pipeline.py examples/sample_vault \
  --questions 30 \
  --model llama3.2:3b \
  --formats markdown quizlet_csv anki_csv \
  --output my_quiz_output
```

## 📖 Detailed Usage

### Command Line Interface

The `run_pipeline.py` script provides a comprehensive CLI:

```bash
python run_pipeline.py <vault_path> [OPTIONS]

Options:
  --config, -c PATH          Configuration file (YAML/JSON)
  --output, -o DIR           Output directory (default: output)
  --questions, -q INT        Max questions to generate (default: 50)
  --model, -m TEXT           Model name (default: llama3.2:1b)
  --model-type TEXT          Model type: ollama, openai, huggingface
  --topic TEXT               Focus on specific topic
  --quiz-types LIST          Question types: flashcard, multiple_choice, etc.
  --formats, -f LIST         Export formats: markdown, quizlet_csv, anki_csv, json
  --chunk-size INT           Content chunk size (default: 512)
  --embedding-model TEXT     Sentence transformer model
  --verbose, -v              Enable verbose logging
  --force-rebuild            Force rebuild semantic index
  --stats-only               Only show vault statistics
```

### Configuration File

Create a configuration file for advanced customization:

```bash
python -c "from obsidian_ai_study_pipeline.utils import ConfigManager; ConfigManager().create_default_config_file('config.yaml')"
```

Example configuration:
```yaml
# Vault settings
vault:
  vault_path: "./my_vault"
  ignore_patterns: [".obsidian", "__pycache__", ".git"]
  include_attachments: true

# AI model settings
generation:
  model_type: "ollama"  # or "openai", "huggingface"
  model_name: "llama3.2:3b"
  api_key: null  # for OpenAI
  questions_per_chunk: 2
  max_questions: 100
  quiz_types: ["flashcard", "multiple_choice", "short_answer"]

# Output settings
output:
  output_dir: "quiz_output"
  export_formats: ["markdown", "quizlet_csv", "json"]
  group_by_source: true
```

### Programmatic Usage

```python
from obsidian_ai_study_pipeline import ObsidianStudyPipeline

# Initialize pipeline
pipeline = ObsidianStudyPipeline()

# Configure
pipeline.config.vault.vault_path = "/path/to/vault"
pipeline.config.generation.max_questions = 50

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Or run steps individually
notes = pipeline.parse_vault("/path/to/vault")
chunks = pipeline.preprocess_content()
pipeline.build_semantic_index()
questions = pipeline.generate_questions()
exported_files = pipeline.export_questions()

# Search content
relevant_chunks = pipeline.search_content("neural networks", k=5)

# Generate topic-focused questions
ml_questions = pipeline.generate_targeted_questions("machine learning", 20)
```

## 🏗️ Architecture

The pipeline consists of several modular components:

### 1. Vault Parser (`vault_parser/`)
- Traverses Obsidian vault directory
- Parses Markdown files with frontmatter
- Extracts content, metadata, math blocks, images, and links
- Handles Obsidian-specific syntax

### 2. Content Preprocessor (`preprocessing/`)
- Chunks content into manageable pieces
- Cleans and normalizes text
- Preserves structure (sections, lists, code, math)
- Configurable chunk size and overlap

### 3. Semantic Retriever (`retrieval/`)
- Uses sentence transformers for embeddings
- FAISS index for efficient similarity search
- Supports various search strategies
- Caches embeddings for performance

### 4. Quiz Generator (`generation/`)
- Supports multiple AI model types (Ollama, OpenAI, etc.)
- Generates various question types
- Uses structured prompts for consistency
- Validates and filters generated questions

### 5. Output Formatter (`output_formatting/`)
- Exports to multiple formats
- Preserves source references
- Customizable output structure
- Ready for import into study platforms

## 📊 Supported Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **Markdown** | Structured markdown with questions and answers | Review in Obsidian, GitHub, etc. |
| **Quizlet CSV** | CSV format for Quizlet import | Create flashcards on Quizlet |
| **Anki CSV** | CSV format for Anki import | Spaced repetition with Anki |
| **JSON** | Structured data with full metadata | Integration with custom apps |
| **Study Guide** | Questions-only format for testing | Print for offline study |

## 🔧 Model Configuration

### Ollama (Recommended - Free & Local)
```yaml
generation:
  model_type: "ollama"
  model_name: "llama3.2:1b"  # or llama3.2:3b, mistral, etc.
```

### OpenAI
```yaml
generation:
  model_type: "openai"
  model_name: "gpt-3.5-turbo"
  api_key: "your-api-key"
```

### Local Models via Ollama
Available models:
- `llama3.2:1b` - Fast, lightweight
- `llama3.2:3b` - Better quality
- `mistral:7b` - Good balance
- `codellama:7b` - Better for technical content

## 📝 Quiz Types

1. **Flashcard**: Question and answer pairs
2. **Multiple Choice**: 4-option questions with one correct answer
3. **Cloze Deletion**: Fill-in-the-blank questions
4. **Short Answer**: Open-ended questions requiring brief responses
5. **True/False**: Boolean questions with explanations

## 🎯 Advanced Usage

### Topic-Focused Generation
```bash
# Generate questions about specific topics
python run_pipeline.py vault/ --topic "linear algebra" --questions 20
python run_pipeline.py vault/ --topic "python programming" --questions 15
```

### Batch Processing
```bash
# Process multiple topics
for topic in "machine learning" "statistics" "calculus"; do
  python run_pipeline.py vault/ --topic "$topic" --questions 10 --output "output_$topic"
done
```

### Custom Quiz Types
```bash
python run_pipeline.py vault/ --quiz-types flashcard multiple_choice --questions 50
```

### Performance Optimization
```bash
# Use smaller chunks for faster processing
python run_pipeline.py vault/ --chunk-size 256 --questions 30

# Use faster embedding model
python run_pipeline.py vault/ --embedding-model "all-MiniLM-L6-v2"
```

## 📋 Example Output

### Generated Markdown Quiz
```markdown
# Quiz Questions

## Machine Learning Fundamentals

### Question 1
**Type:** Multiple Choice
**Source:** Machine Learning Fundamentals

**Question:** What is the primary goal of supervised learning?

**Options:**
- A. To discover hidden patterns in unlabeled data
- B. To learn a mapping function from input to output using labeled data
- C. To maximize cumulative reward through interaction
- D. To reduce the dimensionality of the dataset

**Answer:** B

**Explanation:** Supervised learning uses labeled data to learn a mapping function that can make predictions on new, unseen data.

**Source Content:** Supervised learning involves training a model on labeled data, where both the input and the correct output are provided...
```

### Quizlet CSV Output
```csv
Term,Definition
"What is supervised learning?","A type of machine learning that uses labeled data to train models that can make predictions on new data."
"Name three types of unsupervised learning","Clustering, Association Rules, and Dimensionality Reduction"
```

## 🧪 Testing

Run tests to verify installation:
```bash
# Basic functionality test (no heavy dependencies)
python tests/test_lightweight.py

# Full test suite (requires all dependencies)
python tests/test_basic.py
```

## 🐛 Troubleshooting

### Common Issues

**1. "No module named 'numpy'" or similar dependency errors**
```bash
pip install -r requirements.txt
```

**2. Ollama connection errors**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
ollama pull llama3.2:1b
```

**3. Empty question generation**
- Check that your vault has substantial content
- Try a different model: `--model llama3.2:3b`
- Use `--verbose` flag for debugging
- Test with the example vault first

**4. Memory issues with large vaults**
- Reduce `--chunk-size` (e.g., `--chunk-size 256`)
- Process smaller batches: `--questions 20`
- Use a more efficient embedding model

**5. Performance issues**
- Use `--force-rebuild` only when necessary
- Cache is stored in `.cache/` directory
- Consider using GPU-enabled models for large vaults

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional AI model integrations
- New quiz question types
- Enhanced export formats
- Performance optimizations
- Better error handling
- UI/web interface

## 📚 Resources

- [RAG (Retrieval Augmented Generation)](https://huggingface.co/docs/haystack/v1.21.2/guides/retrieval_augmented_generation)
- [Quizlet API](https://quizlet.com/api/2.0/docs/)
- [Ollama Models](https://ollama.ai/library)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Obsidian community for the excellent note-taking platform
- Hugging Face for the transformers and sentence-transformers libraries
- Meta for FAISS semantic search
- Ollama team for making local LLMs accessible

---

**Feel free to open issues or pull requests to contribute ideas, tools, or improvements!**