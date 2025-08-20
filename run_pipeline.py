#!/usr/bin/env python3
"""
Main pipeline runner script for the Obsidian AI Study Pipeline.

This script provides a simple way to run the complete pipeline
from the command line without installing the package.
"""

import sys
import argparse
import logging
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from obsidian_ai_study_pipeline import ObsidianStudyPipeline
from obsidian_ai_study_pipeline.utils import ConfigManager

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description='Obsidian AI Study Pipeline - Generate quiz questions from your Obsidian vault',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py examples/sample_vault
  python run_pipeline.py /path/to/vault --questions 30 --model llama3.2:1b
  python run_pipeline.py vault_path --config config.yaml --output my_output
  python run_pipeline.py vault_path --topic "machine learning" --questions 15
  
Free Online Models (with .env file):
  python run_pipeline.py vault_path --model-type openrouter --model "microsoft/phi-3-mini-128k-instruct:free"
  python run_pipeline.py vault_path --model-type gemini --model "gemini-2.0-flash-exp"

Rate Limiting (for Google AI Studio quotas):
  python run_pipeline.py vault_path --model-type gemini --model "gemini-2.0-flash-exp" --rate-limit 0.5
  python run_pipeline.py vault_path --model-type gemini --rate-limit-delay 2.0
  python run_pipeline.py vault_path --model-type gemini --model "gemini-1.5-flash"
        """
    )
    
    # Required arguments
    parser.add_argument('vault_path', 
                       help='Path to the Obsidian vault directory')
    
    # Configuration options
    parser.add_argument('--config', '-c',
                       help='Path to configuration file (YAML or JSON)')
    
    # Output options
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for generated files (default: output)')
    
    # Generation options
    parser.add_argument('--questions', '-q', type=int, default=50,
                       help='Maximum number of questions to generate (default: 50)')
    
    parser.add_argument('--model', '-m', default='llama3.2:1b',
                       help='Model name for question generation (default: llama3.2:1b)')
    
    parser.add_argument('--model-type', default='ollama',
                       choices=['ollama', 'openai', 'openrouter', 'gemini', 'huggingface'],
                       help='Type of model to use (default: ollama)')
    
    # API configuration (will check environment variables)
    parser.add_argument('--api-key',
                       help='API key for online models (optional - will check environment variables: OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY)')
    
    # Rate limiting options
    parser.add_argument('--rate-limit', type=float,
                       help='Rate limit for API calls (requests per second). Useful for Google AI Studio quotas.')
    
    parser.add_argument('--rate-limit-delay', type=float,
                       help='Fixed delay between API calls in seconds. Alternative to rate-limit.')
    
    # Filtering options
    parser.add_argument('--topic',
                       help='Generate questions focused on a specific topic')
                       
    parser.add_argument('--quiz-types', nargs='+',
                       choices=['flashcard', 'multiple_choice', 'cloze_deletion', 'short_answer', 'true_false'],
                       default=['flashcard', 'multiple_choice', 'short_answer'],
                       help='Types of quiz questions to generate')
    
    # Export options
    parser.add_argument('--formats', '-f', nargs='+',
                       choices=['markdown', 'quizlet_csv', 'anki_csv', 'json', 'study_guide'],
                       default=['markdown', 'quizlet_csv'],
                       help='Export formats (default: markdown quizlet_csv)')
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Maximum characters per content chunk (default: 512)')
    
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)')
    
    # Control options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuilding the semantic index')
    
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show vault statistics, do not generate questions')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate vault path
        vault_path = Path(args.vault_path)
        if not vault_path.exists():
            print(f"❌ Error: Vault path does not exist: {args.vault_path}")
            sys.exit(1)
        
        print("🚀 Obsidian AI Study Pipeline")
        print("=" * 50)
        print(f"📂 Vault: {vault_path}")
        print(f"📁 Output: {args.output}")
        print(f"🤖 Model: {args.model_type}:{args.model}")
        print(f"❓ Max Questions: {args.questions}")
        print(f"📤 Formats: {', '.join(args.formats)}")
        if args.topic:
            print(f"🎯 Topic Focus: {args.topic}")
        print()
        
        # Initialize pipeline
        if args.config:
            pipeline = ObsidianStudyPipeline(config_path=args.config)
            print(f"📄 Loaded configuration from: {args.config}")
        else:
            pipeline = ObsidianStudyPipeline()
        
        # Update configuration with command line arguments
        pipeline.config.vault.vault_path = str(vault_path)
        pipeline.config.output.output_dir = args.output
        pipeline.config.generation.max_questions = args.questions
        pipeline.config.generation.model_name = args.model
        pipeline.config.generation.model_type = args.model_type
        if args.api_key:
            pipeline.config.generation.api_key = args.api_key
        if args.rate_limit:
            pipeline.config.generation.rate_limit = args.rate_limit
        if args.rate_limit_delay:
            pipeline.config.generation.rate_limit_delay = args.rate_limit_delay
        pipeline.config.generation.quiz_types = args.quiz_types
        pipeline.config.output.export_formats = args.formats
        pipeline.config.preprocessing.chunk_size = args.chunk_size
        pipeline.config.retrieval.model_name = args.embedding_model
        
        # Step 1: Parse vault
        print("📖 Parsing vault...")
        pipeline.parse_vault(str(vault_path))
        vault_stats = pipeline.get_vault_stats()
        
        print(f"   ✅ Found {vault_stats.get('total_notes', 0)} notes")
        print(f"   📷 Images: {vault_stats.get('total_images', 0)}")
        print(f"   🔗 Links: {vault_stats.get('total_links', 0)}")
        print(f"   🏷️ Tags: {vault_stats.get('unique_tags', 0)}")
        
        # Step 2: Preprocess content
        print("\n⚙️ Preprocessing content...")
        pipeline.preprocess_content()
        preprocessing_stats = pipeline.get_preprocessing_stats()
        
        print(f"   ✅ Generated {preprocessing_stats.get('total_chunks', 0)} content chunks")
        print(f"   📊 Avg chunk length: {preprocessing_stats.get('avg_chunk_length', 0):.0f} chars")
        
        # If stats only, exit here
        if args.stats_only:
            print("\n📊 Vault Analysis Complete!")
            chunk_types = preprocessing_stats.get('chunk_types', {})
            if chunk_types:
                print("\n📝 Content Types:")
                for chunk_type, count in chunk_types.items():
                    print(f"   • {chunk_type}: {count}")
            return
        
        # Step 3: Build semantic index
        print("\n🔍 Building semantic index...")
        pipeline.build_semantic_index(force_rebuild=args.force_rebuild)
        retrieval_stats = pipeline.get_retrieval_stats()
        
        print(f"   ✅ Built index with {retrieval_stats.get('total_chunks', 0)} vectors")
        print(f"   🧠 Model: {retrieval_stats.get('model_name', 'unknown')}")
        
        # Step 4: Generate questions
        print("\n❓ Generating quiz questions...")
        
        if args.topic:
            # Generate topic-focused questions
            questions = pipeline.generate_targeted_questions(args.topic, args.questions)
            print(f"   🎯 Generated {len(questions)} questions about '{args.topic}'")
        else:
            # Generate general questions
            pipeline.generate_questions()
            questions = pipeline.questions
            print(f"   ✅ Generated {len(questions)} questions")
        
        if not questions:
            print("   ⚠️ No questions were generated. Check your model configuration.")
            return
        
        generation_stats = pipeline.generator.get_generation_stats(questions) if pipeline.generator else {}
        question_types = generation_stats.get('question_types', {})
        if question_types:
            print("   📝 Question types:")
            for qtype, count in question_types.items():
                print(f"      • {qtype}: {count}")
        
        # Step 5: Export questions
        print("\n📤 Exporting questions...")
        exported_files = pipeline.export_questions(questions)
        
        print(f"   ✅ Exported {len(questions)} questions to {len(exported_files)} files")
        
        # Display results
        print("\n" + "=" * 50)
        print("✅ Pipeline completed successfully!")
        print("\n📊 Final Results:")
        print(f"   • Notes processed: {vault_stats.get('total_notes', 0)}")
        print(f"   • Content chunks: {preprocessing_stats.get('total_chunks', 0)}")
        print(f"   • Questions generated: {len(questions)}")
        print(f"   • Files exported: {len(exported_files)}")
        
        print("\n📁 Generated files:")
        for file_path in exported_files:
            print(f"   • {file_path}")
        
        print(f"\n🎉 Check the '{args.output}' directory for your quiz questions!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print(f"\n❌ Pipeline failed: {e}")
        print("\n💡 Tips:")
        print("   • Make sure Ollama is running if using local models")
        print("   • For online models, set environment variables in .env file:")
        print("     - OPENROUTER_API_KEY for Openrouter models")
        print("     - GOOGLE_API_KEY for Google Gemini models")
        print("     - OPENAI_API_KEY for OpenAI models")
        print("   • Check that your vault path is correct")
        print("   • Try running with --verbose for more details")
        print("   • Use --stats-only to test vault parsing")
        sys.exit(1)

if __name__ == '__main__':
    main()