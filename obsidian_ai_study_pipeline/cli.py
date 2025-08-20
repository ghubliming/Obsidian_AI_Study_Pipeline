"""
Command-line interface for the Obsidian AI Study Pipeline.
"""

import os
import sys
import logging
import click
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

from .pipeline import ObsidianStudyPipeline
from .utils import ConfigManager

logger = logging.getLogger(__name__)

@click.group()
@click.option('--config', '-c', default=None, 
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Obsidian AI Study Pipeline - Generate quiz questions from your Obsidian vault."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@cli.command()
@click.argument('vault_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='output', 
              help='Output directory for generated files')
@click.option('--questions', '-q', default=50, 
              help='Maximum number of questions to generate')
@click.option('--model', '-m', default='llama3.2:1b', 
              help='Model name for question generation')
@click.option('--model-type', default='ollama',
              type=click.Choice(['ollama', 'openai', 'openrouter', 'gemini']),
              help='Type of model to use')
@click.option('--api-key', default=None,
              help='API key for online models (optional - will check environment variables)')
@click.option('--formats', '-f', multiple=True, 
              default=['markdown', 'quizlet_csv'], 
              help='Export formats (can be used multiple times)')
@click.option('--rate-limit', default=None, type=float,
              help='Rate limit for API calls (requests per second). Useful for Google AI Studio quotas.')
@click.option('--rate-limit-delay', default=None, type=float,
              help='Fixed delay between API calls in seconds. Alternative to rate-limit.')
@click.pass_context
def run(ctx, vault_path, output, questions, model, model_type, api_key, formats, rate_limit, rate_limit_delay):
    """Run the complete pipeline on an Obsidian vault."""
    try:
        click.echo(f"🚀 Starting Obsidian AI Study Pipeline")
        click.echo(f"📂 Vault path: {vault_path}")
        click.echo(f"📁 Output directory: {output}")
        click.echo(f"❓ Max questions: {questions}")
        click.echo(f"🤖 Model: {model_type}:{model}")
        click.echo(f"📤 Export formats: {', '.join(formats)}")
        
        # Show rate limiting info
        if rate_limit:
            click.echo(f"⏱️ Rate limit: {rate_limit} requests/second")
        elif rate_limit_delay:
            click.echo(f"⏱️ API delay: {rate_limit_delay} seconds between calls")
        elif model_type == 'gemini':
            click.echo("⚠️ Consider using --rate-limit or --rate-limit-delay for Gemini API to avoid quota issues")
        
        # Show environment variable status for online models
        if model_type in ['openrouter', 'gemini', 'openai']:
            env_vars = {
                'openrouter': 'OPENROUTER_API_KEY',
                'gemini': 'GOOGLE_API_KEY',
                'openai': 'OPENAI_API_KEY'
            }
            env_var = env_vars.get(model_type)
            if env_var and os.getenv(env_var):
                click.echo(f"🔑 API key loaded from environment variable: {env_var}")
            elif api_key:
                click.echo(f"🔑 API key provided via command line")
            else:
                click.echo(f"⚠️ No API key found. Set {env_var} in .env file or use --api-key")
        
        click.echo()
        
        # Initialize pipeline
        pipeline = ObsidianStudyPipeline(config_path=ctx.obj['config_path'])
        
        # Update configuration with command line options
        pipeline.config.vault.vault_path = vault_path
        pipeline.config.output.output_dir = output
        pipeline.config.generation.max_questions = questions
        pipeline.config.generation.model_name = model
        pipeline.config.generation.model_type = model_type
        if api_key:
            pipeline.config.generation.api_key = api_key
        if rate_limit:
            pipeline.config.generation.rate_limit = rate_limit
        if rate_limit_delay:
            pipeline.config.generation.rate_limit_delay = rate_limit_delay
        pipeline.config.output.export_formats = list(formats)
        
        # Run pipeline
        with click.progressbar(range(5), label='Pipeline Progress') as bar:
            results = {}
            
            # Step 1: Parse vault
            click.echo("📖 Parsing vault...")
            pipeline.parse_vault(vault_path)
            bar.update(1)
            
            # Step 2: Preprocess content
            click.echo("⚙️ Preprocessing content...")
            pipeline.preprocess_content()
            bar.update(1)
            
            # Step 3: Build semantic index
            click.echo("🔍 Building semantic index...")
            pipeline.build_semantic_index()
            bar.update(1)
            
            # Step 4: Generate questions
            click.echo("❓ Generating quiz questions...")
            pipeline.generate_questions()
            bar.update(1)
            
            # Step 5: Export questions
            click.echo("📤 Exporting questions...")
            exported_files = pipeline.export_questions()
            bar.update(1)
        
        # Display results
        vault_stats = pipeline.get_vault_stats()
        preprocessing_stats = pipeline.get_preprocessing_stats()
        generation_stats = pipeline.get_generation_stats()
        
        click.echo()
        click.echo("✅ Pipeline completed successfully!")
        click.echo()
        click.echo("📊 Results Summary:")
        click.echo(f"  • Notes processed: {vault_stats.get('total_notes', 0)}")
        click.echo(f"  • Content chunks: {preprocessing_stats.get('total_chunks', 0)}")
        click.echo(f"  • Questions generated: {generation_stats.get('total_questions', 0)}")
        click.echo(f"  • Files exported: {len(exported_files)}")
        click.echo()
        click.echo("📁 Exported files:")
        for file_path in exported_files:
            click.echo(f"  • {file_path}")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.argument('vault_path', type=click.Path(exists=True))
@click.argument('topic')
@click.option('--questions', '-q', default=10, 
              help='Number of questions to generate')
@click.option('--output', '-o', default='output', 
              help='Output directory')
@click.pass_context
def topic(ctx, vault_path, topic, questions, output):
    """Generate questions focused on a specific topic."""
    try:
        click.echo(f"🎯 Generating questions for topic: {topic}")
        
        pipeline = ObsidianStudyPipeline(config_path=ctx.obj['config_path'])
        pipeline.config.output.output_dir = output
        
        # Parse and preprocess
        click.echo("📖 Processing vault...")
        pipeline.parse_vault(vault_path)
        pipeline.preprocess_content()
        pipeline.build_semantic_index()
        
        # Generate targeted questions
        click.echo(f"❓ Generating {questions} questions about '{topic}'...")
        targeted_questions = pipeline.generate_targeted_questions(topic, questions)
        
        if not targeted_questions:
            click.echo(f"⚠️ No relevant content found for topic: {topic}")
            return
        
        # Export questions
        exported_files = pipeline.export_questions(targeted_questions)
        
        click.echo()
        click.echo("✅ Topic-focused questions generated!")
        click.echo(f"  • Questions: {len(targeted_questions)}")
        click.echo(f"  • Topic: {topic}")
        click.echo()
        click.echo("📁 Exported files:")
        for file_path in exported_files:
            click.echo(f"  • {file_path}")
            
    except Exception as e:
        click.echo(f"❌ Failed to generate topic questions: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('vault_path', type=click.Path(exists=True))
@click.argument('query')
@click.option('--limit', '-l', default=5, 
              help='Number of results to show')
@click.pass_context
def search(ctx, vault_path, query, limit):
    """Search vault content semantically."""
    try:
        click.echo(f"🔍 Searching for: {query}")
        
        pipeline = ObsidianStudyPipeline(config_path=ctx.obj['config_path'])
        
        # Build index
        click.echo("📖 Processing vault...")
        pipeline.parse_vault(vault_path)
        pipeline.preprocess_content()
        pipeline.build_semantic_index()
        
        # Search
        results = pipeline.search_content(query, k=limit)
        
        click.echo()
        click.echo(f"📋 Found {len(results)} relevant chunks:")
        click.echo()
        
        for i, chunk in enumerate(results, 1):
            click.echo(f"  {i}. From: {chunk.source_note}")
            click.echo(f"     Type: {chunk.chunk_type}")
            click.echo(f"     Content: {chunk.text[:200]}...")
            if chunk.tags:
                click.echo(f"     Tags: {', '.join(chunk.tags)}")
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', default='config.yaml', 
              help='Output path for configuration file')
def init_config(output):
    """Create a sample configuration file."""
    try:
        config_manager = ConfigManager()
        config_path = config_manager.create_default_config_file(output)
        
        click.echo(f"✅ Created configuration file: {config_path}")
        click.echo()
        click.echo("📝 Edit the configuration file to customize:")
        click.echo(f"  • Vault path")
        click.echo(f"  • Model settings")
        click.echo(f"  • Output preferences")
        click.echo(f"  • Processing parameters")
        click.echo()
        click.echo(f"🚀 Then run: obsidian-ai-pipeline run <vault_path> --config {output}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('vault_path', type=click.Path(exists=True))
@click.pass_context
def stats(ctx, vault_path):
    """Show statistics about a vault."""
    try:
        click.echo(f"📊 Analyzing vault: {vault_path}")
        
        pipeline = ObsidianStudyPipeline(config_path=ctx.obj['config_path'])
        
        # Parse vault
        pipeline.parse_vault(vault_path)
        vault_stats = pipeline.get_vault_stats()
        
        # Preprocess content
        pipeline.preprocess_content()
        preprocessing_stats = pipeline.get_preprocessing_stats()
        
        click.echo()
        click.echo("📈 Vault Statistics:")
        click.echo(f"  • Total notes: {vault_stats.get('total_notes', 0)}")
        click.echo(f"  • Total images: {vault_stats.get('total_images', 0)}")
        click.echo(f"  • Total math blocks: {vault_stats.get('total_math_blocks', 0)}")
        click.echo(f"  • Total links: {vault_stats.get('total_links', 0)}")
        click.echo(f"  • Unique tags: {vault_stats.get('unique_tags', 0)}")
        click.echo(f"  • Avg content length: {vault_stats.get('average_content_length', 0):.0f} chars")
        
        click.echo()
        click.echo("⚙️ Processing Statistics:")
        click.echo(f"  • Content chunks: {preprocessing_stats.get('total_chunks', 0)}")
        click.echo(f"  • Unique source notes: {preprocessing_stats.get('unique_source_notes', 0)}")
        click.echo(f"  • Avg chunk length: {preprocessing_stats.get('avg_chunk_length', 0):.0f} chars")
        click.echo(f"  • Chunks with math: {preprocessing_stats.get('chunks_with_math', 0)}")
        click.echo(f"  • Chunks with images: {preprocessing_stats.get('chunks_with_images', 0)}")
        
        # Show chunk types
        chunk_types = preprocessing_stats.get('chunk_types', {})
        if chunk_types:
            click.echo()
            click.echo("📝 Content Types:")
            for chunk_type, count in chunk_types.items():
                click.echo(f"  • {chunk_type}: {count}")
        
        # Show popular tags
        all_tags = vault_stats.get('all_tags', [])
        if all_tags:
            click.echo()
            click.echo("🏷️ Popular Tags:")
            # Show first 10 tags
            for tag in all_tags[:10]:
                click.echo(f"  • #{tag}")
            if len(all_tags) > 10:
                click.echo(f"  • ... and {len(all_tags) - 10} more")
                
    except Exception as e:
        click.echo(f"❌ Failed to analyze vault: {e}", err=True)
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()