"""
Configuration utilities for the Obsidian AI Study Pipeline.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class VaultConfig:
    """Configuration for vault parsing."""
    vault_path: str = ""
    ignore_patterns: list = field(default_factory=lambda: ['.obsidian', '__pycache__', '.git'])
    include_attachments: bool = True

@dataclass
class PreprocessingConfig:
    """Configuration for content preprocessing."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

@dataclass
class RetrievalConfig:
    """Configuration for semantic retrieval."""
    model_name: str = "all-MiniLM-L6-v2"
    index_type: str = "flat"
    cache_dir: str = ".cache"

@dataclass
class GenerationConfig:
    """Configuration for quiz generation."""
    model_type: str = "ollama"
    model_name: str = "llama3.2:1b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    questions_per_chunk: int = 2
    max_questions: int = 50
    quiz_types: list = field(default_factory=lambda: ["flashcard", "multiple_choice", "short_answer"])

@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    output_dir: str = "output"
    export_formats: list = field(default_factory=lambda: ["markdown", "quizlet_csv", "json"])
    group_by_source: bool = True
    include_metadata: bool = True

@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    vault: VaultConfig = field(default_factory=VaultConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Global settings
    log_level: str = "INFO"
    seed: int = 42

class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = PipelineConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> PipelineConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self.config = self._dict_to_config(config_data)
            logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
        
        return self.config
    
    def save_config(self, config_path: str, format: str = "yaml") -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path where to save configuration
            format: Format to save in ('yaml' or 'json')
        """
        try:
            config_dict = self._config_to_dict(self.config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        config = PipelineConfig()
        
        # Update vault config
        if 'vault' in config_data:
            vault_data = config_data['vault']
            config.vault = VaultConfig(
                vault_path=vault_data.get('vault_path', ''),
                ignore_patterns=vault_data.get('ignore_patterns', config.vault.ignore_patterns),
                include_attachments=vault_data.get('include_attachments', config.vault.include_attachments)
            )
        
        # Update preprocessing config
        if 'preprocessing' in config_data:
            prep_data = config_data['preprocessing']
            config.preprocessing = PreprocessingConfig(
                chunk_size=prep_data.get('chunk_size', config.preprocessing.chunk_size),
                chunk_overlap=prep_data.get('chunk_overlap', config.preprocessing.chunk_overlap),
                min_chunk_size=prep_data.get('min_chunk_size', config.preprocessing.min_chunk_size)
            )
        
        # Update retrieval config
        if 'retrieval' in config_data:
            ret_data = config_data['retrieval']
            config.retrieval = RetrievalConfig(
                model_name=ret_data.get('model_name', config.retrieval.model_name),
                index_type=ret_data.get('index_type', config.retrieval.index_type),
                cache_dir=ret_data.get('cache_dir', config.retrieval.cache_dir)
            )
        
        # Update generation config
        if 'generation' in config_data:
            gen_data = config_data['generation']
            config.generation = GenerationConfig(
                model_type=gen_data.get('model_type', config.generation.model_type),
                model_name=gen_data.get('model_name', config.generation.model_name),
                api_key=gen_data.get('api_key', config.generation.api_key),
                base_url=gen_data.get('base_url', config.generation.base_url),
                questions_per_chunk=gen_data.get('questions_per_chunk', config.generation.questions_per_chunk),
                max_questions=gen_data.get('max_questions', config.generation.max_questions),
                quiz_types=gen_data.get('quiz_types', config.generation.quiz_types)
            )
        
        # Update output config
        if 'output' in config_data:
            out_data = config_data['output']
            config.output = OutputConfig(
                output_dir=out_data.get('output_dir', config.output.output_dir),
                export_formats=out_data.get('export_formats', config.output.export_formats),
                group_by_source=out_data.get('group_by_source', config.output.group_by_source),
                include_metadata=out_data.get('include_metadata', config.output.include_metadata)
            )
        
        # Update global settings
        config.log_level = config_data.get('log_level', config.log_level)
        config.seed = config_data.get('seed', config.seed)
        
        return config
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert PipelineConfig object to dictionary."""
        return {
            'vault': {
                'vault_path': config.vault.vault_path,
                'ignore_patterns': config.vault.ignore_patterns,
                'include_attachments': config.vault.include_attachments
            },
            'preprocessing': {
                'chunk_size': config.preprocessing.chunk_size,
                'chunk_overlap': config.preprocessing.chunk_overlap,
                'min_chunk_size': config.preprocessing.min_chunk_size
            },
            'retrieval': {
                'model_name': config.retrieval.model_name,
                'index_type': config.retrieval.index_type,
                'cache_dir': config.retrieval.cache_dir
            },
            'generation': {
                'model_type': config.generation.model_type,
                'model_name': config.generation.model_name,
                'api_key': config.generation.api_key,
                'base_url': config.generation.base_url,
                'questions_per_chunk': config.generation.questions_per_chunk,
                'max_questions': config.generation.max_questions,
                'quiz_types': config.generation.quiz_types
            },
            'output': {
                'output_dir': config.output.output_dir,
                'export_formats': config.output.export_formats,
                'group_by_source': config.output.group_by_source,
                'include_metadata': config.output.include_metadata
            },
            'log_level': config.log_level,
            'seed': config.seed
        }
    
    def get_config(self) -> PipelineConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def create_default_config_file(self, output_path: str = "config.yaml") -> str:
        """
        Create a default configuration file.
        
        Args:
            output_path: Path where to save the default config
            
        Returns:
            Path to the created configuration file
        """
        # Create a sample configuration with comments
        config_content = """# Obsidian AI Study Pipeline Configuration

# Vault settings
vault:
  vault_path: "./example_vault"  # Path to your Obsidian vault
  ignore_patterns:  # Patterns to ignore when parsing
    - ".obsidian"
    - "__pycache__"
    - ".git"
  include_attachments: true  # Whether to include image references

# Content preprocessing settings
preprocessing:
  chunk_size: 512  # Maximum characters per chunk
  chunk_overlap: 50  # Overlap between chunks
  min_chunk_size: 100  # Minimum chunk size to consider

# Semantic retrieval settings
retrieval:
  model_name: "all-MiniLM-L6-v2"  # Sentence transformer model
  index_type: "flat"  # FAISS index type (flat, ivf, hnsw)
  cache_dir: ".cache"  # Directory for caching embeddings

# Quiz generation settings
generation:
  model_type: "ollama"  # Type of model (ollama, openai, huggingface)
  model_name: "llama3.2:1b"  # Specific model name
  api_key: null  # API key if needed
  base_url: null  # Base URL if using custom API
  questions_per_chunk: 2  # Questions to generate per content chunk
  max_questions: 50  # Maximum total questions
  quiz_types:  # Types of questions to generate
    - "flashcard"
    - "multiple_choice"
    - "short_answer"

# Output formatting settings
output:
  output_dir: "output"  # Directory for output files
  export_formats:  # Formats to export
    - "markdown"
    - "quizlet_csv"
    - "json"
  group_by_source: true  # Group questions by source note
  include_metadata: true  # Include metadata in exports

# Global settings
log_level: "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
seed: 42  # Random seed for reproducibility
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Created default configuration file: {output_path}")
        return output_path