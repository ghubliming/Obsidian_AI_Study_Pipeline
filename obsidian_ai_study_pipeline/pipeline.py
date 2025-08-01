"""
Main pipeline orchestrator for the Obsidian AI Study Pipeline.
"""

import os
import logging
import random
from typing import List, Dict, Optional
import numpy as np

from .vault_parser import VaultParser, ObsidianNote
from .preprocessing import ContentPreprocessor, ContentChunk
from .retrieval import SemanticRetriever
from .generation import QuizGenerator, QuizQuestion, QuizType
from .output_formatting import QuizExporter
from .utils import ConfigManager, PipelineConfig

logger = logging.getLogger(__name__)

class ObsidianStudyPipeline:
    """Main pipeline for generating quiz questions from Obsidian vaults."""
    
    def __init__(self, config: Optional[PipelineConfig] = None, config_path: Optional[str] = None):
        """
        Initialize the Obsidian Study Pipeline.
        
        Args:
            config: Pipeline configuration object
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.get_config()
        elif config:
            self.config = config
            self.config_manager = ConfigManager()
            self.config_manager.config = config
        else:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.get_config()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize components
        self.vault_parser = None
        self.preprocessor = None
        self.retriever = None
        self.generator = None
        self.exporter = None
        
        # Pipeline state
        self.notes: List[ObsidianNote] = []
        self.chunks: List[ContentChunk] = []
        self.questions: List[QuizQuestion] = []
        
        logger.info("Initialized Obsidian Study Pipeline")
    
    def run_full_pipeline(self, vault_path: Optional[str] = None) -> Dict:
        """
        Run the complete pipeline from vault parsing to question export.
        
        Args:
            vault_path: Path to Obsidian vault (overrides config)
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info("Starting full pipeline execution")
        
        try:
            # Step 1: Parse vault
            results = {}
            vault_path = vault_path or self.config.vault.vault_path
            self.parse_vault(vault_path)
            results['vault_stats'] = self.get_vault_stats()
            
            # Step 2: Preprocess content
            self.preprocess_content()
            results['preprocessing_stats'] = self.get_preprocessing_stats()
            
            # Step 3: Build semantic index
            self.build_semantic_index()
            results['retrieval_stats'] = self.get_retrieval_stats()
            
            # Step 4: Generate quiz questions
            self.generate_questions()
            results['generation_stats'] = self.get_generation_stats()
            
            # Step 5: Export questions
            exported_files = self.export_questions()
            results['exported_files'] = exported_files
            
            results['total_questions'] = len(self.questions)
            results['pipeline_status'] = 'completed'
            
            logger.info(f"Pipeline completed successfully. Generated {len(self.questions)} questions.")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'pipeline_status': 'failed', 'error': str(e)}
    
    def parse_vault(self, vault_path: str) -> List[ObsidianNote]:
        """
        Parse the Obsidian vault and extract notes.
        
        Args:
            vault_path: Path to the Obsidian vault
            
        Returns:
            List of parsed ObsidianNote objects
        """
        logger.info(f"Parsing vault: {vault_path}")
        
        if not os.path.exists(vault_path):
            raise ValueError(f"Vault path does not exist: {vault_path}")
        
        self.vault_parser = VaultParser(vault_path)
        self.notes = self.vault_parser.parse_vault()
        
        logger.info(f"Parsed {len(self.notes)} notes from vault")
        return self.notes
    
    def preprocess_content(self) -> List[ContentChunk]:
        """
        Preprocess the parsed notes into content chunks.
        
        Returns:
            List of processed ContentChunk objects
        """
        if not self.notes:
            raise ValueError("No notes available. Run parse_vault first.")
        
        logger.info("Preprocessing content into chunks")
        
        self.preprocessor = ContentPreprocessor(
            chunk_size=self.config.preprocessing.chunk_size,
            chunk_overlap=self.config.preprocessing.chunk_overlap,
            min_chunk_size=self.config.preprocessing.min_chunk_size
        )
        
        self.chunks = self.preprocessor.preprocess_notes(self.notes)
        
        logger.info(f"Generated {len(self.chunks)} content chunks")
        return self.chunks
    
    def build_semantic_index(self, force_rebuild: bool = False) -> None:
        """
        Build semantic search index from content chunks.
        
        Args:
            force_rebuild: Whether to force rebuilding the index
        """
        if not self.chunks:
            raise ValueError("No chunks available. Run preprocess_content first.")
        
        logger.info("Building semantic search index")
        
        self.retriever = SemanticRetriever(
            model_name=self.config.retrieval.model_name,
            index_type=self.config.retrieval.index_type,
            cache_dir=self.config.retrieval.cache_dir
        )
        
        self.retriever.build_index(self.chunks, force_rebuild=force_rebuild)
        
        logger.info("Semantic index built successfully")
    
    def generate_questions(self, 
                          target_chunks: Optional[List[ContentChunk]] = None) -> List[QuizQuestion]:
        """
        Generate quiz questions from content chunks.
        
        Args:
            target_chunks: Specific chunks to generate questions from (uses all if None)
            
        Returns:
            List of generated quiz questions
        """
        chunks_to_use = target_chunks or self.chunks
        
        if not chunks_to_use:
            raise ValueError("No chunks available for question generation")
        
        logger.info(f"Generating quiz questions from {len(chunks_to_use)} chunks")
        
        # Convert quiz type strings to enum values
        quiz_types = []
        for qt_str in self.config.generation.quiz_types:
            try:
                quiz_types.append(QuizType(qt_str))
            except ValueError:
                logger.warning(f"Unknown quiz type: {qt_str}")
        
        if not quiz_types:
            quiz_types = [QuizType.FLASHCARD, QuizType.MULTIPLE_CHOICE]
        
        self.generator = QuizGenerator(
            model_type=self.config.generation.model_type,
            model_name=self.config.generation.model_name,
            api_key=self.config.generation.api_key,
            base_url=self.config.generation.base_url
        )
        
        self.questions = self.generator.generate_quiz_questions(
            chunks=chunks_to_use,
            quiz_types=quiz_types,
            questions_per_chunk=self.config.generation.questions_per_chunk,
            max_questions=self.config.generation.max_questions
        )
        
        # Validate questions
        self.questions = self.generator.validate_questions(self.questions)
        
        logger.info(f"Generated and validated {len(self.questions)} quiz questions")
        return self.questions
    
    def export_questions(self, 
                        custom_questions: Optional[List[QuizQuestion]] = None) -> List[str]:
        """
        Export quiz questions to various formats.
        
        Args:
            custom_questions: Custom list of questions to export (uses generated ones if None)
            
        Returns:
            List of exported file paths
        """
        questions_to_export = custom_questions or self.questions
        
        if not questions_to_export:
            raise ValueError("No questions available for export")
        
        logger.info(f"Exporting {len(questions_to_export)} questions")
        
        self.exporter = QuizExporter(output_dir=self.config.output.output_dir)
        exported_files = []
        
        for format_type in self.config.output.export_formats:
            try:
                if format_type == "markdown":
                    file_path = self.exporter.export_to_markdown(
                        questions_to_export,
                        group_by_source=self.config.output.group_by_source
                    )
                    exported_files.append(file_path)
                
                elif format_type == "quizlet_csv":
                    file_path = self.exporter.export_to_quizlet_csv(questions_to_export)
                    exported_files.append(file_path)
                
                elif format_type == "anki_csv":
                    file_path = self.exporter.export_to_anki_csv(questions_to_export)
                    exported_files.append(file_path)
                
                elif format_type == "json":
                    file_path = self.exporter.export_to_json(
                        questions_to_export,
                        include_metadata=self.config.output.include_metadata
                    )
                    exported_files.append(file_path)
                
                elif format_type == "study_guide":
                    file_path = self.exporter.export_study_guide(questions_to_export)
                    exported_files.append(file_path)
                
                else:
                    logger.warning(f"Unknown export format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Error exporting to {format_type}: {e}")
        
        logger.info(f"Exported questions to {len(exported_files)} files")
        return exported_files
    
    def search_content(self, query: str, k: int = 5) -> List[ContentChunk]:
        """
        Search for relevant content chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant content chunks
        """
        if not self.retriever:
            raise ValueError("Semantic index not built. Run build_semantic_index first.")
        
        results = self.retriever.search(query, k=k)
        return [chunk for chunk, score in results]
    
    def generate_targeted_questions(self, 
                                  topic: str,
                                  num_questions: int = 10) -> List[QuizQuestion]:
        """
        Generate questions focused on a specific topic.
        
        Args:
            topic: Topic to focus on
            num_questions: Number of questions to generate
            
        Returns:
            List of topic-focused quiz questions
        """
        if not self.retriever:
            raise ValueError("Semantic index not built. Run build_semantic_index first.")
        
        # Find relevant chunks for the topic
        relevant_results = self.retriever.search_by_topic(topic, k=num_questions * 2)
        relevant_chunks = [chunk for chunk, score in relevant_results]
        
        if not relevant_chunks:
            logger.warning(f"No relevant content found for topic: {topic}")
            return []
        
        # Generate questions from relevant chunks
        targeted_questions = self.generate_questions(target_chunks=relevant_chunks)
        
        # Limit to requested number
        return targeted_questions[:num_questions]
    
    def get_vault_stats(self) -> Dict:
        """Get statistics about the parsed vault."""
        if not self.vault_parser or not self.notes:
            return {}
        return self.vault_parser.get_vault_stats(self.notes)
    
    def get_preprocessing_stats(self) -> Dict:
        """Get statistics about content preprocessing."""
        if not self.preprocessor or not self.chunks:
            return {}
        return self.preprocessor.get_chunk_statistics(self.chunks)
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the semantic index."""
        if not self.retriever:
            return {}
        return self.retriever.get_index_stats()
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about question generation."""
        if not self.generator or not self.questions:
            return {}
        return self.generator.get_generation_stats(self.questions)
    
    def save_pipeline_state(self, filepath: str) -> None:
        """Save the current pipeline state to disk."""
        if self.retriever:
            self.retriever.save_index(f"{filepath}_index")
        
        # Save configuration
        self.config_manager.save_config(f"{filepath}_config.yaml")
        
        logger.info(f"Saved pipeline state to {filepath}")
    
    def load_pipeline_state(self, filepath: str) -> None:
        """Load pipeline state from disk."""
        # Load configuration
        config_path = f"{filepath}_config.yaml"
        if os.path.exists(config_path):
            self.config = self.config_manager.load_config(config_path)
        
        # Initialize components with loaded config
        self.retriever = SemanticRetriever(
            model_name=self.config.retrieval.model_name,
            index_type=self.config.retrieval.index_type,
            cache_dir=self.config.retrieval.cache_dir
        )
        
        # Load index
        index_path = f"{filepath}_index"
        if os.path.exists(f"{index_path}.faiss"):
            self.retriever.load_index(index_path)
            self.chunks = self.retriever.chunks
        
        logger.info(f"Loaded pipeline state from {filepath}")
    
    def create_sample_config(self, config_path: str = "config.yaml") -> str:
        """Create a sample configuration file."""
        return self.config_manager.create_default_config_file(config_path)