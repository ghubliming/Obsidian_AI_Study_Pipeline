"""
Quiz generation module using AI models to create various types of quiz questions.
"""

import json
import os
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

from ..preprocessing import ContentChunk

logger = logging.getLogger(__name__)

class QuizType(Enum):
    """Types of quiz questions that can be generated."""
    FLASHCARD = "flashcard"
    MULTIPLE_CHOICE = "multiple_choice"
    CLOZE_DELETION = "cloze_deletion"
    SHORT_ANSWER = "short_answer"
    TRUE_FALSE = "true_false"

@dataclass
class QuizQuestion:
    """Represents a generated quiz question with metadata."""
    
    question: str
    answer: str
    quiz_type: QuizType
    source_chunk: str
    source_note: str
    source_path: str
    options: List[str] = field(default_factory=list)  # For multiple choice
    explanation: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

class QuizGenerator:
    """Generates quiz questions from content chunks using AI models."""
    
    def __init__(self, 
                 model_type: str = "ollama",
                 model_name: str = "llama3.2:1b",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize quiz generator.
        
        Args:
            model_type: Type of model to use ('ollama', 'openai', 'openrouter', 'gemini', 'huggingface')
            model_name: Name of the specific model
            api_key: API key if required (will check environment variables if not provided)
            base_url: Base URL for API if required (will use defaults for known services)
        """
        self.model_type = model_type
        self.model_name = model_name
        
        # Get API key from environment variables if not provided
        self.api_key = self._get_api_key(model_type, api_key)
        
        # Set base URL with defaults for known services
        self.base_url = self._get_base_url(model_type, base_url)
        
        # Initialize model client
        self.client = self._initialize_client()
        
        logger.info(f"Initialized QuizGenerator with {model_type}:{model_name}")
        if self.api_key:
            logger.info("API key loaded successfully")
        elif model_type in ["openrouter", "gemini", "openai"]:
            logger.warning(f"No API key found for {model_type}. Set environment variable or pass api_key parameter.")
    
    def _get_api_key(self, model_type: str, provided_key: Optional[str]) -> Optional[str]:
        """Get API key from provided parameter or environment variables."""
        if provided_key:
            return provided_key
        
        # Map model types to environment variable names
        env_var_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        
        env_var = env_var_map.get(model_type)
        if env_var:
            api_key = os.getenv(env_var)
            if api_key:
                logger.debug(f"Loaded API key from environment variable: {env_var}")
                return api_key
        
        return None
    
    def _get_base_url(self, model_type: str, provided_url: Optional[str]) -> Optional[str]:
        """Get base URL from provided parameter or use defaults."""
        if provided_url:
            return provided_url
        
        # Check environment variables for custom base URLs
        env_var_map = {
            "openrouter": "OPENROUTER_BASE_URL", 
            "openai": "OPENAI_BASE_URL"
        }
        
        env_var = env_var_map.get(model_type)
        if env_var:
            base_url = os.getenv(env_var)
            if base_url:
                logger.debug(f"Loaded base URL from environment variable: {env_var}")
                return base_url
        
        # Default base URLs for known services
        defaults = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1"
        }
        
        return defaults.get(model_type)
    
    def _initialize_client(self):
        """Initialize the appropriate model client."""
        if self.model_type == "ollama":
            try:
                import ollama
                return ollama
            except ImportError:
                logger.error("Ollama not installed. Install with: pip install ollama")
                return None
        elif self.model_type == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                return client
            except ImportError:
                logger.error("OpenAI not installed. Install with: pip install openai")
                return None
        elif self.model_type == "openrouter":
            try:
                import openai
                # Openrouter uses OpenAI-compatible API
                base_url = self.base_url or "https://openrouter.ai/api/v1"
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=base_url
                )
                return client
            except ImportError:
                logger.error("OpenAI client not installed. Install with: pip install openai")
                return None
        elif self.model_type == "gemini":
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                return genai
            except ImportError:
                logger.error("Google Generative AI not installed. Install with: pip install google-generativeai")
                return None
        else:
            logger.warning(f"Unsupported model type: {self.model_type}")
            return None
    
    def generate_quiz_questions(self,
                              chunks: List[ContentChunk],
                              quiz_types: List[QuizType] = None,
                              questions_per_chunk: int = 2,
                              max_questions: int = 50) -> List[QuizQuestion]:
        """
        Generate quiz questions from content chunks.
        
        Args:
            chunks: List of content chunks to generate questions from
            quiz_types: Types of quiz questions to generate
            questions_per_chunk: Number of questions per chunk
            max_questions: Maximum total questions to generate
            
        Returns:
            List of generated quiz questions
        """
        if quiz_types is None:
            quiz_types = [QuizType.FLASHCARD, QuizType.MULTIPLE_CHOICE, QuizType.SHORT_ANSWER]
        
        all_questions = []
        questions_generated = 0
        
        for chunk in chunks:
            if questions_generated >= max_questions:
                break
            
            try:
                # Generate questions for this chunk
                chunk_questions = self._generate_questions_for_chunk(
                    chunk, quiz_types, questions_per_chunk
                )
                
                all_questions.extend(chunk_questions)
                questions_generated += len(chunk_questions)
                
                logger.debug(f"Generated {len(chunk_questions)} questions from chunk in {chunk.source_note}")
                
            except Exception as e:
                logger.error(f"Error generating questions for chunk from {chunk.source_note}: {e}")
        
        logger.info(f"Generated {len(all_questions)} total quiz questions")
        return all_questions
    
    def _generate_questions_for_chunk(self,
                                    chunk: ContentChunk,
                                    quiz_types: List[QuizType],
                                    num_questions: int) -> List[QuizQuestion]:
        """Generate questions for a single content chunk."""
        questions = []
        
        for i in range(num_questions):
            # Randomly select quiz type
            quiz_type = random.choice(quiz_types)
            
            try:
                question = self._generate_single_question(chunk, quiz_type)
                if question:
                    questions.append(question)
            except Exception as e:
                logger.error(f"Error generating {quiz_type.value} question: {e}")
        
        return questions
    
    def _generate_single_question(self, chunk: ContentChunk, quiz_type: QuizType) -> Optional[QuizQuestion]:
        """Generate a single question of the specified type."""
        if self.client is None:
            return self._generate_fallback_question(chunk, quiz_type)
        
        prompt = self._create_prompt(chunk, quiz_type)
        
        try:
            if self.model_type == "ollama":
                response = self.client.chat(model=self.model_name, messages=[
                    {"role": "user", "content": prompt}
                ])
                response_text = response['message']['content']
            elif self.model_type in ["openai", "openrouter"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            elif self.model_type == "gemini":
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                response_text = response.text
            else:
                return self._generate_fallback_question(chunk, quiz_type)
            
            return self._parse_response(response_text, chunk, quiz_type)
            
        except Exception as e:
            logger.error(f"Error calling {self.model_type} API: {e}")
            return self._generate_fallback_question(chunk, quiz_type)
    
    def _create_prompt(self, chunk: ContentChunk, quiz_type: QuizType) -> str:
        """Create a prompt for question generation."""
        base_context = f"""
Content: {chunk.text}
Source: {chunk.source_note} ({chunk.source_path})
Content Type: {chunk.chunk_type}

Generate a {quiz_type.value.replace('_', ' ')} question based on this content.
"""
        
        if quiz_type == QuizType.FLASHCARD:
            return base_context + """
Create a flashcard with a clear question and concise answer.
Format your response as JSON:
{
    "question": "Your question here",
    "answer": "Your answer here",
    "explanation": "Brief explanation if needed"
}
"""
        
        elif quiz_type == QuizType.MULTIPLE_CHOICE:
            return base_context + """
Create a multiple choice question with 4 options (A, B, C, D).
Make sure only one option is correct and the others are plausible distractors.
Format your response as JSON:
{
    "question": "Your question here",
    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    "answer": "A",
    "explanation": "Why this answer is correct"
}
"""
        
        elif quiz_type == QuizType.CLOZE_DELETION:
            return base_context + """
Create a cloze deletion (fill-in-the-blank) question by removing a key term or phrase.
Format your response as JSON:
{
    "question": "Text with _____ for the missing part",
    "answer": "The missing word or phrase",
    "explanation": "Context about the missing term"
}
"""
        
        elif quiz_type == QuizType.SHORT_ANSWER:
            return base_context + """
Create a short answer question that requires understanding of the content.
Format your response as JSON:
{
    "question": "Your question here",
    "answer": "Expected short answer",
    "explanation": "Additional context"
}
"""
        
        elif quiz_type == QuizType.TRUE_FALSE:
            return base_context + """
Create a true/false question based on the content.
Format your response as JSON:
{
    "question": "Statement to evaluate as true or false",
    "answer": "true" or "false",
    "explanation": "Why this is true or false"
}
"""
        
        return base_context
    
    def _parse_response(self, response_text: str, chunk: ContentChunk, quiz_type: QuizType) -> Optional[QuizQuestion]:
        """Parse the AI model response into a QuizQuestion object."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                # Fallback parsing for non-JSON responses
                return self._parse_non_json_response(response_text, chunk, quiz_type)
            
            # Extract common fields
            question = response_data.get('question', '').strip()
            answer = response_data.get('answer', '').strip()
            explanation = response_data.get('explanation', '').strip()
            
            if not question or not answer:
                logger.warning("Generated question or answer is empty")
                return None
            
            # Handle options for multiple choice
            options = []
            if quiz_type == QuizType.MULTIPLE_CHOICE:
                options = response_data.get('options', [])
                if len(options) != 4:
                    logger.warning("Multiple choice question doesn't have 4 options")
                    return None
            
            return QuizQuestion(
                question=question,
                answer=answer,
                quiz_type=quiz_type,
                source_chunk=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                source_note=chunk.source_note,
                source_path=chunk.source_path,
                options=options,
                explanation=explanation,
                tags=chunk.metadata.get('tags', []) if chunk.metadata else [],
                confidence_score=0.8  # Default confidence for AI-generated questions
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None
    
    def _parse_non_json_response(self, response_text: str, chunk: ContentChunk, quiz_type: QuizType) -> Optional[QuizQuestion]:
        """Parse non-JSON responses with fallback logic."""
        lines = response_text.strip().split('\n')
        question = ""
        answer = ""
        
        # Simple pattern matching for common formats
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('question:', 'q:')):
                question = line.split(':', 1)[1].strip()
            elif line.lower().startswith(('answer:', 'a:')):
                answer = line.split(':', 1)[1].strip()
        
        if question and answer:
            return QuizQuestion(
                question=question,
                answer=answer,
                quiz_type=quiz_type,
                source_chunk=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                source_note=chunk.source_note,
                source_path=chunk.source_path,
                tags=chunk.metadata.get('tags', []) if chunk.metadata else [],
                confidence_score=0.6  # Lower confidence for parsed responses
            )
        
        return None
    
    def _generate_fallback_question(self, chunk: ContentChunk, quiz_type: QuizType) -> Optional[QuizQuestion]:
        """Generate a simple fallback question when AI model is unavailable."""
        text = chunk.text
        
        if quiz_type == QuizType.FLASHCARD:
            # Create a simple what/who/when question
            if any(word in text.lower() for word in ['what', 'definition', 'meaning']):
                question = f"What is the main concept discussed in this text from {chunk.source_note}?"
                answer = f"Based on the content: {text[:100]}..."
            else:
                question = f"Explain the key idea from {chunk.source_note}"
                answer = f"Key idea: {text[:150]}..."
            
            return QuizQuestion(
                question=question,
                answer=answer,
                quiz_type=quiz_type,
                source_chunk=text[:200] + "..." if len(text) > 200 else text,
                source_note=chunk.source_note,
                source_path=chunk.source_path,
                tags=chunk.metadata.get('tags', []) if chunk.metadata else [],
                confidence_score=0.3  # Low confidence for fallback
            )
        
        # For other types, return None to indicate failure
        return None
    
    def validate_questions(self, questions: List[QuizQuestion]) -> List[QuizQuestion]:
        """Validate and filter generated questions."""
        valid_questions = []
        
        for question in questions:
            if self._is_valid_question(question):
                valid_questions.append(question)
            else:
                logger.debug(f"Filtered out invalid question: {question.question[:50]}...")
        
        logger.info(f"Validated {len(valid_questions)} out of {len(questions)} questions")
        return valid_questions
    
    def _is_valid_question(self, question: QuizQuestion) -> bool:
        """Check if a question meets quality criteria."""
        # Basic validation
        if len(question.question.strip()) < 10:
            return False
        
        if len(question.answer.strip()) < 2:
            return False
        
        # Check for question marks where appropriate
        if question.quiz_type != QuizType.CLOZE_DELETION and not question.question.strip().endswith('?'):
            if not any(word in question.question.lower() for word in ['what', 'who', 'when', 'where', 'why', 'how']):
                return False
        
        # Validate multiple choice options
        if question.quiz_type == QuizType.MULTIPLE_CHOICE:
            if len(question.options) != 4:
                return False
            if question.answer not in ['A', 'B', 'C', 'D']:
                return False
        
        return True
    
    def get_generation_stats(self, questions: List[QuizQuestion]) -> Dict:
        """Get statistics about generated questions."""
        if not questions:
            return {}
        
        type_counts = {}
        source_counts = {}
        difficulty_counts = {}
        avg_confidence = 0
        
        for q in questions:
            type_counts[q.quiz_type.value] = type_counts.get(q.quiz_type.value, 0) + 1
            source_counts[q.source_note] = source_counts.get(q.source_note, 0) + 1
            difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
            avg_confidence += q.confidence_score
        
        avg_confidence /= len(questions)
        
        return {
            "total_questions": len(questions),
            "question_types": type_counts,
            "questions_per_source": source_counts,
            "difficulty_distribution": difficulty_counts,
            "average_confidence": avg_confidence,
            "unique_sources": len(source_counts)
        }