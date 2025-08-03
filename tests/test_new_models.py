"""
Test for new model types (Openrouter and Gemini) integration.
"""

import pytest
from unittest.mock import Mock, patch
from obsidian_ai_study_pipeline.generation.quiz_generator import QuizGenerator, QuizType
from obsidian_ai_study_pipeline.preprocessing import ContentChunk


def test_openrouter_client_initialization():
    """Test that Openrouter client is initialized correctly."""
    with patch('obsidian_ai_study_pipeline.generation.quiz_generator.openai') as mock_openai:
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        generator = QuizGenerator(
            model_type="openrouter",
            model_name="microsoft/phi-3-mini-128k-instruct:free",
            api_key="test-key"
        )
        
        # Verify OpenAI client was called with correct parameters
        mock_openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1"
        )
        assert generator.client == mock_client


def test_gemini_client_initialization():
    """Test that Gemini client is initialized correctly."""
    with patch('obsidian_ai_study_pipeline.generation.quiz_generator.google.generativeai') as mock_genai:
        generator = QuizGenerator(
            model_type="gemini",
            model_name="gemini-1.5-flash",
            api_key="test-key"
        )
        
        # Verify Gemini client was configured correctly
        mock_genai.configure.assert_called_once_with(api_key="test-key")
        assert generator.client == mock_genai


def test_unsupported_model_type():
    """Test handling of unsupported model types."""
    generator = QuizGenerator(
        model_type="unsupported",
        model_name="some-model"
    )
    
    assert generator.client is None


def test_openrouter_custom_base_url():
    """Test that custom base URL is used for Openrouter."""
    with patch('obsidian_ai_study_pipeline.generation.quiz_generator.openai') as mock_openai:
        custom_url = "https://custom-openrouter.com/api/v1"
        
        generator = QuizGenerator(
            model_type="openrouter",
            model_name="test-model",
            api_key="test-key",
            base_url=custom_url
        )
        
        mock_openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url
        )


def test_model_type_validation():
    """Test that model types are correctly identified."""
    # Test all supported model types
    supported_types = ["ollama", "openai", "openrouter", "gemini"]
    
    for model_type in supported_types:
        generator = QuizGenerator(model_type=model_type, model_name="test-model")
        assert generator.model_type == model_type


@pytest.fixture
def sample_content_chunk():
    """Create a sample content chunk for testing."""
    return ContentChunk(
        text="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        chunk_type="paragraph",
        source_note="ML Basics",
        source_path="/notes/ml_basics.md",
        metadata={"tags": ["ml", "ai"]}
    )


def test_fallback_question_generation_with_new_models(sample_content_chunk):
    """Test that fallback generation works with new model types."""
    # Test with openrouter (when client is None)
    generator = QuizGenerator(model_type="openrouter", model_name="test-model")
    generator.client = None  # Simulate failed initialization
    
    fallback_question = generator._generate_fallback_question(
        sample_content_chunk, 
        QuizType.FLASHCARD
    )
    
    assert fallback_question is not None
    assert fallback_question.quiz_type == QuizType.FLASHCARD
    assert "ML Basics" in fallback_question.question
    assert fallback_question.confidence_score == 0.3


if __name__ == "__main__":
    pytest.main([__file__])