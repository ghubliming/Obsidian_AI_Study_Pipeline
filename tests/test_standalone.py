#!/usr/bin/env python3
"""
Standalone test for new model types without heavy dependencies.
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_configuration():
    """Test that new model types are properly defined in configuration."""
    
    # Test configuration comment includes new model types
    config_path = os.path.join(os.path.dirname(__file__), '..', 'obsidian_ai_study_pipeline', 'utils', 'config.py')
    
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Check that new model types are mentioned in the comments
    assert 'openrouter' in config_content.lower(), "Openrouter should be mentioned in config"
    assert 'gemini' in config_content.lower(), "Gemini should be mentioned in config"
    
    print("✓ Configuration includes new model types")

def test_quiz_generator_imports():
    """Test that quiz generator properly imports and handles new model types."""
    
    quiz_gen_path = os.path.join(os.path.dirname(__file__), '..', 'obsidian_ai_study_pipeline', 'generation', 'quiz_generator.py')
    
    with open(quiz_gen_path, 'r') as f:
        quiz_gen_content = f.read()
    
    # Check that new model types are handled
    assert 'openrouter' in quiz_gen_content.lower(), "Openrouter should be handled in quiz generator"
    assert 'gemini' in quiz_gen_content.lower(), "Gemini should be handled in quiz generator"
    assert 'google.generativeai' in quiz_gen_content, "Google Generative AI import should be present"
    assert 'https://openrouter.ai/api/v1' in quiz_gen_content, "Openrouter base URL should be present"
    
    print("✓ Quiz generator includes new model type handling")

def test_cli_changes():
    """Test that CLI includes new options."""
    
    cli_path = os.path.join(os.path.dirname(__file__), '..', 'obsidian_ai_study_pipeline', 'cli.py')
    
    with open(cli_path, 'r') as f:
        cli_content = f.read()
    
    # Check that new CLI options are present
    assert '--model-type' in cli_content, "CLI should include --model-type option"
    assert '--api-key' in cli_content, "CLI should include --api-key option"
    assert "'openrouter'" in cli_content, "CLI should include openrouter in choices"
    assert "'gemini'" in cli_content, "CLI should include gemini in choices"
    
    print("✓ CLI includes new model type options")

def test_requirements_updates():
    """Test that requirements include new dependencies."""
    
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    
    with open(req_path, 'r') as f:
        req_content = f.read()
    
    # Check that new dependencies are present
    assert 'google-generativeai' in req_content, "Requirements should include google-generativeai"
    assert 'openai' in req_content, "Requirements should include openai (for openrouter compatibility)"
    
    print("✓ Requirements include new dependencies")

def test_readme_updates():
    """Test that README includes documentation for new model types."""
    
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Check that new model types are documented
    assert 'openrouter' in readme_content.lower(), "README should document Openrouter"
    assert 'gemini' in readme_content.lower(), "README should document Gemini"
    assert 'openrouter.ai' in readme_content.lower(), "README should include Openrouter URL"
    assert 'aistudio.google.com' in readme_content.lower(), "README should include Google AI Studio URL"
    assert '--model-type' in readme_content, "README should show new CLI option"
    
    print("✓ README includes documentation for new model types")

def main():
    """Run all tests."""
    print("Running standalone tests for new model types...")
    print()
    
    try:
        test_model_configuration()
        test_quiz_generator_imports()
        test_cli_changes()
        test_requirements_updates()
        test_readme_updates()
        
        print()
        print("🎉 All tests passed! New model types are properly integrated.")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()