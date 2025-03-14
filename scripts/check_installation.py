#!/usr/bin/env python
"""
Check if the thoughtful-agents package is installed correctly.

Usage:
    python scripts/check_installation.py
"""

import importlib
import sys

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Check if the package is installed correctly."""
    print("Checking if thoughtful-agents is installed correctly...")
    
    # List of modules to check
    modules = [
        "thoughtful_agents",
        "thoughtful_agents.models",
        "thoughtful_agents.models.participant",
        "thoughtful_agents.models.conversation",
        "thoughtful_agents.models.thought",
        "thoughtful_agents.models.memory",
        "thoughtful_agents.models.mental_object",
        "thoughtful_agents.models.enums",
        "thoughtful_agents.utils",
        "thoughtful_agents.utils.thinking_engine",
        "thoughtful_agents.utils.turn_taking_engine",
        "thoughtful_agents.utils.llm_api",
        "thoughtful_agents.utils.saliency",
        "thoughtful_agents.utils.text_splitter",
    ]
    
    # Check each module
    all_passed = all(check_module(module) for module in modules)
    
    # Check if spaCy is installed
    try:
        import spacy
        print(f"✅ spacy {spacy.__version__} imported successfully")
        
        # Check if the English model is installed
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ en_core_web_sm loaded successfully")
        except OSError:
            print("❌ en_core_web_sm not found. Please install it with:")
            print("   python -m spacy download en_core_web_sm")
            all_passed = False
    except ImportError:
        print("❌ Failed to import spacy")
        all_passed = False
    
    # Print summary
    if all_passed:
        print("\n✅ All checks passed! The package is installed correctly.")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 