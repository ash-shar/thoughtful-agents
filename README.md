# Inner Thoughts AI

A framework for modeling agent thoughts and conversations, enabling more natural and human-like interactions between AI agents and humans.

## Overview

Inner Thoughts AI provides a structured approach to modeling the internal thought processes of AI agents during conversations. The framework includes:

- Mental object management (thoughts, memories)
- Conversation and event tracking
- Turn-taking prediction
- Saliency-based memory and thought retrieval

## Installation

1. Install the package and its dependencies:

```bash
pip install -e .
```

2. Download the required spaCy model:

```bash
python scripts/download_spacy_model.py
```

Or manually:

```bash
python -m spacy download en_core_web_sm
```

3. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Project Structure

The project is organized as follows:

- `inner_thoughts_ai/models/`: Core model classes
  - `mental_object.py`: Base class for mental objects
  - `saliency.py`: Saliency computation
  - `turn_taking.py`: Turn-taking prediction
  - `conversation.py`: Conversation and Event classes
  - `enums.py`: Enumeration types
  - `memory.py`: Memory-related classes
  - `participant.py`: Participant, Human, and Agent classes
  - `thought.py`: Thought-related classes
- `inner_thoughts_ai/utils/`: Utility functions
  - `text_splitter.py`: Text splitting using spaCy
  - `llm_api.py`: OpenAI API interaction

## License

This project is licensed under the MIT License - see the LICENSE file for details.
