# Thoughtful Agents

A framework for modeling agent thoughts and conversations, enabling more natural and human-like interactions between multiple AI agents and humans.

## Overview

Thoughtful Agents provides a structured approach to modeling the internal thought processes of AI agents during conversations. Rather than simply predicting conversational turns, this framework enables proactive AI driven by its own internal "thoughts".

This framework is based on the paper [Proactive Conversational Agents with Inner Thoughts](https://arxiv.org/pdf/2501.00383), published at [CHI 2025](https://doi.org/10.1145/3706598.3713760).

## Key Features

- Thinking engine for thought generation, evaluation, selection, and articulation
- System 1 (fast, automatic) and System 2 (slow, deliberate) thinking
- Mental object management (thoughts, memories)
- Saliency-based memory and thought retrieval
- Conversation and event tracking
- Turn-taking prediction and engine for determining when and who should speak next
- Proactivity configuration for agents

## Installation

```bash
pip install thoughtful-agents
```

Download the required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

```python
import asyncio
from thoughtful_agents.models import Agent, Conversation
from thoughtful_agents.utils.turn_taking_engine import decide_next_speaker_and_utterance, predict_turn_taking_type

async def main():
    # Create a conversation with a simple context
    conversation = Conversation(context="A friendly chat between Alice and Bob.")
    
    # Create agents with specific proactivity configurations
    alice = Agent(name="Alice", proactivity_config={
        'im_threshold': 3.2, 
        'system1_prob': 0.3,
        'interrupt_threshold': 4.5
    })
    
    bob = Agent(name="Bob", proactivity_config={
        'im_threshold': 3.2,
        'system1_prob': 0.3,
        'interrupt_threshold': 4.5
    })
    
    # Add background knowledge to the agents
    alice.initialize_memory("I am a software engineer who likes to code.")
    bob.initialize_memory("I am a cognitive scientist who works on understanding the human mind.")
    
    # Add agents to the conversation
    conversation.add_participant(alice)
    conversation.add_participant(bob)
    
    # Alice starts the conversation
    new_event = await alice.send_message("I'm recently thinking about adopting a cat. What do you think about this?", conversation)
    
    # Predict the next speaker before broadcasting the event
    turn_allocation_type = await predict_turn_taking_type(conversation)
    
    # Broadcast the event to let all agents think
    await conversation.broadcast_event(new_event)
    
    # Decide the next speaker and their utterance
    next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
    
    if next_speaker:
        await next_speaker.send_message(utterance, conversation)
        print(f"{next_speaker.name}: {utterance}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

For more detailed documentation and examples, visit the [GitHub repository](https://github.com/xybruceliu/thoughtful-agents).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
@inproceedings{liu2025inner,
    title={Proactive Conversational Agents with Inner Thoughts},
    author={Liu, Xingyu Bruce and Fang, Shitao and Shi, Weiyan and Wu, Chien-Sheng and Igarashi, Takeo and Chen, Xiang Anthony},
    booktitle = {Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems},
    year = {2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    location = {Yokohama, Japan},
    series = {CHI '25},
    keywords = {Full},    
    url = {https://doi.org/10.1145/3706598.3713760},
    doi = {10.1145/3706598.3713760},
}
``` 