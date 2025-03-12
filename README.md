# Proactive Agents with Inner Thoughts 💭

A framework for modeling agent thoughts and conversations, enabling more natural and human-like interactions between multiple AI agents and humans.

## Overview

Thoughtful Agents provides a structured approach to modeling the internal thought processes of AI agents during conversations. Rather than simply predicting conversational turns, this framework enables proactive AI driven by its own internal "thoughts".

This framework is based on the paper [Proactive Conversational Agents with Inner Thoughts](https://arxiv.org/pdf/2501.00383), published at [CHI 2025](https://doi.org/10.1145/3706598.3713760).

![Inner Thoughts Framework Architecture](assets/images/framework_architecture.png)

Inspired by cognitive architectures and LLM prompting techniques, the framework comprises five stages:
1. **Trigger** - Initiating the thought process
2. **Retrieval** - Accessing relevant memories and context
3. **Thought Formation** - Generating potential thoughts
4. **Evaluation** - Assessing intrinsic motivation to express thoughts
5. **Participation** - Deciding when and how to engage in conversation

The Python implementation includes:
- Thinking engine for thought generation, evaluation, selection, and articulation
- System 1 (fast, automatic) and System 2 (slow, deliberate) thinking
- Mental object management (thoughts, memories)
- Saliency-based memory and thought retrieval
- Conversation and event tracking
- Turn-taking prediction and engine for determining when and who should speak next
- Proactivity configuration for agents

## Research Background

Most current LLM-based systems treat AI as passive respondents, responding only to explicit human prompts. The Inner Thoughts framework takes an alternative approach inspired by human cognition, where we process others' words, reflect on our experiences, and develop an internal train of thoughts before deciding to participate.

This approach models the distinction between *covert responses* (internal thoughts and feelings) and *overt responses* (verbal utterances) in human communication. The AI participant determines whether to engage based on an evaluation of its intrinsic motivation to express a particular thought at that moment.

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
  - `participant.py`: Participant, Human, and Agent classes
  - `thought.py`: Thought-related classes
  - `memory.py`: Memory-related classes
  - `conversation.py`: Conversation and Event classes
  - `mental_object.py`: Base class for mental objects
  - `enums.py`: Enumeration types
- `inner_thoughts_ai/utils/`: Utility functions
  - `llm_api.py`: OpenAI API interaction
  - `saliency.py`: Saliency computation
  - `thinking_engine.py`: Functions for thought generation, evaluation, and articulation
  - `turn_taking_engine.py`: Turn-taking prediction 
  - `text_splitter.py`: Text splitting using spaCy
- `examples/`: Example implementations
  - `hello_world.py`: Simple example with multiple agents in conversation
  - `ai_thought_process.py`: Detailed example showing the AI's thought process
  - `lecture_practice.py`: Example of an AI providing proactive feedback during a lecture practice

## Key Components

### Participants

The framework models different types of conversation participants:
- `Participant`: Base class for all conversation participants
- `Human`: Represents human participants in the conversation
- `Agent`: AI agents that can generate thoughts, evaluate them, and decide when to participate

### Thoughts and Memory

Both `Thought` and `Memory` are subclasses of `MentalObject`.
The framework distinguishes between:
- `Thought`: Temporary mental objects representing current thinking
- `Memory`: Longer-term mental objects stored for future retrieval

They are managed through the `ThoughtReservoir` and `MemoryStore` classes respectively.

### Conversation and Events

The `Conversation` class manages the overall conversation state, while `Event` objects represent individual utterances or any other actions within the conversation.

### Mental Objects

The `MentalObject` class serves as the base for all mental entities in the system. It includes attributes like content, embedding, saliency, and weight that determine how important and relevant the object is in a given context.

### Thinking Engine

Key functions in `thinking_engine.py` include:
- `generate_system1_thought()`: Creates quick, intuitive thoughts based on immediate context
- `generate_system2_thoughts()`: Produces deliberate, reflective thoughts with deeper reasoning
- `evaluate_thought()`: Assesses thoughts and assigns intrinsic motivation scores (1-5)
- `articulate_thought()`: Transforms internal thoughts into natural language utterances

### Turn-Taking

The turn-taking engine predicts appropriate moments for participation and decides which agent should speak next based on their intrinsic motivation scores.

### Proactivity Configuration

Inner Thoughts offers fine-grained control over AI conversation participation through three proactivity layers:

1. **Overt Proactivity**: Controls conversation engagement tendency via the `system1_prob` parameter (0-1).

2. **Covert Proactivity**: Sets motivation threshold for expression using the `im_threshold` parameter (1-5).

3. **Tonal Proactivity**: Adjusts language assertiveness with the `proactive_tone` parameter (true/false).

The framework also supports **interruption** through the `interrupt_threshold` parameter (1-5), allowing AIs to override turn allocation when highly motivated.

To determine when and how the AI participates:
- For open turns: AI speaks if motivation exceeds threshold
- For allocated turns: AI uses highest-rated thought
- For others' turns: AI interrupts only with sufficient motivation

## Usage Examples

### Basic Example

The `hello_world.py` example demonstrates a simple conversation between multiple AI agents:

```python
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
await alice.send_message("I'm recently thinking about adopting a cat. What do you think about this?", conversation)

# Use the turn-taking engine to decide who speaks next
next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
```

### Detailed Thought Process Example

The `ai_thought_process.py` provides a more detailed look at the AI's internal thought process:

```python
# Create a human and an AI agent
human = Human(name="Human")
ai_agent = Agent(name="AI Assistant", proactivity_config={
    'im_threshold': 3.2,
    'system1_prob': 0.3,
    'interrupt_threshold': 4.5
})

# Human starts the conversation
human_event = await human.send_message("How are AI agents designed to participate in conversations?", conversation)

# AI thinking process
await ai_agent.recalibrate_saliency_for_event(human_event)
ai_agent.add_event_to_memory(human_event)
new_thoughts = await ai_agent.generate_thoughts(conversation, num_system1=1, num_system2=2)
await ai_agent.evaluate_thoughts(new_thoughts, conversation)

# AI selects and articulates thoughts
selected_thoughts = await ai_agent.select_thoughts(new_thoughts, conversation)
if selected_thoughts:
    ai_response = await ai_agent.articulate_thought(selected_thoughts[0], conversation)
    await ai_agent.send_message(ai_response, conversation)
```

### Lecture Practice Example

The `lecture_practice.py` example demonstrates how an AI assistant can provide proactive feedback during a lecture practice, without the user having to ask for it:

```python
# Create a conversation with context for practicing a lecture
conversation = Conversation(context="A user is practicing a lecture on artificial intelligence, and an AI assistant is providing feedback.")

# Create the human presenter and AI feedback assistant
human = Human(name="Presenter")
ai_assistant = Agent(name="Feedback Assistant", proactivity_config={
    'im_threshold': 3.0,  
    'system1_prob': 0.0,
    'interrupt_threshold': 3.5  # Higher threshold to reduce interruptions
})

# Add background knowledge to the AI assistant
background_knowledge = """I'm an AI assistant designed to provide helpful feedback on presentations and lectures.
My goal is to be helpful but not intrusive. I should:
1. Only interrupt for critical feedback that would significantly improve the presentation.
2. Note minor issues but save them for when the presenter pauses or asks for feedback.
3. Pay attention to content accuracy, delivery style, pacing, and engagement."""

ai_assistant.initialize_memory(background_knowledge, by_paragraphs=True)

# Process each lecture segment in a loop
for i, segment in enumerate(lecture_segments):
    # Human presenter speaks
    human_event = await human.send_message(segment["content"].strip(), conversation)
    
    # Broadcast the event to let the AI think
    await conversation.broadcast_event(human_event)
    
    # Use the turn-taking engine to decide if AI should provide feedback
    next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
    
    if next_speaker and next_speaker.name == "Feedback Assistant":
        await ai_assistant.send_message(utterance, conversation)
```

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

## Contact

For questions or feedback, please feel free to reach out to [Xingyu Bruce Liu](https://liubruce.me/)!
