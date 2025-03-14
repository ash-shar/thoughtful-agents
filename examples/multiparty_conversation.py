"""
Multi-Party Conversation Example for Inner Thoughts AI

This example demonstrates a casual conversation between three AI agents (Alice, Bob, and Charlie)
discussing what they did last weekend. The conversation continues for a specified number of turns,
controlled by the `num_turns` parameter.

You can customize the LLM models used by setting environment variables:
```bash
# Set the completion model (default: gpt-4o)
export COMPLETION_MODEL=gpt-4-turbo

# Set the embedding model (default: text-embedding-3-small)
export EMBEDDING_MODEL=text-embedding-3-large

# Then run the example
python examples/multiparty_conversation.py
```
"""

import asyncio
import sys
import os
from typing import Optional

from thoughtful_agents.models import (
    Agent, 
    Conversation,
)
from thoughtful_agents.utils.turn_taking_engine import decide_next_speaker_and_utterance, predict_turn_taking_type
from thoughtful_agents.utils.llm_api import DEFAULT_COMPLETION_MODEL, DEFAULT_EMBEDDING_MODEL

async def run_conversation(num_turns: int = 5, verbose: bool = True):
    """
    Run a conversation between Alice, Bob, and Charlie about their weekend activities.
    
    Args:
        num_turns: Number of conversation turns to simulate
        verbose: Whether to print detailed thought processes and turn-taking predictions
    """
    # Create a conversation with a simple context
    conversation = Conversation(context="A casual conversation between friends Alice, Bob, and Charlie about what they did last weekend.")
    
    # Create three agents with their respective proactivity configurations
    alice = Agent(name="Alice", proactivity_config={
        'im_threshold': 3.0,  # Moderate threshold for speaking
        'system1_prob': 0.4,  # Moderate chance of quick, intuitive responses
        'interrupt_threshold': 4.2  # High threshold to avoid frequent interruptions
    })
    
    bob = Agent(name="Bob", proactivity_config={
        'im_threshold': 2.8,  # Slightly more eager to speak
        'system1_prob': 0.5,  # Higher chance of quick, intuitive responses
        'interrupt_threshold': 4.0  # More likely to interrupt when highly motivated
    })
    
    charlie = Agent(name="Charlie", proactivity_config={
        'im_threshold': 3.0,  # Moderate threshold for speaking
        'system1_prob': 0.3,  # More deliberate responses
        'interrupt_threshold': 4.5  # Very unlikely to interrupt
    })
    
    # Initialize long-term memories for each agent
    alice_memories = """Last weekend I spent the afternoon reading a new novel from the local library. 
Last weekend I visited an international food festival with friends, sampling different cuisines. 
Last weekend I enjoyed a picnic lunch outside with my family, appreciating the cool weather. 
I like listening to all genres of music except country music. 
I would travel the world if I could. 
I like to read a lot of books. 
I like spending time with my friends and family. 
I am not much of a fan of hot weather. 
My favorite book that I've read lately is Middlesex."""
    
    bob_memories = """Last weekend I prepared a homemade Italian dinner, trying out a new pasta recipe. 
Last weekend I took a trip to the beach with my dog, enjoying a cooler, breezy day by the water. 
Last weekend I spent the evening singing along to favorite songs while taking a break from grad school studies. 
I enjoy cooking Italian food. 
I am obsessed with my dog. 
I'm in my last year of grad school. 
I love trips to the beach. 
My favorite book that I've read lately is Middlesex. 
I am not much of a fan of hot weather."""
    
    charlie_memories = """Last weekend I cooked a delicious Italian dinner to enjoy with my partner. 
Last weekend I spent a relaxing afternoon doing yoga and writing poetry in the park. 
Last weekend I watched a romantic comedy movie at home. 
I like to write poetry in my spare time. 
I love romantic comedies. 
I am a yoga instructor. 
My favorite musician is Ed Sheeran. 
I like listening to all genres of music except country music. 
I'm proud to have a happy and healthy relationship with my partner."""
    
    # Add memories to the agents (splitting by paragraphs)
    alice.initialize_memory(alice_memories, by_paragraphs=True)
    bob.initialize_memory(bob_memories, by_paragraphs=True)
    charlie.initialize_memory(charlie_memories, by_paragraphs=True)
    
    # Add agents to the conversation
    conversation.add_participant(alice)
    conversation.add_participant(bob)
    conversation.add_participant(charlie)
    
    print("\n==== 🚀 Starting Multi-Party Conversation 🚀 ====\n")
    
    # Alice starts the conversation with a question about the weekend
    new_event = await alice.send_message("Hey everyone! What did you all do last weekend?", conversation)
    print(f"👤 Alice: Hey everyone! What did you all do last weekend?")
    
    # Predict the next speaker before broadcasting the event
    turn_allocation_type = await predict_turn_taking_type(conversation)
    # Use the turn-taking engine to predict who should speak next (this is just a prediction)
    if verbose:
        print(f"🎯 Turn-taking engine predicts that the turn is allocated to {turn_allocation_type}")  
    
    # Broadcast the event to let all agents think
    await conversation.broadcast_event(new_event)
    
    # Run the conversation for the specified number of turns
    for turn in range(num_turns):
        print(f"\n---- Turn {turn + 1} ----")
        
        # Show each agent's thoughts and their intrinsic motivation scores if verbose
        if verbose:
            for participant in conversation.get_agents():
                print(f"🧠 {participant.name}'s thoughts:")
                for thought in participant.thought_reservoir.thoughts:
                    if thought.generated_turn == conversation.turn_number:
                        print(f"  💭 {thought.content} (Intrinsic Motivation: {thought.intrinsic_motivation['score']})")
        
        # Determine the actual next speaker based on intrinsic motivation
        next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
        
        if next_speaker:
            # Send the message
            new_event = await next_speaker.send_message(utterance, conversation)
            print(f"👤 {next_speaker.name}: {utterance}")
            
            # Predict the next speaker before broadcasting the event
            turn_allocation_type = await predict_turn_taking_type(conversation)
            if verbose:
                print(f"🎯 Turn-taking engine predicts that the turn is allocated to {turn_allocation_type}")
            
            # Broadcast the event to let all agents think
            await conversation.broadcast_event(new_event)
        else:
            print("❌ No agent has thoughts to articulate.")
            break
    
    print("\n==== 🏁 End of Conversation 🏁 ====\n")
    
    # Summary
    print("📋 Conversation Summary:")
    for i, event in enumerate(conversation.event_history):
        participant_name = event.participant_name
        content = event.content
        print(f"🔄 Turn {i+1}: {participant_name}: \"{content}\"")

async def main():
    # Get number of turns from command line arguments or use default
    num_turns = 5  # Default number of turns
    verbose = True  # Default verbose setting
    
    if len(sys.argv) > 1:
        try:
            num_turns = int(sys.argv[1])
            if num_turns < 1:
                print("Number of turns must be at least 1. Using default of 5.")
                num_turns = 5
        except ValueError:
            print("Invalid number of turns. Using default of 5.")
    
    if len(sys.argv) > 2:
        verbose_arg = sys.argv[2].lower()
        if verbose_arg in ['false', 'no', '0']:
            verbose = False
    
    # Print the models being used
    print(f"Using completion model: {DEFAULT_COMPLETION_MODEL}")
    print(f"Using embedding model: {DEFAULT_EMBEDDING_MODEL}")
    
    await run_conversation(num_turns, verbose)

if __name__ == "__main__":
    asyncio.run(main()) 