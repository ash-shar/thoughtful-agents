#!/usr/bin/env python
"""
Hello World Example for Inner Thoughts AI

This is a simple example where two AI agents, Alice and Bob, participate in a
conversation for one turn. Alice initiates by sending a message, and Bob responds.
This example uses the turn-taking engine to decide which agent speaks next.
"""

import asyncio
from inner_thoughts_ai.models import (
    Agent, 
    Conversation, 
    Event, 
    EventType
)
from inner_thoughts_ai.utils.turn_taking_engine import predict_next_speaker, decide_next_speaker

async def main():
    # Create a conversation with a simple context
    conversation = Conversation(context="A friendly chat between Alice and Bob.")
    
    # Create two agents: Alice and Bob
    alice = Agent(name="Alice", proactivity_config={
        'im_threshold': 0.6,  # Lower threshold for articulating thoughts
        'system1_prob': 0.5,  # Higher probability for system1 thoughts 
    })
    
    bob = Agent(name="Bob", proactivity_config={
        'im_threshold': 0.6,  # Lower threshold for articulating thoughts
        'system1_prob': 0.5,  # Higher probability for system1 thoughts
    })
    
    # Add agents to the conversation
    conversation.add_participant(alice)
    conversation.add_participant(bob)
    
    print("\n==== Starting Conversation ====\n")
    
    # Alice starts the conversation
    alice.send_message("Hello Bob! How are you today?", conversation)
    print(f"Alice: Hello Bob! How are you today?")
    
    # Broadcast the event to let all agents think
    await conversation.broadcast_event()
    
    # Use the turn-taking engine to predict who should speak next
    predicted_speaker = await predict_next_speaker(conversation)
    print(f"Turn-taking engine predicts {predicted_speaker} should speak next")
    
    # Use the turn-taking engine to decide the next speaker and their utterance
    next_speaker, utterance = await decide_next_speaker(conversation)
    
    if next_speaker:
        # Send the message
        next_speaker.send_message(utterance, conversation)
        print(f"{next_speaker.name}: {utterance}")
        
        # Get the thought that led to this response (for demonstration)
        thoughts = await next_speaker.act(conversation)
        if thoughts:
            print(f"\n{next_speaker.name}'s thought: {thoughts[0].content}")
    else:
        print("No agent has thoughts to articulate.")
    
    print("\n==== End of Conversation ====\n")
    
    # Summary
    print("Conversation Summary:")
    for i, event in enumerate(conversation.event_history):
        if event.type == EventType.UTTERANCE:
            participant_name = event.participant_name
            content = event.content
            print(f"Turn {i+1}: {participant_name} said: \"{content}\"")

if __name__ == "__main__":
    asyncio.run(main()) 