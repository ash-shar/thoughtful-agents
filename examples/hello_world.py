"""
Hello World Example for Thoughtful Agents

This is a simple example where two AI agents, Alice and Bob, participate in a
conversation for one turn. Alice initiates by sending a message, and Bob responds.
This example uses the turn-taking engine to decide which agent speaks next.
"""

import asyncio
from thoughtful_agents.models import (
    Agent, 
    Conversation, 
)
from thoughtful_agents.utils.turn_taking_engine import decide_next_speaker_and_utterance, predict_turn_taking_type

async def main():

    # Create a conversation with a simple context
    conversation = Conversation(context="A friendly chat between Alice and Bob.")
    
    # Create two agents: Alice and Bob
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

    charlie = Agent(name="Charlie", proactivity_config={
        'im_threshold': 3.2,
        'system1_prob': 0.3,
        'interrupt_threshold': 4.5
    })

    # Add background knowledge to the agents
    alice.initialize_memory("I am a software engineer who likes to code. I am currently working on a project to develop a new AI model.")
    bob.initialize_memory("I am a cognitive scientist who works on understanding the human mind and modeling cognitive architectures. I am recently thinking about the nature of proactivity.")
    charlie.initialize_memory("I recently adopted a cat. I'm thinking about getting a second one.")
    
    # Add agents to the conversation
    conversation.add_participant(alice)
    conversation.add_participant(bob)
    conversation.add_participant(charlie)

    print("\n==== 🚀 Starting Conversation 🚀 ====\n")
    
    # Alice starts the conversation
    new_event = await alice.send_message("I'm recently thinking about adopting a cat. What do you think about this?", conversation)
    print(f"👤 Alice: I'm recently thinking about adopting a cat. What do you think about this?")

    # Predict the next speaker before broadcasting the event. This to determine whether the next_turn is turn-allocation or self-selection.
    turn_allocation_type = await predict_turn_taking_type(conversation)
    print(f"🎯 Turn-taking engine predicts that the turn is allocated to {turn_allocation_type}")
    
    # Broadcast the event to let all agents think
    await conversation.broadcast_event(new_event)

    # Show each agent's thoughts and their intrinsic motivation scores
    for participant in conversation.participants:
        print(f"🧠 {participant.name}'s thoughts:")
        for thought in participant.thought_reservoir.thoughts:
            print(f"  💭 {thought.content} (Intrinsic Motivation: {thought.intrinsic_motivation['score']})")
    
    # Use the turn-taking engine to deduce the next speaker and their utterance
    next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
    
    if next_speaker:
        # Send the message
        await next_speaker.send_message(utterance, conversation)
        print(f"🗣️ {next_speaker.name}: {utterance}")
    else:
        print("❌ No agent has thoughts to articulate.")
    
    print("\n==== 🏁 End of Conversation 🏁 ====\n")
    
    # Summary
    print("📋 Conversation Summary:")
    for i, event in enumerate(conversation.event_history):
        participant_name = event.participant_name
        content = event.content
        print(f"🔄 Turn {i+1}: {participant_name}: \"{content}\"")

if __name__ == "__main__":
    asyncio.run(main()) 