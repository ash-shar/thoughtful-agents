"""
A simple example of how to allow an AI to generate thoughts, evaluate them, and participate in a conversation, instead of using a high-level API.
In this example, we simulate a single turn in a conversation where a human speaks first, then AI responds.
"""

import asyncio
from thoughtful_agents.models import (
    Agent,
    Human,
    Conversation,
)
from thoughtful_agents.utils.turn_taking_engine import predict_turn_taking_type

async def main():
    # Create a conversation with a simple context
    conversation = Conversation(context="A chat between a human and an AI assistant.")
    
    # Create a human and an AI agent
    human = Human(name="Human")
    ai_agent = Agent(name="AI Assistant", proactivity_config={
        'im_threshold': 3.2,  # Threshold for intrinsic motivation
        'system1_prob': 0.3,  # Probability for system1 thoughts
        'interrupt_threshold': 4.5  # Threshold for interrupting
    })
    
    # Add background knowledge to the AI agent
    ai_agent.initialize_memory("I'm an AI assistant that is designed in a new way. I process information by generating both quick intuitive thoughts and deliberate reflective ones. I evaluate my thoughts based on their relevance and potential impact before deciding to express them. Unlike passive AI systems, I aim to contribute proactively when my intrinsic motivation to participate is high.")
    
    # Add participants to the conversation
    conversation.add_participant(human)
    conversation.add_participant(ai_agent)

    print("\n==== 🚀 Starting Conversation 🚀 ====\n")
    
    # Human starts the conversation
    human_msg = "I've been trying to learn more about how AI agents could proactively participate in conversations. How are you designed to do that?"
    human_event = await human.send_message(human_msg, conversation)
    print(f"👤 Human: {human_msg}")
    
    # Predict who should speak next and update the event
    print("\n-- 🔄 Predicting who should speak next --")
    predicted_speaker = await predict_turn_taking_type(conversation)
    human_event.pred_next_turn = predicted_speaker
    print(f"\n🎯 Turn-taking engine predicts that the turn is allocated to: {predicted_speaker}")
    
    # ==== Directly using Agent's thought operations ====
    print("\n==== 🧠 AI's thinking process 🧠 ====")
    
    # Step 1: Recalibrate saliency scores for the event
    print("\n-- 📊 Recalibrating saliency scores --")
    await ai_agent.recalibrate_saliency_for_event(human_event)
        
    # Step 2: Add event to short-term memory
    print("\n-- 💾 Adding event to short-term memory --")
    ai_agent.add_event_to_memory(human_event)
        
    # Step 3: Generate thoughts
    print("\n-- 💭 Generating thoughts --")
    new_thoughts = await ai_agent.generate_thoughts(conversation, num_system1=1, num_system2=2)
    
    # Print generated thoughts
    print("\n⚡ System 1 thought:")
    print(f"- {new_thoughts[0].content}")
    print("\n🔍 System 2 thoughts:")
    for i, thought in enumerate(new_thoughts[1:]):
        print(f"- {thought.content}")
    
    # Step 4: Evaluate thoughts
    print("\n-- ⚖️ Evaluating thoughts --")
    await ai_agent.evaluate_thoughts(new_thoughts, conversation)
    # Print evaluation results
    for thought in new_thoughts:
        print(f"💭 Thought: '{thought.content}' - Motivation score: {thought.intrinsic_motivation['score']}")
        print(f"🤔 Reasoning: {thought.intrinsic_motivation['reasoning']}")
    
    # Step 5: Add thoughts to reservoir
    print("\n-- 🌊 Adding thoughts to reservoir --")
    for thought in new_thoughts:
        ai_agent.thought_reservoir.add(thought)
    
    # Step 6: Select thoughts to articulate
    print("\n-- 🎯 Selecting thoughts to articulate --")
    selected_thoughts = await ai_agent.select_thoughts(new_thoughts, conversation)
    
    if selected_thoughts:
        # Find the selected thought 
        selected_thought = selected_thoughts[0]
        print(f"\n✨ Selected thought with highest motivation ({selected_thought.intrinsic_motivation['score']}): '{selected_thought.content}'")
        
        # Articulate the selected thought into natural language
        print("\n-- 🗣️ Articulating thought --")
        ai_response = await ai_agent.articulate_thought(selected_thought, conversation)
        print(f"\n🤖 AI Assistant: {ai_response}")
        
        # Add the AI's response to the conversation
        await ai_agent.send_message(ai_response, conversation)
    else:
        print("\n❌ No thoughts selected for articulation")

    print("\n==== 🏁 End of Conversation 🏁 ====\n")
    
    # Summary
    print("📋 Conversation Summary:")
    for i, event in enumerate(conversation.event_history):
        participant_name = event.participant_name
        content = event.content
        print(f"🔄 Turn {i+1}: {participant_name}: \"{content}\"")
    
    # Print all thoughts in the reservoir 
    print("\n🧠 All thoughts in AI's thought reservoir:")
    for i, thought in enumerate(ai_agent.thought_reservoir.thoughts):
        print(f"{i+1}. '{thought.content}' (Motivation: {thought.intrinsic_motivation['score']})")

if __name__ == "__main__":
    asyncio.run(main()) 