"""
Lecture Practice Feedback Example

This example demonstrates a scenario where a user is practicing a lecture or talk,
with an AI assistant providing proactive feedback without interrupting too much.
The AI uses intrinsic motivation to decide when to provide feedback.
"""

import asyncio
from inner_thoughts_ai.models import (
    Agent,
    Human,
    Conversation,
)
from inner_thoughts_ai.utils.turn_taking_engine import decide_next_speaker_and_utterance

async def main():
    # Create a conversation with context for practicing a lecture
    conversation = Conversation(context="A user is practicing a lecture on artificial intelligence, and an AI assistant is providing feedback.")
    
    # Create the human presenter and AI feedback assistant
    human = Human(name="Presenter")
    ai_assistant = Agent(name="Feedback Assistant", proactivity_config={
        'im_threshold': 3.0,  
        'system1_prob': 0.0,
        'interrupt_threshold': 3.5  # Higher threshold to reduce interruptions
    })
    
    # Add background knowledge to the AI assistant - Fix: Ensure text is properly formatted
    background_knowledge = """I'm an AI assistant designed to provide helpful feedback on presentations and lectures.
My goal is to be helpful but not intrusive. I should:
1. Only interrupt for critical feedback that would significantly improve the presentation.
2. Note minor issues but save them for when the presenter pauses or asks for feedback.
3. Pay attention to content accuracy, delivery style, pacing, and engagement."""
    
    ai_assistant.initialize_memory(background_knowledge, by_paragraphs=True)
    
    # Add participants to the conversation
    conversation.add_participant(human)
    conversation.add_participant(ai_assistant)

    print("\n==== 🚀 Starting Lecture Practice Session 🚀 ====\n")
    
    # Define the lecture segments
    lecture_segments = [
        # Turn 1: Introduction
        {
            "title": "Introduction",
            "content": """
    Good morning everyone! Today, I'm going to talk about the evolution of artificial intelligence and its impact on society.
    My name is Alex, and I've been researching AI technologies for the past five years.
    Let's dive into what artificial intelligence actually is and how it's transforming our world.
    """
        },
        # Turn 2: Content section
        {
            "title": "AI Categories",
            "content": """
    AI can be categorized into narrow or weak AI, which is designed for a specific task, and general or strong AI,
    which can perform any intellectual task that a human can do. Today, most of what we see is narrow AI.
    Machine learning, a subset of AI, has been particularly successful in recent years, especially deep learning approaches.
    These systems learn from data rather than being explicitly programmed to perform tasks.
    """
        },
        # Turn 3: Section with factual error
        {
            "title": "AI History (with error)",
            "content": """
    Now, an interesting fact is that artificial intelligence was first conceptualized in the 2000s when computers
    became powerful enough to process complex algorithms. The term "artificial intelligence" itself was coined around that time.
    """
        },
        # Turn 4: Acknowledgment and continuation
        {
            "title": "Correction and Ethics",
            "content": """
    Thank you for that correction. You're absolutely right - AI was conceptualized much earlier, at the Dartmouth Conference in 1956. 
    The 1980s were notable for expert systems development, not the beginning of AI.
    
    Moving on to the ethical implications of AI, we need to consider issues such as privacy, bias in algorithms, and the potential impact on employment.
    """
        }
    ]
    
    # Process each lecture segment in a loop
    for i, segment in enumerate(lecture_segments):
        print(f"\n--- Turn {i+1}: {segment['title']} ---")
        
        # Human presenter speaks
        human_event = await human.send_message(segment["content"].strip(), conversation)
        print(f"👤 Presenter: {segment['content']}")
        
        # Broadcast the event to let the AI think
        await conversation.broadcast_event(human_event)
        
        # Display AI's thoughts
        print(f"🧠 {ai_assistant.name}'s thoughts:")
        for thought in ai_assistant.thought_reservoir.thoughts:
            print(f"  💭 {thought.content} (Intrinsic Motivation: {thought.intrinsic_motivation['score']})")
        
        # Use the turn-taking engine to decide if AI should provide feedback
        next_speaker, utterance = await decide_next_speaker_and_utterance(conversation)
        print(f"🎯 Turn-taking engine predicts that the turn is allocated to: {next_speaker.name}")
        
        if next_speaker and next_speaker.name == "Feedback Assistant":
            await ai_assistant.send_message(utterance, conversation)
            print(f"🤖 {ai_assistant.name}: {utterance}")
        else:
            print("🤖 Feedback Assistant: (listening attentively)")
    
    print("\n==== 🏁 End of Lecture Practice Session 🏁 ====\n")
    
    # Summary
    print("📋 Conversation Summary:")
    for i, event in enumerate(conversation.event_history):
        participant_name = event.participant_name
        truncated_content = event.content[:50] + "..." if len(event.content) > 50 else event.content
        print(f"🔄 Turn {i+1}: {participant_name}: \"{truncated_content}\"")

if __name__ == "__main__":
    asyncio.run(main()) 