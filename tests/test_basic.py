import unittest
import asyncio
from thoughtful_agents.models import Agent, Conversation, Human
from thoughtful_agents.models.enums import MentalObjectType

class TestBasicFunctionality(unittest.TestCase):
    def test_imports(self):
        """Test that all necessary modules can be imported."""
        # If we got this far, imports are working
        self.assertTrue(True)
    
    def test_agent_creation(self):
        """Test that an agent can be created."""
        agent = Agent(name="TestAgent")
        self.assertEqual(agent.name, "TestAgent")
    
    def test_conversation_creation(self):
        """Test that a conversation can be created."""
        conversation = Conversation(context="Test conversation")
        self.assertEqual(conversation.context, "Test conversation")
    
    def test_memory_initialization(self):
        """Test that an agent's memory can be initialized."""
        agent = Agent(name="TestAgent")
        agent.initialize_memory("Test memory", memory_type=MentalObjectType.MEMORY_LONG_TERM)
        self.assertEqual(len(agent.memory_store.long_term_memory), 1)
        self.assertEqual(agent.memory_store.long_term_memory[0].content, "Test memory")

    def test_async_functionality(self):
        """Test that async functionality works."""
        async def async_test():
            agent = Agent(name="TestAgent")
            conversation = Conversation(context="Test conversation")
            conversation.add_participant(agent)
            
            # Initialize memory
            agent.initialize_memory("Test memory")
            
            # Generate a thought
            thoughts = await agent.generate_thoughts(conversation, num_system1=1, num_system2=0)
            
            # Check that a thought was generated
            self.assertEqual(len(thoughts), 1)
            
            return True
        
        # Run the async test
        result = asyncio.run(async_test())
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main() 