from typing import List, Optional

class Memory:
    """Memory class as specified in the UML design."""
    pass

class MemoryStore:
    def __init__(self):
        self.long_term_memory: List[Memory] = []
        self.short_term_memory: List[Memory] = []
    
    def add(self, memory: Memory) -> None:
        """Add a memory to the appropriate store."""
        if memory.type == "long_term":
            self.long_term_memory.append(memory)
        else:
            self.short_term_memory.append(memory)
    
    def remove(self, memory: Memory) -> None:
        """Remove a memory from the store."""
        if memory.type == "long_term":
            self.long_term_memory.remove(memory)
        else:
            self.short_term_memory.remove(memory)
    
    def retrieve_top_k(self, k: int, by: str = "saliency", type: str = "long_term") -> List[Memory]:
        """Retrieve top k memories based on the specified attribute."""
        pass 