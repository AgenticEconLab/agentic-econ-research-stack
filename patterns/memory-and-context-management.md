# Memory and Context Management

## Overview

Memory and context management are crucial for building agents that can maintain information across interactions, remember previous economics research conversations, and build on past economics work. This is especially important for long-running economics research tasks.

**Note**: These patterns apply to finance (financial research projects), political science (policy research), and other long-running research disciplines.

## Pattern 1: Conversation Memory for Economics Research

Remember previous messages in an economics research conversation.

### LangChain Example

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# First economics research interaction
chain.predict(input="What is behavioral economics?")

# Second interaction (remembers first)
chain.predict(input="How can agentic AI be used in behavioral economics research?")
```

## Pattern 2: Entity Memory for Economics Research

Remember specific economics entities and facts.

### Example: Economics Research Entity Memory

```python
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain

memory = ConversationEntityMemory(llm=llm)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Agent remembers economics entities
chain.predict(input="I'm researching agentic AI in behavioral economics")
chain.predict(input="What economics papers did I mention?")  # Remembers economics papers
```

## Pattern 3: Long-Term Economics Research Memory

Persistent memory for economics research projects.

### Example: Economics Research Project Memory

```python
import json
from datetime import datetime

class EconomicsResearchMemory:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.memory_file = f"{project_name}_memory.json"
        self.memories = self.load()
    
    def load(self):
        """Load economics research memories from file."""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"economics_papers": [], "findings": [], "notes": []}
    
    def save(self):
        """Save economics research memories to file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def remember_economics_paper(self, paper: dict):
        """Remember an economics paper."""
        self.memories["economics_papers"].append({
            **paper,
            "added": datetime.now().isoformat()
        })
        self.save()
    
    def remember_economics_finding(self, finding: str):
        """Remember an economics research finding."""
        self.memories["findings"].append({
            "finding": finding,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def get_relevant_economics_papers(self, query: str) -> list:
        """Get relevant economics papers for query."""
        relevant = [p for p in self.memories["economics_papers"] 
                   if query.lower() in p.get("title", "").lower() 
                   or query.lower() in p.get("abstract", "").lower()]
        return relevant

# Use in economics research
memory = EconomicsResearchMemory("agentic_ai_economics_research")
memory.remember_economics_paper({"title": "Agentic AI in Economics", "abstract": "..."})
relevant = memory.get_relevant_economics_papers("agentic")
```

**Note**: This pattern works for finance (financial research memory), political science (policy research memory), and other research disciplines.

## Best Practices

### 1. Choose Appropriate Memory Type
- **Short economics conversations**: ConversationBufferMemory
- **Long economics conversations**: ConversationSummaryMemory
- **Entity tracking**: ConversationEntityMemory
- **Economics research projects**: Custom persistent memory

### 2. Manage Context Size
- Monitor token usage
- Summarize old economics research context
- Remove irrelevant information
- Prioritize important economics research context

## Next Steps

- Review [Research Workflows](research-workflows.md) for economics workflow patterns
- Check [Frameworks](../frameworks/) for framework-specific memory implementations
- Explore [Examples](../examples/) for complete economics research implementations

