# LangGraph Getting Started

## Installation

### Basic Installation

```bash
pip install langgraph
```

### With LangChain

```bash
pip install langgraph langchain langchain-openai
```

## Environment Setup

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Your First Economics Research Graph

### Simple Linear Graph

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict

# Define state for economics research
class EconomicsResearchState(TypedDict):
    messages: list
    research_topic: str
    papers_found: list
    analysis_complete: bool

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Define nodes
def research_node(state: EconomicsResearchState):
    """Research step for economics papers."""
    response = llm.invoke(f"Research economics topic: {state['research_topic']}")
    return {"messages": state["messages"] + [response], "papers_found": ["paper1", "paper2"]}

def analyze_node(state: EconomicsResearchState):
    """Analysis step for economics research."""
    response = llm.invoke(f"Analyze economics research findings: {state['papers_found']}")
    return {"messages": state["messages"] + [response], "analysis_complete": True}

# Build graph
workflow = StateGraph(EconomicsResearchState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", END)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [],
    "research_topic": "agentic AI in behavioral economics",
    "papers_found": [],
    "analysis_complete": False
})
print(result)
```

## Basic Concepts

### 1. State

Define the state your graph will manage:

```python
from typing import TypedDict

class EconomicsResearchState(TypedDict):
    topic: str
    papers: list
    econometric_analysis: str
    summary: str
```

### 2. Nodes

Nodes are functions that process state:

```python
def research_node(state: EconomicsResearchState):
    # Process state
    papers = search_economics_papers(state["topic"])
    return {"papers": papers}
```

### 3. Conditional Routing

Route based on state:

```python
def route(state: EconomicsResearchState) -> Literal["analyze", "synthesize"]:
    if state["analysis_complete"]:
        return "synthesize"
    return "analyze"
```

## Common Pitfalls and Solutions

### Pitfall 1: State Mutation
**Problem**: Modifying state incorrectly
**Solution**: Always return new state dictionaries, don't mutate input state

### Pitfall 2: Infinite Loops
**Problem**: Graph loops forever
**Solution**: Add termination conditions in conditional edges

### Pitfall 3: State Type Mismatches
**Problem**: State structure doesn't match TypedDict
**Solution**: Ensure all nodes return compatible state structures

## Next Steps

- Explore [Economics Research Examples](scientific-research-examples.md)
- Check out [Resources](resources.md) for advanced patterns
- Review [Research Workflows](../../patterns/research-workflows.md) patterns

