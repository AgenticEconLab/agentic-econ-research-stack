# LangChain Basics: Comprehensive Guide

> **Last Updated**: December 2025
> **Sources**: [LangChain Docs](https://docs.langchain.com/oss/python/langchain/), [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/), [Deep Agents Docs](https://docs.langchain.com/oss/python/deepagents/)

---

## Table of Contents

1. [Overview of the LangChain Ecosystem](#1-overview-of-the-langchain-ecosystem)
2. [LangChain Core Concepts](#2-langchain-core-concepts)
3. [LangChain Expression Language (LCEL)](#3-langchain-expression-language-lcel)
4. [Agents and Tools](#4-agents-and-tools)
5. [LangGraph: Stateful Workflows](#5-langgraph-stateful-workflows)
6. [Deep Agents: Complex Task Handling](#6-deep-agents-complex-task-handling)
7. [Code Examples](#7-code-examples)
8. [When to Use What](#8-when-to-use-what)
9. [Learning Resources](#9-learning-resources)

---

## 1. Overview of the LangChain Ecosystem

### What is LangChain?

LangChain is an **open-source framework** designed to facilitate agent and application development powered by large language models (LLMs). The platform emphasizes ease of use combined with flexibility, enabling developers to build sophisticated AI systems with minimal initial code.

> "LangChain is an open source framework with a pre-built agent architecture and integrations for any model or tool — so you can build agents that adapt as fast as the ecosystem evolves"

### Ecosystem Components

The LangChain ecosystem consists of three main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     DEEP AGENTS                             │
│        (Complex multi-step tasks, planning, subagents)      │
├─────────────────────────────────────────────────────────────┤
│                      LANGCHAIN                              │
│     (High-level agents, models, tools, 1000+ integrations)  │
├─────────────────────────────────────────────────────────────┤
│                      LANGGRAPH                              │
│    (Low-level orchestration, state management, workflows)   │
└─────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Best For |
|-----------|---------|----------|
| **LangChain** | High-level agent framework | Quick prototyping, standard agents |
| **LangGraph** | Low-level orchestration runtime | Complex workflows, stateful agents |
| **Deep Agents** | Advanced multi-step task handling | Planning, file systems, subagents |

### Key Characteristics

- **Ease of Entry**: Connect to major model providers (OpenAI, Anthropic, Google) with fewer than 10 lines of code
- **Flexibility**: 1000+ integrations for models, tools, and databases
- **No Vendor Lock-in**: Swap models and tools without rewriting your application
- **Production Ready**: Built for deployment with durability and scalability

---

## 2. LangChain Core Concepts

### Building Blocks

LangChain organizes functionality around several essential components:

#### Models

Standardized interfaces for interacting with various LLM providers:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Anthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Google
llm = ChatGoogleGenerativeAI(model="gemini-pro")
```

#### Messages

Structured communication units within the system:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?"),
    AIMessage(content="LangChain is a framework for building LLM applications...")
]
```

#### Prompts

Templates for structuring LLM inputs:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert."),
    ("user", "{question}")
])

# Format the prompt
formatted = prompt.format(role="Python", question="What is LCEL?")
```

#### Output Parsers

Type-safe response formatting:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Answer(BaseModel):
    response: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score 0-1")

parser = PydanticOutputParser(pydantic_object=Answer)
```

#### Memory Systems

Both short-term (conversation context) and long-term storage:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, {"output": "Hello!"})
```

---

## 3. LangChain Expression Language (LCEL)

### What is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to chain LangChain components. It introduces a powerful concept: connecting components using the **pipe operator (`|`)**.

### Core Syntax

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Chain with pipe operator
chain = prompt | llm | parser

# Invoke the chain
result = chain.invoke({"topic": "LangChain"})
```

### LCEL Benefits

| Feature | Description |
|---------|-------------|
| **Readability** | Clean, intuitive syntax with pipe operator |
| **Composability** | Easy to combine components |
| **Streaming** | Built-in support for streaming outputs |
| **Async Support** | Native async/await compatibility |
| **Batching** | Efficient batch processing |

### When to Use LCEL

- **Best for**: Simple orchestration tasks without complex branching
- **Not ideal for**: Dynamic agents, decision trees, stateful conversations
- **Alternative**: Use LangGraph for complex workflows (LCEL can still be used within LangGraph nodes)

---

## 4. Agents and Tools

### What is an Agent?

An **agent** is a language model that decides on a sequence of actions to execute. Unlike chains where the sequence is hard-coded, agents use the LLM to determine which actions to take and in what order.

### Agent Architecture Evolution

```
┌─────────────────────────────────────────────────────────────┐
│                    MODERN (Recommended)                     │
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │ create_react_   │    │ LangGraph Custom Agents     │    │
│  │ agent()         │    │ (StateGraph + Nodes)        │    │
│  └─────────────────┘    └─────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                      LEGACY (Deprecated)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ AgentExecutor, initialize_agent, load_tools         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Defining Tools with @tool Decorator

The `@tool` decorator is the primary way to define custom tools:

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for relevant information.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    # Implementation
    return f"Results for: {query}"

@tool("calculator", return_direct=True)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    return str(eval(expression))
```

### Tool Decorator Features

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Custom tool name | `@tool("my_tool")` |
| `return_direct` | Return output directly without LLM | `@tool(return_direct=True)` |
| `args_schema` | Pydantic model for arguments | `@tool(args_schema=MyArgs)` |

### Creating a ReAct Agent

The **ReAct** (Reasoning + Acting) pattern is the standard agent architecture:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny."

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

tools = [get_weather, search_web]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Get ReAct prompt from hub
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Run agent
result = agent_executor.invoke({
    "input": "What's the weather in San Francisco?"
})
```

### Modern LangGraph Agent (Preferred)

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather, search_web]

# Create agent with LangGraph
agent = create_react_agent(llm, tools)

# Invoke
result = agent.invoke({
    "messages": [("user", "What's the weather in NYC?")]
})
```

---

## 5. LangGraph: Stateful Workflows

### What is LangGraph?

**LangGraph** is a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents. It's inspired by Google's Pregel and Apache Beam, with an interface influenced by NetworkX.

> "LangGraph is a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents."

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        StateGraph                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                       STATE                          │   │
│  │    (Shared data structure - snapshot of app)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  NODE A  │─────▶│  NODE B  │─────▶│  NODE C  │         │
│  │(Function)│      │(Function)│      │(Function)│         │
│  └──────────┘      └──────────┘      └──────────┘         │
│       │                  │                  │              │
│       └──────────────────┼──────────────────┘              │
│                     EDGES                                  │
│            (Direct or Conditional)                         │
└─────────────────────────────────────────────────────────────┘
```

#### State

A shared data structure representing the current snapshot of your application:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    step_count: int
```

#### Nodes

Functions that encode agent logic, receiving state and returning updated state:

```python
def process_node(state: AgentState) -> AgentState:
    """Process the current state and return updates."""
    messages = state["messages"]
    # Process logic here
    return {"messages": messages, "step_count": state["step_count"] + 1}
```

#### Edges

Determine which node to execute next:

- **Direct Edges**: Fixed transitions from one node to another
- **Conditional Edges**: Dynamic routing based on state

```python
from langgraph.graph import StateGraph, END

# Direct edge
graph.add_edge("node_a", "node_b")

# Conditional edge
def should_continue(state):
    if state["step_count"] > 5:
        return END
    return "continue_node"

graph.add_conditional_edges("check_node", should_continue)
```

### Building a LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define Nodes
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build Graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 4. Compile
app = graph.compile()

# 5. Invoke
result = app.invoke({"messages": [("user", "Hello!")]})
```

### Key LangGraph Capabilities

| Capability | Description |
|------------|-------------|
| **Durable Execution** | Agents persist through failures, resume from checkpoints |
| **Human-in-the-Loop** | Inspect and modify agent state at any point |
| **Memory** | Working memory + persistent cross-session memory |
| **Streaming** | Real-time output streaming |
| **LangSmith Integration** | Tracing, state transitions, detailed metrics |

### LangGraph vs LCEL

| Feature | LCEL | LangGraph |
|---------|------|-----------|
| Complexity | Simple chains | Complex workflows |
| State Management | Limited | Full state control |
| Branching | Sequential | Conditional routing |
| Cycles | Not supported | Fully supported |
| Human-in-the-Loop | Manual | Built-in |

---

## 6. Deep Agents: Complex Task Handling

### What is Deep Agents?

**Deep Agents** is a standalone Python library built on LangGraph that enables construction of sophisticated agents capable of handling complex, multi-step tasks. Inspired by applications like Claude Code, Deep Research, and Manus.

> "Build agents that can plan, use subagents, and leverage file systems for complex tasks."

### When to Use Deep Agents

| Use Case | Deep Agents | Alternative |
|----------|-------------|-------------|
| Complex multi-step tasks | Yes | - |
| Large context management | Yes | - |
| Work delegation to subagents | Yes | - |
| Persistent memory across conversations | Yes | - |
| Simple single-step tasks | No | `create_agent` |
| Standard workflows | No | Custom LangGraph |

### Key Capabilities

#### 1. Planning & Task Decomposition

Built-in `write_todos` tool for breaking down complex tasks:

```python
# Agent can create and track todos
agent.write_todos([
    {"task": "Research topic", "status": "pending"},
    {"task": "Write outline", "status": "pending"},
    {"task": "Draft content", "status": "pending"}
])
```

#### 2. Context Management (File System Tools)

Prevent context window overflow by offloading to memory:

```python
# Available tools
ls()          # List directory contents
read_file()   # Read file content
write_file()  # Write content to file
edit_file()   # Edit existing file
```

#### 3. Subagent Architecture

Spawn specialized subagents for isolated, focused work:

```python
# Task tool spawns subagents
agent.task(
    prompt="Research quantum computing advances",
    subagent_type="researcher"
)
```

#### 4. Long-term Memory

Integration with LangGraph's Store for persistent memory:

```python
from langgraph.store import InMemoryStore

store = InMemoryStore()
# Memory persists across threads and conversations
```

---

## 7. Code Examples

### Example 1: Simple LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer: {question}"
)
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"question": "What is Python?"})
print(response)
```

### Example 2: Agent with Custom Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [multiply, add]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is 5 multiplied by 3, then add 10?"})
print(result["output"])
```

### Example 3: LangGraph Stateful Workflow

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }

def should_continue(state: State) -> str:
    if state["iteration"] >= 3:
        return END
    return "process"

# Build graph
graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_continue)

app = graph.compile()

result = app.invoke({
    "messages": [("user", "Count to 3")],
    "iteration": 0
})
```

### Example 4: RAG with LCEL

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["LangChain is a framework for LLM apps"],
    embeddings
)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
Answer based on context:
Context: {context}
Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is LangChain?")
```

---

## 8. When to Use What

### Decision Tree

```
                    ┌─────────────────────┐
                    │   What's your need? │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐    ┌───────▼───────┐   ┌──────▼──────┐
    │   Simple    │    │   Complex     │   │  Multi-step │
    │   Chain     │    │   Workflow    │   │   Planning  │
    └──────┬──────┘    └───────┬───────┘   └──────┬──────┘
           │                   │                   │
    ┌──────▼──────┐    ┌───────▼───────┐   ┌──────▼──────┐
    │    LCEL     │    │   LangGraph   │   │ Deep Agents │
    └─────────────┘    └───────────────┘   └─────────────┘
```

### Quick Reference

| Scenario | Solution | Reason |
|----------|----------|--------|
| Simple Q&A | LCEL Chain | No state needed |
| Tool-using agent | `create_react_agent` | Standard pattern |
| Multi-step workflow | LangGraph StateGraph | State management |
| Human approval needed | LangGraph HITL | Built-in support |
| Complex planning | Deep Agents | Todo + subagents |
| RAG application | LCEL + Retriever | Composable |
| Production deployment | LangGraph + LangSmith | Durability + observability |

---

## 9. Learning Resources

### Official Documentation

- [LangChain Overview](https://docs.langchain.com/oss/python/langchain/)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/)
- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/)
- [LangChain Conceptual Guide](https://python.langchain.com/v0.2/docs/concepts/)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

### Tutorials

- [Real Python: LangGraph Build Stateful AI Agents](https://realpython.com/langgraph-python/) - Comprehensive 58-minute tutorial
- [Real Python: Build an LLM RAG Chatbot](https://realpython.com/build-llm-rag-chatbot-with-langchain/)
- [DataCamp: LangGraph Agents Tutorial](https://www.datacamp.com/tutorial/langgraph-agents)
- [Codecademy: Building AI Workflows](https://www.codecademy.com/article/building-ai-workflow-with-langgraph)
- [DeepLearning.AI: Functions, Tools and Agents](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)

### GitHub Repositories

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)

### Installation

```bash
# Core LangChain
pip install langchain langchain-core langchain-community

# Model providers
pip install langchain-openai langchain-anthropic langchain-google-genai

# LangGraph
pip install langgraph

# Vector stores
pip install faiss-cpu chromadb
```

---

## Summary

The LangChain ecosystem provides a comprehensive toolkit for building LLM-powered applications:

1. **LangChain**: High-level framework with 1000+ integrations for quick development
2. **LCEL**: Declarative chaining with the pipe operator for simple workflows
3. **Agents**: Autonomous decision-making with the ReAct pattern and custom tools
4. **LangGraph**: Low-level orchestration for complex, stateful workflows
5. **Deep Agents**: Advanced multi-step planning with subagent delegation

Choose the right tool for your use case:
- **Simple chains** → LCEL
- **Standard agents** → `create_react_agent`
- **Complex workflows** → LangGraph
- **Multi-step planning** → Deep Agents

---

*This documentation was compiled from official LangChain sources and community resources. For the most up-to-date information, always refer to the [official documentation](https://docs.langchain.com/).*
