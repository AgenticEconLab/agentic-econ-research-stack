# AutoGen Getting Started

## Installation

### Basic Installation

```bash
pip install pyautogen
```

### With OpenAI Support

```bash
pip install pyautogen[openai]
```

### Full Installation with Dependencies

```bash
pip install pyautogen[openai,lmm] docker
```

## Environment Setup

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
# Optional: for other providers
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Your First Multi-Agent Economics Research System

### Simple Two-Agent Conversation

```python
import autogen

# Configure LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
    }
]

# Create assistant agent
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful economics research assistant."
)

# Create user proxy agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="What are the key applications of agentic AI in behavioral economics research?"
)
```

### Agent with Econometric Code Execution

```python
import autogen

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
    }
]

# Create assistant that can write econometric code
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful assistant that can write Python code for econometric analysis."
)

# Create user proxy that can execute code
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

# Ask for econometric code generation
user_proxy.initiate_chat(
    assistant,
    message="Write Python code to calculate correlation between GDP growth and inflation rates"
)
```

## Basic Concepts

### 1. Assistant Agent

An agent that uses LLM to respond:

```python
assistant = autogen.AssistantAgent(
    name="economist",
    llm_config={"config_list": config_list},
    system_message="You are an expert economist specializing in behavioral economics."
)
```

### 2. User Proxy Agent

An agent that can execute code and get human input:

```python
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",  # Options: NEVER, ALWAYS, TERMINATE
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)
```

### 3. Group Chat for Economics Research

Multiple agents collaborating on economics research:

```python
from autogen import GroupChat, GroupChatManager

# Create multiple economics research agents
theorist = autogen.AssistantAgent(
    name="theorist",
    llm_config={"config_list": config_list},
    system_message="You are an economic theorist expert."
)

econometrician = autogen.AssistantAgent(
    name="econometrician",
    llm_config={"config_list": config_list},
    system_message="You are an econometrician expert in statistical analysis."
)

policy_analyst = autogen.AssistantAgent(
    name="policy_analyst",
    llm_config={"config_list": config_list},
    system_message="You are a policy analyst expert."
)

# Create group chat
groupchat = GroupChat(
    agents=[theorist, econometrician, policy_analyst],
    messages=[],
    max_round=12
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Start group chat
user_proxy.initiate_chat(manager, message="Research agentic AI in behavioral economics")
```

## Common Pitfalls and Solutions

### Pitfall 1: Infinite Loops
**Problem**: Agents keep responding to each other indefinitely
**Solution**: Set `max_consecutive_auto_reply` limit and use `human_input_mode="TERMINATE"`

### Pitfall 2: Code Execution Errors
**Problem**: Generated econometric code fails to execute
**Solution**: Use Docker for isolation, or set `use_docker=False` and handle errors gracefully

### Pitfall 3: Token Costs
**Problem**: High costs from multi-agent conversations
**Solution**: Use cheaper models for some agents, limit conversation rounds

### Pitfall 4: Agent Coordination
**Problem**: Agents don't collaborate effectively on economics research
**Solution**: Use clear system messages, implement GroupChat with proper roles

## Next Steps

- Explore [Economics Research Examples](scientific-research-examples.md)
- Check out [Resources](resources.md) for advanced patterns
- Review [Multi-Agent Collaboration](../../patterns/multi-agent-collaboration.md) patterns

