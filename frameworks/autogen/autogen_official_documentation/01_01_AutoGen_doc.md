# Agent Building Platform Knowledgebase 1: AutoGen

## AutoGen

> **📝 Optimized for Learning & Development**: This knowledge base has been cleaned and organized to remove redundant content, consolidate examples, and focus on practical implementation. All code examples are complete and runnable.

### Table of Contents

- [Overview](#overview)
- [Installation &amp; Setup](#installation--setup)
- [Core Architecture](#core-architecture)
- [AgentChat Framework](#agentchat-framework)
- [Development Patterns](#development-patterns)
- [Workbench (and MCP)](#workbench--and-mcp)
- [Security &amp; Best Practices](#security--best-practices)
- [Production Deployment](#production-deployment)

### Overview

AutoGen is a multi-agent conversation framework enabling the creation of LLM workflows with distributed, event-driven agent systems.

#### Key Components

- **AutoGen Core**: Low-level framework for custom agent systems using the Actor model
- **AgentChat**: High-level API for rapid multi-agent application development
- **AutoGen Studio**: Low-code interface for prototyping (research prototype only)

#### Key Features

- Event-driven, asynchronous messaging
- Distributed scalability across processes/machines
- Strongly-typed message protocols
- Built-in observability and debugging
- Multi-language support (Python, .NET)

### Installation & Setup

#### Requirements

- Python 3.10+
- Virtual environment recommended

#### Installation

```bash
# Virtual environment setup
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Core packages
pip install -U "autogen-agentchat"  # Includes core
pip install -U "autogen-core"       # Core only

# Model client extensions
pip install "autogen-ext[openai]"     # OpenAI/Azure OpenAI
pip install "autogen-ext[anthropic]"  # Anthropic
pip install "autogen-ext[google]"     # Google models
```

#### Basic Configuration

```python
import autogen
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Model client setup
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key automatically read from OPENAI_API_KEY env var
)

# Verify installation
print(f"AutoGen version: {autogen.__version__}")
```

### Core Architecture

#### Message-Driven Architecture

- **Agents**: Autonomous entities processing messages
- **Messages**: Structured, typed data for communication
- **Topics**: Named channels for message routing
- **Subscriptions**: Define which agents receive which messages
- **Runtime**: Manages agent lifecycle and message delivery

#### Agent Implementation

```python
from dataclasses import dataclass
from autogen_core import RoutedAgent, MessageContext, message_handler

@dataclass
class TaskMessage:
    content: str
    priority: int = 1

class MyAgent(RoutedAgent):
    def __init__(self, name: str):
        super().__init__(name)
  
    @message_handler
    async def handle_task(self, message: TaskMessage, ctx: MessageContext) -> None:
        print(f"{self.id.type} processing: {message.content}")
        # Process the task and optionally publish results
```

#### Runtime Management

```python
from autogen_core import SingleThreadedAgentRuntime, AgentId

async def setup_runtime():
    runtime = SingleThreadedAgentRuntime()
  
    # Register agent type
    await MyAgent.register(
        runtime, 
        "my_agent_type", 
        lambda: MyAgent("agent_instance")
    )
  
    # Start processing
    runtime.start()
  
    # Send message
    await runtime.send_message(
        TaskMessage("Hello World"), 
        AgentId("my_agent_type", "default")
    )
  
    # Wait for completion and cleanup
    await runtime.stop_when_idle()
    await runtime.close()
```

#### Topic & Subscription Patterns

**1. Single Scope (All agents get all messages)**

```python
from autogen_core import DefaultSubscription, DefaultTopicId

# All agents subscribe to default topic
await runtime.add_subscription(DefaultSubscription(agent_type="MyAgent"))
await runtime.publish_message(message, topic_id=DefaultTopicId())
```

**2. Multi-Tenant (Isolated by client/tenant)**

```python
# Tenant-specific topics
tenant_id = "client123"
topic_id = DefaultTopicId(source=tenant_id)
await runtime.publish_message(message, topic_id=topic_id)
```

**3. Specialized Topics (Role-based routing)**

```python
from autogen_core import TypeSubscription, TopicId

# Agents subscribe to specific roles
await runtime.add_subscription(
    TypeSubscription(topic_type="analysis", agent_type="AnalystAgent")
)
await runtime.publish_message(
    message, 
    topic_id=TopicId(type="analysis", source="default")
)
```

### Components Guide

#### Model Clients

AutoGen provides a suite of built-in model clients for using ChatCompletion API. All model clients implement the ChatCompletionClient protocol class.

Currently we support the following built-in model clients:

- `OpenAIChatCompletionClient`: for OpenAI models and models with OpenAI API compatibility (e.g., Gemini).
- `AzureOpenAIChatCompletionClient`: for Azure OpenAI models.
- `AzureAIChatCompletionClient`: for GitHub models and models hosted on Azure.
- `OllamaChatCompletionClient` (Experimental): for local models hosted on Ollama.
- `AnthropicChatCompletionClient` (Experimental): for models hosted on Anthropic.
- `SKChatCompletionAdapter`: adapter for Semantic Kernel AI connectors.

#### Log Model Calls

AutoGen uses standard Python logging module to log events like model calls and responses. The logger name is `autogen_core.EVENT_LOGGER_NAME`, and the event type is `LLMCall`.

```python
import logging

from autogen_core import EVENT_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
```

#### Call Model Client

```python
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4", temperature=0.3
)  # assuming OPENAI_API_KEY is set in the environment.

result = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
print(result)
# finish_reason='stop' content='The capital of France is Paris.' usage=RequestUsage(prompt_tokens=15, completion_tokens=8) cached=False logprobs=None thought=None
```

#### Structured Output

Structured output can be enabled by setting the `response_format` field in `OpenAIChatCompletionClient` and `AzureOpenAIChatCompletionClient` to as a Pydantic BaseModel class.

```python
from typing import Literal
from pydantic import BaseModel

class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=AgentResponse,
)

messages = [
    UserMessage(content="I am happy.", source="user"),
]
response = await model_client.create(messages=messages)
assert isinstance(response.content, str)
parsed_response = AgentResponse.model_validate_json(response.content)
print(parsed_response.thoughts)
print(parsed_response.response)
```

### Model Context

A model context supports storage and retrieval of Chat Completion messages. It is always used together with a model client to generate LLM-based responses.

For example, `BufferedChatCompletionContext` is a most-recent-used (MRU) context that stores the most recent `buffer_size` number of messages. This is useful to avoid context overflow in many LLMs.

```python
from autogen_core.model_context import BufferedChatCompletionContext

# Initialize with buffer size
model_context = BufferedChatCompletionContext(buffer_size=5)

# Add and retrieve messages
await model_context.add_message(user_message)
messages = await model_context.get_messages()
```

### Tools

Tools are code that can be executed by an agent to perform actions. A tool can be a simple function such as a calculator, or an API call to a third-party service such as stock price lookup or weather forecast.

#### Built-in Tools

One of the built-in tools is the `PythonCodeExecutionTool`, which allows agents to execute Python code snippets.

```python
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

# Create the tool.
code_executor = DockerCommandLineCodeExecutor()
await code_executor.start()
code_execution_tool = PythonCodeExecutionTool(code_executor)
cancellation_token = CancellationToken()

# Use the tool directly without an agent.
code = "print('Hello, world!')"
result = await code_execution_tool.run_json({"code": code}, cancellation_token)
print(code_execution_tool.return_value_as_string(result))
# Hello, world!
```

#### Custom Function Tools

```python
import random
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    return random.uniform(10, 200)

# Create a function tool.
stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

# Run the tool.
cancellation_token = CancellationToken()
result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token)
print(stock_price_tool.return_value_as_string(result))
```

### Workbench (and MCP)

A Workbench provides a collection of tools that share state and resources. Different from Tool, which provides an interface to a single tool, a workbench provides an interface to call different tools and receive results as the same types.

#### Using MCP Workbench

`McpWorkbench` is an implementation of `Workbench` that enables an agent to call tools on an MCP server. For example, you can use `McpWorkbench` to call tools on the `Playwright` MCP server to browse the web.

```python
from autogen_core import AgentId, Message
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.mcp.mcp_workbench import McpWorkbench
from autogen_ext.mcp.playwright import PlaywrightMcpServer

playwright_server_params = PlaywrightMcpServer()

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
runtime = SingleThreadedAgentRuntime()

async with McpWorkbench(playwright_server_params) as workbench:
    await WorkbenchAgent.register(
        runtime,
        "web_agent",
        lambda: WorkbenchAgent(
            model_client=model_client,
            workbench=workbench,
        ),
    )

    runtime.start()
  
    web_agent = AgentId("web_agent", "default")
    response = await runtime.send_message(
        Message("Use Bing to find out the address of Microsoft Building 99"), web_agent
    )
    print(response.content)
  
    await runtime.stop()

await model_client.close()
```

### AgentChat Framework

#### Quick Start

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def basic_example():
    # Create model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
  
    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant."
    )
  
    # Create team and run
    team = RoundRobinGroupChat([assistant], max_turns=1)
  
    # Stream interaction
    async for message in team.run_stream(task="Explain quantum computing"):
        print(f"{message.source}: {message.content}")
```

#### Tool Integration

```python
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 22°C and sunny."

# Tool-enabled agent
weather_agent = AssistantAgent(
    name="weather_agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    tools=[get_weather]
)
```

### Development Patterns

#### 1. Reflection Pattern

Iterative improvement through review cycles.

```python
from dataclasses import dataclass
from autogen_core import RoutedAgent, MessageContext, message_handler, TopicId

@dataclass
class CodeTask:
    task: str
    max_iterations: int = 3

@dataclass
class CodeReview:
    code: str
    approved: bool
    feedback: str

class CoderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__("Coder agent")
        self._model_client = model_client
  
    @message_handler
    async def handle_task(self, message: CodeTask, ctx: MessageContext):
        code = await self._generate_code(message.task)
        await self.publish_message(
            CodeReview(code=code, approved=False, feedback=""),
            topic_id=TopicId("review", self.id.key)
        )
```

#### 2. Sequential Workflow

Deterministic processing pipeline.

```python
from dataclasses import dataclass
from autogen_core import type_subscription

@dataclass
class DataMessage:
    data: str

@dataclass  
class ProcessedDataMessage:
    data: str

@type_subscription(topic_type="data_processor")
class DataProcessorAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Data processor")
  
    @message_handler
    async def process_data(self, message: DataMessage, ctx: MessageContext):
        processed_data = await self._process(message.data)
        await self.publish_message(
            ProcessedDataMessage(data=processed_data),
            topic_id=TopicId("validator", self.id.key)
        )
```

#### 3. Multi-Agent Debate

Collaborative problem-solving with multiple perspectives.

#### 4. Mixture of Agents

Hierarchical agent orchestration for complex tasks.

### Security & Best Practices

#### Security Considerations

**Input Validation**

```python
from pydantic import BaseModel, validator

class SecureMessage(BaseModel):
    content: str
    user_id: str
  
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10000:
            raise ValueError('Content too long')
        return v
```

**Authentication & Authorization**

```python
class SecureAgent(RoutedAgent):
    @message_handler
    async def handle_secure_message(self, message: SecureMessage, ctx: MessageContext):
        if not await self._verify_permissions(message.user_id, ctx):
            raise PermissionError("Unauthorized access")
        await self._process_message(message)
```

**Secrets Management**

```python
import os

def get_secure_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to Key Vault or other secure storage
        raise ValueError("API key not found")
    return OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
```

#### Development Best Practices

**Error Handling**

```python
class RobustAgent(RoutedAgent):
    @message_handler
    async def handle_message(self, message: TaskMessage, ctx: MessageContext):
        try:
            result = await self._process_task(message)
            await self.publish_message(result, topic_id=TopicId("results", self.id.key))
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            await self.publish_message(
                ErrorMessage(error=str(e), original_task=message),
                topic_id=TopicId("errors", self.id.key)
            )
```

**Resource Management**

```python
class ResourceManagedAgent(RoutedAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
  
    @message_handler
    async def handle_task(self, message: TaskMessage, ctx: MessageContext):
        async with self._semaphore:
            async with asyncio.timeout(30):  # 30-second timeout
                result = await self._process_task(message)
                return result
```

### Production Deployment

#### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Security: non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Environment Configuration

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    azure_endpoint: str = None
    max_concurrent_requests: int = 100
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379"
  
    class Config:
        env_file = ".env"

settings = Settings()
```

### Quick Reference

#### Essential Imports

```python
# Core
from autogen_core import (
    RoutedAgent, MessageContext, message_handler,
    SingleThreadedAgentRuntime, AgentId, TopicId,
    DefaultSubscription, TypeSubscription
)

# AgentChat
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, UserMessage

# Model Clients
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient
```

#### Common Patterns

```python
# Basic agent setup
runtime = SingleThreadedAgentRuntime()
await MyAgent.register(runtime, "agent_type", lambda: MyAgent())
runtime.start()

# Send message
await runtime.send_message(message, AgentId("agent_type", "default"))

# Publish to topic
await agent.publish_message(message, TopicId("topic_type", "source"))

# Cleanup
await runtime.stop_when_idle()
await runtime.close()
```

---




## LangChain

## Microsoft Agent Framework

## IBM Watson Assistant
