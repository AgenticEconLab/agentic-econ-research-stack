# AutoGen Knowledge Base - Developer Guide

> **📝 Optimized for Learning & Development**: This knowledge base has been cleaned and organized to remove redundant content, consolidate examples, and focus on practical implementation. All code examples are complete and runnable.

## Table of Contents

- [Overview](#overview)
- [Installation &amp; Setup](#installation--setup)
- [Core Architecture](#core-architecture)
- [AgentChat Framework](#agentchat-framework)
- [Development Patterns](#development-patterns)
- [Workbench (and MCP)](#workbench--and-mcp)
- [Security &amp; Best Practices](#security--best-practices)
- [Production Deployment](#production-deployment)

## Overview

AutoGen is a multi-agent conversation framework enabling the creation of LLM workflows with distributed, event-driven agent systems.

### Key Components

- **AutoGen Core**: Low-level framework for custom agent systems using the Actor model
- **AgentChat**: High-level API for rapid multi-agent application development
- **AutoGen Studio**: Low-code interface for prototyping (research prototype only)

### Key Features

- Event-driven, asynchronous messaging
- Distributed scalability across processes/machines
- Strongly-typed message protocols
- Built-in observability and debugging
- Multi-language support (Python, .NET)

## Installation & Setup

### Requirements

- Python 3.10+
- Virtual environment recommended

### Installation

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

### Basic Configuration

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

## Core Architecture

### Message-Driven Architecture

- **Agents**: Autonomous entities processing messages
- **Messages**: Structured, typed data for communication
- **Topics**: Named channels for message routing
- **Subscriptions**: Define which agents receive which messages
- **Runtime**: Manages agent lifecycle and message delivery

### Agent Implementation

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

### Runtime Management

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

### Topic & Subscription Patterns

#### 1. Single Scope (All agents get all messages)

```python
from autogen_core import DefaultSubscription, DefaultTopicId

# All agents subscribe to default topic
await runtime.add_subscription(DefaultSubscription(agent_type="MyAgent"))
await runtime.publish_message(message, topic_id=DefaultTopicId())
```

#### 2. Multi-Tenant (Isolated by client/tenant)

```python
# Tenant-specific topics
tenant_id = "client123"
topic_id = DefaultTopicId(source=tenant_id)
await runtime.publish_message(message, topic_id=topic_id)
```

#### 3. Specialized Topics (Role-based routing)

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

## Components Guide

### Model Clients

AutoGen provides a suite of built-in model clients for using ChatCompletion API. All model clients implement the ChatCompletionClient protocol class.

Currently we support the following built-in model clients:

- `OpenAIChatCompletionClient`: for OpenAI models and models with OpenAI API compatibility (e.g., Gemini).
- `AzureOpenAIChatCompletionClient`: for Azure OpenAI models.
- `AzureAIChatCompletionClient`: for GitHub models and models hosted on Azure.
- `OllamaChatCompletionClient` (Experimental): for local models hosted on Ollama.
- `AnthropicChatCompletionClient` (Experimental): for models hosted on Anthropic.
- `SKChatCompletionAdapter`: adapter for Semantic Kernel AI connectors.

For more information on how to use these model clients, please refer to the documentation of each client.

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

To call a model client, you can use the `create()` method. This example uses the `OpenAIChatCompletionClient` to call an OpenAI model.

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

#### Streaming Tokens

You can use the `create_stream()` method to create a chat completion request with streaming token chunks.

```python
from autogen_core.models import CreateResult, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")  # assuming OPENAI_API_KEY is set in the environment.

messages = [
    UserMessage(content="Write a very short story about a dragon.", source="user"),
]

# Create a stream.
stream = model_client.create_stream(messages=messages)

# Iterate over the stream and print the responses.
print("Streamed responses:")
async for chunk in stream:  # type: ignore
    if isinstance(chunk, str):
        # The chunk is a string.
        print(chunk, flush=True, end="")
    else:
        # The final chunk is a CreateResult object.
        assert isinstance(chunk, CreateResult) and isinstance(chunk.content, str)
        # The last response is a CreateResult object with the complete message.
        print("\n\n------------\n")
        print("The complete response:", flush=True)
        print(chunk.content, flush=True)
# Streamed responses:
# In the heart of an ancient forest, beneath the shadow of snow-capped peaks, a dragon named Elara lived secretly for centuries. Elara was unlike any dragon from the old tales; her scales shimmered with a deep emerald hue, each scale engraved with symbols of lost wisdom. The villagers in the nearby valley spoke of mysterious lights dancing across the night sky, but none dared venture close enough to solve the enigma.

# One cold winter's eve, a young girl named Lira, brimming with curiosity and armed with the innocence of youth, wandered into Elara’s domain. Instead of fire and fury, she found warmth and a gentle gaze. The dragon shared stories of a world long forgotten and in return, Lira gifted her simple stories of human life, rich in laughter and scent of earth.

# From that night on, the villagers noticed subtle changes—the crops grew taller, and the air seemed sweeter. Elara had infused the valley with ancient magic, a guardian of balance, watching quietly as her new friend thrived under the stars. And so, Lira and Elara’s bond marked the beginning of a timeless friendship that spun tales of hope whispered through the leaves of the ever-verdant forest.

# ------------

# The complete response:
# In the heart of an ancient forest, beneath the shadow of snow-capped peaks, a dragon named Elara lived secretly for centuries. Elara was unlike any dragon from the old tales; her scales shimmered with a deep emerald hue, each scale engraved with symbols of lost wisdom. The villagers in the nearby valley spoke of mysterious lights dancing across the night sky, but none dared venture close enough to solve the enigma.

# One cold winter's eve, a young girl named Lira, brimming with curiosity and armed with the innocence of youth, wandered into Elara’s domain. Instead of fire and fury, she found warmth and a gentle gaze. The dragon shared stories of a world long forgotten and in return, Lira gifted her simple stories of human life, rich in laughter and scent of earth.

# From that night on, the villagers noticed subtle changes—the crops grew taller, and the air seemed sweeter. Elara had infused the valley with ancient magic, a guardian of balance, watching quietly as her new friend thrived under the stars. And so, Lira and Elara’s bond marked the beginning of a timeless friendship that spun tales of hope whispered through the leaves of the ever-verdant forest.


# ------------

# The token usage was:
# RequestUsage(prompt_tokens=0, completion_tokens=0)
# Note

# The last response in the streaming response is always the final response of the type CreateResult.

# Note

# The default usage response is to return zero values. To enable usage, see create_stream() for more details.
```

#### Structured Output

Structured output can be enabled by setting the `response_format` field in `OpenAIChatCompletionClient` and `AzureOpenAIChatCompletionClient` to as a Pydantic BaseModel class.

Note

Structured output is only available for models that support it. It also requires the model client to support structured output as well. Currently, the `OpenAIChatCompletionClient` and `AzureOpenAIChatCompletionClient` support structured output.

```python
from typing import Literal

from pydantic import BaseModel


# The response format for the agent as a Pydantic base model.
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


# Create an agent that uses the OpenAI GPT-4o model with the custom response format.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=AgentResponse,  # type: ignore
)

# Send a message list to the model and await the response.
messages = [
    UserMessage(content="I am happy.", source="user"),
]
response = await model_client.create(messages=messages)
assert isinstance(response.content, str)
parsed_response = AgentResponse.model_validate_json(response.content)
print(parsed_response.thoughts)
print(parsed_response.response)

# Close the connection to the model client.
await model_client.close()
# I'm glad to hear that you're feeling happy! It's such a great emotion that can brighten your whole day. Is there anything in particular that's bringing you joy today? 😊
# happy
```

You also use the `extra_create_args` parameter in the `create()` method to set the `response_format` field so that the structured output can be configured for each request.

#### Caching Model Responses

`autogen_ext` implements `ChatCompletionCache` that can wrap any `ChatCompletionClient`. Using this wrapper avoids incurring token usage when querying the underlying client with the same prompt multiple times.

`ChatCompletionCache` uses a `CacheStore` protocol. We have implemented some useful variants of `CacheStore` including `DiskCacheStore` and `RedisStore`.

Here’s an example of using diskcache for local caching:

```bash
# pip install -U "autogen-ext[openai, diskcache]"
```

```python
import asyncio
import tempfile

from autogen_core.models import UserMessage
from autogen_ext.cache_store.diskcache import DiskCacheStore
from autogen_ext.models.cache import CHAT_CACHE_VALUE_TYPE, ChatCompletionCache
from autogen_ext.models.openai import OpenAIChatCompletionClient
from diskcache import Cache


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize the original client
        openai_model_client = OpenAIChatCompletionClient(model="gpt-4o")

        # Then initialize the CacheStore, in this case with diskcache.Cache.
        # You can also use redis like:
        # from autogen_ext.cache_store.redis import RedisStore
        # import redis
        # redis_instance = redis.Redis()
        # cache_store = RedisCacheStore[CHAT_CACHE_VALUE_TYPE](redis_instance)
        cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(tmpdirname))
        cache_client = ChatCompletionCache(openai_model_client, cache_store)

        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)  # Should print response from OpenAI
        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)  # Should print cached response

        await openai_model_client.close()
        await cache_client.close()


asyncio.run(main())
# True
```

Inspecting `cached_client.total_usage()` (or `model_client.total_usage()`) before and after a cached response should yield idential counts.

Note that the caching is sensitive to the exact arguments provided to `cached_client.create` or `cached_client.create_stream`, so changing tools or `json_output` arguments might lead to a cache miss.

#### Build an Agent with a Model Client

Let’s create a simple AI agent that can respond to messages using the ChatCompletion API.

```python
from dataclasses import dataclass

from autogen_core import MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class Message:
    content: str


class SimpleAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        # Return with the model's response.
        assert isinstance(response.content, str)
        return Message(content=response.content)
```

The `SimpleAgent` class is a subclass of the `autogen_core.RoutedAgent` class for the convenience of automatically routing messages to the appropriate handlers. It has a single handler, `handle_user_message`, which handles message from the user. It uses the `ChatCompletionClient` to generate a response to the message. It then returns the response to the user, following the direct communication model.

Note

The `cancellation_token` of the type `autogen_core.CancellationToken` is used to cancel asynchronous operations. It is linked to async calls inside the message handlers and can be used by the caller to cancel the handlers.

```python
from autogen_core import AgentId

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY set in the environment.
)

runtime = SingleThreadedAgentRuntime()
await SimpleAgent.register(
    runtime,
    "simple_agent",
    lambda: SimpleAgent(model_client=model_client),
)
# Start the runtime processing messages.
runtime.start()
# Send a message to the agent and get the response.
message = Message("Hello, what are some fun things to do in Seattle?")
response = await runtime.send_message(message, AgentId("simple_agent", "default"))
print(response.content)
# Stop the runtime processing messages.
await runtime.stop()
await model_client.close()
# Seattle is a vibrant city with a wide range of activities and attractions. Here are some fun things to do in Seattle:

# 1. **Space Needle**: Visit this iconic observation tower for stunning views of the city and surrounding mountains.

# 2. **Pike Place Market**: Explore this historic market where you can see the famous fish toss, buy local produce, and find unique crafts and eateries.

# 3. **Museum of Pop Culture (MoPOP)**: Dive into the world of contemporary culture, music, and science fiction at this interactive museum.

# 4. **Chihuly Garden and Glass**: Marvel at the beautiful glass art installations by artist Dale Chihuly, located right next to the Space Needle.

# 5. **Seattle Aquarium**: Discover the diverse marine life of the Pacific Northwest at this engaging aquarium.

# 6. **Seattle Art Museum**: Explore a vast collection of art from around the world, including contemporary and indigenous art.

# 7. **Kerry Park**: For one of the best views of the Seattle skyline, head to this small park on Queen Anne Hill.

# 8. **Ballard Locks**: Watch boats pass through the locks and observe the salmon ladder to see salmon migrating.

# 9. **Ferry to Bainbridge Island**: Take a scenic ferry ride across Puget Sound to enjoy charming shops, restaurants, and beautiful natural scenery.

# 10. **Olympic Sculpture Park**: Stroll through this outdoor park with large-scale sculptures and stunning views of the waterfront and mountains.

# 11. **Underground Tour**: Discover Seattle's history on this quirky tour of the city's underground passageways in Pioneer Square.

# 12. **Seattle Waterfront**: Enjoy the shops, restaurants, and attractions along the waterfront, including the Seattle Great Wheel and the aquarium.

# 13. **Discovery Park**: Explore the largest green space in Seattle, featuring trails, beaches, and views of Puget Sound.

# 14. **Food Tours**: Try out Seattle’s diverse culinary scene, including fresh seafood, international cuisines, and coffee culture (don’t miss the original Starbucks!).

# 15. **Attend a Sports Game**: Catch a Seahawks (NFL), Mariners (MLB), or Sounders (MLS) game for a lively local experience.

# Whether you're interested in culture, nature, food, or history, Seattle has something for everyone to enjoy!
```

The `SimpleAgent` always responds with a fresh context that contains only the system message and the latest user’s message. We can use model context classes from `autogen_core.model_context` to make the agent “remember” previous conversations. See the Model Context page for more details.

#### API Keys From Environment Variables

In the examples above, we show that you can provide the API key through the `api_key` argument. Importantly, the OpenAI and Azure OpenAI clients use the `openai` package, which will automatically read an api key from the environment variable if one is not provided.

For OpenAI, you can set the `OPENAI_API_KEY` environment variable.

For Azure OpenAI, you can set the `AZURE_OPENAI_API_KEY` environment variable.

In addition, for Gemini (Beta), you can set the `GEMINI_API_KEY` environment variable.

This is a good practice to explore, as it avoids including sensitive api keys in your code.

### Model Context

A model context supports storage and retrieval of Chat Completion messages. It is always used together with a model client to generate LLM-based responses.

For example, `BufferedChatCompletionContext` is a most-recent-used (MRU) context that stores the most recent `buffer_size` number of messages. This is useful to avoid context overflow in many LLMs.

Let’s see an example that uses `BufferedChatCompletionContext`.

```python
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
@dataclass
class Message:
    content: str
class SimpleAgentWithContext(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        # Add message to model context.
        await self._model_context.add_message(user_message)
        # Generate a response.
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        # Return with the model's response.
        assert isinstance(response.content, str)
        # Add message to model context.
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))
        return Message(content=response.content)
```

Now let’s try to ask follow up questions after the first one.

```python
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY set in the environment.
)

runtime = SingleThreadedAgentRuntime()
await SimpleAgentWithContext.register(
    runtime,
    "simple_agent_context",
    lambda: SimpleAgentWithContext(model_client=model_client),
)
# Start the runtime processing messages.
runtime.start()
agent_id = AgentId("simple_agent_context", "default")

# First question.
message = Message("Hello, what are some fun things to do in Seattle?")
print(f"Question: {message.content}")
response = await runtime.send_message(message, agent_id)
print(f"Response: {response.content}")
print("-----")

# Second question.
message = Message("What was the first thing you mentioned?")
print(f"Question: {message.content}")
response = await runtime.send_message(message, agent_id)
print(f"Response: {response.content}")

# Stop the runtime processing messages.
await runtime.stop()
await model_client.close()
# Question: Hello, what are some fun things to do in Seattle?
# Response: Seattle offers a variety of fun activities and attractions. Here are some highlights:

# 1. **Pike Place Market**: Visit this iconic market to explore local vendors, fresh produce, artisanal products, and watch the famous fish throwing.

# 2. **Space Needle**: Take a trip to the observation deck for stunning panoramic views of the city, Puget Sound, and the surrounding mountains.

# 3. **Chihuly Garden and Glass**: Marvel at the stunning glass art installations created by artist Dale Chihuly, located right next to the Space Needle.

# 4. **Seattle Waterfront**: Enjoy a stroll along the waterfront, visit the Seattle Aquarium, and take a ferry ride to nearby islands like Bainbridge Island.

# 5. **Museum of Pop Culture (MoPOP)**: Explore exhibits on music, science fiction, and pop culture in this architecturally striking building.

# 6. **Seattle Art Museum (SAM)**: Discover an extensive collection of art from around the world, including contemporary and Native American art.

# 7. **Gas Works Park**: Relax in this unique park that features remnants of an old gasification plant, offering great views of the Seattle skyline and Lake Union.

# 8. **Discovery Park**: Enjoy nature trails, beaches, and beautiful views of the Puget Sound and the Olympic Mountains in this large urban park.

# 9. **Ballard Locks**: Watch boats navigate the locks and see fish swimming upstream during the salmon migration season.

# 10. **Fremont Troll**: Check out this quirky public art installation under a bridge in the Fremont neighborhood.

# 11. **Underground Tour**: Take an entertaining guided tour through the underground passages of Pioneer Square to learn about Seattle's history.

# 12. **Brewery Tours**: Seattle is known for its craft beer scene. Visit local breweries for tastings and tours.

# 13. **Seattle Center**: Explore the cultural complex that includes the Space Needle, MoPOP, and various festivals and events throughout the year.

# These are just a few options, and Seattle has something for everyone, whether you're into outdoor activities, culture, history, or food!
# -----
# Question: What was the first thing you mentioned?
# Response: The first thing I mentioned was **Pike Place Market**. It's an iconic market in Seattle known for its local vendors, fresh produce, artisanal products, and the famous fish throwing by the fishmongers. It's a vibrant place full of sights, sounds, and delicious food.
```

From the second response, you can see the agent now can recall its own previous responses.

### Tools

Tools are code that can be executed by an agent to perform actions. A tool can be a simple function such as a calculator, or an API call to a third-party service such as stock price lookup or weather forecast. In the context of AI agents, tools are designed to be executed by agents in response to model-generated function calls.

AutoGen provides the `autogen_core.tools` module with a suite of built-in tools and utilities for creating and running custom tools.

#### Built-in Tools

One of the built-in tools is the `PythonCodeExecutionTool`, which allows agents to execute Python code snippets.

Here is how you create the tool and use it.

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

The `DockerCommandLineCodeExecutor` class is a built-in code executor that runs Python code snippets in a subprocess in the command line environment of a docker container. The `PythonCodeExecutionTool` class wraps the code executor and provides a simple interface to execute Python code snippets.

Examples of other built-in tools

- `LocalSearchTool` and `GlobalSearchTool` for using GraphRAG.
- `mcp_server_tools` for using Model Context Protocol (MCP) servers as tools.
- `HttpTool` for making HTTP requests to REST APIs.
- `LangChainToolAdapter` for using LangChain tools.

#### Custom Function Tools

A tool can also be a simple Python function that performs a specific action. To create a custom function tool, you just need to create a Python function and use the `FunctionTool` class to wrap it.

The `FunctionTool` class uses descriptions and type annotations to inform the LLM when and how to use a given function. The description provides context about the function’s purpose and intended use cases, while type annotations inform the LLM about the expected parameters and return type.

For example, a simple tool to obtain the stock price of a company might look like this:

```python
import random

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated


async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    # Returns a random stock price for demonstration purposes.
    return random.uniform(10, 200)


# Create a function tool.
stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

# Run the tool.
cancellation_token = CancellationToken()
result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token)

# Print the result.
print(stock_price_tool.return_value_as_string(result))
# 143.83831971965762
```

#### Calling Tools with Model Clients

In AutoGen, every tool is a subclass of `BaseTool`, which automatically generates the JSON schema for the tool. For example, to get the JSON schema for the `stock_price_tool`, we can use the `schema` property.

```
stock_price_tool.schema
# {'name': 'get_stock_price',
#  'description': 'Get the stock price.',
#  'parameters': {'type': 'object',
#   'properties': {'ticker': {'description': 'ticker',
#     'title': 'Ticker',
#     'type': 'string'},
#    'date': {'description': 'Date in YYYY/MM/DD',
#     'title': 'Date',
#     'type': 'string'}},
#   'required': ['ticker', 'date'],
#   'additionalProperties': False},
#  'strict': False}
```

Model clients use the JSON schema of the tools to generate tool calls.

Here is an example of how to use the `FunctionTool` class with a `OpenAIChatCompletionClient`. Other model client classes can be used in a similar way. See [Model Clients](#model-clients) for more details.

```python
import json

from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create the OpenAI chat completion client. Using OPENAI_API_KEY from environment variable.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create a user message.
user_message = UserMessage(content="What is the stock price of AAPL on 2021/01/01?", source="user")

# Run the chat completion with the stock_price_tool defined above.
cancellation_token = CancellationToken()
create_result = await model_client.create(
    messages=[user_message], tools=[stock_price_tool], cancellation_token=cancellation_token
)
create_result.content
# [FunctionCall(id='call_tpJ5J1Xoxi84Sw4v0scH0qBM', arguments='{"ticker":"AAPL","date":"2021/01/01"}', name='get_stock_price')]
```

What is actually going on under the hood of the call to the `create` method? The model client takes the list of tools and generates a JSON schema for the parameters of each tool. Then, it generates a request to the model API with the tool’s JSON schema and the other messages to obtain a result.

Many models, such as OpenAI’s GPT-4o and Llama-3.2, are trained to produce tool calls in the form of structured JSON strings that conform to the JSON schema of the tool. AutoGen’s model clients then parse the model’s response and extract the tool call from the JSON string.

The result is a list of `FunctionCall` objects, which can be used to run the corresponding tools.

We use `json.loads` to parse the JSON string in the `arguments` field into a Python dictionary. The `run_json()` method takes the dictionary and runs the tool with the provided arguments.

```python
assert isinstance(create_result.content, list)
arguments = json.loads(create_result.content[0].arguments)  # type: ignore
tool_result = await stock_price_tool.run_json(arguments, cancellation_token)
tool_result_str = stock_price_tool.return_value_as_string(tool_result)
tool_result_str
# '32.381250753393104'
```

Now you can make another model client call to have the model generate a reflection on the result of the tool execution.

The result of the tool call is wrapped in a `FunctionExecutionResult` object, which contains the result of the tool execution and the ID of the tool that was called. The model client can use this information to generate a reflection on the result of the tool execution.

```python
# Create a function execution result
exec_result = FunctionExecutionResult(
    call_id=create_result.content[0].id,  # type: ignore
    content=tool_result_str,
    is_error=False,
    name=stock_price_tool.name,
)

# Make another chat completion with the history and function execution result message.
messages = [
    user_message,
    AssistantMessage(content=create_result.content, source="assistant"),  # assistant message with tool call
    FunctionExecutionResultMessage(content=[exec_result]),  # function execution result message
]
create_result = await model_client.create(messages=messages, cancellation_token=cancellation_token)  # type: ignore
print(create_result.content)
await model_client.close()
# The stock price of AAPL (Apple Inc.) on January 1, 2021, was approximately $32.38.
```

#### Tool-Equipped Agent

Putting the model client and the tools together, you can create a tool-equipped agent that can use tools to perform actions, and reflect on the results of those actions.

Note

The Core API is designed to be minimal and you need to build your own agent logic around model clients and tools. For “pre-built” agents that can use tools, please refer to the AgentChat API.

```python
import asyncio
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    AgentId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)
```

When handling a user message, the `ToolUseAgent` class first use the model client to generate a list of function calls to the tools, and then run the tools and generate a reflection on the results of the tool execution. The reflection is then returned to the user as the agent’s response.

To run the agent, let’s create a runtime and register the agent with the runtime.

```python
# Create the model client.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
# Create a runtime.
runtime = SingleThreadedAgentRuntime()
# Create the tools.
tools: List[Tool] = [FunctionTool(get_stock_price, description="Get the stock price.")]
# Register the agents.
await ToolUseAgent.register(
    runtime,
    "tool_use_agent",
    lambda: ToolUseAgent(
        model_client=model_client,
        tool_schema=tools,
    ),
)
# AgentType(type='tool_use_agent')
```

This example uses the `OpenAIChatCompletionClient`, for Azure OpenAI and other clients, see [Model Clients](#model-clients). Let’s test the agent with a question about stock price.

```python
# Start processing messages.
runtime.start()
# Send a direct message to the tool agent.
tool_use_agent = AgentId("tool_use_agent", "default")
response = await runtime.send_message(Message("What is the stock price of NVDA on 2024/06/01?"), tool_use_agent)
print(response.content)
# Stop processing messages.
await runtime.stop()
await model_client.close()
# The stock price of NVIDIA (NVDA) on June 1, 2024, was approximately $140.05.
```

### Workbench (and MCP)

A Workbench provides a collection of tools that share state and resources. Different from Tool, which provides an interface to a single tool, a workbench provides an interface to call different tools and receive results as the same types.

#### Using Workbench

Here is an example of how to create an agent using Workbench.

```python
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import ToolResult, Workbench

@dataclass
class Message:
    content: str


class WorkbenchAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, workbench: Workbench) -> None:
        super().__init__("A workbench agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._workbench.tools,  # Pass the workbench tools to the model client.
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(self, call: FunctionCall, cancellation_token: CancellationToken) -> ToolResult:
        # Run the tool and capture the result.
        try:
            result = await self._workbench.run_tool(call, cancellation_token)
            return ToolResult(
                call_id=call.id, content=result.content, is_error=result.is_error, name=result.name
            )
        except Exception as e:
            return ToolResult(call_id=call.id, content=str(e), is_error=True, name=call.name)
```

When handling a user message, the `WorkbenchAgent` class first use the model client to generate a list of function calls to the tools, and then run the tools and generate a reflection on the results of the tool execution. The reflection is then returned to the user as the agent’s response.

To run the agent, let’s create a runtime and register the agent with the runtime.

```python
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.mcp.mcp_workbench import McpWorkbench

# Create the model client.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
# Create a runtime.
runtime = SingleThreadedAgentRuntime()
# Create the workbench.
workbench = McpWorkbench()
# Register the agents.
await WorkbenchAgent.register(
    runtime,
    "workbench_agent",
    lambda: WorkbenchAgent(
        model_client=model_client,
        workbench=workbench,
    ),
)
# AgentType(type='workbench_agent')
```

This example uses the `OpenAIChatCompletionClient`, for Azure OpenAI and other clients, see [Model Clients](#model-clients). Let’s test the agent with a question about stock price.

```python
# Start processing messages.
runtime.start()
# Send a direct message to the workbench agent.
workbench_agent = AgentId("workbench_agent", "default")
response = await runtime.send_message(Message("What is the stock price of NVDA on 2024/06/01?"), workbench_agent)
print(response.content)
# Stop processing messages.
await runtime.stop()
await model_client.close()
# The stock price of NVIDIA (NVDA) on June 1, 2024, was approximately $140.05.
```

#### Using MCP Workbench

`McpWorkbench` is an implementation of `Workbench` that enables an agent to call tools on an MCP server. For example, you can use `McpWorkbench` to call tools on the `Playwright` MCP server to browse the web.

Here is an example of how to use `McpWorkbench`.

```python
from autogen_core import AgentId, Message
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.mcp.mcp_workbench import McpWorkbench
from autogen_ext.mcp.playwright import PlaywrightMcpServer

# Create the Playwright MCP server parameters.
playwright_server_params = PlaywrightMcpServer(
    # headless=False,  # Uncomment to see the browser UI.
)

# Create the model client.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create a runtime.
runtime = SingleThreadedAgentRuntime()

# Create the workbench.
async with McpWorkbench(playwright_server_params) as workbench:
    # Register the agents.
    await WorkbenchAgent.register(
        runtime,
        "web_agent",
        lambda: WorkbenchAgent(
            model_client=model_client,
            workbench=workbench,
        ),
    )

    # Start processing messages.
    runtime.start()

    # Send a direct message to the web agent.
    web_agent = AgentId("web_agent", "default")
    response = await runtime.send_message(
        Message("Use Bing to find out the address of Microsoft Building 99"), web_agent
    )
    print(response.content)

    # Stop processing messages.
    await runtime.stop()

await model_client.close()
# The address of Microsoft Building 99 is 15760 NE 32nd St, Redmond, WA 98052, USA.
```

#### Web Browsing Agent Example

Here is an example of how to create a web browsing agent that uses `Playwright` MCP server to browse the web and answer questions.

```python
import asyncio
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    AgentId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import ToolResult
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.mcp.mcp_workbench import McpWorkbench
from autogen_ext.mcp.playwright import PlaywrightMcpServer


@dataclass
class Message:
    content: str


class WebAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, workbench: McpWorkbench) -> None:
        super().__init__("A web browsing agent")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content="You are a helpful AI assistant that can browse the web.")
        ]
        self._model_client = model_client
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._workbench.tools,  # Pass the workbench tools to the model client.
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(self, call: FunctionCall, cancellation_token: CancellationToken) -> ToolResult:
        # Run the tool and capture the result.
        try:
            result = await self._workbench.run_tool(call, cancellation_token)
            return ToolResult(
                call_id=call.id, content=result.content, is_error=result.is_error, name=result.name
            )
        except Exception as e:
            return ToolResult(call_id=call.id, content=str(e), is_error=True, name=call.name)


# Create the Playwright MCP server parameters.
playwright_server_params = PlaywrightMcpServer(
    # headless=False,  # Uncomment to see the browser UI.
)

# Create the model client.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create a runtime.
runtime = SingleThreadedAgentRuntime()

# Create the workbench.
async with McpWorkbench(playwright_server_params) as workbench:
    # Register the agents.
    await WebAgent.register(
        runtime,
        "web_agent",
        lambda: WebAgent(
            model_client=model_client,
            workbench=workbench,
        ),
    )

    # Start processing messages.
    runtime.start()

    # Send a direct message to the web agent.
    web_agent = AgentId("web_agent", "default")
    response = await runtime.send_message(
        Message("Use Bing to find out the address of Microsoft Building 99"), web_agent
    )
    print(response.content)

    # Stop processing messages.
    await runtime.stop()

await model_client.close()
# The address of Microsoft Building 99 is 15760 NE 32nd St, Redmond, WA 98052, USA.


### Command Line Code Executors
Command line code execution is the simplest form of code execution. Generally speaking, it will save each code block to a file and then execute that file. This means that each code block is executed in a new process. There are two forms of this executor:

Docker (DockerCommandLineCodeExecutor) - this is where all commands are executed in a Docker container

Local (LocalCommandLineCodeExecutor) - this is where all commands are executed on the host machine

#### Docker
Note

To use DockerCommandLineCodeExecutor, ensure the autogen-ext[docker] package is installed. For more details, see the Packages Documentation.

The DockerCommandLineCodeExecutor will create a Docker container and run all commands within that container. The default image that is used is python:3-slim, this can be customized by passing the image parameter to the constructor. If the image is not found locally then the class will try to pull it. Therefore, having built the image locally is enough. The only thing required for this image to be compatible with the executor is to have sh and python installed. Therefore, creating a custom image is a simple and effective way to ensure required system dependencies are available.

You can use the executor as a context manager to ensure the container is cleaned up after use. Otherwise, the atexit module will be used to stop the container when the program exits.

Inspecting the container
If you wish to keep the container around after AutoGen is finished using it for whatever reason (e.g. to inspect the container), then you can set the auto_remove parameter to False when creating the executor. stop_container can also be set to False to prevent the container from being stopped at the end of the execution.

Example
```python
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:  # type: ignore
    print(
        await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
    )
# CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='coding/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.python')
```

Combining an Application in Docker with a Docker based executor
It is desirable to bundle your application into a Docker image. But then, how do you allow your containerised application to execute code in a different container?

The recommended approach to this is called “Docker out of Docker”, where the Docker socket is mounted to the main AutoGen container, so that it can spawn and control “sibling” containers on the host. This is better than what is called “Docker in Docker”, where the main container runs a Docker daemon and spawns containers within itself. You can read more about this here.

To do this you would need to mount the Docker socket into the container running your application. This can be done by adding the following to the docker run command:

`-v /var/run/docker.sock:/var/run/docker.sock`
This will allow your application’s container to spawn and control sibling containers on the host.

If you need to bind a working directory to the application’s container but the directory belongs to your host machine, use the bind_dir parameter. This will allow the application’s container to bind the host directory to the new spawned containers and allow it to access the files within the said directory. If the bind_dir is not specified, it will fallback to work_dir.

#### Local

Attention

The local version will run code on your local system. Use it with caution.

To execute code on the host machine, as in the machine running your application, LocalCommandLineCodeExecutor can be used.

Example

```python
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
print(
    await local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
)
# CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='/home/ekzhu/agnext/python/packages/autogen-core/docs/src/user-guide/core-user-guide/framework/coding/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.py')
```

Local within a Virtual Environment
If you want the code to run within a virtual environment created as part of the application’s setup, you can specify a directory for the newly created environment and pass its context to LocalCommandLineCodeExecutor. This setup allows the executor to use the specified virtual environment consistently throughout the application’s lifetime, ensuring isolated dependencies and a controlled runtime environment.

```python
import venv
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

venv_dir = work_dir / ".venv"
venv_builder = venv.EnvBuilder(with_pip=True)
venv_builder.create(venv_dir)
venv_context = venv_builder.ensure_directories(venv_dir)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir, virtual_env_context=venv_context)
await local_executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="bash", code="pip install matplotlib"),
    ],
    cancellation_token=CancellationToken(),
)
# CommandLineCodeResult(exit_code=0, output='', code_file='/Users/gziz/Dev/autogen/python/packages/autogen-core/docs/src/user-guide/core-user-guide/framework/coding/tmp_code_d2a7db48799db3cc785156a11a38822a45c19f3956f02ec69b92e4169ecbf2ca.bash')
```

As we can see, the code has executed successfully, and the installation has been isolated to the newly created virtual environment, without affecting our global environment.

## AgentChat Framework

### Quick Start

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

# asyncio.run(basic_example())
```

### Tool Integration

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

### Structured Output

```python
from pydantic import BaseModel, Field
from typing import List

class ReasoningStep(BaseModel):
    explanation: str = Field(description="Step explanation")
    output: str = Field(description="Step output")

class MathReasoning(BaseModel):
    steps: List[ReasoningStep] = Field(description="Reasoning steps")
    final_answer: str = Field(description="Final answer")

# Use with structured output
response = await model_client.create(
    messages=[{"role": "user", "content": "What is 25 + 17?"}],
    extra_create_args={"response_format": MathReasoning}
)
```


## Development Patterns

### 1. Reflection Pattern

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
        # Generate code
        code = await self._generate_code(message.task)
    
        # Send for review
        await self.publish_message(
            CodeReview(code=code, approved=False, feedback=""),
            topic_id=TopicId("review", self.id.key)
        )
  
    async def _generate_code(self, task: str) -> str:
        # Implementation details for code generation
        response = await self._model_client.create([
            {"role": "user", "content": f"Write code for: {task}"}
        ])
        return response.content
```

### 2. Sequential Workflow

Deterministic processing pipeline.

```python
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
    
        # Pass to next stage
        await self.publish_message(
            ProcessedDataMessage(data=processed_data),
            topic_id=TopicId("validator", self.id.key)
        )
  
    async def _process(self, data: str) -> str:
        # Processing logic here
        return f"processed_{data}"
```

### 3. Multi-Agent Debate

Collaborative problem-solving with multiple perspectives.

```python
@dataclass
class DebateQuestion:
    question: str

@dataclass
class DebatePrompt:
    question: str
    round: int

class DebateOrchestrator(RoutedAgent):
    def __init__(self, model_client, num_agents=3, rounds=2):
        super().__init__("Debate orchestrator")
        self._model_client = model_client
        self._num_agents = num_agents
        self._rounds = rounds
  
    @message_handler
    async def start_debate(self, message: DebateQuestion, ctx: MessageContext):
        # Distribute question to all debaters
        for agent_id in range(self._num_agents):
            await self.publish_message(
                DebatePrompt(question=message.question, round=1),
                topic_id=TopicId(f"debater_{agent_id}", self.id.key)
            )
```

### 4. Mixture of Agents

Hierarchical agent orchestration for complex tasks.

```python
import asyncio
from typing import List
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage

class WorkerAgent(BaseChatAgent):
    """Simple worker agent for demonstration"""
    async def on_messages(self, messages, cancellation_token):
        task = messages[-1].content
        result = f"Worker processed: {task}"
        return Response(chat_message=TextMessage(content=result, source=self.name))

class OrchestratorAgent(BaseChatAgent):
    def __init__(self, name: str, worker_agents: List[WorkerAgent]):
        super().__init__(name)
        self._workers = worker_agents
  
    async def on_messages(self, messages, cancellation_token):
        task = messages[-1].content
    
        # Distribute to workers
        worker_tasks = [
            worker.on_messages([TextMessage(content=task, source="user")], cancellation_token)
            for worker in self._workers
        ]
    
        # Gather results
        responses = await asyncio.gather(*worker_tasks)
    
        # Synthesize final response
        return await self._synthesize_responses(responses)
  
    async def _synthesize_responses(self, responses):
        # Combine worker responses into final result
        combined = " | ".join([r.chat_message.content for r in responses])
        return Response(chat_message=TextMessage(content=f"Combined: {combined}", source=self.name))
```

### Security & Best Practices

### Security & Best Practices

@dataclass
class Message:
    content: str

class WorkbenchAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, model_context: ChatCompletionContext, workbench: Workbench
    ) -> None:
        super().__init__("An agent with a workbench")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Add the user message to the model context.
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        print("---------User Message-----------")
        print(message.content)

    # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=(await self._workbench.list_tools()),
            cancellation_token=ctx.cancellation_token,
        )

    # Run tool call loop.
        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            print("---------Function Calls-----------")
            for call in create_result.content:
                print(call)

    # Add the function calls to the model context.
            await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

    # Call the tools using the workbench.
            print("---------Function Call Results-----------")
            results: List[ToolResult] = []
            for call in create_result.content:
                result = await self._workbench.call_tool(
                    call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                )
                results.append(result)
                print(result)

    # Add the function execution results to the model context.
            await self._model_context.add_message(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
            )

    # Run the chat completion again to reflect on the history and function execution results.
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )

    # Now we have a single message as the result.
        assert isinstance(create_result.content, str)

    print("---------Final Response-----------")
        print(create_result.content)

    # Add the assistant message to the model context.
        await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

    # Return the result as a message.
        return Message(content=create_result.content)

```
In this example, the agent calls the tools provided by the workbench in a loop until the model returns a final answer.

### Security & Best Practices

```bash
# npx playwright install chrome
```

Start the Playwright MCP server in a terminal.

```bash
# npx @playwright/mcp@latest --port 8931
```

Then, create the agent using the WorkbenchAgent class and McpWorkbench with the Playwright MCP server URL.

```python
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

playwright_server_params = SseServerParams(
    url="http://localhost:8931/sse",
)

# Start the workbench in a context manager.
# You can also start and stop the workbench using `workbench.start()` and `workbench.stop()`.
async with McpWorkbench(playwright_server_params) as workbench:  # type: ignore
    # Create a single-threaded agent runtime.
    runtime = SingleThreadedAgentRuntime()

    # Register the agent with the runtime.
    await WorkbenchAgent.register(
        runtime=runtime,
        type="WebAgent",
        factory=lambda: WorkbenchAgent(
            model_client=OpenAIChatCompletionClient(model="gpt-4.1-nano"),
            model_context=BufferedChatCompletionContext(buffer_size=10),
            workbench=workbench,
        ),
    )

    # Start the runtime.
    runtime.start()

    # Send a message to the agent.
    await runtime.send_message(
        Message(content="Use Bing to find out the address of Microsoft Building 99"),
        recipient=AgentId("WebAgent", "default"),
    )

    # Stop the runtime.
    await runtime.stop()
```

```
---------User Message-----------
Use Bing to find out the address of Microsoft Building 99
---------Function Calls-----------
FunctionCall(id='call_oJl0E0hWvmKZrzAM7huiIyus', arguments='{"url": "https://www.bing.com"}', name='browser_navigate')
FunctionCall(id='call_Qfab5bAsveZIVg2v0aHl4Kgv', arguments='{}', name='browser_snapshot')
---------Function Call Results-----------
ToolResult(type='ToolResult', name='browser_navigate', result=[TextResultContent...], is_error=False)
ToolResult(type='ToolResult', name='browser_snapshot', result=[TextResultContent...], is_error=False)
---------Final Response-----------
Microsoft Building 99 is located at 1 Microsoft Way, Redmond, WA 98052, USA.
---------Assistant Message-----------
Microsoft Building 99 is located at 1 Microsoft Way, Redmond, WA 98052, USA.

```

## Security & Best Practices

### Security Considerations

#### Input Validation

```python
from pydantic import BaseModel, validator

class SecureMessage(BaseModel):
    content: str
    user_id: str
  
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10000:
            raise ValueError('Content too long')
        # Add additional validation
        return v
```

#### Authentication & Authorization

```python
class SecureAgent(RoutedAgent):
    @message_handler
    async def handle_secure_message(self, message: SecureMessage, ctx: MessageContext):
        # Verify user permissions
        if not await self._verify_permissions(message.user_id, ctx):
            raise PermissionError("Unauthorized access")
    
        # Process message
        await self._process_message(message)
```

#### Secrets Management

```python
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_secure_client():
    # Use environment variables or Azure Key Vault
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to Key Vault
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url="https://vault.vault.azure.net", credential=credential)
        api_key = client.get_secret("openai-api-key").value
  
    return OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
```

### Development Best Practices

#### Error Handling

```python
class RobustAgent(RoutedAgent):
    @message_handler
    async def handle_message(self, message: TaskMessage, ctx: MessageContext):
        try:
            result = await self._process_task(message)
            await self.publish_message(result, topic_id=TopicId("results", self.id.key))
        except Exception as e:
            # Log error
            logger.error(f"Task processing failed: {e}")
        
            # Publish error for handling
            await self.publish_message(
                ErrorMessage(error=str(e), original_task=message),
                topic_id=TopicId("errors", self.id.key)
            )
```

#### Monitoring & Observability

```python
import logging
import time
import asyncio
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

class MonitoredAgent(RoutedAgent):
    def __init__(self, name: str):
        super().__init__(name)
  
    @monitor_performance
    @message_handler
    async def handle_message(self, message: TaskMessage, ctx: MessageContext):
        # Agent logic here
        await asyncio.sleep(0.1)  # Simulate work
        print(f"Processed message: {message.content}")
```

#### Resource Management

```python
class ResourceManagedAgent(RoutedAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
  
    @message_handler
    async def handle_task(self, message: TaskMessage, ctx: MessageContext):
        async with self._semaphore:
            # Process with resource limits
            async with asyncio.timeout(30):  # 30-second timeout
                result = await self._process_task(message)
                return result
```

## Production Deployment

### Docker Configuration

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

### Environment Configuration

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

### Scaling Considerations

```python
# Horizontal scaling with Redis pub/sub
import redis.asyncio as redis

class DistributedRuntime:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url)
  
    async def publish_message(self, message, topic):
        # Publish to Redis for cross-instance communication
        await self.redis.publish(f"autogen:{topic}", message.json())
  
    async def subscribe_to_topic(self, topic, handler):
        # Subscribe to Redis channels
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"autogen:{topic}")
    
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await handler(message['data'])
```

### Performance Optimization

```python
import aiohttp
import asyncio
import hashlib
import json
from cachetools import TTLCache

# Connection pooling and caching
class OptimizedModelClient:
    def __init__(self):
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
  
    def _get_cache_key(self, messages, kwargs):
        """Generate cache key from messages and parameters"""
        content = json.dumps(messages, sort_keys=True) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
  
    async def create_with_cache(self, messages, **kwargs):
        cache_key = self._get_cache_key(messages, kwargs)
    
        if cache_key in self._cache:
            return self._cache[cache_key]
    
        result = await self._create(messages, **kwargs)
        self._cache[cache_key] = result
        return result
  
    async def _create(self, messages, **kwargs):
        # Actual API call implementation
        # This would be your real model client call
        return {"content": "Mock response"}
```

### Health Monitoring

```python
import time
import psutil
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Global variables for demonstration
start_time = time.time()
runtime = None  # Your runtime instance
model_client = None  # Your model client instance

@app.get("/health")
async def health_check():
    try:
        # Check runtime health
        if runtime and not hasattr(runtime, 'is_healthy'):
            # Simple health check - verify runtime is running
            if not runtime._running:
                raise HTTPException(status_code=503, detail="Runtime not running")
    
        # Check model client connectivity (if applicable)
        # await model_client.health_check()  # Uncomment if your client has this method
    
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/metrics")
async def get_metrics():
    agent_count = len(runtime._agents) if runtime else 0
    queue_size = 0  # You would implement this based on your runtime
  
    return {
        "active_agents": agent_count,
        "message_queue_size": queue_size,
        "memory_usage": psutil.Process().memory_info().rss,
        "uptime": time.time() - start_time
    }
```

## AutoGen Studio (Research Prototype)

**⚠️ WARNING**: AutoGen Studio is a research prototype and **NOT production-ready**.

### Key Limitations

- No built-in authentication/authorization
- Potential security vulnerabilities
- Not designed for production workloads
- Code execution security risks

### Safe Usage

```python
# Use Docker for isolation
docker_config = {
    "use_docker": True,
    "docker_image": "python:3.11-slim",
    "timeout": 30,
    "work_dir": "/tmp/autogen_work"
}

# Limited permissions
code_executor = DockerCommandLineCodeExecutor(
    **docker_config,
    auto_remove=True,
    stop_container=True
)
```

---

## Quick Reference

### Essential Imports

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

### Common Patterns

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

This knowledge base provides a comprehensive yet focused guide for both learning AutoGen concepts and implementing production systems efficiently.
