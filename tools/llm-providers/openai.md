# OpenAI

## Overview

OpenAI provides the most popular LLM APIs, including GPT-4, GPT-3.5, and embedding models. It's the default choice for many agentic AI applications in economics research due to its performance, reliability, and extensive framework support.

## Key Models

- **GPT-4**: Most capable, best for complex economics reasoning
- **GPT-3.5-turbo**: Fast and cost-effective, good for most economics research tasks
- **text-embedding-ada-002**: Embedding model for RAG with economics papers

## Installation

```bash
pip install openai
# or with LangChain
pip install langchain-openai
```

## Quick Start

### Basic Usage for Economics Research

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an economics research assistant."},
        {"role": "user", "content": "What is agentic AI in economics research?"}
    ]
)

print(response.choices[0].message.content)
```

### With LangChain for Economics Research

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

response = llm.invoke("What is agentic AI in behavioral economics?")
print(response.content)
```

## Economics Research Use Cases

### Economics Research Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create economics research agent with tools
agent = create_openai_functions_agent(llm, tools, prompt)
```

### Embeddings for Economics RAG

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Use with vector stores for economics papers
vectorstore = Chroma.from_documents(economics_papers, embeddings)
```

## Pricing

- **GPT-4**: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **GPT-3.5-turbo**: ~$0.0015 per 1K input tokens, ~$0.002 per 1K output tokens
- **Embeddings**: ~$0.0001 per 1K tokens

## Pros and Cons

### Pros
- Best performance and capabilities
- Extensive framework support
- Reliable and stable
- Good documentation
- Large community

### Cons
- Higher cost for GPT-4
- API dependency
- Rate limits

## Resources

- **Website**: [https://openai.com/](https://openai.com/)
- **API Documentation**: [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **LangChain Integration**: [https://python.langchain.com/docs/integrations/llms/openai](https://python.langchain.com/docs/integrations/llms/openai)

