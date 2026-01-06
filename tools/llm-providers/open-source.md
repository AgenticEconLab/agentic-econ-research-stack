# Open Source LLMs

## Overview

Open-source LLMs allow you to run models locally or on your own infrastructure, providing full control over data and privacy. Popular options include Llama, Mistral, and other open-source models. These are useful for economics research when data privacy is a concern.

## Popular Models

- **Llama 2/3**: Meta's open-source models
- **Mistral**: High-performance open-source models
- **Falcon**: Open-source models from UAE
- **Vicuna**: Fine-tuned Llama models

## Installation

### Using Ollama (Easiest)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
ollama pull mistral
```

### Using Hugging Face

```bash
pip install transformers torch
```

## Quick Start

### With Ollama for Economics Research

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",
        "prompt": "What is agentic AI in economics research?",
        "stream": False
    }
)

print(response.json()["response"])
```

### With LangChain

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

response = llm.invoke("What is agentic AI in behavioral economics?")
print(response)
```

## Economics Research Use Cases

### Local Economics Research Agent

```python
from langchain_community.llms import Ollama
from langchain.agents import create_openai_functions_agent

# Use local model for economics research
llm = Ollama(model="mistral")

# Create economics research agent (works with local models too)
agent = create_openai_functions_agent(llm, tools, prompt)
```

## Pros and Cons

### Pros
- Full data privacy for economics research
- No API costs
- Full control
- Customizable

### Cons
- Requires infrastructure
- Lower performance than GPT-4
- Setup complexity
- Resource intensive

## Resources

- **Ollama**: [https://ollama.ai/](https://ollama.ai/)
- **Hugging Face**: [https://huggingface.co/](https://huggingface.co/)
- **Llama**: [https://llama.meta.com/](https://llama.meta.com/)
- **Mistral**: [https://mistral.ai/](https://mistral.ai/)

