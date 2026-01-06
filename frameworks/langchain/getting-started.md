# LangChain Getting Started

## Installation

### Basic Installation

```bash
pip install langchain
```

### With Common Integrations

```bash
pip install langchain langchain-openai langchain-community langchain-chroma
```

### For Economics Research Applications

```bash
pip install langchain langchain-openai langchain-community \
    langchain-chroma langchain-arxiv langchain-pubmed \
    langchain-python langchain-pandas
```

## Environment Setup

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Your First Economics Research Agent

### Simple Chat Agent

```python
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful economics research assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, [], prompt)
agent_executor = AgentExecutor(agent=agent, verbose=True)

# Run the agent
result = agent_executor.invoke({"input": "What are the latest findings on behavioral economics?"})
print(result["output"])
```

### Agent with Economics Data Tools

```python
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import numpy as np

# Define economics analysis tool
def calculate_economic_statistics(data):
    """Calculate economic statistics from a dataset."""
    df = pd.DataFrame(data)
    return {
        "gdp_growth": df['gdp'].pct_change().mean() if 'gdp' in df.columns else None,
        "inflation_rate": df['cpi'].pct_change().mean() if 'cpi' in df.columns else None,
        "unemployment_rate": df['unemployment'].mean() if 'unemployment' in df.columns else None,
        "correlation": df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else None
    }

# Create tool
tools = [
    Tool(
        name="EconomicStatistics",
        func=calculate_economic_statistics,
        description="Calculates economic statistics (GDP growth, inflation, unemployment, correlations) from economic data"
    )
]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an economics research assistant with access to economic analysis tools."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create and run agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Analyze economic indicators: GDP, inflation, and unemployment rates"
})
print(result["output"])
```

## Basic Concepts

### 1. Chains

Chains combine multiple components:

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Summarize this economics research topic: {topic}")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="agentic AI in behavioral economics")
```

### 2. Memory

Add conversation memory:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

chain.predict(input="What is behavioral economics?")
chain.predict(input="How can agentic AI be applied to it?")
```

### 3. Retrieval (RAG) for Economics Papers

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load and split economics papers
loader = TextLoader("economics_paper.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = qa_chain.run("What are the main findings about market efficiency?")
```

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Initialize Memory
**Problem**: Agent doesn't remember previous conversations
**Solution**: Always initialize and pass memory to your chain/agent

### Pitfall 2: Tool Description Quality
**Problem**: Agent doesn't use tools correctly
**Solution**: Write clear, detailed tool descriptions with examples

### Pitfall 3: Token Limits
**Problem**: Exceeding context window limits with long economics papers
**Solution**: Use text splitters, summarization, or reduce chunk sizes

### Pitfall 4: Cost Management
**Problem**: High API costs from verbose agents
**Solution**: Use `verbose=False` in production, monitor token usage

## Next Steps

- Explore [Economics Research Examples](scientific-research-examples.md)
- Check out [Resources](resources.md) for advanced tutorials
- Review [LangGraph](../langgraph/) for stateful workflows

