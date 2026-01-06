# Weaviate

## What is Weaviate?

Weaviate is an open-source vector database that can be self-hosted or used as a managed service. It provides a GraphQL API and advanced features like hybrid search, making it ideal for complex economics research applications.

## Key Features

- **GraphQL API**: Flexible query interface
- **Hybrid Search**: Combine vector and keyword search
- **Self-Hosted or Managed**: Deploy yourself or use Weaviate Cloud
- **Advanced Filtering**: Complex metadata filtering
- **Multi-tenancy**: Support for multiple tenants

## Installation

### Self-Hosted (Docker)

```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  weaviate/weaviate:latest
```

### Python Client

```bash
pip install weaviate-client
```

## Quick Start

### Basic Usage with Economics Papers

```python
import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Create schema for economics papers
schema = {
    "class": "EconomicsPaper",
    "properties": [
        {"name": "title", "dataType": ["string"]},
        {"name": "abstract", "dataType": ["text"]},
        {"name": "year", "dataType": ["int"]},
        {"name": "field", "dataType": ["string"]},  # e.g., "behavioral_economics"
    ]
}

client.schema.create_class(schema)

# Add economics papers
client.data_object.create(
    data_object={
        "title": "Agentic AI in Economics",
        "abstract": "...",
        "year": 2024,
        "field": "behavioral_economics"
    },
    class_name="EconomicsPaper"
)
```

### With LangChain

```python
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split economics papers
loader = TextLoader("economics_paper.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(
    documents=texts,
    embedding=embeddings,
    weaviate_url="http://localhost:8080"
)

# Query economics papers
query = "What are the main findings about market efficiency?"
docs = vectorstore.similarity_search(query, k=3)
```

## Economics Research Use Cases

### Hybrid Search for Economics Papers

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Hybrid search (vector + keyword) for economics papers
result = (
    client.query
    .get("EconomicsPaper", ["title", "abstract"])
    .with_hybrid(
        query="agentic AI behavioral economics",
        alpha=0.7  # 0.7 vector, 0.3 keyword
    )
    .with_where({
        "path": ["field"],
        "operator": "Equal",
        "valueString": "behavioral_economics"
    })
    .with_limit(10)
    .do()
)
```

**Note**: This pattern works for finance (financial documents), political science (policy papers), and other complex research document collections.

## Pros and Cons

### Pros
- Advanced features (hybrid search, GraphQL)
- Self-hosted option
- Good for complex economics research applications
- Flexible querying
- Open-source

### Cons
- More complex setup
- Steeper learning curve
- Requires more infrastructure knowledge

## Resources

- **Website**: [https://weaviate.io/](https://weaviate.io/)
- **Documentation**: [https://weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)
- **LangChain Integration**: [https://python.langchain.com/docs/integrations/vectorstores/weaviate](https://python.langchain.com/docs/integrations/vectorstores/weaviate)

