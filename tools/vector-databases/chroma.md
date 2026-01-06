# Chroma

## What is Chroma?

Chroma is an open-source embedding database that makes it easy to build LLM applications with embeddings. It's lightweight, easy to use, and perfect for prototyping and small to medium-scale applications with economics papers.

## Key Features

- **Easy Setup**: Simple Python API
- **Local First**: Runs locally, no external dependencies
- **LangChain Integration**: Built-in LangChain support
- **Persistent Storage**: Save and load collections
- **Metadata Filtering**: Filter by metadata alongside vector search

## Installation

```bash
pip install chromadb
```

## Quick Start

### Basic Usage with Economics Papers

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection for economics papers
collection = client.create_collection(
    name="economics_papers",
    metadata={"description": "Economics research papers on agentic AI"}
)

# Add economics papers
collection.add(
    documents=[
        "Agentic AI enables autonomous economics research agents.",
        "Vector databases support semantic search for economics papers.",
        "RAG improves LLM accuracy with economics research context."
    ],
    ids=["econ_paper1", "econ_paper2", "econ_paper3"],
    metadatas=[
        {"source": "arxiv", "year": 2024, "category": "behavioral_economics"},
        {"source": "arxiv", "year": 2024, "category": "econometrics"},
        {"source": "arxiv", "year": 2023, "category": "economic_theory"}
    ]
)

# Query economics papers
results = collection.query(
    query_texts=["What is agentic AI in economics?"],
    n_results=2
)
print(results)
```

### With LangChain for Economics Papers

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load economics papers
loader = TextLoader("economics_paper.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Query economics papers
query = "What are the main findings about market efficiency?"
docs = vectorstore.similarity_search(query, k=3)
```

## Economics Research Use Cases

### Economics Literature Search

```python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create collection for economics papers
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="economics_papers",
    embedding_function=embeddings,
    persist_directory="./papers_db"
)

# Add economics papers (from ArXiv, etc.)
economics_papers = [
    {"title": "Agentic AI in Behavioral Economics", "abstract": "..."},
    # ... more economics papers
]

for paper in economics_papers:
    vectorstore.add_texts(
        texts=[paper["abstract"]],
        metadatas=[{"title": paper["title"], "source": "arxiv", "field": "economics"}]
    )

# Semantic search for economics papers
results = vectorstore.similarity_search(
    "agentic AI applications in behavioral economics",
    k=5,
    filter={"field": "economics"}
)
```

**Note**: This pattern works for finance (financial papers), political science (policy papers), and other research document collections.

## Pros and Cons

### Pros
- Easy to use and set up
- No external dependencies
- Good for prototyping economics research
- Free and open-source
- LangChain integration

### Cons
- Limited scalability
- Not ideal for very large economics paper collections
- Performance may lag for millions of vectors

## Resources

- **GitHub**: [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)
- **Documentation**: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- **LangChain Integration**: [https://python.langchain.com/docs/integrations/vectorstores/chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma)

