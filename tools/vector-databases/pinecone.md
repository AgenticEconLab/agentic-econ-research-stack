# Pinecone

## What is Pinecone?

Pinecone is a managed vector database service that provides high-performance, scalable vector search. It's ideal for production applications that need to handle large-scale semantic search over millions of economics research documents.

## Key Features

- **Managed Service**: Fully managed, no infrastructure to maintain
- **High Performance**: Optimized for fast similarity search
- **Scalability**: Handles millions of economics paper vectors
- **Metadata Filtering**: Filter by metadata alongside vector search
- **Real-time Updates**: Add and update economics paper vectors in real-time

## Installation

```bash
pip install pinecone-client
```

## Quick Start

### Setup for Economics Papers

```python
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Initialize Pinecone
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"  # Your environment
)

# Create index for economics papers (if it doesn't exist)
index_name = "economics-papers"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine"
    )

# Connect to index
index = pinecone.Index(index_name)
```

### With LangChain for Economics Papers

```python
from langchain.vectorstores import Pinecone
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
vectorstore = Pinecone.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name="economics-papers"
)

# Query economics papers
query = "What are the main findings about behavioral economics?"
docs = vectorstore.similarity_search(query, k=3)
```

## Economics Research Use Cases

### Large-Scale Economics Paper Search

```python
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Initialize
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(
    index_name="economics-papers",
    embedding=embeddings
)

# Add economics papers in batches
economics_papers = load_economics_papers_from_arxiv()  # Your function
batch_size = 100

for i in range(0, len(economics_papers), batch_size):
    batch = economics_papers[i:i+batch_size]
    texts = [p["abstract"] for p in batch]
    metadatas = [{"title": p["title"], "year": p["year"], "field": "economics"} for p in batch]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

# Semantic search with filtering
results = vectorstore.similarity_search(
    "agentic AI in economics",
    k=10,
    filter={"year": 2024, "field": "economics"}  # Filter by year and field
)
```

**Note**: Similar patterns work for finance (financial documents), political science (policy papers), and other large document collections.

## Pros and Cons

### Pros
- High performance and scalability
- Fully managed service
- Real-time updates
- Good for production economics research
- Reliable and stable

### Cons
- Paid service (free tier available)
- Requires internet connection
- Less control over infrastructure

## Pricing

- **Free Tier**: 1 index, 100K vectors
- **Starter**: $70/month, 1M vectors
- **Standard**: $140/month, 5M vectors

## Resources

- **Website**: [https://www.pinecone.io/](https://www.pinecone.io/)
- **Documentation**: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- **LangChain Integration**: [https://python.langchain.com/docs/integrations/vectorstores/pinecone](https://python.langchain.com/docs/integrations/vectorstores/pinecone)

