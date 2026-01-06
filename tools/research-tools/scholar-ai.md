# ScholarAI

## What is ScholarAI?

ScholarAI provides APIs and tools for accessing academic databases, searching papers, and retrieving metadata. It helps agents access economics research information programmatically.

## Key Features

- **Economics Paper Search**: Search across academic databases for economics papers
- **Metadata Retrieval**: Get economics paper metadata (title, authors, abstract)
- **Citation Information**: Access citation data for economics papers
- **Database Access**: Connect to various academic databases

## Installation

```bash
pip install scholar-ai
# or use APIs directly
```

## Quick Start

### Basic Usage for Economics Papers

```python
from scholar_ai import ScholarAI

# Initialize (may require API key)
scholar = ScholarAI(api_key="your-api-key")

# Search economics papers
results = scholar.search(
    query="agentic AI economics",
    limit=10
)

for paper in results:
    print(f"{paper.title} - {paper.authors}")
```

### With LangChain for Economics Research

```python
from langchain.tools import Tool
from scholar_ai import ScholarAI

scholar = ScholarAI(api_key="your-api-key")

def search_economics_papers(query: str) -> str:
    """Search for economics research papers."""
    results = scholar.search(query=query, limit=5)
    return "\n".join([f"{p.title}: {p.abstract[:200]}" for p in results])

# Create tool
economics_paper_search_tool = Tool(
    name="EconomicsPaperSearch",
    func=search_economics_papers,
    description="Search for economics research papers"
)

# Use with economics research agent
agent = create_openai_functions_agent(llm, [economics_paper_search_tool], prompt)
```

## Economics Research Use Cases

### Automated Economics Literature Search

```python
from scholar_ai import ScholarAI
from langchain.agents import create_openai_functions_agent

scholar = ScholarAI(api_key="your-api-key")

def comprehensive_economics_search(query: str) -> str:
    """Comprehensive economics paper search."""
    results = scholar.search(query=query, limit=20)
    summaries = []
    for paper in results:
        summaries.append(f"Title: {paper.title}\nAbstract: {paper.abstract[:300]}")
    return "\n\n".join(summaries)

tool = Tool(name="EconomicsLiteratureSearch", func=comprehensive_economics_search, description="Search academic economics papers")
agent = create_openai_functions_agent(llm, [tool], prompt)
```

**Note**: This pattern works for finance (financial papers), political science (policy papers), and other research disciplines.

## Alternative: Direct API Usage

Many academic databases have APIs:

```python
import requests

# Example: Semantic Scholar API for economics papers
def search_semantic_scholar_economics(query: str):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": f"{query} economics", "limit": 10}
    response = requests.get(url, params=params)
    return response.json()
```

## Pros and Cons

### Pros
- Access to academic databases for economics papers
- Programmatic economics paper search
- Metadata retrieval
- Can integrate with economics research agents

### Cons
- May require API keys
- Rate limits
- Varies by database

## Resources

- **Semantic Scholar API**: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
- **PubMed API**: [https://www.ncbi.nlm.nih.gov/books/NBK25497/](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- **Google Scholar**: Check for unofficial APIs or scraping tools

