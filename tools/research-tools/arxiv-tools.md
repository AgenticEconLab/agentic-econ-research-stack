# ArXiv Tools

## What is ArXiv?

ArXiv is a preprint server for research papers. ArXiv tools help agents search, retrieve, and analyze economics papers from ArXiv programmatically.

## Key Features

- **Economics Paper Search**: Search ArXiv by keywords, authors, categories (e.g., econ.EM, econ.TH)
- **Economics Paper Retrieval**: Download economics papers and metadata
- **Category Filtering**: Filter by economics research categories
- **Integration**: Easy integration with agent frameworks

## Installation

```bash
pip install arxiv
# or with LangChain
pip install langchain-arxiv
```

## Quick Start

### Basic Usage for Economics Papers

```python
import arxiv

# Search economics papers
search = arxiv.Search(
    query="agentic AI economics",
    max_results=10,
    sort_by=arxiv.SortCriterion.Relevance
)

# Retrieve economics papers
for paper in search.results():
    print(f"Title: {paper.title}")
    print(f"Authors: {paper.authors}")
    print(f"Abstract: {paper.summary[:200]}...")
    print(f"PDF: {paper.pdf_url}")
```

### With LangChain for Economics Papers

```python
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load economics paper by ArXiv ID
loader = ArxivLoader("2308.08155")
documents = loader.load()

# Split and process
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Use with vector store for economics papers
vectorstore = Chroma.from_documents(texts, embeddings)
```

### As Agent Tool for Economics Research

```python
from langchain.tools import Tool
import arxiv

def search_arxiv_economics(query: str, max_results: int = 5) -> str:
    """Search ArXiv for economics papers."""
    search = arxiv.Search(
        query=f"{query} AND (cat:econ.EM OR cat:econ.TH)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in search.results():
        results.append(f"Title: {paper.title}\nAbstract: {paper.summary[:300]}")
    return "\n\n".join(results)

# Create tool
arxiv_economics_tool = Tool(
    name="ArXivEconomicsSearch",
    func=search_arxiv_economics,
    description="Search ArXiv for economics research papers"
)

# Use with economics research agent
agent = create_openai_functions_agent(llm, [arxiv_economics_tool], prompt)
```

## Economics Research Use Cases

### Automated Economics Paper Discovery

```python
import arxiv
from langchain.agents import create_openai_functions_agent

def find_relevant_economics_papers(topic: str, max_results: int = 10) -> str:
    """Find relevant economics papers on a topic."""
    search = arxiv.Search(
        query=f"{topic} AND (cat:econ.EM OR cat:econ.TH)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for paper in search.results():
        papers.append({
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "abstract": paper.summary,
            "date": paper.published,
            "url": paper.entry_id
        })
    
    return format_economics_papers(papers)  # Your formatting function

tool = Tool(name="FindEconomicsPapers", func=find_relevant_economics_papers, description="Find economics research papers")
agent = create_openai_functions_agent(llm, [tool], prompt)
```

**Note**: This pattern works for finance (cat:q-fin), political science, and other ArXiv categories.

## Pros and Cons

### Pros
- Free and open access to economics papers
- Large collection of economics preprints
- Easy to use API
- Good for economics research
- LangChain integration

### Cons
- Limited to preprints
- No peer review status
- May have incomplete metadata

## Resources

- **ArXiv**: [https://arxiv.org/](https://arxiv.org/)
- **ArXiv API**: [https://arxiv.org/help/api](https://arxiv.org/help/api)
- **Economics Categories**: econ.EM (Econometrics), econ.TH (Theoretical Economics), econ.GN (General Economics)
- **Python Library**: [https://github.com/lukasschwab/arxiv.py](https://github.com/lukasschwab/arxiv.py)
- **LangChain Integration**: [https://python.langchain.com/docs/integrations/document_loaders/arxiv](https://python.langchain.com/docs/integrations/document_loaders/arxiv)

