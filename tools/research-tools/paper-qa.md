# PaperQA

## What is PaperQA?

PaperQA is a tool for question-answering over research papers. It automatically retrieves relevant papers, extracts information, and answers questions based on the paper content. Perfect for economics research papers.

## Key Features

- **Automatic Economics Paper Retrieval**: Finds relevant economics papers automatically
- **Question Answering**: Answers questions based on economics paper content
- **Citation Support**: Provides citations for answers
- **Multiple Economics Papers**: Can answer across multiple economics papers

## Installation

```bash
pip install paper-qa
```

## Quick Start

### Basic Usage with Economics Papers

```python
from paperqa import Docs

# Initialize
docs = Docs()

# Add economics papers (by DOI, ArXiv ID, or text)
docs.add("10.1038/nature12373")  # By DOI
docs.add("arxiv:2308.08155")  # By ArXiv ID (economics paper)

# Ask questions about economics
answer = docs.query("What is the main finding about behavioral economics?")
print(answer.answer)
print(answer.citations)
```

### With Agent Integration for Economics Research

```python
from langchain.tools import Tool
from paperqa import Docs

docs = Docs()
docs.add("arxiv:2308.08155")  # Economics paper

def economics_paper_qa(query: str) -> str:
    """Answer questions about economics research papers."""
    answer = docs.query(query)
    return f"{answer.answer}\n\nCitations: {answer.citations}"

# Create tool for economics research agent
paper_tool = Tool(
    name="EconomicsPaperQA",
    func=economics_paper_qa,
    description="Answer questions about economics research papers"
)

# Use with economics research agent
agent = create_openai_functions_agent(llm, [paper_tool], prompt)
```

## Economics Research Use Cases

### Economics Literature Q&A Agent

```python
from paperqa import Docs
from langchain.agents import create_openai_functions_agent

# Initialize with multiple economics papers
docs = Docs()
docs.add("arxiv:2308.08155")  # Economics paper 1
docs.add("arxiv:2303.11366")  # Economics paper 2

# Create Q&A function for economics
def economics_research_qa(query: str) -> str:
    answer = docs.query(query)
    return f"Answer: {answer.answer}\nSources: {', '.join(answer.citations)}"

# Integrate with economics research agent
tool = Tool(name="EconomicsResearchQA", func=economics_research_qa, description="Answer economics research questions")
agent = create_openai_functions_agent(llm, [tool], prompt)
```

**Note**: This pattern works for finance (financial papers), political science (policy papers), and other research disciplines.

## Pros and Cons

### Pros
- Easy to use
- Automatic economics paper retrieval
- Good citation support
- Works with multiple economics papers

### Cons
- Requires economics paper access
- May be slow for many economics papers
- Limited to papers it can access

## Resources

- **GitHub**: [https://github.com/whitead/paper-qa](https://github.com/whitead/paper-qa)
- **Documentation**: Check GitHub README

