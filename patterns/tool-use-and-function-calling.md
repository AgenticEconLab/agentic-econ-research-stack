# Tool Use and Function Calling

## Overview

Tool use and function calling allow agents to interact with external systems, APIs, and tools. This pattern is essential for building capable economics research agents that can access economic data, perform econometric computations, and interact with economics research tools.

**Note**: These patterns can be adapted for finance (financial data tools), political science (policy data tools), and other research disciplines.

## Pattern 1: Simple Tool Integration for Economics Research

Basic tool integration with economics research agents.

### LangChain Example

```python
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent

def search_economics_papers(query: str) -> str:
    """Search for economics research papers."""
    # Implementation
    return "Found economics papers..."

# Create tool
economics_paper_tool = Tool(
    name="SearchEconomicsPapers",
    func=search_economics_papers,
    description="Search for economics research papers on a topic"
)

# Use with economics research agent
agent = create_openai_functions_agent(llm, [economics_paper_tool], prompt)
```

## Pattern 2: Multiple Tools for Economics Research

Agents with access to multiple economics research tools.

### Example: Economics Research Agent with Multiple Tools

```python
from langchain.tools import Tool

def search_economics_papers(query: str) -> str:
    """Search for economics research papers."""
    return "Economics papers found..."

def analyze_economic_data(data: str) -> str:
    """Analyze economics research data."""
    return "Economics analysis complete..."

def generate_economics_visualization(data: str) -> str:
    """Generate economics data visualization."""
    return "Economics visualization created..."

# Create multiple economics research tools
tools = [
    Tool(name="SearchEconomicsPapers", func=search_economics_papers, description="Search economics papers"),
    Tool(name="AnalyzeEconomicData", func=analyze_economic_data, description="Analyze economics data"),
    Tool(name="VisualizeEconomics", func=generate_economics_visualization, description="Create economics visualization")
]

# Economics research agent with multiple tools
agent = create_openai_functions_agent(llm, tools, prompt)
```

## Pattern 3: Economics Research-Specific Tools

Tools for economics research applications.

### Example: Economics Literature Review Tools

```python
from langchain.tools import Tool
import arxiv

def search_arxiv_economics(query: str, max_results: int = 5) -> str:
    """Search ArXiv for economics papers."""
    search = arxiv.Search(
        query=f"{query} AND (cat:econ.EM OR cat:econ.TH)",
        max_results=max_results
    )
    results = []
    for paper in search.results():
        results.append(f"{paper.title}: {paper.summary[:200]}")
    return "\n".join(results)

def get_economics_citations(paper_id: str) -> str:
    """Get economics paper citations."""
    # Implementation
    return "Economics citations found..."

# Economics research tools
economics_research_tools = [
    Tool(name="ArXivEconomicsSearch", func=search_arxiv_economics, description="Search ArXiv for economics papers"),
    Tool(name="GetEconomicsCitations", func=get_economics_citations, description="Get economics paper citations")
]

agent = create_openai_functions_agent(llm, economics_research_tools, prompt)
```

## Best Practices

### 1. Clear Tool Descriptions
- Write detailed, clear descriptions for economics tools
- Include parameter information
- Specify expected economics research outputs

### 2. Error Handling
- Always handle errors gracefully
- Return meaningful error messages
- Log errors for debugging

## Next Steps

- Review [Research Workflows](research-workflows.md) for economics workflow patterns
- Check [Tools](../tools/) for specific economics tool documentation
- Explore [Examples](../examples/) for complete economics research implementations

