# Economics Literature Review Agent

## Overview

A complete economics literature review agent that automatically searches for economics papers, summarizes them, and synthesizes findings. This example demonstrates a real-world economics research application using agentic AI.

## Features

- **Economics Paper Search**: Search ArXiv and other sources for economics papers
- **Economics Paper Summarization**: Extract key information from economics papers
- **Economics Synthesis**: Combine findings across economics papers
- **Economics Gap Analysis**: Identify economics research gaps
- **Economics Report Generation**: Create comprehensive economics literature review

## Implementation: LangChain Version

### Setup

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import Chroma
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import arxiv
import os

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
```

### Tools

```python
def search_arxiv_economics_papers(query: str, max_results: int = 10) -> str:
    """Search ArXiv for economics research papers."""
    search = arxiv.Search(
        query=f"{query} AND (cat:econ.EM OR cat:econ.TH OR cat:econ.GN)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for paper in search.results():
        papers.append({
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "abstract": paper.summary,
            "published": str(paper.published),
            "arxiv_id": paper.entry_id.split('/')[-1]
        })
    
    # Format results
    result = "Found economics papers:\n\n"
    for i, p in enumerate(papers, 1):
        result += f"{i}. {p['title']}\n"
        result += f"   Authors: {', '.join(p['authors'][:3])}\n"
        result += f"   Abstract: {p['abstract'][:300]}...\n"
        result += f"   ArXiv ID: {p['arxiv_id']}\n\n"
    return result

def summarize_economics_paper(arxiv_id: str) -> str:
    """Summarize an economics paper by ArXiv ID."""
    try:
        loader = ArxivLoader(arxiv_id)
        documents = loader.load()
        
        if not documents:
            return f"Could not load economics paper {arxiv_id}"
        
        paper_text = documents[0].page_content
        
        # Use LLM to summarize economics paper
        summary_prompt = f"""Summarize this economics research paper. Extract:
        1. Main economics research question
        2. Econometric methodology
        3. Key economics findings
        4. Limitations
        5. Economics contributions
        
        Economics paper text:
        {paper_text[:4000]}  # Limit length
        """
        
        summary = llm.invoke(summary_prompt)
        return summary.content
    except Exception as e:
        return f"Error summarizing economics paper: {str(e)}"

def synthesize_economics_findings(summaries: str) -> str:
    """Synthesize findings from multiple economics paper summaries."""
    synthesis_prompt = f"""Synthesize these economics paper summaries into a coherent economics literature review.
    Identify:
    1. Common economics themes
    2. Conflicting economics findings
    3. Economics research gaps
    4. Future economics research directions
    
    Economics summaries:
    {summaries}
    """
    
    synthesis = llm.invoke(synthesis_prompt)
    return synthesis.content

# Create tools
tools = [
    Tool(
        name="SearchArXivEconomics",
        func=search_arxiv_economics_papers,
        description="Search ArXiv for economics research papers. Input: search query string, optionally max_results (default 10)"
    ),
    Tool(
        name="SummarizeEconomicsPaper",
        func=summarize_economics_paper,
        description="Summarize an economics research paper. Input: ArXiv ID (e.g., '2308.08155')"
    ),
    Tool(
        name="SynthesizeEconomicsFindings",
        func=synthesize_economics_findings,
        description="Synthesize findings from multiple economics papers. Input: concatenated economics paper summaries"
    )
]
```

### Agent

```python
# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an economics literature review assistant. Your task is to:
    1. Search for relevant economics research papers
    2. Summarize key economics papers
    3. Synthesize findings across economics papers
    4. Identify economics research gaps
    5. Generate a comprehensive economics literature review
    
    Use the available tools to search, summarize, and synthesize economics papers."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)

# Execute
result = agent_executor.invoke({
    "input": "Conduct an economics literature review on agentic AI applications in economics research. Find at least 5 recent economics papers, summarize them, and synthesize the findings."
})

print(result["output"])
```

**Note**: This pattern can be adapted for finance (financial literature review), political science (political economy literature), and other research disciplines.

## Usage Tips

1. **Start Small**: Begin with 3-5 economics papers, then scale up
2. **Refine Queries**: Adjust economics search queries based on initial results
3. **Review Summaries**: Check economics summaries for accuracy before synthesis
4. **Iterate**: Refine the economics literature review based on findings

## Common Issues

### Issue 1: Too Many Economics Papers
**Solution**: Limit initial economics search, filter by relevance

### Issue 2: Poor Economics Summaries
**Solution**: Use longer context windows, refine prompts

## Next Steps

- Explore [Econometric Data Analysis Agent](data-analysis-agent.md)
- Review [Patterns](../patterns/research-workflows.md) for economics workflow patterns
- Check [Frameworks](../frameworks/) for more framework examples

