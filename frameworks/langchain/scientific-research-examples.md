# LangChain Economics Research Examples

## Example 1: Economics Literature Review Agent

Automate discovery and analysis of economics papers:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import arxiv

def search_arxiv_economics(query: str, max_results: int = 5) -> str:
    """Search arXiv for economics papers matching the query."""
    # Add economics category filter
    search = arxiv.Search(
        query=f"{query} AND (cat:econ.EM OR cat:econ.TH OR cat:econ.GN)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in search.results():
        results.append(f"Title: {paper.title}\nAbstract: {paper.summary[:500]}...")
    return "\n\n".join(results)

# Create tools
tools = [
    Tool(
        name="ArXivEconomicsSearch",
        func=search_arxiv_economics,
        description="Search arXiv for economics research papers. Input: search query string"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an economics literature review assistant. Help researchers find and summarize relevant economics papers."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use the agent
result = agent_executor.invoke({
    "input": "Find recent papers on agentic AI applications in behavioral economics research"
})
print(result["output"])
```

**Note**: This pattern can be adapted for other disciplines by changing the arXiv category filters (e.g., `cat:q-fin` for finance, `cat:stat` for statistics).

## Example 2: Econometric Data Analysis Agent

Agent that can analyze economic datasets and generate insights:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import numpy as np
from scipy import stats

def analyze_economic_dataframe(df_description: str, operation: str) -> str:
    """Perform econometric analysis on a dataframe.
    
    Args:
        df_description: Description of the dataframe structure
        operation: Operation to perform (correlation, regression, time_series, panel)
    """
    # In practice, you'd load the actual dataframe
    if operation == "correlation":
        return "Correlation matrix showing relationships between economic variables"
    elif operation == "regression":
        return "Regression results with coefficients, p-values, and R-squared"
    elif operation == "time_series":
        return "Time series analysis: trends, seasonality, stationarity tests"
    elif operation == "panel":
        return "Panel data analysis: fixed effects, random effects, Hausman test"
    return "Analysis complete"

def generate_economics_visualization(df_description: str, chart_type: str) -> str:
    """Generate a visualization code snippet for economic data."""
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns
# {chart_type} visualization for economic data
# Examples: time series plots, scatter plots for economic relationships, heatmaps for correlations
plt.show()
"""

# Create tools
tools = [
    Tool(
        name="EconometricAnalysis",
        func=analyze_economic_dataframe,
        description="Perform econometric analysis on datasets. Input: dataframe description and operation type (correlation, regression, time_series, panel)"
    ),
    Tool(
        name="EconomicsVisualization",
        func=generate_economics_visualization,
        description="Generate Python code for economic data visualizations. Input: dataframe description and chart type"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an econometric analysis assistant. Help researchers analyze economic datasets and create visualizations."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Analyze a dataset with GDP, inflation, and unemployment data. Perform correlation analysis and create a time series plot"
})
print(result["output"])
```

**Note**: Similar patterns work for finance (financial data analysis), political science (voting data), and other quantitative disciplines.

## Example 3: RAG-Based Economics Paper Q&A

Question-answering system over economics research papers:

```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load economics papers
loader = DirectoryLoader("./economics_papers/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create custom prompt for economics research
template = """Use the following pieces of context from economics research papers to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: Provide a detailed answer based on the context, and cite which economics papers support your answer."""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Ask questions about economics
query = "What are the main findings about market efficiency in behavioral economics?"
result = qa_chain({"query": query})

print(f"Answer: {result['result']}")
print(f"\nSources: {len(result['source_documents'])} papers referenced")
```

## Example 4: Multi-Step Economics Research Workflow

Chain multiple operations for complex economics research tasks:

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Step 1: Generate research questions for economics
question_prompt = ChatPromptTemplate.from_template(
    "Given an economics research topic: {topic}, generate 5 specific research questions."
)
question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="questions")

# Step 2: Find relevant economics papers
paper_prompt = ChatPromptTemplate.from_template(
    "For these economics research questions: {questions}, suggest 3 relevant papers for each question."
)
paper_chain = LLMChain(llm=llm, prompt=paper_prompt, output_key="papers")

# Step 3: Create economics research plan
plan_prompt = ChatPromptTemplate.from_template(
    "Based on these economics papers: {papers}, create a detailed economics research plan with methodology."
)
plan_chain = LLMChain(llm=llm, prompt=plan_prompt, output_key="plan")

# Chain them together
overall_chain = SimpleSequentialChain(
    chains=[question_chain, paper_chain, plan_chain],
    verbose=True
)

result = overall_chain.run("Agentic AI applications in behavioral economics")
print(result)
```

## Example 5: Code Generation for Econometric Analysis

Agent that generates and executes econometric research code:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import subprocess
import tempfile
import os

def execute_econometric_code(code: str) -> str:
    """Execute econometric Python code and return the output."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return f"Execution successful:\n{result.stdout}"
        else:
            return f"Error:\n{result.stderr}"
    except Exception as e:
        return f"Execution error: {str(e)}"

# Create tools
tools = [
    Tool(
        name="EconometricCodeExecutor",
        func=execute_econometric_code,
        description="Execute Python code for econometric analysis. Input: Python code as a string"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an econometric code generation assistant. Generate and execute Python code for econometric data analysis."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Generate Python code to calculate correlation between GDP growth and inflation from sample economic data"
})
print(result["output"])
```

## Best Practices for Economics Research

1. **Use Specific Tool Descriptions**: Clear descriptions help agents use tools correctly for economic analysis
2. **Implement Error Handling**: Always handle tool execution errors gracefully
3. **Monitor Token Usage**: Economics research agents can be verbose; track costs
4. **Use Appropriate Chunk Sizes**: For RAG with economics papers, balance context vs. retrieval quality
5. **Validate Tool Outputs**: Check tool results before passing to LLM, especially for econometric calculations

## Adapting to Other Disciplines

These examples can be adapted for:
- **Finance**: Change search queries to finance papers, use financial data analysis tools
- **Political Science**: Adapt for political economy, voting data analysis
- **Sociology**: Modify for social economics, survey data analysis
- **Public Health**: Adapt for health economics, policy evaluation

## Next Steps

- Explore more examples in the [Examples](../../examples/) section
- Review [Resources](resources.md) for advanced patterns
- Check out [Patterns](../../patterns/) for common economics research workflows

