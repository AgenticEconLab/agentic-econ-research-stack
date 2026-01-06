# Econometric Data Analysis Agent

## Overview

A complete econometric data analysis agent that loads economics datasets, performs econometric analysis, creates visualizations, and generates insights. This example demonstrates how to build agents for data-driven economics research.

## Features

- **Economics Data Loading**: Load economics datasets from various sources
- **Economics Data Cleaning**: Handle missing values, outliers, and errors in economics data
- **Econometric Analysis**: Perform correlations, regressions, time series tests
- **Economics Visualization**: Create charts and graphs for economics data
- **Economics Insight Generation**: Generate economics research insights

## Implementation: LangChain Version

### Setup

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### Tools

```python
def load_economics_dataset(file_path: str) -> str:
    """Load an economics dataset from file."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file format: {file_path}"
        
        # Store in global variable for other tools
        global current_economics_dataset
        current_economics_dataset = df
        
        return f"Loaded economics dataset: {len(df)} rows, {len(df.columns)} columns\nColumns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        return f"Error loading economics dataset: {str(e)}"

def clean_economics_dataset(description: str) -> str:
    """Clean the loaded economics dataset."""
    global current_economics_dataset
    if 'current_economics_dataset' not in globals():
        return "No economics dataset loaded. Load a dataset first."
    
    df = current_economics_dataset.copy()
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()  # Or use fillna() with strategy
    
    # Remove outliers (simple IQR method for numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    current_economics_dataset = df
    
    return f"Cleaned economics dataset: {initial_rows} -> {len(df)} rows. Removed duplicates, nulls, and outliers."

def analyze_economics_dataset(description: str) -> str:
    """Perform econometric analysis on the economics dataset."""
    global current_economics_dataset
    if 'current_economics_dataset' not in globals():
        return "No economics dataset loaded. Load a dataset first."
    
    df = current_economics_dataset
    analysis = []
    
    # Basic statistics
    analysis.append("=== Basic Economics Statistics ===")
    analysis.append(str(df.describe()))
    
    # Correlations (if numeric columns exist)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        analysis.append("\n=== Economic Variable Correlations ===")
        corr = df[numeric_cols].corr()
        analysis.append(str(corr))
    
    return "\n".join(analysis)

def generate_economics_insights(analysis: str) -> str:
    """Generate economics research insights from analysis."""
    insight_prompt = f"""Based on this econometric analysis, generate economics research insights:
    1. Key economics findings
    2. Economic patterns and trends
    3. Surprising economics results
    4. Economics research implications
    5. Economics policy recommendations
    
    Econometric analysis results:
    {analysis}
    """
    
    insights = llm.invoke(insight_prompt)
    return insights.content

# Create tools
tools = [
    Tool(
        name="LoadEconomicsDataset",
        func=load_economics_dataset,
        description="Load an economics dataset from file. Input: file path (CSV or Excel)"
    ),
    Tool(
        name="CleanEconomicsDataset",
        func=clean_economics_dataset,
        description="Clean the loaded economics dataset (remove duplicates, nulls, outliers). Input: description of cleaning needed"
    ),
    Tool(
        name="AnalyzeEconomicsDataset",
        func=analyze_economics_dataset,
        description="Perform econometric analysis on the economics dataset. Input: description of analysis needed"
    ),
    Tool(
        name="GenerateEconomicsInsights",
        func=generate_economics_insights,
        description="Generate economics research insights from analysis. Input: econometric analysis results"
    )
]
```

### Agent

```python
# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an econometric analysis assistant. Your task is to:
    1. Load economics research datasets
    2. Clean and preprocess economics data
    3. Perform econometric analysis
    4. Create economics visualizations
    5. Generate economics research insights
    
    Use the available tools to analyze economics data and generate insights."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# Execute
result = agent_executor.invoke({
    "input": "Analyze the economics dataset at 'economics_data.csv': load it, clean it, perform econometric analysis, and generate insights."
})

print(result["output"])
```

**Note**: This pattern can be adapted for finance (financial data analysis), political science (political data analysis), and other quantitative research disciplines.

## Usage Tips

1. **Economics Data Quality**: Ensure economics data is in good format before loading
2. **Econometric Analysis Scope**: Start with basic econometric analysis, then expand
3. **Economics Visualization**: Choose appropriate chart types for economics data
4. **Economics Insights**: Review and validate generated economics insights

## Common Issues

### Issue 1: Economics Data Loading Errors
**Solution**: Check file format, path, and permissions

### Issue 2: Econometric Analysis Quality
**Solution**: Validate econometric methods, check assumptions

## Next Steps

- Explore [Economics Hypothesis Testing Agent](hypothesis-testing-agent.md)
- Review [Patterns](../patterns/research-workflows.md) for economics workflow patterns
- Check [Tools](../tools/) for economics data analysis tools

