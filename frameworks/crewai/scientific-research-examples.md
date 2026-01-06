# CrewAI Economics Research Examples

## Example 1: Economics Literature Review Crew

A team of agents for comprehensive economics literature review:

```python
from crewai import Agent, Task, Crew

# Economics research agent - finds papers
researcher = Agent(
    role='Economics Literature Researcher',
    goal='Find and collect relevant economics research papers',
    backstory='You are an expert in academic economics literature search with deep knowledge of economics research databases.',
    verbose=True
)

# Summarizer agent - summarizes papers
summarizer = Agent(
    role='Economics Paper Summarizer',
    goal='Create concise summaries of economics research papers',
    backstory='You are skilled at extracting key information from economics academic papers.',
    verbose=True
)

# Synthesizer agent - synthesizes findings
synthesizer = Agent(
    role='Economics Research Synthesizer',
    goal='Synthesize economics research findings into coherent insights',
    backstory='You are an expert at identifying patterns and connections across multiple economics research papers.',
    verbose=True
)

# Define tasks
research_task = Task(
    description='Search for and collect 10 recent economics papers on agentic AI in economics research',
    agent=researcher,
    expected_output='A list of 10 economics papers with titles, authors, and abstracts'
)

summarize_task = Task(
    description='Summarize each economics paper, highlighting key findings and methodologies',
    agent=summarizer,
    context=[research_task],
    expected_output='Detailed summaries for each economics paper'
)

synthesize_task = Task(
    description='Synthesize the economics findings into a comprehensive literature review highlighting trends and gaps',
    agent=synthesizer,
    context=[research_task, summarize_task],
    expected_output='A comprehensive economics literature review document'
)

# Create crew
crew = Crew(
    agents=[researcher, summarizer, synthesizer],
    tasks=[research_task, summarize_task, synthesize_task],
    verbose=True,
    process='sequential'
)

# Execute
result = crew.kickoff()
print(result)
```

**Note**: This pattern can be adapted for finance (financial literature review), political science (political economy literature), and other disciplines.

## Example 2: Econometric Analysis Team

Specialized agents for econometric data analysis:

```python
from crewai import Agent, Task, Crew

# Data loader
data_loader = Agent(
    role='Economic Data Engineer',
    goal='Load and prepare economic datasets for analysis',
    backstory='You are an expert in economic data engineering and ETL processes.',
    verbose=True
)

# Econometrician
econometrician = Agent(
    role='Econometrician',
    goal='Perform econometric analysis on economic datasets',
    backstory='You are a PhD econometrician with expertise in time series, panel data, and causal inference.',
    verbose=True
)

# Visualizer
visualizer = Agent(
    role='Economic Data Visualizer',
    goal='Create informative visualizations for economic data',
    backstory='You are an expert in economic data visualization and storytelling with data.',
    verbose=True
)

# Report writer
report_writer = Agent(
    role='Economics Analysis Report Writer',
    goal='Write comprehensive econometric analysis reports',
    backstory='You are a technical writer specializing in econometric analysis reports.',
    verbose=True
)

# Define tasks
load_task = Task(description='Load the economic dataset and perform initial data cleaning', agent=data_loader)
analyze_task = Task(description='Perform econometric analysis: correlations, regressions, time series tests', agent=econometrician, context=[load_task])
visualize_task = Task(description='Create visualizations: charts, graphs, and dashboards', agent=visualizer, context=[load_task, analyze_task])
write_task = Task(description='Write comprehensive econometric analysis report', agent=report_writer, context=[load_task, analyze_task, visualize_task])

# Create crew
crew = Crew(
    agents=[data_loader, econometrician, visualizer, report_writer],
    tasks=[load_task, analyze_task, visualize_task, write_task],
    verbose=True,
    process='sequential'
)

result = crew.kickoff()
```

## Best Practices

1. **Clear Role Definitions**: Each agent should have a distinct, well-defined role in economics research
2. **Specific Goals**: Goals should be clear and measurable
3. **Detailed Backstories**: Rich backstories help agents understand their purpose
4. **Task Context**: Use context to link dependent tasks
5. **Expected Outputs**: Always specify expected outputs for tasks
6. **Sequential Process**: Use sequential process for dependent tasks

## Next Steps

- Explore more examples in the [Examples](../../examples/) section
- Review [Resources](resources.md) for advanced patterns
- Check out [Multi-Agent Collaboration](../../patterns/multi-agent-collaboration.md) patterns

