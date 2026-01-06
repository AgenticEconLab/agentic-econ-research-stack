# CrewAI Getting Started

## Installation

### Basic Installation

```bash
pip install crewai
```

### With Tools

```bash
pip install crewai[tools]
```

### Full Installation

```bash
pip install crewai[tools] crewai[local]
```

## Environment Setup

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Your First Economics Research Crew

### Simple Two-Agent Crew

```python
from crewai import Agent, Task, Crew

# Define economics research agents
researcher = Agent(
    role='Economics Research Analyst',
    goal='Find and analyze relevant economics information',
    backstory='You are an expert economics researcher with years of experience in academic economics research.',
    verbose=True
)

writer = Agent(
    role='Economics Technical Writer',
    goal='Write clear and comprehensive economics research reports',
    backstory='You are a skilled technical writer specializing in economics research documentation.',
    verbose=True
)

# Define tasks
research_task = Task(
    description='Research the latest developments in agentic AI for economics research',
    agent=researcher
)

writing_task = Task(
    description='Write a comprehensive economics research report based on the research findings',
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

### Crew with Sequential Tasks for Economics Research

```python
from crewai import Agent, Task, Crew

# Define economics research agents
researcher = Agent(
    role='Economics Research Analyst',
    goal='Conduct thorough economics research on given topics',
    backstory='You are a meticulous economics researcher with expertise in academic literature review.',
    verbose=True
)

econometrician = Agent(
    role='Econometrician',
    goal='Analyze economics research findings and extract insights using econometric methods',
    backstory='You are an expert in econometric analysis and data interpretation.',
    verbose=True
)

writer = Agent(
    role='Economics Report Writer',
    goal='Synthesize economics research and analysis into comprehensive reports',
    backstory='You are a skilled technical writer who creates clear, well-structured economics research reports.',
    verbose=True
)

# Define sequential tasks
research_task = Task(
    description='Research agentic AI applications in behavioral economics',
    agent=researcher
)

analysis_task = Task(
    description='Analyze the economics research findings and identify key trends using econometric methods',
    agent=econometrician,
    context=[research_task]  # Depends on research_task
)

writing_task = Task(
    description='Write a comprehensive economics research report based on research and analysis',
    agent=writer,
    context=[research_task, analysis_task]  # Depends on both
)

# Create crew
crew = Crew(
    agents=[researcher, econometrician, writer],
    tasks=[research_task, analysis_task, writing_task],
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

## Basic Concepts

### 1. Agents

Agents have roles, goals, and backstories:

```python
agent = Agent(
    role='Economics Research Assistant',
    goal='Help with economics literature review',
    backstory='You are a PhD student specializing in economics.',
    verbose=True,
    allow_delegation=False  # Can this agent delegate tasks?
)
```

### 2. Tasks

Tasks define what agents should do:

```python
task = Task(
    description='Find relevant economics papers on agentic AI',
    agent=researcher,
    expected_output='A list of 10 relevant economics papers with summaries'
)
```

### 3. Crews

Crews orchestrate agents and tasks:

```python
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    verbose=True,
    process='sequential'  # or 'hierarchical'
)
```

## Common Pitfalls and Solutions

### Pitfall 1: Unclear Task Descriptions
**Problem**: Agents don't understand what to do
**Solution**: Write detailed, specific task descriptions with expected outputs

### Pitfall 2: Missing Context
**Problem**: Tasks don't have access to previous task results
**Solution**: Use `context` parameter to link tasks

### Pitfall 3: Agent Conflicts
**Problem**: Agents have overlapping or conflicting roles
**Solution**: Define clear, distinct roles and goals for each agent

### Pitfall 4: Infinite Loops
**Problem**: Agents keep delegating to each other
**Solution**: Set `allow_delegation=False` or use sequential process

## Next Steps

- Explore [Economics Research Examples](scientific-research-examples.md)
- Check out [Resources](resources.md) for advanced patterns
- Review [Multi-Agent Collaboration](../../patterns/multi-agent-collaboration.md) patterns

