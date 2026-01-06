# AutoGen Economics Research Examples

## Example 1: Economics Research Team with Specialized Roles

Create a team of agents for comprehensive economics research:

```python
import autogen
from autogen import GroupChat, GroupChatManager

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
    }
]

# Economic theorist agent - develops theoretical frameworks
theorist = autogen.AssistantAgent(
    name="theorist",
    llm_config={"config_list": config_list},
    system_message="""You are an economic theorist. Your role is to:
    1. Develop theoretical frameworks
    2. Formulate economic models
    3. Identify key economic mechanisms
    Be rigorous and cite economic theory."""
)

# Econometrician agent - analyzes data and statistics
econometrician = autogen.AssistantAgent(
    name="econometrician",
    llm_config={"config_list": config_list},
    system_message="""You are an econometrician. Your role is to:
    1. Analyze economic data statistically
    2. Perform econometric tests
    3. Estimate economic models
    Use Python code when needed."""
)

# Policy analyst agent - synthesizes findings for policy
policy_analyst = autogen.AssistantAgent(
    name="policy_analyst",
    llm_config={"config_list": config_list},
    system_message="""You are a policy analyst. Your role is to:
    1. Synthesize research findings
    2. Draw policy implications
    3. Create clear, structured reports
    Write professionally and clearly."""
)

# User proxy
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Create group chat
groupchat = GroupChat(
    agents=[theorist, econometrician, policy_analyst, user_proxy],
    messages=[],
    max_round=20
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Start economics research task
user_proxy.initiate_chat(
    manager,
    message="Research the impact of agentic AI on economic modeling. Create a comprehensive report with theory, empirical analysis, and policy implications."
)
```

**Note**: This pattern works for finance (theorist, quant analyst, risk manager), political science (theorist, data analyst, policy expert), and other multi-disciplinary research.

## Example 2: Economics Literature Review with Code Execution

Agent that searches economics papers and analyzes data:

```python
import autogen
import arxiv

def search_arxiv_economics(query: str, max_results: int = 5) -> str:
    """Search arXiv for economics papers."""
    search = arxiv.Search(
        query=f"{query} AND (cat:econ.EM OR cat:econ.TH)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in search.results():
        results.append(f"Title: {paper.title}\nAbstract: {paper.summary[:300]}")
    return "\n\n".join(results)

# Configure function calling
function_map = {
    "search_arxiv_economics": search_arxiv_economics
}

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
        "functions": [
            {
                "name": "search_arxiv_economics",
                "description": "Search arXiv for economics research papers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum number of results"}
                    },
                    "required": ["query"]
                }
            }
        ],
        "function_call_map": function_map
    }
]

# Create research agent
researcher = autogen.AssistantAgent(
    name="researcher",
    llm_config=config_list,
    system_message="You are an economics literature review assistant. Search for papers and analyze them."
)

# Create user proxy with code execution
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Start research
user_proxy.initiate_chat(
    researcher,
    message="Find and summarize recent papers on agentic AI in behavioral economics"
)
```

## Example 3: Economic Hypothesis Testing Workflow

Multiple agents testing an economics research hypothesis:

```python
import autogen
from autogen import GroupChat, GroupChatManager

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
    }
]

# Hypothesis generator for economics
hypothesis_agent = autogen.AssistantAgent(
    name="hypothesis_generator",
    llm_config={"config_list": config_list},
    system_message="""You generate testable economics research hypotheses based on theory and literature.
    Format: 'If [economic condition], then [economic outcome] because [economic mechanism]'"""
)

# Experimental designer for economics
designer = autogen.AssistantAgent(
    name="experimental_designer",
    llm_config={"config_list": config_list},
    system_message="""You design experiments to test economic hypotheses.
    Specify: variables, controls, methodology, identification strategy, expected outcomes."""
)

# Econometrician
econometrician = autogen.AssistantAgent(
    name="econometrician",
    llm_config={"config_list": config_list},
    system_message="""You analyze experimental results using econometric methods.
    Perform statistical tests, calculate p-values, interpret results in economic context."""
)

# User proxy
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=15,
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Group chat
groupchat = GroupChat(
    agents=[hypothesis_agent, designer, econometrician, user_proxy],
    messages=[],
    max_round=25
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Test hypothesis
user_proxy.initiate_chat(
    manager,
    message="Generate and test a hypothesis about agentic AI improving economics research productivity"
)
```

## Best Practices

1. **Define Clear Roles**: Each agent should have a specific, well-defined role in economics research
2. **Use System Messages**: Detailed system messages improve agent behavior
3. **Limit Rounds**: Set appropriate `max_round` to prevent infinite loops
4. **Human Oversight**: Use `human_input_mode="TERMINATE"` for critical decisions
5. **Code Execution**: Use Docker for isolation in production
6. **Cost Management**: Use cheaper models for simpler agent roles

## Next Steps

- Explore more examples in the [Examples](../../examples/) section
- Review [Resources](resources.md) for advanced patterns
- Check out [Multi-Agent Collaboration](../../patterns/multi-agent-collaboration.md) patterns

