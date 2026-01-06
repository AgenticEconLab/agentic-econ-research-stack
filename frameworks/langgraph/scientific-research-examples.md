# LangGraph Economics Research Examples

## Example 1: Iterative Economics Literature Review

Agent that refines search based on results:

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
import arxiv

class EconomicsReviewState(TypedDict):
    query: str
    papers: list
    iterations: int
    satisfied: bool

llm = ChatOpenAI(model="gpt-4")

def search_economics_papers(state: EconomicsReviewState):
    """Search for economics papers."""
    search = arxiv.Search(
        query=f"{state['query']} AND (cat:econ.EM OR cat:econ.TH)",
        max_results=5
    )
    papers = [{"title": p.title, "summary": p.summary} for p in search.results()]
    return {"papers": papers, "iterations": state["iterations"] + 1}

def evaluate_results(state: EconomicsReviewState):
    """Evaluate if we have enough good economics papers."""
    if len(state["papers"]) >= 10 or state["iterations"] >= 3:
        return {"satisfied": True}
    return {"satisfied": False}

def refine_query(state: EconomicsReviewState):
    """Refine economics search query."""
    response = llm.invoke(f"Refine this economics search query: {state['query']}")
    return {"query": response.content}

def should_continue(state: EconomicsReviewState) -> Literal["refine", "summarize", "end"]:
    """Route based on satisfaction."""
    if not state.get("satisfied", False) and state["iterations"] < 3:
        return "refine"
    elif state.get("satisfied", False):
        return "summarize"
    return "end"

def summarize_papers(state: EconomicsReviewState):
    """Summarize found economics papers."""
    papers_text = "\n".join([f"{p['title']}: {p['summary'][:200]}" for p in state["papers"]])
    response = llm.invoke(f"Summarize these economics papers:\n{papers_text}")
    return {"summary": response.content}

# Build graph
workflow = StateGraph(EconomicsReviewState)
workflow.add_node("search", search_economics_papers)
workflow.add_node("evaluate", evaluate_results)
workflow.add_node("refine", refine_query)
workflow.add_node("summarize", summarize_papers)

workflow.set_entry_point("search")
workflow.add_edge("search", "evaluate")
workflow.add_conditional_edges("evaluate", should_continue)
workflow.add_edge("refine", "search")  # Loop back
workflow.add_edge("summarize", END)

app = workflow.compile()
result = app.invoke({
    "query": "agentic AI economics",
    "papers": [],
    "iterations": 0,
    "satisfied": False
})
```

**Note**: This pattern can be adapted for finance (financial literature), political science (political economy), and other iterative research workflows.

## Best Practices

1. **Clear State Structure**: Define TypedDict for type safety
2. **Idempotent Nodes**: Nodes should be safe to re-run
3. **Termination Conditions**: Always have clear exit conditions for loops
4. **State Updates**: Return new state, don't mutate input
5. **Checkpointing**: Use checkpointing for long-running economics research workflows

## Next Steps

- Explore more examples in the [Examples](../../examples/) section
- Review [Resources](resources.md) for advanced patterns
- Check out [Research Workflows](../../patterns/research-workflows.md) patterns

