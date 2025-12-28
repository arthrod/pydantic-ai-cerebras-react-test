There are several patterns for this. The cleanest depends on whether you want the "done" signal to be implicit (via structured output) or explicit (via a named tool).

## Pattern 1: Implicit Done via `output_type` (Recommended)

The model keeps calling tools until it decides to produce the structured output. This is the most idiomatic PydanticAI approach:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

class ScrapingResult(BaseModel):
    """Call this ONLY when you have found enough documents or exhausted all options."""
    documents_found: list[dict]
    total_agreements: int
    total_briefs: int
    search_queries_tried: list[str]
    reason_for_stopping: str

@dataclass
class Deps:
    http_client: httpx.AsyncClient
    max_documents: int = 100

agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=Deps,
    output_type=ScrapingResult,
    instructions="""
    You are a Yoruba legal document scraper. Keep searching and scraping until you have 
    collected enough documents OR you've exhausted reasonable search strategies.
    
    Do NOT conclude prematurely. Try multiple search queries, follow links, and explore 
    different sources before finishing.
    """,
)

@agent.tool
async def search_web(ctx: RunContext[Deps], query: str) -> list[dict]:
    """Search for Yoruba legal documents. Use Yoruba terms without diacritics."""
    # ... search implementation
    return [{"url": "...", "title": "...", "snippet": "..."}]

@agent.tool
async def scrape_page(ctx: RunContext[Deps], url: str) -> dict:
    """Scrape and extract content from a URL."""
    # ... scraping implementation
    return {"content": "...", "language": "yoruba", "doc_type": "agreement"}

@agent.tool
async def classify_document(ctx: RunContext[Deps], content: str) -> dict:
    """Classify if content is a legal agreement, brief, or other."""
    # ... classification implementation
    return {"type": "legal_agreement", "confidence": 0.92}

# The agent loops internally until it calls the output tool
result = await agent.run(
    "Find Yoruba legal agreements and briefs. Target: 10 of each.",
    deps=Deps(http_client=httpx.AsyncClient()),
)
print(result.output)  # ScrapingResult
```

## Pattern 2: Explicit "Finish" Tool with External Loop

When you need more control or want to inspect intermediate states:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from dataclasses import dataclass, field

class TaskStatus(BaseModel):
    is_complete: bool
    documents: list[dict] = []
    message: str = ""

@dataclass
class Deps:
    collected_docs: list[dict] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = ""

agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=Deps,
    output_type=str,  # Simple output, we track state in deps
    instructions="Keep searching until you call the finish tool.",
)

@agent.tool
async def search_web(ctx: RunContext[Deps], query: str) -> list[dict]:
    """Search for documents."""
    return [{"url": "example.com", "title": "Doc"}]

@agent.tool
async def save_document(ctx: RunContext[Deps], url: str, doc_type: str) -> str:
    """Save a found document to the collection."""
    ctx.deps.collected_docs.append({"url": url, "type": doc_type})
    return f"Saved. Total: {len(ctx.deps.collected_docs)} documents."

@agent.tool
async def finish(ctx: RunContext[Deps], reason: str) -> str:
    """Call this when you're done collecting documents.
    
    Args:
        reason: Why you're finishing (enough docs, exhausted searches, etc.)
    """
    ctx.deps.is_finished = True
    ctx.deps.finish_reason = reason
    return f"Finishing: {reason}"


async def run_until_done(prompt: str, max_iterations: int = 20) -> Deps:
    deps = Deps()
    messages = None
    
    for i in range(max_iterations):
        result = await agent.run(
            prompt if messages is None else "Continue your task.",
            deps=deps,
            message_history=messages,
        )
        messages = result.all_messages()
        
        if deps.is_finished:
            print(f"✅ Agent finished: {deps.finish_reason}")
            break
        
        print(f"🔄 Iteration {i+1}: {len(deps.collected_docs)} docs collected")
    
    return deps

# Usage
deps = await run_until_done("Find 20 Yoruba legal documents")
print(f"Collected {len(deps.collected_docs)} documents")
```

## Pattern 3: Using `iter()` for Fine-Grained Control

Step through the agent graph node-by-node:

```python
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun

agent = Agent(
    'google-gla:gemini-2.5-flash',
    output_type=ScrapingResult,
)

# ... register tools ...

async def run_with_monitoring(prompt: str):
    async with agent.iter(prompt, deps=Deps()) as run:
        async for node in run:
            # Inspect each step
            if Agent.is_call_tools_node(node):
                print(f"📞 Calling tools: {[p.tool_name for p in node.tool_calls]}")
            elif Agent.is_model_request_node(node):
                print("🤖 Model is thinking...")
            
            # You could inject conditions here to stop early
            # if some_condition:
            #     break
        
        return run.result

result = await run_with_monitoring("Find Yoruba legal docs")
```

## Pattern 4: Union Output for "Continue vs Done" Decision

Let the model explicitly choose to continue or finish:

```python
from pydantic import BaseModel
from typing import Annotated, Literal

class ContinueSearching(BaseModel):
    """Call this to continue searching. Explain your next steps."""
    type: Literal["continue"] = "continue"
    next_queries: list[str]
    reasoning: str

class FinishedSearching(BaseModel):
    """Call this when done collecting documents."""
    type: Literal["finished"] = "finished"
    documents: list[dict]
    summary: str

# Union output: model must call one of these
OutputType = ContinueSearching | FinishedSearching

agent = Agent(
    'google-gla:gemini-2.5-flash',
    output_type=OutputType,
)

async def agentic_loop(initial_prompt: str) -> FinishedSearching:
    messages = None
    prompt = initial_prompt
    
    while True:
        result = await agent.run(prompt, message_history=messages)
        messages = result.all_messages()
        
        match result.output:
            case ContinueSearching(next_queries=queries, reasoning=reason):
                print(f"🔄 Continuing: {reason}")
                print(f"   Next queries: {queries}")
                prompt = f"Continue with your plan. Queries to try: {queries}"
            
            case FinishedSearching() as final:
                print(f"✅ Done: {final.summary}")
                return final
```

## Which Pattern to Use?

| Pattern | Use When |
|---------|----------|
| **1. Implicit output_type** | Simple cases; let the model decide naturally when it's done |
| **2. Explicit finish tool + loop** | Need to track state across iterations, inspect progress |
| **3. `iter()` API** | Need to monitor/modify behavior at each graph node |
| **4. Union output** | Want model to explicitly reason about continue vs. stop |

For your Yoruba scraper, I'd recommend **Pattern 4** (Union output) since it forces the model to explicitly justify continuing or stopping, which gives you visibility into its decision-making process—very much in the ReAct spirit.