"""
Four agentic loop patterns for comparison testing.
Each pattern receives the same prompt and produces a standardized result.

Features:
- DuckDuckGo search integration
- Proper logging with loguru
- Time to completion tracking
- Cerebras model with zai-glm-4.6

Run with: uv run python agent_patterns.py
"""

import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Union
import uvloop
from ddgs import DDGS
import logging
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.output import ToolOutput

PROMPTS: list[str] = [
    # Employment-focused (mix of BR and US)
    'Collect 4 employment documents: 2 Brazilian (1 CLT employment contract, 1 TST labor appeal brief) and 2 American (1 at-will employment agreement, 1 EEOC discrimination complaint). Each must be ≥3 pages. Output as PDF with filenames following pattern: {jurisdiction}_{doc_type}_{date}.pdf',

    'Collect 4 labor/employment docs: 2 from TST or TRT (any region) published 2022-2024, 2 from PACER (SDNY or NDCA) employment disputes. Include case numbers in filenames.',

    # Contract-focused
    'Collect 4 commercial contracts: 2 Brazilian (1 prestação de serviços, 1 compra e venda) and 2 American (1 SaaS agreement, 1 asset purchase agreement). Minimum 5 pages each. Source from public filings (CVM or SEC EDGAR) only.',

    'Collect 4 lease agreements: 2 Brazilian commercial leases (locação não-residencial) from São Paulo registry, 2 American commercial leases from SEC 10-K/10-Q exhibits. Each ≥10 pages.',

    # Brief-focused
    'Collect 4 court briefs: 2 Brazilian petições iniciais from TJSP e-SAJ (cível), 2 American motions to dismiss from PACER (any federal district). Each ≥8 pages. Include docket numbers.',

    'Collect 4 civil litigation briefs: 2 Brazilian (1 contestação, 1 recurso de apelação) from any TJ, 2 American (1 summary judgment motion, 1 opposition brief) from PACER. Cite case numbers.',

    # Opinion-focused
    'Collect 4 legal opinions: 2 Brazilian pareceres jurídicos (1 from PGE/PGR if available, 1 from law firm published in academic journal), 2 American opinion letters (from SEC no-action letters database). Each ≥4 pages.',

    'Collect 4 corporate/tax opinions: 2 Brazilian (CVM or RFB consultas), 2 American (IRS PLRs or SEC staff legal bulletins). Include document reference numbers.',

    # Real estate / property
    'Collect 4 real estate documents: 2 Brazilian (1 escritura de compra e venda, 1 contrato de locação comercial) from cartório models or CVM filings, 2 American (1 deed, 1 commercial lease) from SEC EDGAR exhibits. ≥5 pages each.',

    'Collect 4 property documents: 2 Brazilian imóveis documents from TJSP jurisprudência (ações possessórias), 2 American real property disputes from state court databases (NY or CA). Include citation format.',

    # Tax-related
    'Collect 4 tax documents: 2 Brazilian (1 CARF acórdão, 1 mandado de segurança tributário from TRF) and 2 American (1 Tax Court opinion, 1 IRS Chief Counsel Advice). Each ≥6 pages with citation.',

    'Collect 4 tax compliance docs: 2 Brazilian (ICMS-related from SEFAZ consultation or CONFAZ), 2 American (IRS Revenue Rulings or Procedures from 2020-2024). Include official publication numbers.',

    # Corporate / M&A
    'Collect 4 M&A documents from public filings only: 2 Brazilian (CVM fatos relevantes or acquisition agreements from Form 6-K), 2 American (SEC merger proxy statements or 8-K acquisition exhibits). ≥15 pages each.',

    'Collect 4 securities documents: 2 Brazilian (1 prospecto de oferta pública, 1 CVM auto de infração), 2 American (1 S-1 registration, 1 SEC enforcement action). Include CVM/SEC file numbers.',

    # Mixed queries with specific requirements
    'Collect exactly: 1 Brazilian service contract (≥4pp), 1 American MSA (≥6pp), 1 Brazilian TJSP civil brief (≥5pp), 1 American SDNY motion (≥8pp). All from 2020-2024. Output as separate PDFs with source URLs documented.',

    'Collect 4 documents from São Paulo or New York only: 2 from TJSP (1 sentença, 1 acórdão), 2 from NY state courts (1 trial court decision, 1 appellate decision). Include full citations in Bluebook/ABNT format.',

    # Consumer / civil
    'Collect 4 consumer protection docs: 2 Brazilian (PROCON administrative decisions or CDC-related TJSP cases), 2 American (CFPB enforcement actions or FTC complaints). Each ≥5 pages with case/matter numbers.',

    'Collect 4 damages/indemnification docs: 2 Brazilian (indenização decisions from STJ or TJSP), 2 American (personal injury complaints or insurance coverage opinions). Include monetary values discussed if available.',

    # Compliance / regulatory
    'Collect 4 compliance documents: 2 Brazilian (1 LGPD-related ANPD decision or guidance, 1 anticorruption agreement/TAC from CGU/MPF), 2 American (1 SEC enforcement, 1 DOJ FCPA resolution). Include official reference numbers.',

    'Collect 4 documents and summarize each in ≤100 words: 2 Brazilian regulatory (CVM, BACEN, or ANPD), 2 American regulatory (SEC, CFTC, or FTC). Summary must include: jurisdiction, document type, parties, key holding, date.',
]
import logfire as logger
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================



# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def get_model() -> CerebrasModel:
    """Create the Cerebras model with reasoning disabled."""
    return CerebrasModel(
        "zai-glm-4.6",
        settings=CerebrasModelSettings(cerebras_disable_reasoning=True),
    )


# =============================================================================
# STANDARDIZED OUTPUT (same for all patterns)
# =============================================================================

class AgentResult(BaseModel):
    """Standardized result across all patterns."""
    items_collected: list[dict[str, Any]]
    actions_taken: list[str]
    iterations: int
    finish_reason: str
    elapsed_seconds: float


# =============================================================================
# SHARED DEPENDENCIES
# =============================================================================

@dataclass
@dataclass
class Deps:
    """Shared state across tool calls."""
    items: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    target_count: int = 5
    pattern_name: str = "unknown"

    # Pattern 2 specific
    is_finished: bool = False
    finish_reason: str = ""

    def log(self) -> Any:
        """Get a logger bound to this pattern."""
        return logger.bind(pattern=self.pattern_name)


# =============================================================================
# DUCKDUCKGO SEARCH IMPLEMENTATION
# =============================================================================

def ddg_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Perform a DuckDuckGo search synchronously.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, href, body
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:200],
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"DDGS search failed for '{query}': {e}")
        return []


# =============================================================================
# SHARED TOOL IMPLEMENTATIONS
# =============================================================================

async def _search_impl(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Search using DuckDuckGo."""
    log = ctx.deps.log()
    log.info("🔍 Searching: '%s'", query)

    # Run sync DDGS in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, ddg_search, query, 5)

    ctx.deps.actions.append(f"search:{query}")
    log.debug(f"   Found {len(results)} results")  # noqa: G004
    return results


async def _fetch_impl(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Simulated fetch - in real impl would fetch URL content."""
    log = ctx.deps.log()
    log.info(f"📥 Fetching: {url[:50]}...")  # noqa: G004

    await asyncio.sleep(0.1)  # Simulate network delay
    ctx.deps.actions.append(f"fetch:{url[:30]}")

    return {
        "url": url,
        "content": f"Content from {url}",
        "status": "success",
    }


async def _save_impl(ctx: RunContext[Deps], item_id: str, content: str) -> str:  # noqa: RUF029
    """Save an item to the collection."""
    log = ctx.deps.log()

    ctx.deps.items.append({"id": item_id, "content": content[:100]})
    ctx.deps.actions.append(f"save:{item_id[:20]}")

    current = len(ctx.deps.items)
    target = ctx.deps.target_count
    log.info(f"💾 Saved '{item_id[:30]}...' | Progress: {current}/{target}")  # noqa: G004

    return f"Saved. Progress: {current}/{target} items."


# =============================================================================
# PATTERN 1: Implicit Done via output_type
# =============================================================================

class Pattern1Result(BaseModel):
    """Final output - model calls this tool when done."""
    items_collected: list[dict[str, Any]] = Field(description="All collected items")
    search_queries_used: list[str] = Field(description="Queries that were tried")
    reason_for_stopping: str = Field(description="Why you decided to stop")


pattern1_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions="""
    You are a data collector. Your goal is to collect items by:
    1. Searching for relevant items using DuckDuckGo
    2. Fetching promising results
    3. Saving good items

    Keep going until you've collected enough items (check save responses for progress)
    OR you've exhausted reasonable search strategies.

    Be thorough - try multiple search queries before concluding.
    """,
)


@pattern1_agent.tool(strict=True)
async def _search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Search DuckDuckGo for items. Returns list of {title, url, snippet}."""
    return await _search_impl(ctx, query)


@pattern1_agent.tool(strict=True)
async def _fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch content from a URL."""
    return await _fetch_impl(ctx, url)


@pattern1_agent.tool(strict=True)
async def _save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save an item to collection. Returns progress."""
    return await _save_impl(ctx, item_id, content)


async def run_pattern1(prompt: str, target: int = 5) -> AgentResult:
    """Pattern 1: Model loops internally until it produces structured output."""
    deps = Deps(target_count=target, pattern_name="P1-implicit")
    log = deps.log()

    log.info("Starting Pattern 1: Implicit output_type")
    start = time.perf_counter()

    result = await pattern1_agent.run(prompt, deps=deps)

    elapsed = time.perf_counter() - start
    log.success(f"Completed in {elapsed:.2f}s | Items: {len(deps.items)}")

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=1,
        finish_reason=result.output.reason_for_stopping,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 2: Explicit "finish" tool with external loop
# =============================================================================

pattern2_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=str,
    instructions="""
    You are a data collector. Search DuckDuckGo, fetch, and save items.

    Call the 'finish' tool ONLY when:
    - You've collected enough items, OR
    - You've exhausted all reasonable search strategies

    Check save responses for your progress toward the target.
    """,
)


@pattern2_agent.tool(strict=True)
async def __search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Search DuckDuckGo for items."""
    return await _search_impl(ctx, query)


@pattern2_agent.tool(strict=True)
async def __fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch content from a URL."""
    return await _fetch_impl(ctx, url)


@pattern2_agent.tool(strict=True)
async def ____save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save an item. Returns progress."""
    return await _save_impl(ctx, item_id, content)


@pattern2_agent.tool(strict=True)
async def finish(ctx: RunContext[Deps], reason: str) -> str:  # noqa: RUF029
    """Call this when you're done collecting.

    Args:
        reason: Why you're finishing (e.g., "collected enough", "no more results")
    """
    log = ctx.deps.log()
    log.info("🏁 Finish called: %s", reason)

    ctx.deps.is_finished = True
    ctx.deps.finish_reason = reason
    return f"Marked as finished: {reason}"


async def run_pattern2(prompt: str, target: int = 5, max_iterations: int = 10) -> AgentResult:
    """Pattern 2: External loop checking for explicit finish tool call."""
    deps = Deps(target_count=target, pattern_name="P2-explicit")
    log = deps.log()

    log.info("Starting Pattern 2: Explicit finish tool")
    start = time.perf_counter()

    messages = None
    iterations = 0

    for i in range(max_iterations):
        iterations += 1
        log.debug("Iteration %s/%s", iterations, max_iterations)

        current_prompt = prompt if messages is None else "Continue your task."

        result = await pattern2_agent.run(
            current_prompt,
            deps=deps,
            message_history=messages,
        )
        messages = result.all_messages()

        if deps.is_finished:
            break

    elapsed = time.perf_counter() - start
    log.success(f"Completed in {elapsed:.2f}s | Iterations: {iterations} | Items: {len(deps.items)}")

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=deps.finish_reason or "max_iterations_reached",
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 3: iter() API for fine-grained control
# =============================================================================

pattern3_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions="""
    You are a data collector. Search DuckDuckGo, fetch, and save items until you have enough.
    Be thorough with your searches before concluding.
    """,
)


@pattern3_agent.tool(strict=True)
async def ___search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Search DuckDuckGo for items."""
    return await _search_impl(ctx, query)


@pattern3_agent.tool(strict=True)
async def ___fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch content from a URL."""
    return await _fetch_impl(ctx, url)


@pattern3_agent.tool(strict=True)
async def __save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save an item. Returns progress."""
    return await _save_impl(ctx, item_id, content)


async def run_pattern3(prompt: str, target: int = 5) -> AgentResult:
    """Pattern 3: Using iter() to step through and monitor each node."""
    deps = Deps(target_count=target, pattern_name="P3-iter")
    log = deps.log()

    log.info("Starting Pattern 3: iter() API")
    start = time.perf_counter()

    node_count = 0

    async with pattern3_agent.iter(prompt, deps=deps) as run:
        async for node in run:
            node_count += 1
            try:
                node_type = node.__class__.__name__

                if Agent.is_call_tools_node(node):
                    # Safely extract tool names without assuming a single attribute layout.
                    def _extract_from_obj(obj):
                        names: list[str] = []
                        if isinstance(obj, (list, tuple)):
                            for item in obj:
                                if isinstance(item, str):
                                    names.append(item)
                                else:
                                    nm = getattr(item, "tool_name", None) or getattr(item, "name", None) or getattr(item, "tool", None)
                                    if nm is None:
                                        try:
                                            nm = str(item)
                                        except Exception:
                                            nm = "<unknown>"
                                    names.append(nm)
                        else:
                            if isinstance(obj, str):
                                names.append(obj)
                            else:
                                nm = getattr(obj, "tool_name", None) or getattr(obj, "name", None) or getattr(obj, "tool", None)
                                if nm is None:
                                    try:
                                        nm = str(obj)
                                    except Exception:
                                        nm = "<unknown>"
                                names.append(nm)
                        return names

                    tool_names: list[str] = []

                    # Try a sequence of likely attribute names using getattr to avoid attribute access diagnostics.
                    for attr in ("tool_calls", "calls", "call", "tool_call", "tools"):
                        val = getattr(node, attr, None)
                        if val:
                            tool_names = _extract_from_obj(val)
                            break

                    # Fallback single-name attributes
                    if not tool_names:
                        single = getattr(node, "tool_name", None) or getattr(node, "tool", None)
                        if single:
                            tool_names = _extract_from_obj(single)

                    # Last resort: look for any attributes mentioning 'tool' or 'call' and report candidates for diagnostics.
                    if not tool_names:
                        candidates = []
                        for attr in dir(node):
                            if "tool" in attr.lower() or "call" in attr.lower():
                                try:
                                    val = getattr(node, attr)
                                    candidates.append((attr, repr(val)[:200]))
                                except Exception:
                                    candidates.append((attr, "<error>"))
                        log.debug("Node %s: CallTools node (no standard call attributes found). Candidates: %s", node_count, candidates[:6])
                        tool_names = ["<unknown>"]

                    log.debug("Node %s: Tools → %s", node_count, tool_names)

                elif Agent.is_model_request_node(node):
                    # Extract a short preview of the model request for diagnostics.
                    preview_attr = None
                    preview_val = None
                    for attr in ("prompt", "request", "messages", "model_input", "input", "input_text", "text"):
                        if hasattr(node, attr):
                            try:
                                preview_val = getattr(node, attr)
                            except Exception as e:
                                preview_val = f"<error reading {attr}: {e}>"
                            preview_attr = attr
                            break

                    if isinstance(preview_val, (list, tuple)):
                        try:
                            pv = preview_val[:2]
                        except Exception:
                            pv = str(preview_val)[:200]
                        log.debug("Node %s: Model request | %s: %s", node_count, preview_attr, pv)
                    else:
                        log.debug("Node %s: Model request | %s: %s", node_count, preview_attr, str(preview_val)[:200])

                else:
                    # Generic node diagnostic: show class name and a short list of attributes.
                    attrs = [a for a in dir(node) if not a.startswith("_")]
                    log.debug("Node %s: %s | attrs: %s", node_count, node_type, attrs[:8])

            except Exception as e:
                log.exception("Failed to process node %s: %s", node_count, e)

    # Safely access run.result (run may not exist if context failed)
    result = None
    try:
        result = getattr(run, "result", None)
    except Exception:
        result = None

    elapsed = time.perf_counter() - start

    # Robustly determine finish reason from whatever output shape is present.
    finish_reason = "no_output"
    if result is not None:
        out = getattr(result, "output", None)
        if out is not None:
            finish_reason = (
                getattr(out, "reason_for_stopping", None)
                or getattr(out, "reason", None)
                or getattr(out, "decision", None)
            )
            if finish_reason is None:
                # Try dictionary-style inspection as a fallback
                try:
                    dictifier = getattr(out, "dict", None)
                    if callable(dictifier):
                        od = out.dict()
                        for key in ("reason_for_stopping", "reason", "finish_reason", "decision"):
                            if key in od:
                                finish_reason = od.get(key)
                                break
                except Exception:
                    pass
            if finish_reason is None:
                try:
                    finish_reason = str(out)
                except Exception:
                    finish_reason = "unknown_output"

    log.success(f"Completed in {elapsed:.2f}s | Nodes: {node_count} | Items: {len(deps.items)}")

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=node_count,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 4: Union output for explicit Continue vs Done decision
# =============================================================================

class ContinueAction(BaseModel):
    """Choose this to continue working. Explain your plan."""
    model_config = ConfigDict(extra='forbid')
    decision: Literal["continue"] = "continue"
    next_steps: list[str] = Field(description="What you plan to do next")
    reasoning: str = Field(description="Why you're continuing")


class FinishAction(BaseModel):
    """Choose this when done collecting."""
    model_config = ConfigDict(extra='forbid')
    decision: Literal["finish"] = "finish"
    summary: str = Field(description="Summary of what was accomplished")
    reason: str = Field(description="Why you're stopping")


DecisionOutput = Union[ContinueAction, FinishAction]
DecisionOutputSpec: list[ToolOutput[ContinueAction] | ToolOutput[FinishAction]] = [
    ToolOutput(ContinueAction, strict=True),
    ToolOutput(FinishAction, strict=True)
]


pattern4_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=DecisionOutputSpec,
    instructions="""
    You are a data collector. Each turn you must:
    1. Use tools to search DuckDuckGo, fetch, and save items
    2. Then decide: continue or finish?

    Choose 'continue' if you haven't reached your target and have more strategies to try.
    Choose 'finish' if you've collected enough OR exhausted all options.

    Always explain your reasoning.
    """,
)


@pattern4_agent.tool(strict=True)
async def search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    """Search DuckDuckGo for items."""
    return await _search_impl(ctx, query)


@pattern4_agent.tool(strict=True)
async def fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch content from a URL."""
    return await _fetch_impl(ctx, url)


@pattern4_agent.tool(strict=True)
async def save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save an item. Returns progress."""
    return await _save_impl(ctx, item_id, content)


async def run_pattern4(prompt: str, target: int = 5, max_iterations: int = 10) -> AgentResult:
    """Pattern 4: Model explicitly chooses Continue or Finish each iteration."""
    deps = Deps(target_count=target, pattern_name="P4-union")
    log = deps.log()

    log.info("Starting Pattern 4: Union Continue/Finish")
    start = time.perf_counter()

    messages = None
    iterations = 0
    finish_reason = "max_iterations_reached"

    for i in range(max_iterations):
        iterations += 1
        log.debug("Iteration %s/%s", iterations, max_iterations)

        current_prompt = prompt if messages is None else "Continue or finish based on your progress."

        result = await pattern4_agent.run(
            current_prompt,
            deps=deps,
            message_history=messages,
        )
        messages = result.all_messages()

        match result.output:
            case ContinueAction(reasoning=reason, next_steps=steps):
                log.info("➡️  Continue: %s", reason)
                log.debug("   Next steps: %s", steps)

            case FinishAction(reason=reason, summary=summary):
                log.info("🏁 Finish: %s", reason)
                log.debug("   Summary: %s", summary)
                finish_reason = reason
                break

    elapsed = time.perf_counter() - start
    log.success(f"Completed in {elapsed:.2f}s | Iterations: {iterations} | Items: {len(deps.items)}")

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# RESULTS PERSISTENCE
# =============================================================================

def setup_results_directories():
    """Create doc_results/pat1-4 directories."""
    base_dir = Path("doc_results")
    for i in range(1, 5):
        (base_dir / f"pat{i}").mkdir(parents=True, exist_ok=True)
    return base_dir


def save_pattern_result(pattern_num: int, prompt_idx: int, result: AgentResult):
    """Save a pattern's result to its directory."""
    result_dir = Path("doc_results") / f"pat{pattern_num}"

    # Save each scraped item as a separate JSON file
    items_saved = []
    for idx, item in enumerate(result.items_collected, start=1):
        item_file = result_dir / f"prompt_{prompt_idx}_item_{idx}.json"
        with open(item_file, "w") as f:
            json.dump(item, f, indent=2)

        # Calculate file size
        file_size = item_file.stat().st_size
        items_saved.append({
            "filename": item_file.name,
            "size_bytes": file_size,
            "item": item
        })

    # Calculate total size of all items
    total_size = sum(item["size_bytes"] for item in items_saved)

    # Save detailed result summary
    result_file = result_dir / f"prompt_{prompt_idx}_summary.txt"
    summary_content = f"""Prompt Index: {prompt_idx}
Elapsed Seconds: {result.elapsed_seconds:.3f}
Iterations: {result.iterations}
Finish Reason: {result.finish_reason}
Items Collected: {len(result.items_collected)}
Actions Taken: {len(result.actions_taken)}
Total Data Size: {total_size:,} bytes ({total_size / 1024:.2f} KB)

Items Saved:
"""

    for item_info in items_saved:
        summary_content += f"  - {item_info['filename']} ({item_info['size_bytes']:,} bytes)\n"

    summary_content += "\nActions:\n"
    for action in result.actions_taken:
        summary_content += f"  - {action}\n"

    with open(result_file, "w") as f:
        f.write(summary_content)

    # Return file size for CSV logging
    return total_size


def append_to_csv(prompt_idx: int, prompt: str, times: dict[int, float], sizes: dict[int, int]):
    """Append timing results to CSV file.

    Args:
        prompt_idx: Index of the prompt (1-based)
        prompt: The prompt text
        times: Dictionary mapping pattern number (1-4) to elapsed seconds
        sizes: Dictionary mapping pattern number (1-4) to total bytes scraped
    """
    csv_file = Path("doc_results") / "timing_results.csv"

    # Check if file exists to determine if we need headers
    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "prompt_num",
                "pattern1_sec", "pattern2_sec", "pattern3_sec", "pattern4_sec",
                "pattern1_bytes", "pattern2_bytes", "pattern3_bytes", "pattern4_bytes",
                "prompt_text"
            ])

        # Write data row
        writer.writerow([
            f"prompt_{prompt_idx}",
            times.get(1, 0.0),
            times.get(2, 0.0),
            times.get(3, 0.0),
            times.get(4, 0.0),
            sizes.get(1, 0),
            sizes.get(2, 0),
            sizes.get(3, 0),
            sizes.get(4, 0),
            prompt[:100] + "..." if len(prompt) > 100 else prompt
        ])


# =============================================================================
# TEST HARNESS
# =============================================================================

async def run_comparison(prompt: str, target: int = 5, prompt_idx: int = 0) -> dict[str, AgentResult]:
    """Run all 4 patterns with the same prompt and compare results."""

    logger.info("=" * 70)
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"TARGET: {target} items")
    logger.info("=" * 70)

    patterns: list[tuple[str, Callable[..., Any], int]] = [
        ("Pattern 1: Implicit output_type", run_pattern1, 1),
        ("Pattern 2: Explicit finish tool", run_pattern2, 2),
        ("Pattern 3: iter() API", run_pattern3, 3),
        ("Pattern 4: Union Continue/Finish", run_pattern4, 4),
    ]

    results: dict[str, AgentResult] = {}
    timing_data: dict[int, float] = {}
    size_data: dict[int, int] = {}

    for name, runner, pattern_num in patterns:
        logger.info(f"\n{'─' * 70}")
        logger.info(f"▶ {name}")
        logger.info("─" * 70)

        try:
            result = await runner(prompt, target)
            results[name] = result
            timing_data[pattern_num] = result.elapsed_seconds

            # Save individual pattern result and get total size
            total_size = save_pattern_result(pattern_num, prompt_idx, result)
            size_data[pattern_num] = total_size

        except Exception as e:
            logger.exception(f"❌ {name} failed: {e}")
            timing_data[pattern_num] = 0.0
            size_data[pattern_num] = 0

    # Summary comparison
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    header = f"{'Pattern':<40} {'Items':<8} {'Actions':<10} {'Iters':<8} {'Time(s)':<10}"
    logger.info(header)
    logger.info("-" * 70)

    for name, result in results.items():
        short_name = name.split(":")[0]
        logger.info(
            f"{short_name:<40} "
            f"{len(result.items_collected):<8} "
            f"{len(result.actions_taken):<10} "
            f"{result.iterations:<8} "
            f"{result.elapsed_seconds:<10.2f}"
        )

    # Save timing data to CSV
    append_to_csv(prompt_idx, prompt, timing_data, size_data)

    return results


async def run_single_pattern(
    pattern: Literal[1, 2, 3, 4],
    prompt: str,
    target: int = 5,
) -> AgentResult:
    """Run a single pattern for isolated testing."""
    runners = {
        1: run_pattern1,
        2: run_pattern2,
        3: run_pattern3,
        4: run_pattern4,
    }
    return await runners[pattern](prompt, target)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run comparison tests."""

    # Setup result directories
    setup_results_directories()
    logger.info("Created doc_results/pat1-4 directories")

    for idx, prompt in enumerate(PROMPTS, start=1):
        await run_comparison(prompt, target=5, prompt_idx=idx)
        logger.info("\n" * 2)


if __name__ == "__main__":
    uvloop.run(main())
