"""Config 4: Union output for continue vs done decision.

Uses a union type (ContinuePlan | FinalResult) so the model explicitly decides to continue or stop.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Union

import logfire
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ToolDefinition

from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings

from corpus import CORPUS
from logging_config import configure_logging

configure_logging()


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Deps:
    saved_doc_ids: set[str] = field(default_factory=set)
    queries_tried: list[str] = field(default_factory=list)


class ContinuePlan(BaseModel):
    type: Literal["continue"] = "continue"
    next_queries: list[str]
    status: str


class FinalResult(BaseModel):
    type: Literal["finished"] = "finished"
    documents: list[dict[str, Any]]
    total_saved: int
    queries_tried: list[str]
    summary: str


# Type alias for annotations; list form for Agent output_type parameter
OutputType = Union[ContinuePlan, FinalResult]
OutputTypeSpec: list[type[ContinuePlan] | type[FinalResult]] = [ContinuePlan, FinalResult]


async def _force_tool_strict_true(ctx: RunContext[Deps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
    return [replace(tool_def, strict=True) for tool_def in tool_defs]


def _build_agent() -> Agent[Deps, OutputType]:
    logger.info("Building agent with CerebrasModel zai-glm-4.6 (reasoning disabled, union output pattern)")
    model = CerebrasModel(
        "zai-glm-4.6",
        settings=CerebrasModelSettings(cerebras_disable_reasoning=True),
    )
    instructions = (
        "You collect legal documents from a corpus containing Brazilian and American law. "
        "The corpus has contracts, court briefs, and legal opinions from both jurisdictions. "
        "Use tools to search, retrieve, and save documents. Never invent documents. "
        "Try multiple search queries with relevant legal terms. "
        "After each run, choose either ContinuePlan (with next queries to try) or FinalResult (when done)."
    )
    agent: Agent[Deps, OutputType] = Agent(
        model,
        deps_type=Deps,
        output_type=OutputTypeSpec,
        instructions=instructions,
        prepare_tools=_force_tool_strict_true,
        prepare_output_tools=_force_tool_strict_true,
    )
    logger.debug("Agent created with output_type=ContinuePlan|FinalResult (union)")

    @agent.tool
    async def search_corpus(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
        """Search the legal corpus by keyword. Returns up to 5 matching documents."""
        logger.info(f"[TOOL] search_corpus called with query='{query}'")
        q = query.strip().lower()
        ctx.deps.queries_tried.append(query)
        matches: list[dict[str, Any]] = []
        for doc in CORPUS:
            hay = f"{doc['title']} {doc['snippet']} {doc['jurisdiction']} {doc['doc_type']} {' '.join(doc['tags'])}".lower()
            if not q or q in hay:
                matches.append({k: doc[k] for k in ("id", "title", "snippet", "doc_type", "jurisdiction", "tags")})
        result = matches[:5]
        logger.debug(f"[TOOL] search_corpus found {len(result)} matches: {[m['id'] for m in result]}")
        return result

    @agent.tool
    async def get_document(ctx: RunContext[Deps], doc_id: str) -> dict[str, Any]:
        """Get the full content of a document by its ID."""
        logger.info(f"[TOOL] get_document called with doc_id='{doc_id}'")
        for doc in CORPUS:
            if doc["id"] == doc_id:
                logger.debug(f"[TOOL] get_document found: {doc['title']}")
                return {k: doc[k] for k in ("id", "title", "doc_type", "jurisdiction", "content", "tags")}
        logger.warning(f"[TOOL] get_document not found: {doc_id}")
        return {"id": doc_id, "error": "not_found"}

    @agent.tool
    async def save_document(ctx: RunContext[Deps], doc_id: str) -> str:
        """Save a document to your collection."""
        logger.info(f"[TOOL] save_document called with doc_id='{doc_id}'")
        ctx.deps.saved_doc_ids.add(doc_id)
        logger.debug(f"[TOOL] Total saved: {len(ctx.deps.saved_doc_ids)} docs")
        return f"saved:{doc_id}"

    @agent.tool
    async def get_saved_documents(ctx: RunContext[Deps]) -> list[dict[str, Any]]:
        """Get all documents you have saved so far."""
        logger.info(f"[TOOL] get_saved_documents called, {len(ctx.deps.saved_doc_ids)} docs saved")
        saved: list[dict[str, Any]] = []
        for doc_id in sorted(ctx.deps.saved_doc_ids):
            saved.append(await get_document(ctx, doc_id))
        return saved

    return agent


async def run_prompt(prompt: str, *, max_iterations: int = 20) -> dict[str, Any]:
    _load_dotenv()
    logger.info(f"[CONFIG_4] Starting run with prompt: '{prompt[:60]}...'")
    with logfire.span("config_4_union_output.run_prompt", prompt=prompt[:100]):
        agent = _build_agent()
        deps = Deps()
        messages = None
        current_prompt = prompt
        started = time.perf_counter()
        final_output: dict[str, Any] | None = None

        for iteration in range(max_iterations):
            logger.debug(f"[CONFIG_4] Iteration {iteration + 1}/{max_iterations}")
            with logfire.span(f"iteration_{iteration + 1}"):
                result = await agent.run(current_prompt, deps=deps, message_history=messages)
            messages = result.all_messages()

            match result.output:
                case ContinuePlan(next_queries=next_queries, status=status):
                    logger.info(f"[CONFIG_4] ContinuePlan: status='{status}', next_queries={next_queries}")
                    current_prompt = f"Continue. Try these queries: {next_queries}. Current status: {status}"
                case FinalResult() as final:
                    logger.success(f"[CONFIG_4] FinalResult at iteration {iteration + 1}: summary='{final.summary[:50]}...'")
                    final_output = final.model_dump()
                    break

        elapsed = time.perf_counter() - started
        logger.success(
            f"[CONFIG_4] Completed in {elapsed:.2f}s | "
            f"saved={len(deps.saved_doc_ids)} | queries={len(deps.queries_tried)} | "
            f"messages={len(messages or [])} | finished={final_output is not None}"
        )
        if final_output:
            logger.debug(f"[CONFIG_4] Output: {final_output}")

        return {
            "config": "config_4_union_output",
            "seconds": elapsed,
            "prompt": prompt,
            "output": final_output,
            "saved_count": len(deps.saved_doc_ids),
            "queries_tried": len(deps.queries_tried),
            "message_count": len(messages or []),
            "finished": final_output is not None,
        }


def main() -> None:
    prompt = "Collect 4 employment-related legal documents (contracts or briefs) from both Brazilian and US law."
    out = asyncio.run(run_prompt(prompt))
    print(out)


if __name__ == "__main__":
    main()
