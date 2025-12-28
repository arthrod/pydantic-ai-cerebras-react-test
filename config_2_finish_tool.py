"""Config 2: Explicit finish tool with external loop.

Uses a finish() tool to signal completion, with an external loop that continues until finished.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import logfire
from loguru import logger
from pydantic_ai import Agent, RunContext, ToolDefinition, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
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
    is_finished: bool = False
    finish_reason: str = ""


async def _force_tool_strict_true(
    ctx: RunContext[Deps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    return [replace(tool_def, strict=True) for tool_def in tool_defs]


def _build_agent() -> Agent[Deps, str]:
    logger.info("Building agent with CerebrasModel zai-glm-4.6 (reasoning disabled, finish tool pattern)")
    model = CerebrasModel(
        "zai-glm-4.6",
        settings=CerebrasModelSettings(cerebras_disable_reasoning=True),
    )
    instructions = (
        "You collect legal documents from a corpus containing Brazilian and American law. "
        "The corpus has contracts, court briefs, and legal opinions from both jurisdictions. "
        "Use tools to search, retrieve, and save documents. Never invent documents. "
        "Try multiple search queries with relevant legal terms. "
        "When you have collected enough documents, call the finish(reason) tool to complete."
    )
    agent: Agent[Deps, str] = Agent(
        model,
        deps_type=Deps,
        output_type=str,
        instructions=instructions,
        prepare_tools=_force_tool_strict_true,
        prepare_output_tools=_force_tool_strict_true,
    )
    logger.debug("Agent created with output_type=str and finish tool")

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

    @agent.tool
    async def finish(ctx: RunContext[Deps], reason: str) -> str:
        """Call this when you have finished collecting documents. Provide the reason for stopping."""
        logger.success(f"[TOOL] finish called with reason='{reason}'")
        ctx.deps.is_finished = True
        ctx.deps.finish_reason = reason
        return f"finishing:{reason}"

    return agent


async def run_prompt(prompt: str, *, max_iterations: int = 20) -> dict[str, Any]:
    _load_dotenv()
    logger.info(f"[CONFIG_2] Starting run with prompt: '{prompt[:60]}...'")
    with logfire.span("config_2_finish_tool.run_prompt", prompt=prompt[:100]):
        agent = _build_agent()
        deps = Deps()
        messages = None
        started = time.perf_counter()
        last_result = None

        for iteration in range(max_iterations):
            logger.debug(f"[CONFIG_2] Iteration {iteration + 1}/{max_iterations}")
            try:
                current_prompt = prompt if messages is None else "Continue your task."
                with logfire.span(f"iteration_{iteration + 1}"):
                    last_result = await agent.run(
                        current_prompt,
                        deps=deps,
                        message_history=messages,
                        usage_limits=UsageLimits(request_limit=20, tool_calls_limit=60),
                    )
                messages = last_result.all_messages()
                logger.debug(f"[CONFIG_2] Iteration {iteration + 1} complete, is_finished={deps.is_finished}")
                if deps.is_finished:
                    logger.info(f"[CONFIG_2] Agent finished at iteration {iteration + 1}")
                    break
            except UsageLimitExceeded as e:
                logger.warning(f"[CONFIG_2] UsageLimitExceeded: {e}")
                deps.is_finished = True
                deps.finish_reason = str(e)
                break

        elapsed = time.perf_counter() - started
        logger.success(
            f"[CONFIG_2] Completed in {elapsed:.2f}s | "
            f"saved={len(deps.saved_doc_ids)} | queries={len(deps.queries_tried)} | "
            f"messages={len(messages or [])} | reason='{deps.finish_reason[:50]}'"
        )

        return {
            "config": "config_2_finish_tool",
            "seconds": elapsed,
            "prompt": prompt,
            "output": getattr(last_result, "output", None),
            "saved_count": len(deps.saved_doc_ids),
            "queries_tried": len(deps.queries_tried),
            "message_count": len(messages or []),
            "is_finished": deps.is_finished,
            "finish_reason": deps.finish_reason,
        }


def main() -> None:
    prompt = "Collect 4 employment-related legal documents (contracts or briefs) from both Brazilian and US law."
    out = asyncio.run(run_prompt(prompt))
    print(out)


if __name__ == "__main__":
    main()
