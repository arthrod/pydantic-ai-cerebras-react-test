"""Config 3: Using iter() for fine-grained control.

Steps through the agent graph node-by-node for monitoring and control.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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


class CollectedResult(BaseModel):
    documents: list[dict[str, Any]]
    total_saved: int
    queries_tried: list[str]
    reason_for_stopping: str


async def _force_tool_strict_true(ctx: RunContext[Deps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
    return [replace(tool_def, strict=True) for tool_def in tool_defs]


def _build_agent() -> Agent[Deps, CollectedResult]:
    logger.info("Building agent with CerebrasModel zai-glm-4.6 (reasoning disabled, iter pattern)")
    model = CerebrasModel(
        "zai-glm-4.6",
        settings=CerebrasModelSettings(cerebras_disable_reasoning=True),
    )
    instructions = (
        "You collect legal documents from a corpus containing Brazilian and American law. "
        "The corpus has contracts, court briefs, and legal opinions from both jurisdictions. "
        "Use tools to search, retrieve, and save documents. Never invent documents. "
        "Try multiple search queries with relevant legal terms. "
        "When you have collected enough documents, produce CollectedResult with your findings."
    )
    agent: Agent[Deps, CollectedResult] = Agent(
        model,
        deps_type=Deps,
        output_type=CollectedResult,
        instructions=instructions,
        prepare_tools=_force_tool_strict_true,
        prepare_output_tools=_force_tool_strict_true,
    )
    logger.debug("Agent created with output_type=CollectedResult (iter monitoring)")

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


async def run_prompt(prompt: str) -> dict[str, Any]:
    _load_dotenv()
    logger.info(f"[CONFIG_3] Starting run with prompt: '{prompt[:60]}...'")
    with logfire.span("config_3_iter.run_prompt", prompt=prompt[:100]):
        agent = _build_agent()
        deps = Deps()
        started = time.perf_counter()
        node_counts: dict[str, int] = {"call_tools": 0, "model_request": 0, "user_prompt": 0, "end": 0, "other": 0}

        logger.debug("[CONFIG_3] Starting agent.iter() - node-by-node iteration")
        async with agent.iter(prompt, deps=deps) as run:
            async for node in run:
                if Agent.is_call_tools_node(node):
                    node_counts["call_tools"] += 1
                    logger.debug(f"[CONFIG_3] CallToolsNode #{node_counts['call_tools']}")
                elif Agent.is_model_request_node(node):
                    node_counts["model_request"] += 1
                    logger.debug(f"[CONFIG_3] ModelRequestNode #{node_counts['model_request']}")
                elif Agent.is_user_prompt_node(node):
                    node_counts["user_prompt"] += 1
                    logger.debug(f"[CONFIG_3] UserPromptNode #{node_counts['user_prompt']}")
                elif Agent.is_end_node(node):
                    node_counts["end"] += 1
                    logger.info("[CONFIG_3] EndNode reached - agent complete")
                else:
                    node_counts["other"] += 1
                    logger.debug(f"[CONFIG_3] OtherNode: {type(node).__name__}")

        result = run.result
        elapsed = time.perf_counter() - started
        output = result.output

        logger.success(
            f"[CONFIG_3] Completed in {elapsed:.2f}s | "
            f"saved={len(deps.saved_doc_ids)} | queries={len(deps.queries_tried)} | "
            f"messages={len(result.all_messages())} | nodes={node_counts}"
        )
        logger.debug(f"[CONFIG_3] Output: {output.model_dump()}")

        return {
            "config": "config_3_iter",
            "seconds": elapsed,
            "prompt": prompt,
            "output": output.model_dump(),
            "saved_count": len(deps.saved_doc_ids),
            "queries_tried": len(deps.queries_tried),
            "message_count": len(result.all_messages()),
            "node_counts": node_counts,
        }


def main() -> None:
    prompt = "Collect 4 employment-related legal documents (contracts or briefs) from both Brazilian and US law."
    out = asyncio.run(run_prompt(prompt))
    print(out)


if __name__ == "__main__":
    main()
