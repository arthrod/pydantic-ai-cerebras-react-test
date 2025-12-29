"""Five agentic loop patterns for comparison testing.
Each pattern receives the same prompt and produces a standardized result.

Features:
- Jina AI Search and Reader API integration (correct POST requests)
- Sub-agent document analysis to keep main-agent context small
- Deterministic, token-aware history trimming using tiktoken (middle-out)
- Proper logging with logfire
- Time to completion tracking
- Cerebras model with zai-glm-4.6

Run with: uv run python agent_patterns.py

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import logfire as logger
import tiktoken
import uvloop
from dotenv import load_dotenv

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import ToolDefinition

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

logger.configure()
logger.instrument_httpx(capture_all=True)

load_dotenv()


async def set_all_tools_strict(
    ctx: RunContext[Any], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Prepare tools callback that sets strict=True on ALL tools.

    Cerebras requires all tools to have the same strict value.
    """
    return [replace(tool_def, strict=True) for tool_def in tool_defs]


# NOTE: History limit is in TOKENS, not messages.
MAX_HISTORY_TOKENS = 120_000

# How much of a fetched document we allow into the *sub-agent* prompt.
# Keep this comfortably below your model context window.
SUBAGENT_MAX_CONTENT_TOKENS = 20_000

JINA_API_KEY = os.getenv('JINA_API_KEY')
JINA_SEARCH_URL = 'https://s.jina.ai/'  # POST with {"q": "..."}
JINA_READER_URL = 'https://r.jina.ai/'  # POST with {"url": "..."}

_ENCODING: tiktoken.Encoding | None = None

CORE_INSTRUCTIONS = """
You are a document collector. Your task is to collect items by:
1. Searching for relevant items using Jina AI Search
2. Fetching promising URLs to get a sub-agent analysis (do NOT ask for full content in your own context)
3. Saving good items to your collection

Continue until you've collected enough items (check save responses for progress)
OR you've exhausted all reasonable search strategies.

Be thorough - try multiple search queries before concluding.

IMPORTANT:
- The fetch tool returns a doc_id and an analysis (summary + save suggestion). It does NOT return the full document text.
- To persist a document to disk, call save(doc_id, filename). Prefer the filename suggested by the sub-agent, but you may adjust it.
"""

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


# =============================================================================
# TOKENIZATION HELPERS
# =============================================================================


def _get_encoding() -> tiktoken.Encoding:
    """Get or create the tiktoken encoding (lazy init)."""
    global _ENCODING
    if _ENCODING is None:
        # cl100k_base works well as a general-purpose tokenizer
        _ENCODING = tiktoken.get_encoding('cl100k_base')
    return _ENCODING


def _count_tokens_text(text: str) -> int:
    """Count tokens for a string using tiktoken."""
    return len(_get_encoding().encode(text))


def _count_tokens_msg(msg: Any) -> int:
    """Count tokens for a message using tiktoken (best-effort deterministic).

    NOTE: We use model_dump_json() if present so that token counts are stable.
    """
    try:
        content = msg.model_dump_json() if hasattr(msg, 'model_dump_json') else str(msg)
        return _count_tokens_text(content)
    except Exception:
        # Deterministic fallback
        return _count_tokens_text(str(msg))


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum token length (deterministic)."""
    if max_tokens <= 0:
        return ''
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + '\n\n[... truncated by token limit ...]'


# =============================================================================
# CONTEXT TRIMMING (DETERMINISTIC, MIDDLE-OUT)
# =============================================================================


def is_context_length_error(e: Exception) -> bool:
    """Check if exception is a context length exceeded error."""
    if isinstance(e, ModelHTTPError) and e.status_code == 400:
        body_str = str(e.body).lower()
        return 'context_length' in body_str or 'context length' in body_str
    return False


def _has_tool_calls(msg: Any) -> bool:
    """Check if a ModelResponse contains tool calls."""
    if not hasattr(msg, 'parts'):
        return False
    return any(isinstance(p, ToolCallPart) for p in msg.parts)


def _has_tool_returns(msg: Any) -> bool:
    """Check if a ModelRequest contains tool returns."""
    if not hasattr(msg, 'parts'):
        return False
    return any(isinstance(p, ToolReturnPart) for p in msg.parts)


def _tool_pair_span(messages: list[Any], idx: int) -> set[int]:
    """Indices to remove to avoid leaving unpaired tool call/return messages."""
    span = {idx}

    if idx < 0 or idx >= len(messages):
        return span

    if _has_tool_calls(messages[idx]):
        nxt = idx + 1
        if nxt < len(messages) and _has_tool_returns(messages[nxt]):
            span.add(nxt)

    if _has_tool_returns(messages[idx]):
        prev = idx - 1
        if prev >= 0 and _has_tool_calls(messages[prev]):
            span.add(prev)

    return span


def _validate_and_fix_tool_sequence(messages: list[Any], log: Any | None = None) -> list[Any]:
    """Ensure no orphaned tool returns exist (each tool return must follow a tool_calls message).
    
    This is critical for APIs like Cerebras that require tool messages to follow tool_calls.
    """
    if len(messages) <= 1:
        return messages
    
    result = []
    i = 0
    removed_count = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if _has_tool_returns(msg):
            # Check if previous message in result has tool_calls
            if result and _has_tool_calls(result[-1]):
                # Valid pair, keep it
                result.append(msg)
            else:
                # Orphaned tool return - skip it
                removed_count += 1
                if log is not None:
                    log.debug(f'Removing orphaned tool return at index {i}')
        elif _has_tool_calls(msg):
            # Check if next message has tool returns
            if i + 1 < len(messages) and _has_tool_returns(messages[i + 1]):
                # Valid pair, keep both
                result.append(msg)
                result.append(messages[i + 1])
                i += 1  # Skip the next message since we already added it
            else:
                # Tool call without response - skip it to avoid issues
                removed_count += 1
                if log is not None:
                    log.debug(f'Removing tool call without response at index {i}')
        else:
            # Regular message, keep it
            result.append(msg)
        
        i += 1
    
    if removed_count > 0 and log is not None:
        log.warning(f'🔧 Fixed tool sequence: removed {removed_count} orphaned messages')
    
    return result


def deterministic_trim_history(messages: list[Any], max_tokens: int, log: Any | None = None) -> list[Any]:
    """Trim message history deterministically by removing from the middle-out.

    Algorithm:
    - Always keep messages[0] (system prompt) and messages[-1] (most recent).
    - Count tokens with tiktoken for each message.
    - While total > max_tokens, remove the message closest to the middle among messages[1:-1].
      If the chosen message is part of a tool call/return pair, remove the adjacent pair
      to preserve consistency.
    - Uses alternating left/right bias from middle to spread removals evenly.

    Example (12 messages excluding system, all equal size):
      remove 6, 5, 7, 4, 8, 3, 9, 2, 10 ...

    This preserves *both* early instructions and the most recent state, dropping the "middle" first.
    """
    if not messages:
        return messages

    msgs = list(messages)
    token_counts = [_count_tokens_msg(m) for m in msgs]
    total = sum(token_counts)

    if total <= max_tokens:
        return msgs

    if log is not None:
        log.warning(
            f'⚠️  History over limit: ~{total:,} tokens > {max_tokens:,} '
            f'({len(msgs)} messages). Trimming deterministically from the middle.'
        )

    removed = 0
    removed_tokens = 0
    iteration = 0

    while total > max_tokens and len(msgs) > 2:  # Keep at least system + last message
        # Removable range: messages[1] to messages[-2] (exclude system at 0 and last at -1)
        removable_start = 1
        removable_end = len(msgs) - 2  # inclusive index of last removable

        if removable_start > removable_end:
            # Only system and last message remain, can't trim more
            break

        # Calculate middle of removable range
        removable_count = removable_end - removable_start + 1
        middle_offset = removable_count // 2

        # Alternate between left and right of middle to spread removals
        if iteration % 2 == 0:
            # Pick from left-middle
            idx = removable_start + middle_offset
        else:
            # Pick from right-middle (or same if odd count)
            idx = removable_start + (middle_offset - 1 if removable_count > 1 else middle_offset)

        # Clamp to valid range
        idx = max(removable_start, min(idx, removable_end))

        span = _tool_pair_span(msgs, idx)
        span.discard(0)  # never remove system
        span.discard(len(msgs) - 1)  # never remove last message
        if not span:
            # If we can't remove anything at this position, try next iteration
            iteration += 1
            if iteration > len(msgs) * 2:  # Safety break
                break
            continue

        for i in sorted(span, reverse=True):
            removed += 1
            removed_tokens += token_counts[i]
            total -= token_counts[i]
            del msgs[i]
            del token_counts[i]

        iteration += 1

    if log is not None:
        log.info(
            f'✂️  Trimmed history: removed {removed} messages '
            f'(~{removed_tokens:,} tokens). Remaining: {len(msgs)} messages, ~{total:,} tokens.'
        )

    return msgs


async def token_limit_history_processor(ctx: RunContext[Any], messages: list[Any]) -> list[Any]:
    """pydantic_ai history processor that enforces MAX_HISTORY_TOKENS deterministically.

    Works for agents with or without deps (sub-agent has no deps).
    """
    if not messages:
        return messages

    total = sum(_count_tokens_msg(m) for m in messages)
    if total <= MAX_HISTORY_TOKENS:
        return messages

    log = None
    try:
        deps = getattr(ctx, 'deps', None)
        if deps is not None and hasattr(deps, 'log'):
            log = deps.log()
    except Exception:
        log = None

    return deterministic_trim_history(messages, MAX_HISTORY_TOKENS, log=log)


def emergency_trim_history(messages: list[Any], max_tokens: int = 60_000) -> list[Any]:
    """Emergency trim (used after a context-length error)."""
    return deterministic_trim_history(messages, max_tokens=max_tokens, log=None)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


def get_model() -> CerebrasModel:
    """Create the Cerebras model with reasoning disabled."""
    return CerebrasModel('zai-glm-4.6', settings=CerebrasModelSettings(cerebras_disable_reasoning=True))


# =============================================================================
# STANDARDIZED OUTPUT
# =============================================================================


class AgentResult(BaseModel):
    """Standardized result across all patterns."""

    items_collected: list[dict[str, Any]]
    actions_taken: list[str]
    iterations: int
    finish_reason: str
    elapsed_seconds: float


# =============================================================================
# SHARED DEPENDENCIES / STATE
# =============================================================================


class FetchedDocAnalysis(BaseModel):
    """Analysis result stored in FetchedDoc."""

    model_config = ConfigDict(extra='allow')

    should_save: bool = False
    answer_to_prompt: str = ''
    file_summary: str = ''
    suggested_filename: str = ''
    save_instructions: str = ''


@dataclass
class FetchedDoc:
    doc_id: str
    url: str
    title: str = ''
    description: str = ''
    content_md: str = ''
    content_tokens: int = 0
    analyzed: bool = False
    analysis: FetchedDocAnalysis | None = None
    fetched_at: float = field(default_factory=time.time)


@dataclass
class Deps:
    """Shared state across tool calls."""

    items: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    target_count: int = 5
    pattern_name: str = 'unknown'

    # For directory layout / persistence
    run_dir: Path | None = None
    pattern_dir: Path | None = None
    docs_dir: Path | None = None
    prompt_idx: int = 0
    prompt_text: str = ''

    # Cache for fetched docs (so save() doesn't need full content in the model context)
    fetched_docs: dict[str, FetchedDoc] = field(default_factory=dict)

    # Pattern 2 specific
    is_finished: bool = False
    finish_reason: str = ''

    def log(self) -> Any:
        return logger.with_tags(self.pattern_name)


# =============================================================================
# SUB-AGENT: DOCUMENT ANALYSIS
# =============================================================================


class DocAnalysisResult(BaseModel):
    """Result from the document-analysis sub-agent.

    Requirements:
    - answers the main agent's question (evaluate against the original prompt)
    - provides a concise summary
    - tells how to save using the naming convention
    """

    model_config = ConfigDict(extra='forbid')

    should_save: bool = Field(description='Whether this document should be saved for the collection task.')
    answer_to_prompt: str = Field(description='How this document matches (or fails) the original prompt.')
    file_summary: str = Field(description='A short, content-focused summary of the document.')
    suggested_filename: str = Field(
        description='A filename that follows the prompt naming convention (use underscores; include extension .pdf or .md).'
    )
    save_instructions: str = Field(
        description='Concrete instructions on how/where to save this file using the naming convention (include any required identifiers).'
    )


DOC_ANALYZER_INSTRUCTIONS = """
You are a document analysis sub-agent.

You will receive:
- The original collection prompt (what the main agent is trying to collect)
- The candidate document URL, title/description (if available)
- The candidate document extracted content (markdown)

Your job is to return a STRICT structured result that includes:
1) should_save: True/False
2) answer_to_prompt: Explain whether/how this document satisfies the prompt constraints
   (jurisdiction, document type, date range, page count, docket/case numbers, public-filing source constraints, etc.)
3) file_summary: A concise summary of the document (aim for <= 150 words)
4) suggested_filename: A safe filename with underscores, no spaces. Follow the naming convention in the prompt.
   - If the prompt demands a PDF filename convention, still provide a .pdf suggestion.
   - If the source is clearly not a PDF, you may suggest .md, but explain in save_instructions.
   - If the prompt requires case/docket numbers, include them if found; otherwise include a placeholder like "NO_DOCKET".
5) save_instructions: Explain how to save (including the naming convention details and what to do if PDF bytes are unavailable).

Be pragmatic: if you're missing info (e.g., page count), say so explicitly and base should_save on best evidence.
"""

doc_analyzer_agent: Agent[None, DocAnalysisResult] = Agent(
    get_model(),
    output_type=ToolOutput(DocAnalysisResult, strict=True),
    instructions=DOC_ANALYZER_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


# =============================================================================
# JINA AI SEARCH/READER IMPLEMENTATION (POST)
# =============================================================================

_httpx_client: httpx.AsyncClient | None = None


async def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            follow_redirects=True,
        )
    return _httpx_client


async def close_httpx_client() -> None:
    global _httpx_client
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None


async def jina_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Perform a Jina AI search using the correct POST API."""
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if JINA_API_KEY:
            headers['Authorization'] = f'Bearer {JINA_API_KEY}'

        request_body = {'q': query, 'num': max_results}
        client = await get_httpx_client()
        response = await client.post(JINA_SEARCH_URL, headers=headers, json=request_body)

        logger.info(f"Jina search '{query}': status={response.status_code}")

        if response.status_code != 200:
            logger.warning(f'Jina search non-200: {response.status_code} - {response.text[:500]}')
            return {'error': f'HTTP {response.status_code}', 'query': query, 'results': []}

        data = response.json()
        if 'data' in data:
            results = data['data']
            return {'query': query, 'results': results, 'count': len(results)}

        return {'query': query, 'results': [], 'raw_response': data}
    except Exception as e:
        logger.exception(f"Jina search unexpected error for '{query}': {e}")
        return {'error': str(e), 'query': query, 'results': []}


async def jina_fetch(url: str) -> dict[str, Any]:
    """Fetch URL content using Jina AI Reader API (POST). Returns markdown content."""
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if JINA_API_KEY:
            headers['Authorization'] = f'Bearer {JINA_API_KEY}'

        request_body = {'url': url}
        client = await get_httpx_client()
        response = await client.post(JINA_READER_URL, headers=headers, json=request_body)

        logger.info(f"Jina fetch '{url[:50]}...': status={response.status_code}")

        if response.status_code != 200:
            return {
                'url': url,
                'content': '',
                'status': 'error',
                'status_code': response.status_code,
                'error': response.text[:500],
            }

        data = response.json()
        if 'data' in data:
            page_data = data['data']
            content = page_data.get('content', '') or ''
            title = page_data.get('title', '') or ''
            description = page_data.get('description', '') or ''

            return {
                'url': url,
                'title': title,
                'description': description,
                'content': content,
                'content_length_chars': len(content),
                'status': 'success',
                'status_code': 200,
            }

        return {
            'url': url,
            'content': '',
            'status': 'error',
            'error': 'Unexpected response structure',
            'raw_response': data,
        }
    except Exception as e:
        logger.exception(f'Jina fetch unexpected error {url[:40]}: {e}')
        return {'url': url, 'content': '', 'status': 'error', 'error': str(e)}


# =============================================================================
# SHARED TOOL IMPLEMENTATIONS (SEARCH / FETCH->SUBAGENT / SAVE)
# =============================================================================


def _stable_doc_id(url: str) -> str:
    """Deterministic doc id based on URL (stable across runs)."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]


_filename_safe_re = re.compile(r'[^A-Za-z0-9._-]+')


def _sanitize_filename(name: str) -> str:
    """Create a filesystem-safe filename (keeps extension if present)."""
    name = name.strip().replace(' ', '_')
    name = _filename_safe_re.sub('_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('._')
    return name or 'document'


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(2, 10_000):
        candidate = parent / f'{stem}_{i}{suffix}'
        if not candidate.exists():
            return candidate
    return parent / f'{stem}_{int(time.time())}{suffix}'


async def _search_impl(deps: Deps, query: str) -> dict[str, Any]:
    log = deps.log()
    log.info(f"🔍 Searching: '{query}'")

    results = await jina_search(query, max_results=10)
    deps.actions.append(f'search:{query}')

    return results


async def _fetch_impl(deps: Deps, url: str) -> dict[str, Any]:
    """Fetch via Jina Reader, then run sub-agent analysis with token-limited content.

    Returns a compact dict with:
    - doc_id
    - url, title, description
    - analysis (structured)
    """
    log = deps.log()
    log.info(f'📥 Fetching (with sub-agent analysis): {url[:80]}')

    deps.actions.append(f'fetch:{url[:60]}')
    result = await jina_fetch(url)

    if result.get('status') != 'success':
        return {'url': url, 'status': 'error', 'error': result.get('error', 'unknown error')}

    content = result.get('content', '') or ''
    title = result.get('title', '') or ''
    description = result.get('description', '') or ''

    doc_id = _stable_doc_id(url)
    content_tokens = _count_tokens_text(content)

    fetched = deps.fetched_docs.get(doc_id)
    if fetched is None:
        fetched = FetchedDoc(
            doc_id=doc_id,
            url=url,
            title=title,
            description=description,
            content_md=content,
            content_tokens=content_tokens,
        )
        deps.fetched_docs[doc_id] = fetched
    else:
        fetched.title = title
        fetched.description = description
        fetched.content_md = content
        fetched.content_tokens = content_tokens

    # Token-limit what we send to the sub-agent (NOT what we save to disk).
    content_for_subagent = truncate_text_to_tokens(content, SUBAGENT_MAX_CONTENT_TOKENS)

    sub_prompt = f"""ORIGINAL PROMPT:
{deps.prompt_text}

CANDIDATE URL:
{url}

TITLE:
{title}

DESCRIPTION:
{description}

EXTRACTED CONTENT (markdown):
{content_for_subagent}
"""

    try:
        analysis_run = await doc_analyzer_agent.run(sub_prompt)
        fetched.analyzed = True
        # Convert DocAnalysisResult to FetchedDocAnalysis (same fields, different classes)
        doc_result: DocAnalysisResult = analysis_run.output  # type: ignore[assignment]
        fetched.analysis = FetchedDocAnalysis.model_validate(doc_result.model_dump())
    except ModelHTTPError as e:
        if is_context_length_error(e):
            log.warning(
                '🔥 Sub-agent context length exceeded (unexpected after truncation). Returning minimal analysis.'
            )
            fetched.analyzed = False
            fetched.analysis = FetchedDocAnalysis(
                should_save=False,
                answer_to_prompt='Sub-agent exceeded context length.',
                suggested_filename=_sanitize_filename(title) + '.md',
                save_instructions='Retry with smaller SUBAGENT_MAX_CONTENT_TOKENS.',
            )
        else:
            raise

    return {
        'status': 'success',
        'doc_id': doc_id,
        'url': url,
        'title': title,
        'description': description,
        'content_tokens': content_tokens,
        'analysis': fetched.analysis.model_dump() if fetched.analysis else {},
    }


async def _try_download_pdf_bytes(url: str) -> tuple[bytes | None, str | None]:
    """Best-effort PDF download from the source URL."""
    try:
        client = await get_httpx_client()
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; doc-collector/1.0)'}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return None, f'HTTP {resp.status_code}'
        content_type = (resp.headers.get('content-type') or '').lower()
        data = resp.content
        is_pdf = 'application/pdf' in content_type or data[:4] == b'%PDF' or url.lower().endswith('.pdf')
        if not is_pdf:
            return None, f'Not a PDF (content-type={content_type})'
        return data, None
    except Exception as e:
        return None, str(e)


async def _save_impl(deps: Deps, doc_id: str, filename: str) -> str:
    """Persist a fetched doc to disk under deps.docs_dir.

    - Saves PDF bytes if possible (when filename ends with .pdf).
    - Otherwise saves the Jina markdown as .md.
    """
    log = deps.log()

    if deps.docs_dir is None:
        raise RuntimeError('deps.docs_dir is not set; cannot save documents.')

    fetched = deps.fetched_docs.get(doc_id)
    if fetched is None:
        return f"ERROR: Unknown doc_id '{doc_id}'. You must fetch() before save()."

    analysis = fetched.analysis or FetchedDocAnalysis()
    suggested = analysis.suggested_filename
    if not filename:
        filename = suggested or _sanitize_filename(fetched.title or doc_id)

    filename = _sanitize_filename(filename)
    if '.' not in filename:
        # Choose extension: honor suggestion if present; else .md
        if suggested.lower().endswith('.pdf'):
            filename += '.pdf'
        elif suggested.lower().endswith('.md'):
            filename += '.md'
        else:
            filename += '.md'

    # Prompt-scoped directory (recommended to avoid collisions across prompts)
    prompt_dir = deps.docs_dir / f'prompt_{deps.prompt_idx:02d}'
    prompt_dir.mkdir(parents=True, exist_ok=True)

    out_path = _ensure_unique_path(prompt_dir / filename)

    saved_as = out_path.suffix.lower().lstrip('.') or 'md'
    download_error = None

    if out_path.suffix.lower() == '.pdf':
        pdf_bytes, download_error = await _try_download_pdf_bytes(fetched.url)
        if pdf_bytes is not None:
            out_path.write_bytes(pdf_bytes)
        else:
            # Fallback: save markdown instead, but keep the user's base name.
            md_path = out_path.with_suffix('.md')
            md_path = _ensure_unique_path(md_path)
            md_path.write_text(fetched.content_md or '', encoding='utf-8')
            out_path = md_path
            saved_as = 'md'
    else:
        out_path.write_text(fetched.content_md or '', encoding='utf-8')

    file_size = out_path.stat().st_size if out_path.exists() else 0

    item_record = {
        'doc_id': doc_id,
        'url': fetched.url,
        'title': fetched.title,
        'description': fetched.description,
        'saved_path': str(out_path),
        'saved_as': saved_as,
        'file_size_bytes': file_size,
        'content_tokens': fetched.content_tokens,
        'analysis': analysis.model_dump(),
        'pdf_download_error': download_error,
    }

    deps.items.append(item_record)
    deps.actions.append(f'save:{out_path.name}')

    current = len(deps.items)
    target = deps.target_count
    log.info(f"💾 Saved '{out_path.name}' | Progress: {current}/{target}")

    return f'Saved {out_path.name}. Progress: {current}/{target} items.'


# =============================================================================
# PATTERN 1: Implicit Done via output_type
# =============================================================================


class Pattern1Result(BaseModel):
    """Final output schema for pattern 1 (and 3)."""

    items_collected: list[dict[str, Any]] = Field(description='All collected items')
    search_queries_used: list[str] = Field(description='Queries that were tried')
    reason_for_stopping: str = Field(description='Why you decided to stop')


pattern1_agent: Agent[Deps, Pattern1Result] = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
    prepare_tools=set_all_tools_strict,
)


@pattern1_agent.tool(strict=True)
async def p1_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search for documents using Jina AI Search."""
    return await _search_impl(ctx.deps, query)


@pattern1_agent.tool(strict=True)
async def p1_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch a URL and return doc_id + sub-agent analysis (NOT full content)."""
    return await _fetch_impl(ctx.deps, url)


@pattern1_agent.tool(strict=True)
async def p1_save(ctx: RunContext[Deps], doc_id: str, filename: str = '') -> str:
    """Save the fetched document to disk under this run's pat1/docs/prompt_XX."""
    return await _save_impl(ctx.deps, doc_id, filename)


async def run_pattern1(prompt: str, target: int, run_dir: Path, prompt_idx: int) -> AgentResult:
    deps = Deps(
        target_count=target,
        pattern_name='P1-implicit',
        run_dir=run_dir,
        pattern_dir=run_dir / 'pat1',
        docs_dir=(run_dir / 'pat1' / 'docs'),
        prompt_idx=prompt_idx,
        prompt_text=prompt,
    )
    log = deps.log()

    log.info('Starting Pattern 1: Implicit output_type')
    start = time.perf_counter()

    try:
        result = await pattern1_agent.run(prompt, deps=deps)
        finish_reason = result.output.reason_for_stopping
        iterations = 1
    except ModelHTTPError as e:
        if is_context_length_error(e):
            log.warning('🔥 Context length exceeded in internal loop, returning partial results')
            finish_reason = 'context_length_exceeded_partial'
            iterations = 1
        else:
            raise

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Items: {len(deps.items)}')

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 2: Explicit "finish" tool with external loop
# =============================================================================

pattern2_agent: Agent[Deps, str] = Agent(
    get_model(),
    deps_type=Deps,
    output_type=str,
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


@pattern2_agent.tool(strict=True)
async def p2_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    return await _search_impl(ctx.deps, query)


@pattern2_agent.tool(strict=True)
async def p2_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    return await _fetch_impl(ctx.deps, url)


@pattern2_agent.tool(strict=True)
async def p2_save(ctx: RunContext[Deps], doc_id: str, filename: str = '') -> str:
    return await _save_impl(ctx.deps, doc_id, filename)


@pattern2_agent.tool(strict=True)
async def finish(ctx: RunContext[Deps], reason: str) -> str:
    log = ctx.deps.log()
    log.info(f'🏁 Finish called: {reason}')

    ctx.deps.is_finished = True
    ctx.deps.finish_reason = reason
    return f'Marked as finished: {reason}'


async def run_pattern2(
    prompt: str, target: int, run_dir: Path, prompt_idx: int, max_iterations: int = 10
) -> AgentResult:
    deps = Deps(
        target_count=target,
        pattern_name='P2-explicit',
        run_dir=run_dir,
        pattern_dir=run_dir / 'pat2',
        docs_dir=(run_dir / 'pat2' / 'docs'),
        prompt_idx=prompt_idx,
        prompt_text=prompt,
    )
    log = deps.log()

    log.info('Starting Pattern 2: Explicit finish tool')
    start = time.perf_counter()

    messages = None
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        current_prompt = prompt if messages is None else 'Continue your task.'

        try:
            result = await pattern2_agent.run(current_prompt, deps=deps, message_history=messages)
        except ModelHTTPError as e:
            if is_context_length_error(e) and messages:
                log.warning('🔥 Context length exceeded, emergency trim and retry')
                messages = emergency_trim_history(messages, max_tokens=60_000)
                result = await pattern2_agent.run(current_prompt, deps=deps, message_history=messages)
            else:
                raise

        messages = result.all_messages()

        if deps.is_finished:
            break

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Iterations: {iterations} | Items: {len(deps.items)}')

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=deps.finish_reason or 'max_iterations_reached',
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 3: iter() API for fine-grained control
# =============================================================================

pattern3_agent: Agent[Deps, Pattern1Result] = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


@pattern3_agent.tool(strict=True)
async def p3_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    return await _search_impl(ctx.deps, query)


@pattern3_agent.tool(strict=True)
async def p3_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    return await _fetch_impl(ctx.deps, url)


@pattern3_agent.tool(strict=True)
async def p3_save(ctx: RunContext[Deps], doc_id: str, filename: str = '') -> str:
    return await _save_impl(ctx.deps, doc_id, filename)


def _extract_tool_names(node: Any) -> list[str]:
    """Extract tool names from a CallTools node (best-effort across pydantic_ai versions)."""
    names: list[str] = []
    for attr in ('tool_calls', 'calls', 'call', 'tool_call', 'tools'):
        val = getattr(node, attr, None)
        if val:
            if isinstance(val, (list, tuple)):
                for item in val:
                    nm = getattr(item, 'tool_name', None) or getattr(item, 'name', None) or str(item)
                    names.append(nm)
            else:
                nm = getattr(val, 'tool_name', None) or getattr(val, 'name', None) or str(val)
                names.append(nm)
            break
    return names or ['<unknown>']


async def run_pattern3(prompt: str, target: int, run_dir: Path, prompt_idx: int) -> AgentResult:
    deps = Deps(
        target_count=target,
        pattern_name='P3-iter',
        run_dir=run_dir,
        pattern_dir=run_dir / 'pat3',
        docs_dir=(run_dir / 'pat3' / 'docs'),
        prompt_idx=prompt_idx,
        prompt_text=prompt,
    )
    log = deps.log()

    log.info('Starting Pattern 3: iter() API')
    start = time.perf_counter()

    node_count = 0
    finish_reason = 'no_output'

    try:
        async with pattern3_agent.iter(prompt, deps=deps) as run:
            async for node in run:
                node_count += 1
                try:
                    if Agent.is_call_tools_node(node):
                        tool_names = _extract_tool_names(node)
                        log.debug(f'Node {node_count}: Tools → {tool_names}')
                    elif Agent.is_model_request_node(node):
                        log.debug(f'Node {node_count}: Model request')
                    else:
                        log.debug(f'Node {node_count}: {node.__class__.__name__}')
                except Exception as e:
                    log.exception(f'Failed to process node {node_count}: {e}')

        result = getattr(run, 'result', None)
        if result is not None:
            out = getattr(result, 'output', None)
            if out is not None:
                finish_reason = getattr(out, 'reason_for_stopping', None) or str(out)
    except ModelHTTPError as e:
        if is_context_length_error(e):
            log.warning('🔥 Context length exceeded in internal loop, returning partial results')
            finish_reason = 'context_length_exceeded_partial'
        else:
            raise

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Nodes: {node_count} | Items: {len(deps.items)}')

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
    model_config = ConfigDict(extra='forbid')
    decision: Literal['continue'] = 'continue'
    next_steps: list[str] = Field(description='What you plan to do next')
    reasoning: str = Field(description="Why you're continuing")


class FinishAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['finish'] = 'finish'
    summary: str = Field(description='Summary of what was accomplished')
    reason: str = Field(description="Why you're stopping")


DecisionOutputSpec = [
    ToolOutput(ContinueAction, strict=True),
    ToolOutput(FinishAction, strict=True),
]


pattern4_agent: Agent[Deps, ContinueAction | FinishAction] = Agent(
    get_model(),
    deps_type=Deps,
    output_type=DecisionOutputSpec,
    instructions=CORE_INSTRUCTIONS
    + """

After using tools, you must decide: output ContinueAction or FinishAction.
""",
    history_processors=[token_limit_history_processor],
)


@pattern4_agent.tool(strict=True)
async def p4_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    return await _search_impl(ctx.deps, query)


@pattern4_agent.tool(strict=True)
async def p4_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    return await _fetch_impl(ctx.deps, url)


@pattern4_agent.tool(strict=True)
async def p4_save(ctx: RunContext[Deps], doc_id: str, filename: str = '') -> str:
    return await _save_impl(ctx.deps, doc_id, filename)


async def run_pattern4(
    prompt: str, target: int, run_dir: Path, prompt_idx: int, max_iterations: int = 10
) -> AgentResult:
    deps = Deps(
        target_count=target,
        pattern_name='P4-union',
        run_dir=run_dir,
        pattern_dir=run_dir / 'pat4',
        docs_dir=(run_dir / 'pat4' / 'docs'),
        prompt_idx=prompt_idx,
        prompt_text=prompt,
    )
    log = deps.log()

    log.info('Starting Pattern 4: Union Continue/Finish')
    start = time.perf_counter()

    messages = None
    iterations = 0
    finish_reason = 'max_iterations_reached'

    for _ in range(max_iterations):
        iterations += 1
        current_prompt = prompt if messages is None else 'Continue or finish based on your progress.'

        try:
            result = await pattern4_agent.run(current_prompt, deps=deps, message_history=messages)
        except ModelHTTPError as e:
            if is_context_length_error(e) and messages:
                log.warning('🔥 Context length exceeded, emergency trim and retry')
                messages = emergency_trim_history(messages, max_tokens=60_000)
                result = await pattern4_agent.run(current_prompt, deps=deps, message_history=messages)
            else:
                raise

        messages = result.all_messages()

        match result.output:
            case ContinueAction(reasoning=reason, next_steps=steps):
                log.info(f'➡️  Continue: {reason}')
                log.debug(f'   Next steps: {steps}')
            case FinishAction(reason=reason, summary=summary):
                log.info(f'🏁 Finish: {reason}')
                log.debug(f'   Summary: {summary}')
                finish_reason = reason
                break

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Iterations: {iterations} | Items: {len(deps.items)}')

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# PATTERN 5: Explicit Action Router (Search / Fetch / Save / Finish) with External Executor
# =============================================================================


class P5Search(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['search'] = 'search'
    query: str


class P5Fetch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['fetch'] = 'fetch'
    url: str


class P5Save(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['save'] = 'save'
    doc_id: str
    filename: str


class P5Finish(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['finish'] = 'finish'
    reason: str


P5OutputSpec = [
    ToolOutput(P5Search, strict=True),
    ToolOutput(P5Fetch, strict=True),
    ToolOutput(P5Save, strict=True),
    ToolOutput(P5Finish, strict=True),
]


pattern5_agent: Agent[Deps, P5Search | P5Fetch | P5Save | P5Finish] = Agent(
    get_model(),
    deps_type=Deps,
    output_type=P5OutputSpec,
    instructions=CORE_INSTRUCTIONS
    + """

You must choose EXACTLY one next action each turn:
- P5Search(query=...)
- P5Fetch(url=...)
- P5Save(doc_id=..., filename=...)
- P5Finish(reason=...)

Guidance:
- Use search first to discover candidate URLs.
- Use fetch to get a compact sub-agent analysis (doc_id + suggested filename).
- Use save to persist good docs.
- Finish only when you have enough saved documents or further searching is futile.

NOTE: In this pattern you DO NOT call tools directly. The outer loop executes your chosen action and feeds
back an observation.
""",
    history_processors=[token_limit_history_processor],
)


async def run_pattern5(
    prompt: str, target: int, run_dir: Path, prompt_idx: int, max_iterations: int = 25
) -> AgentResult:
    deps = Deps(
        target_count=target,
        pattern_name='P5-router',
        run_dir=run_dir,
        pattern_dir=run_dir / 'pat5',
        docs_dir=(run_dir / 'pat5' / 'docs'),
        prompt_idx=prompt_idx,
        prompt_text=prompt,
    )
    log = deps.log()

    log.info('Starting Pattern 5: External executor (router outputs actions)')
    start = time.perf_counter()

    messages = None
    iterations = 0
    finish_reason = 'max_iterations_reached'
    last_observation = ''

    for _ in range(max_iterations):
        iterations += 1

        current_prompt = (
            prompt
            if messages is None
            else f"""Continue.

Progress: {len(deps.items)}/{deps.target_count} saved.

Last observation:
{last_observation}
"""
        )

        try:
            result = await pattern5_agent.run(current_prompt, deps=deps, message_history=messages)
        except ModelHTTPError as e:
            if is_context_length_error(e) and messages:
                log.warning('🔥 Context length exceeded, emergency trim and retry')
                messages = emergency_trim_history(messages, max_tokens=60_000)
                result = await pattern5_agent.run(current_prompt, deps=deps, message_history=messages)
            else:
                raise

        messages = result.all_messages()
        out = result.output

        if isinstance(out, P5Search):
            search_res = await _search_impl(deps, out.query)
            top = []
            for r in (search_res.get('results') or [])[:5]:
                url = r.get('url') or r.get('link') or ''
                title = r.get('title') or ''
                if url:
                    top.append(f'- {title} | {url}')
            last_observation = 'Search done. Top results:\n' + ('\n'.join(top) if top else '(no results)')
        elif isinstance(out, P5Fetch):
            fetch_res = await _fetch_impl(deps, out.url)
            analysis = fetch_res.get('analysis') or {}
            last_observation = (
                'Fetch done.\n'
                f'doc_id: {fetch_res.get("doc_id")}\n'
                f'url: {fetch_res.get("url")}\n'
                f'should_save: {analysis.get("should_save")}\n'
                f'suggested_filename: {analysis.get("suggested_filename")}\n'
                f'summary: {str(analysis.get("file_summary") or "")[:400]}'
            )
        elif isinstance(out, P5Save):
            save_msg = await _save_impl(deps, out.doc_id, out.filename)
            last_observation = save_msg
        elif isinstance(out, P5Finish):
            finish_reason = out.reason
            break
        else:
            last_observation = f'Unknown output: {out!r}'

        if len(deps.items) >= deps.target_count:
            last_observation += '\n\nTarget reached. You may finish.'

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Iterations: {iterations} | Items: {len(deps.items)}')

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=iterations,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


# =============================================================================
# RESULTS PERSISTENCE (RUN_DIR/date_hour/patX)
# =============================================================================


def _make_run_dir() -> Path:
    """Create the run directory named date_hour, with pat1..pat5 subdirs."""
    date_hour = datetime.now().strftime('%Y%m%d_%H')
    base = Path(date_hour)

    # Avoid collisions if you run multiple times within the same hour.
    if base.exists():
        for i in range(1, 1000):
            candidate = Path(f'{date_hour}_{i:02d}')
            if not candidate.exists():
                base = candidate
                break

    for i in range(1, 6):
        (base / f'pat{i}' / 'docs').mkdir(parents=True, exist_ok=True)

    return base


def save_pattern_result(run_dir: Path, pattern_num: int, prompt_idx: int, result: AgentResult) -> int:
    """Save a pattern's result summary + item metadata under run_dir/pat{n}."""
    pattern_dir = run_dir / f'pat{pattern_num}'
    pattern_dir.mkdir(parents=True, exist_ok=True)

    # Save item metadata (documents themselves are written by save()).
    items_file = pattern_dir / f'prompt_{prompt_idx:02d}_items.json'
    with items_file.open('w', encoding='utf-8') as f:
        json.dump(result.items_collected, f, indent=2, ensure_ascii=False)

    total_size = sum(int(it.get('file_size_bytes') or 0) for it in result.items_collected)

    summary_file = pattern_dir / f'prompt_{prompt_idx:02d}_summary.txt'
    lines = []
    lines.append(f'Prompt Index: {prompt_idx}')
    lines.append(f'Elapsed Seconds: {result.elapsed_seconds:.3f}')
    lines.append(f'Iterations: {result.iterations}')
    lines.append(f'Finish Reason: {result.finish_reason}')
    lines.append(f'Items Collected: {len(result.items_collected)}')
    lines.append(f'Actions Taken: {len(result.actions_taken)}')
    lines.append(f'Total Saved Size: {total_size:,} bytes ({total_size / 1024:.2f} KB)')
    lines.append('')
    lines.append('Items:')
    for it in result.items_collected:
        saved_path = it.get('saved_path', '')
        size = it.get('file_size_bytes', 0)
        url = it.get('url', '')
        lines.append(f'  - {saved_path} ({size:,} bytes) | {url}')
    lines.append('')
    lines.append('Actions:')
    for action in result.actions_taken:
        lines.append(f'  - {action}')

    summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    return total_size


def append_to_csv(run_dir: Path, prompt_idx: int, prompt: str, times: dict[int, float], sizes: dict[int, int]) -> None:
    """Append timing/size results to run_dir/timing_results.csv."""
    csv_file = run_dir / 'timing_results.csv'
    file_exists = csv_file.exists()

    with csv_file.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                'prompt_num',
                'pattern1_sec',
                'pattern2_sec',
                'pattern3_sec',
                'pattern4_sec',
                'pattern5_sec',
                'pattern1_bytes',
                'pattern2_bytes',
                'pattern3_bytes',
                'pattern4_bytes',
                'pattern5_bytes',
                'prompt_text',
            ])

        writer.writerow([
            f'prompt_{prompt_idx:02d}',
            times.get(1, 0.0),
            times.get(2, 0.0),
            times.get(3, 0.0),
            times.get(4, 0.0),
            times.get(5, 0.0),
            sizes.get(1, 0),
            sizes.get(2, 0),
            sizes.get(3, 0),
            sizes.get(4, 0),
            sizes.get(5, 0),
            prompt[:200] + '...' if len(prompt) > 200 else prompt,
        ])


def save_prompt_comparison(run_dir: Path, prompt_idx: int, prompt: str, results: dict[str, AgentResult]) -> None:
    """Write a consolidated comparison txt for this prompt at the run root."""
    out = run_dir / f'prompt_{prompt_idx:02d}_comparison.txt'
    lines = []
    lines.append('=' * 90)
    lines.append(f'PROMPT {prompt_idx:02d}')
    lines.append('=' * 90)
    lines.append(prompt)
    lines.append('')
    lines.append(f'{"Pattern":<18} {"Items":<6} {"Actions":<8} {"Iters":<6} {"Time(s)":<10} Finish')
    lines.append('-' * 90)

    order = [
        ('Pattern 1', 'Pattern 1: Implicit output_type'),
        ('Pattern 2', 'Pattern 2: Explicit finish tool'),
        ('Pattern 3', 'Pattern 3: iter() API'),
        ('Pattern 4', 'Pattern 4: Union Continue/Finish'),
        ('Pattern 5', 'Pattern 5: Router actions'),
    ]

    for label, key in order:
        r = results.get(key)
        if not r:
            continue
        lines.append(
            f'{label:<18} {len(r.items_collected):<6} {len(r.actions_taken):<8} {r.iterations:<6} {r.elapsed_seconds:<10.2f} {r.finish_reason}'
        )

    lines.append('')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# =============================================================================
# TEST HARNESS
# =============================================================================


async def run_comparison(run_dir: Path, prompt: str, target: int, prompt_idx: int) -> dict[str, AgentResult]:
    """Run all 5 patterns with the same prompt and compare results."""
    logger.info('=' * 70)
    logger.info(f'RUN DIR: {run_dir}')
    logger.info(f'PROMPT {prompt_idx:02d}: {prompt}')
    logger.info(f'TARGET: {target} items')
    logger.info('=' * 70)

    patterns: list[tuple[str, Callable[..., Any], int]] = [
        ('Pattern 1: Implicit output_type', run_pattern1, 1),
        ('Pattern 2: Explicit finish tool', run_pattern2, 2),
        ('Pattern 3: iter() API', run_pattern3, 3),
        ('Pattern 4: Union Continue/Finish', run_pattern4, 4),
        ('Pattern 5: Router actions', run_pattern5, 5),
    ]

    results: dict[str, AgentResult] = {}
    timing_data: dict[int, float] = {}
    size_data: dict[int, int] = {}

    for name, runner, pattern_num in patterns:
        logger.info(f'\n{"─" * 70}')
        logger.info(f'▶ {name}')
        logger.info('─' * 70)

        try:
            result: AgentResult = await runner(prompt, target, run_dir, prompt_idx)
            results[name] = result
            timing_data[pattern_num] = result.elapsed_seconds

            total_size = save_pattern_result(run_dir, pattern_num, prompt_idx, result)
            size_data[pattern_num] = total_size
        except Exception as e:
            logger.exception(f'❌ {name} failed: {e}')
            timing_data[pattern_num] = 0.0
            size_data[pattern_num] = 0

    append_to_csv(run_dir, prompt_idx, prompt, timing_data, size_data)
    save_prompt_comparison(run_dir, prompt_idx, prompt, results)

    return results


async def run_single_pattern(
    pattern: Literal[1, 2, 3, 4, 5], run_dir: Path, prompt: str, target: int = 5, prompt_idx: int = 1
) -> AgentResult:
    """Run a single pattern for isolated testing."""
    runners: dict[int, Callable[..., Any]] = {
        1: run_pattern1,
        2: run_pattern2,
        3: run_pattern3,
        4: run_pattern4,
        5: run_pattern5,
    }
    return await runners[pattern](prompt, target, run_dir, prompt_idx)


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    run_dir = _make_run_dir()
    logger.info(f'Created run directory: {run_dir}')

    try:
        for idx, prompt in enumerate(PROMPTS, start=1):
            await run_comparison(run_dir, prompt, target=5, prompt_idx=idx)
            logger.info('\n' * 2)
    finally:
        await close_httpx_client()
        logger.info('Closed httpx client')


if __name__ == '__main__':
    uvloop.run(main())
