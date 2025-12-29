"""Four agentic loop patterns for comparison testing.
Each pattern receives the same prompt and produces a standardized result.

Features:
- Jina AI Search and Reader API integration (correct POST requests)
- Proper logging with logfire
- Time to completion tracking
- Cerebras model with zai-glm-4.6

Run with: uv run python agent_patterns.py

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

import csv
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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

logger.configure()

logger.instrument_httpx(capture_all=True)
load_dotenv()
MAX_HISTORY_TOKENS = 120_000
CHARS_PER_TOKEN = 4  # Conservative estimate for English text
JINA_API_KEY = os.getenv('JINA_API_KEY')
JINA_SEARCH_URL = 'https://s.jina.ai/'  # POST with {"q": "..."}
JINA_READER_URL = 'https://r.jina.ai/'  # POST with {"url": "..."}

_encoding: tiktoken.Encoding | None = None


CORE_INSTRUCTIONS = """
You are a document collector. Your task is to collect items by:
1. Searching for relevant items using Jina AI Search
2. Fetching promising URLs to get content
3. Saving good items to your collection

Continue until you've collected enough items (check save responses for progress)
OR you've exhausted all reasonable search strategies.

Be thorough - try multiple search queries before concluding.
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


def _estimate_tokens(msg: Any) -> int:
    """Count tokens for a message using tiktoken."""
    try:
        if hasattr(msg, 'model_dump_json'):
            content = msg.model_dump_json()
        else:
            content = str(msg)
        return len(_get_encoding().encode(content))
    except Exception:
        return 100  # Fallback if encoding fails


def is_context_length_error(e: Exception) -> bool:
    """Check if exception is a context length exceeded error."""
    if isinstance(e, ModelHTTPError) and e.status_code == 400:
        body_str = str(e.body).lower()
        return 'context_length' in body_str or 'context length' in body_str
    return False


def emergency_trim_history(messages: list[Any], keep_last: int = 3) -> list[Any]:
    """Emergency trim when 400 error occurs - much more aggressive."""
    if not messages or len(messages) <= keep_last + 1:
        return messages

    safe_indices = _find_safe_cut_indices(messages)
    target_start = len(messages) - keep_last

    for idx in reversed(safe_indices):
        if idx >= target_start:
            return [messages[0]] + messages[idx:]

    return [messages[0]] + messages[-keep_last:]


def _get_encoding() -> tiktoken.Encoding:
    """Get or create the tiktoken encoding (lazy init)."""
    global _encoding
    if _encoding is None:
        # cl100k_base works well as general-purpose tokenizer
        _encoding = tiktoken.get_encoding('cl100k_base')
    return _encoding


def _count_tokens(msg: Any) -> int:
    """Count tokens for a message using tiktoken."""
    try:
        if hasattr(msg, 'model_dump_json'):
            content = msg.model_dump_json()
        else:
            content = str(msg)
        return len(_get_encoding().encode(content))
    except Exception:
        return 100  # Fallback if encoding fails


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


def _find_safe_cut_indices(messages: list[Any]) -> list[int]:
    """Find indices where it's safe to cut (not breaking tool call/return pairs).

    Safe cut points are BEFORE messages that:
    - Are ModelRequests without ToolReturnParts, OR
    - Are ModelResponses without ToolCallParts that need pairing

    We never cut at index 0 (system prompt).
    """
    safe = []
    n = len(messages)

    for i in range(1, n):  # Skip index 0 (system prompt)
        msg = messages[i]
        prev_msg = messages[i - 1] if i > 0 else None

        # If this message has tool returns, the previous MUST have tool calls
        # So cutting here would break the pair - NOT safe
        if _has_tool_returns(msg):
            continue

        # If previous message has tool calls, we need this message to have returns
        # Since we already checked above, if we're here and prev has calls, skip
        if prev_msg and _has_tool_calls(prev_msg):
            continue

        # Safe to cut here
        safe.append(i)

    return safe


async def token_limit_history_processor(ctx: RunContext[Deps], messages: list[Any]) -> list[Any]:
    """Limit message history to ~120k tokens while preserving tool call/return pairs.

    Strategy:
    1. Estimate total tokens
    2. If under limit, return as-is
    3. Otherwise, find safe cut points and remove oldest messages
    4. Always keep first message (system prompt) and recent messages
    """
    if not messages:
        return messages

    # Calculate current token estimate
    token_estimates = [_estimate_tokens(m) for m in messages]
    total_tokens = sum(token_estimates)

    if total_tokens <= MAX_HISTORY_TOKENS:
        return messages

    log = ctx.deps.log()
    original_count = len(messages)
    log.info(
        f'⚠️  History exceeds limit: ~{total_tokens:,} tokens > {MAX_HISTORY_TOKENS:,} limit '
        f'({original_count} messages)'
    )

    # Find safe cut points
    safe_indices = _find_safe_cut_indices(messages)

    if not safe_indices:
        # No safe cuts found - keep first and last few messages
        log.warning('No safe cut points found - keeping first + last 5 messages')
        if len(messages) > 6:
            return [messages[0]] + messages[-5:]
        return messages

    # Binary search for the right cut point
    # We want to keep messages from cut_index onwards (plus first message)
    result = list(messages)

    for cut_idx in safe_indices:
        # Try cutting everything from index 1 to cut_idx
        candidate = [messages[0]] + messages[cut_idx:]
        candidate_tokens = _estimate_tokens(messages[0]) + sum(token_estimates[cut_idx:])

        if candidate_tokens <= MAX_HISTORY_TOKENS:
            result = candidate
            break
    else:
        # Even cutting at the last safe point isn't enough
        # Take first message + everything from last safe index
        if safe_indices:
            last_safe = safe_indices[-1]
            result = [messages[0]] + messages[last_safe:]

    final_tokens = sum(_estimate_tokens(m) for m in result)
    removed = original_count - len(result)

    log.info(
        f'✂️  Trimmed history: {original_count} → {len(result)} messages '
        f'(~{total_tokens:,} → ~{final_tokens:,} tokens, removed {removed})'
    )

    return result


# Synchronous wrapper for agents that might not use async processors
def token_limit_history_processor_sync(messages: list[Any]) -> list[Any]:
    """Sync version without ctx - uses simple token estimation.
    For use when RunContext isn't needed.
    """
    if not messages:
        return messages

    token_estimates = [_estimate_tokens(m) for m in messages]
    total_tokens = sum(token_estimates)

    if total_tokens <= MAX_HISTORY_TOKENS:
        return messages

    safe_indices = _find_safe_cut_indices(messages)

    if not safe_indices:
        if len(messages) > 6:
            return [messages[0]] + messages[-5:]
        return messages

    for cut_idx in safe_indices:
        candidate = [messages[0]] + messages[cut_idx:]
        candidate_tokens = _estimate_tokens(messages[0]) + sum(token_estimates[cut_idx:])

        if candidate_tokens <= MAX_HISTORY_TOKENS:
            return candidate

    if safe_indices:
        return [messages[0]] + messages[safe_indices[-1] :]

    return messages


async def close_httpx_client() -> None:
    """Close the shared httpx client."""
    global _httpx_client
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


def get_model() -> CerebrasModel:
    """Create the Cerebras model with reasoning disabled."""
    return CerebrasModel('zai-glm-4.6', settings=CerebrasModelSettings(cerebras_disable_reasoning=True))


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
class Deps:
    """Shared state across tool calls."""

    items: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    target_count: int = 5
    pattern_name: str = 'unknown'

    # Pattern 2 specific
    is_finished: bool = False
    finish_reason: str = ''

    def log(self) -> Any:
        """Get a logger bound to this pattern."""
        return logger.with_tags(self.pattern_name)


# =============================================================================
# JINA AI SEARCH IMPLEMENTATION (FIXED - POST request)
# =============================================================================
_httpx_client: httpx.AsyncClient | None = None


async def get_httpx_client() -> httpx.AsyncClient:
    """Get or create a shared async httpx client."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),  # Longer timeout for Jina
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _httpx_client


async def jina_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Perform a Jina AI search using the correct POST API.

    Per Jina docs:
    - Endpoint: https://s.jina.ai/
    - Method: POST
    - Body: {"q": "search query"}
    - Headers: Authorization, Content-Type, Accept

    Args:
        query: Search query string
        max_results: Maximum number of results (passed as 'num' parameter)

    Returns:
        Parsed Jina response with search results
    """
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if JINA_API_KEY:
            headers['Authorization'] = f'Bearer {JINA_API_KEY}'

        # Jina Search API request body
        request_body = {'q': query, 'num': max_results}

        client = await get_httpx_client()
        response = await client.post(JINA_SEARCH_URL, headers=headers, json=request_body)

        # Log response status for debugging
        logger.info(f"Jina search '{query}': status={response.status_code}")

        if response.status_code != 200:
            logger.warning(f'Jina search non-200: {response.status_code} - {response.text[:500]}')
            return {'error': f'HTTP {response.status_code}', 'query': query, 'results': []}

        # Parse JSON response
        data = response.json()

        # Jina response structure: {"code": 200, "status": 20000, "data": [...]}
        if 'data' in data:
            results = data['data']
            logger.info(f"Jina search returned {len(results)} results for '{query}'")
            return {'query': query, 'results': results, 'count': len(results)}
        logger.warning(f'Jina search unexpected response structure: {list(data.keys())}')
        return {'query': query, 'results': [], 'raw_response': data}

    except httpx.HTTPStatusError as e:
        logger.warning(f"Jina search HTTP error for '{query}': {e}")
        return {'error': str(e), 'query': query, 'results': []}
    except httpx.RequestError as e:
        logger.warning(f"Jina search request error for '{query}': {e}")
        return {'error': str(e), 'query': query, 'results': []}
    except json.JSONDecodeError as e:
        logger.warning(f"Jina search JSON decode error for '{query}': {e}")
        return {'error': f'JSON decode error: {e}', 'query': query, 'results': []}
    except Exception as e:
        logger.exception(f"Jina search unexpected error for '{query}': {e}")
        return {'error': str(e), 'query': query, 'results': []}


# =============================================================================
# JINA AI READER/FETCH IMPLEMENTATION (FIXED - POST request)
# =============================================================================


async def jina_fetch(url: str) -> dict[str, Any]:
    """Fetch URL content using Jina AI Reader API (correct POST method).

    Per Jina docs:
    - Endpoint: https://r.jina.ai/
    - Method: POST
    - Body: {"url": "https://example.com"}
    - Headers: Authorization, Content-Type, Accept

    Args:
        url: The URL to fetch and convert to markdown

    Returns:
        Dict with url, content, title, and metadata
    """
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if JINA_API_KEY:
            headers['Authorization'] = f'Bearer {JINA_API_KEY}'

        # Jina Reader API request body
        request_body = {'url': url}

        client = await get_httpx_client()
        response = await client.post(JINA_READER_URL, headers=headers, json=request_body)

        logger.info(f"Jina fetch '{url[:50]}...': status={response.status_code}")

        if response.status_code != 200:
            logger.warning(f'Jina fetch non-200: {response.status_code}')
            return {
                'url': url,
                'content': '',
                'status': 'error',
                'status_code': response.status_code,
                'error': response.text[:500],
            }

        # Parse JSON response
        data = response.json()

        # Jina Reader response: {"code": 200, "status": 20000, "data": {"title": ..., "content": ..., ...}}
        if 'data' in data:
            page_data = data['data']
            content = page_data.get('content', '')
            title = page_data.get('title', '')
            description = page_data.get('description', '')

            # Truncate content if too long
            max_content_len = 8000
            if len(content) > max_content_len:
                content = content[:max_content_len] + '\n\n[... content truncated ...]'

            logger.debug(f'Jina fetch got {len(content)} chars from {url[:40]}')

            return {
                'url': url,
                'title': title,
                'description': description,
                'content': content,
                'content_length': len(page_data.get('content', '')),
                'status': 'success',
                'status_code': 200,
            }
        logger.warning(f'Jina fetch unexpected response: {list(data.keys())}')
        return {
            'url': url,
            'content': '',
            'status': 'error',
            'error': 'Unexpected response structure',
            'raw_response': data,
        }

    except httpx.HTTPStatusError as e:
        logger.warning(f'Jina fetch HTTP error {url[:40]}: {e.response.status_code}')
        return {'url': url, 'content': '', 'status': 'error', 'status_code': e.response.status_code, 'error': str(e)}
    except httpx.RequestError as e:
        logger.warning(f'Jina fetch request error {url[:40]}: {e}')
        return {'url': url, 'content': '', 'status': 'error', 'error': str(e)}
    except json.JSONDecodeError as e:
        logger.warning(f'Jina fetch JSON error {url[:40]}: {e}')
        return {'url': url, 'content': '', 'status': 'error', 'error': f'JSON decode error: {e}'}
    except Exception as e:
        logger.exception(f'Jina fetch unexpected error {url[:40]}: {e}')
        return {'url': url, 'content': '', 'status': 'error', 'error': str(e)}


# =============================================================================
# SHARED TOOL IMPLEMENTATIONS (using fixed Jina functions)
# =============================================================================


async def _search_impl(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search using Jina AI. Returns structured search results."""
    log = ctx.deps.log()
    log.info("🔍 Searching: '%s'", query)

    results = await jina_search(query, max_results=10)
    ctx.deps.actions.append(f'search:{query}')

    return results


async def _fetch_impl(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch URL content using Jina AI Reader API."""
    log = ctx.deps.log()
    log.info(f'📥 Fetching: {url[:50]}...')

    ctx.deps.actions.append(f'fetch:{url[:30]}')
    result = await jina_fetch(url)

    return result


async def _save_impl(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save an item to the collection."""
    log = ctx.deps.log()

    ctx.deps.items.append({'id': item_id, 'content': content[:100]})
    ctx.deps.actions.append(f'save:{item_id[:20]}')

    current = len(ctx.deps.items)
    target = ctx.deps.target_count
    log.info(f"💾 Saved '{item_id[:30]}...' | Progress: {current}/{target}")

    return f'Saved. Progress: {current}/{target} items.'


# =============================================================================
# PATTERN 1: Implicit Done via output_type
# =============================================================================


class Pattern1Result(BaseModel):
    """Final output - model calls this tool when done."""

    items_collected: list[dict[str, Any]] = Field(description='All collected items')
    search_queries_used: list[str] = Field(description='Queries that were tried')
    reason_for_stopping: str = Field(description='Why you decided to stop')


pattern1_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


@pattern1_agent.tool
async def p1_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search for documents using Jina AI Search.

    Args:
        query: Search query string to find relevant documents

    Returns:
        Search results with URLs, titles, and snippets
    """
    return await _search_impl(ctx, query)


@pattern1_agent.tool
async def p1_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch and read content from a URL using Jina AI Reader.

    Args:
        url: Full URL to fetch (e.g., https://example.com/document.pdf)

    Returns:
        Page content as markdown with title and metadata
    """
    return await _fetch_impl(ctx, url)


@pattern1_agent.tool
async def p1_save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save a collected document to the collection.

    Args:
        item_id: Unique identifier for the document (e.g., filename or URL)
        content: Summary or key content from the document

    Returns:
        Progress message showing current collection status
    """
    return await _save_impl(ctx, item_id, content)


async def run_pattern1(prompt: str, target: int = 5) -> AgentResult:
    """Pattern 1: Model loops internally until it produces structured output."""
    deps = Deps(target_count=target, pattern_name='P1-implicit')
    log = deps.log()

    log.info('Starting Pattern 1: Implicit output_type')
    start = time.perf_counter()

    try:
        result = await pattern1_agent.run(prompt, deps=deps)
    except ModelHTTPError as e:
        if is_context_length_error(e):
            log.warning('🔥 Context length exceeded in internal loop, returning partial results')
            elapsed = time.perf_counter() - start
            return AgentResult(
                items_collected=deps.items,
                actions_taken=deps.actions,
                iterations=1,
                finish_reason='context_length_exceeded_partial',
                elapsed_seconds=elapsed,
            )
        raise

    elapsed = time.perf_counter() - start
    log.info(f'✅ Completed in {elapsed:.2f}s | Items: {len(deps.items)}')

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
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


@pattern2_agent.tool
async def p2_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search for documents using Jina AI Search.

    Args:
        query: Search query string to find relevant documents

    Returns:
        Search results with URLs, titles, and snippets
    """
    return await _search_impl(ctx, query)


@pattern2_agent.tool
async def p2_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch and read content from a URL using Jina AI Reader.

    Args:
        url: Full URL to fetch (e.g., https://example.com/document.pdf)

    Returns:
        Page content as markdown with title and metadata
    """
    return await _fetch_impl(ctx, url)


@pattern2_agent.tool
async def p2_save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save a collected document to the collection.

    Args:
        item_id: Unique identifier for the document
        content: Summary or key content from the document

    Returns:
        Progress message showing current collection status
    """
    return await _save_impl(ctx, item_id, content)


@pattern2_agent.tool
async def finish(ctx: RunContext[Deps], reason: str) -> str:
    """Call this when you're done collecting documents.

    Args:
        reason: Why you're finishing (e.g., "collected enough", "no more results")

    Returns:
        Confirmation that the task is marked complete
    """
    log = ctx.deps.log()
    log.info('🏁 Finish called: %s', reason)

    ctx.deps.is_finished = True
    ctx.deps.finish_reason = reason
    return f'Marked as finished: {reason}'


async def run_pattern2(prompt: str, target: int = 5, max_iterations: int = 10) -> AgentResult:
    """Pattern 2: External loop checking for explicit finish tool call."""
    deps = Deps(target_count=target, pattern_name='P2-explicit')
    log = deps.log()

    log.info('Starting Pattern 2: Explicit finish tool')
    start = time.perf_counter()

    messages = None
    iterations = 0

    for i in range(max_iterations):
        iterations += 1
        log.debug('Iteration %s/%s', iterations, max_iterations)

        current_prompt = prompt if messages is None else 'Continue your task.'

        try:
            result = await pattern2_agent.run(current_prompt, deps=deps, message_history=messages)
        except ModelHTTPError as e:
            if is_context_length_error(e) and messages:
                log.warning('🔥 Context length exceeded, emergency trim and retry')
                messages = emergency_trim_history(messages, keep_last=3)
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

pattern3_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),
    instructions=CORE_INSTRUCTIONS,
    history_processors=[token_limit_history_processor],
)


@pattern3_agent.tool
async def p3_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search for documents using Jina AI Search.

    Args:
        query: Search query string to find relevant documents

    Returns:
        Search results with URLs, titles, and snippets
    """
    return await _search_impl(ctx, query)


@pattern3_agent.tool
async def p3_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch and read content from a URL using Jina AI Reader.

    Args:
        url: Full URL to fetch

    Returns:
        Page content as markdown with title and metadata
    """
    return await _fetch_impl(ctx, url)


@pattern3_agent.tool
async def p3_save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save a collected document to the collection.

    Args:
        item_id: Unique identifier for the document
        content: Summary or key content from the document

    Returns:
        Progress message showing current collection status
    """
    return await _save_impl(ctx, item_id, content)


async def run_pattern3(prompt: str, target: int = 5) -> AgentResult:
    """Pattern 3: Using iter() to step through and monitor each node."""
    deps = Deps(target_count=target, pattern_name='P3-iter')
    log = deps.log()

    log.info('Starting Pattern 3: iter() API')
    start = time.perf_counter()

    node_count = 0

    try:
        async with pattern3_agent.iter(prompt, deps=deps) as run:
            async for node in run:
                node_count += 1
                try:
                    node_type = node.__class__.__name__

                    if Agent.is_call_tools_node(node):
                        tool_names = _extract_tool_names(node)
                        log.debug('Node %s: Tools → %s', node_count, tool_names)

                    elif Agent.is_model_request_node(node):
                        log.debug('Node %s: Model request', node_count)

                    else:
                        log.debug('Node %s: %s', node_count, node_type)

                except Exception as e:
                    log.exception('Failed to process node %s: %s', node_count, e)
    except ModelHTTPError as e:
        if is_context_length_error(e):
            log.warning('🔥 Context length exceeded in internal loop, returning partial results')
            elapsed = time.perf_counter() - start
            return AgentResult(
                items_collected=deps.items,
                actions_taken=deps.actions,
                iterations=node_count,
                finish_reason='context_length_exceeded_partial',
                elapsed_seconds=elapsed,
            )
        raise

    # Safely access run.result
    result = getattr(run, 'result', None)
    elapsed = time.perf_counter() - start

    # Determine finish reason
    finish_reason = 'no_output'
    if result is not None:
        out = getattr(result, 'output', None)
        if out is not None:
            finish_reason = getattr(out, 'reason_for_stopping', None) or getattr(out, 'reason', None) or str(out)

    log.info(f'✅ Completed in {elapsed:.2f}s | Nodes: {node_count} | Items: {len(deps.items)}')

    return AgentResult(
        items_collected=deps.items,
        actions_taken=deps.actions,
        iterations=node_count,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
    )


def _extract_tool_names(node: Any) -> list[str]:
    """Extract tool names from a CallTools node."""
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


# =============================================================================
# PATTERN 4: Union output for explicit Continue vs Done decision
# =============================================================================


class ContinueAction(BaseModel):
    """Choose this to continue working. Explain your plan."""

    model_config = ConfigDict(extra='forbid')
    decision: Literal['continue'] = 'continue'
    next_steps: list[str] = Field(description='What you plan to do next')
    reasoning: str = Field(description="Why you're continuing")


class FinishAction(BaseModel):
    """Choose this when done collecting."""

    model_config = ConfigDict(extra='forbid')
    decision: Literal['finish'] = 'finish'
    summary: str = Field(description='Summary of what was accomplished')
    reason: str = Field(description="Why you're stopping")


DecisionOutputSpec: list[ToolOutput[ContinueAction] | ToolOutput[FinishAction]] = [
    ToolOutput(ContinueAction, strict=True),
    ToolOutput(FinishAction, strict=True),
]


pattern4_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=DecisionOutputSpec,
    instructions=CORE_INSTRUCTIONS
    + """

After using tools, you must decide: output ContinueAction or FinishAction.
""",
    history_processors=[token_limit_history_processor],
)


@pattern4_agent.tool
async def p4_search(ctx: RunContext[Deps], query: str) -> dict[str, Any]:
    """Search for documents using Jina AI Search.

    Args:
        query: Search query string to find relevant documents

    Returns:
        Search results with URLs, titles, and snippets
    """
    return await _search_impl(ctx, query)


@pattern4_agent.tool
async def p4_fetch(ctx: RunContext[Deps], url: str) -> dict[str, Any]:
    """Fetch and read content from a URL using Jina AI Reader.

    Args:
        url: Full URL to fetch

    Returns:
        Page content as markdown with title and metadata
    """
    return await _fetch_impl(ctx, url)


@pattern4_agent.tool
async def p4_save(ctx: RunContext[Deps], item_id: str, content: str) -> str:
    """Save a collected document to the collection.

    Args:
        item_id: Unique identifier for the document
        content: Summary or key content from the document

    Returns:
        Progress message showing current collection status
    """
    return await _save_impl(ctx, item_id, content)


async def run_pattern4(prompt: str, target: int = 5, max_iterations: int = 10) -> AgentResult:
    """Pattern 4: Model explicitly chooses Continue or Finish each iteration."""
    deps = Deps(target_count=target, pattern_name='P4-union')
    log = deps.log()

    log.info('Starting Pattern 4: Union Continue/Finish')
    start = time.perf_counter()

    messages = None
    iterations = 0
    finish_reason = 'max_iterations_reached'

    for i in range(max_iterations):
        iterations += 1
        log.debug('Iteration %s/%s', iterations, max_iterations)

        current_prompt = prompt if messages is None else 'Continue or finish based on your progress.'

        try:
            result = await pattern4_agent.run(current_prompt, deps=deps, message_history=messages)
        except ModelHTTPError as e:
            if is_context_length_error(e) and messages:
                log.warning('🔥 Context length exceeded, emergency trim and retry')
                messages = emergency_trim_history(messages, keep_last=3)
                result = await pattern4_agent.run(current_prompt, deps=deps, message_history=messages)
            else:
                raise

        messages = result.all_messages()

        match result.output:
            case ContinueAction(reasoning=reason, next_steps=steps):
                log.info('➡️  Continue: %s', reason)
                log.debug('   Next steps: %s', steps)

            case FinishAction(reason=reason, summary=summary):
                log.info('🏁 Finish: %s', reason)
                log.debug('   Summary: %s', summary)
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
# RESULTS PERSISTENCE
# =============================================================================


def setup_results_directories():
    """Create doc_results/pat1-4 directories."""
    base_dir = Path('doc_results')
    for i in range(1, 5):
        (base_dir / f'pat{i}').mkdir(parents=True, exist_ok=True)
    return base_dir


def save_pattern_result(pattern_num: int, prompt_idx: int, result: AgentResult):
    """Save a pattern's result to its directory."""
    result_dir = Path('doc_results') / f'pat{pattern_num}'

    # Save each scraped item as a separate JSON file
    items_saved = []
    for idx, item in enumerate(result.items_collected, start=1):
        item_file = result_dir / f'prompt_{prompt_idx}_item_{idx}.json'
        with Path(item_file).open('w') as f:
            json.dump(item, f, indent=2)

        file_size = item_file.stat().st_size
        items_saved.append({'filename': item_file.name, 'size_bytes': file_size, 'item': item})

    total_size = sum(item['size_bytes'] for item in items_saved)

    # Save detailed result summary
    result_file = result_dir / f'prompt_{prompt_idx}_summary.txt'
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
        summary_content += f'  - {item_info["filename"]} ({item_info["size_bytes"]:,} bytes)\n'

    summary_content += '\nActions:\n'
    for action in result.actions_taken:
        summary_content += f'  - {action}\n'

    with Path(result_file).open('w') as f:
        f.write(summary_content)

    return total_size


def append_to_csv(prompt_idx: int, prompt: str, times: dict[int, float], sizes: dict[int, int]):
    """Append timing results to CSV file."""
    csv_file = Path('doc_results') / 'timing_results.csv'
    file_exists = csv_file.exists()

    with Path(csv_file).open('a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                'prompt_num',
                'pattern1_sec',
                'pattern2_sec',
                'pattern3_sec',
                'pattern4_sec',
                'pattern1_bytes',
                'pattern2_bytes',
                'pattern3_bytes',
                'pattern4_bytes',
                'prompt_text',
            ])

        writer.writerow([
            f'prompt_{prompt_idx}',
            times.get(1, 0.0),
            times.get(2, 0.0),
            times.get(3, 0.0),
            times.get(4, 0.0),
            sizes.get(1, 0),
            sizes.get(2, 0),
            sizes.get(3, 0),
            sizes.get(4, 0),
            prompt[:100] + '...' if len(prompt) > 100 else prompt,
        ])


# =============================================================================
# TEST HARNESS
# =============================================================================


async def run_comparison(prompt: str, target: int = 5, prompt_idx: int = 0) -> dict[str, AgentResult]:
    """Run all 4 patterns with the same prompt and compare results."""
    logger.info('=' * 70)
    logger.info(f'PROMPT: {prompt}')
    logger.info(f'TARGET: {target} items')
    logger.info('=' * 70)

    patterns: list[tuple[str, Callable[..., Any], int]] = [
        ('Pattern 1: Implicit output_type', run_pattern1, 1),
        ('Pattern 2: Explicit finish tool', run_pattern2, 2),
        ('Pattern 3: iter() API', run_pattern3, 3),
        ('Pattern 4: Union Continue/Finish', run_pattern4, 4),
    ]

    results: dict[str, AgentResult] = {}
    timing_data: dict[int, float] = {}
    size_data: dict[int, int] = {}

    for name, runner, pattern_num in patterns:
        logger.info(f'\n{"─" * 70}')
        logger.info(f'▶ {name}')
        logger.info('─' * 70)

        try:
            result = await runner(prompt, target)
            results[name] = result
            timing_data[pattern_num] = result.elapsed_seconds

            total_size = save_pattern_result(pattern_num, prompt_idx, result)
            size_data[pattern_num] = total_size

        except Exception as e:
            logger.exception(f'❌ {name} failed: {e}')
            timing_data[pattern_num] = 0.0
            size_data[pattern_num] = 0

    # Summary comparison
    logger.info(f'\n{"=" * 70}')
    logger.info('COMPARISON SUMMARY')
    logger.info('=' * 70)

    header = f'{"Pattern":<40} {"Items":<8} {"Actions":<10} {"Iters":<8} {"Time(s)":<10}'
    logger.info(header)
    logger.info('-' * 70)

    for name, result in results.items():
        short_name = name.split(':')[0]
        logger.info(
            f'{short_name:<40} '
            f'{len(result.items_collected):<8} '
            f'{len(result.actions_taken):<10} '
            f'{result.iterations:<8} '
            f'{result.elapsed_seconds:<10.2f}'
        )

    append_to_csv(prompt_idx, prompt, timing_data, size_data)

    return results


async def run_single_pattern(pattern: Literal[1, 2, 3, 4], prompt: str, target: int = 5) -> AgentResult:
    """Run a single pattern for isolated testing."""
    runners = {1: run_pattern1, 2: run_pattern2, 3: run_pattern3, 4: run_pattern4}
    return await runners[pattern](prompt, target)


# =============================================================================
# MAIN
# =============================================================================


async def main():
    """Run comparison tests."""
    # Setup result directories
    setup_results_directories()
    logger.info('Created doc_results/pat1-4 directories')

    try:
        for idx, prompt in enumerate(PROMPTS, start=1):
            await run_comparison(prompt, target=5, prompt_idx=idx)
            logger.info('\n' * 2)
    finally:
        await close_httpx_client()
        logger.info('Closed httpx client')


if __name__ == '__main__':
    uvloop.run(main())
