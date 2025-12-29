import argparse
import asyncio
import json
import os
import statistics
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

import config_1_output_type
import config_2_finish_tool
import config_3_iter
import config_4_union_output


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / '.env'
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


PROMPTS: list[str] = [
    # Employment-focused (mix of BR and US)
    'Collect 4 employment-related legal documents (contracts or briefs) from both Brazilian and US law.',
    'Collect 4 documents about labor and employment. Include both Brazilian CLT and American contracts.',
    # Contract-focused
    'Collect 4 contracts: 2 from Brazilian law and 2 from American law. Save diverse types.',
    'Collect 4 commercial contracts or leases from both jurisdictions (brazil and usa).',
    # Brief-focused
    'Collect 4 court briefs: 2 Brazilian petitions and 2 American motions.',
    'Collect 4 legal briefs related to civil litigation from both BR and US courts.',
    # Opinion-focused
    'Collect 4 legal opinions (pareceres or opinion letters) from any jurisdiction.',
    'Collect 4 corporate or tax legal opinions from Brazilian and American law.',
    # Real estate / property
    'Collect 4 documents about real estate (lease, sale, property) from both jurisdictions.',
    'Collect 4 documents about commercial real estate or residential property.',
    # Tax-related
    'Collect 4 tax-related documents (briefs, opinions) from Brazilian and US law.',
    'Collect 4 documents about tax compliance, ICMS, or IRS matters.',
    # Corporate / M&A
    'Collect 4 corporate law documents: mergers, acquisitions, or securities.',
    'Collect 4 M&A-related documents (opinions, agreements) from any jurisdiction.',
    # Mixed queries with specific requirements
    'Collect 4 documents: 1 Brazilian contract, 1 US contract, 1 Brazilian brief, 1 US brief.',
    'Collect 4 documents from Sao Paulo or New York jurisdictions.',
    # Consumer / civil
    'Collect 4 documents related to consumer protection or civil disputes.',
    'Collect 4 documents about appeals, indemnification, or damages.',
    # Compliance / regulatory
    'Collect 4 compliance-related documents (anticorruption, SEC, LGPD).',
    'Collect 4 documents and provide a concise summary of each one.',
]


ConfigRunner = Callable[[str], Coroutine[Any, Any, dict[str, Any]]]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {'n': 0.0, 'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'n': float(len(values)),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
    }


async def _run_one(config_name: str, runner: ConfigRunner, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await runner(prompt)
        elapsed = result.get('seconds', time.perf_counter() - started)
        return {'ok': True, 'config': config_name, 'seconds': elapsed, 'prompt': prompt, 'result': result}
    except Exception as e:
        elapsed = time.perf_counter() - started
        return {'ok': False, 'config': config_name, 'seconds': elapsed, 'prompt': prompt, 'error': repr(e)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=len(PROMPTS))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    _load_dotenv()
    if not os.getenv('CEREBRAS_API_KEY'):
        raise RuntimeError('CEREBRAS_API_KEY is missing. Put it in .env or export it in your shell.')

    configs: list[tuple[str, ConfigRunner]] = [
        ('config_1_output_type', config_1_output_type.run_prompt),
        ('config_2_finish_tool', lambda p: config_2_finish_tool.run_prompt(p, max_iterations=20)),
        ('config_3_iter', config_3_iter.run_prompt),
        ('config_4_union_output', lambda p: config_4_union_output.run_prompt(p, max_iterations=20)),
    ]

    prompts = PROMPTS[: max(0, args.limit)]
    all_runs: list[dict[str, Any]] = []
    for config_name, runner in configs:
        for prompt in prompts:
            all_runs.append(await _run_one(config_name, runner, prompt))

    if args.json:
        print(json.dumps(all_runs, indent=2, ensure_ascii=False))
        return

    console = Console()
    summary = Table(title='Benchmark Summary (zai-glm-4.6)')
    summary.add_column('config')
    summary.add_column('n', justify='right')
    summary.add_column('errors', justify='right')
    summary.add_column('mean_s', justify='right')
    summary.add_column('median_s', justify='right')
    summary.add_column('min_s', justify='right')
    summary.add_column('max_s', justify='right')

    for config_name, _ in configs:
        runs = [r for r in all_runs if r['config'] == config_name]
        ok_seconds = [r['seconds'] for r in runs if r['ok']]
        errors = sum(1 for r in runs if not r['ok'])
        s = _stats(ok_seconds)
        summary.add_row(
            config_name,
            str(int(s['n'])),
            str(errors),
            f'{s["mean"]:.3f}',
            f'{s["median"]:.3f}',
            f'{s["min"]:.3f}',
            f'{s["max"]:.3f}',
        )

    details = Table(title='Per-Run Details')
    details.add_column('config')
    details.add_column('seconds', justify='right')
    details.add_column('ok', justify='right')
    details.add_column('saved', justify='right')
    details.add_column('messages', justify='right')
    details.add_column('error')
    details.add_column('prompt')

    for r in all_runs:
        saved = ''
        messages = ''
        error = ''
        if r['ok']:
            saved = str(r['result'].get('saved_count', ''))
            messages = str(r['result'].get('message_count', ''))
        else:
            error = str(r.get('error', ''))[:120]
        details.add_row(
            r['config'], f'{r["seconds"]:.3f}', 'yes' if r['ok'] else 'no', saved, messages, error, r['prompt']
        )

    console.print(summary)
    console.print(details)


if __name__ == '__main__':
    asyncio.run(main())
