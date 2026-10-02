"""Submit and collect a frozen arena matrix without duplicate launches."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess


def reserved_spend(live: Decimal, baseline: Decimal, runs: dict, cap: Decimal) -> Decimal:
    active = sum(r['status'] in {'submitted', 'launching'} for r in runs.values())
    completed = sum(r['status'] in {'complete', 'failed'} for r in runs.values())
    # Reserving full completed caps also covers meter collection lag.
    return max(live, baseline + completed * cap) + active * cap


def launch_command(protocol: dict, edge: dict, phase: str, spend: Decimal) -> list[str]:
    allocation = protocol['openings'][phase]
    return [
        '/root/.local/bin/modal', 'run', '--detach', '--env', 'chess-dfm-research',
        'research/infra/modal_arena_app.py',
        '--candidate-label', protocol[edge['candidate']]['label'],
        '--opponent-label', protocol[edge['opponent']]['label'],
        '--candidate-policy-mode', edge['candidate_mode'],
        '--opponent-policy-mode', edge['opponent_mode'],
        '--candidate-refinement-passes', str(edge['candidate_passes']),
        '--opponent-refinement-passes', str(edge['opponent_passes']),
        '--pair-count', str(allocation['pairs']),
        '--opening-start-index', str(allocation['start']),
        '--recorded-spend', str(spend),
    ]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def main() -> None:
    import modal
    from research.infra.modal_arena_app import _arena_job_spec, _arena_source_tree_sha256
    from research.infra.modal_train_app import _modal_launch_spend
    from research.interpretability.remote_cost import enforce_modal_budget, estimate_modal_job

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'collect'])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'development'], default='development')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--external-reserved-spend', type=Decimal, default=Decimal('0'),
                        help='Additional full attempt reservation for companion jobs, including meter lag.')
    args = parser.parse_args()
    if not args.external_reserved_spend.is_finite() or args.external_reserved_spend < 0:
        raise ValueError('External budget reservation must be finite and nonnegative')
    root = args.protocol.resolve().parent
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    if args.action == 'launch' and 'tooling' in protocol:
        actual_source = _arena_source_tree_sha256(Path(__file__).resolve().parents[2])
        if actual_source != protocol['tooling']['arena_source_tree_sha256']:
            raise RuntimeError('Arena source changed after protocol freeze')
        preflight_path = root / protocol['tooling']['input_preflight_file']
        preflight_raw = preflight_path.read_bytes()
        if hashlib.sha256(preflight_raw).hexdigest() != protocol['tooling']['input_preflight_sha256']:
            raise RuntimeError('Input preflight evidence changed')
        verified = json.loads(preflight_raw)
        if verified.get('passed') is not True or verified['arena_source_tree_sha256'] != actual_source:
            raise RuntimeError('Missing valid CPU input preflight')
        for name in ('candidate', 'hero1'):
            identity = protocol[name]
            observed = verified['models'][identity['label']]
            if any(observed[key] != identity[key] for key in ('state_sha256', 'plane_history_mode')):
                raise RuntimeError('CPU preflight checkpoint/input contract drift')
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    receipt_path = root / 'runs.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'schema_version': 'chess-dfm-eval-matrix-receipt-v1', 'runs': {},
    }
    if receipt.get('protocol_sha256', protocol_sha) != protocol_sha:
        raise RuntimeError('Protocol changed after submissions')
    receipt['protocol_sha256'] = protocol_sha
    runs = receipt['runs']
    if args.action == 'collect':
        volume = modal.Volume.from_name('chess-dfm-training-results', environment_name='chess-dfm-research')
        for edge_id, run in runs.items():
            if run['status'] != 'submitted':
                continue
            try:
                result = modal.FunctionCall.from_id(run['function_call_id']).get(timeout=0)
            except TimeoutError:
                print(json.dumps({'id': edge_id, 'status': 'running'}), flush=True)
                continue
            except Exception as error:
                run.update(status='failed', error=str(error))
                atomic_json(receipt_path, receipt)
                print(json.dumps({'id': edge_id, **run}), flush=True)
                continue
            label = result['arena_label']
            if not re.fullmatch(r'hero-arena-[a-z0-9-]+', label):
                raise RuntimeError('Unsafe remote arena label')
            raw = b''.join(volume.read_file(f'arenas/{label}/state.json'))
            if hashlib.sha256(raw).hexdigest() != result['state_sha256']:
                raise RuntimeError('Terminal arena state checksum drift')
            destination = root / 'results' / edge_id
            state = json.loads(raw)
            if state['status'] != 'complete':
                raise RuntimeError('Incomplete terminal arena state')
            # Preserve exact downloaded bytes so the remote digest remains valid.
            destination.mkdir(parents=True, exist_ok=True)
            state_path = destination / 'state.json'
            if state_path.exists() and state_path.read_bytes() != raw:
                raise RuntimeError('Refusing to overwrite different terminal evidence')
            state_path.write_bytes(raw)
            atomic_json(destination / 'summary.json', result)
            run.update(status='complete', result=result, collected_utc=datetime.now(UTC).isoformat())
            atomic_json(receipt_path, receipt)
            print(json.dumps({'id': edge_id, 'result': result}), flush=True)
        return

    if not 1 <= args.limit <= protocol['budget']['maximum_simultaneous_attempts']:
        raise ValueError('Submission limit exceeds frozen concurrency')
    if any(r['status'] in {'failed', 'launching'} for r in runs.values()):
        raise RuntimeError('Resolve failed or uncertain submissions before further launches')
    if args.phase == 'development':
        for smoke in protocol['smoke_edges']:
            record = runs.get(smoke['id'], {})
            if record.get('status') != 'complete':
                raise RuntimeError('Both smoke evaluations must complete first')
            state = json.loads((root / 'results' / smoke['id'] / 'state.json').read_text())
            if any(state['aggregate']['fault_counts'].values()):
                raise RuntimeError('Smoke policy faults prevent development launch')
            for model in state['aggregate']['models'].values():
                if model['coverage']['incomplete_coverage_positions']:
                    raise RuntimeError('Smoke move coverage is incomplete')
    cap = Decimal(protocol['budget']['declared_attempt_maximum_dollars'])
    baseline = Decimal(protocol['budget']['launch_month_usage_dollars'])
    for edge in protocol[f'{args.phase}_edges']:
        if edge['id'] in runs:
            continue
        active = sum(r['status'] == 'submitted' for r in runs.values())
        if active >= args.limit:
            break
        spend = reserved_spend(_modal_launch_spend(), baseline, runs, cap) + args.external_reserved_spend
        spec, job_cap = _arena_job_spec(protocol['openings'][args.phase]['pairs'], edge['candidate_passes'], edge['opponent_passes'])
        estimate = estimate_modal_job(spec)
        if Decimal(estimate['declared_attempt_upper_bound_dollars']) != cap:
            raise RuntimeError('Frozen attempt estimate drift')
        decision = enforce_modal_budget(estimate, recorded_month_spend=spend, per_job_cap=job_cap)
        command = launch_command(protocol, edge, args.phase, spend)
        record = {'status': 'launching', 'phase': args.phase, 'cost_guard': decision,
                  'command': command, 'submitted_utc': datetime.now(UTC).isoformat()}
        runs[edge['id']] = record
        atomic_json(receipt_path, receipt)
        print(json.dumps({'id': edge['id'], 'launching': True, 'reserved_spend': str(spend)}), flush=True)
        launched = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        (root / f"launch-{edge['id']}.log").write_text(launched.stdout)
        calls = set(re.findall(r'"function_call_id":\s*"(fc-[a-zA-Z0-9]+)"', launched.stdout))
        apps = set(re.findall(r'/apps/liquidgunay/chess-dfm-research/(ap-[a-zA-Z0-9]+)', launched.stdout))
        if len(calls) != 1 or len(apps) != 1:
            raise RuntimeError(f'Uncertain submission for {edge["id"]}; inspect saved launch log before retrying')
        record.update(status='submitted', function_call_id=calls.pop(), app_id=apps.pop(), launcher_returncode=launched.returncode)
        atomic_json(receipt_path, receipt)
        print(json.dumps({'id': edge['id'], 'function_call_id': record['function_call_id'], 'app_id': record['app_id']}), flush=True)


if __name__ == '__main__':
    main()
