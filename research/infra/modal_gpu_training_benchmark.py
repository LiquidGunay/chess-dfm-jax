"""Budget-gated CPU verification and paired L40S/H100 training benchmark."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import modal
from research.infra import modal_train_app as base
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

app = modal.App('chess-dfm-gpu001-benchmark')
SOURCES = ('research/gpu_training_benchmark.py', 'research/infra/modal_gpu_training_benchmark.py',
           'research/infra/modal_train_app.py', 'research/interpretability/remote_cost.py')
VOLUMES = {base.BASE_INPUT_MOUNT: base.base_input_volume.with_mount_options(read_only=True),
           base.TRAINING_RESULT_MOUNT: base.training_result_volume}


def source_identity(root):
    return base._tree_sha256(Path(root), relative_paths=SOURCES)


def spec(gpu=None):
    return ModalJobSpec(name='GPU-001-'+(gpu or 'preflight'), timeout_seconds=1200 if gpu else 600,
                       startup_timeout_seconds=600 if gpu else 300, retries=0,
                       cpu_cores=Decimal(4), memory_gib=Decimal(32 if gpu else 16), gpu=gpu)


def output_root(protocol):
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return Path(base.TRAINING_RESULT_MOUNT)/'benchmarks'/('gpu001-'+digest[:20])


def verify_source(protocol, root):
    root = Path(root)
    if source_identity(root) != protocol['source_sha256']:
        raise ValueError('Benchmark source drift')
    for name, expected in protocol['frozen_training_sources'].items():
        if base._sha256_file(root/name) != expected:
            raise ValueError('Frozen training source differs: '+name)


def environment(protocol, orchestration_commit, cache):
    env = os.environ.copy()
    env.update(CHESS_DFM_WORKSPACE_ROOT='/workspace',
        CHESS_DFM_TRUSTED_WORKSPACE_ROOTS=os.pathsep.join(str(Path(p).resolve(strict=True))
            for p in (base.BASE_INPUT_MOUNT, base.TRAINING_RESULT_MOUNT)),
        # Legacy checkpoint contracts bind the training-source commit. Every
        # frozen training file is verified against that commit before use;
        # current orchestration identity is recorded separately, not concealed.
        CHESS_DFM_GIT_COMMIT=protocol['training_source_commit'],
        CHESS_DFM_SOURCE_TREE_SHA256=protocol['source_sha256'],
        GPU001_ORCHESTRATION_COMMIT=orchestration_commit,
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), TRITON_CACHE_DIR=str(cache/'triton'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_MAX_THREADS='2')
    return env


@app.function(image=base.gpu_image, cpu=4, memory=16384, timeout=600, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def preflight(protocol, orchestration_commit, cost):
    os.environ.update(environment(protocol, orchestration_commit, Path('/tmp/gpu001-preflight')))
    from research.hero_history_data import verify_dataset, write_json_once
    from research.hero_history_batches import HistoryTrajectoryBatches
    from research.input_contract import audit_full_history_batch
    from research.train_torch import canonicalize_trajectory_batch
    verify_source(protocol, '/root')
    raw = base._verify_raw_bt4_volume()
    if base._sha256_file(raw) != protocol['raw_bt4_sha256']:
        raise ValueError('Raw model mismatch')
    data = (Path(base.TRAINING_RESULT_MOUNT)/protocol['dataset_relative']).resolve(strict=True)
    if base._sha256_file(data/'manifest.json') != protocol['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest mismatch')
    verify_dataset(data)  # Full byte hashes, not a first-batch-only gate.
    checkpoint = Path(base.TRAINING_RESULT_MOUNT)/protocol['checkpoint_relative']
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    if manifest['optimizer_update'] != 500 or manifest['next_data_cursor'] != 500 or not manifest['optimizer_resume_supported']:
        raise ValueError('Not the frozen resume checkpoint')
    state = checkpoint/manifest['state']['path']
    if state.stat().st_size != manifest['state']['size_bytes'] or base._sha256_file(state) != protocol['checkpoint_state_sha256']:
        raise ValueError('Recovery state mismatch')
    if manifest['resume_contract']['git_commit'] != protocol['training_source_commit']:
        raise ValueError('Legacy training-source identity mismatch')
    batches = HistoryTrajectoryBatches(data, split='train', batch_size=1024, horizon=8,
        manifest_sha256=protocol['dataset_manifest_sha256'], canonicalize=canonicalize_trajectory_batch)
    if batches.available_steps != 553 or batches.provenance() != manifest['resume_contract']['data']:
        raise ValueError('Checkpoint data contract differs')
    audits = []
    for cursor in [0, *range(500,553)]:
        audits.append({'cursor': cursor, **audit_full_history_batch(batches.batch_at(cursor), allow_chess960_initial=True)})
    out = output_root(protocol); out.mkdir(parents=True, exist_ok=False)
    result = {'passed': True, 'gpu_active': False, 'protocol': protocol,
              'orchestration_commit': orchestration_commit, 'declared_cost': cost,
              'output_relative': str(out.relative_to(base.TRAINING_RESULT_MOUNT)),
              'checkpoint_manifest_sha256': base._sha256_file(checkpoint/'manifest.json'),
              'input_history_audits': audits, 'full_dataset_hashes_verified': True}
    write_json_once(out/'preflight.json', result); base.training_result_volume.commit()
    return result


def run_benchmark(gpu, protocol, preflight_sha, orchestration_commit, cost):
    from research.hero_history_data import write_json_once
    from research.gpu_training_benchmark import command, summarize
    verify_source(protocol, '/root')
    out = output_root(protocol)
    if base._sha256_file(out/'preflight.json') != preflight_sha:
        raise ValueError('CPU preflight drift')
    pre = json.loads((out/'preflight.json').read_text())
    if pre['protocol'] != protocol or not pre['passed'] or pre['gpu_active']:
        raise ValueError('CPU input verification required')
    env = environment(protocol, orchestration_commit, Path('/tmp')/('gpu001-'+gpu))
    os.environ.update(env)
    import torch
    if not torch.cuda.is_available() or gpu not in torch.cuda.get_device_name(0):
        raise ValueError('Unexpected GPU')
    data = Path(base.TRAINING_RESULT_MOUNT)/protocol['dataset_relative']
    checkpoint = Path(base.TRAINING_RESULT_MOUNT)/protocol['checkpoint_relative']
    if base._sha256_file(data/'manifest.json') != protocol['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest drift')
    if base._sha256_file(checkpoint/'manifest.json') != pre['checkpoint_manifest_sha256']:
        raise ValueError('Recovery manifest drift')
    raw = base._verify_raw_bt4_volume()
    root = out/gpu; root.mkdir(exist_ok=False)
    output = root/'training'
    cmd = command(data, protocol['dataset_manifest_sha256'], raw, checkpoint, output)
    write_json_once(root/'launch.json', {'gpu':gpu, 'command':cmd, 'protocol':protocol,
        'orchestration_commit':orchestration_commit, 'trainer_contract_commit':protocol['training_source_commit'],
        'device_name':torch.cuda.get_device_name(0), 'declared_cost':cost, 'preflight_sha256':preflight_sha})
    base.training_result_volume.commit()
    began = time.monotonic()
    process = subprocess.Popen(cmd, cwd='/root', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_commit = began
    try:
        with (root/'stdout.log').open('x') as log:
            for line in process.stdout:
                log.write(line); log.flush(); print(gpu+': '+line, end='', flush=True)
                if time.monotonic()-last_commit > 60:
                    base.training_result_volume.commit(); last_commit=time.monotonic()
        if process.wait() != 0:
            raise RuntimeError('Benchmark trainer failed: '+gpu)
        records = [json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines()]
        report = json.loads((output/'report.json').read_text())
        if report['checkpoint'] is not None or report['recovery_checkpoints'] or report['segment_start_update'] != 500:
            raise ValueError('Disposable benchmark wrote a model or resumed incorrectly')
        result = {'gpu':gpu, 'device_name':torch.cuda.get_device_name(0), 'measurement':summarize(records),
                  'seconds_with_load_compile_and_profiler':time.monotonic()-began,
                  'profile':json.loads((output/'profile_summary.json').read_text()),
                  'compile_counters':report['compile_counters'], 'gpu_monitor':report['gpu_monitor'],
                  'protocol':protocol, 'orchestration_commit':orchestration_commit, 'declared_cost':cost,
                  'output_relative':str(root.relative_to(base.TRAINING_RESULT_MOUNT))}
        result['artifacts'] = {p.name:{'sha256':base._sha256_file(p),'size_bytes':p.stat().st_size}
                               for p in output.iterdir() if p.is_file()}
        write_json_once(root/'result.json', result)
        return {'result_relative':str((root/'result.json').relative_to(base.TRAINING_RESULT_MOUNT)),
                'result_sha256':base._sha256_file(root/'result.json')}
    except Exception as exc:
        write_json_once(root/'failure.json', {'type':type(exc).__name__, 'message':str(exc)})
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        base.training_result_volume.commit()


@app.function(image=base.gpu_image, gpu='L40S', cpu=4, memory=32768, timeout=1200, startup_timeout=600,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def l40s(protocol, preflight_sha, orchestration_commit, cost):
    return run_benchmark('L40S', protocol, preflight_sha, orchestration_commit, cost)


@app.function(image=base.gpu_image, gpu='H100', cpu=4, memory=32768, timeout=1200, startup_timeout=600,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def h100(protocol, preflight_sha, orchestration_commit, cost):
    return run_benchmark('H100', protocol, preflight_sha, orchestration_commit, cost)


@app.local_entrypoint()
def main(protocol_path: str, stage: str, recorded_spend: str):
    from research.infra.modal_eval_matrix import atomic_json
    path=Path(protocol_path); protocol=json.loads(path.read_text())
    verify_source(protocol, Path.cwd())
    if stage not in {'preflight','benchmark'}:
        raise ValueError('Unknown stage')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Commit inputs before launch')
    receipt_path=path.parent/(stage+'_run.json')
    if receipt_path.exists():
        raise ValueError('Attempt already exists; no automatic retry')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    spend=max(Decimal(recorded_spend),base._modal_launch_spend())
    if stage == 'preflight':
        decision=enforce_modal_budget(estimate_modal_job(spec()),recorded_month_spend=spend,per_job_cap=Decimal('.08'))
        atomic_json(receipt_path,{'status':'launching','cost_guard':decision,'git_commit':commit})
        result=preflight.remote(protocol,commit,decision['declared_attempt_upper_bound'])
        atomic_json(path.parent/'preflight.json',result)
        atomic_json(receipt_path,{'status':'complete','cost_guard':decision,'git_commit':commit,'output_relative':result['output_relative']})
        print(json.dumps({'preflight_complete':result['output_relative']}),flush=True)
    else:
        pre=json.loads((path.parent/'preflight.json').read_text())
        raw=b''.join(base.training_result_volume.read_file(pre['output_relative']+'/preflight.json'))
        if json.loads(raw) != pre or not pre['passed'] or pre['gpu_active'] or pre['protocol'] != protocol:
            raise ValueError('Matching completed CPU preflight required before GPU allocation')
        decisions={}
        for gpu in ('L40S','H100'):
            decisions[gpu]=enforce_modal_budget(estimate_modal_job(spec(gpu)),recorded_month_spend=spend,
                                                per_job_cap=Decimal('1.20' if gpu=='L40S' else '2.20'))
            spend+=Decimal(decisions[gpu]['declared_attempt_upper_bound'])
        receipt={'status':'launching','cost_guards':decisions,'git_commit':commit,'calls':{}}
        atomic_json(receipt_path,receipt)
        for gpu, fn in (('L40S',l40s),('H100',h100)):
            call=fn.spawn(protocol,hashlib.sha256(raw).hexdigest(),commit,decisions[gpu]['declared_attempt_upper_bound'])
            receipt['calls'][gpu]=call.object_id; atomic_json(receipt_path,receipt)
        receipt['status']='submitted'; atomic_json(receipt_path,receipt)
        print(json.dumps(receipt),flush=True)
