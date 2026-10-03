"""Read-only remote collection of GPU-001 results; never starts compute."""
import argparse
import hashlib
import json
from pathlib import Path

import modal


def preserve_bytes(path, payload, sha256=None, size=None):
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError('Artifact checksum mismatch: '+str(path))
    if size is not None and len(payload) != size:
        raise ValueError('Artifact size mismatch: '+str(path))
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to replace different local evidence: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def collect(root):
    receipt = json.loads((root/'benchmark_run.json').read_text())
    protocol = json.loads((root/'preregister.json').read_text())
    volume = modal.Volume.from_name('chess-dfm-training-results', environment_name='chess-dfm-research')
    collected = {}
    for gpu, call_id in receipt['calls'].items():
        call = modal.FunctionCall.from_id(call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            print(json.dumps({'gpu':gpu, 'status':'result_not_ready', 'call_id':call_id}), flush=True)
            continue
        raw = b''.join(volume.read_file(result['result_relative']))
        folder = root/gpu
        preserve_bytes(folder/'result.json', raw, result['result_sha256'])
        document = json.loads(raw)
        if document['gpu'] != gpu or document['protocol'] != protocol:
            raise ValueError('Result/protocol mismatch')
        remote = str(Path(result['result_relative']).parent)
        for name, identity in document['artifacts'].items():
            if Path(name).name != name:
                raise ValueError('Nonlocal artifact name')
            local = folder/name
            if local.exists():
                payload = local.read_bytes()
            else:
                payload = b''.join(volume.read_file(remote+'/training/'+name))
            preserve_bytes(local, payload, identity['sha256'], identity['size_bytes'])
        for name in ('launch.json', 'stdout.log'):
            preserve_bytes(folder/name, b''.join(volume.read_file(remote+'/'+name)))
        collected[gpu] = {'call_id':call_id, **result, 'artifacts_verified':len(document['artifacts'])}
        print(json.dumps({'gpu':gpu, 'status':'collected_and_verified', **collected[gpu]}), flush=True)
    if len(collected) == 2:
        preserve_bytes(root/'completion.json', (json.dumps(collected, indent=2, sort_keys=True)+'\n').encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    collect(parser.parse_args().root)
