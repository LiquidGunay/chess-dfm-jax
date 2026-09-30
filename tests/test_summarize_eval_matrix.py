import json
from pathlib import Path
import shutil

import pytest

from research.analysis.summarize_eval_matrix import summarize


SOURCE = Path(__file__).resolve().parents[1] / 'research/analysis/ev001_modal_20260930'


def test_frozen_result_totals_and_failed_search_gate():
    result = summarize(SOURCE / 'preregister.json')
    assert result['development_games'] == 2048
    assert result['smoke_games'] == 64
    assert not result['confirmation_entry_passed']
    assert not result['candidate_promoted']
    assert result['confirmation_entry_failures'] == [
        'p1_hero1: primary development score does not exceed 0.5',
    ]
    primary = next(row for row in result['comparisons'] if row['id'] == 'p1_hero1')
    assert primary['candidate']['wins'] == 1
    assert primary['candidate']['draws'] == 31
    assert primary['candidate']['losses'] == 224
    assert primary['score'] == 16.5 / 256


def test_refuses_changed_protocol(tmp_path):
    shutil.copy(SOURCE / 'runs.json', tmp_path / 'runs.json')
    protocol = json.loads((SOURCE / 'preregister.json').read_text())
    protocol['openings']['development']['start'] = 512
    path = tmp_path / 'preregister.json'
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Frozen protocol changed'):
        summarize(path)


def test_refuses_altered_terminal_evidence(tmp_path):
    shutil.copy(SOURCE / 'runs.json', tmp_path / 'runs.json')
    shutil.copy(SOURCE / 'preregister.json', tmp_path / 'preregister.json')
    edge_dir = tmp_path / 'results/smoke_p8_hero1'
    edge_dir.mkdir(parents=True)
    (edge_dir / 'state.json').write_text('{}')
    with pytest.raises(ValueError, match='state checksum differs'):
        summarize(tmp_path / 'preregister.json')
