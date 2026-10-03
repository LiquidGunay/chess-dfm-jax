from decimal import Decimal

from research.infra.modal_eval_matrix import launch_command, reserved_spend


def test_matrix_reserves_running_attempts_and_meter_lag() -> None:
    runs = {'a': {'status': 'complete'}, 'b': {'status': 'submitted'}, 'c': {'status': 'launching'}}
    assert reserved_spend(Decimal('17.72'), Decimal('17.72'), runs, Decimal('1')) == Decimal('20.72')
    assert reserved_spend(Decimal('21'), Decimal('17.72'), runs, Decimal('1')) == Decimal('23')
    runs['b']['status'] = 'failed'
    assert reserved_spend(Decimal('17.72'), Decimal('17.72'), runs, Decimal('1')) == Decimal('20.72')


def test_matrix_command_binds_exact_phase_and_edge() -> None:
    protocol = {'candidate': {'label': 'hero-training-test'}, 'raw_bt4': {'label': 'raw-bt4', 'plane_history_mode': 'lc0_full_history_as_preprocessed'},
                'openings': {'development': {'start': 384, 'pairs': 128}}}
    edge = {'candidate': 'candidate', 'opponent': 'raw_bt4', 'candidate_mode': 'dfm',
            'opponent_mode': 'policy_only', 'candidate_passes': 8, 'opponent_passes': 1}
    command = launch_command(protocol, edge, 'development', Decimal('20.1'))
    assert command[command.index('--opening-start-index') + 1] == '384'
    assert command[command.index('--pair-count') + 1] == '128'
    assert command[command.index('--candidate-refinement-passes') + 1] == '8'
    assert command[command.index('--opponent-label') + 1] == 'raw-bt4'
    assert command[-1] == '20.1'
    assert command[command.index('--raw-bt4-plane-history-mode') + 1] == 'lc0_full_history_as_preprocessed'
