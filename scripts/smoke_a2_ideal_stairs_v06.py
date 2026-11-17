"""v0.6 GPU smoke plan: seed42, 8 envs, H00, 2 updates (SDS 9.3).

Cost contract (asserted, not assumed)::

    control samples = 8 * 48 * 2      = 768
    physics substeps = 4 * 768        = 3072
    Adam updates     = 5 * 4 * 2      = 40

The smoke artifact is deliberately marked ``resumable=False`` and carries the
``v06-smoke-state-v1`` container tag, so it can never be offered as a normal
training resume source.  Without ``--execute`` this script only prints the plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(str(ROOT))


def _rooted(path):
    """Run-local paths resolve against the project root, never the CWD."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)

#: SDS 5.2: ``H00`` is the first terrain tile of phase ``H`` — never a substage
#: (only D25/D20/L12/L20/L40 are substages).  The smoke starts at ``level_tick=0``.
SMOKE = dict(seed=42, num_envs=8, phase='H', tile='H00', level_tick=0, updates=2,
             rollout=48, epochs=5, minibatches=4, control_samples=768,
             physics_substeps=3072, adam_steps=40)
SMOKE_CONTAINER = 'v06-smoke-state-v1'


def smoke_environment_kwargs():
    """The H00 environment identity: phase ``H``, no substage, tile 1.

    H00..H15 are H-phase *terrain tiles* selected by ``level_tick``; passing one as
    ``substage`` is invalid (env rejects it) and would silently change the task.
    """
    from src.tasks.velocity.v06 import spec

    tile_id = int(spec.tile_id_for('H00'))
    terrain = spec.terrain_spec(tile_id)
    return dict(
        phase='H', substage=None, level_tick=0,
        tile='H00', frontier_tile_id=tile_id,
        lesson_id=terrain.lesson_id, N=int(terrain.N), h=float(terrain.h),
        step_depth=float(terrain.step_depth), width=float(terrain.width),
        tile_id=int(tile_id))


def smoke_plan():
    from src.tasks.velocity.v06 import ppo as ppo_module
    from src.tasks.velocity.v06 import spec

    assert ppo_module.ROLLOUT == SMOKE['rollout']
    assert ppo_module.EPOCHS == SMOKE['epochs']
    assert ppo_module.MINIBATCHES == SMOKE['minibatches']
    control = SMOKE['num_envs'] * SMOKE['rollout'] * SMOKE['updates']
    substeps = ppo_module.physics_substeps(control)
    adam = SMOKE['updates'] * SMOKE['epochs'] * SMOKE['minibatches']
    if (control, substeps, adam) != (SMOKE['control_samples'],
                                     SMOKE['physics_substeps'], SMOKE['adam_steps']):
        raise SystemExit('smoke cost arithmetic contradicts SDS 9.3')
    environment = smoke_environment_kwargs()
    assert environment['phase'] == SMOKE['phase'] == 'H'
    assert environment['substage'] is None
    assert environment['frontier_tile_id'] == int(spec.tile_id_for('H00')) == 1
    assert environment['lesson_id'] == 'H00'
    assert (environment['N'], environment['h'], environment['step_depth']) == (6, 0.02, 0.30)
    assert tuple(spec.phase_spec('H').substages) == ()
    return dict(plan=dict(SMOKE), environment=environment,
                container=SMOKE_CONTAINER, resumable=False,
                acceptance_scope='engineering_check_only',
                note='smoke evidence may not initialize or resume a v0.6 training run')


def update_cost_deltas(before, after):
    """Per-update billed increments from two ``PPOTrainer.learning`` snapshots."""
    keys = ('control_samples', 'physics_substeps', 'adam_steps', 'update')
    deltas = {}
    for key in keys:
        old = int((before or {}).get(key, 0))
        new = int((after or {}).get(key, 0))
        if new < old:
            raise SystemExit(f'smoke learning counter {key} went backwards')
        deltas[key] = new - old
    return deltas


def history_entry(update_index, before, after, *, loss=None):
    """One smoke-history row: real per-update increments plus the update index.

    ``update_cost_deltas`` already carries an ``update`` increment, so the index is
    assigned after copying the mapping (``dict(update=..., **deltas)`` would be a
    duplicate-keyword TypeError).
    """
    entry = dict(update_cost_deltas(before, after))
    entry['update'] = int(update_index)
    entry['total_control_samples'] = int(after['control_samples'])
    entry['total_physics_substeps'] = int(after['physics_substeps'])
    entry['total_adam_steps'] = int(after['adam_steps'])
    if loss is not None:
        entry['loss'] = loss
    return entry


def smoke_updates(trainer, updates):
    """Run the smoke updates and build the billed history from the trainer ledger."""
    before = dict(trainer.learning)
    history = []
    for index in range(int(updates)):
        _records, metrics = trainer.train_step()
        after = dict(trainer.learning)
        history.append(history_entry(index + 1, before, after,
                                     loss=metrics.get('loss')))
        before = after
    return history, dict(trainer.learning)


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute', action='store_true',
                   help='run the smoke worker (default: print the plan only)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir', default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    record = smoke_plan()
    print(json.dumps(record, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if not args.output_dir:
        raise SystemExit('--execute requires --output-dir')
    output = _rooted(args.output_dir)
    if output.exists():
        raise SystemExit(f'--output-dir already exists (refused): {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'smoke_plan.json').write_text(
        json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    from train_a2_ideal_stairs_v06 import build_environment

    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import ppo as ppo_module

    from src.tasks.velocity.v06 import spec

    contract = dict(num_envs=SMOKE['num_envs'], device=args.device,
                    seed=SMOKE['seed'], eval_num_envs=1)
    model = model_module.V06ActorCritic.migrate_eth_checkpoint(seed=SMOKE['seed'],
                                                               device=args.device)
    environment = smoke_environment_kwargs()
    env = build_environment(contract, phase=environment['phase'],
                            substage=environment['substage'])
    # Assert the *actual* environment schedule, read from spec/env (never hardcoded).
    if str(env.phase).upper() != 'H':
        raise SystemExit(f'smoke env phase is {env.phase!r}, expected H')
    if int(env.frontier_tile_id) != int(environment['frontier_tile_id']):
        raise SystemExit(
            f'smoke env frontier tile is {env.frontier_tile_id}, expected H00 '
            f'({environment["frontier_tile_id"]})')
    terrain = spec.terrain_spec(int(env.frontier_tile_id))
    if (terrain.lesson_id, int(terrain.N), float(terrain.h),
            float(terrain.step_depth)) != (environment['lesson_id'],
                                           environment['N'], environment['h'],
                                           environment['step_depth']):
        raise SystemExit('smoke env frontier terrain differs from the H00 spec')
    trainer = ppo_module.PPOTrainer(model, env)
    history, learning = smoke_updates(trainer, SMOKE['updates'])
    totals = dict(
        control_samples=sum(entry['control_samples'] for entry in history),
        physics_substeps=sum(entry['physics_substeps'] for entry in history),
        adam_steps=sum(entry['adam_steps'] for entry in history),
        learning_updates=len(history))
    expected = dict(control_samples=SMOKE['control_samples'],
                    physics_substeps=SMOKE['physics_substeps'],
                    adam_steps=SMOKE['adam_steps'],
                    learning_updates=SMOKE['updates'])
    if totals != expected:
        raise SystemExit(f'smoke billed cost {totals} differs from the frozen {expected}')
    if int(learning['control_samples']) != SMOKE['control_samples']:
        raise SystemExit('smoke trainer ledger does not reach the frozen control samples')
    result = dict(record, history=history, cost_totals=totals,
                  learning=learning,
                  membership=trainer.membership(),
                  environment_actual=dict(
                      phase=str(env.phase), substage=env.substage,
                      level_tick=None,
                      frontier_tile_id=int(env.frontier_tile_id),
                      lesson_id=terrain.lesson_id, N=int(terrain.N),
                      h=float(terrain.h), step_depth=float(terrain.step_depth)))
    (output / 'smoke.json').write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
