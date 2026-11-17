"""v07 engineering-only observer of the exact frozen production load path.

No seed/reset/model/PPO calls are inserted into production. The source-reload
stage captures the two real 512-environment loads, releasing the first before
building the second; all other stages use the production validation driver.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse
import gc
import hashlib
import json
import os
import sys
import weakref

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
import validate_a2_ideal_stairs_v07 as validation


def verify_observer_inputs():
    frozen = json.loads(validation.FREEZE_PATH.read_text())
    expected = frozen.get('observer_hashes')
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError('v07 observer inputs are not frozen')
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    if relative not in expected:
        raise RuntimeError('this engineering observer is outside the frozen inputs')
    for path, sha in expected.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise RuntimeError('frozen observer input changed: ' + path)


def install_source_observer():
    import torch
    import train_a2_ideal_stairs_v07 as worker
    from src.tasks.velocity.v07 import runtime, orchestration, recovery, model, ppo
    from src.tasks.velocity.v07 import continuation_v07 as migration
    from src.tasks.velocity.v07.curriculum import CurriculumManager, allocate_roles
    from src.tasks.velocity.v07.runner import V07Runner

    original_build = worker.build_environment

    def observed_environment(contract, **kwargs):
        env = original_build(contract, **kwargs)
        events = []
        original_reset = env._reset_ids

        def reset_ids(ids, **reset_kwargs):
            indices = torch.as_tensor(ids).detach().cpu().to(torch.long).flatten().tolist()
            context = dict(phase=env.phase, substage=env.substage,
                           generation=env.generation, stage_iteration=env.stage_iteration)
            result = original_reset(ids, **reset_kwargs)
            events.append(dict(ids=indices, count=len(indices), context=context,
                               resample_tiles=reset_kwargs.get('resample_tiles', True),
                               learning_control_steps=0, completed_episodes=0))
            return result
        env._reset_ids = reset_ids
        env.v07_observed_reset_events = events
        return env

    worker.build_environment = observed_environment

    def environment_evidence(runner, original_source):
        env = runner.env
        roles = list(env.roles)
        tiles = env.tile_ids.detach().cpu().tolist()
        expected_tiles = dict(flat={0}, frontier={17}, near={15, 16}, easy=set(range(1, 15)))
        assert env._tile_override is None
        assert env.phase == runner.phase == 'D' and env.substage == runner.substage == 'D25'
        assert env.stage_iteration == runner.phase_update == 2117
        assert env.generation == runner.curriculum.state.generation == 73
        assert Counter(roles) == Counter(allocate_roles(512, 'D')) == dict(flat=102, frontier=164, near=123, easy=123)
        for role, tile in zip(roles, tiles):
            assert tile in expected_tiles[role], (role, tile)
        command = env.command.detach().cpu()
        assert .3 <= float(command[:, 0].min()) <= float(command[:, 0].max()) <= .6
        assert bool((command[:, 1:] == 0).all())
        fresh = env._observations(update=False)
        actual = runner.checkpoint_state()
        checks = dict(
            obs_is_final_episode=all(torch.equal(runner.ppo.obs[k], v) for k, v in fresh.items()),
            all_episode_steps_zero=bool((env.episode_length_buf == 0).all()),
            no_terminal=env.last_termination is None,
            no_first_step=not bool(env.first_step_acquired.any()),
            no_stop=not bool(env.stop.any()), no_stop_counter=bool((env.stop_steps == 0).all()),
            no_contact_history=not bool(env.contacts.raw.any() or env.contacts.filtered.any()),
            no_climb_history=not bool(env.climb.climbed.any()) and bool((env.climb.support_count == 0).all()),
            no_tracking_count=bool((env._tracking_count == 0).all()),
            no_reward_sums=all(bool((value == 0).all()) for value in env.episode_reward_sums.values()),
            no_stall_samples=bool((env.stall.active_samples == 0).all() and (env.stall.stall_samples == 0).all()),
            no_forward_progress=bool((env.forward_distance_buf == 0).all()),
            start_matches_final_pose=bool(torch.equal(env.episode_start_local_x,
                env.robot.data.root_link_pos_w[:, 0] - env.origins[env.tile_ids, 0])),
            map_matches_final_tile=bool(torch.equal(env.map_cache.sample_tile, env.tile_ids)),
            history_matches_final_proprio=bool(torch.equal(env.history_cache.history,
                env._proprio()[:, None].expand(-1, 5, -1))),
            reward_state_fresh=env.reward_computer.state is None,
            no_learning=runner.update == 4817 and runner.curriculum.state.generation_updates == 0,
            no_frontier_fifo=runner.curriculum.state.episode_fifo == [],
            native_schema=actual['schema'] == 9,
            policy_generation=actual['regression_policy_state']['generation'] == 73,
            no_low_success_queue=actual['regression_policy_state']['low_success_evaluations'] == [],
            no_policy_receipt=actual['regression_policy_state']['last_applied'] is None,
            same_remaining_retention_clock=actual['curriculum']['retention_next_due'] == dict(flat=5000, h00=5000),
            source_parent=actual['source']['continuation']['parent']['checkpoint_sha256'] == migration.SOURCE_SHA256)
        for key in ('model', 'optimizer', 'normalizer', 'learning'):
            checks['source_' + key + '_preserved'] = migration.tree_digest(actual[key]) == migration.tree_digest(original_source[key])
        assert all(checks.values()), checks
        events = deepcopy(env.v07_observed_reset_events)
        assert events and events[-1]['ids'] == list(range(512))
        assert len(events) in (1, 2), events
        return dict(phase=env.phase, substage=env.substage, stage_iteration=env.stage_iteration,
            generation=env.generation, role_counts=dict(Counter(roles)),
            role_to_tile=[dict(env_id=n, role=role, tile_id=tile) for n, (role, tile) in enumerate(zip(roles, tiles))],
            allowed_tiles={key: sorted(values) for key, values in expected_tiles.items()},
            command=dict(min=float(command[:, 0].min()), max=float(command[:, 0].max()), vy_zero=True, wz_zero=True),
            reset_events=events, reset_calls=len(events), reset_slot_initializations=sum(e['count'] for e in events),
            distinct_initialized_slots=len({n for e in events for n in e['ids']}),
            extra_initialization_slots=sum(e['count'] for e in events) - 512,
            rng_audit=deepcopy(runner.v07_initialization_audit), checks=checks,
            pending_reward_reset_records=len(env.reward_computer._pending_resets),
            accepted_extra_rng_consumption=True, full_evaluation_rng_isolation='not_established_Q008')

    def measured_resources(runner, output):
        measured = runner.resource_guard() if runner.resource_guard is not None else orchestration.check_resources(
            orchestration.resource_snapshot(directory=output, device=0), stage='measured')
        validation.write_json(Path(output) / 'resources/source_reload_measured.json', measured)
        if not measured['ok']:
            raise RuntimeError('v07 source-reload measured resources failed')
        return measured

    def second_runner(contract, output, ready):
        state = recovery.preload_checkpoint(ready['path'], ready['sha256'], mode='resume')
        actor = model.V07ActorCritic().to(contract['device'])
        current = state['curriculum']
        env = worker.build_environment(contract, phase=current['phase'], substage=current['substage'], level_tick=current['level_tick'])
        trainer = ppo.PPOTrainer(actor, env)
        manager = CurriculumManager(num_envs=512, seed=42, state=current)
        holder = {}
        runner = V07Runner(model=actor, ppo=trainer, curriculum=manager, env=env, directory=output,
            seed=42, device=contract['device'], mode='resume', source=state['source'], require_resume_evidence=True,
            resource_guard=worker.build_measured_resource_guard(output, contract['device'], holder))
        holder['runner'] = runner
        runner.load(ready['path'], ready['sha256'])
        loaded = runner.save_verified('resume_loaded')
        new_ready = runner.save_verified('resume_ready')
        return runner, dict(resume_loaded=loaded, resume_ready=new_ready,
            source_checkpoint_sha256=ready['sha256'], generation=runner.curriculum.state.generation,
            original_parent_sha256=runner.source['continuation']['parent']['checkpoint_sha256'])

    def observed_source_stage(output, device):
        output = Path(output)
        contract = dict(validation.spec.recipe_config(), device=device)
        source = migration.load_designated_source(ROOT / migration.SOURCE_PATH)
        first, first_receipt = runtime.prepare_runner(contract, output)
        first_environment = environment_evidence(first, source)
        first_resources = measured_resources(first, output)
        ready = dict(first.resume_ready)
        validation.write_json(output / 'first_load_environment.json', first_environment)
        env_reference, model_reference = weakref.ref(first.env), weakref.ref(first.model)
        first.env.close()
        del first
        gc.collect()
        torch.cuda.empty_cache()
        released = dict(first_env_released=env_reference() is None, first_model_released=model_reference() is None,
                        resource_snapshot=orchestration.resource_snapshot(directory=output, device=0))
        validation.write_json(output / 'first_release.json', released)
        assert released['first_env_released'] and released['first_model_released']
        second, second_receipt = second_runner(contract, output / 'second_load', ready)
        second_environment = environment_evidence(second, source)
        second_resources = measured_resources(second, output / 'second_load')
        validation.write_json(output / 'second_load_environment.json', second_environment)
        result = dict(first=first_receipt, first_environment=first_environment, first_resources=first_resources,
            release=released, second=second_receipt, second_environment=second_environment,
            second_resources=second_resources, learning_updates=0, engineering_only=True,
            counts_against_training=False, resumable_for_production=False,
            full_evaluation_rng_isolation='not_established_Q008')
        validation.write_json(output / 'source_reload.json', result)
        second.env.close()
        return result

    original_stage = validation.run_stage

    def run_stage(stage, output, device):
        if stage == 'source-reload':
            return observed_source_stage(output, device)
        return original_stage(stage, output, device)
    validation.run_stage = run_stage


def main(arguments=None):
    args = list(sys.argv[1:] if arguments is None else arguments)
    observer_parser = argparse.ArgumentParser(add_help=False)
    observer_parser.add_argument('--stage', choices=validation.STAGES)
    observer_parser.add_argument('--execute', action='store_true')
    selected, _unknown = observer_parser.parse_known_args(args)
    execute = selected.execute
    if execute:
        verify_observer_inputs()
        if selected.stage == 'source-reload':
            install_source_observer()
    result = validation.main(args)
    if execute:
        verify_observer_inputs()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
