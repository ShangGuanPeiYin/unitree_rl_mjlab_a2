# A2 v1.0 learning stack implementation — 2026-11-26

SDS: `doc/sds/SDS_A2_v1.0_DualTeacher_DAgger_PPO_v1.1.md`, SHA256 `eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`.

## Owned files and behavior

- `src/tasks/velocity/v10/model.py`: exact complete S2 state load with SHA check; DAgger Actor-only trainability; frozen normalizers; PPO actor-preserving independent seed-42 default-initialized Critic (zero output layer), copied frozen statistics, std=0.10, P0/P1 phase control and [0.05,0.30] projection. `initialize_ppo()` returns frozen independent `T_keep`.
- `teachers.py`: frozen exact ETH input normalizer and MLP; independent deep-copied S2 teacher; same-state masked deterministic routing without history mutation.
- `replay.py`: six independent CPU FP32 reservoirs, each bounded at 32768; sequential standard reservoir replacement with saved count/RNG; snapshots include metadata; balanced six-task batches of 3072 with current/history split, history shortfall filled from current.
- `dagger.py`: fixed Adam LR 3e-5, FP32, sample-weighted micro accumulation, 20/10 updates for 1024/512, one clipping and Adam per 3072 logical batch, replay insertion only after the final update.
- `ppo.py`: terminal-masked GAE, full-rollout advantage normalization, five independently shuffled no-replacement epochs/four equal batches; clipped policy/value loss, summed-action entropy, frozen final-student re-labeled retention replay only; P0 critic-only gradients, P1 joint gradients and std projection; learner optimizer/RNG save/load.
- `rewards.py`: all 20 continuous SDS terms, once-only dt, separate displacement/event payments with failure priority; real joint soft-limit penalty and explicit stable input; measured surface validity fallback; reward-contact air timer, initial-fall suppression, stop clearing.
- `tests/test_v10_learning_core.py`: 11 CPU tests with actual local teacher checkpoint identity/parity and numerical tests.

## Interfaces

`V10ActorCritic.from_stairs(path, expected_sha, device)`; callable deterministic mean; `evaluate(obs)` returns `[B,1]`; `action_distribution(obs)` unsquashed Normal; `set_phase(course)`; `initialize_ppo(seed=42)`; `constrain_std()`.

`DualTeacher(stairs, blind_path, blind_sha).label(obs, stairs_mask)` returns `[B,12]`; constructor never freezes the supplied student's own parameters.

`TaskReservoir(capacity=32768,seed=42)` stores dict `{obs:{key:tensor},actions:teacher_labels,task:list[str],teacher:list,case:list,episode:list,control_step:list}`. `sample_logical(current)` and `sample_per_task(n)` return concatenated CPU batches. `add`, `state_dict`, `load_state_dict` preserve standard reservoir sampling.

`DAggerLearner(model).update(current,reservoir,num_envs)`; optimizer is public and must be restored by orchestrator along with model/reservoir.

`PPOLearner(model).update(flat_rollout, keep, reservoir, phase)` uses keys `{obs,actions,log_prob,values,returns,advantages}` with all tensor fields flattened to sample dimension. `actions` are original Gaussian draws, not executed clipped actions. `gae(rewards[T,N],values[T,N],done[T,N],last_value[N])` returns unnormalized advantages and returns. Learner `state_dict` includes optimizer and shuffle RNG.

`compute_reward(frame)` returns reward and `{'raw':terms,'weighted':terms_after_weights_and_dt}`. Required keys agreed directly with environment agent; implementation is sole formula authority. `AirtimeTracker(N,device).update(C_reward,cruise)`, `reset(ids)`.

## Verification evidence

`/home/xxs/anaconda3/envs/a2/bin/python -m pytest tests/test_v10_learning_core.py -q`: **11 passed, 1 upstream Torch JIT deprecation warning**, 10.57 s. Initial core test collection failed with missing v10 modules before implementation. Additional teacher-isolation test failed because the first implementation froze the passed student; fixed by independent deepcopy and rerun green.

Tests cover standard reservoir snapshot isolation and exact RNG restoration, balanced task batches, terminal and truncation GAE, exact epoch coverage, reward-contact timer and clearing, full S2 state identity, exact ETH MLP output parity, Actor preservation/Critic zero-output initialization/global RNG isolation/frozen statistics/std bounds, DAgger microbatch/full logical batch parameter equality, all reward raw terms against hand calculations, event failure priority, P0 gradient isolation, and retention re-labeling rather than stored old labels.

## Limits and unverified work

No GPU environment rollout, complete-stack resource probe, teacher behavior admission, DAgger/PPO formal training, behavior improvement, or independent acceptance was performed by this agent. Environment/runner telemetry integration remains to be exercised by the parent and independently reviewed. Unit tests do not imply behavior success. Whole-project suite is delegated to parent integration because other agents are simultaneously constructing modules. No commit/index changes were made; parent owns log append and Git operations.
