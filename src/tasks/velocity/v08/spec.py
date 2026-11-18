"""v0.8 numerical contract. Inherited physical/model constants stay unchanged."""
from copy import deepcopy
from ..v07.spec import *
from ..v07 import spec as legacy
CODE_VERSION='0.8.0'
RECIPE_VERSION='0.8-stairs-walk-stop-reverse-r1'
PROTOCOL_ID=RECIPE_VERSION
TASK=TASK_ID='Unitree-A2-IdealHeightMap-Stairs-V08'
LOG_ROOT='logs/a2_ideal_stairs_v08'
SCHEMA_VERSION=CHECKPOINT_SCHEMA=10
INIT_PATH='logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt'
INIT_SHA256='30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
SOURCE_UPDATE=6400
MAX_GLOBAL_UPDATE=24400
PHASE_ORDER=('A','B','C','D','E')
PHASE_BUDGETS=dict(A=4000,B=2000,C=2000,D=4000,E=6000)
TOTAL_BUDGET=18000
PHASE_ALLOCATIONS={
 'A':{'up_L40':320,'flat':128,'up_H00':64},
 'B':{'up_L40':320,'flat':128,'up_H00':64},
 'C':{'down_D20':256,'up_L40':128,'flat':96,'down_H00':32},
 'D':{'down_L40':256,'up_L40':128,'flat':96,'down_D20':32},
 'E':{'E1':64,'E2':64,'E3':64,'E4':64,'up_L40':96,'down_L40':64,'flat':64,'up_H00':32}}
EVAL_SPEEDS=(.3,.5,.6)
GATE_STREAK_REQUIRED=2
R_RAMP_UPDATES=400
TRANSITION_STEPS=100
ENDPOINT_HOLD_STEPS=1000
ENDPOINT_WINDOW_STEPS=500
MID_HOLD_STEPS=300
MID_WINDOW_STEPS=150
FLAT_WALK_STEPS=400
FLAT_WALK_SECONDS=8.
FLAT_PROGRESS_FACTOR=.8
SPEED_MIN=.3
SPEED_MAX=.6
TOP_SPAWN_OFFSET=1.
BOTTOM_BODY_X=-.8
SUPPORT_HISTORY_STEPS=50
BOTTOM_ARRIVAL_FRAMES=5
STAND_MEAN_SPEED=.05
STAND_MAX_SPEED=.10
STAND_DRIFT=.10
STAND_OMEGA=.25
STAND_FOOT_SPEED=.05
STAND_FOUR_FRACTION=.9
LEVEL_ROLL=LEVEL_PITCH=10.*3.141592653589793/180.
MID_ROLL=12.*3.141592653589793/180.
MID_PITCH=50.*3.141592653589793/180.
MID_YAW=15.*3.141592653589793/180.
REVERSE_YAW=30.*3.141592653589793/180.
LEVEL_HEIGHT=.25
MID_HEIGHT=.22
REVERSE_PROGRESS=.60
REVERSE_PROGRESS_STEPS=400
ARRIVAL_REWARD=2.
SUCCESS_REWARD=8.
FAILURE_REWARD=-10.
PARK_ANG_WEIGHT=-.20
PARK_PITCH_WEIGHT=-5.
STAND_WEIGHT=1.
TRIGGER_RANGES={'E1':(.45,.75),'E2':(.25,.55),'E3':(.30,.70)}
SAMPLER_SEED_OFFSET=8801
SCENE_EPISODE_SECONDS=1000.
COMMAND_TOLERANCE=1e-7

def phase_budget(phase): return PHASE_BUDGETS[str(phase).upper()]
def total_budget(): return TOTAL_BUDGET
def reward_config():
    out=deepcopy(legacy.reward_config())
    out.update(version=RECIPE_VERSION,
        composition='dt*sum_i(weight_i*rate_i) + 0.4*sign(c_exec)*dx*I_cruise + 2*I_final_arrival + 8*I_success - 10*I_failure',
        forward_progress=dict(per_m=FORWARD_PROGRESS_PER_M,formula='0.4*sign(c_exec)*dx during constant cruise only',transition=0,parking=0),
        arrival=ARRIVAL_REWARD,success=SUCCESS_REWARD,failure=FAILURE_REWARD,timeout_reward=FAILURE_REWARD,
        parking_angular=PARK_ANG_WEIGHT,level_parking_pitch=PARK_PITCH_WEIGHT,standing=STAND_WEIGHT,
        orientation=dict(roll='g_y^2 always',pitch='g_x^2 only level parking deceleration/hold'),
        air_time=dict(target=AIR_TIME_TARGET,cruise_weight=REWARD_WEIGHTS['feet_air_time'],parking_weight=0,clear_timer_on_restart=True,settle_before_clear=True),
        diagnostics=dict(scope='signed_constant_cruise_segments_only',reward_payment=False,termination=False,h00='report_only',history_samples=STALL_HISTORY_SAMPLES,dx_fraction=STALL_DX_FRACTION_PER_SECOND,consecutive_samples=STALL_CONSECUTIVE_SAMPLES))
    return out

def contract(**kwargs):
    return dict(code_version=CODE_VERSION,recipe_version=RECIPE_VERSION,protocol_id=PROTOCOL_ID,source_update=SOURCE_UPDATE,init_path=INIT_PATH,init_sha256=INIT_SHA256,phase_order=PHASE_ORDER,phase_budgets=deepcopy(PHASE_BUDGETS),phase_allocations=deepcopy(PHASE_ALLOCATIONS),transition_steps=TRANSITION_STEPS,endpoint_hold_steps=ENDPOINT_HOLD_STEPS,mid_hold_steps=MID_HOLD_STEPS,reward=reward_config(),ppo=deepcopy(PPO_CONFIG))

def contract_sha256(value=None): return legacy.contract_sha256(contract() if value is None else value)
def reward_config_sha256(): return legacy.contract_sha256(reward_config())

# Native v08 configuration API: functions inherited by ``import *`` retain
# their defining module's globals, so every identity/config entry point must
# resolve here rather than silently returning v07's V/R curriculum.
SDS_PATH='doc/sds/SDS_A2_v0.8_Stairs_Walk_Stop_Reverse.md'
RULINGS_PATH='doc/sds/SDS_A2_v0.8_W0_RULINGS.md'
SDS_SHA256='8aca6444581f0fa4c3f62e82061c2a47b00d98c72df48add93b4168138189afe'
NONFOOT_LIMIT_S=NONFOOT_PERSIST_TIME
BODY_HALF_WIDTH=STAIR_WIDTH/2.
TERRAIN_RNG_OFFSET=7919
BOTTOM_MIN_SUPPORT_FEET=2
FLAT_LATERAL_MAX=.20
BOTTOM_LATERAL_MAX=.20
GATE_CASES=20
GATE_PASS_MIN=18
FINAL_CASES=100
FINAL_PASS_MIN=95
E3_FINAL_SCRIPT_CASES=25
E3_FINAL_SCRIPT_PASS_MIN=23
H00_INTERVAL=200
H00_CASES=5
EVAL_MAX_ENVS=8
TOTAL_CONTROL_SAMPLES=TOTAL_BUDGET*CONTROL_SAMPLES_PER_UPDATE
TOTAL_PHYSICS_SUBSTEPS=TOTAL_BUDGET*PHYSICS_SUBSTEPS_PER_UPDATE
TOTAL_ADAM_UPDATES=TOTAL_BUDGET*ADAM_UPDATES_PER_ITERATION
CURRICULUM=dict(phase_order=PHASE_ORDER,phase_budgets=PHASE_BUDGETS,allocations=PHASE_ALLOCATIONS,gate_streak=GATE_STREAK_REQUIRED,birth_ramp_updates=R_RAMP_UPDATES)
PHASES=tuple(dict(name=p,budget=PHASE_BUDGETS[p],allocation=deepcopy(PHASE_ALLOCATIONS[p])) for p in PHASE_ORDER)
COMPLETE_MEANING='final_2100_case_test_passed_only'
SMOKE_CONTAINER='v08-smoke-state-v1'
PROBE_CONTAINER='v08-probe-state-v1'
CLI_DEFAULTS=dict(mode='formal',seed=42,device='cuda:0',num_envs=512,eval_num_envs=8,rollout=48,epochs=5,minibatches=4,gamma=.99,gae_lambda=.95,clip=.2,entropy=.003,max_grad_norm=1.,source_update=SOURCE_UPDATE,source_checkpoint=INIT_PATH,source_sha256=INIT_SHA256)

def identity_block():
    return dict(code_version=CODE_VERSION,recipe_version=RECIPE_VERSION,schema_version=SCHEMA_VERSION,architecture=ARCHITECTURE,task=TASK,protocol=PROTOCOL_ID,log_root=LOG_ROOT,training_seed=TRAINING_SEED)

def document_identity():return dict(sds_path=SDS_PATH,sds_sha256=SDS_SHA256,rulings_path=RULINGS_PATH)

def recipe_config(**cli):
    result=deepcopy(CLI_DEFAULTS)
    for key,value in cli.items():
        if key not in result:raise ValueError(f'unknown v08 recipe option {key}')
        if key not in ('mode','device','source_checkpoint') and value!=result[key]:raise ValueError(f'frozen v08 recipe option {key}: {value!r} != {result[key]!r}')
        result[key]=value
    result.update(identity=identity_block(),curriculum=deepcopy(CURRICULUM),source_actual_learning_rate='read_from_source_adam',observation=dict(proprio=[PROPRIO_DIM],history=list(HISTORY_SHAPE),heightmap=list(HEIGHTMAP_SHAPE),privileged=[PRIVILEGED_DIM],route_error=[ROUTE_DIM]))
    return result

def source_hashes(root=ROOT,paths=None):
    from pathlib import Path
    base=Path(root)
    files=([base/p for p in paths] if paths is not None else list((base/'src/tasks/velocity/v08').glob('*.py'))+list((base/'scripts').glob('*v08*.py'))+[base/SDS_PATH,base/RULINGS_PATH])
    return {str(p.relative_to(base)):sha256(p) for p in sorted(files) if p.is_file()}

def behavior_hashes(root=ROOT):return source_hashes(root)
def product_entries(root=ROOT):return tuple(source_hashes(root))
def pending_source_entries(root=ROOT):return ()

def contract(**cli):
    from dataclasses import asdict
    return dict(identity=identity_block(),document_identity=document_identity(),recipe=recipe_config(**cli),source_update=SOURCE_UPDATE,init_path=INIT_PATH,init_sha256=INIT_SHA256,phase_order=list(PHASE_ORDER),phase_budgets=deepcopy(PHASE_BUDGETS),phase_allocations=deepcopy(PHASE_ALLOCATIONS),total_budget=TOTAL_BUDGET,max_global_update=MAX_GLOBAL_UPDATE,ppo=deepcopy(PPO_CONFIG),terrain=[asdict(v) for v in TERRAIN_TABLE],timing=dict(control_dt=CONTROL_DT,physics_dt=PHYSICS_DT,decimation=DECIMATION,map_hz=MAP_HZ),command=dict(min=SPEED_MIN,max=SPEED_MAX,transition_steps=TRANSITION_STEPS,flat_walk_steps=FLAT_WALK_STEPS,endpoint_hold_steps=ENDPOINT_HOLD_STEPS,mid_hold_steps=MID_HOLD_STEPS),standing=dict(endpoint_window=ENDPOINT_WINDOW_STEPS,mid_window=MID_WINDOW_STEPS,mean_speed=STAND_MEAN_SPEED,max_speed=STAND_MAX_SPEED,drift=STAND_DRIFT,omega=STAND_OMEGA,foot_speed=STAND_FOOT_SPEED,four_support_fraction=STAND_FOUR_FRACTION,level_roll=LEVEL_ROLL,level_pitch=LEVEL_PITCH,level_height=LEVEL_HEIGHT,mid_roll=MID_ROLL,mid_pitch=MID_PITCH,mid_yaw=MID_YAW,mid_height=MID_HEIGHT),direction=dict(yaw=REVERSE_YAW,progress=REVERSE_PROGRESS,deadline_steps=REVERSE_PROGRESS_STEPS,trigger_ranges=deepcopy(TRIGGER_RANGES)),gates=dict(interval=100,diagnostics_interval=50,h00_interval=H00_INTERVAL,cases=GATE_CASES,pass_min=GATE_PASS_MIN,speeds=EVAL_SPEEDS,streak=GATE_STREAK_REQUIRED,final_cases=FINAL_CASES,final_pass_min=FINAL_PASS_MIN,e3_script_cases=E3_FINAL_SCRIPT_CASES,e3_script_pass_min=E3_FINAL_SCRIPT_PASS_MIN,eval_max_envs=EVAL_MAX_ENVS),reward=reward_config())

# Frozen inherited map/contact numbers, centralized without changing values.
MAP_GRID_STEP=.05
MAP_GRID_X=(-.8,.8)
MAP_GRID_Y=(-.8,.8)
MAP_HEIGHT_SCALE=1.5
MAP_AGE_SECONDS=.5
CONTACT_TOP_TOLERANCE=.005
SUPPORT_NORMAL_MAX_ANGLE=3.141592653589793/6.
PLATFORM_CONTACT_MIN_LENGTH=1.99
FOOT_HEIGHT_INVALID=1.5
FOOT_HEIGHT_MIN=-.05
