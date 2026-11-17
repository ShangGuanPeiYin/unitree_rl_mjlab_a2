"""Explicit, process-local C1 contract selected before any v06 consumer imports.

The original v06 remains the default. C1 only replaces approved identity,
command/gate speeds and D/global budgets; reward and PPO structures stay byte
for byte equal in value to their original definitions.
"""
from copy import deepcopy
from dataclasses import replace

CONTINUATION_ID = 'v06-c1-speed030060-budget36000'
PLAN_PATH = 'doc/sds/SDS_A2_Stairs_v0.6_C1_Continue_030_060.md'
PLAN_SHA256 = '22d3c77a6af62042eb98f8b310ea3d1d7e8554b77a811ad618be2a4e0a35c34c'
LOG_ROOT = 'logs/a2_ideal_stairs_v06_c1'
IMPLEMENTATION = LOG_ROOT + '/implementation'
OUTPUT_DIR = LOG_ROOT + '/continue_seed42_from004224_trial01'
ACCEPTANCE_FILE = IMPLEMENTATION + '/acceptance/INDEPENDENT_ACCEPTANCE.md'
MANIFEST_PATH = IMPLEMENTATION + '/C1_BRIDGE_MANIFEST.json'
SOURCE_PATH = ('logs/a2_ideal_stairs_v06/'
               'train_seed42_trial02_resume_000400_q007/checkpoints/stop_004224.pt')
SOURCE_SHA256 = '108e03dc64effa94d2824bbf36842aee8f9c5fd9c2122be391b337706f4a51b9'
SOURCE_CLOSURE = 'ebec92ba8f9c48a1ae15b1a72a067500a183098748eeff2631ee4c484ab4be60'


def apply(namespace):
    """Apply once while spec imports; no downstream cached values exist yet."""
    if namespace['RECIPE_VERSION'] != '0.6-go2refs-r1':
        raise RuntimeError('C1 contract must be selected before v06 imports')
    reward_before = deepcopy(namespace['REWARD_CONFIG'])
    ppo_before = deepcopy(namespace['PPO_CONFIG'])
    namespace.update(CONTINUATION_ID=CONTINUATION_ID, CODE_VERSION='0.6.1',
                     RECIPE_VERSION='0.6-go2refs-c1',
                     PROTOCOL_ID=CONTINUATION_ID, LOG_ROOT=LOG_ROOT,
                     COMMAND_INITIAL_VX=(.3, .6), COMMAND_FINAL_VX=(.3, .6),
                     GATE_SPEEDS_F0_H_D_L=(.3, .5, .6), GATE_SPEEDS_V_R=(.3, .5, .6))
    phases = tuple(replace(p, budget=9000 if p.phase == 'D' else p.budget,
                           substage_budgets=(4500, 4500) if p.phase == 'D' else p.substage_budgets,
                           gate_speeds=(.3, .5, .6), reward_speeds=(.3, .6))
                   for p in namespace['PHASES'])
    namespace['PHASES'] = phases
    namespace['PHASE_BY_ID'] = {p.phase: p for p in phases}
    namespace['D_SUBSTAGE_BUDGETS'] = (4500, 4500)
    namespace['PHASE_BUDGETS'] = {p.phase: p.budget for p in phases}
    namespace['TOTAL_BUDGET'] = 36000
    for key, per in [('TOTAL_CONTROL_SAMPLES',24576),('TOTAL_PHYSICS_SUBSTEPS',98304),
                     ('TOTAL_ADAM_UPDATES',20)]:
        namespace[key] = 36000 * per
    curriculum = namespace['CURRICULUM']
    curriculum['phase_budgets'] = dict(namespace['PHASE_BUDGETS'])
    curriculum['substages']['D']['budgets'] = (4500, 4500)
    for phase in ['v', 'r']:
        curriculum[phase]['gate_speeds'] = (.3, .5, .6)
    curriculum['command'].update(initial_vx=(.3,.6),final_vx=(.3,.6),
                                 v_ramp='initial + (final-initial)*p; both bounds fixed (.3,.6)')
    namespace['CLI_DEFAULTS'].update(recipe='0.6-go2refs-c1',initial_vx_min=.3,
        initial_vx_max=.6,final_vx_min=.3,final_vx_max=.6,max_updates=36000,
        phase_budgets='F0=1000,H=12000,D=9000,L=6000,V=4000,R=4000',
        acceptance_file=ACCEPTANCE_FILE,output_dir=OUTPUT_DIR)
    namespace['SUPPLEMENTARY_DOCUMENTS'] += (('c1_plan',PLAN_PATH,PLAN_SHA256),)
    original_documents = namespace['document_identity']
    def c1_document_identity():
        return dict(original_documents(), c1_plan_path=PLAN_PATH,
                    c1_plan_sha256=PLAN_SHA256, continuation_id=CONTINUATION_ID)
    namespace['document_identity'] = c1_document_identity
    namespace['SCRIPT_ENTRIES'] += ('scripts/run_a2_ideal_stairs_v06_c1.py',
                                     'scripts/validate_a2_ideal_stairs_v06_c1.py')
    namespace['SOURCE_ENTRIES'] += namespace['SCRIPT_ENTRIES'][-2:]
    assert namespace['REWARD_CONFIG'] == reward_before
    assert namespace['PPO_CONFIG'] == ppo_before
    assert sum(namespace['PHASE_BUDGETS'].values()) == 36000
