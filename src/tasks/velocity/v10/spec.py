"""Immutable training recipe transcribed from the reviewed SDS v1.1."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[4]
SDS_PATH = ROOT / 'doc/sds/SDS_A2_v1.0_DualTeacher_DAgger_PPO_v1.1.md'
SDS_SHA256 = 'eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0'
RECIPE = 'v10-dual-teacher-dagger-ppo-v1.1'
BLIND_PATH = ROOT / 'weights/a2_v10/model_2000.pt'
BLIND_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'
STAIRS_PATH = ROOT / 'logs/good/s2_1_latest.pt'
STAIRS_SHA256 = 'b3f31d1e7197654984a513ec6d9c38030a92852a09061aa748c9615670684831'
ROLES = ('F_WALK','F_STAND','F_STOP','P_WALK','P_STOP','S_ROUTE')
ROLE_WEIGHTS = (.20,.10,.15,.15,.10,.30)
SPEEDS = (.3,.5,.6)
DT = .02
PHYSICS_DT = .005
ROLLOUT = 48
SEED = 42
MAP_SHAPE = (3,40,30)
BUDGETS = {'D1':24576000,'D2':49152000,'D3':73728000,'PPO':49152000}
P0_BUDGET = 4915200
SAVE_SAMPLES = 2457600
RESERVOIR_PER_TASK = 32768
LR = 3e-5
MICROBATCH = 256


def role_counts(n):
    if n not in (512,1024):
        raise ValueError('SDS permits only 1024 or 512 formal environments')
    raw = [n*w for w in ROLE_WEIGHTS]
    out = [math.floor(x) for x in raw]
    for i in sorted(range(6),key=lambda i: (-(raw[i]-out[i]),i))[:n-sum(out)]:
        out[i] += 1
    return tuple(out)


def gate_rounds(course,n):
    rounds=BUDGETS[course]//(n*ROLLOUT)
    fractions=(.25,.5,.75,1.) if course=='PPO' else (.4,.8,1.)
    return tuple(round(rounds*f) for f in fractions)
