"""Episode-weighted actual horizontal speed and first-termination recording."""
import math
import numpy as np


def speed_gate(episode_velocities, command):
    if not episode_velocities or any(len(x) == 0 for x in episode_velocities):
        return {'passed': False, 'reason': 'missing pre-termination speed interval'}
    bias = np.array([np.mean(x)-command for x in episode_velocities])
    mae = np.array([np.mean(np.abs(np.asarray(x)-command)) for x in episode_velocities])
    return {'passed': bool(abs(bias.mean()) <= .05 and mae.mean() <= .10 and np.mean(abs(bias) <= .10) >= .90),
            'bias': float(bias.mean()), 'mae': float(mae.mean()), 'episode_fraction': float(np.mean(abs(bias) <= .10))}


def wilson(successes, count):
    if count <= 0:
        raise ValueError('No episodes')
    z = 1.959963984540054
    p = successes/count
    d = 1+z*z/count
    c = (p+z*z/(2*count))/d
    h = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return c-h, c+h


class FirstTermination:
    def __init__(self):
        self.record = None

    def capture(self, failure, success, timeout, state):
        if self.record is None and (failure or success or timeout):
            import copy
            self.record = {'failure': bool(failure), 'success': bool(success),
                           'timeout': bool(timeout), 'state': copy.deepcopy(state)}
        return self.record


class EpisodeSpeed:
    """Update before reset; terrain+x projection. One instance per episode."""
    def __init__(self, command, flat=False):
        self.command = command
        self.flat = flat
        self.entered = False
        self.entered_top = False
        self.finished = False
        self.velocities = []
        self.start_xy = None
        self.last_xy = None
        self.termination = FirstTermination()

    def update(self, elapsed, terrain_x, stair_end, world_velocity, terrain_yaw,
               world_xy, failure=False, success=False, timeout=False, state=None):
        if self.finished:
            return
        if self.start_xy is None:
            self.start_xy = np.asarray(world_xy).copy()
        self.last_xy = np.asarray(world_xy).copy()
        self.entered |= terrain_x >= 0
        self.entered_top |= terrain_x >= stair_end
        active = (5 <= elapsed <= 30) if self.flat else (self.entered and not self.entered_top)
        if active:
            self.velocities.append(float(world_velocity[0]*math.cos(terrain_yaw)+world_velocity[1]*math.sin(terrain_yaw)))
        self.termination.capture(failure,success,timeout,state)
        self.finished = bool(failure or success or timeout)

    def result(self):
        r = self.termination.record
        return {'velocities': list(self.velocities), 'termination': r,
                'entered': self.entered, 'displacement': None if self.last_xy is None else float(np.linalg.norm(self.last_xy-self.start_xy))}


def stage_gate(records_by_speed, flat=False, success_threshold=.90):
    """Every required episode remains in both outcome and speed denominators."""
    results = {}
    for command, records in records_by_speed.items():
        complete = all(r['termination'] is not None for r in records)
        outcomes = [bool(r['termination'] and (r['termination']['timeout'] if flat else r['termination']['success'])) for r in records]
        fraction = float(np.mean(outcomes)) if outcomes else 0.
        if flat and command == 0:
            speed_ok = bool(records) and all(r['velocities'] and abs(np.mean(r['velocities'])) <= .03 and r['displacement'] <= .20 for r in records)
            speed = {'passed': bool(speed_ok)}
        else:
            speed = speed_gate([r['velocities'] for r in records], command)
        results[command] = {'passed': complete and fraction >= success_threshold and speed['passed'], 'fraction': fraction, 'speed': speed}
    return bool(results) and all(x['passed'] for x in results.values()), results
