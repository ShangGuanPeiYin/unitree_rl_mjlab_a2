"""§5.4 exact flat-anchor loss aggregation (CPU-only)."""
from __future__ import annotations

import numpy as np

BETA_FLAT = 0.1
ACTION_DIM = 12


def clean_flat_mask(flags):
    """Only ordinary flat × nominal × no noise/delay/force are clean."""
    required = ('ordinary_flat', 'nominal_dynamics', 'no_noise', 'no_delay',
                'no_force')
    for key in required:
        if key not in flags:
            raise ValueError(f'clean-flat flags missing {key}')
    return np.logical_and.reduce([np.asarray(flags[key], dtype=bool)
                                  for key in required])


def sample_numerator(mu_student, mu_teacher, clean):
    """Per-sample ``(sum_j diff^2 / 12)`` masked by clean; no mean here."""
    student = np.asarray(mu_student, dtype=np.float64)
    teacher = np.asarray(mu_teacher, dtype=np.float64)
    mask = np.asarray(clean, dtype=bool)
    if student.shape != teacher.shape or student.ndim != 2 \
            or student.shape[1] != ACTION_DIM or mask.shape != (student.shape[0],):
        raise ValueError('expected (N,12) student/teacher and (N,) clean mask')
    if not (np.all(np.isfinite(student)) and np.all(np.isfinite(teacher))):
        raise ValueError('mu arrays must be finite')
    per_sample = np.sum((student - teacher) ** 2, axis=1) / ACTION_DIM
    return float(np.sum(per_sample[mask])), int(mask.sum())


class AnchorAccumulator:
    """Accumulate numerator per micro-batch; divide once by logical n_clean."""

    def __init__(self):
        self.numerator = 0.0
        self.n_clean = 0

    def add(self, mu_student, mu_teacher, clean):
        numerator, n_clean = sample_numerator(mu_student, mu_teacher, clean)
        self.numerator += numerator
        self.n_clean += n_clean
        return self.value()

    def value(self):
        return 0.0 if self.n_clean == 0 else self.numerator / self.n_clean

    def total(self, ppo_loss):
        return float(ppo_loss) + BETA_FLAT * self.value()


def flat_anchor_loss(mu_student, mu_teacher, clean):
    accumulator = AnchorAccumulator()
    return accumulator.add(mu_student, mu_teacher, clean)


def total_loss(ppo_loss, mu_student, mu_teacher, clean):
    return float(ppo_loss) + BETA_FLAT * flat_anchor_loss(
        mu_student, mu_teacher, clean)
