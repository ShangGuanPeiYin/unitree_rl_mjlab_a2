"""Code-level rejection of the invalidated s2.1-r1 checkpoints/derivatives."""
from __future__ import annotations

import hashlib
from pathlib import Path

R1_INVALID_SHA256 = {
    '59cf153d510d605c8f2c718baeb9cb1864dd7d4da89444bfa846c5261d715665':
        's2_1_000005013504.pt',
    '429384de': 's2_1_000010027008.pt',
    '247cc0c4': 's2_1_000015040512.pt',
    '16a06ea3': 's2_1_000020054016.pt',
}
R1_INVALID_PREFIXES = ('59cf153d', '429384de', '247cc0c4', '16a06ea3')


def reject_invalid_checkpoint(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in R1_INVALID_SHA256 or digest.startswith(R1_INVALID_PREFIXES):
        raise ValueError(f'refusing invalidated s2.1-r1 checkpoint {path} ({digest})')
    return digest


def reject_invalid_attempt(attempt_id):
    if str(attempt_id).startswith('s2.1-r1'):
        raise ValueError('s2.1-r1 is invalidated; use a new attempt (s2.1-r2)')
    return True
