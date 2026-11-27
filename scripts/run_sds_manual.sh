#!/usr/bin/env bash
# Foreground execution: Ctrl+C reaches controller and worker in the same terminal group.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
PYTHON=/home/xxs/anaconda3/envs/a2/bin/python
# Keep older shorthand usable; all other flags are explicit Python CLI arguments.
case "${1:-}" in
  --check) shift; set -- --mode check "$@" ;;
  --all) shift; set -- --mode all "$@" ;;
  --teachers) shift; set -- --mode teachers "$@" ;;
esac
exec "$PYTHON" scripts/run_a2_v10_formal.py "$@"
