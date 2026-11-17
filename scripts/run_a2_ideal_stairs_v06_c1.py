"""Fixed C1 continuation from the byte-bound stop004224 source; no fresh mode."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'scripts']:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
if 'src.tasks.velocity.v06.spec' in sys.modules:
    raise RuntimeError('C1 must start in a clean interpreter before spec imports')
os.environ['A2_V06_CONTINUATION']='v06-c1-speed030060-budget36000'
import a2_v06_bootstrap
a2_v06_bootstrap.activate(ROOT)
if __name__=='__main__':
    from src.tasks.velocity.v06.c1_runtime import main
    raise SystemExit(main())
