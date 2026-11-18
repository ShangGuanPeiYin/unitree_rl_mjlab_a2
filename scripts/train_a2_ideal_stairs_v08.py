"""Native v08 train entry; no legacy task registry scan."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for location in (ROOT,ROOT/'scripts'):
    if str(location) not in sys.path:sys.path.insert(0,str(location))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)
def main(argv=None):
    from src.tasks.velocity.v08.runtime import main as native_main
    return native_main(argv,allowed_modes=('formal', 'smoke', 'probe'))
if __name__=='__main__':raise SystemExit(main())
