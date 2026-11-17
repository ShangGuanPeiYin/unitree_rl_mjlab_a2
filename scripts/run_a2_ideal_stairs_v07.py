"""Native v07 fixed4817 continuation or exact same-version recovery."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for entry in (ROOT,ROOT/'scripts'):
    if str(entry) not in sys.path:sys.path.insert(0,str(entry))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
def main(argv=None):
    from src.tasks.velocity.v07.runtime import main as native_main
    return native_main(argv)
if __name__=='__main__':raise SystemExit(main())
