"""Follow an existing pointcloud run's JSONL metrics without restarting training."""
import argparse
import json
import time
import uuid
from pathlib import Path
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project', default='a2-pointcloud')
    parser.add_argument('--entity')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    config = json.loads((folder / 'config.json').read_text())
    identity = folder / 'wandb-follow-id.txt'
    if not identity.exists():
        identity.write_text(uuid.uuid4().hex[:8])
    run = wandb.init(project=args.project, entity=args.entity,
                     id=identity.read_text().strip(), resume='allow',
                     name=folder.name, config=config, dir=str(folder),
                     settings=wandb.Settings(disable_code=True))
    run.define_metric('iteration')
    run.define_metric('*', step_metric='iteration')
    last = int(run.summary.get('iteration', 0))
    try:
        with (folder / 'metrics.jsonl').open() as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    stream.seek(position)
                    if args.once:
                        break
                    time.sleep(5)
                    continue
                row = json.loads(line)
                iteration = int(row['iteration'])
                if iteration > last:
                    run.log(row, step=iteration)
                    last = iteration
    finally:
        run.finish()


if __name__ == '__main__':
    main()
