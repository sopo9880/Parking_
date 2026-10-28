"""Sequential local research queue for the explicitly supported replay tasks."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=21600)
    p.add_argument('--min-free-gb', type=float, default=10)
    p.add_argument('--retries', type=int, default=1, choices=range(3))
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    if args.max_seconds <= 0 or args.min_free_gb < 0:
        p.error('Limits must be positive')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = args.run_dir / 'manager.lock'
    # Exclusive creation avoids simultaneous instances sharing a run directory.
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run directory is locked; inspect the owner before removing a stale lock')
    with os.fdopen(fd, 'w') as f:
        json.dump({'pid': os.getpid(), 'started': datetime.now().isoformat()}, f)
    try:
        statefile = args.run_dir / 'state.json'
        inputs = [args.gt]
        for split, starts in [('original', [0, 330, 900]), ('alternate', [0, 330, 660, 900])]:
            fit = args.source / split
            inputs.extend([fit / 'settings_snapshot.json', fit / 'selected_state_params.json', fit / 'selection_freeze.json'])
            run = sorted((fit / 'restart_experiments').glob('run_*'))[-1]
            inputs.append(run / 'restart_summary.json')
            for start in starts:
                inputs.extend([run / f'restart_{start}' / 'evidence' / 'slot_evidence.csv', run / f'restart_{start}' / 'RESET_SAFE_slot_timeseries.csv'])
        inputs.extend(root / n for n in ('research_replay.py', 'research_diagnose.py', 'slot_engine.py', 'restart_evaluation.py'))
        fingerprint = hashlib.sha256(json.dumps({str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}, sort_keys=True).encode()).hexdigest()
        state = json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else {'fingerprint': fingerprint, 'jobs': {}}
        deadline = time.monotonic() + args.max_seconds
        if state['fingerprint'] != fingerprint:
            raise ValueError('Input or code changed; choose a new run directory')
        jobs = [('replay', 'research_replay.py'), ('diagnosis', 'research_diagnose.py')]
        for name, script in jobs:
            job = state['jobs'].setdefault(name, {'status': 'pending', 'attempts': 0})
            if job['status'] == 'completed':
                continue
            if (args.run_dir / 'STOP').exists() or time.monotonic() >= deadline:
                job['status'] = 'stopped'; break
            while job['attempts'] <= args.retries:
                if shutil.disk_usage(args.run_dir).free < args.min_free_gb * 1024**3:
                    job['status'] = 'disk_limit'; break
                job['attempts'] += 1
                target = args.run_dir / (f'{name}_attempt_{job["attempts"]}' + ('.csv' if name == 'diagnosis' else ''))
                cmd = [sys.executable, str(root / script), '--source', str(args.source), '--output', str(target)]
                if name == 'replay': cmd += ['--gt', str(args.gt)]
                job.update(status='running', output=str(target), command=cmd)
                save(statefile, state)
                with (args.run_dir / f'{name}_attempt_{job["attempts"]}.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        if (args.run_dir / 'STOP').exists() or time.monotonic() >= deadline:
                            child.terminate(); child.wait(timeout=30); job['status'] = 'stopped'; break
                        time.sleep(1)
                    else:
                        job['status'] = 'completed' if child.returncode == 0 else 'failed'
                        job['exit_code'] = child.returncode
                save(statefile, state)
                if job['status'] in ('completed', 'stopped'): break
            if job['status'] != 'completed': break
        state['finished'] = datetime.now().isoformat()
        state['status'] = 'completed' if all(state['jobs'].get(n, {}).get('status') == 'completed' for n, _ in jobs) else 'incomplete'
        save(statefile, state)
        (args.run_dir / 'REPORT.md').write_text('# Local research queue\n\n' + '\n'.join(f'- {name}: {state["jobs"].get(name, {}).get("status", "not started")}' for name, _ in jobs) + '\n\nSaved observations only; same-video count results are not independent validation. See job outputs and logs.\n', encoding='utf-8')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
