"""Sequential local research queue for the explicitly supported replay tasks."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resource_sample(child, run_dir):
    """Sample process memory, disk, and whole-device GPU usage; never infer GPU ownership."""
    try:
        process = psutil.Process(child.pid)
        rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        rss = None
    try:
        info = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        gpu = info.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        gpu = ''
    log = run_dir / 'resources.csv'
    new = not log.exists()
    with log.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new: writer.writerow(['time', 'pid', 'process_tree_rss_bytes', 'free_disk_bytes', 'gpu_total_device_util_mem_used_total'])
        writer.writerow([datetime.now().isoformat(), child.pid, rss, shutil.disk_usage(run_dir).free, gpu])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=21600)
    p.add_argument('--min-free-gb', type=float, default=10)
    p.add_argument('--retries', type=int, default=1, choices=range(3))
    p.add_argument('--fresh-config', type=Path, help='JSON with video, gt, fit and app paths for optional GPU job')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    if args.max_seconds < 0 or args.min_free_gb < 0:
        p.error('Limits cannot be negative; max-seconds 0 means no deadline')
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
        inputs.extend(root / n for n in ('research_manager.py', 'research_replay.py', 'research_diagnose.py', 'slot_engine.py', 'restart_evaluation.py', 'manual_initialization.py', 'utils.py', 'detector.py', 'segmentation.py'))
        fresh = None
        if args.fresh_config:
            fresh = json.loads(args.fresh_config.read_text(encoding='utf-8'))
            if not {'video', 'gt', 'fit', 'app'}.issubset(fresh) or set(fresh) - {'video', 'gt', 'fit', 'app', 'start'}:
                raise ValueError('fresh-config requires video, gt, fit, app and optional start')
            if int(fresh.get('start', 900)) not in (0,330,660,900): raise ValueError('Invalid restart time')
            inputs.extend([args.fresh_config, root / 'fresh_manual_experiment.py', Path(fresh['video']), Path(fresh['gt'])])
            inputs.extend(Path(fresh['app']) / model for model in ('yolov8m.pt', 'yolov8l.pt', 'yolov8m-seg.pt'))
            fit = Path(fresh['fit'])
            inputs.extend(path for path in fit.rglob('*') if path.is_file() and 'restart_experiments' not in path.relative_to(fit).parts)
            latest = sorted((fit / 'restart_experiments').glob('run_*'))[-1]
            inputs.extend([latest / 'restart_summary.json', latest / f"restart_{int(fresh.get('start', 900))}" / 'evidence' / 'slot_evidence.csv'])
        fingerprint = hashlib.sha256(json.dumps({str(path.resolve()): digest(path) for path in inputs}, sort_keys=True).encode()).hexdigest()
        state = json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else {'fingerprint': fingerprint, 'jobs': {}}
        deadline = time.monotonic() + args.max_seconds if args.max_seconds else float('inf')
        if state['fingerprint'] != fingerprint:
            raise ValueError('Input or code changed; choose a new run directory')
        jobs = [('replay', 'research_replay.py'), ('diagnosis', 'research_diagnose.py')]
        if fresh is not None: jobs.append(('fresh_gpu', 'fresh_manual_experiment.py'))
        for name, script in jobs:
            job = state['jobs'].setdefault(name, {'status': 'pending', 'attempts': 0, 'failures': 0})
            if job['status'] == 'completed':
                continue
            if (args.run_dir / 'STOP').exists() or time.monotonic() >= deadline:
                job['status'] = 'stopped'; break
            while job.get('failures', 0) <= args.retries:
                if shutil.disk_usage(args.run_dir).free < args.min_free_gb * 1024**3:
                    job['status'] = 'disk_limit'; break
                job['attempts'] += 1
                target = args.run_dir / (f'{name}_attempt_{job["attempts"]}' + ('.csv' if name == 'diagnosis' else ''))
                if name == 'fresh_gpu':
                    cmd = [sys.executable, str(root / script), '--output', str(target)]
                    for key in ('video', 'gt', 'fit', 'app'): cmd += ['--' + key, fresh[key]]
                    cmd += ['--start', str(fresh.get('start', 900))]
                else:
                    cmd = [sys.executable, str(root / script), '--source', str(args.source), '--output', str(target)]
                    if name == 'replay': cmd += ['--gt', str(args.gt)]
                job.update(status='running', output=str(target), command=cmd)
                save(statefile, state)
                with (args.run_dir / f'{name}_attempt_{job["attempts"]}.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    last_sample = 0
                    while child.poll() is None:
                        if time.monotonic() - last_sample >= 10:
                            resource_sample(child, args.run_dir); last_sample = time.monotonic()
                        if ((args.run_dir / 'STOP').exists() or time.monotonic() >= deadline or
                                shutil.disk_usage(args.run_dir).free < args.min_free_gb * 1024**3):
                            child.terminate(); child.wait(timeout=30); job['status'] = 'stopped'; break
                        time.sleep(1)
                    else:
                        job['status'] = 'completed' if child.returncode == 0 else 'failed'
                        job['exit_code'] = child.returncode
                        if child.returncode != 0: job['failures'] = job.get('failures', 0) + 1
                save(statefile, state)
                if job['status'] in ('completed', 'stopped'): break
            if job['status'] != 'completed': break
        state['finished'] = datetime.now().isoformat()
        state['status'] = 'completed' if all(state['jobs'].get(n, {}).get('status') == 'completed' for n, _ in jobs) else 'incomplete'
        save(statefile, state)
        (args.run_dir / 'REPORT.md').write_text('# Local research queue\n\n' + '\n'.join(f'- {name}: {state["jobs"].get(name, {}).get("status", "not started")}' for name, _ in jobs) + '\n\nThe replay uses saved observations; the optional fresh_gpu task extracts new video observations. Same-video results are not independent validation. See job outputs and logs.\n', encoding='utf-8')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
