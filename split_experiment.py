"""Fresh, isolated DEV fitting followed by frozen TEST evaluation.

These are offline same-video split experiments, not neural-network training or
independent unseen-video validation. No cross-protocol cache is reused.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from split_protocol import validate_protocol, protocol_context, dev_mask, labels, learning_times


def protocols(settings):
    horizon = float(settings.get('eval_end_sec', 1230))
    cfg = settings.get('validation', {}).get('split_experiments', {})
    return [validate_protocol(p, horizon) for p in (
        {'name': 'original', 'dev': [0, settings.get('dev_end_sec', 900)],
         'test': [settings.get('dev_end_sec', 900), horizon]},
        {'name': 'alternate', 'dev': cfg.get('dev', [330, horizon]),
         'test': cfg.get('test', [0, 330])})]


def execute(request, progress=None):
    from utils import load_ground_truth, save_json
    from detector_tuner import run_warped_detector_sweep
    from slot_engine import (learn_slots_and_candidates, extract_evidence,
                             extract_segmentation_assist, search_state_parameters,
                             evaluate_state_output)
    from transition_refiner_v164 import run, RefinerConfig
    out = Path(request['output']); out.mkdir(parents=True, exist_ok=True)
    settings = copy.deepcopy(request['settings'])
    protocol = validate_protocol(request['protocol'], settings.get('eval_end_sec', 1230))
    settings['active_split_protocol'] = protocol
    settings['detector_by_cctv'] = {}
    settings.setdefault('detector_sweep', {})['apply_best_to_settings'] = True
    settings.setdefault('pipeline_cache', {})['enabled'] = False
    slots = out/'slots.json'
    slots.write_bytes(Path(request['slots']).read_bytes())
    gt = load_ground_truth(request['gt'])
    with protocol_context(protocol):
        gt = gt[labels(gt.time_sec) != 'IGNORED'].copy()
        dev = gt[dev_mask(gt.time_sec, settings)].copy()
        warmup = float(settings.get('evaluation_warmup_sec', 10))
        for split in ('DEV', 'TEST'):
            if not ((labels(gt.time_sec) == split) & (gt.time_sec >= warmup)).any():
                raise ValueError(f'{split} has no GT samples after warm-up')
        dev.to_csv(out/'DEV_ground_truth.csv', index=False, encoding='utf-8-sig')
        run_warped_detector_sweep(request['video'], request['rois'],
                                 str(out/'DEV_ground_truth.csv'), settings,
                                 str(out/'detector_tuning'), progress)
        if not all(c in settings['detector_by_cctv'] for c in request['rois']):
            raise RuntimeError('DEV detector sweep did not select a detector for every CCTV')
        learn_slots_and_candidates(request['video'], request['rois'], str(slots),
                                   settings, str(out/'learned'), progress)
        evidence = extract_evidence(request['video'], request['rois'], str(slots),
                                    settings, str(out/'learned'), str(out/'evidence'), progress)
        evidence = extract_segmentation_assist(request['video'], request['rois'], str(slots),
                                              settings, evidence, str(out/'evidence'), progress)
        # Only DEV labels reach selection. Slot event annotations are deliberately
        # excluded here; TEST labels enter below, after the selection is frozen.
        search_state_parameters(evidence, dev, settings, str(out), progress,
                                slot_gt_events_path=None)
        selected = out/'selected_state_params.json'
        frozen = hashlib.sha256(selected.read_bytes()).hexdigest()
        save_json(out/'selection_freeze.json', {'sha256': frozen, 'protocol': protocol,
                                               'selection_gt_times': dev.time_sec.tolist(),
                                               'detectors': settings['detector_by_cctv']})
        save_json(out/'settings_snapshot.json', settings)
        (out/'temporal_variant_comparison.csv').rename(out/'DEV_selection_variant_comparison.csv')
        from experimental_selection import freeze_dev_selection
        experimental_freeze=freeze_dev_selection(out)
        rows = []
        for name, tag in [('SAFE_BASELINE', 'baseline'), ('TRANSITION_GUARD', 'transition_guard'),
                          ('SEG_ASSIST', 'seg_assist'), ('SELECTED', 'selected'),
                          ('EMPTY_REF', 'empty_ref_assist'),
                          ('PRODUCTION_SAFE_SELECTED','production_safe_selected'),
                          ('DEV_SELECTED_EXPERIMENTAL','dev_selected_experimental')]:
            path = out/f'{tag}_global_slot_timeseries.csv'
            if not path.exists():
                continue
            result = evaluate_state_output(pd.read_csv(path), gt,
                                           settings.get('dev_end_sec', 900), warmup)
            result['timeseries'].to_csv(out/f'{tag}_count_timeseries.csv', index=False,
                                        encoding='utf-8-sig')
            rows.extend({'protocol': protocol['name'], 'variant': name, 'split': split,
                         'under_rate': result[split]['false_empty_bias_rate'],
                         **result[split]} for split in ('DEV', 'TEST'))
            if tag == 'selected':
                pd.DataFrame([{'split': split, **result[split]} for split in ('DEV','TEST')]).to_csv(
                    out/'metrics_dev_test.csv', index=False, encoding='utf-8-sig')
                trace=result['timeseries']
                trace[(trace['split']=='TEST')&(trace['error']!=0)].to_csv(
                    out/'test_error_cases.csv', index=False, encoding='utf-8-sig')
        run(out, RefinerConfig())
        candidate = pd.read_csv(out/'candidate_v164_metrics.csv')
        rows.extend({'protocol': protocol['name'], 'variant': 'CANDIDATE', **row}
                    for row in candidate.to_dict('records'))
        experimental_path=out/'dev_selected_experimental_params.json'
        assert experimental_freeze['sha256']==hashlib.sha256(experimental_path.read_bytes()).hexdigest(), 'Experimental selection changed during TEST evaluation'
        assert frozen == hashlib.sha256(selected.read_bytes()).hexdigest(), 'Selection changed during TEST evaluation'
        pd.DataFrame(rows).to_csv(out/'split_final_metrics.csv', index=False, encoding='utf-8-sig')
        (out/'REPORT.txt').write_text(
            'Fresh DEV/TEST split experiment\n'+json.dumps(protocol)+'\n'
            'Detector configuration sweep and slot geometry/templates fit on DEV only. '
            'Fixed YOLO weights; no neural-network retraining. Variant selection excludes slot-event labels.\n'
            'TEST labels used after selection freeze. Chronological inference starts at video time zero.\n'
            'Offline same-video protocol, with historically reviewed data; not independent future validation.\n'
            'DEV_selection_variant_comparison.csv and state_search_leaderboard.csv contain DEV-only selection; '
            'split_final_metrics.csv contains final DEV and TEST evaluations.\n\n'
            +pd.DataFrame(rows).to_string(index=False)+'\n', encoding='utf-8')
        save_json(out/'split_audit.json', {
            'status': 'completed', 'protocol': protocol, 'cache_reused': False,
            'test_used_for_selection': False, 'slot_gt_used_for_selection': False,
            'selected_params_sha256': frozen,
            'dev_selected_experimental': experimental_freeze,
            'learning_times': learning_times(settings, float(settings.get('learn_sample_sec', 2))),
            'evaluation_warmup_sec': warmup,
            'boundary': '[start,end), including horizon endpoint',
            'scope': 'offline same-video experiment; historically reviewed data; not independent validation',
            'segmentation_available': not (out/'evidence/SEGMENTATION_UNAVAILABLE.txt').exists(),
        })
        if settings.get('validation',{}).get('restart_experiments',{}).get('enabled',True):
            from restart_evaluation import run_restart_experiments
            try:
                run_restart_experiments(request['video'],request['rois'],load_ground_truth(request['gt']),settings,out,progress)
            except Exception as exc:
                restart_dir=out/'restart_experiments';restart_dir.mkdir(exist_ok=True)
                save_json(restart_dir/'failure.json',{'status':'failed','error':str(exc)})
                raise
    return rows


def run_suite(video, rois, gt, slots, settings, run_dir, progress=None):
    import gc
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    root = Path(run_dir)/'split_experiments'; root.mkdir(parents=True, exist_ok=True)
    planned = protocols(settings)
    rows, failures = [], []
    for index, protocol in enumerate(planned):
        child = root/protocol['name']; child.mkdir(exist_ok=True)
        request = child/'worker_input.json'
        request.write_text(json.dumps({'video': str(video), 'rois': rois, 'gt': str(gt),
            'slots': str(slots), 'settings': settings, 'protocol': protocol,
            'output': str(child.resolve())}, ensure_ascii=False), encoding='utf-8')
        if progress: progress(index/len(planned), f"Split experiments {index+1}/{len(planned)} | {protocol['name']}")
        with (child/'worker.log').open('w', encoding='utf-8') as log:
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(request)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', env=env)
            for line in process.stdout:
                log.write(line); log.flush()
                if progress: progress(index/len(planned), f"Split experiments {index+1}/{len(planned)} | {line.strip()}")
            returncode = process.wait()
        if returncode:
            failures.append(protocol['name'])
        else:
            rows.extend(pd.read_csv(child/'split_final_metrics.csv').to_dict('records'))
    pd.DataFrame(rows).to_csv(root/'split_experiment_comparison.csv', index=False, encoding='utf-8-sig')
    (root/'suite_summary.json').write_text(json.dumps({'status': 'failed' if failures else 'completed',
        'protocols': planned, 'failures': failures}, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError('Split experiments failed: '+', '.join(failures)+'. See split_experiments/*/worker.log; partial outputs retained.')
    if progress: progress(1, 'Split experiments complete')
    return root


if __name__ == '__main__':
    execute(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')),
            lambda value, message: print(message, flush=True))
