"""Evaluate saved SAFE/Candidate traces; no model loading or training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from window_evaluation import parse_windows,write_window_evaluation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--settings',default=str(ROOT/'settings.json'))
    parser.add_argument('--windows',help='e.g. 0:00-5:30;5:30-11:00;11:00-15:00;15:00-20:30')
    args=parser.parse_args()
    settings=json.loads(Path(args.settings).read_text(encoding='utf-8-sig'))
    if args.windows: settings.setdefault('validation',{})['evaluation_windows']=parse_windows(args.windows)
    print(write_window_evaluation(args.run_dir,settings))

if __name__=='__main__': main()
