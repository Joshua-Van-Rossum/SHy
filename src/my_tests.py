import os
import glob
import pickle
from pprint import pprint

files = ['n2_list.pkl', 'n4_list.pkl', 'r2_list.pkl', 'r4_list.pkl', 'prediction_loss_per_epoch.pkl', 'test_loss_per_epoch.pkl', 'train_average_loss_per_epoch.pkl']

# Locate repo root relative to this script
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_root = os.path.join(repo_root, 'training_logs')

# Prefer the specific training folder; fall back to searching all folders if it's missing.
preferred_subfolder = '12_04_2025M18_33_10__3407__MIMIC_IV'
preferred_dir = os.path.join(logs_root, preferred_subfolder)

for fname in files:
    if os.path.isdir(preferred_dir):
        target = os.path.join(preferred_dir, fname)
        if not os.path.exists(target):
            print(f"File {fname} not found in preferred folder {preferred_subfolder}.")
            # fall back to searching other subfolders
            pattern = os.path.join(logs_root, '*', fname)
            matches = glob.glob(pattern)
            if not matches:
                print(f"  No matches for {fname} under {logs_root}")
                continue
            latest = max(matches, key=os.path.getmtime)
        else:
            latest = target
    else:
        # preferred folder missing; search all folders
        pattern = os.path.join(logs_root, '*', fname)
        matches = glob.glob(pattern)
        if not matches:
            print(f"No matches for {fname} under {logs_root}")
            continue
        latest = max(matches, key=os.path.getmtime)

    print(f"\nFile: {fname} -> Loading: {latest}")
    try:
        with open(latest, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  Failed to load {latest}: {e}")
        continue

    # Extract a sensible 'last' value from common container types
    last_val = None
    try:
        # lists/tuples
        if isinstance(data, (list, tuple)):
            last_val = data[-1] if len(data) > 0 else None
        # numpy arrays and array-like
        else:
            import numpy as _np
            arr = _np.asarray(data)
            if arr.size > 0:
                last_val = arr.flatten()[-1]
            else:
                last_val = None
    except Exception:
        # dicts or other containers: try to get last value by insertion order
        try:
            if isinstance(data, dict):
                vals = list(data.values())
                last_val = vals[-1] if vals else None
            elif hasattr(data, '__len__'):
                last_val = data[-1]
            else:
                last_val = data
        except Exception:
            last_val = data

    print('  Last value:', repr(last_val))