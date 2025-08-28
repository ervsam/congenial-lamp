#!/usr/bin/env python3
import argparse, os, csv, sys, re
from pathlib import Path
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator

# Optional: Google Sheets
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    HAS_GSPREAD = True
except Exception:
    HAS_GSPREAD = False

def load_scalars(run_dir: Path):
    """Return a dict: tag -> list of (step, value) for all scalar tags in this run_dir."""
    # find newest event file (or pass dir directly; EA will scan all)
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 100000}
    )
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get('scalars', []):
        scalars[tag] = [(s.step, s.value, s.wall_time) for s in ea.Scalars(tag)]
    return scalars

def get_last_value(series):
    """series: list of (step,val,time) -> last value or '' if empty"""
    return series[-1][1] if series else ''

def get_max_value(series):
    """Return (best_val, best_step) for max over values, or ('','') if empty."""
    if not series:
        return '', ''
    best = max(series, key=lambda t: t[1])
    return best[1], best[0]

def parse_experiment_fields(run_path: Path):
    """
    Robustly extract (mode, window, num_agents, experiment) from run path.

    Supports:
      runs/<experiment>/w<win>/<agents>
      runs/<mode>/w<win>/<agents>/<experiment>
      and nested experiment names between runs/ and w<win>/ (joined with '/')
    """
    parts = run_path.parts
    try:
        r = parts.index('runs')
    except ValueError:
        return '', '', '', run_path.name

    # Find first 'w<digits>' segment after 'runs'
    wi = None
    for idx in range(r + 1, len(parts)):
        if re.fullmatch(r'w\d+', parts[idx]):
            wi = idx
            break
    if wi is None:
        # No window segment; fall back to last component as experiment
        return '', '', '', parts[-1]

    # Window and agents2
    win = int(parts[wi][1:]) if re.search(r'\d+', parts[wi]) else ''
    agents = int(parts[wi + 1]) if (wi + 1 < len(parts) and parts[wi + 1].isdigit()) else ''

    # Everything between 'runs/' and 'w<win>/' is the experiment string
    if wi - (r + 1) >= 1:
        exp = parts[r + 1] if wi == r + 2 else '/'.join(parts[r + 1:wi])
    else:
        exp = ''

    # Mode heuristic: first token before '_' (e.g., 'threeway_weighted' -> 'threeway')
    mode = exp.split('_')[0] if exp else ''

    return mode, win, agents, exp
def collect_one_run(run_dir: Path):
    scalars = load_scalars(run_dir)

    # Main metrics we expect from your training script
    train_acc = get_last_value(scalars.get('Accuracy/train', []))
    train_loss = get_last_value(scalars.get('Loss/train', []))
    test_acc_series = scalars.get('Accuracy/test', [])
    test_loss_series = scalars.get('Loss/test', [])

    best_val_acc, best_epoch = get_max_value(test_acc_series)
    best_test_loss_at_best = ''
    if test_loss_series and best_epoch != '':
        # find loss at the step closest to best_epoch
        # (your writer logs per-epoch; step should match epoch index)
        by_step = {s: v for (s, v, _) in test_loss_series}
        best_test_loss_at_best = by_step.get(best_epoch, '')

    # Per-class test accuracy (you log these in evaluate)
    test_cls = []
    for cls in range(3):  # threeway head → 3 classes
        tag = f'Accuracy/test_class_{cls}'
        test_cls.append(get_last_value(scalars.get(tag, [])))

    mode, win, agents, expname = parse_experiment_fields(run_dir)
    date_str = datetime.now().strftime('%Y-%m-%d')

    row = {
        'Date': date_str,
        'RunDir': str(run_dir),
        'Mode': mode,
        'Window': win,
        'NumAgents': agents,
        'Experiment': expname,
        'TrainAcc_last': train_acc,
        'TrainLoss_last': train_loss,
        'ValAcc_best': best_val_acc,
        'ValAcc_best_epoch': best_epoch,
        'ValLoss_at_best': best_test_loss_at_best,
        'TestAcc_class0': test_cls[0],
        'TestAcc_class1': test_cls[1],
        'TestAcc_class2': test_cls[2],
    }
    return row

def find_run_dirs(root: Path, regex_filter=None):
    """
    Return all leaf dirs that contain TensorBoard event files.
    Optional regex filter is applied to the path.
    """
    run_dirs = []
    pat = re.compile(regex_filter) if regex_filter else None
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if pat and not pat.search(str(p)):
            continue
        # detect any tfevents.* file
        if any(fn.startswith('events.out.tfevents') or '.tfevents.' in fn for fn in filenames):
            run_dirs.append(p)
    # Keep only leaf-most dirs (avoid duplicating parent + child)
    leaves = []
    run_dirs = sorted(set(run_dirs))
    for d in run_dirs:
        if not any((d != other and str(d).startswith(str(other)+os.sep)) for other in run_dirs):
            leaves.append(d)
    return leaves

def write_csv(rows, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'Date','Experiment','Mode','Window','NumAgents','RunDir',
        'TrainAcc_last','TrainLoss_last',
        'ValAcc_best','ValAcc_best_epoch','ValLoss_at_best',
        'TestAcc_class0','TestAcc_class1','TestAcc_class2'
    ]
    new_file = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv

def append_google_sheet(rows, sheet_id, worksheet_name, creds_json):
    if not HAS_GSPREAD:
        print("gspread is not installed. Skipping Google Sheets upload.")
        return
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)
        # header
        ws.append_row([
            'Date','Experiment','Mode','Window','NumAgents','RunDir',
            'TrainAcc_last','TrainLoss_last',
            'ValAcc_best','ValAcc_best_epoch','ValLoss_at_best',
            'TestAcc_class0','TestAcc_class1','TestAcc_class2'
        ])
    # append rows
    for r in rows:
        ws.append_row([
            r['Date'], r['Experiment'], r['Mode'], r['Window'], r['NumAgents'], r['RunDir'],
            r['TrainAcc_last'], r['TrainLoss_last'],
            r['ValAcc_best'], r['ValAcc_best_epoch'], r['ValLoss_at_best'],
            r['TestAcc_class0'], r['TestAcc_class1'], r['TestAcc_class2']
        ], value_input_option='USER_ENTERED')

def main():
    ap = argparse.ArgumentParser(description="Parse TensorBoard logs and autofill CSV / Google Sheet.")
    ap.add_argument('--runs_dir', type=str, default='runs', help='Root directory that contains TB run subfolders')
    ap.add_argument('--out_csv', type=str, default='experiment_summary.csv', help='CSV to append results')
    ap.add_argument('--filter', type=str, default=None, help='Regex to filter run paths (e.g., \"threeway.*stability\")')
    ap.add_argument('--sheet_id', type=str, default=None, help='Google Sheet ID (optional)')
    ap.add_argument('--worksheet', type=str, default='experiments', help='Worksheet/tab name')
    ap.add_argument('--creds_json', type=str, default=None, help='Path to service-account JSON (for Sheets)')
    ap.add_argument('--dry_run', action='store_true', help='Only print rows; do not write CSV or Sheets')
    args = ap.parse_args()

    root = Path(args.runs_dir)
    if not root.exists():
        print(f"[ERR] runs_dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    run_dirs = find_run_dirs(root, args.filter)
    if not run_dirs:
        print("[WARN] No TensorBoard runs found.")
        sys.exit(0)

    rows = []
    for rd in run_dirs:
        try:
            rows.append(collect_one_run(rd))
        except Exception as e:
            print(f"[WARN] Skipping {rd}: {e}")

    if args.dry_run:
        for r in rows:
            print(r)
        return

    out_csv = write_csv(rows, Path(args.out_csv))
    print(f"[OK] Appended {len(rows)} rows to {out_csv}")

    if args.sheet_id and args.creds_json:
        if not HAS_GSPREAD:
            print("[WARN] gspread not installed; skipping Google Sheets upload.")
        else:
            append_google_sheet(rows, args.sheet_id, args.worksheet, args.creds_json)
            print(f"[OK] Appended {len(rows)} rows to Google Sheet {args.sheet_id} / {args.worksheet}")

if __name__ == '__main__':
    main()