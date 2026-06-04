"""
DNS Project — Search Results Analysis
======================================

Reads all search run results from runs/search/ and prints a comparison table.

Usage:
    python analyse_search.py

Output:
    - Table of all search runs sorted by val loss
    - Best config recommendation for full training
"""

import os
import glob
import csv
import json

import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = os.path.expanduser('~/nn_proj')
SEARCH_DIR  = os.path.join(PROJECT_DIR, 'runs', 'search')

# =============================================================================
# COLLECT RESULTS
# =============================================================================

def read_run(run_dir, model_type):
    """
    Read results from one run directory.
    Returns a dict with all metrics, or None if run is incomplete.
    """
    best_model_path = os.path.join(run_dir, 'best_model.pt')
    log_path        = os.path.join(run_dir, 'training_log.csv')
    emissions_path  = os.path.join(run_dir, 'emissions.csv')

    # skip if training not complete
    if not os.path.exists(best_model_path):
        return None
    if not os.path.exists(log_path):
        return None

    # read training log
    try:
        log_df = pd.read_csv(log_path)
    except Exception as e:
        print(f"  WARNING: could not read {log_path}: {e}")
        return None

    if len(log_df) == 0:
        return None

    # best val loss epoch
    best_idx      = log_df['val_loss'].idxmin()
    best_epoch    = int(log_df.loc[best_idx, 'epoch'])
    best_val_loss = float(log_df.loc[best_idx, 'val_loss'])
    best_train_loss = float(log_df.loc[best_idx, 'train_loss'])
    best_gap      = float(log_df.loc[best_idx, 'gap'])

    # final epoch metrics
    final_row     = log_df.iloc[-1]
    epochs_ran    = int(final_row['epoch'])
    final_lr      = float(final_row['lr'])

    # energy from final epoch cumulative
    total_wh  = float(final_row.get('cumulative_energy_wh', 0.0))
    total_co2 = float(final_row.get('cumulative_co2_g',    0.0))

    # try emissions.csv for more accurate final numbers
    if os.path.exists(emissions_path):
        try:
            em_df    = pd.read_csv(emissions_path)
            if len(em_df) > 0:
                total_wh  = float(em_df['energy_consumed'].iloc[-1]) * 1000
                total_co2 = float(em_df['emissions'].iloc[-1])        * 1000
        except Exception:
            pass

    # parse folder name to extract config
    # folder name format: gru_h161 or crn_h161
    folder_name = os.path.basename(run_dir)
    hidden_size = int(folder_name.split('h')[-1])

    # parse parent folder name: lr5em03_samp500
    parent_name = os.path.basename(os.path.dirname(run_dir))
    parts       = parent_name.split('_')
    lr_str      = parts[0].replace('lr', '').replace('em0', 'e-0').replace('em', 'e-')
    try:
        lr = float(lr_str)
    except Exception:
        lr = 0.0
    samp = int(parts[1].replace('samp', '')) if len(parts) > 1 else 0

    return {
        'model_type':       model_type.upper(),
        'hidden_size':      hidden_size,
        'learning_rate':    lr,
        'train_samples':    samp,
        'epochs_ran':       epochs_ran,
        'best_epoch':       best_epoch,
        'best_val_loss':    round(best_val_loss,   6),
        'best_train_loss':  round(best_train_loss, 6),
        'gap':              round(best_gap,         6),
        'final_lr':         round(final_lr,         6),
        'energy_wh':        round(total_wh,         4),
        'co2_g':            round(total_co2,        4),
        'run_dir':          run_dir,
    }


def collect_all_results(search_dir):
    """
    Walk search_dir and collect results from all completed runs.

    Expected structure:
        search_dir/
            lr<x>_samp<n>/
                gru_h<hidden>/
                crn_h<hidden>/
    """
    results = []

    if not os.path.exists(search_dir):
        print(f"Search directory not found: {search_dir}")
        return results

    # walk two levels deep
    for config_dir in sorted(glob.glob(os.path.join(search_dir, '*'))):
        if not os.path.isdir(config_dir):
            continue

        for model_dir in sorted(glob.glob(os.path.join(config_dir, '*'))):
            if not os.path.isdir(model_dir):
                continue

            folder = os.path.basename(model_dir)
            if folder.startswith('gru'):
                model_type = 'gru'
            elif folder.startswith('crn'):
                model_type = 'crn'
            else:
                continue

            result = read_run(model_dir, model_type)
            if result is not None:
                results.append(result)
            else:
                print(f"  Incomplete or missing: {model_dir}")

    return results


# =============================================================================
# PRINT TABLE
# =============================================================================

def print_table(results):
    """Print a formatted comparison table sorted by val loss."""

    if len(results) == 0:
        print("No completed runs found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values('best_val_loss')

    # display columns
    display_cols = [
        'model_type', 'hidden_size', 'learning_rate', 'train_samples',
        'epochs_ran', 'best_epoch', 'best_val_loss', 'best_train_loss',
        'gap', 'energy_wh', 'co2_g'
    ]

    display_df = df[display_cols].copy()
    display_df.columns = [
        'Model', 'Hidden', 'LR', 'Samp',
        'Epochs', 'BestEp', 'BestVal', 'BestTrain',
        'Gap', 'Energy(Wh)', 'CO2(g)'
    ]

    print()
    print("=" * 110)
    print("SEARCH RESULTS — sorted by best val loss (lower is better)")
    print("=" * 110)
    print(display_df.to_string(index=False))
    print()

    # best per model type
    for mtype in ['GRU', 'CRN']:
        subset = df[df['model_type'] == mtype]
        if len(subset) == 0:
            continue
        best = subset.iloc[0]
        print(f"Best {mtype}: lr={best['learning_rate']}  "
              f"hidden={best['hidden_size']}  "
              f"val_loss={best['best_val_loss']:.6f}  "
              f"energy={best['energy_wh']:.2f}Wh")

    print()

    # recommendation
    best_overall = df.iloc[0]
    print("=" * 110)
    print("RECOMMENDATION FOR FULL TRAINING")
    print("=" * 110)
    print(f"  Best overall model : {best_overall['model_type']}")
    print(f"  Learning rate      : {best_overall['learning_rate']}")
    print(f"  Hidden size        : {best_overall['hidden_size']}")
    print(f"  Best val loss      : {best_overall['best_val_loss']:.6f}")
    print()
    print("  Set in your full training sbatch:")
    print(f"    export LEARNING_RATE={best_overall['learning_rate']}")
    print(f"    export HIDDEN_SIZE_A=<best GRU hidden>")
    print(f"    export HIDDEN_SIZE_B=<best CRN hidden>")
    print(f"    export NUM_EPOCHS=60")
    print(f"    export RUN_TYPE=full_train")
    print(f"    export TRAIN_SAMPLES=500   # then 1000, then 2000")
    print()

    # also save to CSV
    out_csv = os.path.join(PROJECT_DIR, 'runs', 'search_results_summary.csv')
    df[display_cols].to_csv(out_csv, index=False)
    print(f"  Full results saved: {out_csv}")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 110)
    print("DNS Project — Search Results Analysis")
    print("=" * 110)
    print(f"Reading from: {SEARCH_DIR}")
    print()

    results = collect_all_results(SEARCH_DIR)
    print(f"Found {len(results)} completed runs.")

    print_table(results)
