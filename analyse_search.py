"""
DNS Project — Search Results Analysis
======================================

Reads all search run results from runs/search/ and prints comparison tables
including parameter counts, GPU/CPU energy split, and val loss.

Key insight:
    Total energy is dominated by CPU (online data synthesis).
    GPU energy reflects model computation only — fairer comparison.
    Parameter counts calculated from architecture definitions.

Usage:
    python analyse_search.py
"""

import os
import glob
import torch
import torch.nn as nn
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = os.path.expanduser('~/nn_proj')
SEARCH_DIR  = os.path.join(PROJECT_DIR, 'runs', 'search')

# =============================================================================
# PARAMETER COUNT — calculated from architecture definitions
# Mirrors exactly what is in block3.py
# =============================================================================

def count_gru_params(hidden_size, input_size=161, output_size=161):
    """Count GRUModel parameters for a given hidden size."""
    gru1   = nn.GRU(input_size,  hidden_size, batch_first=True)
    gru2   = nn.GRU(hidden_size, hidden_size, batch_first=True)
    fc     = nn.Linear(hidden_size, output_size)
    proj   = nn.Linear(input_size, hidden_size, bias=False) \
             if input_size != hidden_size else nn.Identity()
    total  = sum(p.numel() for p in gru1.parameters())
    total += sum(p.numel() for p in gru2.parameters())
    total += sum(p.numel() for p in fc.parameters())
    total += sum(p.numel() for p in proj.parameters())
    return total


def count_crn_params(hidden_size, input_size=161, output_size=161):
    """Count CRNModel parameters for a given hidden size."""
    h = hidden_size
    enc1 = nn.Sequential(nn.Conv1d(input_size, h*4, 5, padding=2),
                         nn.BatchNorm1d(h*4), nn.ELU())
    enc2 = nn.Sequential(nn.Conv1d(h*4, h*2, 5, padding=2),
                         nn.BatchNorm1d(h*2), nn.ELU())
    enc3 = nn.Sequential(nn.Conv1d(h*2, h, 5, padding=2),
                         nn.BatchNorm1d(h), nn.ELU())
    gru  = nn.GRU(h, h, num_layers=2, batch_first=True)
    dec1 = nn.Sequential(nn.ConvTranspose1d(h, h*2, 5, padding=2),
                         nn.BatchNorm1d(h*2), nn.ELU())
    dec2 = nn.Sequential(nn.ConvTranspose1d(h*2, h*4, 5, padding=2),
                         nn.BatchNorm1d(h*4), nn.ELU())
    dec3 = nn.ConvTranspose1d(h*4, output_size, 5, padding=2)

    total  = sum(p.numel() for p in enc1.parameters())
    total += sum(p.numel() for p in enc2.parameters())
    total += sum(p.numel() for p in enc3.parameters())
    total += sum(p.numel() for p in gru.parameters())
    total += sum(p.numel() for p in dec1.parameters())
    total += sum(p.numel() for p in dec2.parameters())
    total += sum(p.numel() for p in dec3.parameters())
    return total


def get_param_count(model_type, hidden_size):
    """Return parameter count for a given model type and hidden size."""
    if model_type == 'GRU':
        return count_gru_params(hidden_size)
    else:
        return count_crn_params(hidden_size)

# =============================================================================
# COLLECT RESULTS
# =============================================================================

def read_run(run_dir, model_type):
    """
    Read results from one completed run directory.
    Returns a dict with all metrics, or None if run is incomplete.
    """
    best_model_path = os.path.join(run_dir, 'best_model.pt')
    log_path        = os.path.join(run_dir, 'training_log.csv')
    emissions_path  = os.path.join(run_dir, 'emissions.csv')

    if not os.path.exists(best_model_path):
        return None
    if not os.path.exists(log_path):
        return None

    try:
        log_df = pd.read_csv(log_path)
    except Exception as e:
        print(f"  WARNING: could not read {log_path}: {e}")
        return None

    if len(log_df) == 0:
        return None

    # best val loss epoch
    best_idx        = log_df['val_loss'].idxmin()
    best_epoch      = int(log_df.loc[best_idx, 'epoch'])
    best_val_loss   = float(log_df.loc[best_idx, 'val_loss'])
    best_train_loss = float(log_df.loc[best_idx, 'train_loss'])
    best_gap        = float(log_df.loc[best_idx, 'gap'])
    epochs_ran      = int(log_df.iloc[-1]['epoch'])

    # energy — GPU / CPU / RAM split
    total_wh = gpu_wh = cpu_wh = ram_wh = total_co2 = 0.0
    avg_gpu_w = avg_cpu_w = 0.0

    if os.path.exists(emissions_path):
        try:
            em_df = pd.read_csv(emissions_path)
            if len(em_df) > 0:
                row       = em_df.iloc[-1]
                total_wh  = float(row['energy_consumed']) * 1000
                gpu_wh    = float(row['gpu_energy'])      * 1000
                cpu_wh    = float(row['cpu_energy'])      * 1000
                ram_wh    = float(row['ram_energy'])      * 1000
                total_co2 = float(row['emissions'])       * 1000
                avg_gpu_w = float(row.get('gpu_power', 0))
                avg_cpu_w = float(row.get('cpu_power', 0))
        except Exception as e:
            print(f"  WARNING: could not read {emissions_path}: {e}")
    else:
        try:
            total_wh  = float(log_df.iloc[-1].get('cumulative_energy_wh', 0.0))
            total_co2 = float(log_df.iloc[-1].get('cumulative_co2_g',    0.0))
        except Exception:
            pass

    # parse folder name: gru_h161 or crn_h161
    folder_name = os.path.basename(run_dir)
    hidden_size = int(folder_name.split('h')[-1])

    # parse parent folder: lr5em03_samp500
    parent_name = os.path.basename(os.path.dirname(run_dir))
    parts       = parent_name.split('_')
    lr_str      = parts[0].replace('lr', '') \
                           .replace('em03', 'e-03') \
                           .replace('em02', 'e-02') \
                           .replace('em01', 'e-01')
    try:
        lr = float(lr_str)
    except Exception:
        lr = 0.0
    samp = int(parts[1].replace('samp', '')) if len(parts) > 1 else 0

    gpu_pct = (gpu_wh / total_wh * 100) if total_wh > 0 else 0.0
    cpu_pct = (cpu_wh / total_wh * 100) if total_wh > 0 else 0.0

    # parameter count from architecture definition
    n_params = get_param_count(model_type.upper(), hidden_size)

    return {
        'model_type':       model_type.upper(),
        'hidden_size':      hidden_size,
        'n_params':         n_params,
        'learning_rate':    lr,
        'train_samples':    samp,
        'epochs_ran':       epochs_ran,
        'best_epoch':       best_epoch,
        'best_val_loss':    round(best_val_loss,   6),
        'best_train_loss':  round(best_train_loss, 6),
        'gap':              round(best_gap,         6),
        'total_wh':         round(total_wh,  2),
        'gpu_wh':           round(gpu_wh,    2),
        'cpu_wh':           round(cpu_wh,    2),
        'ram_wh':           round(ram_wh,    2),
        'gpu_pct':          round(gpu_pct,   1),
        'cpu_pct':          round(cpu_pct,   1),
        'avg_gpu_w':        round(avg_gpu_w, 1),
        'avg_cpu_w':        round(avg_cpu_w, 1),
        'total_co2_g':      round(total_co2, 4),
        'run_dir':          run_dir,
    }


def collect_all_results(search_dir):
    results = []
    if not os.path.exists(search_dir):
        print(f"Search directory not found: {search_dir}")
        return results

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
                print(f"  Incomplete / still running: {model_dir}")

    return results

# =============================================================================
# PRINT TABLES
# =============================================================================

def print_tables(results):
    if len(results) == 0:
        print("No completed runs found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values('best_val_loss')

    # ── Table 1: Performance ──────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("TABLE 1 — Performance (sorted by best val loss)")
    print("=" * 120)
    t1 = df[['model_type', 'hidden_size', 'n_params', 'learning_rate',
             'train_samples', 'epochs_ran', 'best_epoch',
             'best_val_loss', 'best_train_loss', 'gap']].copy()
    t1.columns = ['Model', 'Hidden', 'Params', 'LR', 'Samp',
                  'Epochs', 'BestEp', 'BestVal', 'BestTrain', 'Gap']
    print(t1.to_string(index=False))

    # ── Table 2: Energy breakdown ─────────────────────────────────────────────
    print()
    print("=" * 120)
    print("TABLE 2 — Energy breakdown: GPU vs CPU vs RAM")
    print("(CPU energy = data synthesis — same for all models regardless of params)")
    print("=" * 120)
    t2 = df[['model_type', 'hidden_size', 'n_params', 'learning_rate',
             'total_wh', 'gpu_wh', 'gpu_pct',
             'cpu_wh', 'cpu_pct', 'ram_wh',
             'avg_gpu_w', 'avg_cpu_w']].copy()
    t2.columns = ['Model', 'Hidden', 'Params', 'LR',
                  'Total(Wh)', 'GPU(Wh)', 'GPU%',
                  'CPU(Wh)', 'CPU%', 'RAM(Wh)',
                  'AvgGPU(W)', 'AvgCPU(W)']
    print(t2.to_string(index=False))

    # ── Table 3: Fair GPU-only comparison ─────────────────────────────────────
    print()
    print("=" * 120)
    print("TABLE 3 — Fair model comparison: params vs GPU energy vs val loss")
    print("(GPU energy only — excludes synthesis CPU cost)")
    print("=" * 120)
    t3 = df[['model_type', 'hidden_size', 'n_params', 'learning_rate',
             'best_val_loss', 'gpu_wh', 'avg_gpu_w']].copy()
    t3.columns = ['Model', 'Hidden', 'Params', 'LR',
                  'BestVal', 'GPU(Wh)', 'AvgGPU(W)']
    t3 = t3.sort_values('BestVal')
    print(t3.to_string(index=False))

    # ── Key insight: params vs GPU energy ─────────────────────────────────────
    print()
    print("=" * 120)
    print("KEY INSIGHT — Parameter count vs GPU energy correlation")
    print("=" * 120)
    completed = df[df['gpu_wh'] > 0].copy()
    if len(completed) > 1:
        corr = completed[['n_params', 'gpu_wh']].corr().iloc[0, 1]
        print(f"  Correlation between n_params and GPU(Wh): {corr:.3f}")
        print(f"  (1.0 = perfect, 0.0 = no relationship)")
        print()
        for mtype in ['GRU', 'CRN']:
            subset = completed[completed['model_type'] == mtype]
            if len(subset) == 0:
                continue
            avg_params = subset['n_params'].mean()
            avg_gpu    = subset['gpu_wh'].mean()
            avg_cpu    = subset['cpu_wh'].mean()
            print(f"  {mtype}:")
            print(f"    avg params   : {avg_params:,.0f}")
            print(f"    avg GPU (Wh) : {avg_gpu:.2f}")
            print(f"    avg CPU (Wh) : {avg_cpu:.2f}  "
                  f"← synthesis cost, should be ~same as GRU")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("SUMMARY — Best per model type")
    print("=" * 120)
    gru_df = df[df['model_type'] == 'GRU']
    crn_df = df[df['model_type'] == 'CRN']

    for mtype, subset in [('GRU', gru_df), ('CRN', crn_df)]:
        if len(subset) == 0:
            print(f"  {mtype}: no completed runs")
            continue
        best = subset.iloc[0]
        print(f"  Best {mtype}:")
        print(f"    lr={best['learning_rate']}  "
              f"hidden={best['hidden_size']}  "
              f"params={best['n_params']:,}")
        print(f"    val_loss={best['best_val_loss']:.6f}")
        print(f"    total={best['total_wh']:.2f}Wh  "
              f"gpu={best['gpu_wh']:.2f}Wh ({best['gpu_pct']:.1f}%)  "
              f"cpu={best['cpu_wh']:.2f}Wh ({best['cpu_pct']:.1f}%)")

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("RECOMMENDATION FOR FULL TRAINING")
    print("=" * 120)
    if len(gru_df) > 0:
        best_gru = gru_df.iloc[0]
        print(f"  GRU: LEARNING_RATE={best_gru['learning_rate']}  "
              f"HIDDEN_SIZE_A={best_gru['hidden_size']}  "
              f"params={best_gru['n_params']:,}")
    if len(crn_df) > 0:
        best_crn = crn_df.iloc[0]
        print(f"  CRN: LEARNING_RATE={best_crn['learning_rate']}  "
              f"HIDDEN_SIZE_B={best_crn['hidden_size']}  "
              f"params={best_crn['n_params']:,}")
    print()
    print("  Submit 6 full training jobs:")
    print("    TRAIN_SAMPLES = 500, 1000, 2000")
    print("    NUM_EPOCHS    = 60")
    print("    RUN_TYPE      = full_train")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(PROJECT_DIR, 'runs', 'search_results_summary.csv')
    save_cols = ['model_type', 'hidden_size', 'n_params', 'learning_rate',
                 'train_samples', 'epochs_ran', 'best_epoch',
                 'best_val_loss', 'best_train_loss', 'gap',
                 'total_wh', 'gpu_wh', 'cpu_wh', 'ram_wh',
                 'gpu_pct', 'cpu_pct', 'avg_gpu_w', 'avg_cpu_w',
                 'total_co2_g']
    df[save_cols].to_csv(out_csv, index=False)
    print()
    print(f"  Full results saved: {out_csv}")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 120)
    print("DNS Project — Search Results Analysis")
    print("=" * 120)
    print(f"Reading from: {SEARCH_DIR}")
    print()

    results = collect_all_results(SEARCH_DIR)
    print(f"Found {len(results)} completed runs.")

    print_tables(results)
