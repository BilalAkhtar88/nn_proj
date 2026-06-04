"""
DNS Project — Block 4 (Audio Reconstruction) + Block 5 (DNSMOS Evaluation)
===========================================================================

What this script does:
    1. Scans runs/full_train/ and finds every completed model (best_model.pt)
    2. Scores the unprocessed noisy test audio with DNSMOS (the floor)
    3. Scores the official ONNX baseline with DNSMOS
    4. For each completed model: enhances the test set and scores with DNSMOS
    5. Prints a running summary table after every system and saves summary.csv

Block 4 — audio reconstruction:
    Load model -> predict mask -> apply mask to noisy magnitude ->
    reuse noisy phase -> inverse STFT -> waveform

Block 5 — DNSMOS evaluation:
    DNSMOS returns three scores per clip:
        SIG  — speech signal quality
        BAK  — background suppression quality (higher = less noise)
        OVRL — overall perceptual quality

Why combined:
    Writing all enhanced audio to disk would cost ~15GB.
    Instead each clip is enhanced in memory, scored immediately, then discarded.
    Only the first WAV_SAVE_COUNT clips per model are kept for listening.

ONNX baseline notes:
    The baseline was built for the full DNS track with far-end (loudspeaker)
    signals. Our task is nearend-only — no loudspeaker, no echo, no double-talk.
    The baseline script needs _mic.wav and _lpb.wav file pairs. We create a
    silent zeros file as _lpb.wav because there is no far-end signal in our task.
    The baseline outputs 48kHz audio. We resample to 16kHz (factor of 3) before
    scoring. Do NOT pass --output_sr to the baseline script — it causes a TypeError.

Safe to run while training is still going:
    This script only reads from runs/full_train/ and writes to results/.
    Training jobs write to runs/full_train/ and never touch results/.
    No conflict. Whatever runs have best_model.pt when this job starts will
    be evaluated. Run again later to pick up any newly finished runs.

Run via sbatch:
    sbatch block4and5.sbatch

Or interactively (GPU speeds up the enhancement step):
    python3 block4and5.py
"""

import os
import sys
import glob
import csv
import time
import subprocess
import shutil
import builtins

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# Import model classes from block3.py
sys.path.insert(0, os.path.expanduser('~/nn_proj'))
from block3 import GRUModel, CRNModel
from block1and2 import N_FREQS, EVAL_DIR

# =============================================================================
# FORCE UNBUFFERED OUTPUT — every print appears immediately in sbatch log
# =============================================================================

_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _orig_print(*args, **kwargs)
builtins.print = print

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR  = os.path.expanduser('~/nn_proj')
TEST_PT_DIR  = os.path.join(EVAL_DIR, 'test', 'pt_files')
TEST_WAV_DIR = os.path.join(EVAL_DIR, 'test', 'wav_files')
RUNS_DIR     = os.path.join(PROJECT_DIR, 'runs', 'full_train')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results')

# Challenge repo paths.
# CHALLENGE_DIR can be overridden with the CHALLENGE_DIR environment variable
# set in the sbatch. Default guesses the repo sits beside nn_proj. If you see
# "not found" warnings for dnsmos_local.py or enhance.py, set CHALLENGE_DIR to
# the real repo path in the sbatch before running.
CHALLENGE_DIR       = os.environ.get(
    'CHALLENGE_DIR',
    os.path.join(PROJECT_DIR, '..', 'nnp_track02-master'),
)
ONNX_MODEL_PATH     = os.path.join(CHALLENGE_DIR, 'baseline',
                                   'dec-baseline-model-icassp2022.onnx')
BASELINE_ENHANCE_PY = os.path.join(CHALLENGE_DIR, 'baseline', 'enhance.py')
DNSMOS_LOCAL_PY     = os.path.join(CHALLENGE_DIR, 'dnsmos_local.py')
DNSMOS_MODEL_DIR    = os.path.join(CHALLENGE_DIR, 'DNSMOS')

# =============================================================================
# STFT CONSTANTS — must match block1and2.py exactly
# =============================================================================

SAMPLE_RATE = 16_000
N_SAMPLES   = SAMPLE_RATE * 15          # 240,000 samples per 15s clip
FRAME_SIZE  = int(0.02 * SAMPLE_RATE)   # 320 samples = 20ms
HOP_SIZE    = int(0.01 * SAMPLE_RATE)   # 160 samples = 10ms
DFT_SIZE    = 320
N_FREQS     = DFT_SIZE // 2 + 1        # 161 frequency bins

# Number of enhanced wav files to keep per model (for listening and report)
WAV_SAVE_COUNT = 20

# Number of test samples to score (2000 = full test set)
N_SCORE = 2000

# =============================================================================
# BLOCK 4 — AUDIO RECONSTRUCTION FROM MASK
# =============================================================================

def reconstruct_audio(noisy_mag, noisy_phase, mask):
    """
    [Block 4] Convert a predicted mask back into a waveform.

    Steps:
        1. enhanced_mag  = mask * noisy_mag      suppress noisy frequency bins
        2. enhanced_spec = enhanced_mag * phase  reuse noisy phase (standard
                                                 approximation — ear is tolerant
                                                 of phase error)
        3. irfft per frame                       back to time-domain frames
        4. apply synthesis window + overlap-add  reconstruct waveform
        5. remove left-padding                   undo the one-hop padding from STFT
        6. clamp to [-1, 1]                      prevent clipping

    Args:
        noisy_mag   : (T, 161) float32 tensor — linear magnitude from .pt file
        noisy_phase : (T, 161) complex64 tensor — phase from .pt file
        mask        : (T, 161) float32 tensor — model output, values in [0, 1]

    Returns:
        wav : (N_SAMPLES,) float32 numpy array at 16kHz
    """
    # Step 1 + 2: apply mask, form complex spectrogram
    enhanced_mag  = mask * noisy_mag
    enhanced_spec = enhanced_mag * noisy_phase

    # Step 3: inverse FFT per frame -> (T, FRAME_SIZE)
    frames = torch.fft.irfft(enhanced_spec, n=DFT_SIZE, dim=-1)

    # Step 4: synthesis window (same sqrt Hanning as analysis in block1and2)
    window = torch.from_numpy(
        np.sqrt(np.hanning(FRAME_SIZE + 1)[:-1]).astype(np.float32)
    )
    frames = frames * window

    # Overlap-add: the forward STFT left-padded by one HOP_SIZE so the
    # output buffer needs to be N_SAMPLES + HOP_SIZE before trimming
    wav = torch.zeros(N_SAMPLES + HOP_SIZE)
    for t, frame in enumerate(frames):
        start = t * HOP_SIZE
        wav[start : start + FRAME_SIZE] += frame

    # Step 5: remove the left-padding added during STFT
    wav = wav[HOP_SIZE : HOP_SIZE + N_SAMPLES]

    # Step 6: clamp to prevent clipping
    wav = torch.clamp(wav, -1.0, 1.0)

    return wav.numpy()


def enhance_one_sample(model, noisy_feat, noisy_mag, noisy_phase, device):
    """
    [Block 4] Run model forward pass and reconstruct enhanced audio.

    Args:
        model       : trained GRU or CRN model in eval mode on device
        noisy_feat  : (T, 161) float32 — log-power spectrogram
        noisy_mag   : (T, 161) float32 — linear magnitude
        noisy_phase : (T, 161) complex64 — phase
        device      : torch.device

    Returns:
        enhanced_wav : (N_SAMPLES,) float32 numpy array
    """
    x = noisy_feat.unsqueeze(0).to(device)   # (1, T, 161)

    with torch.no_grad():
        mask = model(x)                       # (1, T, 161)

    mask = mask.squeeze(0).cpu()             # (T, 161)
    return reconstruct_audio(noisy_mag, noisy_phase, mask)

# =============================================================================
# BLOCK 5 — DNSMOS SCORING
# =============================================================================

def run_dnsmos(wav_dir):
    """
    [Block 5] Score all *_mic.wav files in wav_dir using dnsmos_local.py.

    dnsmos_local.py returns SIG, BAK, OVRL scores for each file.
    Do NOT pass --output_sr — it causes a TypeError in the challenge version.

    Args:
        wav_dir : directory containing *_mic.wav files

    Returns:
        scores : list of dicts {filename, SIG, BAK, OVRL}
                 empty list if dnsmos_local.py is not found
    """
    if not os.path.exists(DNSMOS_LOCAL_PY):
        print(f"  WARNING: dnsmos_local.py not found at {DNSMOS_LOCAL_PY}")
        return []

    tmp_csv = os.path.join(wav_dir, '_tmp_scores.csv')

    result = subprocess.run(
        [sys.executable, DNSMOS_LOCAL_PY,
         '-t', wav_dir,
         '-o', tmp_csv,
         '--model_path', DNSMOS_MODEL_DIR],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  ERROR from dnsmos_local.py:\n{result.stderr[:400]}")
        return []

    scores = []
    if os.path.exists(tmp_csv):
        with open(tmp_csv) as f:
            for row in csv.DictReader(f):
                scores.append({
                    'filename': row.get('filename', row.get('file', '')),
                    'SIG':  float(row.get('SIG',  row.get('sig',  0))),
                    'BAK':  float(row.get('BAK',  row.get('bak',  0))),
                    'OVRL': float(row.get('OVRL', row.get('ovrl', 0))),
                })
        os.remove(tmp_csv)

    return scores


def mean_scores(scores):
    """Return mean SIG, BAK, OVRL from a list of score dicts."""
    if not scores:
        return {'SIG': 0.0, 'BAK': 0.0, 'OVRL': 0.0}
    n = len(scores)
    return {
        'SIG':  sum(s['SIG']  for s in scores) / n,
        'BAK':  sum(s['BAK']  for s in scores) / n,
        'OVRL': sum(s['OVRL'] for s in scores) / n,
    }


def save_scores_csv(scores, path):
    """Save per-sample DNSMOS scores to a csv file."""
    if not scores:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'SIG', 'BAK', 'OVRL'])
        writer.writeheader()
        writer.writerows(scores)

# =============================================================================
# RUNNING SUMMARY TABLE — printed after every system
# =============================================================================

def print_summary_table(summary_rows):
    """Print a formatted table of all systems scored so far."""
    print()
    print("  Running summary so far:")
    print(f"  {'System':<45}  {'SIG':>5}  {'BAK':>5}  {'OVRL':>5}")
    print("  " + "-" * 63)
    for row in summary_rows:
        print(f"  {row['system']:<45}  {row['SIG']:>5.3f}"
              f"  {row['BAK']:>5.3f}  {row['OVRL']:>5.3f}")
    print()

# =============================================================================
# AUTO-DISCOVERY OF COMPLETED TRAINING RUNS
# =============================================================================

def find_completed_runs():
    """
    Scan runs/full_train/ for configs that have a best_model.pt file.

    A completed config looks like:
        runs/full_train/lr5em03_samp512/gru_h161/best_model.pt

    Returns list of dicts with run metadata.
    """
    runs = []

    if not os.path.isdir(RUNS_DIR):
        return runs

    for lr_samp_dir in sorted(glob.glob(os.path.join(RUNS_DIR, 'lr*_samp*'))):
        folder_name = os.path.basename(lr_samp_dir)

        # Extract train sample size from folder name e.g. lr5em03_samp512 -> 512
        try:
            train_samples = int(folder_name.split('_samp')[1])
        except (IndexError, ValueError):
            continue

        for model_dir in sorted(glob.glob(os.path.join(lr_samp_dir, '*_h*'))):
            best_model_path = os.path.join(model_dir, 'best_model.pt')

            if not os.path.exists(best_model_path):
                continue   # training not finished yet

            model_name = os.path.basename(model_dir)

            if   model_name.startswith('gru'): model_type = 'gru'
            elif model_name.startswith('crn'): model_type = 'crn'
            else: continue

            try:
                hidden_size = int(model_name.split('_h')[1])
            except (IndexError, ValueError):
                continue

            runs.append({
                'run_name':        f"{folder_name}_{model_name}",
                'model_dir':       model_dir,
                'model_type':      model_type,
                'hidden_size':     hidden_size,
                'train_samples':   train_samples,
                'best_model_path': best_model_path,
            })

    return runs

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_type, hidden_size, checkpoint_path, device):
    """
    Load a trained GRUModel or CRNModel from a checkpoint.

    Strips the DataParallel 'module.' prefix that was added during training
    and puts the model in eval mode on device.

    Args:
        model_type      : 'gru' or 'crn'
        hidden_size     : e.g. 161
        checkpoint_path : path to best_model.pt
        device          : torch.device

    Returns:
        model in eval mode on device
    """
    if model_type == 'gru':
        model = GRUModel(input_size=N_FREQS, hidden_size=hidden_size)
    else:
        model = CRNModel(input_size=N_FREQS, hidden_size=hidden_size)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # block3.py saves weights under the 'state_dict' key (not 'model_state_dict').
    # The checkpoint dict also holds: epoch, val_loss, train_loss, lr,
    # hidden_size, model_type. We read 'state_dict' first, then fall back.
    state = ckpt.get('state_dict',
                     ckpt.get('model_state_dict', ckpt))

    # Strip 'module.' prefix added by DataParallel during training
    state = {k.replace('module.', ''): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model

# =============================================================================
# EVALUATE ONE TRAINED MODEL (BLOCK 4 + BLOCK 5)
# =============================================================================

def evaluate_model(run_info, test_pt_files, device, summary_rows):
    """
    [Block 4 + Block 5] Enhance test set with one model and score with DNSMOS.

    Process:
        For each test sample:
            1. Load noisy features from .pt file
            2. Run model forward pass -> mask  [Block 4]
            3. Apply mask + iSTFT -> enhanced wav  [Block 4]
            4. Save wav as _mic.wav to temp score folder
        Then:
            5. Run DNSMOS on temp score folder  [Block 5]
            6. Delete temp folder (large, not needed after scoring)
            7. Print result and update running summary table

    Only the first WAV_SAVE_COUNT enhanced clips are kept permanently.

    Args:
        run_info      : dict from find_completed_runs()
        test_pt_files : sorted list of .pt file paths
        device        : torch.device
        summary_rows  : list to append result row to (for running table)
    """
    out_dir   = os.path.join(RESULTS_DIR, run_info['run_name'])
    score_dir = os.path.join(out_dir, '_score_wavs')   # temp, deleted after scoring
    save_dir  = os.path.join(out_dir, 'enhanced_wavs') # kept permanently
    csv_path  = os.path.join(out_dir, 'dnsmos_scores.csv')

    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(save_dir,  exist_ok=True)

    print(f"  Loading {run_info['model_type'].upper()} h={run_info['hidden_size']}"
          f"  samp={run_info['train_samples']}")
    model = load_model(
        run_info['model_type'],
        run_info['hidden_size'],
        run_info['best_model_path'],
        device,
    )

    pt_files = test_pt_files[:N_SCORE]
    t_start  = time.time()

    print(f"  Enhancing {len(pt_files)} test samples ...")

    for i, pt_path in enumerate(pt_files):

        # Load saved features produced by block1and2.py
        data        = torch.load(pt_path, map_location='cpu', weights_only=True)
        noisy_feat  = data['noisy_feat']    # (T, 161) float32  log-power
        noisy_mag   = data['noisy_mag']     # (T, 161) float32  linear magnitude
        noisy_phase = data['noisy_phase']   # (T, 161) complex64 phase

        # Block 4: predict mask and reconstruct audio
        enhanced_wav = enhance_one_sample(
            model, noisy_feat, noisy_mag, noisy_phase, device
        )

        # Save as _mic.wav — required filename format for dnsmos_local.py
        base     = os.path.basename(pt_path).replace('.pt', '')
        mic_path = os.path.join(score_dir, f'{base}_mic.wav')
        sf.write(mic_path, enhanced_wav, SAMPLE_RATE)

        # Keep first WAV_SAVE_COUNT clips for listening and report
        if i < WAV_SAVE_COUNT:
            sf.write(os.path.join(save_dir, f'{base}_enhanced.wav'),
                     enhanced_wav, SAMPLE_RATE)

        # Progress log every 200 samples
        if (i + 1) % 200 == 0 or (i + 1) == len(pt_files):
            print(f"    {i+1}/{len(pt_files)} done  |  {time.time()-t_start:.0f}s")

    # Block 5: score with DNSMOS
    print(f"  Enhancement done. Running DNSMOS ...")
    scores = run_dnsmos(score_dir)

    # Clean up large temp folder — scores are saved to csv, wavs not needed
    shutil.rmtree(score_dir, ignore_errors=True)

    if scores:
        save_scores_csv(scores, csv_path)
        m = mean_scores(scores)
        print(f"  DNSMOS  SIG={m['SIG']:.3f}  BAK={m['BAK']:.3f}  OVRL={m['OVRL']:.3f}")
        summary_rows.append({
            'system':        run_info['run_name'],
            'train_samples': run_info['train_samples'],
            'model_type':    run_info['model_type'].upper(),
            **m,
        })
        print_summary_table(summary_rows)
    else:
        print("  WARNING: no DNSMOS scores returned for this run")

# =============================================================================
# NOISY BASELINE
# =============================================================================

def score_noisy_baseline(test_wav_dir, summary_rows):
    """
    [Block 5] Score the unprocessed noisy test audio.

    This gives the floor — how bad the audio sounds before any enhancement.
    All enhanced systems should score higher on BAK and OVRL than this.

    dnsmos_local.py needs *_mic.wav filenames so we copy noisy files into
    a temp folder with the right naming before scoring.

    Args:
        test_wav_dir : directory with sample_XXXX_noisy.wav files
        summary_rows : list to append result row to
    """
    noisy_files = sorted(
        glob.glob(os.path.join(test_wav_dir, '*_noisy.wav'))
    )[:N_SCORE]

    if not noisy_files:
        print(f"  WARNING: no noisy wav files found in {test_wav_dir}")
        return

    out_dir   = os.path.join(RESULTS_DIR, 'noisy_baseline')
    tmp_dir   = os.path.join(out_dir, '_tmp')
    csv_path  = os.path.join(out_dir, 'dnsmos_scores.csv')
    os.makedirs(tmp_dir, exist_ok=True)

    # Copy as _mic.wav for dnsmos_local.py
    for path in noisy_files:
        base = os.path.basename(path).replace('_noisy.wav', '')
        shutil.copy(path, os.path.join(tmp_dir, f'{base}_mic.wav'))

    print(f"  Running DNSMOS on {len(noisy_files)} noisy clips ...")
    scores = run_dnsmos(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if scores:
        save_scores_csv(scores, csv_path)
        m = mean_scores(scores)
        print(f"  Noisy  SIG={m['SIG']:.3f}  BAK={m['BAK']:.3f}  OVRL={m['OVRL']:.3f}")
        summary_rows.append({
            'system': 'Noisy input', 'train_samples': '-', 'model_type': '-', **m
        })
        print_summary_table(summary_rows)

# =============================================================================
# ONNX BASELINE
# =============================================================================

def score_onnx_baseline(test_wav_dir, summary_rows):
    """
    [Block 5] Run the official ONNX baseline and score with DNSMOS.

    The baseline was designed for the full DNS track which includes far-end
    loudspeaker signals (double-talk). Our task is nearend-only — there is
    no loudspeaker and no echo. The baseline needs _mic.wav + _lpb.wav pairs.
    We provide a silent zeros _lpb.wav because there is no far-end signal.
    This is the correct fair comparison for a nearend-only evaluation.

    The baseline outputs 48kHz audio. We resample to 16kHz by taking every
    third sample (48000 / 16000 = 3). Do NOT pass --output_sr to the baseline
    script — it causes a TypeError in the challenge version of enhance.py.

    Args:
        test_wav_dir : directory with sample_XXXX_noisy.wav files
        summary_rows : list to append result row to
    """
    if not os.path.exists(BASELINE_ENHANCE_PY):
        print(f"  WARNING: baseline enhance.py not found — skipping ONNX baseline")
        return
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"  WARNING: ONNX model not found — skipping ONNX baseline")
        return

    noisy_files = sorted(
        glob.glob(os.path.join(test_wav_dir, '*_noisy.wav'))
    )[:N_SCORE]

    out_dir    = os.path.join(RESULTS_DIR, 'onnx_baseline')
    tmp_in     = os.path.join(out_dir, '_tmp_in')
    tmp_out    = os.path.join(out_dir, '_tmp_out')
    score_dir  = os.path.join(out_dir, '_score_wavs')
    save_dir   = os.path.join(out_dir, 'enhanced_wavs')
    csv_path   = os.path.join(out_dir, 'dnsmos_scores.csv')

    for d in [tmp_in, tmp_out, score_dir, save_dir]:
        os.makedirs(d, exist_ok=True)

    # Silent far-end signal — no loudspeaker in nearend-only task
    silent_lpb = np.zeros(N_SAMPLES, dtype=np.float32)

    print(f"  Preparing {len(noisy_files)} input pairs ...")
    print(f"  (silent _lpb.wav — nearend-only, no far-end signal)")
    for path in noisy_files:
        base = os.path.basename(path).replace('_noisy.wav', '')
        shutil.copy(path, os.path.join(tmp_in, f'{base}_mic.wav'))
        sf.write(os.path.join(tmp_in, f'{base}_lpb.wav'), silent_lpb, SAMPLE_RATE)

    # Run baseline — do NOT pass --output_sr (TypeError in challenge version)
    print(f"  Running ONNX baseline enhancement ...")
    result = subprocess.run(
        [sys.executable, BASELINE_ENHANCE_PY,
         '--model_path', ONNX_MODEL_PATH,
         '--noisy_dir',  tmp_in,
         '--enh_dir',    tmp_out],
        capture_output=True, text=True
    )
    shutil.rmtree(tmp_in, ignore_errors=True)

    if result.returncode != 0:
        print(f"  ERROR in baseline enhancement:\n{result.stderr[:400]}")
        return

    # Baseline outputs 48kHz — resample to 16kHz by taking every 3rd sample
    # 48000 / 16000 = 3 exactly so this is lossless for our purposes
    print(f"  Resampling baseline output 48kHz -> 16kHz ...")
    enh_files = sorted(glob.glob(os.path.join(tmp_out, '*.wav')))

    for i, path in enumerate(enh_files):
        wav, sr = sf.read(path)
        wav_16k = wav[::3] if sr == 48000 else wav

        # Trim or pad to exactly N_SAMPLES
        if len(wav_16k) > N_SAMPLES:
            wav_16k = wav_16k[:N_SAMPLES]
        elif len(wav_16k) < N_SAMPLES:
            wav_16k = np.pad(wav_16k, (0, N_SAMPLES - len(wav_16k)))

        wav_16k = wav_16k.astype(np.float32)
        base    = os.path.basename(path)

        sf.write(os.path.join(score_dir, base), wav_16k, SAMPLE_RATE)

        if i < WAV_SAVE_COUNT:
            sf.write(os.path.join(save_dir, base), wav_16k, SAMPLE_RATE)

    shutil.rmtree(tmp_out, ignore_errors=True)

    # Score with DNSMOS
    print(f"  Running DNSMOS on ONNX baseline output ...")
    scores = run_dnsmos(score_dir)
    shutil.rmtree(score_dir, ignore_errors=True)

    if scores:
        save_scores_csv(scores, csv_path)
        m = mean_scores(scores)
        print(f"  ONNX baseline  SIG={m['SIG']:.3f}  BAK={m['BAK']:.3f}  OVRL={m['OVRL']:.3f}")
        summary_rows.append({
            'system': 'ONNX baseline', 'train_samples': '-', 'model_type': '-', **m
        })
        print_summary_table(summary_rows)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    t_total = time.time()

    print("=" * 70)
    print("DNS Project — Block 4 (Reconstruction) + Block 5 (DNSMOS Evaluation)")
    print("=" * 70)
    print(f"Test .pt files  : {TEST_PT_DIR}")
    print(f"Test wav files  : {TEST_WAV_DIR}")
    print(f"Runs dir        : {RUNS_DIR}")
    print(f"Results dir     : {RESULTS_DIR}")
    print(f"Samples to score: {N_SCORE}")
    print(f"Wav files saved : first {WAV_SAVE_COUNT} per model")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i} : {torch.cuda.get_device_name(i)}")
    else:
        print("Device: CPU")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Find test pt files ────────────────────────────────────────────────────
    test_pt_files = sorted(glob.glob(os.path.join(TEST_PT_DIR, 'sample_*.pt')))
    if not test_pt_files:
        print(f"ERROR: no .pt files found in {TEST_PT_DIR}")
        print("Run block1and2.py first to generate the test set.")
        sys.exit(1)
    print(f"Found {len(test_pt_files)} test .pt files")

    # ── Find completed training runs ──────────────────────────────────────────
    runs = find_completed_runs()
    print(f"Found {len(runs)} completed training run(s):")
    for r in runs:
        print(f"  {r['run_name']}")
    if not runs:
        print("No completed runs found. Train models with block3.py first.")
        sys.exit(1)
    print()

    # Collect all results for running summary and final csv
    summary_rows = []

    # ── Step 1: noisy baseline ────────────────────────────────────────────────
    print("=" * 70)
    print("Step 1: Scoring noisy input (floor)")
    print("=" * 70)
    score_noisy_baseline(TEST_WAV_DIR, summary_rows)

    # ── Step 2: ONNX baseline ─────────────────────────────────────────────────
    print("=" * 70)
    print("Step 2: Scoring ONNX baseline")
    print("  (nearend-only: silent _lpb, output resampled 48kHz -> 16kHz)")
    print("=" * 70)
    score_onnx_baseline(TEST_WAV_DIR, summary_rows)

    # ── Steps 3+: each trained model ─────────────────────────────────────────
    for i, run_info in enumerate(runs):
        print("=" * 70)
        print(f"Step {i+3}: Evaluating {run_info['run_name']}")
        print("=" * 70)
        evaluate_model(run_info, test_pt_files, device, summary_rows)

    # ── Final summary ─────────────────────────────────────────────────────────
    summary_csv = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['system', 'train_samples', 'model_type',
                           'SIG', 'BAK', 'OVRL']
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'System':<45}  {'SIG':>5}  {'BAK':>5}  {'OVRL':>5}")
    print("  " + "-" * 63)
    for row in summary_rows:
        print(f"  {row['system']:<45}  {row['SIG']:>5.3f}"
              f"  {row['BAK']:>5.3f}  {row['OVRL']:>5.3f}")
    print()
    print(f"Summary saved to : {summary_csv}")
    print()

    elapsed = time.time() - t_total
    print(f"Total time : {elapsed:.0f}s  ({elapsed/3600:.1f}h)")
    print("=" * 70)
