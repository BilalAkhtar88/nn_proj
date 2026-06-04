"""
DNS Project — Block 1 (Data Loader) + Block 2 (Preprocessing)
==============================================================

Block 1 — Data Loader:
    - Speaker-level 80/10/10 split across all speech datasets
    - Creates symlinks in data_splits/ (no audio copying)
    - Writes config_train.yaml, config_val.yaml, config_test.yaml to configs/
    - DNSDataset class: callable by block3.py for online train generation
    - EvalDataset class: loads pre-saved .pt files for val/test

Block 2 — Preprocessing (inside DNSDataset and generate_and_save):
    - Raw waveform -> STFT -> log-power spectrogram (model input)
    - Ideal Ratio Mask (IRM) computation (training target)
    - noisy_mag clamped to 1e-4 to prevent division by zero on silent frames
    - Preprocessing happens inside __getitem__ (train) and generate_and_save
      (val/test) — standard PyTorch pattern: raw data in, tensors out

Why combined in one file:
    Block 2 preprocessing cannot be separated from Block 1 loading —
    every audio sample must be preprocessed immediately after synthesis.
    Keeping them together avoids passing raw audio between files and
    ensures the STFT constants are defined exactly once.

Folders created by this script:
    ~/nn_proj/data_splits/                   <- speaker symlinks per dataset
    ~/nn_proj/configs/                       <- config_train/val/test.yaml
    ~/nn_proj/eval_dataset/val/pt_files/     <- 2000 preprocessed .pt tensors
    ~/nn_proj/eval_dataset/val/wav_files/    <- 200 noisy+clean .wav pairs
    ~/nn_proj/eval_dataset/test/pt_files/    <- 2000 preprocessed .pt tensors
    ~/nn_proj/eval_dataset/test/wav_files/   <- 200 noisy+clean .wav pairs
    ~/nn_proj/logs/                          <- sbatch log files

Reusable by block3.py:
    from block1and2 import DNSDataset, EvalDataset
    from block1and2 import CONFIGS_DIR, EVAL_DIR
    from block1and2 import SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, DFT_SIZE, N_FREQS

Run via sbatch:
    sbatch block1and2.sbatch
"""

import os
import glob
import random
import time
import yaml
import numpy as np
import soundfile as sf
import builtins

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from synthesizer import Synthesizer

# =============================================================================
# FORCE UNBUFFERED OUTPUT — logs appear immediately in sbatch log file
# =============================================================================

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _original_print(*args, **kwargs)
builtins.print = print

# =============================================================================
# PATHS
# =============================================================================

DATASET_ROOT = '/gpfs/helios/projects/neuralnet_course/track02/dataset'
SPEECH_DIR   = os.path.join(DATASET_ROOT, 'clean_speech')
NOISE_DIR    = os.path.join(DATASET_ROOT, 'noise')

PROJECT_DIR  = os.path.expanduser('~/nn_proj')
SPLIT_BASE   = os.path.join(PROJECT_DIR, 'data_splits')
CONFIGS_DIR  = os.path.join(PROJECT_DIR, 'configs')
EVAL_DIR     = os.path.join(PROJECT_DIR, 'eval_dataset')
LOGS_DIR     = os.path.join(PROJECT_DIR, 'logs')

SPEECH_DATASETS = {
    'emotional_speech': os.path.join(SPEECH_DIR, 'emotional_speech'),
    'read_speech':      os.path.join(SPEECH_DIR, 'read_speech'),
    'vocalset':         os.path.join(SPEECH_DIR, 'VocalSet_48kHz_mono'),
}

# =============================================================================
# BLOCK 2 — STFT CONSTANTS
# Must match baseline/enhance.py exactly. Do not change.
# =============================================================================

SAMPLE_RATE  = 16_000
DURATION_SEC = 15
N_SAMPLES    = SAMPLE_RATE * DURATION_SEC      # 240,000 samples
FRAME_SIZE   = int(0.02 * SAMPLE_RATE)         # 320 samples = 20ms per frame
HOP_SIZE     = int(0.02 * SAMPLE_RATE * 0.5)  # 160 samples = 10ms hop
DFT_SIZE     = 320
N_FREQS      = DFT_SIZE // 2 + 1              # 161 frequency bins
N_FRAMES     = (N_SAMPLES + HOP_SIZE - FRAME_SIZE) // HOP_SIZE + 1  # 1500 frames

# =============================================================================
# BLOCK 1 — DATASET SIZE CONSTANTS
# =============================================================================

VAL_SAMPLES    = 2000   # generated once, saved to eval_dataset/val/
TEST_SAMPLES   = 2000   # generated once, saved to eval_dataset/test/
WAV_SAVE_COUNT = 200    # only save wav files for first N samples

# =============================================================================
# BLOCK 1 — SPLIT CONSTANTS
# =============================================================================

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
SPLIT_SEED  = 42   # fixed seed — ensures same split every run

# =============================================================================
# BLOCK 1 — DATALOADER CONSTANTS (imported by block3.py)
# =============================================================================

BATCH_SIZE  = 16
NUM_WORKERS = 0   # Synthesizer is not multiprocessing-safe

# =============================================================================
# BLOCK 1 — SPEAKER ID EXTRACTION
# =============================================================================

def get_speaker_id(filename, dataset_name):
    """
    [Block 1] Extract a unique speaker ID from a filename.

    Each dataset uses a different naming convention:
        emotional_speech : first underscore-separated token
                           e.g. p001_sentence01_normal.wav -> p001
        read_speech      : token after the word 'reader'
                           e.g. arctic_a0001_reader7.wav  -> 7
        vocalset         : second token
                           e.g. singer_01_scale.wav       -> 01

    Used by split_speakers() to group files before the 80/10/10 split.
    Returns a string speaker ID.
    """
    parts = filename.split('_')
    if dataset_name == 'emotional_speech':
        return parts[0]
    elif dataset_name == 'read_speech':
        if 'reader' in parts:
            idx = parts.index('reader')
            return parts[idx + 1]
        return parts[0]
    elif dataset_name == 'vocalset':
        return parts[1] if len(parts) > 1 else parts[0]
    else:
        return parts[0]

# =============================================================================
# BLOCK 1 — SPEAKER SPLIT
# =============================================================================

def split_speakers(
    speech_datasets = SPEECH_DATASETS,
    split_base      = SPLIT_BASE,
    train_ratio     = TRAIN_RATIO,
    val_ratio       = VAL_RATIO,
    seed            = SPLIT_SEED,
):
    """
    [Block 1] Split speakers 80/10/10 into train/val/test groups.

    Why speaker-level split:
        If speaker X appears in both train and val, the model is evaluated
        on a voice it heard during training — data leakage. Speaker-level
        split guarantees zero voice overlap between splits.

    Uses symlinks — no audio files are copied. Fast and space-efficient.
    Idempotent — skips silently if split_base already exists.

    Returns:
        dict: {dataset_name: {split_name: [list of symlink paths]}}
    """
    if os.path.exists(split_base):
        print(f"[Block 1] Split already exists at {split_base} — skipping.")
        result = {}
        for dataset_name in speech_datasets:
            result[dataset_name] = {}
            for split_name in ['train', 'val', 'test']:
                files = sorted(glob.glob(
                    os.path.join(split_base, dataset_name, split_name, '*.wav')
                ))
                result[dataset_name][split_name] = files
                print(f"  {dataset_name}/{split_name}: {len(files)} files")
        return result

    print("[Block 1] Running speaker-level split (80/10/10)...")
    print()
    rng    = random.Random(seed)
    result = {}

    for dataset_name, folder in speech_datasets.items():
        print(f"  Dataset: {dataset_name}  ({folder})")

        all_files = sorted(glob.glob(os.path.join(folder, '*.wav')))
        if len(all_files) == 0:
            print(f"    WARNING: no .wav files found — skipping")
            continue

        # group files by speaker ID
        speaker_to_files = {}
        for filepath in all_files:
            spk = get_speaker_id(os.path.basename(filepath), dataset_name)
            speaker_to_files.setdefault(spk, []).append(filepath)

        all_speakers = sorted(speaker_to_files.keys())
        rng.shuffle(all_speakers)
        n = len(all_speakers)

        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        if n_train + n_val >= n:   # guard: test gets at least 1 speaker
            n_train = max(1, n - 2)
            n_val   = 1

        groups = {
            'train': all_speakers[:n_train],
            'val':   all_speakers[n_train : n_train + n_val],
            'test':  all_speakers[n_train + n_val:],
        }

        result[dataset_name] = {}
        for split_name, speakers in groups.items():
            split_dir = os.path.join(split_base, dataset_name, split_name)
            os.makedirs(split_dir, exist_ok=True)

            files_in_split = []
            for spk in speakers:
                for filepath in speaker_to_files[spk]:
                    link = os.path.join(split_dir, os.path.basename(filepath))
                    if not os.path.exists(link):
                        os.symlink(filepath, link)
                    files_in_split.append(link)

            result[dataset_name][split_name] = files_in_split
            print(f"    {split_name:5s}: {len(speakers):3d} speakers  "
                  f"{len(files_in_split):4d} files  ->  {split_dir}")

        print()

    print("[Block 1] Speaker split complete.")
    return result

# =============================================================================
# BLOCK 1 — YAML CONFIG FILES
# =============================================================================

def create_configs(split_info, noise_dir=NOISE_DIR,
                   split_base=SPLIT_BASE, configs_dir=CONFIGS_DIR):
    """
    [Block 1] Write config_train.yaml, config_val.yaml, config_test.yaml.

    Loads the original synthesizer_config.yaml and only swaps the dataset
    directory paths — all other synthesizer parameters are preserved exactly
    as the instructor wrote them.

    Configs are written to configs_dir (~/nn_proj/configs/).
    Skips writing a config file if it already exists (idempotent).
    """
    os.makedirs(configs_dir, exist_ok=True)

    base_cfg_path = os.path.join(PROJECT_DIR, 'synthesizer_config.yaml')
    if not os.path.exists(base_cfg_path):
        base_cfg_path = 'synthesizer_config.yaml'   # fallback to cwd

    with open(base_cfg_path) as f:
        base = yaml.load(f, Loader=yaml.FullLoader)

    # noise directory is the same for all splits
    base['onlinesynth_nearend_noises'] = {
        'noise_dataset_1': {'weight': 1.0, 'dir': noise_dir}
    }

    for split_name in ['train', 'val', 'test']:
        out_path = os.path.join(configs_dir, f'config_{split_name}.yaml')

        if os.path.exists(out_path):
            print(f"[Block 1] Config already exists — skipping: {out_path}")
            continue

        cfg = dict(base)
        cfg['onlinesynth_nearend_datasets'] = {}

        for dataset_name in split_info:
            split_dir = os.path.join(split_base, dataset_name, split_name)
            if os.path.isdir(split_dir):
                cfg['onlinesynth_nearend_datasets'][dataset_name] = {
                    'weight': 1.0,
                    'dir':    split_dir,
                }

        with open(out_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"[Block 1] Written: {out_path}")

    print("[Block 1] All configs ready.")

# =============================================================================
# BLOCK 1 + BLOCK 2 — DNSDataset
# Reusable by block3.py for online train generation each epoch
# =============================================================================

class DNSDataset(Dataset):
    """
    [Block 1 + Block 2] PyTorch Dataset for online train data generation.

    Block 1 — Data loading:
        Calls Synthesizer.generate() to produce one random noisy+clean
        audio pair drawn from the speaker split for this config.

    Block 2 — Preprocessing:
        Converts raw waveforms to log-power STFT features (model input)
        and computes the Ideal Ratio Mask (IRM) training target.

    Why combined:
        Preprocessing must happen immediately after synthesis — there is
        no intermediate format. Keeping them together in __getitem__ is
        the standard PyTorch pattern: raw data in, tensors out.

    Reuse in block3.py for the training DataLoader:
        from block1and2 import DNSDataset, CONFIGS_DIR
        train_ds = DNSDataset(
            cfg_path    = os.path.join(CONFIGS_DIR, 'config_train.yaml'),
            num_samples = 2000,
        )
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                  num_workers=0)

    For val/test: use EvalDataset (loads pre-saved .pt files) — faster.

    Args:
        cfg_path    : path to yaml config (config_train/val/test.yaml)
        num_samples : dataset length — how many samples per epoch

    Returns per __getitem__ (shapes assuming 15s audio at 16kHz):
        noisy_feat  : (1500, 161) float32 — log-power spectrogram, model input
        ideal_mask  : (1500, 161) float32 — IRM target in [0, 1]
        noisy_mag   : (1500, 161) float32 — linear magnitude, Block 4 reconstruction
        noisy_phase : (1500, 161) complex64 — phase, Block 4 reconstruction
    """

    def __init__(self, cfg_path, num_samples):
        self.cfg_path    = cfg_path
        self.num_samples = num_samples
        # [Block 2] Hanning window — must match baseline/enhance.py
        w = np.sqrt(np.hanning(FRAME_SIZE + 1)[:-1]).astype(np.float32)
        self.window = torch.from_numpy(w)

    def __len__(self):
        return self.num_samples

    def _compute_stft(self, wav_np):
        """
        [Block 2] Convert raw waveform to log-power spectrogram.

        Matches baseline/enhance.py exactly:
            - sqrt Hanning window (same as enhance.py line 22)
            - left-pad by one hop to align frames
            - rfft with n=DFT_SIZE=320 -> 161 bins
            - feature = log10(mag^2) / 20

        Args:
            wav_np : (240000,) numpy float32 waveform at 16kHz
        Returns:
            logpow : (1500, 161) log-power features — model input
            mag    : (1500, 161) linear magnitude   — reconstruction
            phase  : (1500, 161) complex phase      — reconstruction
        """
        wav    = torch.from_numpy(wav_np)
        wav    = F.pad(wav.unsqueeze(0), (HOP_SIZE, 0)).squeeze(0)
        frames = wav.unfold(0, FRAME_SIZE, HOP_SIZE)
        frames = frames * self.window
        cspec  = torch.fft.rfft(frames, n=DFT_SIZE, dim=-1)
        mag    = cspec.abs()
        phase  = cspec / (mag + 1e-12)
        logpow = torch.log10(torch.clamp(mag ** 2, min=1e-12)) / 20.0
        return logpow, mag, phase

    def __getitem__(self, idx):
        """
        [Block 1] Synthesize one audio clip.
        [Block 2] Preprocess to features and IRM target.

        Block 1 — synthesis:
            Synthesizer.generate() draws a random clean speech file from
            the speaker split for this config, mixes with random noise at
            a random SNR, applies mic distortions -> returns noisy + clean.

        Block 2 — preprocessing:
            Step 1: STFT both waveforms -> log-power features
            Step 2: IRM = clamp(clean_mag / noisy_mag, 0, 1)
                    noisy_mag clamped to 1e-4 before division:
                    silent frames have near-zero noisy_mag ->
                    division gives inf -> nan gradients. Fix: min clamp.

        Note: Synthesizer created fresh each call — not thread-safe.
        NUM_WORKERS must be 0 in the DataLoader.
        """
        # ── Block 1: synthesize ───────────────────────────────────────────────
        synthesizer = Synthesizer(self.cfg_path)
        data        = synthesizer.generate()
        noisy_np    = data['mic'].astype(np.float32)
        clean_np    = data['target'].astype(np.float32)

        # ── Block 2: STFT preprocessing ───────────────────────────────────────
        noisy_feat, noisy_mag, noisy_phase = self._compute_stft(noisy_np)
        clean_feat, _,         _           = self._compute_stft(clean_np)

        # ── Block 2: IRM target ───────────────────────────────────────────────
        noisy_mag_raw = torch.sqrt(torch.pow(10.0, noisy_feat * 20.0))
        clean_mag_raw = torch.sqrt(torch.pow(10.0, clean_feat * 20.0))
        noisy_mag_raw = torch.clamp(noisy_mag_raw, min=1e-4)  # prevent div/0
        ideal_mask    = torch.clamp(clean_mag_raw / noisy_mag_raw, 0.0, 1.0)

        return (
            noisy_feat,
            ideal_mask,
            noisy_mag,
            noisy_phase.to(torch.complex64),
        )

# =============================================================================
# BLOCK 1 — EvalDataset
# Loads pre-saved .pt files for val/test — faster than re-synthesizing
# =============================================================================

class EvalDataset(Dataset):
    """
    [Block 1] Loads pre-saved .pt tensor files for val/test evaluation.

    Use in block3.py for the val/test DataLoader:
        from block1and2 import EvalDataset, EVAL_DIR
        val_ds  = EvalDataset(os.path.join(EVAL_DIR, 'val',  'pt_files'))
        test_ds = EvalDataset(os.path.join(EVAL_DIR, 'test', 'pt_files'))
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                                num_workers=4)

    Each .pt file contains a dict with keys:
        noisy_feat  : (1500, 161) float32
        ideal_mask  : (1500, 161) float32
        noisy_mag   : (1500, 161) float32
        noisy_phase : (1500, 161) complex64

    Args:
        pt_dir : folder containing sample_XXXX.pt files
    """

    def __init__(self, pt_dir):
        self.pt_dir = pt_dir
        self.files  = sorted(glob.glob(os.path.join(pt_dir, 'sample_*.pt')))
        if len(self.files) == 0:
            raise RuntimeError(f"[Block 1] No .pt files found in {pt_dir}")
        print(f"[Block 1] EvalDataset: {len(self.files)} samples from {pt_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu',
                          weights_only=True)
        return (
            data['noisy_feat'],
            data['ideal_mask'],
            data['noisy_mag'],
            data['noisy_phase'],
        )

# =============================================================================
# BLOCK 1 + BLOCK 2 — generate_and_save
# Generates val/test samples once and saves to disk as .pt + .wav
# =============================================================================

def generate_and_save(cfg_path, out_dir, num_samples, split_name,
                      wav_save_count=WAV_SAVE_COUNT):
    """
    [Block 1 + Block 2] Generate num_samples and save to disk.

    Block 1 — synthesis: calls Synthesizer.generate() for each sample
    Block 2 — preprocessing: computes STFT features and IRM target

    Saves:
        out_dir/pt_files/sample_XXXX.pt         — all num_samples as tensors
        out_dir/wav_files/sample_XXXX_noisy.wav — first wav_save_count only
        out_dir/wav_files/sample_XXXX_clean.wav — first wav_save_count only

    Each .pt file is a dict:
        {
            'noisy_feat'  : (1500, 161) float32  — [Block 2] log-power features
            'ideal_mask'  : (1500, 161) float32  — [Block 2] IRM target
            'noisy_mag'   : (1500, 161) float32  — [Block 2] for reconstruction
            'noisy_phase' : (1500, 161) complex64 — [Block 2] for reconstruction
        }

    Idempotent — skips already-existing .pt files so job is restartable.

    Args:
        cfg_path       : yaml config for this split
        out_dir        : e.g. ~/nn_proj/eval_dataset/val
        num_samples    : how many samples to generate
        split_name     : 'val' or 'test' — for progress logging
        wav_save_count : how many wav pairs to save (default 200)
    """
    pt_dir  = os.path.join(out_dir, 'pt_files')
    wav_dir = os.path.join(out_dir, 'wav_files')
    os.makedirs(pt_dir,  exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    existing  = sorted(glob.glob(os.path.join(pt_dir, 'sample_*.pt')))
    start_idx = len(existing)

    if start_idx >= num_samples:
        print(f"[Block 1] [{split_name}] All {num_samples} samples already "
              f"saved — skipping.")
        return

    if start_idx > 0:
        print(f"[Block 1] [{split_name}] Resuming from sample {start_idx} "
              f"({num_samples - start_idx} remaining)...")
    else:
        print(f"[Block 1] [{split_name}] Generating {num_samples} samples...")
        print(f"  pt_files  -> {pt_dir}")
        print(f"  wav_files -> {wav_dir}  (first {wav_save_count} only)")

    print()

    # [Block 2] Hanning window for STFT
    window_np = np.sqrt(np.hanning(FRAME_SIZE + 1)[:-1]).astype(np.float32)
    window    = torch.from_numpy(window_np)

    def compute_stft(wav_np):
        """[Block 2] Same STFT as DNSDataset._compute_stft."""
        wav    = torch.from_numpy(wav_np)
        wav    = F.pad(wav.unsqueeze(0), (HOP_SIZE, 0)).squeeze(0)
        frames = wav.unfold(0, FRAME_SIZE, HOP_SIZE)
        frames = frames * window
        cspec  = torch.fft.rfft(frames, n=DFT_SIZE, dim=-1)
        mag    = cspec.abs()
        phase  = cspec / (mag + 1e-12)
        logpow = torch.log10(torch.clamp(mag ** 2, min=1e-12)) / 20.0
        return logpow, mag, phase

    t_start = time.time()

    for i in range(start_idx, num_samples):

        # ── Block 1: synthesize ───────────────────────────────────────────────
        synthesizer = Synthesizer(cfg_path)
        data        = synthesizer.generate()
        noisy_np    = data['mic'].astype(np.float32)
        clean_np    = data['target'].astype(np.float32)

        # ── Block 2: STFT preprocessing ───────────────────────────────────────
        noisy_feat, noisy_mag, noisy_phase = compute_stft(noisy_np)
        clean_feat, _,         _           = compute_stft(clean_np)

        # ── Block 2: IRM target ───────────────────────────────────────────────
        noisy_mag_raw = torch.sqrt(torch.pow(10.0, noisy_feat * 20.0))
        clean_mag_raw = torch.sqrt(torch.pow(10.0, clean_feat * 20.0))
        noisy_mag_raw = torch.clamp(noisy_mag_raw, min=1e-4)
        ideal_mask    = torch.clamp(clean_mag_raw / noisy_mag_raw, 0.0, 1.0)

        # ── Block 1: save .pt ─────────────────────────────────────────────────
        pt_path = os.path.join(pt_dir, f'sample_{i:04d}.pt')
        torch.save({
            'noisy_feat':  noisy_feat.to(torch.float32),
            'ideal_mask':  ideal_mask.to(torch.float32),
            'noisy_mag':   noisy_mag.to(torch.float32),
            'noisy_phase': noisy_phase.to(torch.complex64),
        }, pt_path)

        # ── Block 1: save wav (first wav_save_count only) ─────────────────────
        if i < wav_save_count:
            sf.write(
                os.path.join(wav_dir, f'sample_{i:04d}_noisy.wav'),
                noisy_np, SAMPLE_RATE
            )
            sf.write(
                os.path.join(wav_dir, f'sample_{i:04d}_clean.wav'),
                clean_np, SAMPLE_RATE
            )

        # ── progress log every 100 samples ───────────────────────────────────
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            elapsed    = time.time() - t_start
            per_sample = elapsed / (i - start_idx + 1)
            remaining  = per_sample * (num_samples - i - 1)
            print(f"  [{split_name}] {i+1:4d}/{num_samples} samples saved"
                  f"  |  elapsed: {elapsed:.0f}s"
                  f"  |  ETA: {remaining:.0f}s"
                  f"  |  {per_sample:.2f}s/sample")

    total     = time.time() - t_start
    pt_count  = len(glob.glob(os.path.join(pt_dir,  'sample_*.pt')))
    wav_count = len(glob.glob(os.path.join(wav_dir, '*_noisy.wav')))

    print()
    print(f"[Block 1+2] [{split_name}] Done.")
    print(f"  .pt files saved : {pt_count}   ->  {pt_dir}")
    print(f"  .wav pairs saved: {wav_count}  ->  {wav_dir}")
    print(f"  Total time      : {total:.0f}s  ({total/60:.1f} min)")
    print()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    t_total = time.time()

    print("=" * 70)
    print("DNS Project — Block 1 (Data Loader) + Block 2 (Preprocessing)")
    print("=" * 70)
    print(f"Project dir  : {PROJECT_DIR}")
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Split base   : {SPLIT_BASE}")
    print(f"Configs dir  : {CONFIGS_DIR}")
    print(f"Eval dir     : {EVAL_DIR}")
    print(f"Val samples  : {VAL_SAMPLES}")
    print(f"Test samples : {TEST_SAMPLES}")
    print(f"Wav saved    : first {WAV_SAVE_COUNT} per split")
    print(f"Split seed   : {SPLIT_SEED}")
    print()

    # ── Step 1: speaker split [Block 1] ───────────────────────────────────────
    print("=" * 70)
    print("Step 1 [Block 1]: Speaker-level split (80/10/10)")
    print("=" * 70)
    split_info = split_speakers()
    print()

    # ── Step 2: yaml configs [Block 1] ────────────────────────────────────────
    print("=" * 70)
    print("Step 2 [Block 1]: Writing yaml configs to configs/")
    print("=" * 70)
    create_configs(split_info)
    print()

    # ── Step 3: generate and save val samples [Block 1 + Block 2] ─────────────
    print("=" * 70)
    print("Step 3 [Block 1 + Block 2]: Generating val samples")
    print("=" * 70)
    val_cfg = os.path.join(CONFIGS_DIR, 'config_val.yaml')
    val_dir = os.path.join(EVAL_DIR, 'val')
    generate_and_save(val_cfg, val_dir, VAL_SAMPLES, split_name='val')

    # ── Step 4: generate and save test samples [Block 1 + Block 2] ────────────
    print("=" * 70)
    print("Step 4 [Block 1 + Block 2]: Generating test samples")
    print("=" * 70)
    test_cfg = os.path.join(CONFIGS_DIR, 'config_test.yaml')
    test_dir = os.path.join(EVAL_DIR, 'test')
    generate_and_save(test_cfg, test_dir, TEST_SAMPLES, split_name='test')

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("=" * 70)
    print("Block 1 + Block 2 complete.")
    print("=" * 70)
    print(f"Total time : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print()
    print("Folders created:")
    print(f"  {SPLIT_BASE}/")
    print(f"  {CONFIGS_DIR}/")
    print(f"  {os.path.join(EVAL_DIR, 'val',  'pt_files')}/")
    print(f"  {os.path.join(EVAL_DIR, 'val',  'wav_files')}/")
    print(f"  {os.path.join(EVAL_DIR, 'test', 'pt_files')}/")
    print(f"  {os.path.join(EVAL_DIR, 'test', 'wav_files')}/")
    print()
    print("Reuse in block3.py:")
    print("  from block1and2 import DNSDataset, EvalDataset")
    print("  from block1and2 import CONFIGS_DIR, EVAL_DIR")
    print()
    print("Next step: submit block3.sbatch to start training.")
