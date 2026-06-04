"""
DNS Project — Block 4 + Block 5
================================
Enhance the fixed test set produced by block1and2.py, compare against the
instructor ONNX baseline, and score everything with dnsmos_local.py.

This script is intentionally standalone: it does not import block1and2.py or
block3.py because those files contain top-level path setup.  The constants and
model definitions below match those files, while all directories come from the
current project directory or sbatch environment variables.
"""

from __future__ import annotations

import builtins
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Unbuffered prints for Slurm logs
# -----------------------------------------------------------------------------
_orig_print = builtins.print

def print(*args, **kwargs):  # noqa: A001 - deliberate replacement
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

builtins.print = print

# -----------------------------------------------------------------------------
# Paths.  The sbatch exports these.  Defaults also work when running manually
# from PROJECT_DIR.
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", os.getcwd())).resolve()
EVAL_DIR = Path(os.environ.get("EVAL_DIR", PROJECT_DIR / "eval_dataset")).resolve()
TEST_PT_DIR = Path(os.environ.get("TEST_PT_DIR", EVAL_DIR / "test" / "pt_files")).resolve()
TEST_WAV_DIR = Path(os.environ.get("TEST_WAV_DIR", EVAL_DIR / "test" / "wav_files")).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", PROJECT_DIR / "runs" / "full_train")).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_DIR / "results")).resolve()

# Instructor source tree.  In your current setup this is usually PROJECT_DIR,
# because baseline/, DNSMOS/, dnsmos_local.py are already there.  If they are in
# a separate unzipped nnp_track02-master folder, set CHALLENGE_DIR in sbatch.

def _find_challenge_dir() -> Path:
    candidates = []
    if os.environ.get("CHALLENGE_DIR"):
        candidates.append(Path(os.environ["CHALLENGE_DIR"]))
    candidates += [
        PROJECT_DIR,
        PROJECT_DIR / "nnp_track02-master",
        PROJECT_DIR.parent / "nnp_track02-master",
    ]
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if (cand / "baseline" / "enhance.py").exists() and (cand / "dnsmos_local.py").exists():
            return cand
    # Return the first candidate so the error message points somewhere useful.
    return candidates[0].expanduser().resolve() if candidates else PROJECT_DIR

CHALLENGE_DIR = _find_challenge_dir()
BASELINE_ENHANCE_PY = CHALLENGE_DIR / "baseline" / "enhance.py"
DNSMOS_LOCAL_PY = CHALLENGE_DIR / "dnsmos_local.py"
DNSMOS_DIR = CHALLENGE_DIR / "DNSMOS"


def _find_onnx_model() -> Optional[Path]:
    if os.environ.get("ONNX_MODEL_PATH"):
        p = Path(os.environ["ONNX_MODEL_PATH"]).expanduser().resolve()
        return p if p.exists() else None

    baseline_dir = CHALLENGE_DIR / "baseline"
    preferred = baseline_dir / "dec-baseline-model-icassp2022.onnx"
    if preferred.exists():
        return preferred

    # The zip you uploaded contains: "dec-baseline-model-icassp2022 (1).onnx".
    matches = sorted(baseline_dir.glob("dec-baseline-model-icassp2022*.onnx"))
    if matches:
        return matches[0]

    matches = sorted(baseline_dir.glob("*.onnx"))
    return matches[0] if matches else None

ONNX_MODEL_PATH = _find_onnx_model()

# -----------------------------------------------------------------------------
# Constants matching block1and2.py / block3.py
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16_000
DURATION_SEC = 15
N_SAMPLES = SAMPLE_RATE * DURATION_SEC
FRAME_SIZE = int(0.02 * SAMPLE_RATE)       # 320
HOP_SIZE = int(0.02 * SAMPLE_RATE * 0.5)   # 160
DFT_SIZE = 320
N_FREQS = DFT_SIZE // 2 + 1                # 161
INPUT_SIZE = 161
OUTPUT_SIZE = 161

N_SCORE = int(os.environ.get("N_SCORE", "2000"))
WAV_SAVE_COUNT = int(os.environ.get("WAV_SAVE_COUNT", "20"))
SKIP_ONNX_BASELINE = os.environ.get("SKIP_ONNX_BASELINE", "0") == "1"
SKIP_NOISY_BASELINE = os.environ.get("SKIP_NOISY_BASELINE", "0") == "1"

# -----------------------------------------------------------------------------
# Model classes copied from block3.py so this script has no import side effects.
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 161,
                 output_size: int = OUTPUT_SIZE):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.proj = nn.Linear(input_size, hidden_size, bias=False) if input_size != hidden_size else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1 + self.proj(x))
        return torch.clamp(torch.sigmoid(self.fc(out2)), 0.0, 1.0)


class CRNModel(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 161,
                 output_size: int = OUTPUT_SIZE):
        super().__init__()
        h = hidden_size
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_size, h * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 4),
            nn.ELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(h * 4, h * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 2),
            nn.ELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(h * 2, h, kernel_size=5, padding=2),
            nn.BatchNorm1d(h),
            nn.ELU(),
        )
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(h, h * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 2),
            nn.ELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(h * 2, h * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 4),
            nn.ELU(),
        )
        self.dec3 = nn.ConvTranspose1d(h * 4, output_size, kernel_size=5, padding=2)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        enc1 = self.enc1(x_t)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        gru_in = enc3.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)
        bottleneck = gru_out.transpose(1, 2) + enc3
        dec1 = self.dec1(bottleneck) + enc2
        dec2 = self.dec2(dec1) + enc1
        out = self.dec3(dec2).transpose(1, 2)
        return torch.clamp(torch.sigmoid(out), 0.0, 1.0)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_torch_load(path: Path, map_location="cpu"):
    """Use weights_only when available, but keep compatibility with older PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def clean_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def read_pt_sample(pt_path: Path) -> Dict[str, torch.Tensor]:
    data = safe_torch_load(pt_path, map_location="cpu")
    needed = ["noisy_feat", "noisy_mag", "noisy_phase"]
    missing = [k for k in needed if k not in data]
    if missing:
        raise KeyError(f"{pt_path} is missing keys: {missing}")
    return data

# Precompute the synthesis window once on CPU.
SYNTH_WINDOW = torch.from_numpy(np.sqrt(np.hanning(FRAME_SIZE + 1)[:-1]).astype(np.float32))


def reconstruct_wav(noisy_mag: torch.Tensor, noisy_phase: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    target_len: int = N_SAMPLES) -> np.ndarray:
    """Inverse of block1and2.py STFT.  mask=None reconstructs the noisy mic."""
    noisy_mag = noisy_mag.detach().cpu().to(torch.float32)
    noisy_phase = noisy_phase.detach().cpu().to(torch.complex64)
    if mask is None:
        enhanced_mag = noisy_mag
    else:
        enhanced_mag = noisy_mag * mask.detach().cpu().to(torch.float32)

    enhanced_spec = enhanced_mag * noisy_phase
    frames = torch.fft.irfft(enhanced_spec, n=DFT_SIZE, dim=-1)
    frames = frames * SYNTH_WINDOW

    n_frames = frames.shape[0]
    wav_len_with_left_pad = FRAME_SIZE + (n_frames - 1) * HOP_SIZE
    wav = torch.zeros(wav_len_with_left_pad, dtype=torch.float32)
    for t in range(n_frames):
        start = t * HOP_SIZE
        wav[start:start + FRAME_SIZE] += frames[t]

    # Remove the one-hop left padding used in block1and2.py.
    wav = wav[HOP_SIZE:HOP_SIZE + target_len]
    if wav.numel() < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
    elif wav.numel() > target_len:
        wav = wav[:target_len]

    wav = torch.clamp(wav, -1.0, 1.0)
    return wav.numpy().astype(np.float32)


def enhance_one_sample(model: nn.Module, data: Dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    x = data["noisy_feat"].unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(x).squeeze(0).cpu()
    return reconstruct_wav(data["noisy_mag"], data["noisy_phase"], mask=mask)


def write_reconstructed_noisy_wavs(pt_files: Iterable[Path], out_dir: Path,
                                   save_copy_dir: Optional[Path] = None) -> int:
    """Write *_mic.wav files reconstructed from block1and2 .pt files."""
    clean_dir(out_dir)
    if save_copy_dir is not None:
        save_copy_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, pt_path in enumerate(pt_files):
        data = read_pt_sample(pt_path)
        wav = reconstruct_wav(data["noisy_mag"], data["noisy_phase"], mask=None)
        base = pt_path.stem
        sf.write(out_dir / f"{base}_mic.wav", wav, SAMPLE_RATE)
        if save_copy_dir is not None and i < WAV_SAVE_COUNT:
            sf.write(save_copy_dir / f"{base}_noisy.wav", wav, SAMPLE_RATE)
        count += 1
        if (i + 1) % 200 == 0:
            print(f"    reconstructed noisy {i+1} files")
    return count


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None, label: str = "command") -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.stdout:
        print(f"  {label} stdout:\n{result.stdout[-1500:]}")
    if result.returncode != 0:
        if result.stderr:
            print(f"  {label} stderr:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    if result.stderr:
        # tqdm often writes progress to stderr; show only the tail.
        print(f"  {label} stderr:\n{result.stderr[-1500:]}")
    return result

# -----------------------------------------------------------------------------
# DNSMOS
# -----------------------------------------------------------------------------
def run_dnsmos(wav_dir: Path, csv_path: Path) -> List[Dict[str, float]]:
    require_file(DNSMOS_LOCAL_PY, "dnsmos_local.py")
    require_dir(DNSMOS_DIR, "DNSMOS model directory")
    require_file(DNSMOS_DIR / "sig_bak_ovr.onnx", "DNSMOS sig_bak_ovr.onnx")
    require_file(DNSMOS_DIR / "model_v8.onnx", "DNSMOS model_v8.onnx")

    if csv_path.exists():
        csv_path.unlink()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    wav_count = len(list(wav_dir.glob("*.wav")))
    if wav_count == 0:
        print(f"  WARNING: no wav files found in {wav_dir}")
        return []

    print(f"  Running DNSMOS on {wav_count} wav files ...")
    run_subprocess(
        [sys.executable, str(DNSMOS_LOCAL_PY), "-t", str(wav_dir), "-o", str(csv_path)],
        cwd=CHALLENGE_DIR,
        label="DNSMOS",
    )

    if not csv_path.exists():
        raise RuntimeError(f"DNSMOS did not create expected CSV: {csv_path}")

    scores: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append({
                "filename": row.get("filename", ""),
                "SIG": float(row.get("SIG", 0.0)),
                "BAK": float(row.get("BAK", 0.0)),
                "OVRL": float(row.get("OVRL", 0.0)),
                "P808_MOS": float(row.get("P808_MOS", 0.0)),
            })
    return scores


def mean_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {"SIG": 0.0, "BAK": 0.0, "OVRL": 0.0, "P808_MOS": 0.0}
    keys = ["SIG", "BAK", "OVRL", "P808_MOS"]
    return {k: float(np.mean([s[k] for s in scores])) for k in keys}


def add_summary(summary_rows: List[Dict[str, object]], system: str, scores: List[Dict[str, float]],
                train_samples: object = "-", model_type: str = "-") -> None:
    m = mean_scores(scores)
    row = {
        "system": system,
        "train_samples": train_samples,
        "model_type": model_type,
        "num_clips": len(scores),
        **m,
    }
    summary_rows.append(row)
    print(f"  {system}: SIG={m['SIG']:.3f}  BAK={m['BAK']:.3f}  "
          f"OVRL={m['OVRL']:.3f}  P808={m['P808_MOS']:.3f}  n={len(scores)}")
    print_summary_table(summary_rows)


def print_summary_table(summary_rows: List[Dict[str, object]]) -> None:
    print("\n  Running summary:")
    print(f"  {'System':<46} {'n':>5} {'SIG':>6} {'BAK':>6} {'OVRL':>6} {'P808':>6}")
    print("  " + "-" * 84)
    for row in summary_rows:
        print(f"  {str(row['system']):<46} {int(row['num_clips']):>5} "
              f"{float(row['SIG']):>6.3f} {float(row['BAK']):>6.3f} "
              f"{float(row['OVRL']):>6.3f} {float(row['P808_MOS']):>6.3f}")
    print()

# -----------------------------------------------------------------------------
# Run discovery / model loading
# -----------------------------------------------------------------------------
def find_completed_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    if not RUNS_DIR.is_dir():
        return runs

    for ckpt_path in sorted(RUNS_DIR.glob("lr*_samp*/*_h*/best_model.pt")):
        model_dir = ckpt_path.parent
        config_dir = model_dir.parent
        folder_name = config_dir.name
        model_name = model_dir.name

        try:
            train_samples = int(folder_name.split("_samp", 1)[1])
        except (IndexError, ValueError):
            train_samples = "?"

        if model_name.startswith("gru"):
            model_type = "gru"
        elif model_name.startswith("crn"):
            model_type = "crn"
        else:
            continue

        try:
            hidden_size = int(model_name.split("_h", 1)[1])
        except (IndexError, ValueError):
            # Fall back to checkpoint metadata if the folder name is unusual.
            ckpt = safe_torch_load(ckpt_path, map_location="cpu")
            hidden_size = int(ckpt.get("hidden_size", 161))

        runs.append({
            "run_name": f"{folder_name}_{model_name}",
            "model_dir": str(model_dir),
            "model_type": model_type,
            "hidden_size": hidden_size,
            "train_samples": train_samples,
            "best_model_path": ckpt_path,
        })
    return runs


def load_model(run_info: Dict[str, object], device: torch.device) -> nn.Module:
    ckpt_path = Path(run_info["best_model_path"])
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    hidden_size = int(ckpt.get("hidden_size", run_info["hidden_size"])) if isinstance(ckpt, dict) else int(run_info["hidden_size"])
    model_type = str(ckpt.get("model_type", run_info["model_type"])) if isinstance(ckpt, dict) else str(run_info["model_type"])

    if model_type == "gru":
        model = GRUModel(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE)
    elif model_type == "crn":
        model = CRNModel(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE)
    else:
        raise ValueError(f"Unknown model_type in {ckpt_path}: {model_type}")

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Evaluation steps
# -----------------------------------------------------------------------------
def score_noisy_baseline(test_pt_files: List[Path], summary_rows: List[Dict[str, object]]) -> None:
    out_dir = RESULTS_DIR / "noisy_baseline"
    tmp_dir = out_dir / "_score_wavs"
    keep_dir = out_dir / "noisy_wavs"
    csv_path = out_dir / "dnsmos_scores.csv"

    print(f"  Reconstructing noisy mic wavs from {len(test_pt_files)} .pt files ...")
    write_reconstructed_noisy_wavs(test_pt_files, tmp_dir, save_copy_dir=keep_dir)
    scores = run_dnsmos(tmp_dir, csv_path)
    clean_dir(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    add_summary(summary_rows, "Noisy input", scores)


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return wav.astype(np.float32)
    try:
        from scipy.signal import resample_poly  # type: ignore
        # Exact, filtered conversion for 48k -> 16k and generally safe otherwise.
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        return resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    except Exception:
        pass
    try:
        import librosa  # type: ignore
        return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    except Exception:
        # Final fallback for 48k baseline output.  Less ideal than filtered
        # resampling, but prevents a crash if scipy/librosa is unavailable here.
        if sr % SAMPLE_RATE == 0:
            return wav[:: sr // SAMPLE_RATE].astype(np.float32)
        raise RuntimeError(f"Cannot resample from {sr} Hz to {SAMPLE_RATE} Hz: install scipy or librosa")


def score_onnx_baseline(test_pt_files: List[Path], summary_rows: List[Dict[str, object]]) -> None:
    if SKIP_ONNX_BASELINE:
        print("  SKIP_ONNX_BASELINE=1, skipping ONNX baseline.")
        return
    require_file(BASELINE_ENHANCE_PY, "baseline/enhance.py")
    if ONNX_MODEL_PATH is None or not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX baseline model not found under {CHALLENGE_DIR / 'baseline'}")

    out_dir = RESULTS_DIR / "onnx_baseline"
    tmp_in = out_dir / "_tmp_in"
    tmp_out = out_dir / "_tmp_out"
    score_dir = out_dir / "_score_wavs"
    keep_dir = out_dir / "enhanced_wavs"
    csv_path = out_dir / "dnsmos_scores.csv"

    clean_dir(tmp_in)
    clean_dir(tmp_out)
    clean_dir(score_dir)
    keep_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Preparing {len(test_pt_files)} ONNX baseline input pairs ...")
    print("  Far-end/lpb is silent zeros because this project is nearend-only.")
    for i, pt_path in enumerate(test_pt_files):
        data = read_pt_sample(pt_path)
        noisy_wav = reconstruct_wav(data["noisy_mag"], data["noisy_phase"], mask=None)
        base = pt_path.stem
        sf.write(tmp_in / f"{base}_mic.wav", noisy_wav, SAMPLE_RATE)
        sf.write(tmp_in / f"{base}_lpb.wav", np.zeros_like(noisy_wav, dtype=np.float32), SAMPLE_RATE)
        if (i + 1) % 200 == 0:
            print(f"    ONNX input pairs {i+1}/{len(test_pt_files)}")

    print("  Running instructor ONNX baseline enhance.py ...")
    run_subprocess(
        [
            sys.executable,
            str(BASELINE_ENHANCE_PY),
            "--model_path", str(ONNX_MODEL_PATH),
            "--data_dir", str(tmp_in),
            "--output_dir", str(tmp_out),
        ],
        cwd=CHALLENGE_DIR,
        label="ONNX baseline",
    )
    shutil.rmtree(tmp_in, ignore_errors=True)

    out_files = sorted(tmp_out.glob("*.wav"))
    if not out_files:
        raise RuntimeError(f"ONNX baseline produced no wav files in {tmp_out}")

    print(f"  Resampling/copying {len(out_files)} ONNX outputs to 16 kHz for DNSMOS ...")
    for i, wav_path in enumerate(out_files):
        wav, sr = sf.read(wav_path)
        wav = np.asarray(wav, dtype=np.float32)
        wav_16k = resample_to_16k(wav, int(sr))
        if len(wav_16k) > N_SAMPLES:
            wav_16k = wav_16k[:N_SAMPLES]
        elif len(wav_16k) < N_SAMPLES:
            wav_16k = np.pad(wav_16k, (0, N_SAMPLES - len(wav_16k)))

        sf.write(score_dir / wav_path.name, wav_16k, SAMPLE_RATE)
        if i < WAV_SAVE_COUNT:
            sf.write(keep_dir / wav_path.name, wav_16k, SAMPLE_RATE)

    shutil.rmtree(tmp_out, ignore_errors=True)
    scores = run_dnsmos(score_dir, csv_path)
    shutil.rmtree(score_dir, ignore_errors=True)
    add_summary(summary_rows, "ONNX baseline", scores)


def evaluate_model(run_info: Dict[str, object], test_pt_files: List[Path],
                   device: torch.device, summary_rows: List[Dict[str, object]]) -> None:
    out_dir = RESULTS_DIR / str(run_info["run_name"])
    score_dir = out_dir / "_score_wavs"
    keep_dir = out_dir / "enhanced_wavs"
    csv_path = out_dir / "dnsmos_scores.csv"

    clean_dir(score_dir)
    keep_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {run_info['model_type'].upper()} h={run_info['hidden_size']} from:")
    print(f"    {run_info['best_model_path']}")
    model = load_model(run_info, device)

    t0 = time.time()
    print(f"  Enhancing {len(test_pt_files)} test samples ...")
    for i, pt_path in enumerate(test_pt_files):
        data = read_pt_sample(pt_path)
        enhanced_wav = enhance_one_sample(model, data, device)
        base = pt_path.stem
        sf.write(score_dir / f"{base}_mic.wav", enhanced_wav, SAMPLE_RATE)
        if i < WAV_SAVE_COUNT:
            sf.write(keep_dir / f"{base}_enhanced.wav", enhanced_wav, SAMPLE_RATE)
        if (i + 1) % 200 == 0 or (i + 1) == len(test_pt_files):
            print(f"    enhanced {i+1}/{len(test_pt_files)}  |  {time.time() - t0:.0f}s")

    scores = run_dnsmos(score_dir, csv_path)
    shutil.rmtree(score_dir, ignore_errors=True)
    add_summary(
        summary_rows,
        str(run_info["run_name"]),
        scores,
        train_samples=run_info["train_samples"],
        model_type=str(run_info["model_type"]).upper(),
    )


def save_summary(summary_rows: List[Dict[str, object]]) -> Path:
    summary_csv = RESULTS_DIR / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["system", "train_samples", "model_type", "num_clips", "SIG", "BAK", "OVRL", "P808_MOS"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_csv

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    t_total = time.time()

    print("=" * 72)
    print("DNS Block 4 + Block 5 — Enhancement and DNSMOS Evaluation")
    print("=" * 72)
    print(f"Project dir       : {PROJECT_DIR}")
    print(f"Eval dir          : {EVAL_DIR}")
    print(f"Test pt dir       : {TEST_PT_DIR}")
    print(f"Test wav dir      : {TEST_WAV_DIR}  (not required for full scoring)")
    print(f"Runs dir          : {RUNS_DIR}")
    print(f"Results dir       : {RESULTS_DIR}")
    print(f"Challenge dir     : {CHALLENGE_DIR}")
    print(f"DNSMOS script     : {DNSMOS_LOCAL_PY}")
    print(f"Baseline script   : {BASELINE_ENHANCE_PY}")
    print(f"ONNX model        : {ONNX_MODEL_PATH}")
    print(f"N_SCORE           : {N_SCORE}")
    print(f"WAV_SAVE_COUNT    : {WAV_SAVE_COUNT}")
    print("=" * 72)

    require_dir(TEST_PT_DIR, "test pt directory")
    require_dir(RUNS_DIR, "full_train runs directory")
    require_file(DNSMOS_LOCAL_PY, "dnsmos_local.py")
    require_dir(DNSMOS_DIR, "DNSMOS directory")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_pt_files = sorted(TEST_PT_DIR.glob("sample_*.pt"))
    if not all_pt_files:
        raise RuntimeError(f"No sample_*.pt files found in {TEST_PT_DIR}. Run block1and2.py first.")
    test_pt_files = all_pt_files[: min(N_SCORE, len(all_pt_files))]
    print(f"Found {len(all_pt_files)} test .pt files; scoring {len(test_pt_files)}.")

    runs = find_completed_runs()
    if not runs:
        raise RuntimeError(f"No completed runs found under {RUNS_DIR}. Run block3.py full_train first.")
    print(f"Found {len(runs)} completed model run(s):")
    for r in runs:
        print(f"  - {r['run_name']}  ({r['best_model_path']})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device            : {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    summary_rows: List[Dict[str, object]] = []

    if not SKIP_NOISY_BASELINE:
        print("=" * 72)
        print("Step 1: Score noisy input reconstructed from test .pt files")
        print("=" * 72)
        score_noisy_baseline(test_pt_files, summary_rows)

    print("=" * 72)
    print("Step 2: Score instructor ONNX baseline")
    print("=" * 72)
    score_onnx_baseline(test_pt_files, summary_rows)

    for idx, run in enumerate(runs, start=3):
        print("=" * 72)
        print(f"Step {idx}: Evaluate {run['run_name']}")
        print("=" * 72)
        evaluate_model(run, test_pt_files, device, summary_rows)

    summary_csv = save_summary(summary_rows)
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print_summary_table(summary_rows)
    print(f"Summary saved to  : {summary_csv}")
    print(f"Elapsed time      : {time.time() - t_total:.0f}s ({(time.time() - t_total) / 3600:.2f} h)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc)
        print("\nCheck the printed paths above.  The most common fixes are:")
        print("  1. Put this file in PROJECT_DIR as block4and5.py.")
        print("  2. Make sure block1and2.py has created eval_dataset/test/pt_files.")
        print("  3. Make sure block3.py full_train runs have best_model.pt files.")
        print("  4. Set CHALLENGE_DIR to the folder containing baseline/, DNSMOS/, dnsmos_local.py.")
        raise
