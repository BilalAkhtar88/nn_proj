"""
DNS Project — Block 3: Model Training
======================================

Trains two models sequentially in one job:
    Model A — GRU-based DNS model (baseline re-implementation)
    Model B — CRN-based DNS model (encoder-decoder with GRU bottleneck)

Each model is fully trained before the next begins.
Results saved under runs/search/ or runs/full_train/ depending on RUN_TYPE.

Architecture — Model A (GRU):
    Input (batch, 1500, 161)
    GRU1 (input → hidden_A)
    Residual add: out1 + proj(x)
    GRU2 (hidden_A → hidden_A)
    Linear (hidden_A → 161)
    Sigmoid + Clip → mask [0,1]

Architecture — Model B (CRN):
    Encoder:
        Conv1d(161 → h*4) + BN + ELU    → enc1
        Conv1d(h*4 → h*2) + BN + ELU   → enc2
        Conv1d(h*2 → h)   + BN + ELU   → enc3
    Bottleneck:
        GRU(h → h, 2 layers)            → gru_out
        gru_out + enc3 (residual add, dims match)
    Decoder:
        ConvTranspose1d(h → h*2)   + BN + ELU → dec1
        dec1 + enc2 (residual add)
        ConvTranspose1d(h*2 → h*4) + BN + ELU → dec2
        dec2 + enc1 (residual add)
        ConvTranspose1d(h*4 → 161)             → output
        Sigmoid + Clip → mask [0,1]

Key design decisions:
    - DNSDataset and EvalDataset imported from block1and2.py
    - Fresh DNSDataset every epoch for train (online synthesis)
    - EvalDataset loads pre-saved .pt files for val and test
    - Weight init: Normal(mean=0, std=0.01) — same as lab exercises
    - noisy_mag clamped to 1e-4 in block1and2.py — prevents IRM overflow
    - DataParallel bug fix: forward() returns mask only, not hidden states
    - Gradient clipping: max_norm=1.0
    - CodeCarbon tracks energy for both models — ignored during search phase
    - Checkpoints every 5 epochs
    - Loss curves saved per model

Folder structure created:
    runs/
        search/                          ← RUN_TYPE=search
            lr<lr>_samp<n>/
                gru_h<hidden_A>/
                    best_model.pt
                    checkpoints/
                    loss_curves.png
                    training_log.csv
                    emissions.csv
                crn_h<hidden_B>/
                    best_model.pt
                    checkpoints/
                    loss_curves.png
                    training_log.csv
                    emissions.csv
        full_train/                      ← RUN_TYPE=full_train
            lr<lr>_samp<n>/
                gru_h<hidden_A>/
                crn_h<hidden_B>/

How to run:
    sbatch block3_search_lr5em03_h161.sbatch
    sbatch block3_full_lr5em03_h161_samp500.sbatch
    etc.

Pitfalls fixed (from train.py experience):
    1. IRM overflow: noisy_mag clamped to 1e-4 in block1and2.py
    2. DataParallel GRU hidden state corruption: return mask only
    3. Weight init: Normal(0, 0.01) prevents early instability
    4. Unbuffered output: flush=True on all prints for live log viewing
"""

import os
import sys
import glob
import time
import csv
import builtins

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from codecarbon import EmissionsTracker

# =============================================================================
# IMPORT FROM block1and2.py
# =============================================================================

sys.path.insert(0, os.path.expanduser('~/nn_proj'))

from block1and2 import (
    DNSDataset,       # online synthesis for train
    EvalDataset,      # loads pre-saved .pt files for val/test
    CONFIGS_DIR,      # ~/nn_proj/configs/
    EVAL_DIR,         # ~/nn_proj/eval_dataset/
    PROJECT_DIR,      # ~/nn_proj/
    SAMPLE_RATE,
    FRAME_SIZE, HOP_SIZE, DFT_SIZE, N_FREQS, N_FRAMES,
    BATCH_SIZE, NUM_WORKERS,
)

# =============================================================================
# FORCE UNBUFFERED OUTPUT
# =============================================================================

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs['flush'] = True
    _original_print(*args, **kwargs)
builtins.print = print

# =============================================================================
# HYPERPARAMETERS — set via sbatch environment variables
# =============================================================================

LEARNING_RATE  = float(os.environ.get('LEARNING_RATE',  '0.005'))
HIDDEN_SIZE_A  = int(os.environ.get('HIDDEN_SIZE_A',   '161'))   # GRU hidden
HIDDEN_SIZE_B  = int(os.environ.get('HIDDEN_SIZE_B',   '161'))   # CRN bottleneck
TRAIN_SAMPLES  = int(os.environ.get('TRAIN_SAMPLES',   '500'))
NUM_EPOCHS     = int(os.environ.get('NUM_EPOCHS',       '20'))
RUN_TYPE       = os.environ.get('RUN_TYPE', 'search')            # 'search' or 'full_train'

# =============================================================================
# FIXED CONSTANTS
# =============================================================================

INPUT_SIZE       = 161
OUTPUT_SIZE      = 161
CHECKPOINT_EVERY = 5
PRINT_EVERY      = 10
GRAD_CLIP        = 1.0
LR_STEP_SIZE     = 5
LR_GAMMA         = 0.95

# =============================================================================
# PATHS
# =============================================================================

RUNS_DIR = os.path.join(PROJECT_DIR, 'runs')

LR_STR        = f"{LEARNING_RATE:.0e}".replace('-', 'm').replace('+', '')
CONFIG_SUBDIR = f"lr{LR_STR}_samp{TRAIN_SAMPLES}"

RUN_BASE = os.path.join(RUNS_DIR, RUN_TYPE, CONFIG_SUBDIR)

MODEL_A_DIR = os.path.join(RUN_BASE, f'gru_h{HIDDEN_SIZE_A}')
MODEL_B_DIR = os.path.join(RUN_BASE, f'crn_h{HIDDEN_SIZE_B}')

for d in [MODEL_A_DIR, MODEL_B_DIR,
          os.path.join(MODEL_A_DIR, 'checkpoints'),
          os.path.join(MODEL_B_DIR, 'checkpoints')]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DEVICE
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# STARTUP INFO
# =============================================================================

print("=" * 70)
print("DNS Project — Block 3: Model Training")
print("=" * 70)
print(f"RUN_TYPE       : {RUN_TYPE}")
print(f"LEARNING_RATE  : {LEARNING_RATE}")
print(f"HIDDEN_SIZE_A  : {HIDDEN_SIZE_A}  (GRU)")
print(f"HIDDEN_SIZE_B  : {HIDDEN_SIZE_B}  (CRN bottleneck)")
print(f"TRAIN_SAMPLES  : {TRAIN_SAMPLES} per epoch (online synthesis)")
print(f"NUM_EPOCHS     : {NUM_EPOCHS}")
print(f"BATCH_SIZE     : {BATCH_SIZE}")
print(f"GRAD_CLIP      : {GRAD_CLIP}")
print(f"LR scheduler   : StepLR step={LR_STEP_SIZE} gamma={LR_GAMMA}")
print(f"Device         : {device}")
if torch.cuda.is_available():
    print(f"GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}        : {torch.cuda.get_device_name(i)}")
print(f"Model A dir    : {MODEL_A_DIR}")
print(f"Model B dir    : {MODEL_B_DIR}")
print()

# =============================================================================
# MODEL A — GRU
# =============================================================================

class GRUModel(nn.Module):
    """
    [Block 3 — Model A] Re-implementation of baseline ONNX DNS model.

    Architecture:
        GRU1 (input → hidden)
        Residual add: out1 + proj(x)
            proj = Identity if input == hidden
            proj = Linear   if input != hidden
        GRU2 (hidden → hidden)
        Linear (hidden → 161)
        Sigmoid + Clip → mask [0,1]

    Pitfall fixed:
        forward() returns mask ONLY — not hidden states.
        DataParallel gathers outputs along dim=0.
        Hidden state shape (1, batch, hidden) gets corrupted to
        (num_gpus, batch, hidden) when returned. Returning mask only
        avoids this entirely.
        Ref: pytorch/pytorch issues #7890, #15260

    Weight init:
        Normal(mean=0, std=0.01) — same as lab exercises.
        Small init keeps sigmoid outputs near 0.5 at start.
        Stable gradients from epoch 1.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=161,
                 output_size=OUTPUT_SIZE):
        super().__init__()
        self.gru1 = nn.GRU(input_size,  hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

        # residual projection — handles input_size != hidden_size
        if input_size != hidden_size:
            self.proj = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Normal(0, 0.01) — same as lab exercises."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        """
        Args:
            x    : (batch, frames, 161)
        Returns:
            mask : (batch, frames, 161) values in [0,1]
        """
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1 + self.proj(x))
        mask = torch.sigmoid(self.fc(out2))
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask

# =============================================================================
# MODEL B — CRN (Convolutional Recurrent Network)
# =============================================================================

class CRNModel(nn.Module):
    """
    [Block 3 — Model B] Convolutional Recurrent Network for DNS.

    Architecture:
        Encoder (Conv1d + BN + ELU):
            161 → h*4 → h*2 → h
        Bottleneck:
            GRU(h → h, 2 layers)
            Residual add with enc3 (same dim, direct add)
        Decoder (ConvTranspose1d + BN + ELU):
            h  → h*2  (+ residual enc2)
            h*2 → h*4 (+ residual enc1)
            h*4 → 161
        Sigmoid + Clip → mask [0,1]

    Why ELU not ReLU:
        Log power spectrogram features are negative for quiet frames.
        ReLU zeros all negative values, destroying quiet speech info.
        ELU(x) = x if x>0, alpha*(exp(x)-1) if x<=0 — preserves negatives.

    Why BN after Conv but not after GRU:
        Conv produces raw linear combinations — BN stabilises training.
        GRU has internal gating that naturally regulates activations.
        BN after GRU would interfere with hidden state dynamics.

    Why no BN before final Sigmoid:
        BN constrains output distribution, limiting mask dynamic range.
        Final layer is bare ConvTranspose1d → Sigmoid → Clip.

    Skip connections:
        Encoder outputs (enc1, enc2, enc3) added directly to decoder
        inputs where dimensions match. All matches are exact so no
        projection layers needed — direct residual addition throughout.

    Weight init:
        Normal(mean=0, std=0.01) — same as GRUModel and lab exercises.

    Pitfall fixed:
        Same DataParallel fix as GRUModel — return mask only.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=161,
                 output_size=OUTPUT_SIZE):
        super().__init__()
        h = hidden_size

        # ── Encoder ──────────────────────────────────────────────────────────
        # kernel_size=5, padding=2 keeps time dimension unchanged (1500 → 1500)
        # only frequency dimension is transformed
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

        # ── Bottleneck GRU ────────────────────────────────────────────────────
        # input = h, output = h — same size so residual add works directly
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True)

        # ── Decoder ──────────────────────────────────────────────────────────
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
        # no BN, no ELU before Sigmoid — keep output free to span full [0,1]
        self.dec3 = nn.ConvTranspose1d(h * 4, output_size, kernel_size=5,
                                       padding=2)

        self._init_weights()

    def _init_weights(self):
        """Normal(0, 0.01) — same as GRUModel and lab exercises."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        """
        Args:
            x    : (batch, frames, 161)  — (batch, time, freq)
        Returns:
            mask : (batch, frames, 161) values in [0,1]

        Note on transpose:
            Conv1d expects (batch, channels, length).
            Our data is (batch, time, freq).
            We treat freq as channels and time as length.
            So we transpose: (batch, time, freq) → (batch, freq, time)
            before conv, then transpose back after decoder.
        """
        # x: (batch, time, freq) → (batch, freq, time) for Conv1d
        x_t = x.transpose(1, 2)          # (batch, 161, 1500)

        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.enc1(x_t)              # (batch, h*4, 1500)
        e2 = self.enc2(e1)               # (batch, h*2, 1500)
        e3 = self.enc3(e2)               # (batch, h,   1500)

        # ── Bottleneck GRU ────────────────────────────────────────────────────
        # GRU expects (batch, time, features) — transpose enc3 output
        e3_t = e3.transpose(1, 2)        # (batch, 1500, h)
        gru_out, _ = self.gru(e3_t)      # (batch, 1500, h)

        # residual add with enc3 — both are (batch, 1500, h)
        gru_out = gru_out + e3_t

        # back to (batch, h, 1500) for ConvTranspose1d
        d = gru_out.transpose(1, 2)      # (batch, h, 1500)

        # ── Decoder with skip connections ─────────────────────────────────────
        d = self.dec1(d)                 # (batch, h*2, 1500)
        d = d + e2                       # skip from enc2 — both (batch, h*2, 1500)

        d = self.dec2(d)                 # (batch, h*4, 1500)
        d = d + e1                       # skip from enc1 — both (batch, h*4, 1500)

        d = self.dec3(d)                 # (batch, 161, 1500)

        # back to (batch, time, freq)
        out = d.transpose(1, 2)          # (batch, 1500, 161)

        mask = torch.sigmoid(out)
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask

# =============================================================================
# TRAINING HELPERS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    """One full training epoch. Returns mean train loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (noisy_feat, ideal_mask, _, _) in enumerate(loader):
        noisy_feat = noisy_feat.to(device)
        ideal_mask = ideal_mask.to(device)

        pred_mask = model(noisy_feat)
        loss      = criterion(pred_mask, ideal_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % PRINT_EVERY == 0:
            print(f"  [epoch {epoch}/{num_epochs}]"
                  f"  batch {batch_idx+1}/{len(loader)}"
                  f"  loss={loss.item():.6f}")

    return total_loss / n_batches


def validate(model, fixed_data, criterion):
    """Evaluate on fixed val or test set. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy_feat, ideal_mask, _, _ in fixed_data:
            pred_mask  = model(noisy_feat)
            total_loss += criterion(pred_mask, ideal_mask).item()

    return total_loss / len(fixed_data)


def save_checkpoint(model, epoch, val_loss, train_loss, lr,
                    hidden_size, model_type, out_dir):
    """Save periodic checkpoint."""
    state_dict = (model.module.state_dict()
                  if isinstance(model, nn.DataParallel)
                  else model.state_dict())
    path = os.path.join(out_dir, 'checkpoints',
                        f'checkpoint_epoch{epoch}.pt')
    torch.save({
        'epoch':       epoch,
        'state_dict':  state_dict,
        'val_loss':    val_loss,
        'train_loss':  train_loss,
        'lr':          lr,
        'hidden_size': hidden_size,
        'model_type':  model_type,
    }, path)
    print(f"  Checkpoint saved: {path}")


def save_best(model, epoch, val_loss, train_loss, lr,
              hidden_size, model_type, out_dir):
    """Save best model."""
    state_dict = (model.module.state_dict()
                  if isinstance(model, nn.DataParallel)
                  else model.state_dict())
    path = os.path.join(out_dir, 'best_model.pt')
    torch.save({
        'epoch':       epoch,
        'state_dict':  state_dict,
        'val_loss':    val_loss,
        'train_loss':  train_loss,
        'lr':          lr,
        'hidden_size': hidden_size,
        'model_type':  model_type,
    }, path)
    print(f"  ✓ Best model saved (val_loss={val_loss:.6f})")


def plot_loss_curves(train_losses, val_losses, config_name, out_dir,
                     emissions_kwh=0.0, co2_g=0.0):
    """Save loss curve plot to out_dir."""
    epochs = list(range(1, len(train_losses) + 1))
    gaps   = [v - t for v, t in zip(val_losses, train_losses)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label='train loss', color='steelblue')
    axes[0].plot(epochs, val_losses,   label='val loss',   color='crimson')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title(f'Loss Curves — {config_name}')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, gaps, color='darkorange', label='val - train gap')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Gap (val - train)')
    axes[1].set_title('Overfitting Gap')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(
        f'{config_name}\n'
        f'best_val={min(val_losses):.4f} | '
        f'energy={emissions_kwh*1000:.2f}Wh | '
        f'CO2={co2_g:.2f}g',
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Loss curves saved: {path}")


def run_training(model, model_name, model_type, hidden_size,
                 out_dir, fixed_val_data, fixed_test_data,
                 train_cfg_path):
    """
    Full training loop for one model.

    Args:
        model          : GRUModel or CRNModel instance (already on device)
        model_name     : string label for logging e.g. 'GRU h=161'
        model_type     : 'gru' or 'crn'
        hidden_size    : int — for checkpoint saving
        out_dir        : folder to save outputs
        fixed_val_data : list of batches from EvalDataset
        fixed_test_data: list of batches from EvalDataset
        train_cfg_path : path to config_train.yaml for DNSDataset
    """

    print()
    print("=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # wrap with DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    # ── CodeCarbon ────────────────────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name = f"{model_type}_{out_dir.split('/')[-1]}",
        output_dir   = out_dir,
        log_level    = "error",
    )
    tracker.start()

    # ── training log CSV ──────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, 'training_log.csv')
    log_fields = ['epoch', 'train_loss', 'val_loss', 'gap', 'lr',
                  'time_sec', 'cumulative_energy_wh', 'cumulative_co2_g']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── training state ────────────────────────────────────────────────────────
    train_losses  = []
    val_losses    = []
    best_val_loss = float('inf')
    t_start       = time.time()

    config_name = (f"{model_type}_h{hidden_size}_"
                   f"lr{LR_STR}_samp{TRAIN_SAMPLES}_{RUN_TYPE}")

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()

        # fresh train DataLoader every epoch — online synthesis
        train_loader = DataLoader(
            DNSDataset(train_cfg_path, TRAIN_SAMPLES),
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = NUM_WORKERS,
            drop_last   = True,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}  lr={current_lr:.2e}")
        print("-" * 60)

        # train
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, epoch, NUM_EPOCHS)

        # validate
        val_loss = validate(model, fixed_val_data, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        gap        = val_loss - train_loss

        # read cumulative energy from CodeCarbon
        try:
            em_df    = pd.read_csv(os.path.join(out_dir, 'emissions.csv'))
            cum_wh   = em_df['energy_consumed'].iloc[-1] * 1000  # kWh → Wh
            cum_co2  = em_df['emissions'].iloc[-1] * 1000         # kg → g
        except Exception:
            cum_wh  = 0.0
            cum_co2 = 0.0

        print("-" * 60)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}"
              f"  train={train_loss:.6f}"
              f"  val={val_loss:.6f}"
              f"  gap={gap:+.6f}"
              f"  time={epoch_time:.0f}s"
              f"  energy={cum_wh:.2f}Wh"
              f"  CO2={cum_co2:.2f}g")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best(model, epoch, val_loss, train_loss,
                      current_lr, hidden_size, model_type, out_dir)
        else:
            print(f"  No improvement. Best val={best_val_loss:.6f}")

        # checkpoint every 5 epochs
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, epoch, val_loss, train_loss,
                            current_lr, hidden_size, model_type, out_dir)

        # append to training log
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                'epoch':               epoch,
                'train_loss':          round(train_loss, 6),
                'val_loss':            round(val_loss,   6),
                'gap':                 round(gap,        6),
                'lr':                  current_lr,
                'time_sec':            round(epoch_time, 1),
                'cumulative_energy_wh':round(cum_wh,     4),
                'cumulative_co2_g':    round(cum_co2,    4),
            })

    # ── training complete ─────────────────────────────────────────────────────
    emissions = tracker.stop()
    total_time = time.time() - t_start

    # read final CodeCarbon CSV
    try:
        em_df        = pd.read_csv(os.path.join(out_dir, 'emissions.csv'))
        row          = em_df.iloc[-1]
        total_wh     = row['energy_consumed'] * 1000
        gpu_wh       = row['gpu_energy']      * 1000
        cpu_wh       = row['cpu_energy']      * 1000
        ram_wh       = row['ram_energy']      * 1000
        total_co2_g  = row['emissions']       * 1000
        avg_gpu_w    = row['gpu_power']
        avg_cpu_w    = row['cpu_power']
    except Exception as e:
        print(f"Warning: could not read emissions CSV: {e}")
        total_wh = gpu_wh = cpu_wh = ram_wh = total_co2_g = 0.0
        avg_gpu_w = avg_cpu_w = 0.0

    print()
    print("=" * 70)
    print(f"Training complete: {model_name}")
    print("=" * 70)
    print(f"Total time     : {total_time:.0f}s  ({total_time/3600:.2f}h)")
    print(f"Best val loss  : {best_val_loss:.6f}")
    print(f"Total energy   : {total_wh:.4f} Wh")
    print(f"GPU energy     : {gpu_wh:.4f} Wh")
    print(f"CPU energy     : {cpu_wh:.4f} Wh")
    print(f"RAM energy     : {ram_wh:.4f} Wh")
    print(f"Total CO2      : {total_co2_g:.4f} gCO2eq")
    print(f"Avg GPU power  : {avg_gpu_w:.1f} W")
    print(f"Avg CPU power  : {avg_cpu_w:.1f} W")

    # plot loss curves
    plot_loss_curves(train_losses, val_losses, config_name, out_dir,
                     emissions_kwh=total_wh/1000, co2_g=total_co2_g)

    # ── final test evaluation ─────────────────────────────────────────────────
    print()
    print("Final test evaluation using best model...")

    best_ckpt = torch.load(os.path.join(out_dir, 'best_model.pt'),
                           map_location=device)

    if model_type == 'gru':
        eval_model = GRUModel(
            input_size  = INPUT_SIZE,
            hidden_size = best_ckpt['hidden_size'],
            output_size = OUTPUT_SIZE,
        ).to(device)
    else:
        eval_model = CRNModel(
            input_size  = INPUT_SIZE,
            hidden_size = best_ckpt['hidden_size'],
            output_size = OUTPUT_SIZE,
        ).to(device)

    eval_model.load_state_dict(best_ckpt['state_dict'])
    eval_model.eval()

    test_loss = validate(eval_model, fixed_test_data, criterion)

    print(f"Best model epoch : {best_ckpt['epoch']}")
    print(f"Train loss       : {best_ckpt['train_loss']:.6f}")
    print(f"Val loss         : {best_ckpt['val_loss']:.6f}")
    print(f"Test loss        : {test_loss:.6f}")
    print()

    return {
        'model_name':    model_name,
        'best_val_loss': best_val_loss,
        'test_loss':     test_loss,
        'total_wh':      total_wh,
        'total_co2_g':   total_co2_g,
        'total_time':    total_time,
        'params':        sum(p.numel() for p in eval_model.parameters()),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    t_total = time.time()

    # ── paths ─────────────────────────────────────────────────────────────────
    train_cfg  = os.path.join(CONFIGS_DIR, 'config_train.yaml')
    val_pt_dir = os.path.join(EVAL_DIR, 'val',  'pt_files')
    tst_pt_dir = os.path.join(EVAL_DIR, 'test', 'pt_files')

    for p in [train_cfg, val_pt_dir, tst_pt_dir]:
        if not os.path.exists(p):
            raise RuntimeError(
                f"Required path not found: {p}\n"
                f"Run block1and2.py first to generate splits and eval data."
            )

    # ── load fixed val and test sets ──────────────────────────────────────────
    print("Loading fixed val and test sets from disk...")

    val_loader = DataLoader(
        EvalDataset(val_pt_dir),
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,       # pt files safe for multiprocessing
    )
    test_loader = DataLoader(
        EvalDataset(tst_pt_dir),
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
    )

    # pre-load all val and test batches into memory once
    print("Pre-loading val batches into memory...")
    fixed_val_data = [
        (nf.to(device), im.to(device), nm.to(device), np_.to(device))
        for nf, im, nm, np_ in val_loader
    ]
    print(f"Val batches loaded: {len(fixed_val_data)}")

    print("Pre-loading test batches into memory...")
    fixed_test_data = [
        (nf.to(device), im.to(device), nm.to(device), np_.to(device))
        for nf, im, nm, np_ in test_loader
    ]
    print(f"Test batches loaded: {len(fixed_test_data)}")
    print()

    results = []

    # ── Model A: GRU ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("MODEL A — GRU")
    print("=" * 70)

    model_a = GRUModel(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE_A,
        output_size = OUTPUT_SIZE,
    ).to(device)

    result_a = run_training(
        model        = model_a,
        model_name   = f"GRU h={HIDDEN_SIZE_A}",
        model_type   = 'gru',
        hidden_size  = HIDDEN_SIZE_A,
        out_dir      = MODEL_A_DIR,
        fixed_val_data  = fixed_val_data,
        fixed_test_data = fixed_test_data,
        train_cfg_path  = train_cfg,
    )
    results.append(result_a)

    # ── Model B: CRN ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("MODEL B — CRN")
    print("=" * 70)

    model_b = CRNModel(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE_B,
        output_size = OUTPUT_SIZE,
    ).to(device)

    result_b = run_training(
        model        = model_b,
        model_name   = f"CRN h={HIDDEN_SIZE_B}",
        model_type   = 'crn',
        hidden_size  = HIDDEN_SIZE_B,
        out_dir      = MODEL_B_DIR,
        fixed_val_data  = fixed_val_data,
        fixed_test_data = fixed_test_data,
        train_cfg_path  = train_cfg,
    )
    results.append(result_b)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>10} {'BestVal':>10} "
          f"{'TestLoss':>10} {'Energy(Wh)':>12} {'CO2(g)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model_name']:<20} {r['params']:>10,} "
              f"{r['best_val_loss']:>10.6f} {r['test_loss']:>10.6f} "
              f"{r['total_wh']:>12.4f} {r['total_co2_g']:>10.4f}")
    print()
    print(f"Total job time : {total_elapsed:.0f}s  ({total_elapsed/3600:.2f}h)")
    print(f"RUN_TYPE       : {RUN_TYPE}")
    print(f"Results saved  : {RUN_BASE}")
    print()
    print("Next steps:")
    if RUN_TYPE == 'search':
        print("  Compare val losses across search runs.")
        print("  Pick best lr and hidden sizes.")
        print("  Submit full_train jobs with best config.")
    else:
        print("  Run block4and5.py to enhance audio and compute DNSMOS.")
