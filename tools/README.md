# Human-like Mouse Motion Tools (Recording + Train/Save/Load + Generate)

## Overview
This toolkit helps you record real mouse movements and train models to generate human-like mouse trajectories. It includes:
- ProMP-DMP (probabilistic dynamic movement primitives) for fast, smooth, interpretable generation
- LSTM behavior cloning for higher-fidelity, more human-like motion (recommended for final use)

You can record your own data, train models, save/load them, and generate trajectories for automation or simulation.

## Files
- tools/record_mouse.py — Record mouse trajectories to CSV (120–240 Hz). Hotkeys to start/stop.
- tools/mouse_models.py — Train, save, load, and generate trajectories (ProMP-DMP and LSTM).
- tools/requirements.txt — Python dependencies (PyTorch optional for LSTM).

## Installation
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\pip install -r tools/requirements.txt
# macOS/Linux
. .venv/bin/activate && pip install -r tools/requirements.txt
```
Notes:
- On macOS, grant Terminal/IDE "Accessibility" and "Input Monitoring" permissions.
- On Windows, run the terminal with sufficient permissions; some security software may require allow-listing.

## Recording Data
Hotkeys:
- Start a segment: Ctrl+Alt+S
- End a segment: First left-click after start ends the segment and sets the target at the click position
- Quit: Ctrl+Alt+Q (if recording, it finalizes the current segment using last position if no click)

Run:
```bash
python tools/record_mouse.py --out data/train.csv --fps 240 --target-w 40
```
Output CSV columns per sample:
- time_ms, x, y, target_x, target_y, target_w, start_flag, end_flag

Tip: Record a few hundred segments, each 100–500 samples (≈0.5–2 s), covering various distances and target sizes.

## Training, Saving, Loading, Generating
Train and save both models:
```bash
python tools/mouse_models.py --data data/train.csv \
  --train-promp --save-promp models/promp_model.npz \
  --train-lstm  --save-lstm  models/lstm_mouse.pt
```
Load saved models and generate trajectories (recommended to use LSTM for higher fidelity):
```bash
python tools/mouse_models.py --data data/train.csv \
  --load-promp models/promp_model.npz \
  --load-lstm  models/lstm_mouse.pt \
  --start 200,200 --goal 1000,600 --target-w 40 \
  --gen-out out_promp.csv --gen-out-lstm out_lstm.csv
```
Key flags:
- `--train-promp` / `--load-promp`: Train or load ProMP-DMP.
- `--train-lstm` / `--load-lstm`: Train or load LSTM (requires torch).
- `--save-promp` / `--save-lstm`: Save models.
- `--start x,y` and `--goal x,y`: Specify start/goal for generation.
- `--target-w`: Target width for Fitts-law timing (default 40 px).
- `--fps`: Export frequency (default 240 Hz).

## Recommendations
- Final use: Prefer LSTM for the most accurate mimicry; keep ProMP-DMP as a fast fallback.
- Keep lateral noise small (0.3–1.5 px) and tie it to speed to retain naturalness.
- Duration is estimated via Fitts’ law and can be adjusted by target width or by scaling after generation.

## CSV Compatibility
Generated CSVs use the same schema as recordings, allowing unified visualization and evaluation.

---

## Troubleshooting
- If LSTM training is unstable, reduce learning rate (e.g., 5e-4) or increase dataset size.
- If generation is jittery, lower noise/temperature or add a light low-pass filter post-generation.
- If PyTorch installation on macOS arm64 is problematic, consider Conda-forge or PyTorch nightly wheels.