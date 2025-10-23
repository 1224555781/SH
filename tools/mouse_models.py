[BEGIN]
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Optional: torch only needed for LSTM behavior cloning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ------------- Data utilities -------------
@dataclass
class Movement:
    t: np.ndarray   # seconds, shape [T]
    xy: np.ndarray  # positions, shape [T, 2]
    target: np.ndarray  # [2]
    target_w: float     # scalar

def load_movements_from_csv(path: str) -> List[Movement]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return []

    # Group by movements using start_flag/end_flag if present
    movements = []
    cur = []
    for r in rows:
        sf = int(r.get("start_flag", "0") or 0)
        ef = int(r.get("end_flag", "0") or 0)
        cur.append(r)
        if ef == 1:
            movements.append(cur)
            cur = []
    if cur:
        movements.append(cur)

    result = []
    for seg in movements:
        t = np.array([float(r.get("time_ms", "0")) for r in seg], dtype=np.float64) / 1000.0
        t = t - t[0]
        x = np.array([float(r["x"]) for r in seg], dtype=np.float64)
        y = np.array([float(r["y"]) for r in seg], dtype=np.float64)
        xy = np.stack([x, y], axis=-1)
        tx = float(seg[-1].get("target_x", seg[0].get("target_x", x[-1])))
        ty = float(seg[-1].get("target_y", seg[0].get("target_y", y[-1])))
        tw = float(seg[-1].get("target_w", 40.0))
        result.append(Movement(t=t, xy=xy, target=np.array([tx, ty], dtype=np.float64), target_w=tw))
    return result


# ------------- Fitts' law (duration model) -------------
class FittsLaw:
    def __init__(self):
        self.a = 0.08  # seconds
        self.b = 0.12  # seconds per bit

    def fit(self, demos: List[Movement]):
        X = []
        y = []
        for m in demos:
            D = float(np.linalg.norm(m.xy[-1] - m.xy[0]))
            W = max(float(m.target_w), 1.0)
            ID = math.log2(D / W + 1.0) if W > 0 else 0.0
            T = float(m.t[-1] - m.t[0])
            if np.isfinite(ID) and np.isfinite(T) and D > 2.0 and T > 0.05:
                X.append([1.0, ID])
                y.append(T)
        if len(X) >= 2:
            X = np.array(X)
            y = np.array(y)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.a, self.b = float(beta[0]), float(beta[1])

    def predict_T(self, D: float, W: float) -> float:
        ID = math.log2(D / max(W, 1e-6) + 1.0)
        T = max(0.05, self.a + self.b * ID)
        return T


# ------------- DMP + ProMP -------------
class DMP2D:
    def __init__(self, n_basis=30, alpha_z=25.0, beta_z=None, alpha_s=4.0, l2=1e-3):
        self.n_basis = n_basis
        self.alpha_z = alpha_z
        self.beta_z = beta_z if beta_z is not None else alpha_z / 4.0
        self.alpha_s = alpha_s
        self.c = np.exp(-self.alpha_s * np.linspace(0, 1, n_basis))  # centers in s
        self.h = np.ones(n_basis) * (n_basis ** 1.5) / self.c / self.alpha_s
        self.l2 = l2
        self.w = np.zeros((2, n_basis))  # weights per DOF

    def _basis(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(s, 1e-6, 1.0)
        psi = np.exp(-self.h * (s[:, None] - self.c[None, :]) ** 2)
        denom = np.sum(psi, axis=1, keepdims=True) + 1e-9
        return psi / denom

    def fit_from_demo(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        T = float(t[-1] - t[0]) if t[-1] > 0 else 1.0
        tau = max(T, 1e-3)
        yd = np.gradient(y, t, axis=0, edge_order=2)
        ydd = np.gradient(yd, t, axis=0, edge_order=2)
        y0 = y[0]
        g = y[-1]
        s = np.exp(-self.alpha_s * (t - t[0]) / tau)
        Psi = self._basis(s)
        eps = 1e-6
        denom_vec = (g - y0) + eps
        F = (tau ** 2) * ydd - self.alpha_z * (self.beta_z * (g - y) - tau * yd)
        F = F / denom_vec
        A = Psi
        reg = self.l2 * np.eye(self.n_basis)
        AtA = A.T @ A + reg
        AtF = A.T @ F
        w_mat = np.linalg.solve(AtA, AtF)
        self.w = w_mat.T.copy()
        return self.w

    def rollout(self, y0: np.ndarray, g: np.ndarray, T: float, dt: float = 1/240,
                w_override: Optional[np.ndarray] = None, noise_scale: float = 0.6) -> np.ndarray:
        steps = max(2, int(T / dt))
        tau = max(T, 1e-3)
        y = y0.copy()
        yd = np.zeros_like(y0)
        out = [y.copy()]
        w_use = self.w if w_override is None else w_override
        ou_state = 0.0
        ou_theta, ou_sigma = 6.0, noise_scale
        d = g - y0
        if np.linalg.norm(d) < 1e-6:
            ortho = np.array([0.0, 0.0])
        else:
            ortho = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)

        for i in range(1, steps):
            t_now = i * dt
            s = math.exp(-self.alpha_s * t_now / tau)
            psi = self._basis(np.array([s]))[0]
            f_s = (psi @ w_use.T)
            f_s = f_s * (g - y0)
            ydd = (self.alpha_z * (self.beta_z * (g - y) - tau * yd) + f_s) / (tau ** 2 + 1e-9)
            yd = yd + ydd * dt
            y = y + yd * dt
            speed = float(np.linalg.norm(yd))
            ou_state += ou_theta * (-ou_state) * dt + ou_sigma * math.sqrt(dt) * np.random.randn()
            y_noisy = y + ortho * ou_state * (0.2 + 0.8 * min(1.0, speed / 1500.0))
            out.append(y_noisy.copy())
        return np.stack(out, axis=0)


class ProMP:
    def __init__(self, dmp: DMP2D):
        self.dmp = dmp
        self.W = []
        self.mu = None
        self.cov = None

    def fit(self, demos: List[Movement]):
        W = []
        for m in demos:
            w = self.dmp.fit_from_demo(m.t, m.xy)
            W.append(w.reshape(-1))
        W = np.stack(W, axis=0)
        self.W = W
        self.mu = np.mean(W, axis=0)
        S = np.cov(W.T)
        lam = 1e-3
        self.cov = (1 - lam) * S + lam * np.eye(S.shape[0])

    def sample_weights(self, kappa: float = 1.0) -> np.ndarray:
        cov = self.cov * (kappa ** 2)
        w = np.random.multivariate_normal(self.mu, cov)
        return w.reshape(2, -1)

    def generate(self, start: np.ndarray, goal: np.ndarray, T: float, dt: float = 1/240,
                 noise_scale: float = 0.6, kappa: float = 1.0) -> np.ndarray:
        w = self.sample_weights(kappa=kappa)
        return self.dmp.rollout(start, goal, T, dt=dt, w_override=w, noise_scale=noise_scale)


# ------------- LSTM Behavior Cloning -------------
class LSTMMouse(nn.Module):
    def __init__(self, input_dim=4, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        y, _ = self.lstm(x)
        out = self.head(y)
        return out

def prepare_bc_sequences(movs: List[Movement], hist=5, stride=1):
    X, Y = [], []
    for m in movs:
        xy = m.xy
        tgt = m.target
        rel = tgt[None, :] - xy
        dxy = np.diff(xy, axis=0, prepend=xy[[0], :])
        Tn = len(xy)
        for t in range(hist, Tn, stride):
            inp = np.concatenate([dxy[t-1], rel[t] / (np.linalg.norm(rel[t]) + 1e-6)], axis=0)
            X.append(inp.astype(np.float32))
            Y.append((xy[t] - xy[t-1]).astype(np.float32))
    if not X:
        return np.zeros((0,1,4), dtype=np.float32), np.zeros((0,1,2), dtype=np.float32)
    X = np.stack(X, axis=0)[:, None, :]
    Y = np.stack(Y, axis=0)[:, None, :]
    return X, Y

def train_lstm_bc(movs: List[Movement], epochs=8, lr=1e-3, batch=1024):
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available. Install torch to use LSTM.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMMouse().to(device)
    X, Y = prepare_bc_sequences(movs)
    if len(X) == 0:
        raise RuntimeError("Not enough data to train LSTM.")
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"[LSTM] epoch {ep+1}/{epochs} loss={{total/len(ds):.6f}}")
    return model

@torch.no_grad()
def generate_lstm(model: LSTMMouse, start: np.ndarray, goal: np.ndarray, T: float,
                  dt: float = 1/240, vmax: float = 3000.0, temp: float = 0.5) -> np.ndarray:
    device = next(model.parameters()).device
    steps = max(2, int(T / dt))
    xy = start.copy()
    out = [xy.copy()]
    dxy_prev = np.zeros(2, dtype=np.float32)
    for i in range(1, steps):
        rel = goal - xy
        reln = rel / (np.linalg.norm(rel) + 1e-6)
        inp = np.concatenate([dxy_prev, reln], axis=0).astype(np.float32)[None, None, :]
        pred = model(torch.from_numpy(inp).to(device)).cpu().numpy()[0, 0]
        noise = np.random.randn(2).astype(np.float32) * temp * 0.4
        step_vec = pred + noise
        spd = np.linalg.norm(step_vec) / max(dt, 1e-6)
        if spd > vmax:
            step_vec = step_vec * (vmax * dt / (spd + 1e-9))
        xy = xy + step_vec
        dxy_prev = step_vec.astype(np.float32)
        out.append(xy.copy())
        if np.linalg.norm(goal - xy) < 1.0:
            break
    return np.stack(out, axis=0)


# ------------- Save/Load helpers -------------
def save_promp_model(promp: ProMP, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dmp = promp.dmp
    np.savez(path,
             mu=promp.mu, cov=promp.cov,
             n_basis=dmp.n_basis, alpha_z=dmp.alpha_z, beta_z=dmp.beta_z,
             alpha_s=dmp.alpha_s, c=dmp.c, h=dmp.h)
    print(f"[ProMP] saved to {{path}}")

def load_promp_model(path: str) -> ProMP:
    data = np.load(path)
    dmp = DMP2D(n_basis=int(data["n_basis"),
                alpha_z=float(data["alpha_z"]), beta_z=float(data["beta_z"]),
                alpha_s=float(data["alpha_s"]))
    dmp.c = data["c"]
    dmp.h = data["h"]
    promp = ProMP(dmp)
    promp.mu = data["mu"]
    promp.cov = data["cov"]
    print(f"[ProMP] loaded from {{path}}")
    return promp

def save_lstm_model(model: LSTMMouse, path: str):
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available. Cannot save LSTM.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[LSTM] saved to {{path}}")

def load_lstm_model(path: str) -> LSTMMouse:
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available. Cannot load LSTM.")
    model = LSTMMouse()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"[LSTM] loaded from {{path}}")
    return model


# ------------- CSV export -------------
def save_csv(path: str, traj: np.ndarray, target: np.ndarray, target_w: float = 40.0, fps: int = 240):
    """
    Export a single trajectory as one segment with start/end flags.
    Columns: time_ms, x, y, target_x, target_y, target_w, start_flag, end_flag
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["time_ms", "x", "y", "target_x", "target_y", "target_w", "start_flag", "end_flag"])
        dt_ms = 1000.0 / max(1, fps)
        for i, p in enumerate(traj):
            sf = 1 if i == 0 else 0
            ef = 1 if i == len(traj) - 1 else 0
            w.writerow([int(round(i * dt_ms)), float(p[0]), float(p[1]),
                        float(target[0]), float(target[1]), float(target_w), sf, ef])


# ------------- Main CLI -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV path for training data")
    parser.add_argument("--train-promp", action="store_true")
    parser.add_argument("--train-lstm", action="store_true")
    parser.add_argument("--load-promp", type=str, default="")
    parser.add_argument("--load-lstm", type=str, default="")
    parser.add_argument("--save-promp", type=str, default="")
    parser.add_argument("--save-lstm", type=str, default="")
    parser.add_argument("--gen-out", type=str, default="out_promp.csv")
    parser.add_argument("--gen-out-lstm", type=str, default="out_lstm.csv")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--goal", type=str, default="")
    parser.add_argument("--target-w", type=float, default=40.0)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--noise", type=float, default=0.6)
    parser.add_argument("--use-promp-only", action="store_true", help="Only generate via ProMP if both models available")
    args = parser.parse_args()

    moves = load_movements_from_csv(args.data)
    if not moves:
        raise SystemExit("No movements loaded. Check CSV and schema.")

    fitts = FittsLaw()
    fitts.fit(moves)

    def parse_xy(s: str) -> Optional[np.ndarray]:
        if not s:
            return None
        parts = s.split(",")
        if len(parts) != 2:
            return None
        return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)

    start = parse_xy(args.start)
    goal = parse_xy(args.goal)
    if start is None or goal is None:
        start = start if start is not None else moves[0].xy[0].astype(np.float64)
        goal = goal if goal is not None else moves[0].target.astype(np.float64)

    D = float(np.linalg.norm(goal - start))
    T = fitts.predict_T(D, args.target_w)
    dt = 1.0 / max(30, args.fps)

    promp = None
    if args.load_promp:
        promp = load_promp_model(args.load_promp)
    elif args.train_promp:
        dmp = DMP2D(n_basis=30)
        promp = ProMP(dmp)
        promp.fit(moves)
        if args.save_promp:
            save_promp_model(promp, args.save_promp)

    lstm_model = None
    if args.load_lstm:
        lstm_model = load_lstm_model(args.load_lstm)
    elif args.train_lstm:
        lstm_model = train_lstm_bc(moves, epochs=8, lr=1e-3, batch=1024)
        if args.save_lstm:
            save_lstm_model(lstm_model, args.save_lstm)

    if promp is not None:
        traj = promp.generate(start, goal, T, dt=dt, noise_scale=args.noise, kappa=1.0)
        save_csv(args.gen_out, traj, goal, target_w=args.target_w, fps=args.fps)
        print(f"Saved ProMP-DMP trajectory to {{args.gen_out}} (T≈{{T:.3f}}s, steps={{len(traj)}})")

    if lstm_model is not None and not args.use_promp-only:
        traj_lstm = generate_lstm(lstm_model, start, goal, T, dt=dt, vmax=3000.0, temp=0.6)
        save_csv(args.gen_out_lstm, traj_lstm, goal, target_w=args.target_w, fps=args.fps)
        print(f"Saved LSTM trajectory to {{args.gen_out_lstm}} (T≈{{T:.3f}}s, steps={{len(traj_lstm)}})")

if __name__ == "__main__":
    main()
[END]