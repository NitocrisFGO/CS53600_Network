#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

@dataclass
class Q3Result:
    server_id: str
    trace_path: str
    out_dir: str
    metrics: Dict

# ---------- utilities ----------
def split_servers_global(latest: Dict[str, str], test_count: int = 5, seed: int = 42):
    """
    Split servers into:
      - train servers: all except selected test servers
      - test servers: randomly chosen held-out servers
    """
    server_ids = list(latest.keys())

    def keyfun(sid: str):
        m = re.fullmatch(r"\d+", sid)
        return (0, int(sid)) if m else (1, sid)

    server_ids = sorted(server_ids, key=keyfun)

    if len(server_ids) <= test_count:
        raise ValueError(f"Need more than {test_count} servers, got {len(server_ids)}")

    rng = random.Random(seed)
    test_ids = sorted(rng.sample(server_ids, test_count), key=keyfun)
    train_ids = [sid for sid in server_ids if sid not in test_ids]

    return train_ids, test_ids


def find_all_trace_csv(q2_root: str) -> List[str]:
    traces = []
    for root, _, files in os.walk(q2_root):
        for fn in files:
            if fn.lower() == "trace.csv":
                traces.append(os.path.join(root, fn))
    return traces


def parse_server_id_from_path(path: str) -> Optional[str]:
    """
    Try to infer server id folder name from .../results/q2/<server_id>/.../trace.csv
    If not found, return None.
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    # look for "q2" then take next component as server_id
    for i in range(len(parts) - 1):
        if parts[i].lower() == "q2" and i + 1 < len(parts):
            # q2/<server_id>/...
            return parts[i + 1]
    return None


def pick_latest_trace_per_server(trace_paths: List[str]) -> Dict[str, str]:
    """
    For each server_id, pick the trace.csv with latest mtime.
    If server_id cannot be inferred, group into 'unknown'.
    """
    best: Dict[str, Tuple[float, str]] = {}
    for p in trace_paths:
        sid = parse_server_id_from_path(p) or "unknown"
        mtime = os.path.getmtime(p)
        if sid not in best or mtime > best[sid][0]:
            best[sid] = (mtime, p)
    return {sid: p for sid, (t, p) in best.items()}


def ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def safe_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------- core Q3 ----------
# ✅ change 1: add cwnd(t) as a feature
# ✅ change 2: use retrans_delta (increment) instead of cumulative retrans
FEATURE_COLS = [
    "goodput_mbps",
    "rtt_ms",
    "loss",
    "rttvar_ms",
    "retrans_delta",
    "cwnd",
    "goodput_mbps_prev",
    "rtt_ms_prev",
    "loss_prev",
    "delta_cwnd_prev",
]

REQUIRED_COLS = ["t", "goodput_mbps", "cwnd", "rtt_ms", "rttvar_ms", "retrans", "loss"]

def build_dataset_from_trace(trace_csv: str, alpha: float = 0.02, beta: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(trace_csv)

    # required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"trace missing columns {missing}: {trace_csv}")

    # numeric conversion
    for c in REQUIRED_COLS:
        df[c] = ensure_numeric(df, c)

    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

    # retrans is cumulative -> convert to per-interval increment
    df["retrans_delta"] = df["retrans"].diff().fillna(0.0)
    df["retrans_delta"] = df["retrans_delta"].clip(lower=0.0)

    # label: next cwnd update
    df["delta_cwnd"] = df["cwnd"].shift(-1) - df["cwnd"]

    # objective value uses performance observed at t for decision at t-1
    df["eta_next"] = (
        df["goodput_mbps"].shift(-1)
        - alpha * df["rtt_ms"].shift(-1)
        - beta * df["loss"].shift(-1)
    )

    # -------- lag features --------
    df["goodput_mbps_prev"] = df["goodput_mbps"].shift(1)
    df["rtt_ms_prev"] = df["rtt_ms"].shift(1)
    df["loss_prev"] = df["loss"].shift(1)
    df["delta_cwnd_prev"] = df["delta_cwnd"].shift(1)

    # drop rows that cannot form full lagged feature + label
    df = df.dropna(subset=[
        "delta_cwnd",
        "eta_next",
        "goodput_mbps_prev",
        "rtt_ms_prev",
        "loss_prev",
        "delta_cwnd_prev",
    ]).reset_index(drop=True)

    return df

def predict_on_test_server(server_id: str, trace_path: str, model_pack, out_root: str) -> Q3Result:
    """
    Use the global model to predict on one held-out server trace.

    IMPORTANT:
    Since cwnd(t) is a feature, test-time rollout must use the PREVIOUSLY
    PREDICTED cwnd, not the real cwnd from the trace.

    Since lag features are also used, we recursively maintain:
      goodput(t-1), rtt(t-1), loss(t-1), delta_cwnd(t-1)
    """
    df = build_dataset_from_trace(trace_path)

    model = model_pack["model"]
    mu = model_pack["mu"]
    sigma = model_pack["sigma"]

    # real sequences (already aligned with dataset rows)
    t_test = df["t"].values.astype(float)
    cwnd_real = df["cwnd"].values.astype(float)
    y_test = df["delta_cwnd"].values.astype(float)

    goodput = df["goodput_mbps"].values.astype(float)
    rtt = df["rtt_ms"].values.astype(float)
    loss = df["loss"].values.astype(float)
    rttvar = df["rttvar_ms"].values.astype(float)
    retrans_delta = df["retrans_delta"].values.astype(float)

    # rollout
    cur = float(cwnd_real[0])

    # lag states：初始化用数据集第一行已有的 lag 值
    prev_goodput = float(df["goodput_mbps_prev"].iloc[0])
    prev_rtt = float(df["rtt_ms_prev"].iloc[0])
    prev_loss = float(df["loss_prev"].iloc[0])
    prev_delta_cwnd = float(df["delta_cwnd_prev"].iloc[0])

    cwnd_pred = [cur]
    yhat = []

    for i in range(len(df)):
        x_raw = np.array([
            goodput[i],          # goodput(t)
            rtt[i],              # rtt(t)
            loss[i],             # loss(t)
            rttvar[i],           # rttvar(t)
            retrans_delta[i],    # retrans_delta(t)
            cur,                 # cwnd(t) - use predicted rollout state
            prev_goodput,        # goodput(t-1)
            prev_rtt,            # rtt(t-1)
            prev_loss,           # loss(t-1)
            prev_delta_cwnd,     # delta_cwnd(t-1)
        ], dtype=float)

        x_std = (x_raw - mu) / sigma
        d = float(model.predict(x_std.reshape(1, -1))[0])

        # safety clamp
        d = clamp(d, -0.5 * cur, 0.5 * cur)

        yhat.append(d)

        nxt = max(1.0, cur + d)
        cwnd_pred.append(nxt)

        # 更新 lag 状态，供下一步使用
        prev_goodput = goodput[i]
        prev_rtt = rtt[i]
        prev_loss = loss[i]
        prev_delta_cwnd = d

        cur = nxt

    cwnd_pred = np.array(cwnd_pred[:-1], dtype=float)  # align with current-step cwnd
    yhat = np.array(yhat, dtype=float)

    m = min(len(cwnd_real), len(cwnd_pred), len(t_test), len(y_test), len(yhat))

    metrics = {
        "n_test_points": int(len(df)),
        "features": FEATURE_COLS,
        "rmse_cwnd": safe_rmse(cwnd_real[:m], cwnd_pred[:m]),
        "mae_cwnd": safe_mae(cwnd_real[:m], cwnd_pred[:m]),
        "rmse_delta": safe_rmse(y_test[:m], yhat[:m]),
        "mae_delta": safe_mae(y_test[:m], yhat[:m]),
    }

    out_dir = os.path.join(out_root, f"server_{server_id}")
    os.makedirs(out_dir, exist_ok=True)

    plot_pdf = os.path.join(out_dir, "cwnd_pred_vs_real.pdf")
    plot_cwnd(
        t_test[:m],
        cwnd_real[:m],
        cwnd_pred[:m],
        plot_pdf,
        title=f"Q3 global-model cwnd prediction ({server_id})"
    )

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    algo_path = os.path.join(out_dir, "cwnd_update_algorithm.txt")
    write_update_algorithm_text(algo_path, model_pack["metrics"])

    return Q3Result(
        server_id=server_id,
        trace_path=trace_path,
        out_dir=out_dir,
        metrics=metrics
    )

def fit_global_model(
    train_ids: List[str],
    latest: Dict[str, str],
    alpha: float = 0.02,
    beta: float = 1.0,
    ridge_alpha: float = 1.0
):
    """
    Build one global training dataset from multiple servers,
    fit ONE weighted model using eta-based sample weights.
    """
    from sklearn.linear_model import Ridge

    dfs = []
    for sid in train_ids:
        trace_path = latest[sid]
        df = build_dataset_from_trace(trace_path, alpha=alpha, beta=beta)
        df["server_id"] = sid
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True)

    X_train = train_df[FEATURE_COLS].values.astype(float)
    y_train = train_df["delta_cwnd"].values.astype(float)

    # standardize for more stable global training
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    X_train_std = (X_train - mu) / sigma

    # eta-based weights: larger eta => more desirable decision outcome
    eta = train_df["eta_next"].values.astype(float)

    # normalize to positive weights
    lo = np.percentile(eta, 10)
    hi = np.percentile(eta, 90)
    if hi - lo < 1e-9:
        weights = np.ones_like(eta)
    else:
        weights = (eta - lo) / (hi - lo)
        weights = np.clip(weights, 0.1, 2.0)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_std, y_train, sample_weight=weights)

    # convert coefficients back to raw feature space for interpretation / algorithm text
    raw_coef = model.coef_ / sigma
    raw_intercept = float(model.intercept_ - np.sum((mu / sigma) * model.coef_))

    metrics = {
        "n_train_points": int(len(train_df)),
        "n_train_servers": int(len(train_ids)),
        "features": FEATURE_COLS,
        "alpha": float(alpha),
        "beta": float(beta),
        "ridge_alpha": float(ridge_alpha),
        "coef": {FEATURE_COLS[i]: float(raw_coef[i]) for i in range(len(FEATURE_COLS))},
        "intercept": raw_intercept,
    }

    return {
        "model": model,
        "mu": mu,
        "sigma": sigma,
        "metrics": metrics,
    }

def plot_cwnd(t_test: np.ndarray, cwnd_real: np.ndarray, cwnd_pred: np.ndarray, out_pdf: str, title: str):
    plt.figure()
    m = min(len(t_test), len(cwnd_pred), len(cwnd_real))
    plt.plot(t_test[:m], cwnd_real[:m], label="real cwnd")
    plt.plot(t_test[:m], cwnd_pred[:m], label="predicted cwnd")
    plt.xlabel("time (s)")
    plt.ylabel("cwnd")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

def write_update_algorithm_text(out_txt: str, metrics: Dict):
    coef = metrics["coef"]
    b0 = metrics["intercept"]

    lines = []
    lines.append("Congestion Window Update Algorithm (learned + safe-guards)\n")
    lines.append("Features at time t:\n")
    lines.append("  goodput(t), rtt(t), loss(t), rttvar(t), retrans_delta(t), cwnd(t)\n")
    lines.append("  goodput(t-1), rtt(t-1), loss(t-1), delta_cwnd(t-1)\n")

    lines.append("\nLearned linear model for delta_cwnd:\n")
    lines.append(f"  delta = {b0:.6f}")
    for k, v in coef.items():
        lines.append(f"        + ({v:.6f}) * {k}")

    lines.append("\nUpdate rule per sampling interval:\n")
    lines.append("  if loss(t) > 0:\n")
    lines.append("      cwnd <- max(1, 0.5 * cwnd)")
    lines.append("  else:\n")
    lines.append("      cwnd <- max(1, cwnd + clamp(delta, -0.5*cwnd, 0.5*cwnd))")

    lines.append("\nNotes:\n")
    lines.append("- retrans_delta is derived from cumulative retrans counters.")
    lines.append("- loss is used directly as provided by the trace (not differenced).")
    lines.append("- lag features capture temporal dependence in TCP cwnd evolution.")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():

    # path settings
    q2_root = r"results/q2"
    out_root = r"results/q3"

    print("[Q3] scanning:", q2_root)

    traces = find_all_trace_csv(q2_root)

    if not traces:
        raise SystemExit(f"No trace.csv found under: {q2_root}")

    latest = pick_latest_trace_per_server(traces)

    server_ids = sorted(latest.keys())

    print("[Q3] total servers found:", len(server_ids))

    # -------------------------
    # choose 5 servers as test
    # -------------------------

    random.seed(42)
    test_ids = random.sample(server_ids, 5)

    train_ids = [sid for sid in server_ids if sid not in test_ids]

    print("[Q3] train servers:", train_ids)
    print("[Q3] test servers:", test_ids)

    os.makedirs(out_root, exist_ok=True)

    # -------------------------
    # train global model
    # -------------------------

    print("[Q3] training global model...")

    model_pack = fit_global_model(train_ids, latest)

    # save global model description
    with open(os.path.join(out_root, "global_model.json"), "w", encoding="utf-8") as f:
        json.dump(model_pack["metrics"], f, indent=2)

    # -------------------------
    # run prediction on test servers
    # -------------------------

    summary = {}

    for sid in test_ids:

        trace_path = latest[sid]

        print(f"[Q3] testing server {sid}")

        res = predict_on_test_server(
            sid,
            trace_path,
            model_pack,
            out_root
        )

        summary[sid] = {
            "trace": res.trace_path,
            "out_dir": res.out_dir,
            "rmse_cwnd": res.metrics["rmse_cwnd"],
            "mae_cwnd": res.metrics["mae_cwnd"],
        }

    # -------------------------
    # save summary
    # -------------------------

    summary_path = os.path.join(out_root, "summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Q3] done.")
    print("[Q3] outputs:", out_root)
    print("[Q3] summary:", summary_path)


if __name__ == "__main__":
    main()