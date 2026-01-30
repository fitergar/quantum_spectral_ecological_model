# src/predict_gray_local.py
from __future__ import annotations

import argparse
import json
import pickle as pkl
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, inv as sparse_inv

from src.utils import dataset_paths, step, die, ensure_dir
from src.featurize import _resolve_cols as _cols, build_Tdata_grey25
from src.linalg import laplacian_4nbrs, minevec_iter
from src.models import MLPDriver, MLPModulatedGrey25Local


# ----------------- args saving -----------------

def save_args(out_dir: Path, args: argparse.Namespace):
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
    }
    (out_dir / "args_predict_grey_local.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


# ----------------- ground state -----------------

def solve_ground_state(Hs: np.ndarray, iters: int, pow_k: int):
    """
    Inverse-power method on Hs:
      - normalize Hs to Hn for numerical stability
      - compute Hi = inv(Hn), then Hi^pow_k
      - iterate smallest-eigenvector of Hi^pow_k
    Returns:
      Phi (unit norm), Rayleigh lambda for Hs, residual norm, relative residual
    """
    N = Hs.shape[0]

    # normalize for inversion stability
    fro = np.linalg.norm(Hs, "fro") + 1e-60
    Hn = Hs / fro

    Hi = sparse_inv(csc_matrix(Hn))
    Hi_pow = Hi
    for _ in range(max(1, pow_k) - 1):
        Hi_pow = Hi_pow @ Hi

    Phi = minevec_iter(Hi_pow.toarray(), np.random.rand(N), iters=iters)
    Phi = Phi / (np.linalg.norm(Phi) + 1e-60)

    # ---- convergence metrics computed on ORIGINAL Hs ----
    HsPhi = Hs @ Phi
    lam = float(Phi @ HsPhi)  # Rayleigh quotient (since ||Phi||=1)
    r = HsPhi - lam * Phi
    res = float(np.linalg.norm(r))
    relres = float(res / (np.linalg.norm(HsPhi) + 1e-60))

    return Phi, lam, res, relres



# ----------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Local GREY25+Z predictions with frozen driver and neighborhood averaging"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--cuda", action="store_true")

    p.add_argument("--step", type=int, default=10)
    p.add_argument("--shift", type=float, default=5000.0)
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--pow", type=int, default=10)

    p.add_argument("--neigh-file", default="VecindadesLinf.pkl")
    p.add_argument("--out-csv", default="pred_grey_localavg.csv")
    p.add_argument("--save-args", action="store_true")
    p.add_argument("--conv-print-every", type=int, default=25,
                   help="Print inverse-power convergence every N neighborhoods (0 disables).")
    p.add_argument("--conv-warn-relres", type=float, default=1e-6,
                   help="Warn if relative residual exceeds this threshold.")


    return p.parse_args()


# ----------------- main -----------------

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    step(f"Device: {device}")

    paths = dataset_paths(args.dataset)
    out_dir = Path(paths["out_dir"])
    ensure_dir(out_dir)
    if args.save_args:
        save_args(out_dir, args)

    prepared = paths["prepared_csv"]
    if not prepared.exists():
        die("prepared.csv not found. Run prepare.py first.")

    df_all = pd.read_csv(prepared)
    C = _cols(df_all)

    idcol = C["id"]
    geom_cols = [c for c in (C["id"], C["x"], C["y"], C["drio"]) if c in df_all.columns]

    # ---------- neighborhoods ----------
    neigh_path = out_dir / args.neigh_file
    if not neigh_path.exists():
        die("Neighborhoods file not found. Run neighborhoods.py first.")

    with open(neigh_path, "rb") as f:
        neighborhoods = pkl.load(f)

    if not neighborhoods:
        die("Neighborhood list is empty.")

    step(f"Loaded {len(neighborhoods)} neighborhoods")

    # ---------- driver ----------
    dstate = torch.load(out_dir / "model_driver.pt", map_location=device)
    dmeta = dstate["meta"]

    driver = MLPDriver(
        dmeta["poly_degree"],
        *dmeta["hidden"],
    ).to(device)
    driver.load_state_dict(dstate["state_dict"])
    driver.eval()
    for p in driver.parameters():
        p.requires_grad = False

    # ---------- grey modulator ----------
    gstate = torch.load(out_dir / "model_grey.pt", map_location=device)
    gmeta = gstate["meta"]

    drio_max = float(df_all[C["drio"]].max())
    d0_T = gmeta["d0_meters"] / drio_max
    k_T = gmeta["k_per_meter"] * drio_max

    grey = MLPModulatedGrey25Local(
        gmeta["train_args"]["mod_poly"],
        gmeta["train_args"]["mod_h1"],
        gmeta["train_args"]["mod_h2"],
        gmeta["train_args"]["mod_h3"],
        driver,
        a=gmeta["a"],
        b=gmeta["b"],
        d0=d0_T,
        k=k_T,
        ngrey=25,
        grey_start=6,
        use_z=True,
        z_index=4,
    ).to(device)

    grey.load_state_dict(gstate["state_dict"])
    grey.eval()

    # ---------- accumulation ----------
    sum_prob = defaultdict(float)
    count = defaultdict(int)

    for k, nb in enumerate(neighborhoods):
        step(f"[{k+1}/{len(neighborhoods)}] Neighborhood N={len(nb)}")

        T_nb = build_Tdata_grey25(nb, df_all, n_grey=25).to(torch.float32).to(device)

        with torch.no_grad():
            V_nb = grey(T_nb).cpu().numpy().astype(np.float64)

        L = laplacian_4nbrs(nb, step=args.step).numpy().astype(np.float64)
        H = L + np.diag(V_nb)
        Hs = H + np.eye(len(H)) * args.shift

        Phi, lam, res, relres = solve_ground_state(Hs, iters=args.iters, pow_k=args.pow)
        Prob = Phi ** 2

        if args.conv_print_every and ((k + 1) % args.conv_print_every == 0 or (k == 0)):
            msg = f"  invpow: lam≈{lam:.6e} | res={res:.3e} | relres={relres:.3e}"
            if relres > args.conv_warn_relres:
                msg += "  [WARN]"
            step(msg)

        
        for cid, p in zip(nb[idcol].to_numpy(), Prob):
            sum_prob[int(cid)] += float(p)
            count[int(cid)] += 1

    # ---------- global average ----------
    df_geom = df_all[geom_cols].drop_duplicates(subset=idcol)
    df_geom = df_geom[df_geom[idcol].isin(sum_prob)]

    df_geom["Probav"] = df_geom[idcol].map(sum_prob) / df_geom[idcol].map(count)
    df_geom["Probav"] /= df_geom["Probav"].sum()

    # ---------- population calibration ----------
    vcol = C["v"]
    if vcol in df_all.columns:
        train = df_all[[idcol, vcol]].dropna()
        if not train.empty:
            m = df_geom.merge(train, on=idcol)
            scale = m[vcol].sum() / m["Probav"].sum()
            df_geom["Pop_est"] = df_geom["Probav"] * scale
            step(f"Population calibration scale={scale:.6e}")

    # ---------- save ----------
    out_csv = out_dir / args.out_csv
    cols = geom_cols + ["Probav"] + (["Pop_est"] if "Pop_est" in df_geom.columns else [])
    df_geom[cols].to_csv(out_csv, index=False)
    step(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
