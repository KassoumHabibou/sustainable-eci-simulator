"""
complexity.py — RCA / ECI / PCI / relatedness, vectorized
=========================================================
Mathematically identical to the authors' `ecioptimization.rca` and
`ecioptimization.cplex_rank`, but the O(P²·C) Python loops are replaced
with matrix operations (~100x faster). The eigenvector extraction and
sign convention are IDENTICAL to the authors' code, so the outputs match.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig


def rca(X):
    """Revealed Comparative Advantage:  R_cp = (X_cp * X) / (X_c * X_p)."""
    X = np.asarray(X, dtype=float)
    Xc = X.sum(axis=1, keepdims=True)
    Xp = X.sum(axis=0)
    return (X * X.sum()) / (Xc * Xp)


def proximity(Mcp):
    """Product-proximity matrix:  PHI_pp' = (M'M)_pp' / max(k_p, k_p')."""
    Kp0 = Mcp.sum(axis=0)
    co  = Mcp.T @ Mcp
    return co / np.maximum.outer(Kp0, Kp0)


def relatedness(Mcp, PHIpp):
    """Relatedness (density):  ω_cp = Σ_p' M_cp' PHI_pp' / Σ_p' PHI_pp'."""
    return (Mcp @ PHIpp) / PHIpp.sum(axis=1)


def cplex_rank(RCAcp, countries, products):
    """
    Vectorized equivalent of the authors' `cplex_rank`.

    Returns (CountryRankings, ProductRankings, Relatedness, PHIpp).
    NOTE: it additionally returns PHIpp (the authors return OpportunityGain
    in that slot; we don't need it and PHIpp is reused downstream).
    """
    RCAcp = np.nan_to_num(np.asarray(RCAcp, dtype=float))
    Mcp = (RCAcp >= 1).astype(float)

    Kp0 = Mcp.sum(axis=0)          # ubiquity
    Kc0 = Mcp.sum(axis=1)          # diversity

    PHIpp = proximity(Mcp)
    Rel   = relatedness(Mcp, PHIpp)

    # Country-country and product-product transition matrices
    #   Mcc[i,j] = Σ_p M_ip M_jp / (Kp0_p · Kc0_i)
    #   Mpp[i,j] = Σ_c M_ci M_cj / (Kc0_c · Kp0_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        Mcc = (Mcp / Kc0[:, None]) @ (Mcp / Kp0[None, :]).T
        Mpp = (Mcp / Kp0[None, :]).T @ (Mcp / Kc0[:, None])
    Mcc = np.nan_to_num(Mcc)
    Mpp = np.nan_to_num(Mpp)

    # Second eigenvector — same extraction as the authors (column index 1
    # of scipy.linalg.eig output), same z-scoring, same sign correction.
    Vc = np.real(eig(Mcc)[1][:, 1])
    Vp = np.real(eig(Mpp)[1][:, 1])
    ECI = (Vc - Vc.mean()) / Vc.std()
    PCI = (Vp - Vp.mean()) / Vp.std()

    Kc1 = (Mcp @ Kp0) / Mcp.sum(axis=1)          # avg ubiquity of exports
    if np.corrcoef(ECI, Kc1)[0, 1] > 0:
        ECI *= -1
    if np.corrcoef(PCI, Kp0)[0, 1] > 0:
        PCI *= -1

    CountryRankings = pd.DataFrame({"Country": countries, "ECI": ECI})
    ProductRankings = pd.DataFrame({"Product": products, "PCI": PCI})
    CountryRankings["COI"] = (Rel * (1 - Mcp)) @ PCI
    return CountryRankings, ProductRankings, Rel, PHIpp


def eci_not_normalized(Mcp, PCI):
    """Average PCI of the activities a location specializes in (mean-PCI scale)."""
    div = Mcp.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (Mcp @ PCI) / div
    out[div == 0] = np.nan
    return out
