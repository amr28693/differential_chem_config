#!/usr/bin/env python3
"""
polarizability_curvature_analysis.py
=====================================
Self-contained analysis correlating the second difference (curvature) of the
CFC configuration field Φ with experimental atomic static dipole polarizabilities.

Data sources
------------
- First ionization energies: NIST Atomic Spectra Database v5.11 (2023)
- Covalent radii: Cordero et al., Dalton Trans 2832 (2008)
- Atomic polarizabilities: Schwerdtfeger & Nagle, Mol. Phys. 117, 1200 (2019),
  2025 update from https://ctcp.massey.ac.nz/2025Tablepol.pdf
  (recommended values in atomic units, converted to Å³ where noted)

Methodology
-----------
Matches the published manuscript:
  Φ_i = IE_zscore_i + λ * R_zscore_i    (λ = 0.5)
  ∇²Φ[i] = Φ[i+1] + Φ[i-1] - 2Φ[i]   (second difference along Z)

Correlations reported with BCa bootstrap 95% CIs (10,000 resamples).

Author: Anderson M. Rodriguez
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ============================================================
# EMBEDDED DATA: Z=1 to Z=90
# IE (eV) from NIST ASD v5.11;  R (pm) from Cordero et al. 2008
# Polarizabilities (atomic units) from Schwerdtfeger & Nagle 2019/2025
#   recommended values. 1 a.u. = 0.148185 Å³
# ============================================================
# ============================================================

ELEMENTS = {
    # Z: (symbol, IE_eV, R_pm, alpha_au)
    #   alpha_au = None if no recommended value available
    1:  ("H",  13.598, 31,   4.507),
    2:  ("He", 24.587, 28,   1.384),
    3:  ("Li",  5.392, 128, 164.11),
    4:  ("Be",  9.323, 96,  37.74),
    5:  ("B",   8.298, 84,  20.5),
    6:  ("C",  11.260, 73,  11.3),
    7:  ("N",  14.534, 71,   7.4),
    8:  ("O",  13.618, 66,   5.3),
    9:  ("F",  17.423, 57,   3.74),
    10: ("Ne", 21.565, 58,   2.661),
    11: ("Na",  5.139, 166, 162.7),
    12: ("Mg",  7.646, 141,  71.2),
    13: ("Al",  5.986, 121,  57.8),
    14: ("Si",  8.152, 111,  37.3),
    15: ("P",  10.487, 107,  25.0),
    16: ("S",  10.360, 105,  19.4),
    17: ("Cl", 12.968, 102,  14.6),
    18: ("Ar", 15.760,  106, 11.083),
    19: ("K",   4.341, 203, 289.7),
    20: ("Ca",  6.113, 176, 160.8),
    21: ("Sc",  6.562, 170,  97.0),
    22: ("Ti",  6.828, 160,  87.0),
    23: ("V",   6.746, 153,  87.0),
    24: ("Cr",  6.767, 139,  83.0),
    25: ("Mn",  7.434, 150,  68.0),
    26: ("Fe",  7.902, 142,  62.0),
    27: ("Co",  7.881, 138,  55.0),
    28: ("Ni",  7.640, 124,  49.0),
    29: ("Cu",  7.726, 132,  46.5),
    30: ("Zn",  9.394, 122,  38.67),
    31: ("Ga",  5.999, 122,  50.0),
    32: ("Ge",  7.900, 120,  40.0),
    33: ("As",  9.789, 119,  30.0),
    34: ("Se",  9.752, 120,  28.9),
    35: ("Br", 11.814, 120,  21.0),
    36: ("Kr", 14.000, 116,  16.78),
    37: ("Rb",  4.177, 220, 319.8),
    38: ("Sr",  5.695, 195, 197.2),
    39: ("Y",   6.217, 190, 162.0),
    40: ("Zr",  6.634, 175, 112.0),
    41: ("Nb",  6.759, 164,  98.0),
    42: ("Mo",  7.092, 154,  87.0),
    43: ("Tc",  7.119, 147,  79.0),
    44: ("Ru",  7.361, 146,  72.0),
    45: ("Rh",  7.459, 142,  66.0),
    46: ("Pd",  8.337, 139,  26.14),
    47: ("Ag",  7.576, 145,  55.0),
    48: ("Cd",  8.994, 144,  46.0),
    49: ("In",  5.786, 142,  65.0),
    50: ("Sn",  7.344, 139,  53.0),
    51: ("Sb",  8.608, 139,  43.0),
    52: ("Te",  9.010, 138,  38.0),
    53: ("I",  10.451, 139,  32.9),
    54: ("Xe", 12.130, 140,  27.32),
    55: ("Cs",  3.894, 244, 400.9),
    56: ("Ba",  5.212, 215, 272.0),
    57: ("La",  5.577, 207, 215.0),
    58: ("Ce",  5.539, 204, 205.0),
    59: ("Pr",  5.473, 203, 216.0),
    60: ("Nd",  5.525, 201, 208.0),
    61: ("Pm",  5.582, 199, 200.0),
    62: ("Sm",  5.644, 198, 192.0),
    63: ("Eu",  5.670, 198, 184.0),
    64: ("Gd",  6.150, 196, 158.0),
    65: ("Tb",  5.864, 194, 170.0),
    66: ("Dy",  5.939, 192, 163.0),
    67: ("Ho",  6.022, 192, 155.0),
    68: ("Er",  6.108, 189, 150.0),
    69: ("Tm",  6.184, 190, 144.0),
    70: ("Yb",  6.254, 187, 139.0),
    71: ("Lu",  5.426, 187, 137.0),
    72: ("Hf",  6.825, 175, 103.0),
    73: ("Ta",  7.550, 170,  74.0),
    74: ("W",   7.864, 162,  68.0),
    75: ("Re",  7.834, 151,  62.0),
    76: ("Os",  8.438, 144,  57.0),
    77: ("Ir",  8.967, 141,  54.0),
    78: ("Pt",  8.959, 136,  48.0),
    79: ("Au",  9.226, 136,  36.0),
    80: ("Hg", 10.437, 132,  33.91),
    81: ("Tl",  6.108, 145,  50.0),
    82: ("Pb",  7.417, 146,  47.0),
    83: ("Bi",  7.286, 148,  48.0),
    84: ("Po",  8.414, 140,  44.0),
    85: ("At",  9.318, 150,  42.0),
    86: ("Rn", 10.749, 150,  35.0),
    87: ("Fr",  4.073, 260, None),  # no recommended alpha
    88: ("Ra",  5.278, 221, None),  # no recommended alpha
    89: ("Ac",  5.380, 215, None),  # no recommended alpha
    90: ("Th",  6.307, 206, None),  # no recommended alpha
}

# ============================================================
# ============================================================
# FIELD CONSTRUCTION (exactly as in manuscript)
# ============================================================
# ============================================================

def build_field(lam=0.5):
    """Build Φ = IE_zscore + λ * R_zscore for Z=1..90."""
    Zs = sorted(ELEMENTS.keys())
    ie_arr = np.array([ELEMENTS[z][1] for z in Zs])
    r_arr  = np.array([ELEMENTS[z][2] for z in Zs], dtype=float)

    ie_z = (ie_arr - ie_arr.mean()) / ie_arr.std()
    r_z  = (r_arr  - r_arr.mean())  / r_arr.std()

    phi = ie_z + lam * r_z
    return Zs, phi


def second_difference(Zs, phi):
    """
    ∇²Φ[i] = Φ[i+1] + Φ[i-1] - 2Φ[i]
    Defined for i = 1..N-2 (i.e., Z=2..89).
    """
    sd = {}
    for idx in range(1, len(Zs) - 1):
        z = Zs[idx]
        sd[z] = phi[idx + 1] + phi[idx - 1] - 2.0 * phi[idx]
    return sd

# ============================================================
# ============================================================
# BCa BOOTSTRAP 
# ============================================================
# ============================================================

def bca_ci(x, y, corr_func, n_boot=10000, alpha=0.05, seed=42):
    """BCa bootstrap confidence interval for a correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    theta_hat = corr_func(x, y)[0]

    # Bootstrap distribution
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[b] = corr_func(x[idx], y[idx])[0]

    # Bias correction
    z0 = _norm_ppf(np.mean(boot < theta_hat))

    # Acceleration (jackknife)
    jack = np.empty(n)
    for i in range(n):
        mask = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        jack[i] = corr_func(x[mask], y[mask])[0]
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5
    a_hat = num / den if den != 0 else 0.0

    # Adjusted percentiles
    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)

    def adj(z):
        return _norm_cdf(z0 + (z0 + z) / (1 - a_hat * (z0 + z)))

    p_lo = max(adj(z_lo), 0.5 / n_boot)
    p_hi = min(adj(z_hi), 1 - 0.5 / n_boot)

    ci_lo = np.percentile(boot, 100 * p_lo)
    ci_hi = np.percentile(boot, 100 * p_hi)
    return theta_hat, ci_lo, ci_hi


def _norm_ppf(p):
    from scipy.special import erfinv
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sqrt(2) * erfinv(2 * p - 1)


def _norm_cdf(x):
    from scipy.special import erfc
    return 0.5 * erfc(-x / np.sqrt(2))

# ============================================================
# ============================================================
# MAIN ANALYSIS
# ============================================================
# ============================================================

def main():
    lam = 0.5
    Zs, phi = build_field(lam)
    sd = second_difference(Zs, phi)

    # Collect elements with both ∇²Φ and polarizability
    z_list, sym_list, lap_list, alpha_list = [], [], [], []
    for z in sorted(sd.keys()):
        sym, ie, r, alpha = ELEMENTS[z]
        if alpha is not None and np.isfinite(sd[z]):
            z_list.append(z)
            sym_list.append(sym)
            lap_list.append(sd[z])
            alpha_list.append(alpha)

    lap = np.array(lap_list)
    alp = np.array(alpha_list)
    N = len(lap)

    print("=" * 65)
    print("POLARIZABILITY-CURVATURE CORRELATION ANALYSIS")
    print("=" * 65)
    print(f"  Field: Φ = IE_zscore + {lam} * R_zscore")
    print(f"  Curvature: ∇²Φ[i] = Φ[i+1] + Φ[i-1] - 2Φ[i]")
    print(f"  Polarizabilities: Schwerdtfeger & Nagle (2019/2025)")
    print(f"  N = {N} elements with both ∇²Φ and α")
    print(f"  Bootstrap: BCa, 10,000 resamples, seed=42")
    print()

    # --- Raw polarizability ---
    r_p, p_p = pearsonr(lap, alp)
    rho_s, p_s = spearmanr(lap, alp)

    # BCa CIs
    r_hat, r_lo, r_hi = bca_ci(lap, alp, pearsonr)
    rho_hat, rho_lo, rho_hi = bca_ci(lap, alp, spearmanr)

    print(f"--- Correlations: ∇²Φ vs α (raw polarizability) ---")
    print(f"  Pearson  r  = {r_p:.3f}  95% CI: [{r_lo:.3f}, {r_hi:.3f}]  p = {p_p:.2e}")
    print(f"  Spearman ρ  = {rho_s:.3f}  95% CI: [{rho_lo:.3f}, {rho_hi:.3f}]  p = {p_s:.2e}")
    print()

    # --- Log polarizability (common for alpha which spans orders of magnitude) ---
    log_alp = np.log(alp)
    r_log, p_log = pearsonr(lap, log_alp)
    rho_log, p_log_s = spearmanr(lap, log_alp)

    r_hat_log, r_lo_log, r_hi_log = bca_ci(lap, log_alp, pearsonr)

    print(f"--- Correlations: ∇²Φ vs ln(α) ---")
    print(f"  Pearson  r  = {r_log:.3f}  95% CI: [{r_lo_log:.3f}, {r_hi_log:.3f}]  p = {p_log:.2e}")
    print(f"  Spearman ρ  = {rho_log:.3f}  (same as raw, rank-invariant)")
    print()

    # --- Cube-root transform (motivated by α ∝ η^{-3} relationship) ---
    cbrt_alp = np.cbrt(alp)
    r_cbrt, p_cbrt = pearsonr(lap, cbrt_alp)
    r_hat_cbrt, r_lo_cbrt, r_hi_cbrt = bca_ci(lap, cbrt_alp, pearsonr)

    print(f"--- Correlations: ∇²Φ vs α^(1/3) (cube-root) ---")
    print(f"  Pearson  r  = {r_cbrt:.3f}  95% CI: [{r_lo_cbrt:.3f}, {r_hi_cbrt:.3f}]  p = {p_cbrt:.2e}")
    print()

    # --- Inverse cube-root transform (Blair 2014: η ∝ α^{-1/3}) ---
    inv_cbrt = alp ** (-1.0/3.0)
    r_inv, p_inv = pearsonr(lap, inv_cbrt)
    r_hat_inv, r_lo_inv, r_hi_inv = bca_ci(lap, inv_cbrt, pearsonr)

    print(f"--- Correlations: ∇²Φ vs α^(-1/3) (inv cube-root, Blair 2014) ---")
    print(f"  Pearson  r  = {r_inv:.3f}  95% CI: [{r_lo_inv:.3f}, {r_hi_inv:.3f}]  p = {p_inv:.2e}")
    print()

    # --- Element listing ---
    print(f"--- Element data (N={N}) ---")
    print(f"{'Z':>3}  {'Sym':<3}  {'∇²Φ':>8}  {'α (a.u.)':>10}  {'ln(α)':>8}")
    print("-" * 45)
    for z, sym, l, a in zip(z_list, sym_list, lap_list, alpha_list):
        print(f"{z:>3}  {sym:<3}  {l:>8.4f}  {a:>10.2f}  {np.log(a):>8.3f}")

    # --- Generate scatter plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # (a) ∇²Φ vs α
        ax = axes[0]
        ax.scatter(lap, alp, c="steelblue", s=40, edgecolor="white", linewidth=0.5, zorder=3)
        for z, sym, l, a in zip(z_list, sym_list, lap_list, alpha_list):
            ax.annotate(sym, (l, a), fontsize=6, alpha=0.7, ha="center", va="bottom")
        # regression line
        m, b = np.polyfit(lap, alp, 1)
        xx = np.linspace(lap.min(), lap.max(), 100)
        ax.plot(xx, m * xx + b, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(r"$\nabla^2\Phi$", fontsize=12)
        ax.set_ylabel(r"$\alpha$ (a.u.)", fontsize=12)
        ax.set_title(
            f"(a) Curvature vs polarizability\n"
            f"Pearson r = {r_p:.3f}, p = {p_p:.1e}, N = {N}",
            fontsize=10
        )
        ax.grid(alpha=0.2)

        # (b) ∇²Φ vs ln(α)
        ax = axes[1]
        ax.scatter(lap, log_alp, c="darkorange", s=40, edgecolor="white", linewidth=0.5, zorder=3)
        for z, sym, l, a in zip(z_list, sym_list, lap_list, alpha_list):
            ax.annotate(sym, (l, np.log(a)), fontsize=6, alpha=0.7, ha="center", va="bottom")
        m2, b2 = np.polyfit(lap, log_alp, 1)
        ax.plot(xx, m2 * xx + b2, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(r"$\nabla^2\Phi$", fontsize=12)
        ax.set_ylabel(r"$\ln(\alpha)$", fontsize=12)
        ax.set_title(
            f"(b) Curvature vs ln(polarizability)\n"
            f"Pearson r = {r_log:.3f}, p = {p_log:.1e}, N = {N}",
            fontsize=10
        )
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig("fig_polarizability_curvature.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("fig_polarizability_curvature.png", dpi=300, bbox_inches="tight")
        print(f"\nFigures saved: fig_polarizability_curvature.pdf / .png")

    except ImportError:
        print("\n(matplotlib not available; skipping plot)")

    print("\nDone.")


if __name__ == "__main__":
    main()
