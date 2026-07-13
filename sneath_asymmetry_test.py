#!/usr/bin/env python3
"""
sneath_asymmetry_test.py

Tests whether the configuration field Phi exhibits a directional
asymmetry in elemental similarity noted by Sneath (oral remarks,
ca. 2004; foundation: Found. Chem. 2, 237, 2000): the (n+1, g+1)
neighbor tends to be more similar than the (n+1, g-1) neighbor.

Anderson M. Rodriguez
"""

import numpy as np
from scipy.stats import wilcoxon

# ================================================================
# ELEMENT DATA: Z → (symbol, IE_eV, R_pm)
# IE: NIST ASD v5.11 (2023);  R: Cordero et al. 2008
# Values identical to polarizability_curvature_analysis.py and
# cfc_master_validation_v5.py
# ================================================================

ELEMENTS = {
     1: ("H",  13.598,  31),
     2: ("He", 24.587,  28),
     3: ("Li",  5.392, 128),
     4: ("Be",  9.323,  96),
     5: ("B",   8.298,  84),
     6: ("C",  11.260,  73),
     7: ("N",  14.534,  71),
     8: ("O",  13.618,  66),
     9: ("F",  17.423,  57),
    10: ("Ne", 21.565,  58),
    11: ("Na",  5.139, 166),
    12: ("Mg",  7.646, 141),
    13: ("Al",  5.986, 121),
    14: ("Si",  8.152, 111),
    15: ("P",  10.487, 107),
    16: ("S",  10.360, 105),
    17: ("Cl", 12.968, 102),
    18: ("Ar", 15.760, 106),
    19: ("K",   4.341, 203),
    20: ("Ca",  6.113, 176),
    21: ("Sc",  6.562, 170),
    22: ("Ti",  6.828, 160),
    23: ("V",   6.746, 153),
    24: ("Cr",  6.767, 139),
    25: ("Mn",  7.434, 150),
    26: ("Fe",  7.902, 142),
    27: ("Co",  7.881, 138),
    28: ("Ni",  7.640, 124),
    29: ("Cu",  7.726, 132),
    30: ("Zn",  9.394, 122),
    31: ("Ga",  5.999, 122),
    32: ("Ge",  7.900, 120),
    33: ("As",  9.789, 119),
    34: ("Se",  9.752, 120),
    35: ("Br", 11.814, 120),
    36: ("Kr", 14.000, 116),
    37: ("Rb",  4.177, 220),
    38: ("Sr",  5.695, 195),
    39: ("Y",   6.217, 190),
    40: ("Zr",  6.634, 175),
    41: ("Nb",  6.759, 164),
    42: ("Mo",  7.092, 154),
    43: ("Tc",  7.119, 147),
    44: ("Ru",  7.361, 146),
    45: ("Rh",  7.459, 142),
    46: ("Pd",  8.337, 139),
    47: ("Ag",  7.576, 145),
    48: ("Cd",  8.994, 144),
    49: ("In",  5.786, 142),
    50: ("Sn",  7.344, 139),
    51: ("Sb",  8.608, 139),
    52: ("Te",  9.010, 138),
    53: ("I",  10.451, 139),
    54: ("Xe", 12.130, 140),
    55: ("Cs",  3.894, 244),
    56: ("Ba",  5.212, 215),
    57: ("La",  5.577, 207),
    58: ("Ce",  5.539, 204),
    59: ("Pr",  5.473, 203),
    60: ("Nd",  5.525, 201),
    61: ("Pm",  5.582, 199),
    62: ("Sm",  5.644, 198),
    63: ("Eu",  5.670, 198),
    64: ("Gd",  6.150, 196),
    65: ("Tb",  5.864, 194),
    66: ("Dy",  5.939, 192),
    67: ("Ho",  6.022, 192),
    68: ("Er",  6.108, 189),
    69: ("Tm",  6.184, 190),
    70: ("Yb",  6.254, 187),
    71: ("Lu",  5.426, 187),
    72: ("Hf",  6.825, 175),
    73: ("Ta",  7.550, 170),
    74: ("W",   7.864, 162),
    75: ("Re",  7.834, 151),
    76: ("Os",  8.438, 144),
    77: ("Ir",  8.967, 141),
    78: ("Pt",  8.959, 136),
    79: ("Au",  9.226, 136),
    80: ("Hg", 10.437, 132),
    81: ("Tl",  6.108, 145),
    82: ("Pb",  7.417, 146),
    83: ("Bi",  7.286, 148),
    84: ("Po",  8.414, 140),
    85: ("At",  9.318, 150),
    86: ("Rn", 10.749, 150),
    87: ("Fr",  4.073, 260),
    88: ("Ra",  5.278, 221),
    89: ("Ac",  5.380, 215),
    90: ("Th",  6.307, 206),
}

# ================================================================
# LATTICE POSITIONS: Z → (period, group) on the 7 × 18 table
#
# f-block elements (Z = 57–71 and Z = 89–90) all map to group 3
# in the manuscript's convention, creating multi-occupancy at
# (6, 3) and (7, 3). These positions are excluded from the
# diagonal comparison; the elements remain in the z-score pool.
# ================================================================

POSITION = {
     1: (1,  1),    2: (1, 18),
     3: (2,  1),    4: (2,  2),    5: (2, 13),    6: (2, 14),
     7: (2, 15),    8: (2, 16),    9: (2, 17),   10: (2, 18),
    11: (3,  1),   12: (3,  2),   13: (3, 13),   14: (3, 14),
    15: (3, 15),   16: (3, 16),   17: (3, 17),   18: (3, 18),
    19: (4,  1),   20: (4,  2),   21: (4,  3),   22: (4,  4),
    23: (4,  5),   24: (4,  6),   25: (4,  7),   26: (4,  8),
    27: (4,  9),   28: (4, 10),   29: (4, 11),   30: (4, 12),
    31: (4, 13),   32: (4, 14),   33: (4, 15),   34: (4, 16),
    35: (4, 17),   36: (4, 18),
    37: (5,  1),   38: (5,  2),   39: (5,  3),   40: (5,  4),
    41: (5,  5),   42: (5,  6),   43: (5,  7),   44: (5,  8),
    45: (5,  9),   46: (5, 10),   47: (5, 11),   48: (5, 12),
    49: (5, 13),   50: (5, 14),   51: (5, 15),   52: (5, 16),
    53: (5, 17),   54: (5, 18),
    55: (6,  1),   56: (6,  2),
    # Z = 57–71 (La–Lu): group 3, period 6 — multi-occupancy, excluded
    72: (6,  4),   73: (6,  5),   74: (6,  6),   75: (6,  7),
    76: (6,  8),   77: (6,  9),   78: (6, 10),   79: (6, 11),
    80: (6, 12),   81: (6, 13),   82: (6, 14),   83: (6, 15),
    84: (6, 16),   85: (6, 17),   86: (6, 18),
    87: (7,  1),   88: (7,  2),
    # Z = 89–90 (Ac, Th): group 3, period 7 — multi-occupancy, excluded
}

# Block classification by group
def block_label(g):
    if g <= 2:
        return "s"
    elif 3 <= g <= 12:
        return "d"
    else:
        return "p"


def main():
    # ── Build Φ = IE_zscore + 0.5 * R_zscore (all 90 elements) ──
    Zs = sorted(ELEMENTS.keys())
    ie = np.array([ELEMENTS[z][1] for z in Zs])
    r  = np.array([ELEMENTS[z][2] for z in Zs], dtype=float)
    ie_z = (ie - ie.mean()) / ie.std()
    r_z  = (r  - r.mean())  / r.std()
    phi  = {z: ie_z[i] + 0.5 * r_z[i] for i, z in enumerate(Zs)}

    # ── Reverse lookup: (period, group) → Z ──────────────────────
    grid = {}
    for z, (n, g) in POSITION.items():
        grid[(n, g)] = z

    # ── Compute diagonal differences ─────────────────────────────
    rows = []
    for z in sorted(POSITION.keys()):
        n, g = POSITION[z]
        z_5oc = grid.get((n + 1, g + 1))     # 5 o'clock
        z_7oc = grid.get((n + 1, g - 1))     # 7 o'clock
        if z_5oc is None or z_7oc is None:
            continue
        d_5oc = abs(phi[z] - phi[z_5oc])
        d_7oc = abs(phi[z] - phi[z_7oc])
        winner = ("g+1" if d_5oc < d_7oc else
                  "g-1" if d_7oc < d_5oc else "tie")
        rows.append({
            "z": z, "sym": ELEMENTS[z][0], "n": n, "g": g,
            "blk": block_label(g),
            "sym_5": ELEMENTS[z_5oc][0], "d_5": d_5oc,
            "sym_7": ELEMENTS[z_7oc][0], "d_7": d_7oc,
            "winner": winner,
        })

    # ── Print element-by-element table ────────────────────────────
    print("=" * 72)
    print("SNEATH DIAGONAL ASYMMETRY TEST")
    print("=" * 72)
    print(f"  Field: Φ = IE_zscore + 0.5 × R_zscore  (N_elem = 90)")
    print(f"  Lattice: 7 × 18, f-block excluded (multi-occupancy at group 3)")
    print(f"  Test: |ΔΦ| to (n+1, g+1) vs (n+1, g−1)")
    print()
    print(f"{'Elem':>5} {'Blk':>3} {'(n,g)':>7}  "
          f"{'(g+1)':>5}  {'|ΔΦ|':>7}  {'(g-1)':>5}  {'|ΔΦ|':>7}  {'Closer':>6}")
    print("-" * 72)
    for r in rows:
        print(f"{r['sym']:>5} {r['blk']:>3} ({r['n']},{r['g']:>2})  "
              f"{r['sym_5']:>5}  {r['d_5']:>7.4f}  "
              f"{r['sym_7']:>5}  {r['d_7']:>7.4f}  {r['winner']:>6}")

    # ── Summary function ──────────────────────────────────────────
    def summarize(label, subset):
        d5 = np.array([r["d_5"] for r in subset])
        d7 = np.array([r["d_7"] for r in subset])
        n5 = sum(1 for r in subset if r["winner"] == "g+1")
        n7 = sum(1 for r in subset if r["winner"] == "g-1")
        nt = sum(1 for r in subset if r["winner"] == "tie")
        N  = len(subset)

        print(f"\n--- {label} (N = {N}) ---")
        print(f"  (g+1) closer:  {n5}/{N}  ({100*n5/N:.1f}%)")
        print(f"  (g-1) closer:  {n7}/{N}  ({100*n7/N:.1f}%)")
        if nt:
            print(f"  Ties:          {nt}/{N}")
        print(f"  Median |ΔΦ|  (g+1) = {np.median(d5):.4f}   "
              f"(g-1) = {np.median(d7):.4f}")
        print(f"  Mean   |ΔΦ|  (g+1) = {np.mean(d5):.4f}   "
              f"(g-1) = {np.mean(d7):.4f}")

        if N >= 6:
            stat, p = wilcoxon(d5, d7, alternative="less")
            print(f"  Wilcoxon signed-rank (one-sided, |ΔΦ(g+1)| < |ΔΦ(g-1)|):")
            print(f"    W = {stat:.1f},  p = {p:.4e}")
            sig = "YES" if p < 0.05 else "no"
            print(f"  Significant at α = 0.05:  {sig}")
        else:
            print("  (N too small for Wilcoxon test)")

    # ── Report: pooled and by block ───────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    summarize("ALL TRIPLETS (pooled)", rows)
    for blk in ("s", "d", "p"):
        label = {"s": "s-block (groups 1–2)",
                 "d": "d-block (groups 3–12)",
                 "p": "p-block (groups 13–18)"}[blk]
        subset = [r for r in rows if r["blk"] == blk]
        if subset:
            summarize(label, subset)

    # ── Report: by period transition ──────────────────────────────
    print("\n" + "-" * 72)
    print("BY PERIOD TRANSITION")
    print("-" * 72)
    for pn in sorted(set(r["n"] for r in rows)):
        subset = [r for r in rows if r["n"] == pn]
        if subset:
            d5 = np.array([r["d_5"] for r in subset])
            d7 = np.array([r["d_7"] for r in subset])
            n5 = sum(1 for r in subset if r["winner"] == "g+1")
            N  = len(subset)
            print(f"  Period {pn}→{pn+1}:  {n5}/{N} ({100*n5/N:.0f}%) "
                  f"(g+1) closer   "
                  f"median (g+1)={np.median(d5):.3f}  (g-1)={np.median(d7):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
