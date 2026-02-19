#!/usr/bin/env python3
"""
cfc_master_validation_v5.py — Configuration Field Chemistry
=============================================================

FULLY SELF-CONTAINED one-click validation for the paper:

  "Differential Structure of a Configuration Field on the Periodic Table
   Encodes Chemical Hardness and Predicts Diatomic Bond Energies"

All data is embedded.  No external files, APIs, or network access needed.

Runs:
  1. Build Phi = IE_norm + lam*R_norm on (group, period) lattice
  2. Compute |grad Phi|, Laplacian Phi
  3. Laplacian-hardness/softness validation + BCa bootstrap CIs
  4. Headline diatomic D0 prediction (N=60, gradmag, cardinal, lam=0.5) + CIs
  5. Continuous interpolation benchmark (N=201) + CIs
  6. Full 16-config ablation study
  7. Generate all manuscript figures (PNG + PDF)
  8. Write comprehensive VALIDATION_REPORT_v5.txt

v5 additions:
  - BCa bootstrap 95% confidence intervals on all reported correlations
  - Explicit "lambda fixed a priori" documentation in report

Usage:
  pip install numpy pandas scipy matplotlib
  python cfc_master_validation_v5.py [--out_dir cfc_validation_out]

Author: Anderson M. Rodriguez | ORCID: 0009-0007-5179-9341
Data:   IE from NIST ASD (v5.11, 2024)
        R from Cordero et al., Dalton Trans. 2008, 2832-2838
        D0 from CRC Handbook (104th ed) + Huber & Herzberg (1979)
        EA from Hotop & Lineberger, J. Phys. Chem. Ref. Data 14, 731 (1985)
"""
from __future__ import annotations
import argparse, heapq, math, os, sys, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RBFInterpolator
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ================================================================
def bootstrap_ci(x, y, stat_func, n_boot=10000, alpha=0.05, seed=42):
    """BCa bootstrap confidence interval for a bivariate statistic.
    
    Parameters
    ----------
    x, y : array-like, same length
    stat_func : callable(x, y) -> float  (e.g. lambda a,b: pearsonr(a,b)[0])
    n_boot : int, number of bootstrap resamples
    alpha : float, significance level (0.05 -> 95% CI)
    seed : int, RNG seed for reproducibility
    
    Returns
    -------
    (ci_lo, ci_hi, theta_hat) : tuple of floats
    """
    rng = np.random.RandomState(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    theta_hat = stat_func(x, y)
    
    # Bootstrap resamples
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            boot_stats[b] = stat_func(x[idx], y[idx])
        except:
            boot_stats[b] = np.nan
    boot_stats = boot_stats[np.isfinite(boot_stats)]
    if len(boot_stats) < 100:
        return (np.nan, np.nan, theta_hat)
    
    # Bias-correction factor
    z0 = norm_ppf(np.mean(boot_stats < theta_hat))
    
    # Acceleration factor (jackknife)
    jack = np.empty(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i+1, n)])
        try:
            jack[i] = stat_func(x[idx_j], y[idx_j])
        except:
            jack[i] = np.nan
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        return (np.nan, np.nan, theta_hat)
    jmean = jack.mean()
    num = np.sum((jmean - jack)**3)
    den = 6.0 * (np.sum((jmean - jack)**2))**1.5
    a_hat = num / den if den != 0 else 0.0
    
    # BCa adjusted percentiles
    z_lo = norm_ppf(alpha / 2.0)
    z_hi = norm_ppf(1.0 - alpha / 2.0)
    
    def bca_percentile(z_alpha):
        numer = z0 + z_alpha
        adj = z0 + numer / (1.0 - a_hat * numer)
        return norm_cdf(adj)
    
    p_lo = bca_percentile(z_lo)
    p_hi = bca_percentile(z_hi)
    
    # Clamp to [0,1]
    p_lo = max(0.0, min(1.0, p_lo))
    p_hi = max(0.0, min(1.0, p_hi))
    
    ci_lo = np.percentile(boot_stats, 100.0 * p_lo)
    ci_hi = np.percentile(boot_stats, 100.0 * p_hi)
    return (ci_lo, ci_hi, theta_hat)


def norm_ppf(p):
    """Inverse normal CDF (probit) using rational approximation."""
    # Abramowitz & Stegun 26.2.23
    if p <= 0: return -6.0
    if p >= 1: return 6.0
    if p == 0.5: return 0.0
    if p > 0.5:
        return -norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1*t + c2*t**2) / (1.0 + d1*t + d2*t**2 + d3*t**3))


def norm_cdf(x):
    """Normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_all_cis(df, n_boot=10000):
    """Compute bootstrap CIs for all correlation metrics in a diatomic results df."""
    d = df["D0"].values
    pred = df["pred_score"].values
    man = -df["manhattan"].values.astype(float)
    euc = -df["euclid"].values
    
    pear_fn = lambda a, b: float(pearsonr(a, b)[0])
    spear_fn = lambda a, b: float(spearmanr(a, b)[0])
    
    cis = {}
    for name, xvals in [("pred", pred), ("man", man), ("euc", euc)]:
        cis[f"pearson_{name}_ci"] = bootstrap_ci(xvals, d, pear_fn, n_boot)
        cis[f"spearman_{name}_ci"] = bootstrap_ci(xvals, d, spear_fn, n_boot)
    return cis


def fmt_ci(ci_tuple):
    """Format a CI tuple as string: 'value [lo, hi]'."""
    lo, hi, theta = ci_tuple
    if np.isnan(lo):
        return f"{theta:+.4f} [CI failed]"
    return f"{theta:+.4f} [{lo:+.4f}, {hi:+.4f}]"

# ================================================================
# EMBEDDED DATA
# ================================================================

# (Symbol, Z, group, period, IE_eV, R_cov_pm)
# IE: NIST ASD v5.11 | R: Cordero et al. Dalton Trans. 2008, 2832
# Lanthanides mapped to extended groups 19-32
ELEMENT_DATA = [
    ("H",1,1,1,13.598,31),("He",2,18,1,24.587,28),
    ("Li",3,1,2,5.392,128),("Be",4,2,2,9.323,96),
    ("B",5,13,2,8.298,84),("C",6,14,2,11.260,73),
    ("N",7,15,2,14.534,71),("O",8,16,2,13.618,66),
    ("F",9,17,2,17.423,57),("Ne",10,18,2,21.565,58),
    ("Na",11,1,3,5.139,166),("Mg",12,2,3,7.646,141),
    ("Al",13,13,3,5.986,121),("Si",14,14,3,8.152,111),
    ("P",15,15,3,10.487,107),("S",16,16,3,10.360,105),
    ("Cl",17,17,3,12.968,102),("Ar",18,18,3,15.760,106),
    ("K",19,1,4,4.341,203),("Ca",20,2,4,6.113,176),
    ("Sc",21,3,4,6.562,170),("Ti",22,4,4,6.828,160),
    ("V",23,5,4,6.746,153),("Cr",24,6,4,6.767,139),
    ("Mn",25,7,4,7.434,150),("Fe",26,8,4,7.902,142),
    ("Co",27,9,4,7.881,138),("Ni",28,10,4,7.640,124),
    ("Cu",29,11,4,7.726,132),("Zn",30,12,4,9.394,122),
    ("Ga",31,13,4,5.999,122),("Ge",32,14,4,7.900,120),
    ("As",33,15,4,9.789,119),("Se",34,16,4,9.752,120),
    ("Br",35,17,4,11.814,120),("Kr",36,18,4,14.000,116),
    ("Rb",37,1,5,4.177,220),("Sr",38,2,5,5.695,195),
    ("Y",39,3,5,6.217,190),("Zr",40,4,5,6.634,175),
    ("Nb",41,5,5,6.759,164),("Mo",42,6,5,7.092,154),
    ("Tc",43,7,5,7.119,147),("Ru",44,8,5,7.361,146),
    ("Rh",45,9,5,7.459,142),("Pd",46,10,5,8.337,139),
    ("Ag",47,11,5,7.576,145),("Cd",48,12,5,8.994,144),
    ("In",49,13,5,5.786,142),("Sn",50,14,5,7.344,139),
    ("Sb",51,15,5,8.608,139),("Te",52,16,5,9.010,138),
    ("I",53,17,5,10.451,139),("Xe",54,18,5,12.130,140),
    ("Cs",55,1,6,3.894,244),("Ba",56,2,6,5.212,215),
    ("La",57,3,6,5.577,207),("Ce",58,3,6,5.539,204),
    ("Pr",59,3,6,5.473,203),("Nd",60,3,6,5.525,201),
    ("Pm",61,3,6,5.582,199),("Sm",62,3,6,5.644,198),
    ("Eu",63,3,6,5.670,198),("Gd",64,3,6,6.150,196),
    ("Tb",65,3,6,5.864,194),("Dy",66,3,6,5.939,192),
    ("Ho",67,3,6,6.022,192),("Er",68,3,6,6.108,189),
    ("Tm",69,3,6,6.184,190),("Yb",70,3,6,6.254,187),
    ("Lu",71,3,6,5.426,187),
    ("Hf",72,4,6,6.825,175),("Ta",73,5,6,7.550,170),
    ("W",74,6,6,7.864,162),("Re",75,7,6,7.834,151),
    ("Os",76,8,6,8.438,144),("Ir",77,9,6,8.967,141),
    ("Pt",78,10,6,8.959,136),("Au",79,11,6,9.226,136),
    ("Hg",80,12,6,10.437,132),("Tl",81,13,6,6.108,145),
    ("Pb",82,14,6,7.417,146),("Bi",83,15,6,7.286,148),
    ("Po",84,16,6,8.414,140),("At",85,17,6,9.318,150),
    ("Rn",86,18,6,10.749,150),
    ("Fr",87,1,7,4.073,260),("Ra",88,2,7,5.278,221),
    ("Ac",89,3,7,5.380,215),("Th",90,3,7,6.307,206),
]

# Electron affinities (eV). 0.0 = no stable anion.
# Hotop & Lineberger, J. Phys. Chem. Ref. Data 14, 731 (1985)
EA_DATA = {
    "H":0.754,"He":0.0,"Li":0.618,"Be":0.0,
    "B":0.277,"C":1.263,"N":0.0,"O":1.461,
    "F":3.401,"Ne":0.0,"Na":0.548,"Mg":0.0,
    "Al":0.433,"Si":1.389,"P":0.746,"S":2.077,
    "Cl":3.613,"Ar":0.0,"K":0.501,"Ca":0.025,
    "Sc":0.188,"Ti":0.079,"V":0.525,"Cr":0.666,
    "Mn":0.0,"Fe":0.151,"Co":0.662,"Ni":1.156,
    "Cu":1.235,"Zn":0.0,"Ga":0.43,"Ge":1.233,
    "As":0.814,"Se":2.021,"Br":3.364,"Kr":0.0,
}

# Diatomic D0 (eV). CRC Handbook 104th + Huber & Herzberg 1979
D0_DATA = [
    ("CO","C","O",11.092),("ThO","Th","O",9.0),("BO","B","O",8.28),
    ("SiO","Si","O",8.26),("LaO","La","O",8.23),("HfO","Hf","O",8.19),
    ("ZrO","Zr","O",7.85),("BF","B","F",7.81),("CN","C","N",7.76),
    ("CS","C","S",7.35),("YO","Y","O",7.29),("ScO","Sc","O",6.96),
    ("AlF","Al","F",6.89),("LaF","La","F",6.86),("TiO","Ti","O",6.82),
    ("GeO","Ge","O",6.78),("TbF","Tb","F",6.66),("NO","N","O",6.4968),
    ("IrC","Ir","C",6.45),("SiS","Si","S",6.42),("CeF","Ce","F",6.41),
    ("VO","V","O",6.41),("PN","P","N",6.36),("PtC","Pt","C",6.28),
    ("YF","Y","F",6.2),("ScF","Sc","F",6.17),("PO","P","O",6.15),
    ("GdF","Gd","F",6.12),("BaF","Ba","F",6.015),("BS","B","S",6.01),
    ("RhC","Rh","C",6.01),("PrF","Pr","F",6.0),("GaF","Ga","F",5.98),
    ("LaS","La","S",5.91),("ErF","Er","F",5.9),("HF","H","F",5.869),
    ("BeF","Be","F",5.85),("SmF","Sm","F",5.81),("BaO","Ba","O",5.79),
    ("PmF","Pm","F",5.78),("CF","C","F",5.67),("NdF","Nd","F",5.65),
    ("RaF","Ra","F",5.61),("EuF","Eu","F",5.59),("SrF","Sr","F",5.58),
    ("SiF","Si","F",5.57),("BCl","B","Cl",5.5),("CaF","Ca","F",5.48),
    ("DyF","Dy","F",5.47),("YbF","Yb","F",5.41),("HoF","Ho","F",5.41),
    ("LaCl","La","Cl",5.41),("LuF","Lu","F",5.39),("SO","S","O",5.359),
    ("CP","C","P",5.28),("AlO","Al","O",5.27),("InF","In","F",5.25),
    ("TmF","Tm","F",5.24),("AlCl","Al","Cl",5.12),("GeF","Ge","F",5.0),
    ("AsO","As","O",4.98),("GaCl","Ga","Cl",4.92),("SnF","Sn","F",4.9),
    ("SrO","Sr","O",4.88),("TbCl","Tb","Cl",4.87),("NS","N","S",4.8),
    ("CaO","Ca","O",4.76),("TiS","Ti","S",4.75),("MgF","Mg","F",4.75),
    ("CeCl","Ce","Cl",4.74),("GdCl","Gd","Cl",4.67),("ErCl","Er","Cl",4.65),
    ("LaBr","La","Br",4.62),("BeO","Be","O",4.6),("TlF","Tl","F",4.57),
    ("PS","P","S",4.54),("BaCl","Ba","Cl",4.53),("BBr","B","Br",4.49),
    ("InCl","In","Cl",4.44),("AlBr","Al","Br",4.43),("CuF","Cu","F",4.42),
    ("SeO","Se","O",4.41),("CrO","Cr","O",4.4),("SbF","Sb","F",4.4),
    ("OH","O","H",4.392),("PrCl","Pr","Cl",4.39),("BaS","Ba","S",4.36),
    ("NdCl","Nd","Cl",4.34),("SmCl","Sm","Cl",4.34),("PmCl","Pm","Cl",4.29),
    ("HoCl","Ho","Cl",4.24),("AsF","As","F",4.2),("EuCl","Eu","Cl",4.2),
    ("NiC","Ni","C",4.167),("DyCl","Dy","Cl",4.07),("NSe","N","Se",4.0),
    ("FeC","Fe","C",3.961),("LuCl","Lu","Cl",3.94),("TmCl","Tm","Cl",3.92),
    ("GaO","Ga","O",3.91),("YbCl","Yb","Cl",3.88),("CeBr","Ce","Br",3.87),
    ("AlS","Al","S",3.84),("PbO","Pb","O",3.83),("PtO","Pt","O",3.82),
    ("TlCl","Tl","Cl",3.82),("BeS","Be","S",3.8),("BaBr","Ba","Br",3.71),
    ("MnO","Mn","O",3.7),("SeS","Se","S",3.7),("NiS","Ni","S",3.651),
    ("AgF","Ag","F",3.64),("PbF","Pb","F",3.64),("PrBr","Pr","Br",3.57),
    ("SH","S","H",3.55),("MgO","Mg","O",3.53),("NdBr","Nd","Br",3.52),
    ("TeS","Te","S",3.5),("NF","N","F",3.5),("PbS","Pb","S",3.49),
    ("LiO","Li","O",3.49),("TbI","Tb","I",3.48),("SrS","Sr","S",3.48),
    ("NH","N","H",3.47),("BiO","Bi","O",3.47),("PmBr","Pm","Br",3.47),
    ("GdI","Gd","I",3.46),("CaS","Ca","S",3.46),("CH","C","H",3.46),
    ("PtH","Pt","H",3.44),("SmBr","Sm","Br",3.43),("CuBr","Cu","Br",3.43),
    ("BH","B","H",3.42),("ScCl","Sc","Cl",3.4),("EuBr","Eu","Br",3.38),
    ("MgBr","Mg","Br",3.35),("AuAl","Au","Al",3.34),("GeH","Ge","H",3.3),
    ("MgCl","Mg","Cl",3.29),("CuI","Cu","I",3.27),("ErI","Er","I",3.27),
    ("FeS","Fe","S",3.24),("AgCl","Ag","Cl",3.22),("AuH","Au","H",3.22),
    ("NiSe","Ni","Se",3.218),("BiS","Bi","S",3.17),("PrI","Pr","I",3.17),
    ("NdI","Nd","I",3.12),("AgBr","Ag","Br",3.1),("AuS","Au","S",3.089),
    ("BiCl","Bi","Cl",3.08),("PmI","Pm","I",3.07),("SiH","Si","H",3.06),
    ("AlH","Al","H",3.06),("SmI","Sm","I",3.04),("PH","P","H",3.02),
    ("EuI","Eu","I",2.99),("BiH","Bi","H",2.9),("NBr","N","Br",2.9),
    ("IF","I","F",2.879),("DyI","Dy","I",2.85),("HoI","Ho","I",2.85),
    ("GaH","Ga","H",2.84),("CuS","Cu","S",2.8),("CuO","Cu","O",2.79),
    ("ClO","Cl","O",2.751),("FeSe","Fe","Se",2.739),("LuI","Lu","I",2.73),
    ("CuH","Cu","H",2.73),("TmI","Tm","I",2.7),("YbI","Yb","I",2.67),
    ("ClF","Cl","F",2.617),("AgI","Ag","I",2.6),("CuSe","Cu","Se",2.55),
    ("MnH","Mn","H",2.5),("InH","In","H",2.48),("LiH","Li","H",2.428365),
    ("MgS","Mg","S",2.4),("CuTe","Cu","Te",2.35),("AgH","Ag","H",2.28),
    ("BrCl","Br","Cl",2.233),("ICl","I","Cl",2.1531),("BeH","Be","H",2.034),
    ("BaH","Ba","H",1.95),("AgAl","Ag","Al",1.95),("KH","K","H",1.86),
    ("CsH","Cs","H",1.81),("IO","I","O",1.8),("CaH","Ca","H",1.7),
    ("SrH","Sr","H",1.66),("PbH","Pb","H",1.59),("MgH","Mg","H",1.34),
    ("NaLi","Na","Li",0.88097),("ZnH","Zn","H",0.851),
    ("LiRb","Li","Rb",0.7341),("LiCs","Li","Cs",0.7284639),
    ("NaK","Na","K",0.646175),("NaRb","Na","Rb",0.61713614),
    ("KRb","K","Rb",0.5183),("RbCs","Rb","Cs",0.47562),
    ("XeCl","Xe","Cl",0.03),
]

# ================================================================
# CORE MATH
# ================================================================
def zscore(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / s if s > 0 else x * 0.0

def load_elements():
    return pd.DataFrame([
        {"Symbol":s,"Z":z,"group":g,"period":p,"IE_eV":ie,"R_pm":r}
        for s,z,g,p,ie,r in ELEMENT_DATA])

def load_diatomics():
    return pd.DataFrame(D0_DATA, columns=["chemical_formula","A","B","D0"])

def build_phi_grid(df, lam):
    sym=df["Symbol"].values; grp=df["group"].values.astype(int)
    per=df["period"].values.astype(int); ie=df["IE_eV"].values.astype(float)
    rad=df["R_pm"].values.astype(float)
    ok=np.isfinite(ie)&np.isfinite(rad)&(grp>0)&(per>0)
    sym,grp,per,ie,rad = sym[ok],grp[ok],per[ok],ie[ok],rad[ok]
    ie_n,r_n = zscore(ie),zscore(rad)
    gmax,pmax = int(grp.max()),int(per.max())
    phi = np.full((pmax,gmax), np.nan)
    coords = {}
    for s,g,p,ien,rn,ie_raw,r_raw in zip(sym,grp,per,ie_n,r_n,ie,rad):
        coords[s] = {"group":int(g),"period":int(p),"ie":float(ie_raw),"r":float(r_raw)}
        phi[p-1, g-1] = float(ien + lam*rn)
    return phi, coords

def gradient_magnitude(phi):
    # NaN-propagating gradient: cells adjacent to NaN become NaN.
    # This creates a sparser cost field, forcing geodesic paths through
    # chemically populated corridors rather than through empty cells.
    gy, gx = np.gradient(phi)
    gmag = np.sqrt(gx**2 + gy**2)
    gmag[np.isnan(phi)] = np.nan
    return gmag

def laplacian_5pt(phi):
    H,W = phi.shape
    lap = np.full_like(phi, np.nan)
    for y in range(H):
        for x in range(W):
            c = phi[y,x]
            if not np.isfinite(c): continue
            up = phi[y-1,x] if y>0 else np.nan
            dn = phi[y+1,x] if y<H-1 else np.nan
            lf = phi[y,x-1] if x>0 else np.nan
            rt = phi[y,x+1] if x<W-1 else np.nan
            if all(np.isfinite(v) for v in [up,dn,lf,rt]):
                lap[y,x] = (up+dn+lf+rt) - 4.0*c
    return lap

def dijkstra_cost(cf, start, goal, diag=False):
    H,W = cf.shape; sy,sx=start; gy,gx=goal
    if not (np.isfinite(cf[sy,sx]) and np.isfinite(cf[gy,gx])): return np.inf
    dist=np.full((H,W),np.inf); dist[sy,sx]=0.0; pq=[(0.0,sy,sx)]
    steps=[(-1,0),(1,0),(0,-1),(0,1)]
    if diag: steps+=[(-1,-1),(-1,1),(1,-1),(1,1)]
    while pq:
        d,y,x=heapq.heappop(pq)
        if d>dist[y,x]: continue
        if (y,x)==(gy,gx): return d
        for dy,dx in steps:
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and np.isfinite(cf[ny,nx]):
                sl=math.sqrt(2.0) if (dy!=0 and dx!=0) else 1.0
                w=0.5*(cf[y,x]+cf[ny,nx])*sl; nd=d+w
                if nd<dist[ny,nx]: dist[ny,nx]=nd; heapq.heappush(pq,(nd,ny,nx))
    return float(dist[gy,gx])

def make_cost_field(phi, gmag, mode):
    if mode=="gradmag":
        c=gmag.copy(); fin=c[np.isfinite(c)]
        eps=np.nanmedian(fin)*1e-6 if fin.size else 1e-9
        c[np.isfinite(c)&(c<=0)]=eps; return c
    elif mode=="phi":
        c=phi.copy(); fin=c[np.isfinite(c)]
        if fin.size: c=c-np.nanmin(fin)
        eps=np.nanmedian(c[np.isfinite(c)])*1e-6 if np.any(np.isfinite(c)) else 1e-9
        c[np.isfinite(c)&(c<=0)]=eps; return c
    raise ValueError(f"Unknown cost mode: {mode}")

def interpolate_phi(df, lam, smooth=0.1):
    grp=df["group"].values.astype(float); per=df["period"].values.astype(float)
    ie=df["IE_eV"].values.astype(float); rad=df["R_pm"].values.astype(float)
    ok=np.isfinite(ie)&np.isfinite(rad)&np.isfinite(grp)&np.isfinite(per)
    ie_n,r_n = zscore(ie[ok]),zscore(rad[ok])
    phi_vals=ie_n+lam*r_n; pts=np.column_stack([grp[ok],per[ok]])
    rbf=RBFInterpolator(pts,phi_vals,kernel="thin_plate_spline",smoothing=smooth)
    xs=sorted(df["group"].dropna().unique()); ys=sorted(df["period"].dropna().unique())
    gx,gy=np.meshgrid(xs,ys); flat=np.column_stack([gx.ravel(),gy.ravel()])
    Phi=rbf(flat).reshape(gx.shape)
    xi={x:i for i,x in enumerate(xs)}; yi={y:i for i,y in enumerate(ys)}
    return Phi,xs,ys,xi,yi

# ================================================================
# VALIDATION ROUTINES
# ================================================================
def laplacian_hardness_validation(coords):
    """Correlate 1D second difference of Φ along Z with Parr-Pearson hardness.
    
    The second difference ∇²Φ[i] = Φ[i+1] + Φ[i-1] - 2Φ[i] measures
    how each element's configuration index deviates from the average of
    its atomic-number neighbors. Peaks correspond to hard/inert elements
    (noble gases, filled shells); valleys to soft/reactive elements.
    """
    # Sort elements by Z
    elem_list = sorted(
        [(s, c) for s, c in coords.items()],
        key=lambda x: next(z for sym,z,g,p,ie,r in ELEMENT_DATA if sym==x[0])
    )
    # Compute Φ for each element (zscore of IE + 0.5 * zscore of R)
    ie_arr = np.array([c["ie"] for _,c in elem_list])
    r_arr = np.array([c["r"] for _,c in elem_list])
    ie_z = (ie_arr - ie_arr.mean()) / ie_arr.std()
    r_z = (r_arr - r_arr.mean()) / r_arr.std()
    phi_1d = ie_z + 0.5 * r_z
    
    # 1D second difference (interior elements only)
    lap_1d = np.full(len(phi_1d), np.nan)
    for i in range(1, len(phi_1d)-1):
        lap_1d[i] = phi_1d[i+1] + phi_1d[i-1] - 2*phi_1d[i]
    
    syms,lap_v,eta_v,soft_v = [],[],[],[]
    for i,(s,c) in enumerate(elem_list):
        ea = EA_DATA.get(s)
        if ea is None or not np.isfinite(lap_1d[i]): continue
        eta = (c["ie"]-ea)/2.0
        if eta<=0: continue
        syms.append(s); lap_v.append(lap_1d[i]); eta_v.append(eta); soft_v.append(1.0/eta)
    la,ea,sa = np.array(lap_v),np.array(eta_v),np.array(soft_v)
    n=len(la)
    pr_h=pearsonr(la,ea) if n>=3 else (np.nan,np.nan)
    pr_s=pearsonr(la,sa) if n>=3 else (np.nan,np.nan)
    sp_h=spearmanr(la,ea) if n>=3 else (np.nan,np.nan)
    sp_s=spearmanr(la,sa) if n>=3 else (np.nan,np.nan)
    return dict(n=n,symbols=syms,laplacian=la,hardness=ea,softness=sa,
                pearson_hardness=pr_h,pearson_softness=pr_s,
                spearman_hardness=sp_h,spearman_softness=sp_s)

def run_diatomic_validation(phi, gmag, coords, df_diat, cost_mode, diag):
    cost_field = make_cost_field(phi, gmag, cost_mode)
    results = []
    for _,row in df_diat.iterrows():
        a,b = str(row["A"]),str(row["B"])
        if a not in coords or b not in coords: continue
        ca,cb = coords[a],coords[b]
        start=(ca["period"]-1,ca["group"]-1); goal=(cb["period"]-1,cb["group"]-1)
        gc = dijkstra_cost(cost_field, start, goal, diag=diag)
        if not np.isfinite(gc): continue
        dy,dx = abs(start[0]-goal[0]),abs(start[1]-goal[1])
        results.append(dict(chemical_formula=row["chemical_formula"],A=a,B=b,
            D0=float(row["D0"]),geo_cost=gc,pred_score=-gc,
            manhattan=dy+dx,euclid=math.sqrt(dy**2+dx**2)))
    return pd.DataFrame(results)

def corrs(df):
    if len(df)<3:
        nan2=(np.nan,np.nan)
        return dict(pearson_pred=nan2,spearman_pred=nan2,pearson_man=nan2,
                    spearman_man=nan2,pearson_euc=nan2,spearman_euc=nan2)
    d=df["D0"].values
    return dict(
        pearson_pred=pearsonr(df["pred_score"].values,d),
        spearman_pred=spearmanr(df["pred_score"].values,d),
        pearson_man=pearsonr(-df["manhattan"].values.astype(float),d),
        spearman_man=spearmanr(-df["manhattan"].values.astype(float),d),
        pearson_euc=pearsonr(-df["euclid"].values,d),
        spearman_euc=spearmanr(-df["euclid"].values,d))

# ================================================================
# FIGURES (all PNG + PDF)
# ================================================================
def savefig(fig,stem):
    fig.savefig(f"{stem}.png",dpi=300,bbox_inches="tight")
    fig.savefig(f"{stem}.pdf",bbox_inches="tight")
    plt.close(fig); print(f"      -> {stem}.png / .pdf")

def fig_field_3panel(phi,gmag,lap,stem):
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    im0=axes[0].imshow(phi,origin="upper",aspect="auto",cmap="viridis")
    axes[0].set_title(r"(a) $\Phi$",fontsize=13); axes[0].set_xlabel("Group"); axes[0].set_ylabel("Period")
    fig.colorbar(im0,ax=axes[0],shrink=0.8,label=r"$\Phi$")
    im1=axes[1].imshow(gmag,origin="upper",aspect="auto",cmap="inferno")
    axes[1].set_title(r"(b) $|\nabla\Phi|$",fontsize=13); axes[1].set_xlabel("Group"); axes[1].set_ylabel("Period")
    fig.colorbar(im1,ax=axes[1],shrink=0.8,label=r"$|\nabla\Phi|$")
    lp=lap.copy(); vmax=np.nanmax(np.abs(lp[np.isfinite(lp)])) if np.any(np.isfinite(lp)) else 1
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    im2=axes[2].imshow(lp,origin="upper",aspect="auto",cmap="RdBu_r",norm=norm)
    axes[2].set_title(r"(c) $\nabla^2\Phi$",fontsize=13); axes[2].set_xlabel("Group"); axes[2].set_ylabel("Period")
    fig.colorbar(im2,ax=axes[2],shrink=0.8,label=r"$\nabla^2\Phi$")
    fig.suptitle(r"Configuration field on the periodic table ($\lambda = 0.5$)",fontsize=14,y=1.02)
    fig.tight_layout(); savefig(fig,stem)

def fig_laplacian_hardness(hv,stem):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.5))
    r_h,p_h=hv["pearson_hardness"]
    ax1.scatter(hv["laplacian"],hv["hardness"],c="steelblue",s=50,edgecolors="k",lw=0.5)
    for i,s in enumerate(hv["symbols"]):
        ax1.annotate(s,(hv["laplacian"][i],hv["hardness"][i]),fontsize=7,alpha=0.75,ha="center",va="bottom")
    ax1.set_xlabel(r"$\nabla^2\Phi$",fontsize=12); ax1.set_ylabel(r"Hardness $\eta = (I-A)/2$ (eV)",fontsize=12)
    ax1.set_title(f"(a) Laplacian vs hardness\nPearson r = {r_h:.3f}, p = {p_h:.2e}",fontsize=11)
    m,b=np.polyfit(hv["laplacian"],hv["hardness"],1)
    xr=np.linspace(hv["laplacian"].min(),hv["laplacian"].max(),50)
    ax1.plot(xr,m*xr+b,"k--",alpha=0.4,lw=1)
    r_s,p_s=hv["pearson_softness"]
    ax2.scatter(hv["laplacian"],hv["softness"],c="coral",s=50,edgecolors="k",lw=0.5)
    for i,s in enumerate(hv["symbols"]):
        ax2.annotate(s,(hv["laplacian"][i],hv["softness"][i]),fontsize=7,alpha=0.75,ha="center",va="bottom")
    ax2.set_xlabel(r"$\nabla^2\Phi$",fontsize=12); ax2.set_ylabel(r"Softness $S=1/\eta$ (eV$^{-1}$)",fontsize=12)
    ax2.set_title(f"(b) Laplacian vs softness\nPearson r = {r_s:.3f}, p = {p_s:.2e}",fontsize=11)
    m2,b2=np.polyfit(hv["laplacian"],hv["softness"],1)
    ax2.plot(xr,m2*xr+b2,"k--",alpha=0.4,lw=1)
    fig.tight_layout(); savefig(fig,stem)

def fig_scatter(df,corr,extra,stem):
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(df["pred_score"],df["D0"],s=30,alpha=0.7,edgecolors="k",lw=0.3)
    rho,p=corr["spearman_pred"]; r,pr=corr["pearson_pred"]
    ax.set_xlabel(r"Predicted score ($-\mathcal{G}$)",fontsize=12)
    ax.set_ylabel(r"Experimental $D_0$ (eV)",fontsize=12)
    ax.set_title(f"Geodesic cost vs bond dissociation energy\n"
                 f"Spearman rho = {rho:.3f} (p = {p:.2e})  |  "
                 f"Pearson r = {r:.3f}  |  N = {len(df)}\n{extra}",fontsize=10)
    fig.tight_layout(); savefig(fig,stem)

def fig_ablation(abl,stem):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    for ax,cm,title in [(ax1,"gradmag",r"$|\nabla\Phi|$ cost"),(ax2,"phi",r"$\Phi$ cost")]:
        sub=abl[abl["cost"]==cm]
        if sub.empty: ax.set_title(f"{title}: no data"); continue
        lams=sorted(sub["lam"].unique()); diags=[False,True]
        data=np.full((2,len(lams)),np.nan)
        for i,dg in enumerate(diags):
            for j,la in enumerate(lams):
                row=sub[(sub["lam"]==la)&(sub["diag"]==dg)]
                if not row.empty: data[i,j]=row["spearman_pred"].values[0]
        im=ax.imshow(-data,cmap="YlOrRd",aspect="auto")
        ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l}" for l in lams])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Cardinal","Diagonal"])
        ax.set_xlabel(r"$\lambda$"); ax.set_title(title)
        for i in range(2):
            for j in range(len(lams)):
                if np.isfinite(data[i,j]):
                    ax.text(j,i,f"{data[i,j]:.3f}",ha="center",va="center",
                            fontsize=9,color="white" if -data[i,j]>0.4 else "black")
    fig.suptitle(r"Ablation: Spearman $\rho$(pred, $D_0$)",fontsize=13)
    fig.tight_layout(); savefig(fig,stem)

# ================================================================
# MAIN
# ================================================================
def main():
    ap=argparse.ArgumentParser(description="CFC master validation v5")
    ap.add_argument("--out_dir",default="cfc_validation_out")
    ap.add_argument("--lam_grid",default="0.5,1.0,1.5,2.0")
    ap.add_argument("--smooth",type=float,default=0.1)
    ap.add_argument("--n_boot",type=int,default=10000,help="Bootstrap resamples")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    lam_grid=[float(x) for x in args.lam_grid.split(",")]
    ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    NB=args.n_boot

    print("="*70)
    print("  CONFIGURATION FIELD CHEMISTRY - MASTER VALIDATION v5")
    print("="*70)
    print(f"  {ts}  |  Output: {out}")
    print(f"  All data embedded. No external files needed.")
    print(f"  Bootstrap: {NB} resamples | BCa 95% CIs")
    print("="*70)

    df_elem=load_elements(); df_diat=load_diatomics()
    print(f"\n[1/7] {len(df_elem)} elements, {len(df_diat)} diatomics loaded")

    print("\n[2/7] Building field (lam=0.5, zscore)...")
    phi,coords=build_phi_grid(df_elem,lam=0.5)
    gmag=gradient_magnitude(phi); lap=laplacian_5pt(phi)
    print(f"      Grid: {phi.shape} | Elements: {len(coords)}")

    # === LAPLACIAN-HARDNESS ===
    print("\n[3/7] === LAPLACIAN-HARDNESS VALIDATION + BOOTSTRAP CIs ===")
    hv=laplacian_hardness_validation(coords)
    r_h,p_h=hv["pearson_hardness"]; r_s,p_s=hv["pearson_softness"]
    rho_h,prho_h=hv["spearman_hardness"]; rho_s,prho_s=hv["spearman_softness"]
    
    # Bootstrap CIs for hardness correlations
    pear_fn = lambda a,b: float(pearsonr(a,b)[0])
    spear_fn = lambda a,b: float(spearmanr(a,b)[0])
    la_arr, eta_arr, soft_arr = hv["laplacian"], hv["hardness"], hv["softness"]
    
    ci_pear_eta  = bootstrap_ci(la_arr, eta_arr, pear_fn, NB)
    ci_spear_eta = bootstrap_ci(la_arr, eta_arr, spear_fn, NB)
    ci_pear_S    = bootstrap_ci(la_arr, soft_arr, pear_fn, NB)
    ci_spear_S   = bootstrap_ci(la_arr, soft_arr, spear_fn, NB)
    
    print(f"      N = {hv['n']} elements with EA data")
    print(f"      Lap vs hardness:  Pearson  {fmt_ci(ci_pear_eta)}  (p={p_h:.2e})")
    print(f"                        Spearman {fmt_ci(ci_spear_eta)}  (p={prho_h:.2e})")
    print(f"      Lap vs softness:  Pearson  {fmt_ci(ci_pear_S)}  (p={p_s:.2e})")
    print(f"                        Spearman {fmt_ci(ci_spear_S)}  (p={prho_s:.2e})")
    print(f"\n      {'Sym':>4s} {'Lap':>9s} {'IE':>7s} {'EA':>6s} {'eta':>7s} {'S':>7s}")
    for i,s in enumerate(hv["symbols"]):
        print(f"      {s:>4s} {hv['laplacian'][i]:+9.4f} {coords[s]['ie']:7.3f} "
              f"{EA_DATA[s]:6.3f} {hv['hardness'][i]:7.3f} {hv['softness'][i]:7.4f}")
    hv_df=pd.DataFrame(dict(symbol=hv["symbols"],laplacian=hv["laplacian"],
                             hardness_eta=hv["hardness"],softness_S=hv["softness"]))
    hv_df.to_csv(out/"laplacian_hardness_data.csv",index=False)
    fig_laplacian_hardness(hv,str(out/"fig2_laplacian_hardness"))

    # === HEADLINE ===
    print("\n[4/7] Headline diatomic (gradmag, cardinal, lam=0.5) + CIs...")
    res_h=run_diatomic_validation(phi,gmag,coords,df_diat,"gradmag",False)
    ch=corrs(res_h); sp=ch["spearman_pred"]; pe=ch["pearson_pred"]
    sm=ch["spearman_man"]; se=ch["spearman_euc"]
    
    cis_h = compute_all_cis(res_h, NB)
    
    print(f"      N={len(res_h)}")
    print(f"      Geodesic:  Spearman {fmt_ci(cis_h['spearman_pred_ci'])}  (p={sp[1]:.2e})")
    print(f"                 Pearson  {fmt_ci(cis_h['pearson_pred_ci'])}")
    print(f"      Manhattan: Spearman {fmt_ci(cis_h['spearman_man_ci'])}")
    print(f"      Euclidean: Spearman {fmt_ci(cis_h['spearman_euc_ci'])}")
    res_h.sort_values("D0",ascending=False).to_csv(out/"headline_diatomics.csv",index=False)

    # === DISCRETE PHI-COST (N=201, the primary D0 result in the paper) ===
    print("\n[4b/7] Discrete Phi-cost (cardinal, lam=0.5) + CIs...")
    res_phi=run_diatomic_validation(phi,gmag,coords,df_diat,"phi",False)
    ch_phi=corrs(res_phi); sp_phi=ch_phi["spearman_pred"]
    cis_phi = compute_all_cis(res_phi, NB)
    print(f"      N={len(res_phi)}")
    print(f"      Geodesic:  Spearman {fmt_ci(cis_phi['spearman_pred_ci'])}  (p={sp_phi[1]:.2e})")
    print(f"                 Pearson  {fmt_ci(cis_phi['pearson_pred_ci'])}")
    print(f"      Manhattan: Spearman {fmt_ci(cis_phi['spearman_man_ci'])}")
    print(f"      Euclidean: Spearman {fmt_ci(cis_phi['spearman_euc_ci'])}")
    res_phi.sort_values("D0",ascending=False).to_csv(out/"discrete_phi_diatomics.csv",index=False)

    print("\n[5/7] Figures...")
    fig_field_3panel(phi,gmag,lap,str(out/"fig1_field_3panel"))
    fig_scatter(res_phi,ch_phi,r"$\Phi$ cost, cardinal, $\lambda=0.5$",str(out/"fig4_scatter_discrete"))
    fig_scatter(res_h,ch,r"$|\nabla\Phi|$ cost, cardinal, $\lambda=0.5$",str(out/"fig3_scatter_headline"))

    # === CONTINUOUS ===
    print("\n[6/7] Continuous interpolation (lam=0.5) + CIs...")
    cis_c = {}
    try:
        Phi_c,xs,ys,xi,yi=interpolate_phi(df_elem,lam=0.5,smooth=args.smooth)
        gmag_c=gradient_magnitude(Phi_c); cost_c=make_cost_field(Phi_c,gmag_c,"phi")
        rc=[]
        for _,row in df_diat.iterrows():
            a,b=str(row["A"]),str(row["B"])
            ca,cb=coords.get(a),coords.get(b)
            if not ca or not cb: continue
            if ca["group"] not in xi or cb["group"] not in xi: continue
            if ca["period"] not in yi or cb["period"] not in yi: continue
            start=(yi[ca["period"]],xi[ca["group"]]); goal=(yi[cb["period"]],xi[cb["group"]])
            gc=dijkstra_cost(cost_c,start,goal,diag=False)
            if not np.isfinite(gc): continue
            dy,dx=abs(start[0]-goal[0]),abs(start[1]-goal[1])
            rc.append(dict(chemical_formula=row["chemical_formula"],A=a,B=b,
                D0=float(row["D0"]),pred_score=-gc,manhattan=dy+dx,euclid=math.sqrt(dy**2+dx**2)))
        df_c=pd.DataFrame(rc); cc=corrs(df_c); spc=cc["spearman_pred"]
        
        cis_c = compute_all_cis(df_c, NB)
        
        print(f"      N={len(df_c)}")
        print(f"      Geodesic:  Spearman {fmt_ci(cis_c['spearman_pred_ci'])}  (p={spc[1]:.2e})")
        print(f"                 Pearson  {fmt_ci(cis_c['pearson_pred_ci'])}")
        print(f"      Manhattan: Spearman {fmt_ci(cis_c['spearman_man_ci'])}")
        print(f"      Euclidean: Spearman {fmt_ci(cis_c['spearman_euc_ci'])}")
        df_c.sort_values("D0",ascending=False).to_csv(out/"continuous_diatomics.csv",index=False)
        fig_scatter(df_c,cc,r"$\Phi$ cost, continuous, $\lambda=0.5$",str(out/"fig4_scatter_continuous"))
    except Exception as e:
        print(f"      [WARN] Failed: {e}"); df_c=pd.DataFrame(); cc={}

    # === ABLATION ===
    print("\n[7/7] Ablation (16 configs)...")
    abl_rows=[]
    for cm in ["gradmag","phi"]:
        for dg in [False,True]:
            for lam in lam_grid:
                pa,ca2=build_phi_grid(df_elem,lam=lam); ga=gradient_magnitude(pa)
                ra=run_diatomic_validation(pa,ga,ca2,df_diat,cm,dg); ca=corrs(ra)
                row=dict(cost=cm,diag=dg,lam=lam,N=len(ra),
                    pearson_pred=ca["pearson_pred"][0],spearman_pred=ca["spearman_pred"][0],
                    p_spearman=ca["spearman_pred"][1],
                    spearman_man=ca["spearman_man"][0],spearman_euc=ca["spearman_euc"][0])
                tag=f"{cm}_diag{int(dg)}_lam{lam}"
                print(f"      {tag:30s} N={row['N']:3d} rho={row['spearman_pred']:+.3f} p={row['p_spearman']:.1e}")
                abl_rows.append(row)
    abl_df=pd.DataFrame(abl_rows); abl_df.to_csv(out/"ablation_summary.csv",index=False)
    fig_ablation(abl_df,str(out/"fig5_ablation"))

    # === REPORT ===
    rpt=out/"VALIDATION_REPORT_v5.txt"
    with open(rpt,"w") as f:
        W=lambda s: f.write(s+"\n")
        W("="*70); W("  CONFIGURATION FIELD CHEMISTRY - VALIDATION REPORT v5"); W("="*70)
        W(f"  Generated: {ts}"); W(f"  Output: {out}")
        W(f"  Bootstrap: {NB} resamples, BCa 95% CIs, seed=42\n")
        
        W("-"*70); W("  METHODOLOGY NOTE"); W("-"*70)
        W("  Lambda was fixed a priori at 0.5 from the D0 prediction task")
        W("  (ablation over lam in {0.5, 1.0, 1.5, 2.0}).")
        W("  The hardness validation was performed AFTER freezing lambda.")
        W("  No parameters were tuned to optimize the hardness correlation.\n")
        
        W("-"*70); W("  DATA SOURCES (all embedded)"); W("-"*70)
        W(f"  Elements: {len(df_elem)} | IE: NIST ASD v5.11 | R: Cordero 2008")
        W(f"  Diatomics: {len(df_diat)} | CRC 104th + Huber & Herzberg 1979")
        W(f"  EA: {len(EA_DATA)} elements | Hotop & Lineberger 1985\n")
        
        W("-"*70); W("  LAPLACIAN-HARDNESS VALIDATION (lam=0.5)"); W("-"*70)
        W(f"  N = {hv['n']}")
        W(f"  Lap vs eta:")
        W(f"    Pearson  r  = {fmt_ci(ci_pear_eta)}  (p={p_h:.2e})")
        W(f"    Spearman rho = {fmt_ci(ci_spear_eta)}  (p={prho_h:.2e})")
        W(f"  Lap vs S:")
        W(f"    Pearson  r  = {fmt_ci(ci_pear_S)}  (p={p_s:.2e})")
        W(f"    Spearman rho = {fmt_ci(ci_spear_S)}  (p={prho_s:.2e})\n")
        
        W("-"*70); W("  HEADLINE D0 (gradmag, cardinal, lam=0.5)"); W("-"*70)
        W(f"  N = {len(res_h)}")
        W(f"  Geodesic cost:")
        W(f"    Spearman rho = {fmt_ci(cis_h['spearman_pred_ci'])}  (p={sp[1]:.2e})")
        W(f"    Pearson  r   = {fmt_ci(cis_h['pearson_pred_ci'])}")
        W(f"  Baselines:")
        W(f"    Manhattan  Spearman = {fmt_ci(cis_h['spearman_man_ci'])}")
        W(f"    Euclidean  Spearman = {fmt_ci(cis_h['spearman_euc_ci'])}\n")
        
        W("-"*70); W("  DISCRETE PHI-COST D0 (phi cost, cardinal, lam=0.5)"); W("-"*70)
        W(f"  N = {len(res_phi)}  [PRIMARY D0 RESULT IN MANUSCRIPT]")
        W(f"  Geodesic cost:")
        W(f"    Spearman rho = {fmt_ci(cis_phi['spearman_pred_ci'])}  (p={sp_phi[1]:.2e})")
        W(f"    Pearson  r   = {fmt_ci(cis_phi['pearson_pred_ci'])}")
        W(f"  Baselines:")
        W(f"    Manhattan  Spearman = {fmt_ci(cis_phi['spearman_man_ci'])}")
        W(f"    Euclidean  Spearman = {fmt_ci(cis_phi['spearman_euc_ci'])}\n")
        
        if len(df_c)>0 and cc and cis_c:
            scc=cc["spearman_pred"]
            W("-"*70); W("  CONTINUOUS D0 (phi cost, cardinal, lam=0.5)"); W("-"*70)
            W(f"  N = {len(df_c)}")
            W(f"  Geodesic cost:")
            W(f"    Spearman rho = {fmt_ci(cis_c['spearman_pred_ci'])}  (p={scc[1]:.2e})")
            W(f"    Pearson  r   = {fmt_ci(cis_c['pearson_pred_ci'])}")
            W(f"  Baselines:")
            W(f"    Manhattan  Spearman = {fmt_ci(cis_c['spearman_man_ci'])}")
            W(f"    Euclidean  Spearman = {fmt_ci(cis_c['spearman_euc_ci'])}\n")
        
        W("-"*70); W("  ABLATION (16 configs)"); W("-"*70)
        best=abl_df.loc[abl_df["spearman_pred"].abs().idxmax()]
        W(f"  Best: {best['cost']}, diag={best['diag']}, lam={best['lam']} -> rho={best['spearman_pred']:+.4f}\n")
        W(f"  {'Config':<30s} {'N':>4s} {'rho':>8s} {'Man':>8s} {'Euc':>8s} {'p':>10s}")
        W("  "+"-"*68)
        for _,r in abl_df.iterrows():
            tag=f"{r['cost']}_diag{int(r['diag'])}_lam{r['lam']}"
            W(f"  {tag:<30s} {int(r['N']):>4d} {r['spearman_pred']:>+8.3f} "
              f"{r['spearman_man']:>+8.3f} {r['spearman_euc']:>+8.3f} {r['p_spearman']:>10.2e}")
        W("  "+"-"*68+"\n")
        W("-"*70); W("  OUTPUT FILES"); W("-"*70)
        for p in sorted(out.glob("*")): W(f"  {p.name}")
        W("\n"+"="*70); W("  END OF REPORT"); W("="*70)

    print(f"\n{'='*70}")
    print(f"  DONE -> {out}/"); print(f"  Report: {rpt}")
    print(f"{'='*70}")
    print(f"\n  KEY RESULTS (with 95% BCa bootstrap CIs):")
    print(f"  Lap vs hardness: {fmt_ci(ci_pear_eta)}")
    print(f"  Lap vs softness: {fmt_ci(ci_pear_S)}")
    print(f"  Discrete Phi D0 (N={len(res_phi)}): {fmt_ci(cis_phi['spearman_pred_ci'])}  [PRIMARY]")
    print(f"  Headline |grad| D0 (N={len(res_h)}): {fmt_ci(cis_h['spearman_pred_ci'])}")
    if cis_c: print(f"  Continuous (N={len(df_c)}):  {fmt_ci(cis_c['spearman_pred_ci'])}")
    print()

if __name__=="__main__":
    main()
