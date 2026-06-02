# Geodesic Costs on a Scalar Field over the Periodic Table Predict Diatomic Bond Dissociation Energies

**Companion code and data for:**

> A. M. Rodriguez, "Geodesic Costs on a Scalar Field over the Periodic Table Predict Diatomic Bond Dissociation Energies," *Discover Chemistry* (2026).

---

## Overview

This repository contains fully self-contained Python scripts that reproduce every numerical result, statistical test, and figure reported in the manuscript. All elemental and diatomic data are embedded directly in the scripts — no external files, APIs, or network access are required.

The core idea: a scalar field $\Phi = \widetilde{IE} + \lambda\,\widetilde{R}$ is constructed on the periodic table lattice from z-scored ionization energies and covalent radii. Geodesic costs computed on this field via Dijkstra's algorithm predict experimental diatomic bond dissociation energies, and the field's curvature correlates with chemical hardness and atomic polarizability:

- **Geodesic cost ↔ bond dissociation energy:** $\rho = -0.633$ (Spearman, $N = 60$) on the gradient-magnitude field; $\rho = -0.325$ ($N = 201$) on the full diatomic set
- **Curvature ↔ hardness:** $r = -0.830$ (Pearson, $N = 35$, $p < 10^{-9}$)
- **Curvature ↔ polarizability:** $r = -0.600$ (Pearson, $\alpha^{-1/3}$ transform, $N = 85$, $p = 1.3 \times 10^{-9}$) — polarizability shares no input variable with the field

All correlations are reported with BCa bootstrap 95% confidence intervals (10,000 resamples). No parameters are fitted to any target property.

## Quick start

```bash
pip install numpy pandas scipy matplotlib
python cfc_master_validation_v5.py
```

Output lands in `cfc_validation_out/` by default. To change the directory:

```bash
python cfc_master_validation_v5.py --out_dir my_output
```

Runtime is roughly 2–5 minutes (dominated by bootstrap resampling).

For the polarizability analysis:

```bash
cd polarizability
python polarizability_curvature_analysis.py
```

## Repository structure

```
.
├── README.md
├── LICENSE
├── cfc_master_validation_v5.py              # Main analysis (Sections 2–8)
└── polarizability/
    ├── polarizability_curvature_analysis.py  # Curvature–polarizability analysis (Section 5.1)
    ├── polarizability_verification_table.csv # Source-annotated input data (Z = 1–90)
    ├── fig_polarizability_curvature.pdf      # Figure: ∇²Φ vs α and ln(α)
    └── fig_polarizability_curvature.png
```

## What it produces

### Main script (`cfc_master_validation_v5.py`)

| File | Description |
|------|-------------|
| `VALIDATION_REPORT_v5.txt` | Full numerical report with all correlations, CIs, and p-values |
| `fig1_field_3panel.pdf/.png` | $\Phi$, $\|\nabla\Phi\|$, and $\nabla^2\Phi$ on the periodic table |
| `fig2_laplacian_hardness.pdf/.png` | Curvature vs hardness and softness scatter plots |
| `fig3_scatter_headline.pdf/.png` | Geodesic cost vs $D_0$ ($N = 60$, gradient-magnitude field) |
| `fig4_scatter_discrete.pdf/.png` | Geodesic cost vs $D_0$ ($N = 201$, discrete $\Phi$ field) |
| `fig4_scatter_continuous.pdf/.png` | Geodesic cost vs $D_0$ ($N = 201$, continuous interpolation) |
| `fig5_ablation.pdf/.png` | 16-configuration ablation heatmap |
| `ablation_summary.csv` | Ablation results for all 16 configurations |
| `headline_diatomics.csv` | Headline $D_0$ predictions ($N = 60$) |
| `discrete_phi_diatomics.csv` | Discrete $\Phi$-cost $D_0$ predictions ($N = 201$) |
| `continuous_diatomics.csv` | Continuous-interpolation $D_0$ predictions ($N = 201$) |
| `laplacian_hardness_data.csv` | Curvature–hardness data ($N = 35$) |

### Polarizability script (`polarizability/polarizability_curvature_analysis.py`)

| File | Description |
|------|-------------|
| `fig_polarizability_curvature.pdf/.png` | $\nabla^2\Phi$ vs $\alpha$ and $\ln\alpha$ ($N = 85$) |
| Terminal output | Full correlation table: raw, log, cube-root, and inverse cube-root transforms with BCa CIs |

The subfolder also includes `polarizability_verification_table.csv`, which lists all input data ($Z = 1$–90) with per-value source annotations for independent verification.

## Methodology

1. **Field construction.** Ionization energies (NIST ASD v5.11) and covalent radii (Cordero et al. 2008) are z-score normalized and combined as $\Phi = \widetilde{IE} + \lambda\,\widetilde{R}$ with $\lambda = 0.5$ fixed a priori.

2. **Differential operators.** Discrete gradient magnitude $|\nabla\Phi|$ and second difference $\nabla^2\Phi$ along atomic number are computed on the (group, period) lattice.

3. **Bond-energy prediction.** Dijkstra shortest-path geodesic costs between element pairs are computed on the field and correlated with experimental bond dissociation energies from the CRC Handbook (104th ed.) and Huber & Herzberg (1979).

4. **Hardness validation.** The second difference is compared against Pearson–Parr chemical hardness $\eta = (IE - EA)/2$ for 35 elements with known electron affinities.

5. **Polarizability validation.** The second difference is correlated with static dipole polarizabilities (Schwerdtfeger & Nagle 2019/2025) for 85 elements. Unlike hardness, polarizability shares no input variable with the field, providing a stronger test that the curvature encodes genuine chemical information.

6. **Ablation.** A systematic sweep over $\lambda \in \{0.5, 1.0, 1.5, 2.0\}$, cost field $\in \{|\nabla\Phi|, \Phi\}$, and connectivity $\in \{\text{cardinal}, \text{diagonal}\}$ confirms robustness.

7. **Bootstrap.** All reported correlations include BCa 95% confidence intervals from 10,000 bootstrap resamples.

## Data sources

All data are embedded in the scripts. Original sources:

- **Ionization energies:** NIST Atomic Spectra Database, ver. 5.11 (2023)
- **Covalent radii:** Cordero et al., *Dalton Trans.* (2008) 2832–2838
- **Bond dissociation energies:** CRC Handbook of Chemistry and Physics, 104th ed.; Huber & Herzberg, *Molecular Spectra and Molecular Structure IV* (1979)
- **Electron affinities:** Hotop & Lineberger, *J. Phys. Chem. Ref. Data* **14** (1985) 731–750
- **Static dipole polarizabilities:** Schwerdtfeger & Nagle, *Mol. Phys.* **117** (2019) 1200–1225; 2025 update

## Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- SciPy
- Matplotlib

## License

MIT — see [LICENSE](LICENSE).

## Author

Anderson M. Rodriguez · ORCID [0009-0007-5179-9341](https://orcid.org/0009-0007-5179-9341)
