# Differential Structure of a Configuration Field on the Periodic Table

**Companion code and data for:**

> A. M. Rodriguez, "Differential Structure of a Configuration Field on the Periodic Table Encodes Chemical Hardness and Predicts Diatomic Bond Energies," submitted to *MATCH Communications in Mathematical and in Computer Chemistry* (2026).

---

## Overview

This repository contains a fully self-contained Python script that reproduces every numerical result, statistical test, and figure reported in the manuscript. All elemental and diatomic data are embedded directly in the script — no external files, APIs, or network access are required.

The core idea: a scalar field $\Phi = \widetilde{IE} + \lambda\,\widetilde{R}$ is constructed on the periodic table lattice from z-scored ionization energies and covalent radii. Applying standard discrete differential operators (gradient, Laplacian) to this field recovers known chemical properties:

- **Laplacian ↔ hardness:** $r = -0.830$ (Pearson, $N = 35$, $p < 10^{-9}$)
- **Geodesic cost ↔ bond dissociation energy:** $\rho = -0.633$ (Spearman, $N = 60$) on the gradient-magnitude field; $\rho = -0.325$ ($N = 201$) on the full diatomic set

All correlations are reported with BCa bootstrap 95% confidence intervals (10,000 resamples).

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

## What it produces

| File | Description |
|------|-------------|
| `VALIDATION_REPORT_v5.txt` | Full numerical report with all correlations, CIs, and p-values |
| `fig1_field_3panel.pdf/.png` | $\Phi$, $\|\nabla\Phi\|$, and $\nabla^2\Phi$ on the periodic table |
| `fig2_laplacian_hardness.pdf/.png` | Laplacian vs hardness and softness scatter plots |
| `fig3_scatter_headline.pdf/.png` | Geodesic cost vs $D_0$ ($N = 60$, gradient-magnitude field) |
| `fig4_scatter_continuous.pdf/.png` | Geodesic cost vs $D_0$ ($N = 201$, continuous interpolation) |
| `fig5_ablation.pdf/.png` | 16-configuration ablation heatmap |
| `ablation_summary.csv` | Ablation results for all 16 configurations |
| `headline_diatomics.csv` | Headline $D_0$ predictions ($N = 60$) |
| `discrete_phi_diatomics.csv` | Discrete $\Phi$-cost $D_0$ predictions ($N = 201$) |
| `continuous_diatomics.csv` | Continuous-interpolation $D_0$ predictions ($N = 201$) |
| `laplacian_hardness_data.csv` | Laplacian–hardness data ($N = 35$) |

## Methodology

1. **Field construction.** Ionization energies (NIST ASD v5.11) and covalent radii (Cordero et al. 2008) are z-score normalized and combined as $\Phi = \widetilde{IE} + \lambda\,\widetilde{R}$ with $\lambda = 0.5$ fixed a priori.

2. **Differential operators.** Discrete gradient magnitude $|\nabla\Phi|$ and five-point Laplacian $\nabla^2\Phi$ are computed on the (group, period) lattice.

3. **Hardness validation.** The Laplacian is compared against Pearson–Parr chemical hardness $\eta = (IE - EA)/2$ for 35 elements with known electron affinities.

4. **Bond-energy prediction.** Dijkstra shortest-path geodesic costs between element pairs are computed on the field and correlated with experimental bond dissociation energies from the CRC Handbook (104th ed.) and Huber & Herzberg (1979).

5. **Ablation.** A systematic sweep over $\lambda \in \{0.5, 1.0, 1.5, 2.0\}$, cost field $\in \{|\nabla\Phi|, \Phi\}$, and connectivity $\in \{\text{cardinal}, \text{diagonal}\}$ confirms robustness.

6. **Bootstrap.** All reported correlations include BCa 95% confidence intervals from 10,000 bootstrap resamples (seed = 42).

## Data sources

All data are embedded in the script. Original sources:

- **Ionization energies:** NIST Atomic Spectra Database, ver. 5.11 (2024)
- **Covalent radii:** Cordero et al., *Dalton Trans.* (2008) 2832–2838
- **Bond dissociation energies:** CRC Handbook of Chemistry and Physics, 104th ed.; Huber & Herzberg, *Molecular Spectra and Molecular Structure IV* (1979)
- **Electron affinities:** Hotop & Lineberger, *J. Phys. Chem. Ref. Data* **14** (1985) 731–750

## Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- SciPy
- Matplotlib

## License

This code is provided for scientific reproducibility. If you use it in your own work, please cite the manuscript above.

## Author

Anderson M. Rodriguez · ORCID [0009-0007-5179-9341](https://orcid.org/0009-0007-5179-9341)

MIT License

Copyright (c) 2026 Anderson M. Rodriguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

