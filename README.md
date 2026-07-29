# DyMES Pandemic Model Comparison

Comparative analysis and benchmarking framework comparing the **DyMES** (Dynamic Maximum Entropy across Scales) framework against standard epidemiological modeling approaches (e.g., SIR/SEIR compartmental models) in simulating pandemic dynamics.

---

## 📌 Overview

Traditional compartmental models (such as SIR or SEIR) often rely on macroscopic aggregate state variables, which can struggle to capture dynamic feedback loops between individual micro-level agent behaviors and macro-level epidemic trends.

The **DyMES** framework integrates Maximum Entropy (MaxEnt) top-down inference with micro-level mechanistic rules to model two-way causation between macro state variables (e.g., total infected/recovered counts) and micro-scale probability distributions (e.g., individual risk/contact distributions).

This repository provides simulation code, data pipelines, and comparison benchmarks to evaluate model stability, perturbation response, and predictive performance during pandemic scenarios.

---

## ✨ Features

- **Model Implementations:**
  - **DyMES Framework:** Multi-scale dynamic Maximum Entropy model tailored for disease spread and intervention response.
  - **Baseline Models:** Standard compartmental models (SIR, SEIR, SEIR-D) for direct comparison.
- **Perturbation & Scenario Testing:**
  - Non-pharmaceutical interventions (lockdowns, mobility restrictions).
  - Stochastic parameter variations (varying contact rates, viral transmissibility).
  - System resilience and recovery speed analyses.
- **Visualization Suite:**
  - Comparative time-series plots for macro-variables ($N(t)$, $I(t)$, $R(t)$).
  - Probability distribution evolution over micro-scale variables $P(n, t)$.
  - Power spectral density and sensitivity analyses.

---

## 📁 Repository Structure

```text
dymes-pandemic-model-comparison/
├── data/                   # Empirical or synthetic pandemic dataset files
├── src/
│   ├── models/             # Implementations of DyMES and baseline (SIR/SEIR) models
│   ├── simulation/         # Run loops, scenario configs, and parameter sweeps
│   └── utils/              # Data parsing, metrics, and MaxEnt solvers
├── notebooks/              # Jupyter notebooks with interactive visual comparisons
├── tests/                  # Unit tests for core numerical solvers
├── requirements.txt        # Python package dependencies
└── README.md
