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
  - Non-pharmaceutical interventions (lockdowns, mobility restrictions) represented by the K parameter.
  - Stochastic parameter variations (varying contact rates, viral transmissibility).
  - System resilience and recovery speed analyses.
- **Visualization Suite:**
  - Comparative time-series plots for macro-variables ($N(t)$, $I(t)$, $R(t)$).
  - Probability distribution evolution over micro-scale variables $P(n, t)$.
---

## 📁 Repository Structure

```text
dymes-pandemic-model-comparison/
├── dyMES/
│   ├── default_models.py   # Contains default parameters and transition functions 
│   ├── model.py            # Infrastructure for running DyMES pandemic model with finite K
│   ├── model_k_inf.py      # Infrastructure for running DyMES pandemic model with infinite K
│   └── rfunctions.py       # Contains functions for probability distribution expectation calculations
├── sir/
│   ├── sir_scaled.py       # Run a version of the SIR model that scales parameters for representing whole-system behavior
├── fmd_cattle_code.ipynb   # Run code for foot and mouth disease (FMD) in cattle parameter simulation
├── hpai_code.ipynb         # Run code for high pathogenicity avian influenza (HPAI) parameter simulation
├── ndv_code.ipynb          # Run code for Newcastle Disease Virus (NDV) in commercial broilers parameter simulation
└── README.md
