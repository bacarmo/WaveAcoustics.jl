# WaveAcoustics [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bacarmo.github.io/WaveAcoustics.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bacarmo.github.io/WaveAcoustics.jl/dev/) [![Build Status](https://github.com/bacarmo/WaveAcoustics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bacarmo/WaveAcoustics.jl/actions/workflows/CI.yml?query=branch%3Amain)[![DOI](https://zenodo.org/badge/1113335794.svg)](https://doi.org/10.5281/zenodo.21984456)

This repository contains a Julia implementation of the numerical scheme presented in the article "Numerical analysis for nonlinear wave equations with boundary conditions: Dirichlet, Acoustics and Impenetrability".
The results presented in the paper were obtained using MATLAB source code.
This package is a reimplementation in Julia.
It includes the Crank-Nicolson Galerkin scheme presented in the paper, along with two additional variants developed as a numerical investigation aimed at reducing execution time without sacrificing the order of convergence.

If you find these results useful, please cite the article
```bibtex
@article{ALCANTARA2025,
title    = {Numerical analysis for nonlinear wave equations with boundary conditions: Dirichlet, Acoustics and Impenetrability},
author   = {Adriano A. Alcântara and Juan B. Límaco and Bruno A. Carmo and Ronald R. Guardia and Mauro A. Rincon},
journal  = {Applied Mathematics and Computation},
volume   = {484},
pages    = {129009},
year     = {2025},
doi      = {10.1016/j.amc.2024.129009},
abstract = {In this article, we present an error estimation in the L2 norm referring to three wave models with variable coefficients, supplemented with initial and boundary conditions. The first two models are nonlinear wave equations with Dirichlet, Acoustics, and nonlinear dissipative impenetrability boundary conditions, while the third model is a linear wave equation with Dirichlet, Acoustics, and linear dissipative impenetrability boundary conditions. In the field of numerical analysis, we establish two key theorems for estimating errors to the semi-discrete and totally discrete problems associated with each model. Such theorems provide theoretical results on the convergence rate in both space and time. For conducting numerical simulations, we employ linear, quadratic, and cubic polynomial basis functions for the finite element spaces in the Galerkin method, in conjunction with the Crank-Nicolson method for time discretization. For each time step, we apply Newton's method to the resulting nonlinear problem. The numerical results are presented for all three models in order to corroborate with the theoretical convergence order obtained.}
}
```

This repository can be cited as
```bibtex
@software{WaveAcoustics,
author    = {Bruno Alves do Carmo},
title     = {Numerical solution for a nonlinear wave equation with boundary conditions: Dirichlet, Acoustics, and Impenetrability},
version   = {v0.4},
year      = {2026},
publisher = {Zenodo},
doi       = {10.5281/zenodo.21984457},
url       = {https://doi.org/10.5281/zenodo.21984457}
}
```