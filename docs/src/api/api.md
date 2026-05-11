
## Index
```@index
```

## Public API
### Problem Definition
```@docs
PDEInputData
example1_manufactured
example1_zero_source
example2_manufactured
example2_zero_source
```
### Finite Element Family
```@docs
Lagrange
```
### Time Integration
```@docs
CrankNicolson1
CrankNicolson2
CrankNicolson3
```
### Solver
```@docs
pde_solve
```
### Callbacks
```@docs
L2ErrorCallback
```
### Convergence study
```@docs
convergence_study_coupled
convergence_study_spatial
convergence_study_temporal
print_convergence_table
```

## Internal API
```@autodocs
Modules = [WaveAcoustics]
Filter = t -> !(t in [
    PDEInputData,
    example1_manufactured, example1_zero_source,
    example2_manufactured, example2_zero_source,
    Lagrange,
    CrankNicolson1, CrankNicolson2, CrankNicolson3,
    pde_solve,
    L2ErrorCallback,
    convergence_study_coupled, convergence_study_spatial, convergence_study_temporal, print_convergence_table
])
```