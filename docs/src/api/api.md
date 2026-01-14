
# API Reference
Documentation for [WaveAcoustics](https://github.com/bacarmo/WaveAcoustics.jl).

```@index
```

## Mesh
```@autodocs
Modules = [WaveAcoustics]
Pages = ["mesh.jl"]
```

## Finite Element Families
```@autodocs
Modules = [WaveAcoustics]
Pages = ["fe_families.jl"]
```

## Basis Functions
```@meta
CurrentModule = WaveAcoustics
```

### 1D Lagrange Elements
```@docs
basis_functions(::Lagrange{1,1}, ξ::T) where {T<:Real}
basis_functions_derivatives(::Lagrange{1,1}, ξ::T) where {T<:Real}
```

```@docs
basis_functions(::Lagrange{1,2}, ξ::T) where {T<:Real}
basis_functions_derivatives(::Lagrange{1,2}, ξ::T) where {T<:Real}
```

```@docs
basis_functions(::Lagrange{1,3}, ξ::T) where {T<:Real}
basis_functions_derivatives(::Lagrange{1,3}, ξ::T) where {T<:Real}
```

### 2D Lagrange Elements
```@docs
basis_functions(::Lagrange{2,Deg}, ξ::T, η::T) where {Deg,T<:Real}
basis_functions_derivatives(::Lagrange{2,Deg}, ξ::T, η::T) where {Deg,T<:Real}
```

### 1D Hermite Elements
```@docs
basis_functions(::Hermite{1,3}, ξ::T) where {T<:Real}
basis_functions_derivatives(::Hermite{1,3}, ξ::T) where {T<:Real}
```

### 2D Hermite Elements
```@docs
basis_functions(::Hermite{2,3}, ξ::T, η::T) where {T<:Real}
basis_functions_derivatives(::Hermite{2,3}, ξ::T, η::T) where {T<:Real}
```

## Boundary Conditions
```@autodocs
Modules = [WaveAcoustics]
Pages = ["boundary_conditions.jl"]
```

## DOF Map
```@autodocs
Modules = [WaveAcoustics]
Pages = ["dof_map.jl"]
```

## Quadrature Setup
```@autodocs
Modules = [WaveAcoustics]
Pages = ["quadrature_setup.jl"]
```

## Assemble local matrices
```@autodocs
Modules = [WaveAcoustics]
Pages = ["assembly_local_matrices.jl"]
```

## Assemble global matrices
```@autodocs
Modules = [WaveAcoustics]
Pages = ["assembly_global_matrices.jl"]
```

## Assemble global vectors
```@autodocs
Modules = [WaveAcoustics]
Pages = ["assembly_global_vectors.jl"]
```
## Initial Solution
```@autodocs
Modules = [WaveAcoustics]
Pages = ["initial_solution.jl"]
```

## PDE Input Data
```@autodocs
Modules = [WaveAcoustics]
Pages = ["pde_inputdata.jl"]
```

## PDE Solve
```@autodocs
Modules = [WaveAcoustics]
Pages = ["pde_solve.jl"]
```

## Error norms
```@autodocs
Modules = [WaveAcoustics]
Pages = ["error_norms.jl"]
```

## Convergence test
```@autodocs
Modules = [WaveAcoustics]
Pages = ["convergence_test.jl"]
```