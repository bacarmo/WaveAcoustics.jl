# ========================================
# Type definitions
# ========================================
"""
    AbstractODESolver

Abstract type for time integration schemes used in [`solve_pde`](@ref).

# Concrete subtypes
- [`Scheme1`](@ref)
- [`Scheme2`](@ref)
"""
abstract type AbstractODESolver end

"""
    Scheme1 <: AbstractODESolver

Standard Crank-Nicolson scheme for the PDE system.
See *Scheme 1* in the method documentation for the full problem formulation.
"""
struct Scheme1 <: AbstractODESolver end

"""
    Scheme2 <: AbstractODESolver

Modified Crank-Nicolson scheme for the PDE system.
See *Scheme 2* in the method documentation for the full problem formulation.
"""
struct Scheme2 <: AbstractODESolver end

"""
    Scheme3 <: AbstractODESolver

Linearized Crank-Nicolson scheme for the PDE system.
See *Scheme 3* in the method documentation for the full problem formulation.
"""
struct Scheme3 <: AbstractODESolver end

# ========================================
# Generic interfaces
# ========================================
"""
    build_cache(solver, matrices)

Allocate and return the solver-specific cache for the time integration.
Dispatches on `solver` to construct the appropriate cache type.

# Arguments
- `solver::AbstractODESolver`: time integration scheme (e.g. `Scheme1()`)
- `matrices::SystemMatrices{T, I}`: global FEM matrices

# Extended help
To add support for a new solver `MySolver`, define:
```julia
build_cache(::MySolver, matrices) = MySolverCache(matrices)
```
"""
function build_cache end

"""
    solve_ode(cache, tspan, state, matrices, nel_per_dim, element_side_lengths,
              dof_map_m₁, dof_map_m₂, quad, input_data, callback)


Advance `state` over `tspan` using the time integration scheme `solver`.
Dispatches on `cache` to the appropriate implementation.

# Arguments
- `cache`: pre-allocated workspace; see [`build_cache`](@ref).
- `tspan::StepRangeLen{T}`: time grid defined as `t₀:τ:t_end`.
- `state::State`: solution state; modified in-place at each time step.
- `matrices::SystemMatrices`: global FEM matrices.
- `nel_per_dim::NTuple{2, I}`: Number of elements along each spatial dimension
- `element_side_lengths::NTuple{2, T}`: Element side lengths `(Δx, Δy)`
- `dof_map_m₁::DOFMap`: DOF map for the `m₁`-space
- `dof_map_m₂::DOFMap`: DOF map for the `m₂`-space
- `quad::QuadratureSetup`: precomputed quadrature data
- `input_data::PDEInputData`: input data container for the PDE system
- `callback::AbstractCallback`: invoked after each time step
"""
function solve_ode end

# ========================================
# Implementations
# ========================================
include("utils.jl")

include("scheme1_cache.jl")
include("scheme1.jl")

include("scheme2_cache.jl")
include("scheme2.jl")

include("scheme3_cache.jl")
include("scheme3.jl")