# ========================================
# Type definitions
# ========================================
"""
    ODESolver

Abstract type for time integration schemes used in [`pde_solve`](@ref).

# Concrete subtypes
- [`CrankNicolson1`](@ref)
- [`CrankNicolson2`](@ref)
- [`CrankNicolson3`](@ref)
- [`MCrankNicolson1`](@ref)
"""
abstract type ODESolver end

"""
    CrankNicolson1 <: ODESolver

Standard Crank-Nicolson scheme for the coupled wave-acoustic system.

See *Scheme 1, Strategy 1* in the method documentation for the full problem formulation.
"""
struct CrankNicolson1 <: ODESolver end

"""
    CrankNicolson2 <: ODESolver

Standard Crank-Nicolson scheme for the coupled wave-acoustic system.

See *Scheme 1, Strategy 2* in the method documentation for the full problem formulation.
"""
struct CrankNicolson2 <: ODESolver end

"""
    CrankNicolson3 <: ODESolver

Standard Crank-Nicolson scheme for the coupled wave-acoustic system.

See *Scheme 1, Strategy 3* in the method documentation for the full problem formulation.
"""
struct CrankNicolson3 <: ODESolver end

"""
    MCrankNicolson1 <: ODESolver

Modified Crank-Nicolson scheme for the coupled wave-acoustic system.

See *Scheme 2* in the method documentation for the full problem formulation.
"""
struct MCrankNicolson1 <: ODESolver end

# ========================================
# Generic interfaces
# ========================================
"""
    build_cache(solver, matrices)

Allocate and return the solver-specific cache for the time integration.
Dispatches on `solver` to construct the appropriate cache type.

# Arguments
- `solver::ODESolver`: time integration scheme (e.g. `CrankNicolson1()`).
- `matrices::SystemMatrices{T,I}`: global FEM matrices.
 
# Extended help
To add support for a new solver `MySolver`, define:
```julia
build_cache(::MySolver, matrices) =
    MySolverCache(matrices)
```
"""
function build_cache end

"""
    ode_solve(cache, state, matrices, dof_map_m₁, dof_map_m₂,
              mesh1D, mesh2D, quad, tspan, input_data, callback)

Advance `state` over `tspan` using the time integration scheme `solver`.
Dispatches on `cache` to the appropriate implementation.

# Arguments
- `cache`: pre-allocated workspace; see [`build_cache`](@ref).
- `state::FEMState`: solution state; modified in-place at each time step.
- `matrices::SystemMatrices`: global FEM matrices.
- `dof_map_m₁::DOFMap`: DOF map for the `m₁`-space (wave field).
- `dof_map_m₂::DOFMap`: DOF map for the `m₂`-space (acoustic field).
- `mesh1D::CartesianMesh{1}`: 1D Cartesian mesh.
- `mesh2D::CartesianMesh{2}`: 2D Cartesian mesh.
- `quad::QuadratureSetup`: precomputed quadrature data.
- `tspan::StepRangeLen{T}`: time grid defined as `t₀:τ:t_end`.
- `input_data::PDEInputData`: problem data (source terms, coefficients, boundary data).
- `callback::AbstractCallback`: invoked after each accepted time step.
"""
function ode_solve end

# ========================================
# Implementations
# ========================================
include("common.jl")
include("crank_nicolson_strategy1_cache.jl")
include("crank_nicolson_strategy1.jl")
include("crank_nicolson_strategy2_cache.jl")
include("crank_nicolson_strategy2.jl")
include("crank_nicolson_strategy3_cache.jl")
include("crank_nicolson_strategy3.jl")