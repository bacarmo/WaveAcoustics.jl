# ========================================
# Types
# ========================================

"""
    RefinementData{T}

Per-field scalar data indexed by refinement level.

Stores one value per solution field of the wave-acoustic system at each
refinement level of a convergence study. Used for both L∞(L²) error norms and
convergence rates.

## Fields
- `v`: Wave velocity 
- `d`: Wave displacement
- `r`: Acoustic velocity
- `z`: Acoustic displacement
"""
struct RefinementData{T <: AbstractFloat}
    v::Vector{T}
    d::Vector{T}
    r::Vector{T}
    z::Vector{T}
end

"""
    RefinementData{T}(n::Int)

Construct a `RefinementData{T}` with all vectors initialized to zero, with
length `n` (number of refinement levels).
"""
function RefinementData{T}(n::Int) where {T <: AbstractFloat}
    RefinementData{T}(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
end

"""
    ConvergenceResults{T}

Output of a convergence study, storing discretization parameters,
L∞(L²) error norms, and convergence rates for all solution fields.

## Fields
- `test_info`: Description of the convergence study
- `Nx`: Mesh sizes per direction at each refinement level
- `h`: Element diameters at each refinement level
- `τ`: Time steps at each refinement level
- `errors`: L∞(L²) error norms per field at each refinement level
  (see [`RefinementData`](@ref))
- `rates`: Convergence rates log₂(eᵢ₋₁/eᵢ) per field at each refinement level
  (see [`RefinementData`](@ref))
"""
struct ConvergenceResults{T <: AbstractFloat, I <: Integer}
    test_info::String
    Nx::Vector{I}
    h::Vector{T}
    τ::Vector{T}
    errors::RefinementData{T}
    rates::RefinementData{T}
end

# ========================================
# Public API
# ========================================

"""
    convergence_study_coupled(; input_data, solver, fe, t_end, Nx_exp_range)

Coupled space-time convergence study with `τ = h^{(p+1)/2}`, with `p` the `fe` basis polynomial degree.

Refines spatial and temporal discretizations together. 
Convergence rates are computed as log₂(eᵢ₋₁/eᵢ) between successive levels.

## Keyword Arguments
- `input_data::PDEInputData`: PDE configuration with manufactured solution
  (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration scheme (default: `CrankNicolson1()`)
- `fe::FEFamily`: Finite element family (default: `Lagrange{1}()`)
- `t_end::Float64`: Final simulation time (default: `1.0`)
- `Nx_exp_range::UnitRange{Int}`: Exponents for grid refinement
  (default: `3:6` → Nx = 8, 16, 32, 64)

## Returns
[`ConvergenceResults`](@ref)

## Example
```julia
results = convergence_study_coupled(solver = ModifiedCN())
print_convergence_table(results)
```
"""
function convergence_study_coupled(;
        input_data::PDEInputData = example1_manufactured(),
        solver::ODESolver = CrankNicolson1(),
        fe::FEFamily = Lagrange{1}(),
        t_end::Float64 = 1.0,
        Nx_exp_range::UnitRange{Int} = 3:6)
    Nx_values = 2 .^ collect(Nx_exp_range)
    h_values = [element_diameter(Nx, input_data.pmin, input_data.pmax)
                for Nx in Nx_values]
    p = polynomial_degree(fe)
    τ_values = h_values .^ ((p + 1) / 2)

    test_info = "Coupled space-time convergence (τ = h^{($(p)+1)/2}) · t_end = $(t_end) · $(input_data.name) · $(string(typeof(fe))) · $(string(typeof(solver)))"

    return run_convergence_study(
        solver, fe, input_data, t_end, Nx_values, h_values, τ_values, test_info)
end

"""
    convergence_study_spatial(; input_data, solver, fe, t_end, Nx_exp_range, τ_fixed)

Spatial convergence study with fixed time step.

Refines the mesh while holding τ constant to isolate spatial discretization error.

## Keyword Arguments
- `input_data::PDEInputData`: PDE configuration with manufactured solution
  (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration scheme (default: `CrankNicolson1()`)
- `fe::FEFamily`: Finite element family (default: `Lagrange{1}()`)
- `t_end::Float64`: Final simulation time (default: `1.0`)
- `Nx_exp_range::UnitRange{Int}`: Exponents for grid refinement (default: `3:6`)
- `τ_fixed::Float64`: Fixed time step (default: `2.0^(-7)`)

## Returns
[`ConvergenceResults`](@ref)

## Example
```julia
results = convergence_study_spatial(τ_fixed = 2.0^(-8))
print_convergence_table(results)
```
"""
function convergence_study_spatial(;
        input_data::PDEInputData = example1_manufactured(),
        solver::ODESolver = CrankNicolson1(),
        fe::FEFamily = Lagrange{1}(),
        t_end::Float64 = 1.0,
        Nx_exp_range::UnitRange{Int} = 3:6,
        τ_fixed::Float64 = 2.0^(-7))
    Nx_values = 2 .^ collect(Nx_exp_range)
    h_values = [element_diameter(Nx, input_data.pmin, input_data.pmax)
                for Nx in Nx_values]
    τ_values = fill(τ_fixed, length(Nx_values))

    test_info = "Spatial convergence (τ = $τ_fixed fixed) · t_end = $(t_end) · $(input_data.name) · $(string(typeof(fe))) · $(string(typeof(solver)))"

    return run_convergence_study(
        solver, fe, input_data, t_end, Nx_values, h_values, τ_values, test_info)
end

"""
    convergence_study_temporal(; input_data, solver, fe, t_end, τ_exp_range, Nx_fixed)

Temporal convergence study with fixed spatial grid.

Refines the time step while holding Nx constant to isolate temporal discretization error.

## Keyword Arguments
- `input_data::PDEInputData`: PDE configuration with manufactured solution
  (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration scheme (default: `CrankNicolson1()`)
- `fe::FEFamily`: Finite element family (default: `Lagrange{1}()`)
- `t_end::Float64`: Final simulation time (default: `1.0`)
- `τ_exp_range::UnitRange{Int}`: Exponents for time step refinement
  (default: `3:6` → τ = 2⁻³, …, 2⁻⁶)
- `Nx_fixed::Int`: Fixed mesh size per direction (default: `2^8`)

## Returns
[`ConvergenceResults`](@ref)

## Example
```julia
results = convergence_study_temporal(Nx_fixed = 2^7)
print_convergence_table(results)
```
"""
function convergence_study_temporal(;
        input_data::PDEInputData = example1_manufactured(),
        solver::ODESolver = CrankNicolson1(),
        fe::FEFamily = Lagrange{1}(),
        t_end::Float64 = 1.0,
        τ_exp_range::UnitRange{Int} = 3:6,
        Nx_fixed::Int = 2^8)
    τ_values = 2.0 .^ .-(collect(τ_exp_range))
    n = length(τ_values)
    Nx_values = fill(Nx_fixed, n)
    h_fixed = element_diameter(Nx_fixed, input_data.pmin, input_data.pmax)
    h_values = fill(h_fixed, n)

    test_info = "Temporal convergence " *
                "(Nx = $Nx_fixed, h ≈ $(round(h_fixed; sigdigits = 4)) fixed) · " *
                "t_end = $(t_end) · $(input_data.name) · $(string(typeof(fe))) · $(string(typeof(solver)))"

    return run_convergence_study(
        solver, fe, input_data, t_end, Nx_values, h_values, τ_values, test_info)
end

"""
    print_convergence_table(results)

Print a formatted L∞(L²) convergence table to stdout.

## Example
```julia
results = convergence_study_coupled()
print_convergence_table(results)
```
"""
function print_convergence_table(results::ConvergenceResults)
    sep = "="^110
    println("\n", sep)
    println(results.test_info)
    println(sep)
    @printf("  %5s  %7s  %7s  %10s  %6s  %10s  %6s  %10s  %6s  %10s  %6s\n",
        "Nx", "log₂h", "log₂τ",
        "L∞L²(v)", "rate",
        "L∞L²(d)", "rate",
        "L∞L²(r)", "rate",
        "L∞L²(z)", "rate")
    println("-"^110)

    for i in eachindex(results.Nx)
        @printf("  %5d  %7.2f  %7.2f  %10.2e  %6.3f  %10.2e  %6.3f  %10.2e  %6.3f  %10.2e  %6.3f\n",
            results.Nx[i],
            log2(results.h[i]),
            log2(results.τ[i]),
            results.errors.v[i], results.rates.v[i],
            results.errors.d[i], results.rates.d[i],
            results.errors.r[i], results.rates.r[i],
            results.errors.z[i], results.rates.z[i])
    end

    println(sep)
end

# ========================================
# Internal functions
# ========================================

"""
    element_diameter(Nx, pmin, pmax) -> T

Element diameter h = √(Δx² + Δy²) for a uniform rectangular mesh with `Nx`
subdivisions per direction over the domain `[pmin, pmax]`.
"""
function element_diameter(
        Nx::Int,
        pmin::NTuple{2, T},
        pmax::NTuple{2, T}
) where {T <: AbstractFloat}
    Δx = (pmax[1] - pmin[1]) / Nx
    Δy = (pmax[2] - pmin[2]) / Nx
    return sqrt(Δx^2 + Δy^2)
end

"""
    run_convergence_study(
        solver, fe, input_data, t_end, Nx_values, h_values, τ_values, test_info)

Execute a convergence study over a sequence of refinement levels.

Solves the PDE at each level, computes L∞(L²) error norms for all fields,
and estimates convergence rates as log₂(eᵢ₋₁/eᵢ).
"""
function run_convergence_study(
        solver::ODESolver,
        fe::FEFamily,
        input_data::PDEInputData,
        t_end::T,
        Nx_values::Vector{I},
        h_values::Vector{T},
        τ_values::Vector{T},
        test_info::String
) where {T <: AbstractFloat, I <: Integer}
    n = length(Nx_values)
    errors = RefinementData{T}(n)
    rates = RefinementData{T}(n)

    for i in eachindex(Nx_values)
        tspan = zero(T):τ_values[i]:t_end
        cb = L2ErrorCallback(tspan)
        pde_solve((Nx_values[i], Nx_values[i]), fe, tspan, input_data, solver, cb)

        errors.v[i] = maximum(cb.v_errors)
        errors.d[i] = maximum(cb.d_errors)
        errors.r[i] = maximum(cb.r_errors)
        errors.z[i] = maximum(cb.z_errors)
    end

    compute_rates!(rates, errors)

    return ConvergenceResults{T, I}(test_info, Nx_values, h_values, τ_values, errors, rates)
end

"""
    compute_rates!(rates, errors)

Compute convergence rates in-place as log₂(eᵢ₋₁/eᵢ) for each field.

The first entry of each rate vector remains zero since no previous refinement
level is available. The `rates` argument must be pre-allocated with zeros of
length equal to `errors` (as produced by [`RefinementData{T}(n)`](@ref)).
"""
function compute_rates!(
        rates::RefinementData{T}, errors::RefinementData{T}) where {T <: AbstractFloat}
    for (e, r) in zip(
        (errors.v, errors.d, errors.r, errors.z),
        (rates.v, rates.d, rates.r, rates.z))
        for i in 2:length(e)
            r[i] = log2(e[i - 1] / e[i])
        end
    end
end