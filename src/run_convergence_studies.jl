"""
    ConvergenceResults{T,I}

Stores the output of a mesh-refinement convergence study.

Fields
- `info`  : description of the run
- `Nx`    : number of elements per spatial direction at each refinement level
- `h`     : element diameter `√(Δx²+Δy²)` at each level
- `τ`     : time-step size at each level
- `errors`: L∞(L²) error norms; size `(n_levels, n_fields)`
- `rates` : convergence rates, computed as log(eᵢ₋₁/eᵢ)/log(δᵢ₋₁/δᵢ), where δ denotes the refinement parameter: δ = h for 2D spatial studies, δ = Δx for 1D spatial studies, and δ = τ for temporal studies; same size as `errors`
"""
struct ConvergenceResults{T <: AbstractFloat, I <: Integer}
    info::String
    Nx::Vector{I}
    h::Vector{T}
    τ::Vector{T}
    errors::Matrix{T}  # (n_levels, n_fields)
    rates::Matrix{T}   # (n_levels, n_fields); first row is always zero
end

"""
    run_cases(solver, cases, t_end=1.0) -> Vector{ConvergenceResults}

Runs a convergence study for each case in `cases`, writing each [`ConvergenceResults`](@ref) to a file named
`convergence_studies_<solver>_<YYYY-MM-DD_HH-MM-SS>.txt` as it becomes available, and returns all results as a vector.

# Arguments
- `solver`: time integration scheme, e.g. `Scheme1()`
- `cases::Tuple`: tuple of `NamedTuple`s, one per convergence study, each with keys:
    - `fe`: finite element basis with only the polynomial degree fixed (e.g. `Lagrange{1}`)
    - `id`: input data container for the PDE system, e.g. `example1_manufactured(2.4)`
    - `Nx`: number of elements per spatial direction; `Int` (fixed) or `Vector{Int}` (refined).
    - `τ` : time step; `Float64` (fixed) or `Vector{Float64}` (refined).
    - `t_end::Float64=1.0` : final simulation time.

    At least one of `Nx`/`τ` must vary; if both are vectors they must have the same length.

# Returns
- `Vector{ConvergenceResults}`: one entry per case, in the same order as `cases`.
  Results are also written incrementally to the output file described above.


# Example
```julia-repl
julia> using WaveAcoustics

julia> cases = (
 (fe = Lagrange{1}, id = example1_manufactured(1.76), Nx = [2^i for i in 2:5], τ = 2.0^-13),
 (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
 (fe = Lagrange{2}, id = example1_manufactured(2.58), Nx = [2^i for i in 2:5], τ = 2.0^-13),
 (fe = Lagrange{2}, id = example1_manufactured(3.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
 (fe = Lagrange{3}, id = example1_manufactured(3.51), Nx = [2^i for i in 2:5], τ = 2.0^-13),
 (fe = Lagrange{3}, id = example1_manufactured(4.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13)
 );

julia> result = run_cases(Scheme1(), cases)
```
"""
function run_cases(
        solver::AbstractODESolver,
        cases::Tuple,
        t_end::Float64 = 1.0
)
    filename = "convergence_studies_$(typeof(solver))_" *
               Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * ".txt"

    open(filename, "a") do io
        versioninfo(io)
        println(io, "="^100)
    end

    results = Vector{ConvergenceResults}(undef, length(cases))

    for (i, case) in enumerate(cases)
        # Warmup on a coarse, cheap mesh solely to trigger JIT compilation before timing.
        # Logging is suppressed since the result is discarded and warmup warnings would otherwise be indistinguishable from real ones.
        Base.with_logger(Base.NullLogger()) do
            run_convergence_study(case.fe, [2, 4], [1/4, 1/4], solver, case.id, t_end)
        end

        results[i] = run_convergence_study(case.fe, case.Nx, case.τ, solver, case.id, t_end)

        write_result(filename, results[i])
    end

    return results
end

# ==============================================================================
"""
    write_result(filename, result)

Appends `result` to `filename` in plain-text format.
"""
function write_result(filename::String, result::ConvergenceResults)
    open(filename, "a") do io
        show(io, result)
        println(io)
        flush(io)
    end
    return nothing
end

# ==============================================================================
"""
    run_convergence_study(fe, Nx, τ, solver, id, t_end) -> ConvergenceResults

Runs one PDE solve per entry in `Nx`/`τ`, collects L∞(L²) errors, and returns a [`ConvergenceResults`](@ref).

Three refinement modes are supported, selected by which of `Nx`/`τ` vary:
- **Spatial study**: `Nx` varies, `τ` fixed, e.g. `Nx = [2^i for i in 2:5]`, `τ = 2.0^-11`
- **Temporal study**: `Nx` fixed, `τ` varies, e.g. `Nx = 2^9`, `τ = [2.0^-i for i in 2:5]`
- **Coupled study**: `Nx` and `τ` both vary, e.g. `Nx = [2^i for i in 2:5]`, `h = sqrt{2}./Nx`, `p = 1`, `τ = h.^((p+1)/2)`, where `p` is the basis polynomial degree

# Arguments
- `fe`: finite element basis with only the polynomial degree fixed (e.g. `Lagrange{1}`)
- `Nx`: number of elements per spatial direction; `Int` (fixed) or `Vector{Int}` (refined).
- `τ`: time step; `Float64` (fixed) or `Vector{Float64}` (refined).
- `solver`: time integration scheme, e.g. `Scheme1()`
- `id`: input data container for the PDE system, e.g. `example1_manufactured(4.4)`
- `t_end` : final simulation time.

At least one of `Nx`/`τ` must vary; if both are vectors they must have the same length.

# Returns
- `ConvergenceResults` : L∞(L²) errors and convergence rates for `v`, `d`, `r`, `z`
  across all refinement levels, along with mesh/time-step values and run metadata.


# Example
```julia-repl
julia> using WaveAcoustics

julia> result = run_convergence_study(
           Lagrange{1}, [2, 4, 8, 16], sqrt(2) ./ [2, 4, 8, 16],
           Scheme1(), example1_manufactured(2.4), 1.0)
```
"""
function run_convergence_study(
        fe::Type{FE},
        Nx::Vector{Int},
        τ::Vector{Float64},
        solver::AbstractODESolver,
        id::PDEInputData,
        t_end::Float64
) where {Deg, FE <: AbstractFEBasis{Deg}}
    length(Nx) == length(τ) || throw(ArgumentError(
        "Nx and τ must have the same length (got $(length(Nx)) and $(length(τ)))"
    ))

    pmin, pmax = id.pmin, id.pmax

    n_levels = length(Nx)
    n_fields = 4 # (v, d, r, z)

    errors = zeros(n_levels, n_fields)
    rates = zeros(n_levels, n_fields)
    h_values = zeros(n_levels)
    Δx_values = zeros(n_levels)
    times = zeros(n_levels)

    for i in 1:n_levels
        Δx, Δy = (pmax .- pmin) ./ (Nx[i], Nx[i])
        Δx_values[i] = Δx
        h_values[i] = sqrt(Δx^2 + Δy^2)

        tspan = 0.0:τ[i]:t_end
        callback = L2ErrorCallback(tspan)
        times[i] = @elapsed solve_pde(fe, (Nx[i], Nx[i]), tspan, id, solver, callback)
        errors[i, 1] = maximum(callback.v_errors)
        errors[i, 2] = maximum(callback.d_errors)
        errors[i, 3] = maximum(callback.r_errors)
        errors[i, 4] = maximum(callback.z_errors)
    end

    # Refinement mode determines δ 
    # Temporal (Nx fixed) uses τ for all fields. 
    # Otherwise (Nx varies, spatial or coupled) 2D fields (v, d) use h, and 1D fields (r, z) use Δx.
    temporal = allequal(Nx)
    δ_2D = temporal ? τ : h_values
    δ_1D = temporal ? τ : Δx_values

    compute_rates!(view(rates, :, 1:2), view(errors, :, 1:2), δ_2D)
    compute_rates!(view(rates, :, 3:4), view(errors, :, 3:4), δ_1D)

    total_time = sum(times)
    time_str = total_time ≥ 1.0 ? @sprintf("%.2f s", total_time) :
               @sprintf("%.1f ms", total_time*1e3)

    info = string(
        "ConvergenceResults: t_end=", t_end, ", ", id.name, ", ", fe, ", ", solver, ", elapsed=", time_str)
    return ConvergenceResults(info, Nx, h_values, τ, errors, rates)
end

# Convenience overloads: fix Nx, vary τ (temporal study); or fix τ, vary Nx (spatial study)
function run_convergence_study(
        fe::Type{FE}, Nx::Int, τ::Vector{Float64}, solver::AbstractODESolver,
        id::PDEInputData, t_end::Float64
) where {Deg, FE <: AbstractFEBasis{Deg}}
    run_convergence_study(fe, fill(Nx, length(τ)), τ, solver, id, t_end)
end

function run_convergence_study(
        fe::Type{FE}, Nx::Vector{Int}, τ::Float64, solver::AbstractODESolver,
        id::PDEInputData, t_end::Float64
) where {Deg, FE <: AbstractFEBasis{Deg}}
    run_convergence_study(fe, Nx, fill(τ, length(Nx)), solver, id, t_end)
end

# ==============================================================================
"""
    compute_rates!(rates, errors, δ)

Fills `rates` in-place: `rates[i,j] = log(e[i-1,j]/e[i,j]) / log(δ[i-1]/δ[i])` for `i ≥ 2`.
Row 1 is left as zero (no previous level available).
"""
function compute_rates!(rates::AbstractArray, errors::AbstractArray, δ::Vector)
    n_levels, n_fields = size(errors)
    for j in 1:n_fields, i in 2:n_levels

        rates[i, j] = log(errors[i - 1, j] / errors[i, j]) / log(δ[i - 1] / δ[i])
    end
    return nothing
end

# ==============================================================================
"""
    Base.show(io, r::ConvergenceResults)

Prints a compact table of mesh sizes, errors, and convergence rates.
Invoked automatically by `print`, `display`, and the REPL.
"""
function Base.show(io::IO, r::ConvergenceResults)
    n_levels, n_fields = size(r.errors)
    println(io, r.info)
    println(io,
        "  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate")
    for i in 1:n_levels
        row = @sprintf("%4d  %6.2f  %6.2f", r.Nx[i], log2(r.h[i]), log2(r.τ[i]))
        for j in 1:n_fields
            row *= @sprintf("    %.2e % .3f", r.errors[i, j], r.rates[i, j])
        end
        println(io, row)
    end
end