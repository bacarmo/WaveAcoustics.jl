# ========================================
# Convergence test functions
# ========================================
"""
    convergence_test_coupled(;input_data, solver, Nx_exp_range)

Perform coupled space-time convergence test with τ = h.

Varies spatial and temporal discretization together, assuming linear basis functions.
Convergence rates are computed as log₂(error_{i-1} / error_i) between successive refinements.

# Keyword Arguments
- `input_data`: PDE input data (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration method (default: `CrankNicolson()`)
- `Nx_exp_range`: Grid refinement exponents (default: 3:6 → Nx = 8, 16, 32, 64)

# Returns
`NamedTuple` with fields: `test_info`, `Nx`, `h`, `τ`, `errors`, `rates`
"""
function convergence_test_coupled(;
        input_data = example1_manufactured(),
        solver::ODESolver = CrankNicolson(),
        Nx_exp_range = 3:6)
    Nx_values = [2^i for i in Nx_exp_range]
    h_values = [compute_element_diameter(
                    Nx, input_data.pmin, input_data.pmax)
                for Nx in Nx_values]
    τ_values = h_values

    test_info = "Coupled space-time convergence (τ = h) with $(typeof(solver).name.name)"

    return run_convergence_study(
        Nx_values, h_values, τ_values, input_data, solver, test_info)
end

"""
    convergence_test_spatial(;input_data, solver, Nx_exp_range, τ_fixed)

Perform spatial convergence test with fixed time step.

Varies mesh size h while keeping temporal discretization τ constant to isolate
spatial discretization error.

# Keyword Arguments
- `input_data`: PDE input data (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration method (default: `CrankNicolson()`)
- `Nx_exp_range`: Grid refinement exponents (default: 3:6)
- `τ_fixed::Real`: Fixed time step (default: 2^(-7))

# Returns
`NamedTuple` with fields: `test_info`, `Nx`, `h`, `τ`, `errors`, `rates`
"""
function convergence_test_spatial(;
        input_data = example1_manufactured(),
        solver::ODESolver = CrankNicolson(),
        Nx_exp_range = 3:6,
        τ_fixed::Real = 2^(-7))
    Nx_values = [2^i for i in Nx_exp_range]
    h_values = [compute_element_diameter(
                    Nx, input_data.pmin, input_data.pmax)
                for Nx in Nx_values]
    τ_values = fill(τ_fixed, length(Nx_values))

    test_info = "Spatial convergence (τ = $τ_fixed fixed) with $(typeof(solver).name.name)"

    return run_convergence_study(
        Nx_values, h_values, τ_values, input_data, solver, test_info)
end

"""
    convergence_test_temporal(;input_data, solver, τ_exp_range, Nx_fixed)

Perform temporal convergence test with fixed spatial grid.

Varies time step τ while keeping mesh size h constant to isolate temporal
discretization error.

# Keyword Arguments
- `input_data`: PDE input data (default: `example1_manufactured()`)
- `solver::ODESolver`: Time integration method (default: `CrankNicolson()`)
- `τ_exp_range`: Time step refinement exponents (default: 3:6 → τ = 2^(-3), ..., 2^(-6))
- `Nx_fixed::Int`: Fixed grid size per dimension (default: 2^8)

# Returns
`NamedTuple` with fields: `test_info`, `Nx`, `h`, `τ`, `errors`, `rates`
"""
function convergence_test_temporal(;
        input_data = example1_manufactured(),
        solver::ODESolver = CrankNicolson(),
        τ_exp_range = 3:6,
        Nx_fixed::Int = 2^8)
    τ_values = [1 / 2^i for i in τ_exp_range]
    n_refinements = length(τ_values)

    Nx_values = fill(Nx_fixed, n_refinements)
    h_fixed = compute_element_diameter(
        Nx_fixed, input_data.pmin, input_data.pmax)
    h_values = fill(h_fixed, n_refinements)

    test_info = "Temporal convergence (Nx = $Nx_fixed, h ≈ $(round(h_fixed; sigdigits=4)) fixed) with $(typeof(solver).name.name)"

    return run_convergence_study(
        Nx_values, h_values, τ_values, input_data, solver, test_info)
end

# ========================================
# Helper functions
# ========================================
"""
    compute_element_diameter(Nx, pmin, pmax)

Compute element diameter h = √(Δx² + Δy²) for uniform rectangular mesh.
"""
function compute_element_diameter(
        Nx::I, pmin::NTuple{2, T}, pmax::NTuple{2, T}) where {
        I <: Integer, T <: AbstractFloat}
    Lx = pmax[1] - pmin[1]
    Ly = pmax[2] - pmin[2]

    Δx = Lx / Nx
    Δy = Ly / Nx

    return sqrt(Δx^2 + Δy^2)
end

"""
    run_convergence_study(Nx_values, h_values, τ_values, input_data, solver, test_info)

Execute convergence study computing error norms and convergence rates.

Solves the PDE for each refinement level using the specified `solver`, computes 
L∞(L²) error norms for all fields, and calculates convergence rates as 
log₂(error_{i-1} / error_i) between successive refinements.
"""
function run_convergence_study(
        Nx_values, h_values, τ_values, input_data,
        solver::ODESolver, test_info::String)
    n_refinements = length(Nx_values)

    # Preallocate error storage
    errors = (
        v = zeros(Float64, n_refinements),
        d = zeros(Float64, n_refinements),
        r = zeros(Float64, n_refinements),
        z = zeros(Float64, n_refinements)
    )

    rates = (
        v = zeros(Float64, n_refinements),
        d = zeros(Float64, n_refinements),
        r = zeros(Float64, n_refinements),
        z = zeros(Float64, n_refinements)
    )

    # Convergence loop
    for i in eachindex(Nx_values)
        Nx = Nx_values[i]
        τ = τ_values[i]

        # Solve PDE with specified solver
        output_data = pde_solve((Nx, Nx), τ, input_data, solver, ConvergenceStudy())

        errors.v[i] = maximum(output_data.v_errors)
        errors.d[i] = maximum(output_data.d_errors)
        errors.r[i] = maximum(output_data.r_errors)
        errors.z[i] = maximum(output_data.z_errors)

        # Compute convergence rates (log₂ ratio)
        if i > 1
            rates.v[i] = log2(errors.v[i - 1] / errors.v[i])
            rates.d[i] = log2(errors.d[i - 1] / errors.d[i])
            rates.r[i] = log2(errors.r[i - 1] / errors.r[i])
            rates.z[i] = log2(errors.z[i - 1] / errors.z[i])
        end
    end

    return (
        test_info = test_info,
        Nx = Nx_values,
        h = h_values,
        τ = τ_values,
        errors = errors,
        rates = rates
    )
end

"""
    print_convergence_table(results)

Display formatted convergence table from convergence test results.

# Examples
```julia
results = convergence_test_coupled()
print_convergence_table(results)
```
"""
function print_convergence_table(results)
    println("\n" * "="^110)
    println(results.test_info)
    println("="^110)
    @printf("   Nx    log₂(h)  log₂(τ)   L∞L²_v     rate_v    L∞L²_d     rate_d    L∞L²_r     rate_r    L∞L²_z     rate_z\n")
    println("-"^110)

    for i in eachindex(results.Nx)
        @printf("%5d   %7.2f  %7.2f  %10.2e  %7.3f  %10.2e  %7.3f  %10.2e  %7.3f  %10.2e  %7.3f\n",
            results.Nx[i],
            log2(results.h[i]),
            log2(results.τ[i]),
            results.errors.v[i], results.rates.v[i],
            results.errors.d[i], results.rates.d[i],
            results.errors.r[i], results.rates.r[i],
            results.errors.z[i], results.rates.z[i])
    end

    println("="^110)
end