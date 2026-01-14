"""
    convergence_test(test_type::Symbol; kwargs...)

Perform convergence analysis for PDE solvers using method of manufactured solutions.

Three test strategies are available: `:coupled` varies spatial and temporal discretization
together with τ = h (assuming linear basis functions); `:spatial` varies mesh size h with
fixed time step τ; `:temporal` varies time step τ with fixed mesh size h. Convergence
rates are computed as log₂(error_{i-1} / error_i) between successive refinements.

# Arguments
- `test_type::Symbol`: Convergence test strategy - `:coupled`, `:spatial`, or `:temporal`

# Keyword Arguments
- `Nx_exp_range`: Exponents for grid refinement (e.g., 3:6 gives Nx = 8, 16, 32, 64). Used by `:coupled` and `:spatial` tests
- `τ_fixed`: Fixed time step. Used by `:spatial` test
- `τ_exp_range`: Exponents for time step refinement (e.g., 3:6 gives τ = 2⁻³, 2⁻⁴, 2⁻⁵, 2⁻⁶). Used by `:temporal` test
- `Nx_fixed`: Number of elements per dimension (Ny = Nx assumed). Used by `:temporal` test
- `input_data`: PDE system configuration struct (must contain `pmin` and `pmax` for domain bounds)

# Returns
`NamedTuple` with fields:
- `test_info`: Description of the test performed
- `Nx`: Vector of grid sizes used
- `h`: Vector of element diameters h = √(Δx² + Δy²)
- `τ`: Vector of time steps used
- `errors`: Named tuple of L∞(L²) error norms for wave and acoustic fields (v=velocity, d=displacement, r=acoustic position, z=acoustic velocity)
- `rates`: Named tuple of convergence rates for each field

# Examples
```julia
# Coupled space-time convergence with τ = h
results = convergence_test(:coupled, Nx_exp_range=3:6)

# Spatial convergence with fixed time step
results = convergence_test(:spatial, Nx_exp_range=3:6, τ_fixed=2^(-7))

# Temporal convergence with fixed spatial grid
results = convergence_test(:temporal, τ_exp_range=3:6, Nx_fixed=2^7)

# Display results
print_convergence_table(results)
```

# Extended help
The element diameter h is computed for uniform rectangular meshes as the diagonal length
of each element: h = √(Δx² + Δy²), where Δx and Δy are element dimensions in x and y
directions. The implementation assumes uniform mesh refinement with Ny = Nx elements per
dimension, though the physical domain can be non-square (shape determined by `pmin` and
`pmax` in `input_data`).

For the `:coupled` test, the relationship τ = h is specifically designed for linear
Lagrange finite elements.
"""
function convergence_test(test_type::Symbol;
        Nx_exp_range = 3:6,
        τ_fixed::Real = 2^(-7),
        τ_exp_range = 3:6,
        Nx_fixed::Int = 2^7,
        input_data = example1_manufactured())

    # Validate domain specification
    if !hasfield(typeof(input_data.common), :pmin) ||
       !hasfield(typeof(input_data.common), :pmax)
        throw(ArgumentError("input_data must have pmin and pmax fields for domain bounds"))
    end

    if test_type == :coupled
        return _convergence_test_coupled(Nx_exp_range, input_data)
    elseif test_type == :spatial
        return _convergence_test_spatial(Nx_exp_range, τ_fixed, input_data)
    elseif test_type == :temporal
        return _convergence_test_temporal(τ_exp_range, Nx_fixed, input_data)
    else
        throw(ArgumentError("test_type must be :coupled, :spatial, or :temporal, got :$test_type"))
    end
end

"""
    print_convergence_table(results)

Display formatted convergence table from convergence test results.

# Arguments
- `results`: NamedTuple returned by `convergence_test` containing test results

# Examples
```julia
results = convergence_test(:coupled, Nx_exp_range=3:6)
print_convergence_table(results)
```
"""
function print_convergence_table(results)
    println("\n" * "="^125)
    println(results.test_info)
    println("="^125)
    @printf("   Nx    log₂(h)  log₂(τ)   L∞L²_v     rate_v    L∞L²_d     rate_d    L∞L²_r     rate_r    L∞L²_z     rate_z\n")
    println("-"^125)

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

    println("="^125)
end

"""
    _compute_element_diameter(Nx::Int, pmin, pmax) -> Float64

Compute element diameter h = √(Δx² + Δy²) for uniform rectangular mesh.

Element dimensions Δx and Δy are determined by domain size (pmax - pmin) divided
by number of elements Nx in each direction.
"""
function _compute_element_diameter(Nx::Int, pmin, pmax)
    Lx = pmax[1] - pmin[1]
    Ly = pmax[2] - pmin[2]

    Δx = Lx / Nx
    Δy = Ly / Nx

    return sqrt(Δx^2 + Δy^2)
end

"""
    _convergence_test_coupled(Nx_exp_range, input_data) -> NamedTuple

Execute coupled space-time convergence test with τ = h.
"""
function _convergence_test_coupled(Nx_exp_range, input_data)
    Nx_values = [2^i for i in Nx_exp_range]
    h_values = [_compute_element_diameter(
                    Nx, input_data.common.pmin, input_data.common.pmax)
                for Nx in Nx_values]
    τ_values = h_values  # Coupled refinement

    test_info = "Coupled space-time convergence (τ = h)"

    return _run_convergence_study(Nx_values, h_values, τ_values, input_data, test_info)
end

"""
    _convergence_test_spatial(Nx_exp_range, τ_fixed, input_data) -> NamedTuple

Execute spatial convergence test with fixed time step.

Varies mesh size h while keeping temporal discretization τ constant to isolate
spatial discretization error.
"""
function _convergence_test_spatial(Nx_exp_range, τ_fixed::Real, input_data)
    Nx_values = [2^i for i in Nx_exp_range]
    h_values = [_compute_element_diameter(Nx, input_data.pmin, input_data.pmax)
                for Nx in Nx_values]
    τ_values = fill(τ_fixed, length(Nx_values))

    test_info = "Spatial convergence (τ = $(τ_fixed) fixed)"

    return _run_convergence_study(Nx_values, h_values, τ_values, input_data, test_info)
end

"""
    _convergence_test_temporal(τ_exp_range, Nx_fixed, input_data) -> NamedTuple

Execute temporal convergence test with fixed spatial grid.

Varies time step τ while keeping mesh size h constant to isolate temporal
discretization error.
"""
function _convergence_test_temporal(τ_exp_range, Nx_fixed::Int, input_data)
    τ_values = [2^(-i) for i in τ_exp_range]
    n_refinements = length(τ_values)

    Nx_values = fill(Nx_fixed, n_refinements)
    h_fixed = _compute_element_diameter(Nx_fixed, input_data.pmin, input_data.pmax)
    h_values = fill(h_fixed, n_refinements)

    test_info = "Temporal convergence (Nx = $Nx_fixed, h ≈ $(round(h_fixed; sigdigits=4)) fixed)"

    return _run_convergence_study(Nx_values, h_values, τ_values, input_data, test_info)
end

"""
    _run_convergence_study(Nx_values, h_values, τ_values, input_data, test_info) -> NamedTuple

Execute convergence study computing error norms and convergence rates.

Solves the PDE for each refinement level, computes L∞(L²) error norms for all fields,
and calculates convergence rates as log₂(error_{i-1} / error_i) between successive
refinements.
"""
function _run_convergence_study(
        Nx_values, h_values, τ_values, input_data, test_info::String)
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

        # Solve PDE and collect error norms
        err_v, err_d, err_r, err_z = pde_solve((Nx, Nx), τ, input_data)

        errors.v[i] = err_v
        errors.d[i] = err_d
        errors.r[i] = err_r
        errors.z[i] = err_z

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