# ========================================
# Type definitions
# ========================================
abstract type ODESolver end
struct CrankNicolson <: ODESolver end
struct CrankNicolsonLinearized <: ODESolver end

abstract type OutputMode end
struct ConvergenceStudy <: OutputMode end
struct SolutionHistory <: OutputMode end
struct EnergyHistory <: OutputMode end

# ========================================
# Data containers
# ========================================
struct ConvergenceStudyData{T <: Real}
    v_errors::Vector{T} # length nt
    d_errors::Vector{T} # length nt
    r_errors::Vector{T} # length nt
    z_errors::Vector{T} # length nt
end

struct SolutionHistoryData{T <: Real, R <: AbstractVector{T}}
    d_history::Matrix{T}  # (m₁, nt)
    z_history::Matrix{T}  # (m₂, nt)
    times::R              # length nt
end

struct EnergyHistoryData{T <: Real, R <: AbstractVector{T}}
    energy::Vector{T} # length nt
    times::R          # length nt
end

# ========================================
# Initialize output data
# ========================================
function initialize_output(
        ::ConvergenceStudy, m₁::Integer, m₂::Integer, times::AbstractVector{T}) where {T}
    nt = length(times)
    return ConvergenceStudyData(
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt)
    )
end

function initialize_output(
        ::SolutionHistory, m₁::Integer, m₂::Integer, times::AbstractVector{T}) where {T}
    nt = length(times)
    return SolutionHistoryData(
        Matrix{T}(undef, m₁, nt),
        Matrix{T}(undef, m₂, nt),
        times
    )
end

function initialize_output(
        ::EnergyHistory, m₁::Integer, m₂::Integer, times::AbstractVector{T}) where {T}
    return EnergyHistoryData(zeros(T, length(times)), times)
end

# ========================================
# Process solution at each time step
# ========================================
"""
Process solution for convergence study: compute L² errors.
"""
function process_solution!(
        data::ConvergenceStudyData,
        ::ConvergenceStudy,
        n::Integer,
        tₙ::T,
        vⁿ::AbstractVector{T},
        dⁿ::AbstractVector{T},
        rⁿ::AbstractVector{T},
        zⁿ::AbstractVector{T},
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data
) where {T <: AbstractFloat}
    data.v_errors[n + 1] = L2_error_2d(
        (x, y) -> input_data.v(x, y, tₙ), vⁿ, mesh2D, dof_map_m₁, quad)
    data.d_errors[n + 1] = L2_error_2d(
        (x, y) -> input_data.u(x, y, tₙ), dⁿ, mesh2D, dof_map_m₁, quad)
    data.r_errors[n + 1] = L2_error_1d(
        x -> input_data.r(x, tₙ), rⁿ, mesh1D, dof_map_m₂, quad)
    data.z_errors[n + 1] = L2_error_1d(
        x -> input_data.z(x, tₙ), zⁿ, mesh1D, dof_map_m₂, quad)
    return nothing
end

"""
Process solution for solution history: store current solution.
"""
function process_solution!(
        data::SolutionHistoryData,
        ::SolutionHistory,
        n::Integer,
        tₙ::T,
        vⁿ::AbstractVector{T},
        dⁿ::AbstractVector{T},
        rⁿ::AbstractVector{T},
        zⁿ::AbstractVector{T},
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data
) where {T <: AbstractFloat}
    data.d_history[:, n + 1] .= dⁿ
    data.z_history[:, n + 1] .= zⁿ
    return nothing
end

"""
Process solution for energy history: compute energy functional.
"""
function process_solution!(
        data::EnergyHistoryData,
        ::EnergyHistory,
        n::Integer,
        tₙ::T,
        vⁿ::AbstractVector{T},
        dⁿ::AbstractVector{T},
        rⁿ::AbstractVector{T},
        zⁿ::AbstractVector{T},
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data
) where {T <: AbstractFloat}
    # TODO: In construction
    # Placeholder:
    data.energy[n + 1] = zero(T)
    return nothing
end

# ========================================
# Main solver function
# ========================================
"""
    pde_solve(Nx, τ, input_data)

Solve coupled wave-acoustic PDE system and compute L∞(0,T;L²) errors.

## Arguments
- `Nx::NTuple{2,Integer}`: Number of elements per direction
- `τ::Real`: Time step size
- `input_data::PDEInputData`: Problem configuration with manufactured solution 
- `solver::ODESolver`: Time integration scheme (default: `CrankNicolson()`)
- `output::OutputMode`: Output mode (default: `ConvergenceStudy()`)
  - `ConvergenceStudy()` → `ConvergenceStudyData` with L² errors at each time step
  - `SolutionHistory()` → `SolutionHistoryData` with displacement snapshots
  - `EnergyHistory()` → `EnergyHistoryData` with energy evolution

# Returns
Depends on `output`:
- `ConvergenceStudyData{T}`: Fields `v_errors`, `d_errors`, `r_errors`, `z_errors` (each a vector of length `nt`)
- `SolutionHistoryData{T}`: Fields `d_history` (m₁×nt), `z_history` (m₂×nt), `times` (length `nt`)
- `EnergyHistoryData{T}`: Fields `energy`, `times` (each length `nt`)
"""
function pde_solve(
        Nx::NTuple{2, I},
        τ::T,
        input_data,
        solver::S = CrankNicolson(),
        output::O = ConvergenceStudy()
) where {I <: Integer, T <: Real, S <: ODESolver, O <: OutputMode}
    # ========================================
    # Spatial discretization
    # ========================================
    pmin, pmax = input_data.common.pmin, input_data.common.pmax

    mesh2D = CartesianMesh(pmin, pmax, Nx)
    mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (Nx[1],))

    # ========================================
    # Quadrature setup (Gauss-Legendre)
    # ========================================
    quad = QuadratureSetup(mesh2D.Δx, pmin)

    # ========================================
    # Local-to-global mapping
    # ========================================
    dof_map_m₁ = DOFMap(mesh2D, Lagrange{2, 1}(), LeftRightTop())
    dof_map_m₂ = DOFMap(mesh1D, Lagrange{1, 1}(), LeftRight())

    m₁ = dof_map_m₁.m
    m₂ = dof_map_m₂.m

    # ========================================
    # Assemble local matrices
    # ========================================
    Me_m₁xm₁ = assembly_local_matrix_ϕxϕ(mesh2D, Lagrange{2, 1}())
    Ke_m₁xm₁ = assembly_local_matrix_∇ϕx∇ϕ(mesh2D, Lagrange{2, 1}())
    Me_m₂xm₂ = assembly_local_matrix_ϕxϕ(mesh1D, Lagrange{1, 1}())

    Me_m₁xm₁_sym = Symmetric(Me_m₁xm₁)
    Ke_m₁xm₁_sym = Symmetric(Ke_m₁xm₁)
    Me_m₂xm₂_sym = Symmetric(Me_m₂xm₂)

    # ========================================
    # Assemble global matrices
    # ========================================
    M_m₁xm₁ = assembly_global_matrix(Me_m₁xm₁_sym, dof_map_m₁)
    K_m₁xm₁ = assembly_global_matrix(Ke_m₁xm₁_sym, dof_map_m₁)
    M_m₂xm₂ = assembly_global_matrix(Me_m₂xm₂_sym, dof_map_m₂)

    M_m₁xm₂ = [M_m₂xm₂; spzeros(m₁ - m₂, m₂)]
    M_m₂xm₁ = [M_m₂xm₂ spzeros(m₂, m₁ - m₂)]

    matrices = (
        M_m₁xm₁ = M_m₁xm₁,
        K_m₁xm₁ = K_m₁xm₁,
        M_m₂xm₂ = M_m₂xm₂,
        M_m₁xm₂ = M_m₁xm₂,
        M_m₂xm₁ = M_m₂xm₁)

    # ========================================
    # Compute v⁰ and d⁰
    # ========================================
    v⁰ = zeros(T, m₁)
    d⁰ = zeros(T, m₁)
    compute_v⁰_d⁰!(v⁰, d⁰, K_m₁xm₁, input_data, mesh2D, dof_map_m₁, quad)

    # ========================================
    # Compute r⁰ and z⁰
    # ========================================
    r⁰ = zeros(T, m₂)
    z⁰ = zeros(T, m₂)
    compute_r⁰_z⁰!(r⁰, z⁰, M_m₂xm₂, input_data, mesh1D, dof_map_m₂, quad)

    # ========================================
    # Compute vⁿ, dⁿ, rⁿ, and zⁿ for n ≥ 1
    # ========================================
    result = solve_ode(
        solver, output, v⁰, d⁰, r⁰, z⁰, τ, input_data, mesh1D, mesh2D,
        dof_map_m₁, dof_map_m₂, quad, matrices)

    return result
end

# ========================================
# ODE solver dispatch methods
# ========================================
"""
    solve_ode(::CrankNicolson, output, v⁰, d⁰, r⁰, z⁰, τ, ...)

Crank-Nicolson time integration for the semi-discrete system.
"""
function solve_ode(
        ::CrankNicolson,
        output::OutputMode,
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
    return crank_nicolson(
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices,
        output)
end

"""
    solve_ode(::CrankNicolsonLinearized, output, v⁰, d⁰, r⁰, z⁰, τ, ...)

Linearized Crank-Nicolson time integration for the semi-discrete system.
"""
function solve_ode(
        ::CrankNicolsonLinearized,
        output::OutputMode,
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
    return crank_nicolson_linearized(
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices,
        output)
end