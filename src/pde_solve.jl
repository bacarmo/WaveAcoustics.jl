# ========================================
# Type definitions
# ========================================
abstract type ODESolver end

struct CrankNicolson <: ODESolver end
struct CrankNicolsonLinearized <: ODESolver end

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

## Returns
- `(LooL2_v, LooL2_u, LooL2_r, LooL2_z)`: L∞(0,T;L²) errors
"""
function pde_solve(
        Nx::NTuple{2, I},
        τ::T,
        input_data,
        solver::ODESolver = CrankNicolson()
) where {I <: Integer, T <: Real}
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
    L2_error = solve_ode(
        solver, v⁰, d⁰, r⁰, z⁰, τ, input_data, mesh1D, mesh2D,
        dof_map_m₁, dof_map_m₂, quad, matrices)

    return (
        maximum(L2_error.v), maximum(L2_error.d), maximum(L2_error.r), maximum(L2_error.z))
end

# ========================================
# ODE solver dispatch methods
# ========================================
"""
    solve_ode(::CrankNicolson, v⁰, d⁰, r⁰, z⁰, τ, ...)

Crank-Nicolson time integration for the semi-discrete system.
"""
function solve_ode(
        ::CrankNicolson,
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
    return crank_nicolson(
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
end

"""
    solve_ode(::CrankNicolsonLinearized, v⁰, d⁰, r⁰, z⁰, τ, ...)

Linearized Crank-Nicolson time integration for the semi-discrete system.
"""
function solve_ode(
        ::CrankNicolsonLinearized,
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
    return crank_nicolson_linearized(
        v⁰, d⁰, r⁰, z⁰, τ, input_data,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices)
end