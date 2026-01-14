"""
    pde_solve(Nx, τ, input_data)

Solve coupled wave-acoustic PDE system and compute L∞(0,T;L²) errors.

## Arguments
- `Nx::NTuple{2,Integer}`: Number of elements per direction
- `τ::Real`: Time step size
- `input_data::PDEInputData`: Problem configuration with manufactured solution 

## Returns
- `(LooL2_v, LooL2_u, LooL2_r, LooL2_z)`: L∞(0,T;L²) errors
"""
function pde_solve(
        Nx::NTuple{2, I},
        τ::T,
        input_data
) where {I <: Integer, T <: Real}
    # ========================================
    # Spatial and temporal discretization
    # ========================================
    common = input_data.common
    pmin, pmax = common.pmin, common.pmax

    mesh2D = CartesianMesh(pmin, pmax, Nx)
    mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (Nx[1],))

    times = range(zero(T), common.t_final; step = τ)
    nt = length(times)

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

    Me_m₁xm₁ = Symmetric(Me_m₁xm₁)
    Ke_m₁xm₁ = Symmetric(Ke_m₁xm₁)
    Me_m₂xm₂ = Symmetric(Me_m₂xm₂)

    # ========================================
    # Assemble global matrices
    # ========================================
    M_m₁xm₁ = assembly_global_matrix(Me_m₁xm₁, dof_map_m₁)
    K_m₁xm₁ = assembly_global_matrix(Ke_m₁xm₁, dof_map_m₁)
    M_m₂xm₂ = assembly_global_matrix(Me_m₂xm₂, dof_map_m₂)

    M_m₁xm₂ = [M_m₂xm₂; spzeros(m₁ - m₂, m₂)]
    M_m₂xm₁ = [M_m₂xm₂ spzeros(m₂, m₁ - m₂)]

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
    # Compute L2 error
    # ========================================
    L2_error_v = zeros(T, nt)
    L2_error_d = zeros(T, nt)
    L2_error_r = zeros(T, nt)
    L2_error_z = zeros(T, nt)

    L2_error_v[1] = L2_error_2d(input_data.common.v₀, v⁰, mesh2D, dof_map_m₁, quad)
    L2_error_d[1] = L2_error_2d(input_data.common.u₀, d⁰, mesh2D, dof_map_m₁, quad)

    L2_error_r[1] = L2_error_1d(input_data.common.r₀, r⁰, mesh1D, dof_map_m₂, quad)
    L2_error_z[1] = L2_error_1d(input_data.common.z₀, z⁰, mesh1D, dof_map_m₂, quad)

    return (
        maximum(L2_error_v), maximum(L2_error_d), maximum(L2_error_r), maximum(L2_error_z))
end