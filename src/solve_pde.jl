"""
    solve_pde(fe, nel_per_dim, tspan, input_data, solver, callback)

Solve the PDE system, populating `callback` in-place.

# Arguments
- `fe::Type{FE}`: finite element basis with only the polynomial degree fixed (e.g. `Lagrange{1}`)
- `nel_per_dim::NTuple{2, Int}`: number of elements per spatial direction (e.g. `(4, 4)`)
- `tspan::StepRangeLen{Float64}`: uniform time grid (e.g. `0:0.1:1`)
- `input_data::PDEInputData`: input data container for the PDE system (e.g. `example1_manufactured(2.4)`) 
- `solver::AbstractODESolver`: time integration scheme (e.g. `Scheme1()`)
- `callback::AbstractCallback`: callback invoked at each time step (e.g. `L2ErrorCallback(tspan)`)
"""
function solve_pde(
        fe::Type{FE},
        nel_per_dim::NTuple{2, Int},
        tspan::StepRangeLen{Float64},
        input_data::PDEInputData,
        solver::AbstractODESolver,
        callback::AbstractCallback
) where {Deg, FE <: AbstractFEBasis{Deg}}
    # ========================================
    # Spatial discretization
    # ========================================
    Nx, Ny = nel_per_dim
    pmin, pmax = input_data.pmin, input_data.pmax
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx, Δy = element_side_lengths

    # ========================================
    # Specialize FEBasis
    # ========================================
    fe1D = fe{1}()::Lagrange{Deg, 1}
    fe2D = fe{2}()::Lagrange{Deg, 2}

    # ========================================
    # Local-to-global mapping
    # ========================================
    dof_map_m₁ = DOFMap(fe2D, LeftRightTop(), nel_per_dim)
    dof_map_m₂ = DOFMap(fe1D, AllSides(), (Nx,))

    # ========================================
    # Assemble matrices
    # ========================================
    matrices = SystemMatrices(
        fe1D, fe2D, element_side_lengths, dof_map_m₁, dof_map_m₂)

    # ========================================
    # Quadrature setup (Gauss-Legendre)
    # ========================================
    quad = QuadratureSetup(fe1D, fe2D, element_side_lengths, pmin)

    # ========================================
    # Compute initial state
    # ========================================
    state0 = compute_initial_state(
        input_data, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, matrices, quad)

    apply!(
        callback, state0, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data)

    # ========================================
    # Compute state for n ≥ 1
    # ========================================
    cache = build_cache(solver, matrices)
    solve_ode(
        cache, tspan, state0, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data, callback)

    return nothing
end

function compute_initial_state(
        input_data, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, matrices, quad)
    Δx, Δy = element_side_lengths

    m₁, m₂ = dof_map_m₁.m, dof_map_m₂.m
    v⁰, d⁰ = zeros(m₁), zeros(m₁)
    r⁰, z⁰ = zeros(m₂), zeros(m₂)
    vec_m₁, vec_m₂ = zeros(m₁), zeros(m₂)

    factorized_K_m₁xm₁ = cholesky(matrices.K_m₁xm₁)
    factorized_M_m₂xm₂ = cholesky(matrices.M_m₂xm₂)

    projection_H01!(
        v⁰, input_data.∂ₓv₀, input_data.∂ᵧv₀, nel_per_dim,
        element_side_lengths, dof_map_m₁, factorized_K_m₁xm₁,
        quad.xP, quad.yP, quad.W_∂φ∂ξP, quad.W_∂φ∂ηP, vec_m₁)
    projection_H01!(
        d⁰, input_data.∂ₓu₀, input_data.∂ᵧu₀, nel_per_dim,
        element_side_lengths, dof_map_m₁, factorized_K_m₁xm₁,
        quad.xP, quad.yP, quad.W_∂φ∂ξP, quad.W_∂φ∂ηP, vec_m₁)
    projection_L2!(
        r⁰, input_data.r₀, (Δx,), dof_map_m₂, factorized_M_m₂xm₂,
        quad.xP, quad.W_ϕP, vec_m₂)
    projection_L2!(
        z⁰, input_data.z₀, (Δx,), dof_map_m₂, factorized_M_m₂xm₂,
        quad.xP, quad.W_ϕP, vec_m₂)

    return State(0, 0.0, v⁰, d⁰, r⁰, z⁰)
end
