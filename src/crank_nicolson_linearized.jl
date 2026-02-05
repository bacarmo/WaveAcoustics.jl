"""
    crank_nicolson_linearized(v⁰, d⁰, r⁰, z⁰, τ, input_data, mesh1D, mesh2D, 
                              dof_map_m₁, dof_map_m₂, quad, matrices)

Solve coupled wave-acoustic PDE system using linearized Crank-Nicolson time integration.

# Arguments
- `v⁰::AbstractVector{T}`: Initial wave velocity in Ω
- `d⁰::AbstractVector{T}`: Initial wave displacement in Ω
- `r⁰::AbstractVector{T}`: Initial acoustic velocity on Γ₁
- `z⁰::AbstractVector{T}`: Initial acoustic displacement on Γ₁
- `τ::T`: Time step size
- `input_data`: Problem configuration with manufactured solution
- `mesh1D::CartesianMesh{1}`: 1D mesh for boundary Γ₁
- `mesh2D::CartesianMesh{2}`: 2D mesh for domain Ω
- `dof_map_m₁::DOFMap`: DOF mapping for 2D finite element space
- `dof_map_m₂::DOFMap`: DOF mapping for 1D boundary space
- `quad::QuadratureSetup`: Precomputed quadrature data
- `matrices`: Preassembled global matrices

# Returns
Named tuple containing L² errors at each time step:
- `v`: Wave velocity errors in Ω
- `d`: Wave displacement errors in Ω
- `r`: Acoustic velocity errors on Γ₁
- `z`: Acoustic displacement errors on Γ₁

# Algorithm
- Implements second-order accurate linearized Crank-Nicolson scheme. Nonlinear terms 
  `f(u)` and `g(x,v)`, along with the coupling term `q₄v` in the acoustic equation, 
  are evaluated using values from previous time step(s) to decouple the system and 
  avoid implicit nonlinear solves.
- **Temporal refinement requirement**: Preliminary tests suggest the algorithm requires 
  sufficiently refined time steps for stability. The necessary refinement may be related 
  to the variation of the boundary nonlinearity `g`. Further investigation needed.
- **Note on validation strategy**: During implementation validation, the exact solution 
  was temporarily substituted for `v_ast_n` in the nonlinear term `g(x,v)`, which 
  allowed verification of all other algorithmic components. The current implementation 
  correctly uses the extrapolated velocity `v_ast_n` as specified in the algorithm. 
  Convergence analysis with fixed, well-refined τ demonstrates correct behavior; issues 
  observed in earlier tests with coupled refinement (h = τ) require further analysis.

# Examples

Spatial convergence with τ = 2⁻¹² (well-refined) demonstrates optimal second-order convergence:
```julia
julia> results = convergence_test_spatial(input_data=example1_manufactured(), solver=CrankNicolsonLinearized(), τ_fixed=2^(-12));
julia> print_convergence_table(results)
==============================================================================================================
Spatial convergence (τ = 0.000244140625 fixed) with CrankNicolsonLinearized
==============================================================================================================
   Nx    log₂(h)  log₂(τ)   L∞L²_v     rate_v    L∞L²_d     rate_d    L∞L²_r     rate_r    L∞L²_z     rate_z
--------------------------------------------------------------------------------------------------------------
    8     -2.50   -12.00    4.68e-03    0.000    1.12e-02    0.000    1.50e-02    0.000    1.14e-02    0.000
   16     -3.50   -12.00    1.18e-03    1.983    2.80e-03    2.005    3.61e-03    2.057    2.79e-03    2.036
   32     -4.50   -12.00    2.97e-04    1.994    6.99e-04    2.002    8.88e-04    2.022    6.92e-04    2.012
   64     -5.50   -12.00    7.44e-05    1.998    1.75e-04    2.001    2.21e-04    2.009    1.72e-04    2.004
==============================================================================================================
```

Insufficient temporal refinement (τ = 2⁻¹⁰) leads to algorithm failure for `example1_manufactured()`:
```julia
julia> results = convergence_test_spatial(input_data=example1_manufactured(), solver=CrankNicolsonLinearized(), τ_fixed=2^(-10));
julia> print_convergence_table(results)
==============================================================================================================
Spatial convergence (τ = 0.0009765625 fixed) with CrankNicolsonLinearized
==============================================================================================================
   Nx    log₂(h)  log₂(τ)   L∞L²_v     rate_v    L∞L²_d     rate_d    L∞L²_r     rate_r    L∞L²_z     rate_z
--------------------------------------------------------------------------------------------------------------
    8     -2.50   -10.00    4.68e-03    0.000    1.12e-02    0.000    1.50e-02    0.000    1.14e-02    0.000
   16     -3.50   -10.00    1.18e-03    1.983    2.80e-03    2.005    3.61e-03    2.057    2.79e-03    2.036
   32     -4.50   -10.00    6.73e-02   -5.830    2.15e-03    0.380    4.82e-03   -0.419    2.54e-03    0.138
   64     -5.50   -10.00         NaN      NaN         NaN      NaN         NaN      NaN         NaN      NaN
==============================================================================================================
```

However, the same temporal refinement (τ = 2⁻¹⁰) is sufficient for 
`example2_manufactured()`, suggesting sensitivity to the specific form of `g`:
```julia
julia> results = convergence_test_spatial(input_data=example2_manufactured(), solver=CrankNicolsonLinearized(), τ_fixed=2^(-10));
julia> print_convergence_table(results)
==============================================================================================================
Spatial convergence (τ = 0.0009765625 fixed) with CrankNicolsonLinearized
==============================================================================================================
   Nx    log₂(h)  log₂(τ)   L∞L²_v     rate_v    L∞L²_d     rate_d    L∞L²_r     rate_r    L∞L²_z     rate_z
--------------------------------------------------------------------------------------------------------------
    8     -2.50   -10.00    4.65e-03    0.000    1.21e-02    0.000    5.05e-03    0.000    6.54e-03    0.000
   16     -3.50   -10.00    1.16e-03    2.000    3.02e-03    2.003    1.22e-03    2.047    1.61e-03    2.026
   32     -4.50   -10.00    2.91e-04    1.999    7.55e-04    2.002    3.02e-04    2.018    4.00e-04    2.007
   64     -5.50   -10.00    7.28e-05    1.999    1.88e-04    2.003    7.50e-05    2.008    9.97e-05    2.002
==============================================================================================================
```
"""
function crank_nicolson_linearized(
        v⁰::AbstractVector{T},
        d⁰::AbstractVector{T},
        r⁰::AbstractVector{T},
        z⁰::AbstractVector{T},
        τ::T,
        input_data,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        matrices) where {T <: Real}
    m₁, m₂ = length(v⁰), length(r⁰)
    @assert m₁ == dof_map_m₁.m
    @assert m₂ == dof_map_m₂.m

    times = range(zero(T), input_data.common.t_final; step = τ)
    nt = length(times)

    # ========================================
    # Compute L2 error for initial solutions
    # ========================================
    L2_error = (
        v = zeros(T, nt),
        d = zeros(T, nt),
        r = zeros(T, nt),
        z = zeros(T, nt)
    )

    L2_error.v[1] = L2_error_2d(input_data.common.v₀, v⁰, mesh2D, dof_map_m₁, quad)
    L2_error.d[1] = L2_error_2d(input_data.common.u₀, d⁰, mesh2D, dof_map_m₁, quad)

    L2_error.r[1] = L2_error_1d(input_data.common.r₀, r⁰, mesh1D, dof_map_m₂, quad)
    L2_error.z[1] = L2_error_1d(input_data.common.z₀, z⁰, mesh1D, dof_map_m₂, quad)

    # ========================================
    # Precompute constants
    # ========================================
    q₁, q₂, q₃, q₄ = input_data.common.q₁, input_data.common.q₂, input_data.common.q₃,
    input_data.common.q₄

    lhs_q₁₂₃ = q₁ + (τ / 2) * q₂ + (τ^2 / 4) * q₃
    cst1 = (q₁ - (τ / 2) * q₂ - (τ^2 / 4) * q₃) / lhs_q₁₂₃
    cst2 = (τ * q₃) / lhs_q₁₂₃
    cst3 = (τ * q₄) / lhs_q₁₂₃
    cst4 = τ / lhs_q₁₂₃
    vec_cst = SVector{4, T}(cst1, cst2, cst3, cst4)

    τ_2 = τ / T(2)
    τ_4 = τ / T(4)
    τ²_4 = τ * τ / T(4)

    # ========================================
    # Allocations
    # ========================================
    cache = CNLCache(m₁, m₂, T)

    M_m₂xm₂_factorized = cholesky(matrices.M_m₂xm₂)
    A = similar(matrices.M_m₁xm₁)

    r¹ = zeros(T, m₂)
    v¹ = zeros(T, m₁)

    v_ast_n = zeros(T, m₁)
    d_ast_n = zeros(T, m₁)

    # ========================================
    # Solution at time t₁
    # ========================================
    # Predict solution
    t_half = τ_2
    compute_rⁿ!(r¹, r⁰, z⁰, v⁰, vec_cst, M_m₂xm₂_factorized,
        input_data.f₂, t_half, mesh1D, dof_map_m₂, quad, cache)
    compute_vⁿ!(v¹, v⁰, d⁰, r¹, r⁰, v⁰, d⁰,
        input_data.common.α, input_data.common.g, input_data.common.f, input_data.f₁,
        t_half, τ, τ_4, τ²_4,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices, A, cache)

    # Correct solution
    @. v_ast_n = T(0.5) * v¹ + T(0.5) * v⁰
    @. d_ast_n = τ_2 * v_ast_n + d⁰ # == (d¹ + d⁰)/2

    compute_rⁿ!(r¹, r⁰, z⁰, v_ast_n, vec_cst, M_m₂xm₂_factorized,
        input_data.f₂, t_half, mesh1D, dof_map_m₂, quad, cache)
    ########################################################### TEST
    # v_ast_n_TEMP = zeros(T, m₂)
    # Δx = mesh1D.Δx[1]
    # @. v_ast_n_TEMP = input_data.v(Δx:Δx:(1 - Δx), 0.0, τ_2)
    ###########################################################
    compute_vⁿ!(v¹, v⁰, d⁰, r¹, r⁰,
        v_ast_n, d_ast_n,
        input_data.common.α, input_data.common.g, input_data.common.f, input_data.f₁,
        t_half, τ, τ_4, τ²_4,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices, A, cache)

    # Compute L² errors at current time
    d¹ = τ_2 * (v¹ + v⁰) + d⁰
    z¹ = τ_2 * (r¹ + r⁰) + z⁰

    L2_error.v[2] = L2_error_2d(
        (x, y) -> input_data.v(x, y, τ), v¹, mesh2D, dof_map_m₁, quad)
    L2_error.d[2] = L2_error_2d(
        (x, y) -> input_data.u(x, y, τ), d¹, mesh2D, dof_map_m₁, quad)

    L2_error.r[2] = L2_error_1d(x -> input_data.r(x, τ), r¹, mesh1D, dof_map_m₂, quad)
    L2_error.z[2] = L2_error_1d(x -> input_data.z(x, τ), z¹, mesh1D, dof_map_m₂, quad)

    # ========================================
    # Main Time-Stepping Loop (n ≥ 2)
    # ========================================
    vⁿ⁻² = v⁰
    vⁿ⁻¹ = v¹
    vⁿ = zeros(T, m₁)
    dⁿ⁻² = d⁰
    dⁿ⁻¹ = d¹
    dⁿ = zeros(T, m₁)

    rⁿ⁻¹ = r¹
    rⁿ = r⁰
    zⁿ⁻¹ = z¹
    zⁿ = zeros(T, m₂)

    for n in 2:(nt - 1)
        @. v_ast_n = T(1.5) * vⁿ⁻¹ - T(0.5) * vⁿ⁻²
        @. d_ast_n = T(1.5) * dⁿ⁻¹ - T(0.5) * dⁿ⁻²
        t_half = T(n - 0.5) * τ

        compute_rⁿ!(rⁿ, rⁿ⁻¹, zⁿ⁻¹, v_ast_n, vec_cst, M_m₂xm₂_factorized,
            input_data.f₂, t_half, mesh1D, dof_map_m₂, quad, cache)
        ########################################################### TEST
        # @. v_ast_n_TEMP = input_data.v(Δx:Δx:(1 - Δx), 0.0, t_half)
        ###########################################################
        compute_vⁿ!(vⁿ, vⁿ⁻¹, dⁿ⁻¹, rⁿ, rⁿ⁻¹,
            v_ast_n, d_ast_n,
            input_data.common.α, input_data.common.g, input_data.common.f, input_data.f₁,
            t_half, τ, τ_4, τ²_4,
            mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices, A, cache)

        @. dⁿ = τ_2 * (vⁿ + vⁿ⁻¹) + dⁿ⁻¹
        @. zⁿ = τ_2 * (rⁿ + rⁿ⁻¹) + zⁿ⁻¹

        # Compute L² errors at current time
        tₙ = times[n + 1]
        L2_error.v[n + 1] = L2_error_2d(
            (x, y) -> input_data.v(x, y, tₙ), vⁿ, mesh2D, dof_map_m₁, quad)
        L2_error.d[n + 1] = L2_error_2d(
            (x, y) -> input_data.u(x, y, tₙ), dⁿ, mesh2D, dof_map_m₁, quad)

        L2_error.r[n + 1] = L2_error_1d(
            x -> input_data.r(x, tₙ), rⁿ, mesh1D, dof_map_m₂, quad)
        L2_error.z[n + 1] = L2_error_1d(
            x -> input_data.z(x, tₙ), zⁿ, mesh1D, dof_map_m₂, quad)

        # Rotate array references (no allocation)
        vⁿ⁻², vⁿ⁻¹, vⁿ = vⁿ⁻¹, vⁿ, vⁿ⁻²
        dⁿ⁻², dⁿ⁻¹, dⁿ = dⁿ⁻¹, dⁿ, dⁿ⁻²
        rⁿ⁻¹, rⁿ = rⁿ, rⁿ⁻¹
        zⁿ⁻¹, zⁿ = zⁿ, zⁿ⁻¹
    end

    return L2_error
end

# ============================================
# Cache structure definition
# ============================================

"""
    CNLCache{T}

Cache structure for linearized Crank-Nicolson solver.

# Fields
"""
Base.@kwdef struct CNLCache{T <: Real}
    vec₁m₁::Vector{T}
    vec₂m₁::Vector{T}
    vec₃m₁::Vector{T}
    vec₄m₁::Vector{T}
    vec₁m₂::Vector{T}
    vec₂m₂::Vector{T}
end

"""
    CNLCache(m₁::Integer, m₂::Integer, ::Type{T}) where T

Construct cache with all vectors initialized to zero.

# Arguments
- `m₁`: Number of free DOFs in 2D domain Ω
- `m₂`: Number of free DOFs on 1D boundary Γ₁  
- `T`: Element type
"""
function CNLCache(m₁::Integer, m₂::Integer, ::Type{T}) where {T}
    return CNLCache(
        vec₁m₁ = zeros(T, m₁),
        vec₂m₁ = zeros(T, m₁),
        vec₃m₁ = zeros(T, m₁),
        vec₄m₁ = zeros(T, m₁),
        vec₁m₂ = zeros(T, m₂),
        vec₂m₂ = zeros(T, m₂)
    )
end

# ============================================
# Helper functions for compute solutions
# ============================================
"""
    compute_rⁿ!(rⁿ, rⁿ⁻¹, zⁿ⁻¹, v_ast_n, cst, M_m₂xm₂_factorized, f₂, t_half, mesh1D, dof_map_m₂, quad, cache)

Compute `rⁿ = cst[1]*rⁿ⁻¹ - cst[2]*zⁿ⁻¹ - cst[3]*v_ast_n + M_m₂xm₂ \\ (cst[4]*F(f₂(tₙ₋₁/₂)))`.
"""
function compute_rⁿ!(
        rⁿ::AbstractVector{T},
        rⁿ⁻¹::AbstractVector{T},
        zⁿ⁻¹::AbstractVector{T},
        v_ast_n::AbstractVector{T},
        cst::AbstractVector{T},
        M_m₂xm₂_factorized,
        f₂::Ff₂,
        t_half::T,
        mesh1D::CartesianMesh{1},
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        cache::CNLCache) where {T, Ff₂}

    # Compute cst[4]⋅F(f₂(tₙ₋₁/₂)) → Fm₂
    f = x -> f₂(x, t_half)
    scale = cst[4] * mesh1D.Δx[1] / T(2)
    Fm₂ = cache.vec₁m₂
    assembly_rhs_1d!(Fm₂, f, scale, quad.W_ϕP, mesh1D, dof_map_m₂, quad.xP)

    # Solve M_m₂xm₂ \ Fm₂ → sⁿ
    sⁿ = cache.vec₂m₂
    ldiv!(sⁿ, M_m₂xm₂_factorized, Fm₂)

    # Update rⁿ = cst[1]*rⁿ⁻¹ - cst[2]*zⁿ⁻¹ - cst[3]*v_ast_n + sⁿ
    @inbounds @simd for i in eachindex(rⁿ)
        rⁿ[i] = cst[1] * rⁿ⁻¹[i] - cst[2] * zⁿ⁻¹[i] - cst[3] * v_ast_n[i] + sⁿ[i]
    end

    return nothing
end

"""
    compute_vⁿ!(vⁿ, vⁿ⁻¹, dⁿ⁻¹, rⁿ, rⁿ⁻¹, v_ast_n, d_ast_n, α, g, f, f₁, t_half, τ, τ_4, τ²_4, mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, matrices, A, cache)

Assemble and solve `A vⁿ = b`, where:
- `A = M_m₁×m₁ + (τ²/4)α(tₙ₋₁/₂)K_m₁×m₁`
- `b = M_m₁×m₁ vⁿ⁻¹ - τα(tₙ₋₁/₂)K_m₁×m₁((τ/4)vⁿ⁻¹ + dⁿ⁻¹) + (τ/2)α(tₙ₋₁/₂)M_m₂×m₂(rⁿ + rⁿ⁻¹) - τα(tₙ₋₁/₂)G(v_ast_n) - τF(d_ast_n) + τF(f₁(tₙ₋₁/₂))`
"""
function compute_vⁿ!(
        vⁿ::AbstractVector{T},
        vⁿ⁻¹::AbstractVector{T},
        dⁿ⁻¹::AbstractVector{T},
        rⁿ::AbstractVector{T},
        rⁿ⁻¹::AbstractVector{T},
        v_ast_n::AbstractVector{T},
        d_ast_n::AbstractVector{T},
        α::Fα,
        g::Fg,
        f::Ff,
        f₁::Ff1,
        t_half::T,
        τ::T,
        τ_4::T,
        τ²_4::T,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        matrices,
        A,
        cache::CNLCache) where {T, Fα, Fg, Ff, Ff1}
    # Precompute constants
    α_half = α(t_half)
    τα = τ * α_half
    τα_2 = τα * T(0.5)

    # Compute A = M_m₁×m₁ + (τ²/4)α(tₙ₋₁/₂)K_m₁×m₁
    compute_A!(A, matrices.M_m₁xm₁, τ²_4, α_half, matrices.K_m₁xm₁)

    # Compute M_m₁xm₁*vⁿ⁻¹ → cache.vec₁m₁
    mul!(cache.vec₁m₁, matrices.M_m₁xm₁, vⁿ⁻¹)

    # Compute K_m₁xm₁⋅((τ/4)⋅vⁿ⁻¹+dⁿ⁻¹) → cache.vec₂m₁
    # (will be scaled by -τ⋅α(tₙ₋₁/₂))
    @. cache.vec₃m₁ = muladd(τ_4, vⁿ⁻¹, dⁿ⁻¹)
    mul!(cache.vec₂m₁, matrices.K_m₁xm₁, cache.vec₃m₁)

    # Compute M_m₂xm₂⋅(rⁿ+rⁿ⁻¹) → cache.vec₁m₂
    # (will be scaled by (τ/2)⋅α(tₙ₋₁/₂))
    @. cache.vec₂m₂ = rⁿ + rⁿ⁻¹
    mul!(cache.vec₁m₂, matrices.M_m₂xm₂, cache.vec₂m₂)

    # Compute τα(tₙ₋₁/₂)⋅G(v_ast_n) → cache.vec₂m₂
    assembly_nonlinearity_G!(
        cache.vec₂m₂, τα, g, v_ast_n, mesh1D, dof_map_m₂, quad)

    # Compute τ⋅F(d̂ⁿ) → cache.vec₃m₁
    assembly_nonlinearity_F!(
        cache.vec₃m₁, τ, f, d_ast_n, mesh2D, dof_map_m₁, quad)

    # Compute τ⋅F(f₁(tₙ₋₁/₂)) → cache.vec₄m₁
    f₁_eval = (x, y) -> f₁(x, y, t_half)
    scale = τ * mesh2D.Δx[1] * mesh2D.Δx[2] * T(0.25)
    assembly_rhs_2d!(
        cache.vec₄m₁, f₁_eval, scale, quad.W_φP, mesh2D, dof_map_m₁, quad.xP, quad.yP)

    # Assemble RHS
    m₁ = dof_map_m₁.m
    m₂ = dof_map_m₂.m

    @inbounds for i in 1:m₂
        cache.vec₁m₁[i] += -τα * cache.vec₂m₁[i] + τα_2 * cache.vec₁m₂[i] -
                           cache.vec₂m₂[i] - cache.vec₃m₁[i] + cache.vec₄m₁[i]
    end
    @inbounds for i in (m₂ + 1):m₁
        cache.vec₁m₁[i] += -τα * cache.vec₂m₁[i] - cache.vec₃m₁[i] + cache.vec₄m₁[i]
    end

    # Solve system
    A_fact = cholesky(A)
    ldiv!(vⁿ, A_fact, cache.vec₁m₁)

    return nothing
end

"""
    compute_A!(A, M_m₁xm₁, τ²_4, α_half, K_m₁xm₁)

Compute system matrix `A = M_m₁×m₁ + (τ²/4)α(tₙ₊₁/₂)K_m₁×m₁`.
"""
function compute_A!(
        A::Symmetric{T, S},
        M_m₁xm₁::Symmetric{T, S},
        τ²_4::T,
        α_half::T,
        K_m₁xm₁::Symmetric{T, S}
) where {T, S <: SparseMatrixCSC{T}}
    cst = τ²_4 * α_half
    @. A.data.nzval = M_m₁xm₁.data.nzval + cst * K_m₁xm₁.data.nzval
    return nothing
end