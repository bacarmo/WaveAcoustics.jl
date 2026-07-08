function solve_ode(
        cache::Scheme1Cache,
        tspan::StepRangeLen{T},
        state::State{T},
        matrices::SystemMatrices{T},
        nel_per_dim::NTuple{2, I},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data::PDEInputData,
        callback::AbstractCallback
) where {T, I}
    τ = step(tspan)
    τ_2 = τ / 2
    τ²_2 = τ * τ / 2

    q₁, q₂, q₃, q₄ = input_data.q₁, input_data.q₂, input_data.q₃, input_data.q₄
    q₅ = 2 * q₁ + τ * q₂ + τ²_2 * q₃

    _2q₁ = 2 * q₁
    τq₃ = τ * q₃
    τ²q₄_q₅ = τ*τ*q₄/q₅

    _τq₄_q₅ = -τ * q₄ / q₅
    _2q₁_q₅ = 2 * q₁ / q₅
    _τq₃_q₅ = -τ * q₃ / q₅
    _q₅ = 1 / q₅

    Δx, Δy = element_side_lengths
    τ_jac_2d = τ * Δx * Δy / 4
    τ_jac_1d = τ * Δx / 2

    csts = (τ, τ_2, τ²_2, _2q₁, τq₃, q₅, τ²q₄_q₅,
        _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅, τ_jac_1d, τ_jac_2d)

    for n in 1:(length(tspan) - 1)
        perform_step!(
            cache, state, matrices, nel_per_dim, element_side_lengths,
            dof_map_m₁, dof_map_m₂, quad, input_data, n, csts)
        apply!(
            callback, state, nel_per_dim, element_side_lengths,
            dof_map_m₁, dof_map_m₂, quad, input_data)
    end
    return nothing
end

# ==============================================================================
# perform_step!
# ==============================================================================
function perform_step!(
        cache::Scheme1Cache{T},
        state::State{T},
        matrices::SystemMatrices{T},
        nel_per_dim::NTuple{2, I},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data::PDEInputData,
        n::Int,
        csts::NTuple{13, T}
) where {T, I}
    τ, τ_2, τ²_2, _2q₁, τq₃, q₅, τ²q₄_q₅,
    _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅, τ_jac_1d, τ_jac_2d = csts
    Δx = element_side_lengths[1]

    t_half = (n - T(0.5)) * τ
    α = input_data.α(t_half)

    τα = τ * α
    τα_q₅ = τα / q₅
    τ²α_2 = τ²_2 * α
    τ²q₄α_q₅ = τ²q₄_q₅ * α

    compute_Q!(
        cache.Q, matrices.K_m₁xm₁, matrices._2M_m₁xm₁, matrices.M_m₂xm₂, τ²α_2, τ²q₄α_q₅)

    assembly_rhs_2d!(
        cache.τFf₁, (x, y) -> input_data.f₁(x, y, t_half),
        τ_jac_2d, quad.W_φP, dof_map_m₁, nel_per_dim, element_side_lengths, quad.xP, quad.yP)
    assembly_rhs_1d!(
        cache.τFf₂, x -> input_data.f₂(x, t_half),
        τ_jac_1d, quad.W_ϕP, dof_map_m₂, Δx, quad.xP)

    compute_L!(
        cache.L, state, matrices.K_m₁xm₁, matrices._2M_m₁xm₁,
        matrices.M_m₂xm₂, cache.τFf₁, cache.τFf₂,
        τα, τα_q₅, _2q₁, τq₃,
        cache.vec_m₁, cache.vec_m₂_1, cache.vec_m₂_2)
    solve_nonlinear_system!(
        cache, state, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data,
        τ, τ_2, τ²_2, τα)

    cache.linsolve_m₂.isfresh = false
    LS.solve!(cache.linsolve_m₂)

    compute_r̂ⁿ!(cache.r̂ⁿ, cache.v̂ⁿ, state.r, state.z, cache.linsolve_m₂.u,
        _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅)

    update_state!(state, cache.v̂ⁿ, cache.r̂ⁿ, τ)

    return nothing
end

# ==============================================================================
# Compute L
# ==============================================================================
"""
    compute_L!(
        L, state, K_m₁xm₁, _2M_m₁xm₁, M_m₂xm₂, τFf₁, τFf₂, 
        τα, τα_q₅, _2q₁, τq₃, 
        vec_m₁, vec_m₂_1, vec_m₂_2)

Compute in-place

    L = 2M_m₁xm₁⋅vⁿ⁻¹ - τα⋅K_m₁xm₁⋅dⁿ⁻¹ + τFf₁ + τα_q₅⋅[M_m₂xm₂⋅(2q₁⋅rⁿ⁻¹-τq₃⋅zⁿ⁻¹) + τFf₂; 0_(m₁-m₂)]

where `τα = τ⋅α(tₙ₋₁/₂)`, `τα_q₅ = τ⋅α(tₙ₋₁/₂)/q₅`, `τFf₁ = τ⋅F(f₁(tₙ₋₁/₂))`, and `τFf₂ = τ⋅F(f₂(tₙ₋₁/₂))`.
"""
function compute_L!(
        L::Vector{T},
        state::State{T},
        K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        _2M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        τFf₁::Vector{T},
        τFf₂::Vector{T},
        τα::T, τα_q₅::T, _2q₁::T, τq₃::T,
        vec_m₁::Vector{T}, vec_m₂_1::Vector{T}, vec_m₂_2::Vector{T}) where {T, I}
    vⁿ⁻¹ = state.v
    dⁿ⁻¹ = state.d
    rⁿ⁻¹ = state.r
    zⁿ⁻¹ = state.z

    m₁ = length(vⁿ⁻¹)
    m₂ = length(rⁿ⁻¹)

    # L ← 2M_m₁xm₁ · vⁿ⁻¹
    mul!(L, _2M_m₁xm₁, vⁿ⁻¹)

    # vec_m₁ ← K_m₁xm₁⋅dⁿ⁻¹
    mul!(vec_m₁, K_m₁xm₁, dⁿ⁻¹)

    # vec_m₂_2 ← 2q₁⋅rⁿ⁻¹-τq₃⋅zⁿ⁻¹
    @. vec_m₂_2 = _2q₁ * rⁿ⁻¹ - τq₃ * zⁿ⁻¹

    # vec_m₂_1 ← M_m₂×m₂ · vec_m₂_2
    mul!(vec_m₂_1, M_m₂xm₂, vec_m₂_2)

    # Final assembly
    cst = -τα
    for i in 1:m₂
        tmp1 = muladd(cst, vec_m₁[i], τFf₁[i])
        tmp2 = vec_m₂_1[i] + τFf₂[i]
        L[i] += muladd(τα_q₅, tmp2, tmp1)
    end
    for i in (m₂ + 1):m₁
        L[i] += muladd(cst, vec_m₁[i], τFf₁[i])
    end

    return nothing
end

# ==============================================================================
# solve_nonlinear_system!
# ==============================================================================
"""
    solve_nonlinear_system!(
        cache, state, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data,
        τ, τ_2, τ²_2, τα;
        abstol, reltol, maxiter)

Solve the nonlinear system
```math
H(v̂ⁿ) = Q v̂ⁿ + τα G^{m₁}(v̂ⁿ) + τ F^{m₁}(d̂ⁿ) - L = 0
```
via Newton's method, updating `cache.v̂ⁿ` in-place. 
`cache.v̂ⁿ` is warm-started from `vⁿ⁻¹` and ``d̂ⁿ = (τ/2)v̂ⁿ + dⁿ⁻¹``.

Convergence is declared when either of the following holds at iteration ``k``:
1. Residual criterion:
   ``max|H(Xᵏ)| ≤ abstol``
2. Step criterion:
   ``max|Xᵏ⁺¹-Xᵏ| ≤ abstol + reltol ⋅ max|Xᵏ|``
"""
function solve_nonlinear_system!(
        cache::Scheme1Cache{T},
        state::State{T},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data::PDEInputData,
        τ::T, τ_2::T, τ²_2::T, τα::T;
        abstol::T = T(1e-14),
        reltol::T = T(1e-9),
        maxiter::Int = 5) where {T}
    Δx, Δy = element_side_lengths

    # Warm start: X ← vⁿ⁻¹
    cache.v̂ⁿ .= state.v
    @. cache.d̂ⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)

    normH = zero(T)
    normΔX = zero(T)
    normX = zero(T)

    for _ in 1:maxiter
        assembly_nonlinearity_G!(
            cache.G, τα, input_data.g, cache.v̂ⁿ,
            dof_map_m₂, Δx, quad.xP, quad.ϕP, quad.W_ϕP)      # τα⋅G(v̂ⁿ)
        assembly_nonlinearity_F!(
            cache.F, τ, input_data.f, cache.d̂ⁿ, dof_map_m₁,
            element_side_lengths, quad.φP, quad.W_φP)         # τ⋅F(d̂ⁿ)

        compute_minusH!(cache.minusH, cache.v̂ⁿ, cache.Q, cache.G, cache.F, cache.L)

        normH = maximum(abs, cache.minusH)
        normH ≤ abstol && return nothing                      # criterion 1

        JG = assembly_global_matrix_DG(
            τα, input_data.∂ₛg, cache.v̂ⁿ, dof_map_m₂,
            Δx, quad.xP, quad.ϕP, quad.W_ϕPϕP)                # τα⋅JG(v̂ⁿ)
        JF = assembly_global_matrix_DF(
            τ²_2, input_data.df, cache.d̂ⁿ, dof_map_m₁,
            element_side_lengths, quad.φP, quad.W_φPφP)       # (τ²/2)⋅JF(d̂ⁿ)

        compute_JH_upper!(cache.JH_upper, cache.Q, JG, JF)    # Q + τα⋅JG(v̂ⁿ) + (τ²/2)⋅JF(d̂ⁿ)
        scatter_symmetric!(cache.JH, cache.JH_upper, cache.map_direct, cache.map_mirror)

        cache.linsolve_m₁.isfresh = true
        LS.solve!(cache.linsolve_m₁)

        normΔX = maximum(abs, cache.linsolve_m₁.u)
        normX = maximum(abs, cache.v̂ⁿ)
        cache.v̂ⁿ .+= cache.linsolve_m₁.u
        @. cache.d̂ⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)

        normΔX ≤ abstol + reltol * normX && return nothing     # criterion 2
    end

    @warn "solve_nonlinear_system! did not converge in $maxiter iterations " *
          "(‖H(Xᵏ)‖ = $(@sprintf("%.1e", normH)), " *
          "‖Xᵏ⁺¹-Xᵏ‖ = $(@sprintf("%.1e", normΔX)), " *
          "abstol+reltol*‖Xᵏ‖ = $(@sprintf("%.1e", abstol + reltol * normX)), " *
          "‖Xᵏ‖ = $(@sprintf("%.1e", normX)), " *
          "abstol = $(@sprintf("%.1e", abstol)), reltol = $(@sprintf("%.1e", reltol)))"
    return nothing
end

# ==============================================================================
# compute_minusH!
# ==============================================================================
"""
    compute_minusH!(minusH, v̂ⁿ, Q, G, F, L)

Compute in-place ``-H(X)`` where
```math
-H = - Q⋅v̂ⁿ - τα⋅G(v̂ⁿ) - τF(d̂ⁿ) + L
```
"""
function compute_minusH!(minusH, v̂ⁿ, Q, G, F, L)
    m₁ = length(v̂ⁿ)
    m₂ = length(G)

    mul!(minusH, Q, v̂ⁿ)
    for i in 1:m₂
        minusH[i] = -minusH[i] - G[i] - F[i] + L[i]
    end
    for i in (m₂ + 1):m₁
        minusH[i] = -minusH[i] - F[i] + L[i]
    end

    return nothing
end

# ==============================================================================
# Compute JH_upper
# ==============================================================================
"""
    compute_JH_upper!(JH_upper, Q, JG, JF)

Compute in-place
```math
JH_upper = Q + τα⋅JG(v̂ⁿ) + (τ²/2)⋅JF(d̂ⁿ).
```

Assumes `JH_upper`, `Q`, and `JF` share the same sparsity pattern, and that `JG` occupies
the leading `nnz(JG)` entries of that pattern.
"""
function compute_JH_upper!(
        JH_upper::Symmetric{T, SparseMatrixCSC{T, I}},
        Q::Symmetric{T, SparseMatrixCSC{T, I}},
        JG::Symmetric{T, SparseMatrixCSC{T, I}},
        JF::Symmetric{T, SparseMatrixCSC{T, I}}) where {T, I}
    nnz_m₁ = nnz(JH_upper.data)
    nnz_m₂ = nnz(JG.data)

    nzval_JH = JH_upper.data.nzval
    nzval_Q = Q.data.nzval
    nzval_JF = JF.data.nzval
    nzval_JG = JG.data.nzval

    for i in 1:nnz_m₂
        nzval_JH[i] = nzval_Q[i] + nzval_JG[i] + nzval_JF[i]
    end
    for i in (nnz_m₂ + 1):nnz_m₁
        nzval_JH[i] = nzval_Q[i] + nzval_JF[i]
    end

    return nothing
end