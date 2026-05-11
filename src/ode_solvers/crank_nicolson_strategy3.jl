# ==============================================================================
# ode_solve
# ==============================================================================
function ode_solve(
        cache::CrankNicolson3Cache{T},
        state::FEMState{T},
        matrices::SystemMatrices{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        tspan::StepRangeLen{T},
        input_data::PDEInputData,
        callback::AbstractCallback
) where {T <: AbstractFloat}
    τ = T(step(tspan))
    q₁, q₂, q₃, q₄ = input_data.q₁, input_data.q₂, input_data.q₃, input_data.q₄
    csts = SVector(
        τ, τ / 2, τ / 4, τ^2 / 4, q₁ - τ * q₂ / 2 - τ^2 * q₃ / 4, τ * q₃, τ * q₄ / 2,
        τ * mesh1D.Δx[1] / 2, τ * mesh2D.Δx[1] * mesh2D.Δx[2] / 4)
    cst₂₁ = τ * input_data.q₄ / 2
    cst₂₂ = input_data.q₁ + τ * input_data.q₂ / 2 + τ^2 * input_data.q₃ / 4

    compute_Q̄₂₁_Q₂₂!(cache.Q̄₂₁, cache.Q₂₂, matrices.M_m₂xm₂, cst₂₁, cst₂₂)
    sync_JH!(cache.JH, cache.Q̄₂₁, cache.map_direct_21, cache.map_mirror_21)
    sync_JH!(cache.JH, cache.Q₂₂, cache.map_direct_22, cache.map_mirror_22)

    for n in 1:(length(tspan) - 1)
        perform_step!(cache, state, matrices,
            dof_map_m₁, dof_map_m₂,
            mesh1D, mesh2D, quad, n, csts, input_data)
        apply!(callback, state,
            mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, input_data)
    end

    return nothing
end

# ==============================================================================
# perform_step!
# ==============================================================================

"""
    perform_step!(cache, state, matrices, dof_map_m₁, dof_map_m₂,
                  mesh1D, mesh2D, quad, n, τ, input_data)
 
Advance `state` from time level ``n-1`` to ``n``.
 
On entry, `state` holds ``(v^{n-1}, d^{n-1}, r^{n-1}, z^{n-1})``.
On exit, `state` holds the solution at ``t_n``.
"""
function perform_step!(
        cache::CrankNicolson3Cache{T},
        state::FEMState{T},
        matrices::SystemMatrices{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        n::Int,
        csts::SVector{9, T},
        input_data::PDEInputData
) where {T}
    τ, τ_2, τ_4, τ²_4, cstᵣ, τq₃, τq₄_2, scale_1d, scale_2d = csts
    t_half = (n - T(0.5)) * τ
    α = input_data.α(t_half)
    τα = τ * α
    τα_2 = τ_2 * α

    cst₁₁ = τ²_4 * α
    cst₁₂ = -τα_2

    compute_Q₁₁!(cache.Q₁₁, matrices.M_m₁xm₁, matrices.K_m₁xm₁, cst₁₁)
    compute_Q̄₁₂!(cache.Q̄₁₂, matrices.M_m₂xm₂, cst₁₂)
    sync_JH!(cache.JH, cache.Q̄₁₂, cache.map_direct_12, cache.map_mirror_12)

    assembly_rhs_2d!(
        cache.τFf₁, (x, y) -> input_data.f₁(x, y, t_half),
        scale_2d, quad.W_φP, mesh2D, dof_map_m₁, quad.xP, quad.yP)
    assembly_rhs_1d!(
        cache.τFf₂, x -> input_data.f₂(x, t_half),
        scale_1d, quad.W_ϕP, mesh1D, dof_map_m₂, quad.xP)

    compute_L₁!(
        cache.L₁, matrices.M_m₁xm₁, matrices.K_m₁xm₁, matrices.M_m₂xm₂,
        state.v, state.d, state.r, cache.τFf₁, τ_4, τα, τα_2,
        cache.vec_m₁_1, cache.vec_m₁_2, cache.vec_m₂_1)
    compute_L₂!(
        cache.L₂, matrices.M_m₂xm₂, state.r, state.z, state.v, cache.τFf₂,
        cstᵣ, τq₃, τq₄_2, cache.vec_m₂_1)

    newton_solve!(
        cache, state, dof_map_m₁, dof_map_m₂, mesh1D, mesh2D,
        quad, τ, τ_2, τ²_4, τα, τα_2, input_data)

    update_state!(state, cache.vⁿ, cache.rⁿ, cache.v̂ⁿ, τ, τ_2)

    return nothing
end

# ==============================================================================
# Compute Q₁₁, Q̄₁₂, Q̄₂₁, Q₂₂
# ==============================================================================
"""
    compute_Q₁₁!(Q₁₁, M_m₁xm₁, K_m₁xm₁, cst)

Compute in-place `Q₁₁ = M_m₁xm₁ + cst * K_m₁xm₁` by iterating directly over the nonzero values.

Assumes `Q₁₁`, `M_m₁xm₁`, and `K_m₁xm₁` share the same sparsity pattern.
"""
function compute_Q₁₁!(
        Q₁₁::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        cst::T) where {T, I}
    nzval_Q₁₁ = Q₁₁.data.nzval
    nzval_M = M_m₁xm₁.data.nzval
    nzval_K = K_m₁xm₁.data.nzval

    @. nzval_Q₁₁ = muladd(cst, nzval_K, nzval_M)

    return nothing
end

"""
    compute_Q̄₁₂!(Q̄₁₂, M_m₂xm₂, cst)

Compute in-place `Q̄₁₂ = cst * M_m₂xm₂` by scaling the nonzero values directly.

Assumes `Q̄₁₂` and `M_m₂xm₂` share the same sparsity pattern.
"""
function compute_Q̄₁₂!(
        Q̄₁₂::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        cst::T) where {T, I}
    nzval_Q̄₁₂ = Q̄₁₂.data.nzval
    nzval_M = M_m₂xm₂.data.nzval

    @. nzval_Q̄₁₂ = cst * nzval_M

    return nothing
end

"""
    compute_Q̄₂₁_Q₂₂!(Q̄₂₁, Q₂₂, M_m₂xm₂, cst₂₁, cst₂₂)

Compute in-place `Q̄₂₁ = cst₂₁ * M_m₂xm₂` and `Q₂₂ = cst₂₂ * M_m₂xm₂` by scaling the nonzero values directly.

Assumes `Q̄₂₁`, `Q₂₂`, and `M_m₂xm₂` share the same sparsity pattern.
"""
function compute_Q̄₂₁_Q₂₂!(
        Q̄₂₁::Symmetric{T, SparseMatrixCSC{T, I}},
        Q₂₂::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        cst₂₁::T, cst₂₂::T) where {T, I}
    nnz_M = nnz(M_m₂xm₂.data)
    nzval_Q̄₂₁ = Q̄₂₁.data.nzval
    nzval_Q₂₂ = Q₂₂.data.nzval
    nzval_M = M_m₂xm₂.data.nzval

    @inbounds for i in 1:nnz_M
        nzval_Q̄₂₁[i] = cst₂₁ * nzval_M[i]
        nzval_Q₂₂[i] = cst₂₂ * nzval_M[i]
    end

    return nothing
end

# ==============================================================================
# Compute L₁, L₂
# ==============================================================================

"""
    compute_L₁!(L₁, M_m₁xm₁, K_m₁xm₁, M_m₂xm₂, vⁿ⁻¹, dⁿ⁻¹, rⁿ⁻¹, τFf₁, τ_4, τα, τα_2, vec_m₁, vec_m₂)

Compute in-place

    L₁ = M_m₁xm₁⋅vⁿ⁻¹ - τα⋅K_m₁xm₁⋅((τ/4)⋅vⁿ⁻¹ + dⁿ⁻¹) + τα_2⋅M_m₂xm₂⋅rⁿ⁻¹ + τFf₁

where `τα = τ⋅α(tₙ₋₁/₂)`, `τα_2 = (τ/2)⋅α(tₙ₋₁/₂)`, and `vec_m₁` and `vec_m₂` are preallocated work buffers. 
The `M_m₂xm₂⋅rⁿ⁻¹` contribution applies only to the first `m₂` entries of `L₁`.
"""
function compute_L₁!(
        L₁::Vector{T},
        M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        vⁿ⁻¹::Vector{T}, dⁿ⁻¹::Vector{T}, rⁿ⁻¹::Vector{T}, τFf₁::Vector{T}, τ_4::T, τα::T, τα_2::T,
        vec_m₁_1::Vector{T}, vec_m₁_2::Vector{T}, vec_m₂::Vector{T}) where {T, I}
    # L₁ ← M_m₁xm₁ · vⁿ⁻¹
    mul!(L₁, M_m₁xm₁, vⁿ⁻¹)

    # vec_m₁ ← K_m₁xm₁ · ((τ/4)vⁿ⁻¹ + dⁿ⁻¹)
    @. vec_m₁_2 = muladd(τ_4, vⁿ⁻¹, dⁿ⁻¹)
    mul!(vec_m₁_1, K_m₁xm₁, vec_m₁_2)

    # vec_m₂ ← M_m₂xm₂ · rⁿ⁻¹
    mul!(vec_m₂, M_m₂xm₂, rⁿ⁻¹)

    # Final assembly
    m₁ = length(vec_m₁_1)
    m₂ = length(vec_m₂)
    @inbounds for i in 1:m₂
        L₁[i] = L₁[i] - τα * vec_m₁_1[i] + τFf₁[i] + τα_2 * vec_m₂[i]
    end
    @inbounds for i in (m₂ + 1):m₁
        L₁[i] = L₁[i] - τα * vec_m₁_1[i] + τFf₁[i]
    end

    return nothing
end

"""
    compute_L₂!(L₂, M_m₂xm₂, rⁿ⁻¹, zⁿ⁻¹, vⁿ⁻¹, τFf₂, cstᵣ, τq₃, τq₄_2, vec_m₂)

Compute in-place

    L₂ = M_m₂xm₂⋅(cstᵣ⋅rⁿ⁻¹ - τq₃⋅zⁿ⁻¹) - τq₄_2⋅M_m₂xm₂⋅vⁿ⁻¹[1:m₂] + τFf₂

where `cstᵣ = q₁ - τq₂/2 - τ²q₃/4`, `τq₄_2 = τq₄/2`, and `vec_m₂` is a preallocated work buffer of length `m₂`.
"""
function compute_L₂!(
        L₂::Vector{T},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        rⁿ⁻¹::Vector{T},
        zⁿ⁻¹::Vector{T},
        vⁿ⁻¹::Vector{T},
        τFf₂::Vector{T},
        cstᵣ::T, τq₃::T, τq₄_2::T,
        vec_m₂::Vector{T}) where {T, I}
    # vec_m₂ ← cstᵣ⋅rⁿ⁻¹ - τq₃⋅zⁿ⁻¹
    @. vec_m₂ = cstᵣ * rⁿ⁻¹ - τq₃ * zⁿ⁻¹

    # L₂ ← M_m₂xm₂⋅(cstᵣ⋅rⁿ⁻¹ - τq₃⋅zⁿ⁻¹)
    mul!(L₂, M_m₂xm₂, vec_m₂)

    # vec_m₂ ← M_m₂xm₂⋅vⁿ⁻¹[1:m₂]
    m₂ = length(L₂)
    mul!(vec_m₂, M_m₂xm₂, view(vⁿ⁻¹, 1:m₂))

    # Final assembly
    @. L₂ = L₂ - τq₄_2 * vec_m₂ + τFf₂

    return nothing
end

# ==============================================================================
# newton_solve!
# ==============================================================================

"""
    newton_solve!(cache, state, dof_map_m₁, dof_map_m₂,
                  mesh1D, mesh2D, quad, τ, τ_2, τ²_4, τα, τα_2, input_data; 
                  abstol, maxiter)
 
Solve the nonlinear system
```math
H(X) = Q X + [τα G^{m₁}(v̂ⁿ) + τ F^{m₁}(d̂ⁿ); 0^{m₂}] - [L₁;L₂] = 0
```
for ``X = [v^n;r^n]`` via Newton's method, updating `cache.X` in-place. 
`cache.X` is warm-started from `[vⁿ⁻¹;rⁿ⁻¹]`.
Convergence is declared when ``\\max_i |H_i(X)| \\leq \\texttt{abstol}``.
 
Assumes cache.Q₁₁, cache.Q̄₁₂, cache.Q̄₂₁, cache.Q₂₂, cache.L₁, and cache.L₂ are already populated.
 
# Keyword Arguments
- `abstol::T`: absolute tolerance on ``\\max_i|H_i|`` (default: `T(1e-10)`).
- `maxiter::Int`: maximum number of Newton iterations (default: `10`).
"""
function newton_solve!(
        cache::CrankNicolson3Cache{T},
        state::FEMState{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        τ::T,
        τ_2::T,
        τ²_4::T,
        τα::T,
        τα_2::T,
        input_data::PDEInputData;
        abstol::T = T(1e-10),
        maxiter::Int = 10
) where {T}
    half = T(0.5)

    # Warm start: X ← [vⁿ⁻¹;rⁿ⁻¹]
    cache.vⁿ .= state.v
    cache.rⁿ .= state.r
    cache.v̂ⁿ .= state.v
    @. cache.d̂ⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)

    for _ in 1:maxiter
        assembly_nonlinearity_G!(
            cache.G, τα, input_data.g, cache.v̂ⁿ, mesh1D, dof_map_m₂, quad)
        assembly_nonlinearity_F!(
            cache.F, τ, input_data.f, cache.d̂ⁿ, mesh2D, dof_map_m₁, quad)

        compute_minusH!(
            cache.minusH, cache.vⁿ, cache.rⁿ,
            cache.Q₁₁, cache.Q̄₁₂, cache.Q̄₂₁, cache.Q₂₂,
            cache.G, cache.F, cache.L₁, cache.L₂,
            cache.vec_m₂_1, cache.vec_m₂_2, cache.vec_m₂_3)

        maximum(abs, cache.minusH) ≤ abstol && return nothing

        JG = assembly_global_matrix_DG(
            τα_2, input_data.∂ₛg, cache.v̂ⁿ, mesh1D, dof_map_m₂, quad)
        JF = assembly_global_matrix_DF(
            τ²_4, input_data.df, cache.d̂ⁿ, mesh2D, dof_map_m₁, quad)

        compute_JH₁₁!(cache.JH₁₁, cache.Q₁₁, JF, JG)
        sync_JH!(cache.JH, cache.JH₁₁, cache.map_direct_11, cache.map_mirror_11)

        solve_newton_linear_system!(cache.linsolve)

        cache.X .+= cache.linsolve.u
        @. cache.v̂ⁿ = half * (cache.vⁿ + state.v)
        @. cache.d̂ⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)
    end

    @warn "newton_solve! did not converge within $maxiter iterations " *
          "(max|H| = $(maximum(abs, cache.minusH)), abstol = $abstol)"
    return nothing
end

# ==============================================================================
# compute_minusH!
# ==============================================================================

"""
    compute_minusH!(minusH, vⁿ, rⁿ, Q₁₁, Q̄₁₂, Q̄₂₁, Q₂₂, G, F, L₁, L₂,
                    vec_m₂_1, vec_m₂_2, vec_m₂_3)

Compute in-place ``-H(X)`` where ``H(X) = QX + [\\tau\\alpha G + \\tau F;\\ 0^{m_2}] - [L_1; L_2]``:

```math
-H_1 = L_1 - Q_{11}v^n - \\bar{Q}_{12}r^n - G - F, \\quad
-H_2 = L_2 - \\bar{Q}_{21}v^n_{1:m_2} - Q_{22}r^n.
```

`vec_m₂_1`, `vec_m₂_2`, `vec_m₂_3` are preallocated work buffers of length `m₂`.
"""
function compute_minusH!(
        minusH, vⁿ, rⁿ, Q₁₁, Q̄₁₂, Q̄₂₁, Q₂₂, G, F, L₁, L₂, vec_m₂_1, vec_m₂_2, vec_m₂_3)
    m₁ = length(L₁)
    m₂ = length(L₂)

    mul!(view(minusH, 1:m₁), Q₁₁, vⁿ)
    mul!(vec_m₂_1, Q̄₁₂, rⁿ)
    mul!(vec_m₂_2, Q̄₂₁, view(vⁿ, 1:m₂))
    mul!(vec_m₂_3, Q₂₂, rⁿ)

    @inbounds for i in 1:m₂
        minusH[i] = -minusH[i] - vec_m₂_1[i] - G[i] - F[i] + L₁[i]
        minusH[i + m₁] = -vec_m₂_2[i] - vec_m₂_3[i] + L₂[i]
    end
    @inbounds for i in (m₂ + 1):m₁
        minusH[i] = -minusH[i] - F[i] + L₁[i]
    end

    return nothing
end

"""
    compute_JH₁₁!(JH₁₁, Q₁₁, JF, JG)

Compute in-place the (1,1) block of the Jacobian matrix,

```math
JH_{11} = Q_{11} + J_F + J_G.
```

Assumes `JH₁₁`, `Q₁₁`, and `JF` share the same sparsity pattern, and that `JG` occupies
the leading `nnz(JG)` entries of that pattern.
"""
function compute_JH₁₁!(
        JH₁₁::Symmetric{T, SparseMatrixCSC{T, I}},
        Q₁₁::Symmetric{T, SparseMatrixCSC{T, I}},
        JF::Symmetric{T, SparseMatrixCSC{T, I}},
        JG::Symmetric{T, SparseMatrixCSC{T, I}}) where {T, I}
    nnz_m₁ = nnz(JH₁₁.data)
    nnz_m₂ = nnz(JG.data)

    nzval_JH₁₁ = JH₁₁.data.nzval
    nzval_Q₁₁ = Q₁₁.data.nzval
    nzval_JF = JF.data.nzval
    nzval_JG = JG.data.nzval

    for i in 1:nnz_m₂
        nzval_JH₁₁[i] = nzval_Q₁₁[i] + nzval_JF[i] + nzval_JG[i]
    end
    for i in (nnz_m₂ + 1):nnz_m₁
        nzval_JH₁₁[i] = nzval_Q₁₁[i] + nzval_JF[i]
    end

    return nothing
end