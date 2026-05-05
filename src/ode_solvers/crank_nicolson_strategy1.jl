# ==============================================================================
# ode_solve
# ==============================================================================

"""
    ode_solve(cache, state, matrices, dof_map_m₁, dof_map_m₂,
              mesh1D, mesh2D, quad, tspan, input_data, callback)
 
Advance `state` over all time steps in `tspan` using the Crank–Nicolson
Strategy 1 scheme, applying `callback` after each step.
"""

function ode_solve(
        cache::CrankNicolson1Cache{T},
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
    q₅ = 2 * input_data.q₁ + τ * input_data.q₂ + (τ^2 / 2) * input_data.q₃
    c1 = -τ * input_data.q₄ / q₅
    c2 = 2 * input_data.q₁ / q₅
    c3 = -τ * input_data.q₃ / q₅
    c4 = τ / q₅
    csts = SVector(c1, c2, c3, c4)

    for n in 1:(length(tspan) - 1)
        perform_step!(cache, state, matrices,
            dof_map_m₁, dof_map_m₂,
            mesh1D, mesh2D, quad, n, τ, q₅, csts, input_data)
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
                  mesh1D, mesh2D, quad, n, τ, q₅, input_data)
 
Advance `state` from time level ``n-1`` to ``n``.
 
On entry, `state` holds ``(v^{n-1}, d^{n-1}, r^{n-1}, z^{n-1})``.
On exit, `state` holds the solution at ``t_n``.
 
- `q₅ = 2q_1 + τ q_2 + (τ²/2) q_3`
"""
function perform_step!(
        cache::CrankNicolson1Cache{T},
        state::FEMState{T},
        matrices::SystemMatrices{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        n::Int,
        τ::T,
        q₅::T,
        csts::SVector{4, T},
        input_data::PDEInputData
) where {T}
    t_half = (n - T(0.5)) * τ   # t_{n-1/2}
    α = input_data.α(t_half)
    τα = τ * α

    compute_Q!(cache, matrices, τ, α, input_data.q₄, q₅)
    compute_L!(cache, state, matrices, dof_map_m₁, dof_map_m₂,
        mesh1D, mesh2D, quad, τ, t_half, α, q₅, input_data)

    newton_solve!(cache, state, dof_map_m₁, dof_map_m₂,
        mesh1D, mesh2D, quad, τ, τα, input_data)

    compute_r̂ⁿ!(
        cache.r̂ⁿ, cache.v̂ⁿ, state.r, state.z, cache.Ff₂, csts, cache.M_m₂xm₂_chol, cache.vec_m₂_1)

    update_state!(state, cache.v̂ⁿ, cache.r̂ⁿ, τ)

    return nothing
end

# ==============================================================================
# compute_Q!
# ==============================================================================

"""
    compute_Q!(cache, matrices, τ, α, q₄, q₅)
 
Assemble `cache.Q` in-place:
```math
Q =
M^{m_1 \\times m_1}
+ \\frac{\\tau^2}{4}\\alpha K^{m_1 \\times m_1}
+ \\frac{\\tau^2 q_4}{2q_5}\\alpha
  \\begin{bmatrix}
  M^{m_2\\times m_2}       & 0^{m_2\\times(m_1-m_2)}\\\\[5pt]
  0^{(m_1-m_2)\\times m_2} & 0^{(m_1-m_2)\\times(m_1-m_2)}
  \\end{bmatrix}.
```
"""
function compute_Q!(
        cache::CrankNicolson1Cache{T},
        matrices::SystemMatrices{T},
        τ::T,
        α::T,
        q₄::T,
        q₅::T
) where {T}
    cst1 = (τ^2 / 4) * α
    cst2 = (τ^2 * q₄ / (2 * q₅)) * α

    nzval_Q = cache.Q.data.nzval
    nzval_M₁₁ = matrices.M_m₁xm₁.data.nzval
    nzval_K₁₁ = matrices.K_m₁xm₁.data.nzval
    nzval_M₂₂ = matrices.M_m₂xm₂.data.nzval

    nnz_m₂ = length(nzval_M₂₂)
    nnz_m₁ = length(nzval_Q)

    #  Leading m₂×m₂ sub-block: Q ← M^{m₁×m₁} + cst1·K^{m₁×m₁} + cst2·M^{m₂×m₂}
    @inbounds @simd for i in 1:nnz_m₂
        nzval_Q[i] = nzval_M₁₁[i] + cst1 * nzval_K₁₁[i] + cst2 * nzval_M₂₂[i]
    end

    # Remaining m₁×m₁ entries: Q ← M^{m₁×m₁} + cst1·K^{m₁×m₁}
    @inbounds @simd for i in (nnz_m₂ + 1):nnz_m₁
        nzval_Q[i] = nzval_M₁₁[i] + cst1 * nzval_K₁₁[i]
    end

    return nothing
end

# ==============================================================================
# compute_L!
# ==============================================================================

"""
    compute_L!(cache, state, matrices, dof_map_m₁, dof_map_m₂, mesh1D, mesh2D,
                quad, τ, t_half, α, q₅, input_data)
 
Compute `cache.L` in-place:
```math
L =
  M^{m_1 \\times m_1}v^{n-1}
- \\tau\\alpha K^{m_1 \\times m_1}\\bigl(\\frac{\\tau}{4}v^{n-1}+d^{n-1}\\bigr)
+ \\tau\\mathcal{F}^{m_1}(f_1^{n-1/2})
+ \\frac{\\tau\\alpha}{q_5}
\\begin{bmatrix}
    M^{m_2 \\times m_2}\\bigl(-\\frac{\\tau}{2}q_4v_{1:m_2}^{n-1} + 2q_1 r^{n-1} - \\tau q_3 z^{n-1}\\bigr)
    + \\tau\\mathcal{F}^{m_2}(f_2^{n-1/2})
    \\\\[5pt]
    0^{(m_1-m_2)}
\\end{bmatrix}.
```
"""
function compute_L!(
        cache::CrankNicolson1Cache{T},
        state::FEMState{T},
        matrices::SystemMatrices{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        τ::T,
        t_half::T,
        α::T,
        q₅::T,
        input_data::PDEInputData
) where {T}
    m₁ = dof_map_m₁.m
    m₂ = dof_map_m₂.m
    τα = τ * α
    τα_q₅ = τα / q₅
    τ²α_q₅ = τ * τα_q₅
    q₁2 = 2 * input_data.q₁
    τq₃ = τ * input_data.q₃
    τq₄_2 = τ * input_data.q₄ / 2
    τ_4 = τ / 4

    # vec_m₁_1 ← M_m₁xm₁ · vⁿ⁻¹
    mul!(cache.vec_m₁_1, matrices.M_m₁xm₁, state.v)

    # vec_m₁_3 ← (τ/4)vⁿ⁻¹ + dⁿ⁻¹
    @. cache.vec_m₁_3 = muladd(τ_4, state.v, state.d)

    # vec_m₁_2 ← K_m₁xm₁ · ((τ/4)vⁿ⁻¹ + dⁿ⁻¹)
    mul!(cache.vec_m₁_2, matrices.K_m₁xm₁, cache.vec_m₁_3)

    # vec_m₁_3 ← τ F(f₁(t_half))
    scale_2d = τ * mesh2D.Δx[1] * mesh2D.Δx[2] / 4
    assembly_rhs_2d!(cache.vec_m₁_3, (x, y) -> input_data.f₁(x, y, t_half),
        scale_2d, quad.W_φP, mesh2D, dof_map_m₁, quad.xP, quad.yP)

    # vec_m₂_2 ← 2q₁rⁿ⁻¹ - τq₃zⁿ⁻¹ - (τq₄/2)vⁿ⁻¹[1:m₂]
    for i in 1:m₂
        cache.vec_m₂_2[i] = q₁2 * state.r[i] - τq₃ * state.z[i] - τq₄_2 * state.v[i]
    end

    # vec_m₂_1 ← M_m₂×m₂ · (2q₁rⁿ⁻¹ - τq₃zⁿ⁻¹ - (τq₄/2)vⁿ⁻¹[1:m₂])
    mul!(cache.vec_m₂_1, matrices.M_m₂xm₂, cache.vec_m₂_2)

    # Ff₂ ← F(f₂(t_half))
    scale_1d = mesh1D.Δx[1] / 2
    assembly_rhs_1d!(cache.Ff₂, x -> input_data.f₂(x, t_half),
        scale_1d, quad.W_ϕP, mesh1D, dof_map_m₂, quad.xP)

    # Final assembly
    @inbounds for i in 1:m₂
        cache.L[i] = cache.vec_m₁_1[i] - τα * cache.vec_m₁_2[i] + cache.vec_m₁_3[i] +
                     τα_q₅ * cache.vec_m₂_1[i] + τ²α_q₅ * cache.Ff₂[i]
    end
    @inbounds for i in (m₂ + 1):m₁
        cache.L[i] = cache.vec_m₁_1[i] - τα * cache.vec_m₁_2[i] + cache.vec_m₁_3[i]
    end

    return nothing
end

# ==============================================================================
# newton_solve!
# ==============================================================================

"""
    newton_solve!(cache, state, matrices, dof_map_m₁, dof_map_m₂,
                  mesh1D, mesh2D, quad, τ, τα, input_data; 
                  abstol, maxiter)
 
Solve the nonlinear system
```math
H(X) 
= Q X
+ \\tau\\alpha G^{m_1}(\\frac{X + vⁿ⁻¹}{2})
+ \\tau F^{m_1}\\bigl(\\frac{\\tau}{4}X + \\frac{\\tau}{4}v^{n-1} + d^{n-1}\\bigr)
- L
= 0
```
for ``X = v^n`` via Newton's method, updating `cache.X` in-place. 
`cache.X` is warm-started from ``v^{n-1}``. 
Convergence is declared when ``\\max_i |H_i(X)| \\leq \\texttt{abstol}``.
 
Assumes `cache.Q` and `cache.L` are already populated.
 
# Keyword Arguments
- `abstol::T`: absolute tolerance on ``\\max_i|H_i|`` (default: `T(1e-10)`).
- `maxiter::Int`: maximum number of Newton iterations (default: `10`).
"""
function newton_solve!(
        cache::CrankNicolson1Cache{T},
        state::FEMState{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        τ::T,
        τα::T,
        input_data::PDEInputData;
        abstol::T = T(1e-10),
        maxiter::Int = 10
) where {T}
    half = T(0.5)
    τ_2 = τ * half

    # Warm start: X ← vⁿ⁻¹
    cache.X .= state.v
    cache.v̂ⁿ .= state.v
    @. cache.d̂ⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)

    for _ in 1:maxiter
        compute_minusH!(cache, dof_map_m₁, dof_map_m₂,
            mesh1D, mesh2D, quad, τ, τα, input_data)

        maximum(abs, cache.minusH) ≤ abstol && return nothing

        compute_JH_upper!(
            cache, dof_map_m₁, dof_map_m₂, mesh1D, mesh2D, quad, τ, τα, input_data)
        sync_JH!(cache.JH, cache.JH_upper, cache.map_direct, cache.map_mirror)

        solve_newton_linear_system!(cache.linsolve)

        cache.X .+= cache.linsolve.u
        @. cache.v̂ⁿ = half * (cache.X + state.v)
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
    compute_minusH!(cache, state, matrices, dof_map_m₁, dof_map_m₂,
                    mesh1D, mesh2D, quad, τ, τα, input_data)
 
Compute ``-H(X)`` and store the result in `cache.minusH`:
```math
-H 
= - Q X
- \\tau\\alpha G^{m_1}(\\frac{X + vⁿ⁻¹}{2})
- \\tau F^{m_1}\\bigl(\\frac{\\tau}{4}X + \\frac{\\tau}{4}v^{n-1} + d^{n-1}\\bigr)
+ L.
```
Assumes `cache.Q`, `cache.L`, `cache.X`, cache.v̂ⁿ, and cache.d̂ⁿ are already populated.
"""
function compute_minusH!(
        cache::CrankNicolson1Cache{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        τ::T,
        τα::T,
        input_data::PDEInputData
) where {T}
    m₁ = dof_map_m₁.m
    m₂ = dof_map_m₂.m

    # Q X → vec_m₁_1
    mul!(cache.vec_m₁_1, cache.Q, cache.X)

    # τα G(v̂ⁿ_{1:m₂}) → vec_m₂_1
    assembly_nonlinearity_G!(
        cache.vec_m₂_1, τα, input_data.g, cache.v̂ⁿ, mesh1D, dof_map_m₂, quad)

    # τ F(d̂ⁿ) → vec_m₁_2
    assembly_nonlinearity_F!(
        cache.vec_m₁_2, τ, input_data.f, cache.d̂ⁿ, mesh2D, dof_map_m₁, quad)

    # Final assembly: -H = L - Q X - nonlinear contributions
    @inbounds for i in 1:m₂
        cache.minusH[i] = cache.L[i] - cache.vec_m₁_1[i] - cache.vec_m₁_2[i] -
                          cache.vec_m₂_1[i]
    end
    @inbounds for i in (m₂ + 1):m₁
        cache.minusH[i] = cache.L[i] - cache.vec_m₁_1[i] - cache.vec_m₁_2[i]
    end

    return nothing
end

# ==============================================================================
# compute_JH_upper!
# ==============================================================================

"""
    compute_JH_upper!(cache, dof_map_m₁, dof_map_m₂, mesh1D, mesh2D, quad, τ, τα, input_data)
 
Assemble the upper triangle of the Jacobian into `cache.JH_upper`:
```math
JH 
= Q
+ \\tfrac{\\tau\\alpha}{2}\\, JG\\bigl(\\hat{v}^n_{1:m₂}\\bigr)
 + \\tfrac{\\tau^2}{4}\\, JF\\bigl(\\hat{d}^n\\bigr),
```
where ``JG \\in \\mathbb{R}^{m_2 \\times m_2}`` is embedded in the leading sub-block
and ``JF \\in \\mathbb{R}^{m_1 \\times m_1}`` spans the full block.

Assumes `cache.Q`, `cache.v̂ⁿ`, and `cache.d̂ⁿ` are already populated.
"""
function compute_JH_upper!(
        cache::CrankNicolson1Cache{T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        quad::QuadratureSetup,
        τ::T,
        τα::T,
        input_data::PDEInputData
) where {T}
    # FIXME: allocates
    JG = assembly_global_matrix_DG(
        τα / 2, input_data.∂ₛg, cache.v̂ⁿ, mesh1D, dof_map_m₂, quad)
    # FIXME: allocates
    JF = assembly_global_matrix_DF(
        τ^2 / 4, input_data.df, cache.d̂ⁿ, mesh2D, dof_map_m₁, quad)

    nzval_JH = cache.JH_upper.data.nzval
    nzval_Q = cache.Q.data.nzval
    nzval_JF = JF.data.nzval
    nzval_JG = JG.data.nzval
    nnz_m₂ = length(nzval_JG)
    nnz_m₁ = length(nzval_JH)

    # Leading m₂×m₂ sub-block: Q + JF + JG
    @inbounds @simd for i in 1:nnz_m₂
        nzval_JH[i] = nzval_Q[i] + nzval_JF[i] + nzval_JG[i]
    end

    # Remaining m₁×m₁ entries: Q + JF
    @inbounds @simd for i in (nnz_m₂ + 1):nnz_m₁
        nzval_JH[i] = nzval_Q[i] + nzval_JF[i]
    end

    return nothing
end