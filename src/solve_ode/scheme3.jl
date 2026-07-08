function solve_ode(
        cache::Scheme3Cache,
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

    # Step "1,0" (predictor) + step n=1 (corrector)
    perform_first_step!(
        cache, state, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data, csts)
    apply!(
        callback, state, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data)

    # Steps n ≥ 2
    for n in 2:(length(tspan) - 1)
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
        cache::Scheme3Cache{T},
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

    # v*ⁿ = (3*vⁿ⁻¹ - vⁿ⁻²)/2,    d*ⁿ = (3*dⁿ⁻¹ - dⁿ⁻²)/2
    @. cache.vᵃⁿ = T(0.5) * muladd(3, state.v, -cache.vⁿ⁻²)
    @. cache.dᵃⁿ = T(0.5) * muladd(3, state.d, -cache.dⁿ⁻²)

    solve_v̂ⁿ(
        cache, state, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data,
        τ, τα, τα_q₅, _2q₁, τq₃, τ²α_2, τ²q₄α_q₅, Δx, τ_jac_1d, τ_jac_2d, t_half)

    cache.linsolve_m₂.isfresh = false
    LS.solve!(cache.linsolve_m₂)
    compute_r̂ⁿ!(cache.r̂ⁿ, cache.v̂ⁿ, state.r, state.z, cache.linsolve_m₂.u,
        _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅)

    # vⁿ⁻² ← vⁿ⁻¹, dⁿ⁻² ← dⁿ⁻¹, update state solution
    cache.vⁿ⁻² .= state.v
    cache.dⁿ⁻² .= state.d
    update_state!(state, cache.v̂ⁿ, cache.r̂ⁿ, τ)

    return nothing
end

function perform_first_step!(
        cache::Scheme3Cache{T},
        state::State{T},
        matrices::SystemMatrices{T},
        nel_per_dim::NTuple{2, I},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data::PDEInputData,
        csts::NTuple{13, T}
) where {T, I}
    τ, τ_2, τ²_2, _2q₁, τq₃, q₅, τ²q₄_q₅,
    _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅, τ_jac_1d, τ_jac_2d = csts
    Δx = element_side_lengths[1]

    t_half = T(0.5) * τ
    α = input_data.α(t_half)

    τα = τ * α
    τα_q₅ = τα / q₅
    τ²α_2 = τ²_2 * α
    τ²q₄α_q₅ = τ²q₄_q₅ * α

    # --- Predictor: v*¹ = v⁰, d*¹ = d⁰  ---
    cache.vᵃⁿ .= state.v
    cache.dᵃⁿ .= state.d
    solve_v̂ⁿ(
        cache, state, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data,
        τ, τα, τα_q₅, _2q₁, τq₃, τ²α_2, τ²q₄α_q₅, Δx, τ_jac_1d, τ_jac_2d, t_half)

    # --- Corrector: v*¹ = v̂^{"1,0"}, d*¹ = (τ/2)v̂^{"1,0"} + d⁰ ---
    @. cache.vᵃⁿ = cache.v̂ⁿ
    @. cache.dᵃⁿ = muladd(τ_2, cache.v̂ⁿ, state.d)
    solve_v̂ⁿ(
        cache, state, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data,
        τ, τα, τα_q₅, _2q₁, τq₃, τ²α_2, τ²q₄α_q₅, Δx, τ_jac_1d, τ_jac_2d, t_half)

    # --- Compute r̂ⁿ ---
    cache.linsolve_m₂.isfresh = false
    LS.solve!(cache.linsolve_m₂)
    compute_r̂ⁿ!(cache.r̂ⁿ, cache.v̂ⁿ, state.r, state.z, cache.linsolve_m₂.u,
        _τq₄_q₅, _2q₁_q₅, _τq₃_q₅, _q₅)

    # vⁿ⁻² ← vⁿ⁻¹, dⁿ⁻² ← dⁿ⁻¹, update state solution
    cache.vⁿ⁻² .= state.v
    cache.dⁿ⁻² .= state.d
    update_state!(state, cache.v̂ⁿ, cache.r̂ⁿ, τ)

    return nothing
end

"""
    solve_v̂ⁿ(
        cache, state, matrices, nel_per_dim, element_side_lengths,
        dof_map_m₁, dof_map_m₂, quad, input_data, 
        τ, τα, τα_q₅, _2q₁, τq₃, τ²α_2, τ²q₄α_q₅, Δx, τ_jac_1d, τ_jac_2d, t_half)

Sequentially:
1. Computes `Q(n)`
2. Computes `τ⋅F(f₁(tₙ₋₁/₂))`
3. Computes `τ⋅F(f₂(tₙ₋₁/₂))`
4. Computes `τα⋅G(vᵃⁿ)`
5. Computes `τ⋅F(dᵃⁿ)`
6. Computes `L(n,vᵃⁿ,dᵃⁿ)`
7. Solves the linear system `Q⋅v̂ⁿ = L`

**WARNING**: Assumes that `cache.vᵃⁿ` and `cache.dᵃⁿ` have already been populated.
"""
function solve_v̂ⁿ(
        cache::Scheme3Cache{T},
        state::State{T},
        matrices::SystemMatrices{T},
        nel_per_dim::NTuple{2, I},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data::PDEInputData,
        τ::T, τα::T, τα_q₅::T, _2q₁::T, τq₃::T, τ²α_2::T, τ²q₄α_q₅::T, Δx::T,
        τ_jac_1d::T, τ_jac_2d::T, t_half::T
) where {T, I}
    compute_Q!(
        cache.Q_upper, matrices.K_m₁xm₁, matrices._2M_m₁xm₁, matrices.M_m₂xm₂, τ²α_2, τ²q₄α_q₅)
    scatter_symmetric!(cache.Q, cache.Q_upper, cache.map_direct, cache.map_mirror)

    assembly_rhs_2d!(
        cache.τFf₁, (x, y) -> input_data.f₁(x, y, t_half),
        τ_jac_2d, quad.W_φP, dof_map_m₁, nel_per_dim, element_side_lengths, quad.xP, quad.yP)
    assembly_rhs_1d!(
        cache.τFf₂, x -> input_data.f₂(x, t_half),
        τ_jac_1d, quad.W_ϕP, dof_map_m₂, Δx, quad.xP)

    assembly_nonlinearity_G!(
        cache.G, τα, input_data.g, cache.vᵃⁿ, dof_map_m₂, Δx, quad.xP, quad.ϕP, quad.W_ϕP)
    assembly_nonlinearity_F!(
        cache.F, τ, input_data.f, cache.dᵃⁿ, dof_map_m₁, element_side_lengths, quad.φP, quad.W_φP)

    compute_L!(
        cache.L, state, matrices._2M_m₁xm₁,
        matrices.M_m₂xm₂, matrices.K_m₁xm₁,
        cache.G, cache.F, cache.τFf₁, cache.τFf₂,
        τα, τα_q₅, _2q₁, τq₃,
        cache.vec_m₁, cache.vec_m₂_1, cache.vec_m₂_2)

    cache.linsolve_m₁.isfresh = true
    LS.solve!(cache.linsolve_m₁)
    cache.v̂ⁿ .= cache.linsolve_m₁.u

    return nothing
end

# ==============================================================================
# Compute L
# ==============================================================================
"""
    compute_L!(
        L, _2M_m₁xm₁, M_m₂xm₂, K_m₁xm₁, ταGvᵃⁿ, τFdᵃⁿ, τFf₁, τFf₂, 
        τα, τα_q₅, _2q₁, τq₃,
        vec_m₁, vec_m₂_1, vec_m₂_2)

Compute in-place

    L = 2M_m₁xm₁⋅vⁿ⁻¹ - τα⋅K_m₁xm₁⋅dⁿ⁻¹ - ταGvᵃⁿ - τFdᵃⁿ  + τFf₁ + τα_q₅⋅[M_m₂xm₂⋅(2q₁⋅rⁿ⁻¹-τq₃⋅zⁿ⁻¹) + τFf₂; 0_(m₁-m₂)]

where `τα = τ⋅α(tₙ₋₁/₂)`, `τα_q₅ = τ⋅α(tₙ₋₁/₂)/q₅`, `ταGvᵃⁿ = τ⋅α(tₙ₋₁/₂)⋅G(vᵃⁿ)`,  `τFdᵃⁿ = τ⋅F(dᵃⁿ)`, `τFf₁ = τ⋅F(f₁(tₙ₋₁/₂))`, and `τFf₂ = τ⋅F(f₂(tₙ₋₁/₂))`.
"""
function compute_L!(
        L::Vector{T},
        state::State{T},
        _2M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        ταGvᵃⁿ::Vector{T},
        τFdᵃⁿ::Vector{T},
        τFf₁::Vector{T},
        τFf₂::Vector{T},
        τα::T, τα_q₅::T, _2q₁::T, τq₃::T,
        vec_m₁::Vector{T}, vec_m₂_1::Vector{T}, vec_m₂_2::Vector{T}) where {
        T, I}
    vⁿ⁻¹ = state.v
    dⁿ⁻¹ = state.d
    rⁿ⁻¹ = state.r
    zⁿ⁻¹ = state.z

    m₁ = length(vⁿ⁻¹)
    m₂ = length(rⁿ⁻¹)

    # L₁ ← 2M_m₁xm₁ · vⁿ⁻¹
    mul!(L, _2M_m₁xm₁, vⁿ⁻¹)

    # vec_m₁ ← K_m₁xm₁⋅dⁿ⁻¹
    mul!(vec_m₁, K_m₁xm₁, dⁿ⁻¹)

    # vec_m₂_2 ← 2q₁⋅rⁿ⁻¹-τq₃⋅zⁿ⁻¹
    @. vec_m₂_2 = _2q₁ * rⁿ⁻¹ - τq₃ * zⁿ⁻¹

    # vec_m₂_1 ← M_m₂×m₂ · vec_m₂_2
    mul!(vec_m₂_1, M_m₂xm₂, vec_m₂_2)

    # Final assembly
    for i in 1:m₂
        tmp1 = muladd(τα, vec_m₁[i], τFdᵃⁿ[i])
        tmp2 = vec_m₂_1[i] + τFf₂[i]
        tmp3 = muladd(τα_q₅, tmp2, τFf₁[i])
        L[i] = L[i] - ταGvᵃⁿ[i] - tmp1 + tmp3
    end
    for i in (m₂ + 1):m₁
        tmp1 = muladd(τα, vec_m₁[i], τFdᵃⁿ[i])
        L[i] = L[i] - tmp1 + τFf₁[i]
    end

    return nothing
end