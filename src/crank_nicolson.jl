"""
    crank_nicolson(v⁰, d⁰, r⁰, z⁰, τ, input_data, mesh1D, mesh2D, 
                   dof_map_m₁, dof_map_m₂, quad, matrices, output)

Solve coupled wave-acoustic PDE system using Crank-Nicolson time integration.

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
- `output::OutputMode`: Output mode (default: `ConvergenceStudy()`)
  - `ConvergenceStudy()` → `ConvergenceStudyData` with L² errors at each time step
  - `SolutionHistory()` → `SolutionHistoryData` with displacement snapshots
  - `EnergyHistory()` → `EnergyHistoryData` with energy evolution

# Returns
Depends on `output`:
- `ConvergenceStudyData{T}`: Fields `v_errors`, `d_errors`, `r_errors`, `z_errors` (each a vector of length `nt`)
- `SolutionHistoryData{T}`: Fields `d_history` (m₁×nt), `z_history` (m₂×nt), `times` (length `nt`)
- `EnergyHistoryData{T}`: Fields `energy`, `times` (each length `nt`)

# Algorithm
Uses Crank-Nicolson for time discretization with Newton iteration to handle nonlinearities f(u) and g(x,v). 
The method is second-order accurate in time.
"""
function crank_nicolson(
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
        matrices,
        output::O = ConvergenceStudy()
) where {T <: Real, O <: OutputMode}
    m₁, m₂ = length(v⁰), length(r⁰)
    @assert m₁ == dof_map_m₁.m
    @assert m₂ == dof_map_m₂.m

    times = range(zero(T), input_data.t_final; step = τ)
    nt = length(times)

    # ========================================
    # Initialize output data (dispatch)
    # ========================================
    output_data = initialize_output(output, m₁, m₂, times)

    # ========================================
    # Process solution at t₀ (dispatch)
    # ========================================
    process_solution!(
        output_data, output, 0, times[1], v⁰, d⁰, r⁰, z⁰,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, input_data)
    # ========================================
    # Precompute constants
    # ========================================
    q₁, q₂, q₃, q₄ = input_data.q₁, input_data.q₂, input_data.q₃,
    input_data.q₄

    τ_2 = τ / 2
    τ_4 = τ / 4
    τ²_4 = τ * τ / 4
    τq₄_2 = τ_2 * q₄
    τq₃ = τ * q₃
    cst_q = q₁ - τ_2 * q₂ - τ²_4 * q₃

    # ========================================
    # Allocate cache vectors
    # ========================================
    cache = CNCache(m₁, m₂, T)

    # ========================================
    # Allocate system matrices for Newton
    # ========================================
    A₁₁ = similar(matrices.M_m₁xm₁)
    A₁₂ = similar(matrices.M_m₁xm₂)
    A₂₁ = τq₄_2 * matrices.M_m₂xm₁
    A₂₂ = (q₁ + τ_2 * q₂ + τ²_4 * q₃) * matrices.M_m₂xm₂
    DHX₁₁ = similar(matrices.M_m₁xm₁)

    # ========================================
    # Newton parameters
    # ========================================
    epsilon = 1e-14
    max_newton_iterations = 5

    # ========================================
    # Time-stepping loop
    # ========================================
    vⁿ⁻¹ = v⁰
    dⁿ⁻¹ = d⁰
    rⁿ⁻¹ = r⁰
    zⁿ⁻¹ = z⁰

    vⁿ = similar(v⁰)
    dⁿ = similar(d⁰)
    rⁿ = similar(r⁰)
    zⁿ = similar(z⁰)

    vec_d_v = similar(v⁰)
    d̂ⁿ = similar(v⁰)
    v̂ⁿ_m₂ = similar(r⁰)

    Lm₁ = similar(v⁰)
    Lm₂ = similar(r⁰)

    minusHX = zeros(T, m₁ + m₂)

    for n in 1:(nt - 1)
        # Compute time at half-step
        t_half = (n - 0.5) * τ
        α_half = input_data.α(t_half)
        τα_2 = τ_2 * α_half

        # Compute dⁿ⁻¹ + (τ/4)vⁿ⁻¹
        @. vec_d_v = muladd(τ_4, vⁿ⁻¹, dⁿ⁻¹)

        # Assemble system matrices A
        compute_A₁₁!(A₁₁, matrices.M_m₁xm₁, τ²_4, α_half, matrices.K_m₁xm₁)
        compute_A₁₂!(A₁₂, τ_2, α_half, matrices.M_m₁xm₂)

        # Assemble right-hand side vectors L
        compute_Lm₁!(Lm₁, vⁿ⁻¹, rⁿ⁻¹, vec_d_v, matrices, input_data,
            mesh2D, dof_map_m₁, quad, τ, t_half, α_half, cache)
        compute_Lm₂!(Lm₂, rⁿ⁻¹, zⁿ⁻¹, vⁿ⁻¹, matrices, input_data, mesh1D,
            dof_map_m₂, quad, τ, t_half, τq₄_2, τq₃, cst_q, cache)

        # Initial guess for Newton iteration
        @. vⁿ = vⁿ⁻¹
        @. rⁿ = rⁿ⁻¹
        @. d̂ⁿ = dⁿ⁻¹
        v̂ⁿ_m₂ .= view(vⁿ⁻¹, 1:m₂)

        # Compute initial residual
        compute_minusHX!(
            minusHX, vⁿ, v̂ⁿ_m₂, d̂ⁿ, rⁿ, A₁₁, A₂₂, matrices.M_m₂xm₂, Lm₁, Lm₂,
            input_data, mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad,
            τ, τq₄_2, α_half, cache)

        # Newton iteration
        counter = 0
        while maximum(abs, minusHX) > epsilon
            # Compute Jacobian DHX
            compute_DHX₁₁!(DHX₁₁, A₁₁, v̂ⁿ_m₂, d̂ⁿ, input_data, mesh1D, mesh2D,
                dof_map_m₁, dof_map_m₂, quad, τα_2, τ²_4, m₂)
            DHX = [DHX₁₁ A₁₂; A₂₁ A₂₂]

            # Solve linear system: DHX * sol = -H(X)
            sol = DHX \ minusHX

            # Update solution X 
            vⁿ .+= view(sol, 1:m₁)
            rⁿ .+= view(sol, (m₁ + 1):(m₁ + m₂))

            # Update auxiliary variables d̂ⁿ, v̂ⁿ_m₂
            @. d̂ⁿ = τ_4 * (vⁿ + vⁿ⁻¹) + dⁿ⁻¹
            v̂ⁿ_m₂ .= 0.5 .* (view(vⁿ, 1:m₂) .+ view(vⁿ⁻¹, 1:m₂))

            # Recompute residual minusHX
            compute_minusHX!(
                minusHX, vⁿ, v̂ⁿ_m₂, d̂ⁿ, rⁿ, A₁₁, A₂₂, matrices.M_m₂xm₂, Lm₁, Lm₂,
                input_data, mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad,
                τ, τq₄_2, α_half, cache)

            # Check convergence
            counter += 1
            if maximum(abs, sol) < epsilon || counter ≥ max_newton_iterations
                break
            end
        end

        # After completing the Newton's method, update displacement
        @. dⁿ = τ_2 * (vⁿ + vⁿ⁻¹) + dⁿ⁻¹
        @. zⁿ = τ_2 * (rⁿ + rⁿ⁻¹) + zⁿ⁻¹

        # Process solution at tₙ (dispatch)
        process_solution!(
            output_data, output, n, times[n + 1], vⁿ, dⁿ, rⁿ, zⁿ,
            mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, input_data)

        # Rotate array references (no allocation)
        vⁿ⁻¹, vⁿ = vⁿ, vⁿ⁻¹
        dⁿ⁻¹, dⁿ = dⁿ, dⁿ⁻¹
        rⁿ⁻¹, rⁿ = rⁿ, rⁿ⁻¹
        zⁿ⁻¹, zⁿ = zⁿ, zⁿ⁻¹
    end

    return output_data
end

# ============================================
# Cache structure definition
# ============================================

"""
    CNCache{T}

Cache structure for Crank-Nicolson solver.

# Fields
"""
Base.@kwdef struct CNCache{T <: Real}
    vec₁m₁::Vector{T}
    vec₂m₁::Vector{T}
    vec₃m₁::Vector{T}
    vec₁m₂::Vector{T}
    vec₂m₂::Vector{T}
    vec₃m₂::Vector{T}
end

"""
    CNCache(m₁::Integer, m₂::Integer, ::Type{T}) where T

Construct cache with all vectors initialized to zero.

# Arguments
- `m₁`: Number of free DOFs in 2D domain Ω
- `m₂`: Number of free DOFs on 1D boundary Γ₁  
- `T`: Element type
"""
function CNCache(m₁::Integer, m₂::Integer, ::Type{T}) where {T}
    return CNCache(
        vec₁m₁ = zeros(T, m₁),
        vec₂m₁ = zeros(T, m₁),
        vec₃m₁ = zeros(T, m₁),
        vec₁m₂ = zeros(T, m₂),
        vec₂m₂ = zeros(T, m₂),
        vec₃m₂ = zeros(T, m₂)
    )
end

# ============================================
# Helper functions for system matrix assembly
# ============================================

"""
    compute_A₁₁!(A₁₁, M_m₁xm₁, τ²_4, α_half, K_m₁xm₁)

Compute system matrix A₁₁ = M_m₁×m₁ + (τ²/4)α(tₙ₊₁/₂)K_m₁×m₁.
"""
function compute_A₁₁!(
        A₁₁::AbstractMatrix,
        M_m₁xm₁::AbstractMatrix,
        τ²_4::Real,
        α_half::Real,
        K_m₁xm₁::AbstractMatrix
)
    cst = τ²_4 * α_half
    @. A₁₁.data = M_m₁xm₁.data + cst * K_m₁xm₁.data
    return nothing
end

"""
    compute_A₁₂!(A₁₂, τ_2, α_half, M_m₁xm₂)

Compute system matrix A₁₂ = -(τ/2)α(tₙ₊₁/₂)M_m₁×m₂.
"""
function compute_A₁₂!(
        A₁₂::AbstractMatrix,
        τ_2::Real,
        α_half::Real,
        M_m₁xm₂::AbstractMatrix
)
    cst = -τ_2 * α_half
    @. A₁₂ = cst * M_m₁xm₂
    return nothing
end

# ============================================
# Helper functions for right-hand side assembly
# ============================================

"""
    compute_Lm₁!(Lm₁, vⁿ⁻¹, rⁿ⁻¹, vec_d_v, matrices, input_data, mesh2D, dof_map_m₁, quad, τ, t_half, α_half, cache)

```
Lm₁ = M_m₁×m₁⋅vⁿ⁻¹ - τα(tₙ₊₁/₂)K_m₁×m₁⋅(dⁿ⁻¹+(τ/4)⋅vⁿ⁻¹) + (τ/2)α(tₙ₊₁/₂)M_m₁×m₂⋅rⁿ⁻¹ + τF(f₁(tₙ₊₁/₂))
```
"""
function compute_Lm₁!(Lm₁, vⁿ⁻¹, rⁿ⁻¹, vec_d_v, matrices, input_data,
        mesh2D, dof_map_m₁, quad, τ, t_half, α_half, cache)
    # Compute constants
    τα = τ * α_half
    τα_2 = τα / 2

    # Initialize Lm₁ = M_m₁×m₁⋅vⁿ⁻¹
    mul!(Lm₁, matrices.M_m₁xm₁, vⁿ⁻¹)

    # Compute K_m₁xm₁⋅(dⁿ⁻¹+(τ/4)⋅vⁿ⁻¹)
    # (will be scaled by -τ⋅α(tₙ₋₁/₂))
    mul!(cache.vec₁m₁, matrices.K_m₁xm₁, vec_d_v)

    # Compute M_m₂xm₂⋅rⁿ⁻¹
    # (will be scaled by (τ/2)⋅α(tₙ₋₁/₂))
    mul!(cache.vec₁m₂, matrices.M_m₂xm₂, rⁿ⁻¹)

    # Compute τ⋅F(f₁(tₙ₋₁/₂))
    assembly_rhs_2d!(
        cache.vec₂m₁, (x, y) -> input_data.f₁(x, y, t_half), τ * mesh2D.Δx[1] *
                                                             mesh2D.Δx[2] / 4,
        quad.W_φP, mesh2D, dof_map_m₁, quad.xP, quad.yP)

    # Assemble final Lm₁
    m₁ = length(vⁿ⁻¹)
    m₂ = length(rⁿ⁻¹)
    for i in 1:m₂
        Lm₁[i] = Lm₁[i] - τα * cache.vec₁m₁[i] + τα_2 * cache.vec₁m₂[i] + cache.vec₂m₁[i]
    end
    for i in (m₂ + 1):m₁
        Lm₁[i] = Lm₁[i] - τα * cache.vec₁m₁[i] + cache.vec₂m₁[i]
    end

    return nothing
end

"""
    compute_Lm₂!(Lm₂, rⁿ⁻¹, zⁿ⁻¹, vⁿ⁻¹, matrices, input_data, mesh1D, dof_map_m₂, quad, τ, t_half, τq₄_2, τq₃, cst_q, cache)

```
Lm₂ = M_m₂×m₂⋅(cst_q⋅rⁿ⁻¹ - τq₃⋅zⁿ⁻¹) - (τ/2)q₄⋅M_m₂×m₁⋅vⁿ⁻¹ + τ⋅F(f₂(tₙ₊₁/₂))
```
where `cst_q = q₁ - (τ/2)q₂ - (τ²/4)q₃`.
"""
function compute_Lm₂!(Lm₂, rⁿ⁻¹, zⁿ⁻¹, vⁿ⁻¹, matrices, input_data, mesh1D,
        dof_map_m₂, quad, τ, t_half, τq₄_2, τq₃, cst_q, cache)
    m₂ = length(Lm₂)

    # Initialize Lm₂ = M_m₂×m₂⋅(cst_q⋅rⁿ⁻¹ - τq₃⋅zⁿ⁻¹)
    @. cache.vec₁m₂ = cst_q * rⁿ⁻¹ - τq₃ * zⁿ⁻¹
    mul!(Lm₂, matrices.M_m₂xm₂, cache.vec₁m₂)

    # Compute M_m₂×m₂⋅vⁿ⁻¹[1:m₂]
    # (will be scaled by (τ/2)q₄)
    mul!(cache.vec₁m₂, matrices.M_m₂xm₂, view(vⁿ⁻¹, 1:m₂))

    # Compute τ⋅F(f₂(tₙ₋₁/₂))
    assembly_rhs_1d!(cache.vec₂m₂, x -> input_data.f₂(x, t_half),
        τ * mesh1D.Δx[1] / 2, quad.W_ϕP, mesh1D, dof_map_m₂, quad.xP)

    # Assemble final Lm₂
    @. Lm₂ = Lm₂ - τq₄_2 * cache.vec₁m₂ + cache.vec₂m₂

    return nothing
end

# ============================================
# Helper functions for Newton iteration
# ============================================
"""
    compute_minusHX!(minusHX, vⁿ, v̂ⁿ_m₂, d̂ⁿ, rⁿ, A₁₁, A₂₂, M_m₂xm₂, Lm₁, Lm₂, 
                     input_data, mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, 
                     τ, τq₄_2, α_half, cache)

Compute negative residual -H(X) for Newton's method.

The residual H(X) represents the nonlinear system:
```
H(X) = [A₁₁⋅vⁿ - (τ/2)α(tₙ₊₁/₂)M_m₁×m₂⋅rⁿ + τα(tₙ₊₁/₂)G(v̂ⁿ_m₂) + τF(d̂ⁿ) - Lm₁]
       [(τ/2)q₄⋅M_m₂×m₁⋅vⁿ + A₂₂⋅rⁿ - Lm₂                                    ]
```
"""
function compute_minusHX!(
        minusHX::AbstractVector{T},
        vⁿ::AbstractVector{T},
        v̂ⁿ_m₂::AbstractVector{T},
        d̂ⁿ::AbstractVector{T},
        rⁿ::AbstractVector{T},
        A₁₁::AbstractMatrix{T},
        A₂₂::AbstractMatrix{T},
        M_m₂xm₂::AbstractMatrix{T},
        Lm₁::AbstractVector{T},
        Lm₂::AbstractVector{T},
        input_data,
        mesh1D,
        mesh2D,
        dof_map_m₁,
        dof_map_m₂,
        quad,
        τ::T,
        τq₄_2::T,
        α_half::T,
        cache::CNCache{T}
) where {T}
    m₁ = dof_map_m₁.m
    m₂ = dof_map_m₂.m

    # Precompute constants
    τα = τ * α_half
    τα_2 = τα / 2

    # Compute A₁₁⋅vⁿ → cache.vec₁m₁
    mul!(cache.vec₁m₁, A₁₁, vⁿ)

    # Compute M_m₂×m₂⋅rⁿ → cache.vec₁m₂ (will be scaled by (τ/2)⋅α(tₙ₋₁/₂))
    mul!(cache.vec₁m₂, M_m₂xm₂, rⁿ)

    # Compute τα(tₙ₊₁/₂)⋅G(v̂ⁿ_m₂) → cache.vec₂m₂
    assembly_nonlinearity_G!(
        cache.vec₂m₂, τα, input_data.g, v̂ⁿ_m₂, mesh1D, dof_map_m₂, quad)

    # Compute τ⋅F(d̂ⁿ) → cache.vec₂m₁
    assembly_nonlinearity_F!(
        cache.vec₂m₁, τ, input_data.f, d̂ⁿ, mesh2D, dof_map_m₁, quad)

    # Assemble minusHX[1:m₁]
    for i in 1:m₂
        minusHX[i] = -cache.vec₁m₁[i] + τα_2 * cache.vec₁m₂[i] - cache.vec₂m₂[i] -
                     cache.vec₂m₁[i] +
                     Lm₁[i]
    end
    for i in (m₂ + 1):m₁
        minusHX[i] = -cache.vec₁m₁[i] - cache.vec₂m₁[i] + Lm₁[i]
    end

    # Compute M_m₂×m₂⋅vⁿ[1:m₂] → cache.vec₁m₂ (will be scaled by (τ/2)q₄)
    mul!(cache.vec₁m₂, M_m₂xm₂, view(vⁿ, 1:m₂))

    # Compute A₂₂⋅rⁿ → cache.vec₂m₂
    mul!(cache.vec₂m₂, A₂₂, rⁿ)

    # Assemble minusHX[(m₁+1):(m₁+m₂)]
    for i in 1:m₂
        minusHX[m₁ + i] = -τq₄_2 * cache.vec₁m₂[i] - cache.vec₂m₂[i] + Lm₂[i]
    end

    return nothing
end

"""
    compute_DHX₁₁!(DHX₁₁, A₁₁, v̂ⁿ_m₂, d̂ⁿ, input_data, mesh1D, mesh2D,
                   dof_map_m₁, dof_map_m₂, quad, τα_2, τ²_4, m₂)

```
DHX₁₁ = A₁₁ + (τ²/4)⋅DF(d̂ⁿ) + (τ/2)α(tₙ₋₁/₂)⋅DG(v̂ⁿ_m₂)
```
"""
function compute_DHX₁₁!(
        DHX₁₁, A₁₁, v̂ⁿ_m₂, d̂ⁿ, input_data, mesh1D, mesh2D,
        dof_map_m₁, dof_map_m₂, quad, τα_2, τ²_4, m₂)
    # Compute (τ/2)α(tₙ₋₁/₂)⋅DG(v̂ⁿ_m₃)
    DG = assembly_global_matrix_DG(
        τα_2, input_data.∂ₛg, v̂ⁿ_m₂, mesh1D, dof_map_m₂, quad)

    # Compute (τ²/4)⋅DF(d̂ⁿ)
    DF = assembly_global_matrix_DF(
        τ²_4, input_data.df, d̂ⁿ, mesh2D, dof_map_m₁, quad)

    # Compute DHX₁₁
    @. DHX₁₁.data = A₁₁.data + DF.data
    @. DHX₁₁.data[1:m₂, 1:m₂] += DG.data

    return nothing
end