"""
    State{T}

Mutable snapshot of the FEM solution at time `tₙ`. 
`n` and `t` are updated in-place by the solver loop at each iteration; 
solution vectors are fixed references whose contents are overwritten by the time integrator.

## Fields
- `n::Int`: Current time step index (e.g. `0` for `t₀`)
- `t::T`: Current time
- `v::Vector{T}`: FE coefficients vector of the wave velocity solution Vⁿ(x,y) (size m₁)
- `d::Vector{T}`: FE coefficients vector of the wave displacement solution Uⁿ(x,y) (size m₁)
- `r::Vector{T}`: FE coefficients vector of the acoustic velocity solution Rⁿ(x) (size m₂)
- `z::Vector{T}`: FE coefficients vector of the acoustic displacement solution Zⁿ(x) (size m₂)
"""
mutable struct State{T <: AbstractFloat}
    n::Int
    t::T
    const v::Vector{T}
    const d::Vector{T}
    const r::Vector{T}
    const z::Vector{T}
end

"""
    AbstractCallback

Abstract type for callbacks invoked at each time step via [`apply!`](@ref).
The signature of `apply!` is defined per concrete subtype.
"""
abstract type AbstractCallback end

# ==============================================================================
# L2ErrorCallback
# ==============================================================================
"""
    L2ErrorCallback{T}

Callback that computes the L²(Ω) error between the numerical and exact solutions at each time step.

## Fields
- `v_errors::Vector{T}`: L² errors for the wave velocity field (`length(tspan)`)
- `d_errors::Vector{T}`: L² errors for the wave displacement field
- `r_errors::Vector{T}`: L² errors for the acoustic velocity
- `z_errors::Vector{T}`: L² errors for the acoustic displacement
"""
struct L2ErrorCallback{T} <: AbstractCallback
    v_errors::Vector{T}
    d_errors::Vector{T}
    r_errors::Vector{T}
    z_errors::Vector{T}
end

"""
    L2ErrorCallback(tspan)

Allocate an `L2ErrorCallback` for `length(tspan)` time steps.
"""
function L2ErrorCallback(tspan::StepRangeLen{T}) where {T}
    return L2ErrorCallback(
        zeros(T, length(tspan)), zeros(T, length(tspan)),
        zeros(T, length(tspan)), zeros(T, length(tspan)))
end

"""
    apply!(cb, state, nel_per_dim, element_side_lengths, dof_map_m₁, dof_map_m₂, quad, input_data)

Compute the L²(Ω) error at the current time step.

## Arguments
- `cb::L2ErrorCallback{T}`: Callback accumulating the error history
- `state::State{T}`: Current solver state
- `nel_per_dim::NTuple{2, Int}`: Number of elements along each spatial dimension
- `element_side_lengths::NTuple{2, T}`: Element side lengths `(Δx, Δy)`
- `dof_map_m₁`: DOF map for the `m₁`-space
- `dof_map_m₂`: DOF map for the `m₂`-space
- `quad`: Quadrature setup with fields `W`, `xP`, `yP`, `φP`, `ϕP`
- `input_data`: PDE input data with the exact solutions
"""
function apply!(cb::L2ErrorCallback{T}, state::State{T},
        nel_per_dim::NTuple{2, Int}, element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap, dof_map_m₂::DOFMap,
        quad::QuadratureSetup, input_data::PDEInputData) where {T}
    Δx, Δy = element_side_lengths

    cb.v_errors[state.n + 1] = error_L2(
        (x, y) -> input_data.v(x, y, state.t), state.v,
        dof_map_m₁, nel_per_dim, element_side_lengths,
        quad.W, quad.xP, quad.yP, quad.φP)
    cb.d_errors[state.n + 1] = error_L2(
        (x, y) -> input_data.u(x, y, state.t), state.d,
        dof_map_m₁, nel_per_dim, element_side_lengths,
        quad.W, quad.xP, quad.yP, quad.φP)
    cb.r_errors[state.n + 1] = error_L2(
        x -> input_data.r(x, state.t), state.r,
        dof_map_m₂, Δx, quad.W, quad.xP, quad.ϕP)
    cb.z_errors[state.n + 1] = error_L2(
        x -> input_data.z(x, state.t), state.z,
        dof_map_m₂, Δx, quad.W, quad.xP, quad.ϕP)
    return nothing
end

# ==============================================================================
# SolutionCallback
# ==============================================================================
"""
    SolutionCallback{T}

Callback that stores the full FE coefficient history at every time step.

## Fields
- `tspan::StepRangeLen{T}`: uniform time grid
- `v::Vector{Vector{T}}`: FE coefficient history; entry `n+1` holds the solution at `tspan[n+1]`
- `d::Vector{Vector{T}}`: FE coefficient history; entry `n+1` holds the solution at `tspan[n+1]`
- `r::Vector{Vector{T}}`: FE coefficient history; entry `n+1` holds the solution at `tspan[n+1]`
- `z::Vector{Vector{T}}`: FE coefficient history; entry `n+1` holds the solution at `tspan[n+1]`
"""
struct SolutionCallback{T} <: AbstractCallback
    tspan::StepRangeLen{T}
    v::Vector{Vector{T}}
    d::Vector{Vector{T}}
    r::Vector{Vector{T}}
    z::Vector{Vector{T}}
end

"""
    SolutionCallback(tspan)

Allocate a `SolutionCallback` for `length(tspan)` time steps.
The solution history is grown lazily via `push!` at each call to [`apply!`](@ref).
"""
function SolutionCallback(tspan::StepRangeLen{T}) where {T}
    return SolutionCallback(tspan, Vector{Vector{T}}(),
        Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{Vector{T}}())
end

"""
    apply!(cb, state, args...)

Append a copy of the current FE coefficient vector.
Extra arguments are accepted and ignored so that all `AbstractCallback` subtypes
share a uniform call signature at the solver level.

## Arguments
- `cb::SolutionCallback{T}`: Callback accumulating the solution history
- `state::State{T}`: Current solver state
"""
function apply!(cb::SolutionCallback{T}, state::State{T}, args...) where {T}
    push!(cb.v, copy(state.v))
    push!(cb.d, copy(state.d))
    push!(cb.r, copy(state.r))
    push!(cb.z, copy(state.z))
    return nothing
end