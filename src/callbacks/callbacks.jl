# ========================================
# FEMState
# ========================================
"""
    FEMState{T}

Mutable snapshot of the FEM solution, updated at each time step.
`n` and `t` are updated in-place at each iteration; solution vectors
are fixed references whose contents are overwritten by the time integrator.

## Type Parameters
- `T <: AbstractFloat`: Scalar floating-point type (e.g., `Float64`)
- `V <: AbstractVector{T}`: Concrete vector type (e.g., `Vector{T}`)

## Fields
- `n::Int`: Current time step index
- `t::T`: Current time
- `v::V`: DOF coefficients of the wave velocity FEM solution Vⁿ(x,y) (2D, size m₁)
- `d::V`: DOF coefficients of the wave displacement FEM solution Uⁿ(x,y) (2D, size m₁)
- `r::V`: DOF coefficients of the acoustic velocity FEM solution Rⁿ(x) (1D, size m₂)
- `z::V`: DOF coefficients of the acoustic displacement FEM solution Zⁿ(x) (1D, size m₂)

## Size constraints
- `length(v) == length(d)` (shared mesh size m₁)
- `length(r) == length(z)` (shared mesh size m₂)
"""
mutable struct FEMState{T <: AbstractFloat, V <: AbstractVector{T}}
    n::Int
    t::T
    const v::V
    const d::V
    const r::V
    const z::V
end

# ========================================
# AbstractCallback
# ========================================
"""
    AbstractCallback

Abstract type for callbacks invoked at each accepted time step.

A concrete subtype must implement:
- `affect!(cb, state::FEMState, args...)`: action to perform
- `condition(cb, state::FEMState)` (optional): predicate controlling when to fire (default: `true`)
"""
abstract type AbstractCallback end

"""
    condition(cb::AbstractCallback, state::FEMState) -> Bool

Predicate controlling whether `affect!` is invoked at the current time step.
Default implementation always returns `true`.
"""
condition(::AbstractCallback, ::FEMState) = true

"""
    apply!(cb::AbstractCallback, state::FEMState, args...)

Invoke `affect!(cb, state, args...)` if `condition(cb, state)` is `true`.
"""
function apply!(cb::AbstractCallback, state::FEMState, args...)
    condition(cb, state) && affect!(cb, state, args...)
    return nothing
end

# ========================================
# Concrete callbacks
# ========================================
include("l2_error.jl")
include("solution.jl")
include("energy.jl")