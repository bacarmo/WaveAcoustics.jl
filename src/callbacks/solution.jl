"""
    SolutionCallback{T}

Callback that stores the full solution history at every time step.

## Fields
- `v::Matrix{T}`: DOF coefficients of the wave velocity field (m₁ × nt)
- `d::Matrix{T}`: DOF coefficients of the wave displacement field (m₁ × nt)
- `r::Matrix{T}`: DOF coefficients of the acoustic velocity (m₂ × nt)
- `z::Matrix{T}`: DOF coefficients of the acoustic displacement (m₂ × nt)
- `times::Vector{T}`: Time grid (size `nt`)
"""
struct SolutionCallback{T} <: AbstractCallback
    v::Matrix{T}
    d::Matrix{T}
    r::Matrix{T}
    z::Matrix{T}
    times::Vector{T}
end

"""
    SolutionCallback(m₁, m₂, m₃, times)

Allocate a `SolutionCallback` with spatial dimensions `m₁` (wave fields, 2D),
`m₂` (acoustic fields, 1D), and time grid `times`.
"""
function SolutionCallback(
        m₁::Integer,
        m₂::Integer,
        times::Vector{T}
) where {T}
    nt = length(times)
    return SolutionCallback(
        Matrix{T}(undef, m₁, nt),
        Matrix{T}(undef, m₁, nt),
        Matrix{T}(undef, m₂, nt),
        Matrix{T}(undef, m₂, nt),
        times
    )
end

function affect!(cb::SolutionCallback, state::FEMState, args...)
    cb.v[:, state.n + 1] .= state.v
    cb.d[:, state.n + 1] .= state.d
    cb.r[:, state.n + 1] .= state.r
    cb.z[:, state.n + 1] .= state.z
    return nothing
end