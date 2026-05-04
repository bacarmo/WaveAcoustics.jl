"""
    L2ErrorCallback{T}

Callback that computes the L²(Ω) error between the numerical and exact solutions at each time step.

## Fields
- `v_errors::Vector{T}`: L² errors for the wave velocity field (size `nt`)
- `d_errors::Vector{T}`: L² errors for the wave displacement field (size `nt`)
- `r_errors::Vector{T}`: L² errors for the acoustic velocity (size `nt`)
- `z_errors::Vector{T}`: L² errors for the acoustic displacement (size `nt`)
"""
struct L2ErrorCallback{T} <: AbstractCallback
    v_errors::Vector{T}
    d_errors::Vector{T}
    r_errors::Vector{T}
    z_errors::Vector{T}
end

"""
    L2ErrorCallback(tspan)

Allocate an `L2ErrorCallback` for `nt` time steps with floating-point type `T`.
"""
function L2ErrorCallback(tspan::StepRangeLen{T}) where {T}
    nt = length(tspan)
    return L2ErrorCallback(
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt)
    )
end

function affect!(
        cb::L2ErrorCallback,
        state::FEMState{T},
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap,
        quad::QuadratureSetup,
        input_data
) where {T}
    cb.v_errors[state.n + 1] = L2_error(
        (x, y) -> input_data.v(x, y, state.t), state.v, mesh2D, dof_map_m₁, quad)
    cb.d_errors[state.n + 1] = L2_error(
        (x, y) -> input_data.u(x, y, state.t), state.d, mesh2D, dof_map_m₁, quad)
    cb.r_errors[state.n + 1] = L2_error(
        x -> input_data.r(x, state.t), state.r, mesh1D, dof_map_m₂, quad)
    cb.z_errors[state.n + 1] = L2_error(
        x -> input_data.z(x, state.t), state.z, mesh1D, dof_map_m₂, quad)
    return nothing
end