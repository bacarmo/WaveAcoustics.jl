"""
    pde_solve(Nx, τ, input_data)

Solve coupled wave-acoustic PDE system and compute L∞(0,T;L²) errors.

## Arguments
- `Nx::NTuple{2,Integer}`: Number of elements per direction
- `τ::Real`: Time step size
- `input_data::PDEInputData`: Problem configuration with manufactured solution 

## Returns
- `(LooL2_v, LooL2_u, LooL2_r, LooL2_z)`: L∞(0,T;L²) errors
"""
function pde_solve(
        Nx::NTuple{2, I},
        τ::T,
        input_data
) where {I <: Integer, T <: Real}
    # ========================================
    # Spatial and temporal discretization
    # ========================================
    common = input_data.common
    pmin, pmax = common.pmin, common.pmax

    mesh2D = CartesianMesh(pmin, pmax, Nx)
    mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (Nx[1],))

    times = range(zero(T), common.t_final; step = τ)
    nt = length(times)

    # ========================================
    # Quadrature setup (Gauss-Legendre)
    # ========================================
    quad = QuadratureSetup(mesh2D.Δx, pmin)

    # ========================================
    # Local-to-global mapping
    # ========================================
    dof_map_m₁ = DOFMap(mesh2D, Lagrange{2, 1}(), LeftRightTop())
    dof_map_m₂ = DOFMap(mesh1D, Lagrange{1, 1}(), LeftRight())

    # ========================================
    # Assembly local matrices
    # ========================================

    # ========================================
    # Assembly global matrices
    # ========================================

end