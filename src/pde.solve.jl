"""
    pde_solve(Nx, fe, tspan, input_data, solver, callback)

Solve the coupled wave-acoustic PDE system, populating `callback` in-place.

## Arguments
- `Nx::NTuple{2,Integer}`: Number of elements per direction
- `fe::FEFamily`: Finite element family (e.g. `Lagrange{1}()`, `Hermite{3}()`)
- `tspan::StepRangeLen`: Time grid defined as `0:τ:t_end`
- `input_data::PDEInputData`: Problem configuration with manufactured solution
- `solver::ODESolver`: Time integration scheme
- `callback::AbstractCallback`: Callback invoked at each accepted time step

## Examples
```julia
Nx     = (10, 10)
fe     = Lagrange{1}()
tspan  = 0.0:0.01:1.0
id     = example1_manufactured()
cb     = L2ErrorCallback(tspan)
pde_solve(Nx, fe, tspan, id, CrankNicolson1(), cb)
```
"""
function pde_solve(
        Nx::NTuple{2, Int},
        fe::FEFamily,
        tspan::StepRangeLen{Float64},
        input_data::PDEInputData,
        solver::ODESolver,
        callback::AbstractCallback
)
    # ========================================
    # Spatial discretization
    # ========================================
    pmin, pmax = input_data.pmin, input_data.pmax

    mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (Nx[1],))
    mesh2D = CartesianMesh(pmin, pmax, Nx)

    # ========================================
    # Specialize FEFamily
    # ========================================
    fe1D = specialize(fe, Val(1))
    fe2D = specialize(fe, Val(2))

    # ========================================
    # Quadrature setup (Gauss-Legendre)
    # ========================================
    quad = QuadratureSetup(fe1D, fe2D, mesh2D.Δx, pmin)

    # ========================================
    # Local-to-global mapping
    # ========================================
    dof_map_m₁ = DOFMap(mesh2D, fe2D, LeftRightTop())
    dof_map_m₂ = DOFMap(mesh1D, fe1D, LeftRight())

    # ========================================
    # Assemble matrices
    # ========================================
    matrices = SystemMatrices(
        mesh1D, mesh2D, fe1D, fe2D, dof_map_m₁, dof_map_m₂)

    # ========================================
    # Compute initial state v⁰, d⁰, r⁰, z⁰ 
    # ========================================
    initial_state = compute_initial_state(
        matrices.K_m₁xm₁, matrices.M_m₂xm₂,
        dof_map_m₁, dof_map_m₂,
        mesh2D, mesh1D, quad, input_data)

    apply!(callback, initial_state,
        mesh1D, mesh2D, dof_map_m₁, dof_map_m₂, quad, input_data)

    # ========================================
    # Compute vⁿ, dⁿ, rⁿ, and zⁿ for n ≥ 1
    # ========================================
    cache = build_cache(solver, matrices)
    ode_solve(
        cache,
        initial_state,
        matrices,
        dof_map_m₁, dof_map_m₂,
        mesh1D, mesh2D,
        quad,
        tspan,
        input_data,
        callback)

    return nothing
end