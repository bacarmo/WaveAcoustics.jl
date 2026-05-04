# Key design facts verified:
#    1. `linsolve.A` and `JH` share the same `nzval` array.
#    2. `linsolve.b` and `minusH` share the same array.
#    3. `isfresh = true` triggers refactorization; `isfresh = false` suppresses it.
@testitem "cache (CrankNicolson1)" begin
    using WaveAcoustics: build_cache, CartesianMesh, CrankNicolson1,
                         DOFMap, Lagrange, LeftRight, LeftRightTop, specialize,
                         SystemMatrices
    using LinearAlgebra: Symmetric
    using SparseArrays: SparseMatrixCSC
    import LinearSolve as LS

    # --- Setup ---
    pmin, pmax = (0.0, 0.0), (1.0, 1.0)
    mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (4,))
    mesh2D = CartesianMesh(pmin, pmax, (4, 3))
    fe1D = specialize(Lagrange{1}(), Val(1))
    fe2D = specialize(Lagrange{1}(), Val(2))

    dof_map_m₁ = DOFMap(mesh2D, fe2D, LeftRightTop())
    dof_map_m₂ = DOFMap(mesh1D, fe1D, LeftRight())

    matrices = SystemMatrices(
        mesh1D, mesh2D, fe1D, fe2D, dof_map_m₁, dof_map_m₂)
    cache = build_cache(CrankNicolson1(), matrices)

    # Test types
    @test cache.Q isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test cache.JH isa SparseMatrixCSC{Float64, Int64}

    # Fact 1: the solver holds a reference to JH, not a copy.
    # In-place mutations to JH.nzval are immediately visible to the solver.
    @test cache.linsolve.A === cache.JH
    @test cache.linsolve.A.nzval == cache.JH.nzval

    # Fact 2: the solver holds a reference to minusH, not a copy.
    # In-place mutations to minusH are immediately visible to the solver.
    @test cache.linsolve.b === cache.minusH

    @. cache.minusH = 1.0
    @test all(cache.linsolve.b .== 1.0)

    # Fact 3: `isfresh = true` triggers refactorization; `isfresh = false` suppresses it.
    # ---Baseline: solve A x = b with isfresh = true 
    cache.linsolve.isfresh = true
    LS.solve!(cache.linsolve)
    sol_A = copy(cache.linsolve.u)

    @test cache.linsolve.A == matrices.M_m₁xm₁
    @test sol_A ≈ matrices.M_m₁xm₁ \ cache.linsolve.b

    # --- Mutate A → 2A in-place; set isfresh = true to trigger refactorization ---
    # Expected: (2A) x = b  ⟹  x = sol_A / 2
    cache.JH.nzval .*= 2.0
    cache.linsolve.isfresh = true
    LS.solve!(cache.linsolve)
    sol_2A = copy(cache.linsolve.u)
    @test sol_2A ≈ sol_A / 2

    # --- Change only b; set isfresh = false to reuse the factorization of 2A ---
    # Expected: (2A) x = 2b  ⟹  x = A⁻¹b = sol_A
    cache.linsolve.b .= 2.0
    cache.linsolve.isfresh = false
    LS.solve!(cache.linsolve)
    sol_2A_2b = copy(cache.linsolve.u)
    @test sol_2A_2b ≈ sol_A
end