@testitem "CrankNicolson3Cache: build_maps_* & sync_JH!" begin
    using WaveAcoustics: Lagrange, CartesianMesh, specialize, DOFMap, LeftRightTop,
                         LeftRight, SystemMatrices, CrankNicolson3Cache, sync_JH!
    using SparseArrays: nnz

    # --- Setup ---
    # The Jacobian JH is a (m₁+m₂)×(m₁+m₂) sparse matrix assembled from four blocks:
    #
    #   JH = [ Q₁₁   Q₁₂ ]    Q₁₁ : m₁×m₁
    #        [ Q₂₁   Q₂₂ ]    Q₂₂ : m₂×m₂  (m₁ > m₂)
    #
    # Q₁₂ = [Q̄₁₂; 0] and Q₂₁ = [Q̄₂₁ 0] embed m₂×m₂ blocks with zero padding.
    #
    # sync_JH! uses precomputed index maps (map_direct, map_mirror) to update
    # JH.nzval in-place without recomputing positions at every time step.
    #
    # The four block tests run sequentially on the same cache.JH: each test
    # perturbs one block, syncs it into JH, and checks against a dense reference.

    function setup(fe)
        pmin, pmax = (0.0, 0.0), (1.0, 1.0)
        mesh1D = CartesianMesh((pmin[1],), (pmax[1],), (4,))
        mesh2D = CartesianMesh(pmin, pmax, (4, 3))
        fe1D = specialize(fe, Val(1))
        fe2D = specialize(fe, Val(2))

        dof_map_m₁ = DOFMap(mesh2D, fe2D, LeftRightTop())
        dof_map_m₂ = DOFMap(mesh1D, fe1D, LeftRight())
        m₁, m₂ = dof_map_m₁.m, dof_map_m₂.m

        matrices = SystemMatrices(mesh1D, mesh2D, fe1D, fe2D, dof_map_m₁, dof_map_m₂)
        cache = CrankNicolson3Cache(matrices)

        return cache, m₁, m₂
    end

    @testset "$fe" for fe in (Lagrange{1}(), Lagrange{2}(), Lagrange{3}())
        cache, m₁, m₂ = setup(fe)

        @testset "Q₁₁  → JH[1:m₁, 1:m₁]" begin
            cache.Q₁₁.data.nzval .= 1:nnz(cache.Q₁₁.data)
            JH_ref = copy(cache.JH)
            JH_ref[1:m₁, 1:m₁] .= cache.Q₁₁
            sync_JH!(cache.JH, cache.Q₁₁, cache.map_direct_11, cache.map_mirror_11)
            @test cache.JH == JH_ref
        end

        @testset "Q̄₂₁  → JH[m₁+1:m₁+m₂, 1:m₂]" begin
            cache.Q̄₂₁.data.nzval .= 1:nnz(cache.Q̄₂₁.data)
            JH_ref = copy(cache.JH)
            JH_ref[(m₁ + 1):(m₁ + m₂), 1:m₂] .= cache.Q̄₂₁
            sync_JH!(cache.JH, cache.Q̄₂₁, cache.map_direct_21, cache.map_mirror_21)
            @test cache.JH == JH_ref
        end

        @testset "Q̄₁₂  → JH[1:m₂, m₁+1:m₁+m₂]" begin
            cache.Q̄₁₂.data.nzval .= 1:nnz(cache.Q̄₁₂.data)
            JH_ref = copy(cache.JH)
            JH_ref[1:m₂, (m₁ + 1):(m₁ + m₂)] .= cache.Q̄₁₂
            sync_JH!(cache.JH, cache.Q̄₁₂, cache.map_direct_12, cache.map_mirror_12)
            @test cache.JH == JH_ref
        end

        @testset "Q₂₂   → JH[m₁+1:m₁+m₂, m₁+1:m₁+m₂]" begin
            cache.Q₂₂.data.nzval .= 1:nnz(cache.Q₂₂.data)
            JH_ref = copy(cache.JH)
            JH_ref[(m₁ + 1):(m₁ + m₂), (m₁ + 1):(m₁ + m₂)] .= cache.Q₂₂
            sync_JH!(cache.JH, cache.Q₂₂, cache.map_direct_22, cache.map_mirror_22)
            @test cache.JH == JH_ref
        end
    end
end