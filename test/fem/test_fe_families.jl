@testitem "specialize: Lagrange" begin
    using WaveAcoustics: specialize, Lagrange, LagrangeElement

    # 1D
    @test specialize(Lagrange{1}(), Val(1)) == LagrangeElement{1, 1}()
    @test specialize(Lagrange{2}(), Val(1)) == LagrangeElement{1, 2}()
    @test specialize(Lagrange{3}(), Val(1)) == LagrangeElement{1, 3}()

    # 2D
    @test specialize(Lagrange{1}(), Val(2)) == LagrangeElement{2, 1}()
    @test specialize(Lagrange{2}(), Val(2)) == LagrangeElement{2, 2}()
    @test specialize(Lagrange{3}(), Val(2)) == LagrangeElement{2, 3}()
end

@testitem "specialize: Hermite" begin
    using WaveAcoustics: specialize, Hermite, HermiteElement

    # 1D
    @test specialize(Hermite{3}(), Val(1)) == HermiteElement{1, 3}()

    # 2D
    @test specialize(Hermite{3}(), Val(2)) == HermiteElement{2, 3}()
end

@testitem "num_local_dof: LagrangeElement" begin
    using WaveAcoustics: num_local_dof, LagrangeElement

    # 1D: (Deg + 1)^1
    @test num_local_dof(LagrangeElement{1, 1}()) == 2
    @test num_local_dof(LagrangeElement{1, 2}()) == 3
    @test num_local_dof(LagrangeElement{1, 3}()) == 4

    # 2D: (Deg + 1)^2
    @test num_local_dof(LagrangeElement{2, 1}()) == 4
    @test num_local_dof(LagrangeElement{2, 2}()) == 9
    @test num_local_dof(LagrangeElement{2, 3}()) == 16
end

@testitem "num_local_dof: HermiteElement" begin
    using WaveAcoustics: num_local_dof, HermiteElement

    # 1D: 4^1
    @test num_local_dof(HermiteElement{1, 3}()) == 4

    # 2D: 4^2
    @test num_local_dof(HermiteElement{2, 3}()) == 16
end

@testitem "polynomial_degree: LagrangeElement and Lagrange" begin
    using WaveAcoustics: polynomial_degree, LagrangeElement, Lagrange

    # LagrangeElement (dimension-aware)
    @test polynomial_degree(LagrangeElement{1, 1}()) == 1
    @test polynomial_degree(LagrangeElement{1, 2}()) == 2
    @test polynomial_degree(LagrangeElement{1, 3}()) == 3
    @test polynomial_degree(LagrangeElement{2, 1}()) == 1
    @test polynomial_degree(LagrangeElement{2, 3}()) == 3

    # Lagrange (dimension-agnostic)
    @test polynomial_degree(Lagrange{1}()) == 1
    @test polynomial_degree(Lagrange{2}()) == 2
    @test polynomial_degree(Lagrange{3}()) == 3
end

@testitem "polynomial_degree: HermiteElement and Hermite" begin
    using WaveAcoustics: polynomial_degree, HermiteElement, Hermite

    # HermiteElement (dimension-aware)
    @test polynomial_degree(HermiteElement{1, 3}()) == 3
    @test polynomial_degree(HermiteElement{2, 3}()) == 3

    # Hermite (dimension-agnostic)
    @test polynomial_degree(Hermite{3}()) == 3
end