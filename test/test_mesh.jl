@testitem "CartesianMesh 1D" begin
    using WaveAcoustics: CartesianMesh

    mesh = CartesianMesh((0.0,), (1.0,), (10,))
    @test mesh.pmin == (0.0,)
    @test mesh.pmax == (1.0,)
    @test mesh.Nx == (10,)
    @test mesh.Δx == (0.1,)
end

@testitem "CartesianMesh 2D" begin
    using WaveAcoustics: CartesianMesh

    mesh = CartesianMesh((0.0, 0.0), (1.0, 2.0), (10, 20))
    @test mesh.pmin == (0.0, 0.0)
    @test mesh.pmax == (1.0, 2.0)
    @test mesh.Nx == (10, 20)
    @test mesh.Δx == (0.1, 0.1)
end