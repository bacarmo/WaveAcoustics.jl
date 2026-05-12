@testitem "CartesianMesh: 1D" begin
    using WaveAcoustics: CartesianMesh

    mesh = CartesianMesh((0.0,), (1.0,), (4,))

    @test mesh.pmin == (0.0,)
    @test mesh.pmax == (1.0,)
    @test mesh.Nx == (4,)
    @test mesh.Δx == (0.25,)  # (1.0 - 0.0) / 4
end

@testitem "CartesianMesh: 2D" begin
    using WaveAcoustics: CartesianMesh

    mesh = CartesianMesh((0.0, 0.0), (1.0, 2.0), (4, 5))

    @test mesh.pmin == (0.0, 0.0)
    @test mesh.pmax == (1.0, 2.0)
    @test mesh.Nx == (4, 5)
    @test mesh.Δx == (0.25, 0.4)  # (1.0 - 0.0) / 4, (2.0 - 0.0) / 5
end

@testitem "CartesianMesh: integer type propagation" begin
    using WaveAcoustics: CartesianMesh

    mesh_int64 = CartesianMesh((0.0,), (1.0,), (4,))
    mesh_uint8 = CartesianMesh((0.0,), (1.0,), (UInt8(4),))

    @test mesh_int64 isa CartesianMesh{1, Int64}
    @test mesh_uint8 isa CartesianMesh{1, UInt8}
end