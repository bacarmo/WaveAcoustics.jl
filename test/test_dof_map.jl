@testitem "LG: Lagrange{1, 1}()" begin
    using WaveAcoustics: build_LG, CartesianMesh

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, Lagrange{1, 1}())

    @test length(LG) == 4
    @test LG[1] == [1, 2]
    @test LG[2] == [2, 3]
    @test LG[3] == [3, 4]
    @test LG[4] == [4, 5]
end

@testitem "LG: Lagrange{1, 1}(), UInt8" begin
    using WaveAcoustics: build_LG, CartesianMesh
    using StaticArrays: SVector

    mesh = CartesianMesh((0.0,), (1.0,), (UInt8(4),))
    LG = build_LG(mesh, Lagrange{1, 1}())

    # Test type
    @test eltype(LG) == SVector{2, UInt8}
    @test LG[1] isa SVector{2, UInt8}

    # Test values
    @test length(LG) == 4
    @test LG[1] == [0x01, 0x02]
    @test LG[2] == [0x02, 0x03]
    @test LG[3] == [0x03, 0x04]
    @test LG[4] == [0x04, 0x05]
end

@testitem "LG: Lagrange{1, 2}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, Lagrange

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, Lagrange{1, 2}())

    @test length(LG) == 4
    @test LG[1] == [1, 2, 3]
    @test LG[2] == [3, 4, 5]
    @test LG[3] == [5, 6, 7]
    @test LG[4] == [7, 8, 9]
end

@testitem "LG: Lagrange{1, 3}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, Lagrange

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, Lagrange{1, 3}())

    @test length(LG) == 4
    @test LG[1] == [1, 2, 3, 4]
    @test LG[2] == [4, 5, 6, 7]
    @test LG[3] == [7, 8, 9, 10]
    @test LG[4] == [10, 11, 12, 13]
end

@testitem "LG: Lagrange{2, 1}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, Lagrange

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, Lagrange{2, 1}())

    @test length(LG) == 12
    @test LG[1] == [1, 2, 6, 7]
    @test LG[2] == [2, 3, 7, 8]
    @test LG[3] == [3, 4, 8, 9]
    @test LG[4] == [4, 5, 9, 10]
    @test LG[5] == [6, 7, 11, 12]
    @test LG[6] == [7, 8, 12, 13]
    @test LG[7] == [8, 9, 13, 14]
    @test LG[8] == [9, 10, 14, 15]
    @test LG[9] == [11, 12, 16, 17]
    @test LG[10] == [12, 13, 17, 18]
    @test LG[11] == [13, 14, 18, 19]
    @test LG[12] == [14, 15, 19, 20]
end

@testitem "LG: Lagrange{2, 2}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, Lagrange

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, Lagrange{2, 2}())

    @test length(LG) == 12
    @test LG[1] == [1, 2, 3, 10, 11, 12, 19, 20, 21]
    @test LG[2] == [3, 4, 5, 12, 13, 14, 21, 22, 23]
    @test LG[5] == [19, 20, 21, 28, 29, 30, 37, 38, 39]
    @test LG[12] == [43, 44, 45, 52, 53, 54, 61, 62, 63]
end

@testitem "LG: Lagrange{2, 3}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, Lagrange

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, Lagrange{2, 3}())

    @test length(LG) == 12
    @test LG[1] == [1, 2, 3, 4, 14, 15, 16, 17, 27, 28, 29, 30, 40, 41, 42, 43]
    @test LG[2][[1, 5, 9, 13]] == [4, 17, 30, 43]
    @test LG[5][[1, 2, 3, 4]] == [40, 41, 42, 43]
    @test LG[6][1] == 43
end

@testitem "DOFMap: Lagrange{1, 1}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, Lagrange, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dofmap = DOFMap(mesh, Lagrange{1, 1}(), LeftRight())

    @test dofmap.m == 3
    @test length(dofmap.EQoLG) == 4
    @test dofmap.EQoLG[1] == [4, 1]
    @test dofmap.EQoLG[2] == [1, 2]
    @test dofmap.EQoLG[3] == [2, 3]
    @test dofmap.EQoLG[4] == [3, 4]
end

@testitem "DOFMap: Lagrange{1, 1}(), LeftRight(), UInt8" begin
    using WaveAcoustics: DOFMap, CartesianMesh, Lagrange, LeftRight
    using StaticArrays: SVector

    mesh = CartesianMesh((0.0,), (1.0,), (UInt8(4),))
    dofmap = DOFMap(mesh, Lagrange{1, 1}(), LeftRight())

    # Test type
    @test typeof(dofmap.m) == UInt8
    @test eltype(dofmap.EQoLG) == SVector{2, UInt8}
    @test dofmap.EQoLG[1] isa SVector{2, UInt8}

    # Test values
    @test dofmap.m == 0x03
    @test length(dofmap.EQoLG) == 4
    @test dofmap.EQoLG[1] == [0x04, 0x01]
    @test dofmap.EQoLG[2] == [0x01, 0x02]
    @test dofmap.EQoLG[3] == [0x02, 0x03]
    @test dofmap.EQoLG[4] == [0x03, 0x04]
end

@testitem "DOFMap: Lagrange{1, 2}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, Lagrange, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dofmap = DOFMap(mesh, Lagrange{1, 2}(), LeftRight())

    @test dofmap.m == 7
    @test length(dofmap.EQoLG) == 4
    @test dofmap.EQoLG[1] == [8, 1, 2]
    @test dofmap.EQoLG[2] == [2, 3, 4]
    @test dofmap.EQoLG[3] == [4, 5, 6]
    @test dofmap.EQoLG[4] == [6, 7, 8]
end

@testitem "DOFMap: Lagrange{1, 3}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, Lagrange, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dofmap = DOFMap(mesh, Lagrange{1, 3}(), LeftRight())

    @test dofmap.m == 11
    @test length(dofmap.EQoLG) == 4
    @test dofmap.EQoLG[1] == [12, 1, 2, 3]
    @test dofmap.EQoLG[2] == [3, 4, 5, 6]
    @test dofmap.EQoLG[3] == [6, 7, 8, 9]
    @test dofmap.EQoLG[4] == [9, 10, 11, 12]
end