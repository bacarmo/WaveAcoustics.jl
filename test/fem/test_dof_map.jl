@testitem "EQ: LagrangeElement{2,1}(), LeftRightTop()" begin
    using WaveAcoustics: build_EQ, LagrangeElement, LeftRightTop

    Nx = (4, 3)
    EQ, m = build_EQ(Nx, LagrangeElement{2, 1}(), LeftRightTop())

    nx = 5  # 1*4 + 1
    ny = 4  # 1*3 + 1

    @test m == 9           # nx * ny - 2 * ny - (nx - 2)
    @test length(EQ) == 20 # nx * ny

    # Equation numbering of prescribed DOFs (left, right, top) should be m+1
    for j in 1:ny
        @test EQ[(j - 1) * nx + 1] == m + 1  # Left
        @test EQ[j * nx] == m + 1            # Right
    end
    for i in 1:nx
        @test EQ[(ny - 1) * nx + i] == m + 1  # Top
    end

    # Equation numbering of free DOFs
    @test EQ[[2, 3, 4, 7, 8, 9, 12, 13, 14]] == 1:9
end

@testitem "EQ: LagrangeElement{2,2}(), LeftRightTop()" begin
    using WaveAcoustics: build_EQ, LagrangeElement, LeftRightTop

    Nx = (4, 3)
    EQ, m = build_EQ(Nx, LagrangeElement{2, 2}(), LeftRightTop())

    nx = 9  # 2*4 + 1
    ny = 7  # 2*3 + 1

    @test m == 42          # nx * ny - 2 * ny - (nx - 2)
    @test length(EQ) == 63 # nx * ny

    # Equation numbering of prescribed DOFs (left, right, top) should be m+1
    for j in 1:ny
        @test EQ[(j - 1) * nx + 1] == m + 1  # Left
        @test EQ[j * nx] == m + 1            # Right
    end
    for i in 1:nx
        @test EQ[(ny - 1) * nx + i] == m + 1  # Top
    end

    # Equation numbering of free DOFs (j=1:6, i=2:8)
    expected_indices = [(j - 1) * nx + i for j in 1:6 for i in 2:8]
    @test EQ[expected_indices] == 1:42
end

@testitem "EQ: LagrangeElement{2,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: build_EQ, LagrangeElement, LeftRightBottomTop

    Nx = (4, 3)
    EQ, m = build_EQ(Nx, LagrangeElement{2, 1}(), LeftRightBottomTop())

    nx = 5  # 1*4 + 1
    ny = 4  # 1*3 + 1

    @test m == 6           # nx * ny - 2 * ny - 2 * (nx - 2)
    @test length(EQ) == 20 # nx * ny

    # Equation numbering of prescribed DOFs (left, right, bottom, top) should be m+1
    for j in 1:ny
        @test EQ[(j - 1) * nx + 1] == m + 1  # Left
        @test EQ[j * nx] == m + 1            # Right
    end
    for i in 1:nx
        @test EQ[(ny - 1) * nx + i] == m + 1  # Top
        @test EQ[i] == m + 1                  # Bottom
    end

    # Equation numbering of free DOFs
    @test EQ[[7, 8, 9, 12, 13, 14]] == 1:6
end

@testitem "EQ: LagrangeElement{2,2}(), LeftRightBottomTop()" begin
    using WaveAcoustics: build_EQ, LagrangeElement, LeftRightBottomTop

    Nx = (4, 3)
    EQ, m = build_EQ(Nx, LagrangeElement{2, 2}(), LeftRightBottomTop())

    nx = 9  # 2*4 + 1
    ny = 7  # 2*3 + 1

    @test m == 35          # nx * ny - 2 * ny - 2 * (nx - 2)
    @test length(EQ) == 63 # nx * ny

    # Equation numbering of prescribed DOFs (left, right, bottom, top) should be m+1
    for j in 1:ny
        @test EQ[(j - 1) * nx + 1] == m + 1  # Left
        @test EQ[j * nx] == m + 1            # Right
    end
    for i in 1:nx
        @test EQ[(ny - 1) * nx + i] == m + 1  # Top
        @test EQ[i] == m + 1                  # Bottom
    end

    # Equation numbering of free DOFs (j=2:6, i=2:8)
    free_indices = [(j - 1) * nx + i for j in 2:6 for i in 2:8]
    @test EQ[free_indices] == 1:35
end

@testitem "LG: LagrangeElement{1,1}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, LagrangeElement{1, 1}())

    @test length(LG) == 4
    @test LG[1] == [1, 2]
    @test LG[2] == [2, 3]
    @test LG[3] == [3, 4]
    @test LG[4] == [4, 5]
end

@testitem "LG: LagrangeElement{1,1}(), UInt8" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement
    using StaticArrays: SVector

    mesh = CartesianMesh((0.0,), (1.0,), (UInt8(4),))
    LG = build_LG(mesh, LagrangeElement{1, 1}())

    # Type correctness
    @test eltype(LG) == SVector{2, UInt8}
    @test LG[1] isa SVector{2, UInt8}

    # Values
    @test length(LG) == 4
    @test LG[1] == [0x01, 0x02]
    @test LG[2] == [0x02, 0x03]
    @test LG[3] == [0x03, 0x04]
    @test LG[4] == [0x04, 0x05]
end

@testitem "LG: LagrangeElement{1,2}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, LagrangeElement{1, 2}())

    @test length(LG) == 4
    @test LG[1] == [1, 2, 3]
    @test LG[2] == [3, 4, 5]
    @test LG[3] == [5, 6, 7]
    @test LG[4] == [7, 8, 9]
end

@testitem "LG: LagrangeElement{1,3}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    LG = build_LG(mesh, LagrangeElement{1, 3}())

    @test length(LG) == 4
    @test LG[1] == [1, 2, 3, 4]
    @test LG[2] == [4, 5, 6, 7]
    @test LG[3] == [7, 8, 9, 10]
    @test LG[4] == [10, 11, 12, 13]
end

@testitem "LG: LagrangeElement{2,1}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, LagrangeElement{2, 1}())

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

@testitem "LG: LagrangeElement{2,2}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, LagrangeElement{2, 2}())

    @test length(LG) == 12
    @test LG[1] == [1, 2, 3, 10, 11, 12, 19, 20, 21]
    @test LG[2] == [3, 4, 5, 12, 13, 14, 21, 22, 23]
    @test LG[5] == [19, 20, 21, 28, 29, 30, 37, 38, 39]
    @test LG[12] == [43, 44, 45, 52, 53, 54, 61, 62, 63]
end

@testitem "LG: LagrangeElement{2,3}()" begin
    using WaveAcoustics: build_LG, CartesianMesh, LagrangeElement

    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.1), (4, 3))
    LG = build_LG(mesh, LagrangeElement{2, 3}())

    @test length(LG) == 12
    @test LG[1] == [1, 2, 3, 4, 14, 15, 16, 17, 27, 28, 29, 30, 40, 41, 42, 43]
    @test LG[2][[1, 5, 9, 13]] == [4, 17, 30, 43]
    @test LG[5][[1, 2, 3, 4]] == [40, 41, 42, 43]
    @test LG[6][1] == 43
end

@testitem "DOFMap: LagrangeElement{1,1}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, LagrangeElement, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dof_map = DOFMap(mesh, LagrangeElement{1, 1}(), LeftRight())

    @test dof_map.m == 3
    @test length(dof_map.EQoLG) == 4
    @test dof_map.EQoLG[1] == [4, 1]
    @test dof_map.EQoLG[2] == [1, 2]
    @test dof_map.EQoLG[3] == [2, 3]
    @test dof_map.EQoLG[4] == [3, 4]
end

@testitem "DOFMap: LagrangeElement{1,1}(), LeftRight(), UInt8" begin
    using WaveAcoustics: DOFMap, CartesianMesh, LagrangeElement, LeftRight
    using StaticArrays: SVector

    mesh = CartesianMesh((0.0,), (1.0,), (UInt8(4),))
    dof_map = DOFMap(mesh, LagrangeElement{1, 1}(), LeftRight())

    # Type correctness
    @test typeof(dof_map.m) == UInt8
    @test eltype(dof_map.EQoLG) == SVector{2, UInt8}
    @test dof_map.EQoLG[1] isa SVector{2, UInt8}

    # Values
    @test dof_map.m == 0x03
    @test length(dof_map.EQoLG) == 4
    @test dof_map.EQoLG[1] == [0x04, 0x01]
    @test dof_map.EQoLG[2] == [0x01, 0x02]
    @test dof_map.EQoLG[3] == [0x02, 0x03]
    @test dof_map.EQoLG[4] == [0x03, 0x04]
end

@testitem "DOFMap: LagrangeElement{1,2}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, LagrangeElement, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dof_map = DOFMap(mesh, LagrangeElement{1, 2}(), LeftRight())

    @test dof_map.m == 7
    @test length(dof_map.EQoLG) == 4
    @test dof_map.EQoLG[1] == [8, 1, 2]
    @test dof_map.EQoLG[2] == [2, 3, 4]
    @test dof_map.EQoLG[3] == [4, 5, 6]
    @test dof_map.EQoLG[4] == [6, 7, 8]
end

@testitem "DOFMap: LagrangeElement{1,3}(), LeftRight()" begin
    using WaveAcoustics: DOFMap, CartesianMesh, LagrangeElement, LeftRight

    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dof_map = DOFMap(mesh, LagrangeElement{1, 3}(), LeftRight())

    @test dof_map.m == 11
    @test length(dof_map.EQoLG) == 4
    @test dof_map.EQoLG[1] == [12, 1, 2, 3]
    @test dof_map.EQoLG[2] == [3, 4, 5, 6]
    @test dof_map.EQoLG[3] == [6, 7, 8, 9]
    @test dof_map.EQoLG[4] == [9, 10, 11, 12]
end