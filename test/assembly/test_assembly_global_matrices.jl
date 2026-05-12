@testitem "assembly_global_matrix: LagrangeElement{1,1}(), LeftRight(), Me" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRight
    using SparseArrays: sparse
    using StaticArrays: SMatrix

    # Setup
    # 1D mesh with 4 linear elements; LeftRight() removes boundary DOFs:
    # Basis layout:  [x  1  2  3  x]  (x = removed, 3 free interior DOFs)
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())

    Δx = mesh.Δx[1]
    Me = (Δx / 6) * SMatrix{2, 2}(2, 1, 1, 2)

    # Compute
    M = assembly_global_matrix(Me, dof_map)

    # Analytical solution
    #! format: off
    M_expected = (Δx / 6) * sparse([
    #  1  2  3
       4  1  0;  # DOF 1
       1  4  1;  # DOF 2
       0  1  4   # DOF 3
    ])
    #! format: on

    # Test
    @test M ≈ M_expected
end

@testitem "assembly_global_matrix: LagrangeElement{1,1}(), LeftRight(), Me symmetric" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRight
    using SparseArrays: SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (2^3,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    Me = (Δx / 6) * SMatrix{2, 2}(2, 1, 1, 2)
    Me_sym = Symmetric(Me)

    # Compute
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Test
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix: LagrangeElement{2,1}(), LeftRightTop(), Me" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRightTop
    using StaticArrays: SMatrix
    using SparseArrays: sparse

    # Setup
    # 2D mesh with 4×3 bilinear elements; LeftRightTop() removes boundary DOFs:
    # Basis layout (5×4 grid, 9 free DOFs):
    # Row 4 (top):    [x  x  x  x  x]  - Top BC (removed)
    # Row 3:          [x  7  8  9  x]  - Left/Right BC (removed)
    # Row 2:          [x  4  5  6  x]  - Left/Right BC (removed)
    # Row 1 (bottom): [x  1  2  3  x]  - Left/Right BC (removed), bottom free
    mesh = CartesianMesh((0.0, 0.0), (16.0, 27.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    Δx, Δy = mesh.Δx
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])

    # Compute
    M = assembly_global_matrix(Me, dof_map)

    # Analytical solution
    #! format: off
    M_expected = (Δx * Δy / 36) * sparse([
    #  1   2   3   4   5   6   7   8   9
       8   2   0   4   1   0   0   0   0;  # DOF 1
       2   8   2   1   4   1   0   0   0;  # DOF 2
       0   2   8   0   1   4   0   0   0;  # DOF 3
       4   1   0  16   4   0   4   1   0;  # DOF 4
       1   4   1   4  16   4   1   4   1;  # DOF 5
       0   1   4   0   4  16   0   1   4;  # DOF 6
       0   0   0   4   1   0  16   4   0;  # DOF 7
       0   0   0   1   4   1   4  16   4;  # DOF 8
       0   0   0   0   1   4   0   4  16   # DOF 9
    ])
    #! format: on

    # Test
    @test M ≈ M_expected
end

@testitem "assembly_global_matrix: LagrangeElement{2,1}(), LeftRightTop(), Me symmetric" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRightTop
    using SparseArrays: SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2^3, 2^3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    Δx, Δy = mesh.Δx
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])
    Me_sym = Symmetric(Me)

    # Compute
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Test
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix_DG: LagrangeElement{1,1}(), LeftRight(), ∂ₛg(x,v) = 1.0" begin
    using WaveAcoustics: assembly_global_matrix_DG, assembly_local_matrix_ϕxϕ,
                         assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRight, QuadratureSetup
    using SparseArrays: SparseMatrixCSC
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), (Δx, Δx), (0.0, 0.0))

    ∂ₛg(x, s) = 1.0
    v = ones(Float64, dof_map.m)

    # Compute
    DG_global = assembly_global_matrix_DG(1.0, ∂ₛg, v, mesh, dof_map, quad)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    DG_global_expected = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test
    @test DG_global isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test DG_global ≈ DG_global_expected
    @test size(DG_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DG: LagrangeElement{1,1}(), LeftRight(), ∂ₛg(x,v) = x²+v²" begin
    using WaveAcoustics: assembly_global_matrix_DG, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRight, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), (Δx, Δx), (0.0, 0.0))

    ∂ₛg(x, s) = x^2 + s^2
    v = ones(Float64, dof_map.m)

    # Compute
    DG_global = assembly_global_matrix_DG(1.0, ∂ₛg, v, mesh, dof_map, quad)

    # Expected solution
    #! format: off
    DG_global_expected = (1/8) * [
        17/40 + 11/15      61/160              0.0;
        61/160             191/240 + 211/240   223/480;
        0.0                223/480             59/60 + 101/120
    ]
    #! format: on

    # Test
    @test DG_global ≈ DG_global_expected
    @test size(DG_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DF: LagrangeElement{2,1}(), LeftRightTop(), f(s) = 1.0" begin
    using WaveAcoustics: assembly_global_matrix_DF, assembly_local_matrix_ϕxϕ,
                         assembly_global_matrix, CartesianMesh, LagrangeElement,
                         DOFMap, LeftRightTop, QuadratureSetup
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    f(s) = 1.0
    d = ones(Float64, dof_map.m)

    # Compute
    DF_global = assembly_global_matrix_DF(1.0, f, d, mesh, dof_map, quad)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    DF_global_expected = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test
    @test DF_global ≈ DF_global_expected
    @test size(DF_global) == (dof_map.m, dof_map.m)
end