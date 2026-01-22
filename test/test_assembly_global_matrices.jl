@testitem "assembly_global_matrix: Lagrange{1,1}(), LeftRight(), Me" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, Lagrange, DOFMap, LeftRight
    using SparseArrays: sparse
    using StaticArrays: SMatrix

    # Setup: 1D mesh with 4 linear elements
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())

    # Local mass matrix
    Δx = mesh.Δx[1]
    Me = (Δx / 6) * SMatrix{2, 2}([2 1;
                                   1 2])

    # DOF layout (5 nodes total, 3 DOFs after BC removal):
    # [x  1  2  3  x]  - Left/Right BC (removed), interior free

    # Analytical global mass matrix
    #! format: off
    M_analytical = (Δx / 6) * sparse([
    #  1  2  3
       4  1  0;   # DOF 1
       1  4  1;   # DOF 2
       0  1  4    # DOF 3
    ])
    #! format: on

    # Assemble global mass matrix
    M = assembly_global_matrix(Me, dof_map)

    # Test
    @test M ≈ M_analytical
end

@testitem "assembly_global_matrix: Lagrange{1,1}(), LeftRight(), Me symmetric" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, Lagrange, DOFMap, LeftRight
    using SparseArrays: sparse, SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (2^3,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    Me = (Δx / 6) * SMatrix{2, 2}([2 1;
                                   1 2])
    Me_sym = Symmetric(Me)

    # Assemble global mass matrix
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Type tests
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}

    # Consistency test
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix: Lagrange{2,1}(), LeftRightTop(), Me" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, Lagrange, DOFMap,
                         LeftRightTop
    using StaticArrays: SMatrix
    using SparseArrays: sparse

    # Setup: 2D mesh with 4×3 bilinear elements
    mesh = CartesianMesh((0.0, 0.0), (16.0, 27.0), (4, 3))
    family = Lagrange{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())

    # Local mass matrix
    Δx, Δy = mesh.Δx
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])

    # DOF layout (5×4 nodes total, 9 DOFs after BC removal):
    # Row 4 (top):    [x  x  x  x  x]  - Top BC (removed)
    # Row 3:          [x  7  8  9  x]  - Left/Right BC (removed)
    # Row 2:          [x  4  5  6  x]  - Left/Right BC (removed)
    # Row 1 (bottom): [x  1  2  3  x]  - Left/Right BC (removed), bottom free

    # Analytical global mass matrix
    #! format: off
    M_analytical = (Δx * Δy / 36) * sparse([
    #  1   2   3   4   5   6   7   8   9
       8   2   0   4   1   0   0   0   0;   # DOF 1
       2   8   2   1   4   1   0   0   0;   # DOF 2
       0   2   8   0   1   4   0   0   0;   # DOF 3
       4   1   0  16   4   0   4   1   0;   # DOF 4
       1   4   1   4  16   4   1   4   1;   # DOF 5
       0   1   4   0   4  16   0   1   4;   # DOF 6
       0   0   0   4   1   0  16   4   0;   # DOF 7
       0   0   0   1   4   1   4  16   4;   # DOF 8
       0   0   0   0   1   4   0   4  16    # DOF 9
    ])
    #! format: on

    # Assemble global mass matrix
    M = assembly_global_matrix(Me, dof_map)

    # Test
    @test M ≈ M_analytical
end

@testitem "assembly_global_matrix: Lagrange{2,1}(), LeftRightTop(), Me symmetric" begin
    using WaveAcoustics: assembly_global_matrix, CartesianMesh, Lagrange, DOFMap,
                         LeftRightTop
    using SparseArrays: sparse, SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2^3, 2^3))
    family = Lagrange{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    Δx, Δy = mesh.Δx
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])
    Me_sym = Symmetric(Me)

    # Assemble global mass matrix
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Type tests
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}

    # Consistency test
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix_DG: Lagrange{1,1}(), LeftRight(), ∂ₛg(x,v) = 1.0" begin
    using WaveAcoustics: assembly_global_matrix_DG, assembly_local_matrix_ϕxϕ,
                         assembly_global_matrix, CartesianMesh, Lagrange, DOFMap, LeftRight,
                         QuadratureSetup
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    quad = QuadratureSetup((0.25, 0.25), (0.0, 0.0))

    # Test function and data
    @inline ∂ₛg(x, s) = 1.0
    v = ones(Float64, dof_map.m)

    # Reference solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    expected_DG_global = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test: Correctness
    DG_global = assembly_global_matrix_DG(1.0, ∂ₛg, v, mesh, dof_map, quad)

    @test DG_global ≈ expected_DG_global
    @test size(DG_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DF: Lagrange{2,1}(), LeftRightTop(), f(s) = 1.0" begin
    using WaveAcoustics: assembly_global_matrix_DF, assembly_local_matrix_ϕxϕ,
                         assembly_global_matrix, CartesianMesh, Lagrange, DOFMap,
                         LeftRightTop,
                         QuadratureSetup
    using LinearAlgebra: Symmetric

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = Lagrange{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    quad = QuadratureSetup(mesh.Δx, mesh.pmin)

    # Test function and data
    @inline f(s) = 1.0
    d = ones(Float64, dof_map.m)

    # Reference solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    expected_DF_global = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test: Correctness
    DF_global = assembly_global_matrix_DF(1.0, f, d, mesh, dof_map, quad)

    @test DF_global ≈ expected_DF_global
    @test size(DF_global) == (dof_map.m, dof_map.m)
end
