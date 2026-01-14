@testitem "L2_error_1d: u(x) = x(x-1), Lagrange{1,1}(), LeftRight()" begin
    using WaveAcoustics: L2_error_1d, CartesianMesh, Lagrange, LeftRight, DOFMap,
                         QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    dof_map = DOFMap(mesh, Lagrange{1, 1}(), LeftRight())
    quad = QuadratureSetup((mesh.Δx[1], mesh.Δx[1]), (0.0, 0.0))

    # Exact solution and interpolant (internal nodes only)
    u(x) = x * (x - 1)
    Δx = mesh.Δx[1]
    uₕ_coefs = [u(i * Δx) for i in 1:(dof_map.m)]  # i=1,2,3 → x=0.25,0.5,0.75

    # Test correctness
    computed = L2_error_1d(u, uₕ_coefs, mesh, dof_map, quad)
    expected = sqrt(1 / 7680)  # Analytical L2 error
    @test computed ≈ expected

    # Test performance
    alloc = @allocations L2_error_1d(u, uₕ_coefs, mesh, dof_map, quad)
    @test alloc == 0
end

@testitem "L2_error_2d: u(x,y) = x(x-1)y(y-1), 2x2 mesh, Lagrange{1,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error_2d, CartesianMesh, Lagrange, LeftRightBottomTop, DOFMap,
                         QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    dof_map = DOFMap(mesh, Lagrange{2, 1}(), LeftRightBottomTop())
    quad = QuadratureSetup(mesh.Δx, mesh.pmin)

    # Exact solution and interpolant (internal nodes only)
    u(x, y) = x * (x - 1) * y * (y - 1)
    Δx, Δy = mesh.Δx
    uₕ_coefs = [u(Δx, Δy)]

    # Test correctness
    computed = L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)
    expected = sqrt(4 * 29 / 614_400)  # Analytical L2 error
    @test computed ≈ expected
    @test uₕ_coefs[1] ≈ 1 / 16

    # Test performance
    alloc = @allocations L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)
    @test alloc == 0
end

@testitem "L2_error_2d: u(x,y) = x(x-1)y(y-1), 4x3 mesh, Lagrange{1,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error_2d, CartesianMesh, Lagrange, LeftRightBottomTop, DOFMap,
                         QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    dof_map = DOFMap(mesh, Lagrange{2, 1}(), LeftRightBottomTop())
    quad = QuadratureSetup(mesh.Δx, mesh.pmin)

    # Exact solution and interpolant (internal nodes only)
    u(x, y) = x * (x - 1) * y * (y - 1)
    Δx, Δy = mesh.Δx
    Nx, Ny = mesh.Nx
    uₕ_coefs = [u(i * Δx, j * Δy) for j in 1:(Ny - 1) for i in 1:(Nx - 1)]

    # Test correctness
    computed = L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)
    expected = sqrt(
        4 * 77 / 74_649_600 +
        4 * 631 / 223_948_800 +
        2 * 11 / 6_220_800 +
        2 * 7 / 1_749_600
    )  # Analytical L2 error
    @test computed≈expected rtol=1e-12
    @test all(uₕ_coefs[[1, 3, 4, 6]] .≈ 1 / 24)
    @test all(uₕ_coefs[[2, 5]] .≈ 1 / 18)

    # Test performance
    alloc = @allocations L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)
    @test alloc == 0
end

@testitem "L2_error_2d: ‖φᵢ‖_L2, 4x3 mesh, Lagrange{1,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error_2d, CartesianMesh, Lagrange, LeftRightBottomTop, DOFMap,
                         QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    dof_map = DOFMap(mesh, Lagrange{2, 1}(), LeftRightBottomTop())
    quad = QuadratureSetup(mesh.Δx, mesh.pmin)

    # Exact solution and expected L2 norm of single basis function 
    u(x, y) = 0.0
    Δx, Δy = mesh.Δx
    expected = sqrt(4 * Δx * Δy / 9)

    # Test: ‖φᵢ‖_L2 = expected for each basis function
    for i in 1:(dof_map.m)
        uₕ_coefs = zeros(Float64, dof_map.m)
        uₕ_coefs[i] = 1.0

        computed = L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)
        @test computed≈expected rtol=1e-12
    end
end