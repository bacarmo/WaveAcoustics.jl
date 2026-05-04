@testitem "L2_error: u(x) = x(x-1), LagrangeElement{1,1}(), LeftRight()" begin
    using WaveAcoustics: L2_error, CartesianMesh, LagrangeElement, LeftRight,
                         DOFMap, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), (Δx, Δx), (0.0, 0.0))

    u(x) = x * (x - 1)
    uₕ_coefs = [u(i * Δx) for i in 1:(dof_map.m)]  # x = 0.25, 0.50, 0.75

    # Compute
    L2_err = L2_error(u, uₕ_coefs, mesh, dof_map, quad)

    # Analytical solution
    L2_err_expected = sqrt(1 / 7680)

    # Test
    @test L2_err ≈ L2_err_expected
    @test (@allocated L2_error(u, uₕ_coefs, mesh, dof_map, quad)) == 0
end

@testitem "L2_error: u(x,y) = x(x-1)y(y-1), 2×2 mesh, LagrangeElement{2,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error, CartesianMesh, LagrangeElement,
                         LeftRightBottomTop, DOFMap, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightBottomTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    u(x, y) = x * (x - 1) * y * (y - 1)
    Δx, Δy = mesh.Δx
    uₕ_coefs = [u(Δx, Δy)]  # single interior DOF at (0.5, 0.5)

    # Compute
    L2_err = L2_error(u, uₕ_coefs, mesh, dof_map, quad)

    # Analytical solution
    L2_err_expected = sqrt(4 * 29 / 614_400)

    # Test
    @test uₕ_coefs[1] ≈ 1 / 16
    @test L2_err ≈ L2_err_expected
    @test (@allocated L2_error(u, uₕ_coefs, mesh, dof_map, quad)) == 0
end

@testitem "L2_error: u(x,y) = x(x-1)y(y-1), 4×3 mesh, LagrangeElement{2,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error, CartesianMesh, LagrangeElement,
                         LeftRightBottomTop, DOFMap, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightBottomTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    u(x, y) = x * (x - 1) * y * (y - 1)
    Δx, Δy = mesh.Δx
    Nx, Ny = mesh.Nx
    uₕ_coefs = [u(i * Δx, j * Δy) for j in 1:(Ny - 1) for i in 1:(Nx - 1)]

    # Compute
    L2_err = L2_error(u, uₕ_coefs, mesh, dof_map, quad)

    # Analytical solution
    L2_err_expected = sqrt(
        4 * 77 / 74_649_600 +
        4 * 631 / 223_948_800 +
        2 * 11 / 6_220_800 +
        2 * 7 / 1_749_600
    )

    # Test
    @test all(uₕ_coefs[[1, 3, 4, 6]] .≈ 1 / 24)
    @test all(uₕ_coefs[[2, 5]] .≈ 1 / 18)
    @test L2_err≈L2_err_expected rtol=1e-12
    @test (@allocated L2_error(u, uₕ_coefs, mesh, dof_map, quad)) == 0
end

@testitem "L2_error: ‖φᵢ‖_L2, 4×3 mesh, LagrangeElement{2,1}(), LeftRightBottomTop()" begin
    using WaveAcoustics: L2_error, CartesianMesh, LagrangeElement,
                         LeftRightBottomTop, DOFMap, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightBottomTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    u(x, y) = 0.0
    Δx, Δy = mesh.Δx

    # Analytical solution
    # Each interior basis function φᵢ has ‖φᵢ‖_L2 = sqrt(4 * Δx * Δy / 9)
    L2_err_expected = sqrt(4 * Δx * Δy / 9)

    # Test
    for i in 1:(dof_map.m)
        uₕ_coefs = zeros(Float64, dof_map.m)
        uₕ_coefs[i] = 1.0
        @test L2_error(u, uₕ_coefs, mesh, dof_map, quad)≈L2_err_expected rtol=1e-12
    end
end