@testitem "assembly_rhs_1d!: LagrangeElement{1,1}(), LeftRight(), ∫f(x)*ϕᵢ(x) dx" begin
    using WaveAcoustics: assembly_rhs_1d!, CartesianMesh, LagrangeElement,
                         LeftRight, DOFMap, basis_functions
    using StaticArrays: SVector, @SVector
    using GaussQuadrature: legendre

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]

    P_raw, W_raw = legendre(4)
    P = SVector{4}(P_raw)
    W = SVector{4}(W_raw)
    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]

    scale = Δx / 2
    ϕ(ξ) = basis_functions(family, ξ)
    W_basisP = @SVector [W[k] * ϕ(P[k]) for k in 1:4]

    m = dof_map.m
    F = zeros(Float64, m)

    # Test f(x) = 1
    assembly_rhs_1d!(F, x -> 1.0, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ fill(1 / 4, m)

    # Test f(x) = x
    assembly_rhs_1d!(F, x -> x, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ [1 / 16, 1 / 8, 3 / 16]

    # Test f(x) = sin(x)
    alloc = @allocated assembly_rhs_1d!(F, sin, scale, W_basisP, mesh, dof_map, xP)
    F_expected = [
        8 * sin(1 / 4) - 4 * sin(1 / 2),
        -4 * sin(1 / 4) + 8 * sin(1 / 2) - 4 * sin(3 / 4),
        -4 * sin(1 / 2) + 8 * sin(3 / 4) - 4 * sin(1)
    ]
    @test F ≈ F_expected
    @test alloc == 0
end

@testitem "assembly_rhs_1d!: LagrangeElement{1,1}(), LeftRight(), ∫f(x)*dϕᵢ(x) dx" begin
    using WaveAcoustics: assembly_rhs_1d!, CartesianMesh, LagrangeElement,
                         LeftRight, DOFMap, basis_functions_derivatives
    using StaticArrays: SVector, @SVector
    using GaussQuadrature: legendre

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]

    P_raw, W_raw = legendre(4)
    P = SVector{4}(P_raw)
    W = SVector{4}(W_raw)
    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]

    dϕ(ξ) = basis_functions_derivatives(family, ξ)
    W_basisP = @SVector [W[k] * dϕ(P[k]) for k in 1:4]

    scale = 1.0  # == (Δx/2) * (2/Δx)
    m = dof_map.m
    F = zeros(Float64, m)

    # Test f(x) = 1
    assembly_rhs_1d!(F, x -> 1.0, scale, W_basisP, mesh, dof_map, xP)
    @test F≈[0.0, 0.0, 0.0] atol=1e-15

    # Test f(x) = x
    assembly_rhs_1d!(F, x -> x, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ [-1 / 4, -1 / 4, -1 / 4]

    # Test f(x) = sin(x)
    alloc = @allocated assembly_rhs_1d!(F, sin, scale, W_basisP, mesh, dof_map, xP)
    F_expected = [
        8 * sin(1 / 8)^2 - 4 * cos(1 / 4) + 4 * cos(1 / 2),
        4 * cos(1 / 4) - 8 * cos(1 / 2) + 4 * cos(3 / 4),
        4 * cos(1 / 2) - 8 * cos(3 / 4) + 4 * cos(1)
    ]
    @test F≈F_expected atol=1e-14
    @test alloc == 0
end

@testitem "assembly_rhs_2d!: LagrangeElement{2,1}(), LeftRightBottomTop(), ∫f(x,y)*φᵢ(x,y) dx dy" begin
    using WaveAcoustics: assembly_rhs_2d!, CartesianMesh, LagrangeElement,
                         LeftRightBottomTop, DOFMap, basis_functions
    using StaticArrays: SVector, @SMatrix
    using GaussQuadrature: legendre

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightBottomTop())
    Δx, Δy = mesh.Δx

    Npg = 2
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg, Float64}(P_raw)
    W = SVector{Npg, Float64}(W_raw)

    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]
    yP = @. (Δy / 2) * (P + 1) + mesh.pmin[2]

    φ(ξ, η) = basis_functions(family, ξ, η)
    W_basisP = @SMatrix [W[i] * W[j] * φ(P[i], P[j]) for i in 1:Npg, j in 1:Npg]

    scale = Δx * Δy / 4
    m = dof_map.m
    F = zeros(Float64, m)

    # Test f(x,y) = 1
    assembly_rhs_2d!(F, (x, y) -> 1.0, scale, W_basisP, mesh, dof_map, xP, yP)
    @test F ≈ fill(Δx * Δy, m)

    # Test f(x,y) = x*(x-1)*y*(y-1)
    f₂(x, y) = x * (x - 1) * y * (y - 1)
    alloc = @allocated assembly_rhs_2d!(F, f₂, scale, W_basisP, mesh, dof_map, xP, yP)
    cst1 = 13 / 27648 + 7 / 9216 + 169 / 248_832 + 91 / 82_944
    cst2 = 2 * 23 / 27_648 + 2 * 299 / 248_832
    @test F[1] ≈ F[3] ≈ F[4] ≈ F[6] ≈ cst1
    @test F[2] ≈ F[5] ≈ cst2
    @test alloc == 0
end

@testitem "assembly_nonlinearity_F!: LagrangeElement{2,1}(), LeftRightTop(), f(u) = u, d = ones(m)" begin
    using WaveAcoustics: assembly_nonlinearity_F!, CartesianMesh, LagrangeElement,
                         LeftRightTop, DOFMap, QuadratureSetup,
                         assembly_global_matrix, assembly_local_matrix_ϕxϕ

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    f(u) = u
    scale = 1.0
    d = ones(Float64, dof_map.m)
    F = zeros(Float64, dof_map.m)

    # Compute
    alloc = @allocated assembly_nonlinearity_F!(F, scale, f, d, mesh, dof_map, quad)

    # Expected solution
    # If f(u) = u and d = ones, F = M·d where M is mass matrix
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    F_expected = assembly_global_matrix(Me, dof_map) * d

    # Test
    @test F ≈ F_expected
    @test alloc == 0
end

@testitem "assembly_nonlinearity_G!: LagrangeElement{1,1}(), LeftRight(), g(x,v) = v, v = ones(m)" begin
    using WaveAcoustics: assembly_nonlinearity_G!, CartesianMesh, LagrangeElement,
                         LeftRight, DOFMap, QuadratureSetup,
                         assembly_global_matrix, assembly_local_matrix_ϕxϕ

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), (Δx, Δx), (0.0, 0.0))

    g(x, v) = v
    scale = 1.0
    v = ones(Float64, dof_map.m)
    G = zeros(Float64, dof_map.m)

    # Compute
    alloc = @allocated assembly_nonlinearity_G!(G, scale, g, v, mesh, dof_map, quad)

    # Expected solution
    # If g(x, v) = v and v = ones, G = M·v where M is mass matrix
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    G_expected = assembly_global_matrix(Me, dof_map) * v

    # Test
    @test G ≈ G_expected
    @test alloc == 0
end