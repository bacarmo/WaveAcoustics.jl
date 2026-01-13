@testitem "assembly_rhs_1d!: Lagrange{1,1}(), LeftRight(), ∫f(x)*ϕᵢ(x) dx" begin
    using WaveAcoustics: assembly_rhs_1d!, CartesianMesh, Lagrange, LeftRight, DOFMap,
                         basis_functions
    using StaticArrays: SVector, @SVector
    using GaussQuadrature: legendre

    # Setup: 1D mesh with 4 linear elements
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())

    # Quadrature data
    P_raw, W_raw = legendre(4)
    P = SVector{4}(P_raw)
    W = SVector{4}(W_raw)
    Δx = mesh.Δx[1]
    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]

    # Define scale and W_basisP related to ∫f(x)*ϕᵢ(x) dx
    scale = Δx / 2
    ϕ(ξ) = basis_functions(family, ξ)
    W_basisP = @SVector [W[k] * ϕ(P[k]) for k in 1:4]

    # Initialize global RHS vector
    m = dof_map.m
    F = zeros(Float64, m)

    # Test 1: f(x) = 1
    f₁(x) = 1.0
    alloc = @allocated assembly_rhs_1d!(F, f₁, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ fill(1 / 4, m)
    @test alloc == 0  # No allocations should occur within the function

    # Test 2: f(x) = x
    f₂(x) = x
    alloc = @allocated assembly_rhs_1d!(F, f₂, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ [1 / 16, 1 / 8, 3 / 16]
    @test alloc == 0

    # Test 2: f(x) = sin(x)
    f₃(x) = sin(x)
    alloc = @allocated assembly_rhs_1d!(F, f₃, scale, W_basisP, mesh, dof_map, xP)
    expected = [
        8 * sin(1 / 4) - 4 * sin(1 / 2),
        -4 * sin(1 / 4) + 8 * sin(1 / 2) - 4 * sin(3 / 4),
        -4 * sin(1 / 2) + 8 * sin(3 / 4) - 4 * sin(1)
    ]
    @test F ≈ expected
    @test alloc == 0
end

@testitem "assembly_rhs_1d!: Lagrange{1,1}(), LeftRight(), ∫f(x)*dϕᵢ(x) dx" begin
    using WaveAcoustics: assembly_rhs_1d!, CartesianMesh, Lagrange, LeftRight, DOFMap,
                         basis_functions_derivatives
    using StaticArrays: SVector, @SVector
    using GaussQuadrature: legendre

    # Setup: 1D mesh with 4 linear elements
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())

    # Quadrature data
    P_raw, W_raw = legendre(4)
    P = SVector{4}(P_raw)
    W = SVector{4}(W_raw)
    Δx = mesh.Δx[1]
    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]

    # Define scale and W_basisP related to ∫f(x)*dϕᵢ(x) dx
    scale = 1.0 # == (Δx/2) * (2/Δx)
    dϕ(ξ) = basis_functions_derivatives(family, ξ)
    W_basisP = @SVector [W[k] * dϕ(P[k]) for k in 1:4]

    # Initialize global RHS vector
    m = dof_map.m
    F = zeros(Float64, m)

    # Test 1: f(x) = 1
    f₁(x) = 1.0
    alloc = @allocated assembly_rhs_1d!(F, f₁, scale, W_basisP, mesh, dof_map, xP)
    @test F≈[0.0, 0.0, 0.0] atol=1e-15
    @test alloc == 0  # No allocations should occur within the function

    # Test 2: f(x) = x
    f₂(x) = x
    alloc = @allocated assembly_rhs_1d!(F, f₂, scale, W_basisP, mesh, dof_map, xP)
    @test F ≈ [-1 / 4, -1 / 4, -1 / 4]
    @test alloc == 0

    # Test 3: f(x) = sin(x)
    f₃(x) = sin(x)
    alloc = @allocated assembly_rhs_1d!(F, f₃, scale, W_basisP, mesh, dof_map, xP)
    expected = [
        8 * sin(1 / 8)^2 - 4 * cos(1 / 4) + 4 * cos(1 / 2),
        4 * cos(1 / 4) - 8 * cos(1 / 2) + 4 * cos(3 / 4),
        4 * cos(1 / 2) - 8 * cos(3 / 4) + 4 * cos(1)]
    @test F≈expected atol=1e-14
    @test alloc == 0
end

@testitem "assembly_rhs_2d!: Lagrange{2,1}(), LeftRightBottomTop(), ∫f(x,y)*φᵢ(x,y) dx dy" begin
    using WaveAcoustics: assembly_rhs_2d!, CartesianMesh, Lagrange, LeftRightBottomTop,
                         DOFMap, basis_functions
    using StaticArrays: SVector, @SMatrix
    using GaussQuadrature: legendre

    # Domain: Ω = ]0,1[×]0,1[, uniform mesh with 4×3 elements, Bilinear Lagrange basis and essential BC
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = Lagrange{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightBottomTop())

    # Gauss–Legendre quadrature setup
    Npg = 2
    P_raw, W_raw = legendre(Npg)
    W = SVector{Npg, Float64}(W_raw)
    P = SVector{Npg, Float64}(P_raw)

    φ(ξ, η) = basis_functions(family, ξ, η)
    W_basisP = @SMatrix [@. W[i] * W[j] * φ(P[i], P[j]) for i in 1:Npg, j in 1:Npg]

    Δx, Δy = mesh.Δx
    xP = @. (Δx / 2) * (P + 1) + mesh.pmin[1]
    yP = @. (Δy / 2) * (P + 1) + mesh.pmin[2]

    # Initialize global RHS vector
    m = dof_map.m
    F = zeros(Float64, m)

    # Test 1: f(x,y) = 1
    f₁(x, y) = 1.0
    scale = Δx * Δy / 4
    alloc = @allocated assembly_rhs_2d!(F, f₁, scale, W_basisP, mesh, dof_map, xP, yP)
    @test F ≈ fill(Δx * Δy, m)
    @test alloc == 0  # No allocations should occur within the function

    # Test 2: f(x,y) = x*(x-1)*y*(y-1)
    f₂(x, y) = x * (x - 1) * y * (y - 1)
    scale = Δx * Δy / 4
    alloc = @allocated assembly_rhs_2d!(F, f₂, scale, W_basisP, mesh, dof_map, xP, yP)

    cst1 = 13 / 27648 + 7 / 9216 + 169 / 248_832 + 91 / 82_944
    cst2 = 2 * 23 / 27_648 + 2 * 299 / 248_832

    @test F[1] ≈ F[3] ≈ F[4] ≈ F[6] ≈ cst1
    @test F[2] ≈ F[5] ≈ cst2
    @test alloc == 0
end