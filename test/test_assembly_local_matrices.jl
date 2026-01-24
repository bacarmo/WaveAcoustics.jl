@testitem "assembly_local_matrix_ϕxϕ: Lagrange{1,1}" begin
    using WaveAcoustics: assembly_local_matrix_ϕxϕ, CartesianMesh, Lagrange
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (2,))
    family = Lagrange{1, 1}()

    # Compute using quadrature
    Me_quad = assembly_local_matrix_ϕxϕ(mesh, family)

    # Analytical solution
    Δx = mesh.Δx[1]
    Me_analytical = SMatrix{2, 2}([2 1; 1 2]) * (Δx / 2) / 3

    # Test
    @test Me_quad ≈ Me_analytical
end

@testitem "assembly_local_matrix_ϕxϕ: Lagrange{2,1}" begin
    using WaveAcoustics: assembly_local_matrix_ϕxϕ, CartesianMesh, Lagrange
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    family = Lagrange{2, 1}()

    # Compute using quadrature
    Me_quad = assembly_local_matrix_ϕxϕ(mesh, family)

    # Analytical solution
    Δx, Δy = mesh.Δx
    Me_analytical = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                                    2 4 1 2;
                                                    2 1 4 2;
                                                    1 2 2 4])

    # Test
    @test Me_quad ≈ Me_analytical
end

@testitem "assembly_local_matrix_∇ϕx∇ϕ: Lagrange{2,1}" begin
    using WaveAcoustics: assembly_local_matrix_∇ϕx∇ϕ, CartesianMesh, Lagrange
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    family = Lagrange{2, 1}()

    # Compute using quadrature
    Ke_quad = assembly_local_matrix_∇ϕx∇ϕ(mesh, family)

    # Analytical solution
    Δx, Δy = mesh.Δx
    Ke_analytical = SMatrix{4, 4}([2 -2 1 -1; -2 2 -1 1; 1 -1 2 -2; -1 1 -2 2]) *
                    ((Δy / Δx) / 6) +
                    SMatrix{4, 4}([2 1 -2 -1; 1 2 -1 -2; -2 -1 2 1; -1 -2 1 2]) *
                    ((Δx / Δy) / 6)

    # Test
    @test Ke_quad ≈ Ke_analytical
end

@testitem "assembly_local_matrix_DG!: Lagrange{1,1}(), LeftRight(), ∂ₛg(x,v) = 1.0" begin
    using WaveAcoustics: assembly_local_matrix_DG!, assembly_local_matrix_ϕxϕ,
                         CartesianMesh, Lagrange, DOFMap, LeftRight, QuadratureSetup
    using FixedSizeArrays: FixedSizeArray

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
    expected_DG_local = Me * (2 / mesh.Δx[1])

    # Allocate output
    num_dof_local = length(quad.ϕP[1])
    DG_local = FixedSizeArray{Float64}(undef, num_dof_local, num_dof_local)

    # Test: Correctness
    assembly_local_matrix_DG!(DG_local, ∂ₛg, v, dof_map.m, dof_map.EQoLG[1], quad.xP, quad)

    for b in axes(DG_local, 2), a in 1:b # Upper triangle: a ≤ b
        @test DG_local[a, b] ≈ expected_DG_local[a, b]
    end

    # Test: Performance (allocation-free operation)
    ## Warn: Direct field access (dof_map.m, dof_map.EQoLG[1], quad.xP) is causing allocations. Why?
    m = dof_map.m
    eq = dof_map.EQoLG[1]
    xeP = quad.xP

    alloc = @allocated assembly_local_matrix_DG!(DG_local, ∂ₛg, v, m, eq, xeP, quad)
    @test alloc == 0
end

@testitem "assembly_local_matrix_DG!: Lagrange{1,1}(), LeftRight(), ∂ₛg(x,v) = x²+v²" begin
    using WaveAcoustics: assembly_local_matrix_DG!, CartesianMesh, Lagrange, DOFMap,
                         LeftRight, QuadratureSetup
    using FixedSizeArrays: FixedSizeArray

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = Lagrange{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup((Δx, Δx), (0.0, 0.0))

    # Test function and data
    @inline ∂ₛg(x, s) = x^2 + s^2
    v = ones(Float64, dof_map.m)

    # Reference solution for element e = 1, 2, 3, 4
    expected_DG_local = (
        [17/240 17/160; 17/160 17/40],    # e=1: [0.00, 0.25], Vₕ=(1+ξ)/2
        [11/15 61/160; 61/160 191/240],   # e=2: [0.25, 0.50], Vₕ=1
        [211/240 223/480; 223/480 59/60], # e=3: [0.50, 0.75], Vₕ=1
        [101/120 57/160; 57/160 157/240]  # e=4: [0.75, 1.00], Vₕ=(1-ξ)/2
    )

    # Allocate output
    num_dof_local = length(quad.ϕP[1])
    DG_local = FixedSizeArray{Float64}(undef, num_dof_local, num_dof_local)

    # Test: Correctness
    @testset "Element $e" for e in 1:4
        eq = dof_map.EQoLG[e]
        xeP = quad.xP .+ (e - 1) * Δx
        assembly_local_matrix_DG!(DG_local, ∂ₛg, v, dof_map.m, eq, xeP, quad)

        @testset "Entry ($a,$b)" for b in axes(DG_local, 2), a in 1:b # Upper triangle: a ≤ b
            @test DG_local[a, b] ≈ expected_DG_local[e][a, b]
        end
    end
end

@testitem "assembly_local_matrix_DG!: Lagrange{2,1}(), LeftRightTop(), f(s) = 1.0" begin
    using WaveAcoustics: assembly_local_matrix_DF!, assembly_local_matrix_ϕxϕ,
                         CartesianMesh, Lagrange, DOFMap, LeftRightTop, QuadratureSetup
    using FixedSizeArrays: FixedSizeArray

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
    expected_DF_local = Me * (4 / (mesh.Δx[1] * mesh.Δx[2]))

    # Allocate output
    num_dof_local = length(quad.φP[1, 1])
    DF_local = FixedSizeArray{Float64}(undef, num_dof_local, num_dof_local)

    # Test: Correctness
    assembly_local_matrix_DF!(
        DF_local, f, d, dof_map.m, dof_map.EQoLG[1], quad.W_φPφP, quad.φP)

    for b in axes(DF_local, 2), a in 1:b # Upper triangle: a ≤ b
        @test DF_local[a, b] ≈ expected_DF_local[a, b]
    end

    # Test: Performance (allocation-free operation)
    ## Warn: Direct field access (dof_map.m, dof_map.EQoLG[1], quad.W_φPφP, quad.φP) is causing allocations. Why?
    m = dof_map.m
    eq = dof_map.EQoLG[1]
    W_φPφP = quad.W_φPφP
    φP = quad.φP

    alloc = @allocated assembly_local_matrix_DF!(DF_local, f, d, m, eq, W_φPφP, φP)
    @test alloc == 0
end