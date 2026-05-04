@testitem "assembly_local_matrix_ϕxϕ: LagrangeElement{1,1}" begin
    using WaveAcoustics: assembly_local_matrix_ϕxϕ, CartesianMesh, LagrangeElement
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (2,))
    family = LagrangeElement{1, 1}()

    # Compute
    Me = assembly_local_matrix_ϕxϕ(mesh, family)

    # Expected solution
    Δx = mesh.Δx[1]
    Me_expected = (Δx / 6) * SMatrix{2, 2}([2 1;
                                            1 2])

    # Test
    @test Me ≈ Me_expected
end

@testitem "assembly_local_matrix_ϕxϕ: LagrangeElement{2,1}" begin
    using WaveAcoustics: assembly_local_matrix_ϕxϕ, CartesianMesh, LagrangeElement
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    family = LagrangeElement{2, 1}()

    # Compute
    Me = assembly_local_matrix_ϕxϕ(mesh, family)

    # Expected solution
    Δx, Δy = mesh.Δx
    Me_expected = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                                  2 4 1 2;
                                                  2 1 4 2;
                                                  1 2 2 4])

    # Test
    @test Me ≈ Me_expected
end

@testitem "assembly_local_matrix_∇ϕx∇ϕ: LagrangeElement{2,1}" begin
    using WaveAcoustics: assembly_local_matrix_∇ϕx∇ϕ, CartesianMesh, LagrangeElement
    using StaticArrays: SMatrix

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (2, 2))
    family = LagrangeElement{2, 1}()

    # Compute
    Ke = assembly_local_matrix_∇ϕx∇ϕ(mesh, family)

    # Expected solution
    Δx, Δy = mesh.Δx
    Ke_expected = ((Δy / Δx) / 6) *
                  SMatrix{4, 4}([2 -2 1 -1;
                                 -2 2 -1 1;
                                 1 -1 2 -2;
                                 -1 1 -2 2]) +
                  ((Δx / Δy) / 6) *
                  SMatrix{4, 4}([2 1 -2 -1;
                                 1 2 -1 -2;
                                 -2 -1 2 1;
                                 -1 -2 1 2])

    # Test
    @test Ke ≈ Ke_expected
end

@testitem "assembly_local_matrix_DG!: LagrangeElement{1,1}(), LeftRight(), ∂ₛg(x,v) = 1.0" begin
    using WaveAcoustics: assembly_local_matrix_DG!, assembly_local_matrix_ϕxϕ,
                         CartesianMesh, LagrangeElement, DOFMap, LeftRight,
                         QuadratureSetup

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
    num_dof_local = length(quad.ϕP[1])
    DG_local = zeros(Float64, num_dof_local, num_dof_local)
    assembly_local_matrix_DG!(DG_local, ∂ₛg, v, dof_map.m, dof_map.EQoLG[1], quad.xP, quad)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    DG_local_expected = Me * (2 / mesh.Δx[1])

    # Test correctness
    for b in axes(DG_local, 2), a in 1:b # Upper triangle: a ≤ b
        @test DG_local[a, b] ≈ DG_local_expected[a, b]
    end

    # Test allocation-free operation
    m = dof_map.m
    eq = dof_map.EQoLG[1]
    xeP = quad.xP

    alloc = @allocated assembly_local_matrix_DG!(DG_local, ∂ₛg, v, m, eq, xeP, quad)
    @test alloc == 0
end

@testitem "assembly_local_matrix_DG!: LagrangeElement{1,1}(), LeftRight(), ∂ₛg(x,v) = x²+v²" begin
    using WaveAcoustics: assembly_local_matrix_DG!, CartesianMesh, LagrangeElement,
                         DOFMap,
                         LeftRight, QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0,), (1.0,), (4,))
    family = LagrangeElement{1, 1}()
    dof_map = DOFMap(mesh, family, LeftRight())
    Δx = mesh.Δx[1]
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), (Δx, Δx), (0.0, 0.0))

    ∂ₛg(x, s) = x^2 + s^2
    v = ones(Float64, dof_map.m)

    # Expected solution for elements e = 1, 2, 3, 4
    DG_local_expected = (
        [17/240 17/160; 17/160 17/40],    # e=1: [0.00, 0.25], Vₕ=(1+ξ)/2
        [11/15 61/160; 61/160 191/240],   # e=2: [0.25, 0.50], Vₕ=1
        [211/240 223/480; 223/480 59/60], # e=3: [0.50, 0.75], Vₕ=1
        [101/120 57/160; 57/160 157/240]  # e=4: [0.75, 1.00], Vₕ=(1-ξ)/2
    )

    # Compute and test
    num_dof_local = length(quad.ϕP[1])
    DG_local = zeros(Float64, num_dof_local, num_dof_local)

    @testset "Element $e" for e in 1:4
        xeP = quad.xP .+ (e - 1) * Δx
        assembly_local_matrix_DG!(DG_local, ∂ₛg, v, dof_map.m, dof_map.EQoLG[e], xeP, quad)
        @testset "Entry ($a,$b)" for b in axes(DG_local, 2), a in 1:b # Upper triangle: a ≤ b
            @test DG_local[a, b] ≈ DG_local_expected[e][a, b]
        end
    end
end

@testitem "assembly_local_matrix_DF!: LagrangeElement{2,1}(), LeftRightTop(), f(s) = 1.0" begin
    using WaveAcoustics: assembly_local_matrix_DF!, assembly_local_matrix_ϕxϕ,
                         CartesianMesh, LagrangeElement, DOFMap, LeftRightTop,
                         QuadratureSetup

    # Setup
    mesh = CartesianMesh((0.0, 0.0), (1.0, 1.0), (4, 3))
    family = LagrangeElement{2, 1}()
    dof_map = DOFMap(mesh, family, LeftRightTop())
    quad = QuadratureSetup(
        LagrangeElement{1, 1}(), LagrangeElement{2, 1}(), mesh.Δx, mesh.pmin)

    f(s) = 1.0
    d = ones(Float64, dof_map.m)

    # Compute
    num_dof_local = length(quad.φP[1, 1])
    DF_local = zeros(Float64, num_dof_local, num_dof_local)
    assembly_local_matrix_DF!(
        DF_local, f, d, dof_map.m, dof_map.EQoLG[1], quad.W_φPφP, quad.φP)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(mesh, family)
    DF_local_expected = Me * (4 / (mesh.Δx[1] * mesh.Δx[2]))

    # Test
    for b in axes(DF_local, 2), a in 1:b # Upper triangle: a ≤ b
        @test DF_local[a, b] ≈ DF_local_expected[a, b]
    end

    # Test allocation-free operation
    ## Warn: Direct field access (dof_map.m, dof_map.EQoLG[1], quad.W_φPφP, quad.φP) is causing allocations. Why?
    m = dof_map.m
    eq = dof_map.EQoLG[1]
    W_φPφP = quad.W_φPφP
    φP = quad.φP

    alloc = @allocated assembly_local_matrix_DF!(DF_local, f, d, m, eq, W_φPφP, φP)
    @test alloc == 0
end