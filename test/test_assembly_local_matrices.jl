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