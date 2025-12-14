@testitem "Lagrange num_local_dof" begin
    using WaveAcoustics
    using WaveAcoustics: num_local_dof

    @test num_local_dof(Lagrange{1, 1}()) == 2
    @test num_local_dof(Lagrange{1, 2}()) == 3
    @test num_local_dof(Lagrange{2, 1}()) == 4
    @test num_local_dof(Lagrange{2, 3}()) == 16
end

@testitem "Hermite num_local_dof" begin
    using WaveAcoustics
    using WaveAcoustics: num_local_dof

    @test num_local_dof(Hermite{1, 3}()) == 4
    @test num_local_dof(Hermite{2, 3}()) == 16
end