# =============================================================================
# 1D Lagrange, Degree 1
# =============================================================================

@testitem "1D Lagrange, Degree 1: functions" begin
    using WaveAcoustics: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{1, 1}(), -1.0) ≈ [1.0, 0.0]
    @test basis_functions(Lagrange{1, 1}(), 1.0) ≈ [0.0, 1.0]

    # Partition of unity: Σϕᵢ = 1
    @test sum(basis_functions(Lagrange{1, 1}(), -0.5)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 1}(), 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 1}(), 0.7)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{1, 1}(), 0.0)
    vec32 = basis_functions(Lagrange{1, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 1: derivatives" begin
    using WaveAcoustics: Lagrange, basis_functions_derivatives

    # Linear basis: derivatives are constant
    @test basis_functions_derivatives(Lagrange{1, 1}(), -1.0) ≈ [-0.5, 0.5]
    @test basis_functions_derivatives(Lagrange{1, 1}(), 1.0) ≈ [-0.5, 0.5]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{1, 1}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{1, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Lagrange, Degree 2
# =============================================================================

@testitem "1D Lagrange, Degree 2: functions" begin
    using WaveAcoustics: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{1, 2}(), -1.0) ≈ [1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{1, 2}(), 0.0) ≈ [0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{1, 2}(), 1.0) ≈ [0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{1, 2}(), -0.7)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 2}(), 0.3)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 2}(), 0.9)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{1, 2}(), 0.0)
    vec32 = basis_functions(Lagrange{1, 2}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 2: derivatives" begin
    using WaveAcoustics: Lagrange, basis_functions_derivatives

    # Node values
    @test basis_functions_derivatives(Lagrange{1, 2}(), -1.0) ≈ [-1.5, 2.0, -0.5]
    @test basis_functions_derivatives(Lagrange{1, 2}(), 0.0) ≈ [-0.5, 0.0, 0.5]
    @test basis_functions_derivatives(Lagrange{1, 2}(), 1.0) ≈ [0.5, -2.0, 1.5]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{1, 2}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{1, 2}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Lagrange, Degree 3
# =============================================================================

@testitem "1D Lagrange, Degree 3: functions" begin
    using WaveAcoustics: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{1, 3}(), -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{1, 3}(), -1.0 / 3.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{1, 3}(), 1.0 / 3.0) ≈ [0.0, 0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{1, 3}(), 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{1, 3}(), -0.8)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 3}(), 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 3}(), 0.6)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{1, 3}(), 0.0)
    vec32 = basis_functions(Lagrange{1, 3}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 3: derivatives" begin
    using WaveAcoustics: Lagrange, basis_functions_derivatives

    # Node values
    @test basis_functions_derivatives(Lagrange{1, 3}(), -1.0) ≈
          [-11 / 4, 9 / 2, -9 / 4, 1 / 2]
    @test basis_functions_derivatives(Lagrange{1, 3}(), -1.0 / 3.0) ≈
          [-1 / 2, -3 / 4, 3 / 2, -1 / 4]
    @test basis_functions_derivatives(Lagrange{1, 3}(), 1.0 / 3.0) ≈
          [1 / 4, -3 / 2, 3 / 4, 1 / 2]
    @test basis_functions_derivatives(Lagrange{1, 3}(), 1.0) ≈
          [-1 / 2, 9 / 4, -9 / 2, 11 / 4]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{1, 3}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{1, 3}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 2D Lagrange, Degree 1
# =============================================================================

@testitem "2D Lagrange, Degree 1: functions" begin
    using WaveAcoustics: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    # 1:(−1,−1), 2:(+1,−1), 3:(−1,+1), 4:(+1,+1)
    @test basis_functions(Lagrange{2, 1}(), -1.0, -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 1}(), 1.0, -1.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 1}(), -1.0, 1.0) ≈ [0.0, 0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{2, 1}(), 1.0, 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{2, 1}(), -0.3, -0.7)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 1}(), 0.0, 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 1}(), 0.5, 0.8)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{2, 1}(), 0.0, 0.0)
    vec32 = basis_functions(Lagrange{2, 1}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "2D Lagrange, Degree 1: derivatives" begin
    using WaveAcoustics: Lagrange, basis_functions_derivatives

    # Derivatives at center (0,0)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{2, 1}(), 0.0, 0.0)
    @test ∂ϕ_∂ξ ≈ [-0.25, 0.25, -0.25, 0.25]
    @test ∂ϕ_∂η ≈ [-0.25, -0.25, 0.25, 0.25]

    # Test type
    vec64_ξ, vec64_η = basis_functions_derivatives(Lagrange{2, 1}(), 0.0, 0.0)
    vec32_ξ, vec32_η = basis_functions_derivatives(
        Lagrange{2, 1}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64_ξ) == Float64
    @test eltype(vec64_η) == Float64
    @test eltype(vec32_ξ) == Float32
    @test eltype(vec32_η) == Float32
end

# =============================================================================
# 2D Lagrange, Degree 2
# =============================================================================

@testitem "2D Lagrange, Degree 2: functions" begin
    using WaveAcoustics: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    # 1th layer: 1:(−1,-1), 2:(0,-1), 3:(+1,-1)
    @test basis_functions(Lagrange{2, 2}(), -1.0, -1.0) ≈
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 0.0, -1.0) ≈
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 1.0, -1.0) ≈
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 2th layer: 1:(−1,0), 2:(0,0), 3:(+1,0)
    @test basis_functions(Lagrange{2, 2}(), -1.0, 0.0) ≈
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 0.0, 0.0) ≈
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 1.0, 0.0) ≈
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # 3th layer: 1:(−1,+1), 2:(0,+1), 3:(+1,+1)
    @test basis_functions(Lagrange{2, 2}(), -1.0, 1.0) ≈
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 0.0, 1.0) ≈
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{2, 2}(), 1.0, 1.0) ≈
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{2, 2}(), -0.5, -0.3)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 2}(), 0.7, 0.4)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{2, 2}(), 0.0, 0.0)
    vec32 = basis_functions(Lagrange{2, 2}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Hermite, Degree 3
# =============================================================================
@testitem "1D Hermite, Degree 3: functions" begin
    using WaveAcoustics: Hermite, basis_functions

    # Node values: [H₁(ξ), H₁'(ξ), H₂(ξ), H₂'(ξ)]
    @test basis_functions(Hermite{1, 3}(), -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Hermite{1, 3}(), 1.0) ≈ [0.0, 0.0, 1.0, 0.0]

    # Test type
    vec64 = basis_functions(Hermite{1, 3}(), 0.0)
    vec32 = basis_functions(Hermite{1, 3}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Hermite, Degree 3: derivatives" begin
    using WaveAcoustics: Hermite, basis_functions_derivatives

    # Node values: [dH₁/dξ, dH₁'/dξ, dH₂/dξ, dH₂'/dξ]
    @test basis_functions_derivatives(Hermite{1, 3}(), -1.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions_derivatives(Hermite{1, 3}(), 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Test type
    vec64_ξ, vec64_η = basis_functions_derivatives(Hermite{1, 3}(), 0.0)
    vec32_ξ, vec32_η = basis_functions_derivatives(Hermite{1, 3}(), Float32(0.0))
    @test eltype(vec64_ξ) == Float64
    @test eltype(vec64_η) == Float64
    @test eltype(vec32_ξ) == Float32
    @test eltype(vec32_η) == Float32
end
