"""
    compute_initial_state(
        lhs_wave, lhs_acoustic,
        dof_map_wave, dof_map_acoustic,
        mesh2D, mesh1D, quad, input_data)

Compute the initial [`FEMState`](@ref) for the wave-acoustic coupled problem.

Wave (v⁰, d⁰) fields are obtained via "H₀¹" projection on `mesh2D`;
acoustic fields (r⁰, z⁰) via L² projection on `mesh1D`.
Returns a `FEMState` with `n = 0`, `t = 0`.

# Arguments
- `lhs_wave`: Stiffness matrix for the wave field (m₁ × m₁)
- `lhs_acoustic`: Mass matrix for the acoustic field (m₂ × m₂)
- `dof_map_wave::DOFMap`: DOF mapping for the wave field (m₁ free DOFs)
- `dof_map_acoustic::DOFMap`: DOF mapping for the acoustic field (m₂ free DOFs)
- `mesh2D::CartesianMesh{2}`: 2D mesh for the wave field
- `mesh1D::CartesianMesh{1}`: 1D mesh for the acoustic field
- `quad::QuadratureSetup`: Shared precomputed quadrature data
- `input_data`: Structure with fields `∂ₓv₀`, `∂ᵧv₀`, `∂ₓu₀`, `∂ᵧu₀`, `r₀`, `z₀`
"""
function compute_initial_state(
        lhs_wave, lhs_acoustic,
        dof_map_wave::DOFMap,
        dof_map_acoustic::DOFMap,
        mesh2D::CartesianMesh{2},
        mesh1D::CartesianMesh{1},
        quad::QuadratureSetup,
        input_data
)
    v⁰ = zeros(eltype(lhs_wave), dof_map_wave.m)
    d⁰ = zeros(eltype(lhs_wave), dof_map_wave.m)
    compute_v⁰_d⁰!(v⁰, d⁰, lhs_wave, input_data, mesh2D, dof_map_wave, quad)

    r⁰ = zeros(eltype(lhs_acoustic), dof_map_acoustic.m)
    z⁰ = zeros(eltype(lhs_acoustic), dof_map_acoustic.m)
    compute_r⁰_z⁰!(r⁰, z⁰, lhs_acoustic, input_data, mesh1D, dof_map_acoustic, quad)

    return FEMState(0, zero(eltype(lhs_wave)), v⁰, d⁰, r⁰, z⁰)
end

"""
    compute_v⁰_d⁰!(v⁰, d⁰, lhs_mat, input_data, mesh, dof_map, quad)

Compute the initial wave velocity `v⁰` and displacement `d⁰` via "H₀¹" projection.

# Arguments
- `v⁰::AbstractVector`: Initial velocity coefficients (in-place, length `dof_map.m`)
- `d⁰::AbstractVector`: Initial displacement coefficients (in-place, length `dof_map.m`)
- `lhs_mat`: Stiffness matrix (m₁ × m₁); factorized internally via Cholesky
- `input_data`: Structure with gradient fields `∂ₓv₀`, `∂ᵧv₀`, `∂ₓu₀`, `∂ᵧu₀`
- `mesh::CartesianMesh{2}`: 2D mesh for the wave field
- `dof_map::DOFMap`: DOF mapping with `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function compute_v⁰_d⁰!(v⁰, d⁰, lhs_mat, input_data, mesh, dof_map, quad)
    vec₁ = similar(v⁰)
    vec₂ = similar(v⁰)
    factorized_lhs_mat = cholesky(lhs_mat)

    projection_H01_2d!(v⁰, input_data.∂ₓv₀, input_data.∂ᵧv₀,
        factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)
    projection_H01_2d!(d⁰, input_data.∂ₓu₀, input_data.∂ᵧu₀,
        factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)

    return nothing
end

"""
    compute_r⁰_z⁰!(r⁰, z⁰, lhs_mat, input_data, mesh, dof_map, quad)

Compute the initial acoustic velocity `r⁰` and displacement `z⁰` via L² projection.

# Arguments
- `r⁰::AbstractVector`: Initial acoustic velocity coefficients (in-place, length `dof_map.m`)
- `z⁰::AbstractVector`: Initial acoustic displacement coefficients (in-place, length `dof_map.m`)
- `lhs_mat`: Mass matrix (m₂ × m₂); factorized internally via Cholesky
- `input_data`: Structure with fields `r₀`, `z₀`
- `mesh::CartesianMesh{1}`: 1D mesh for the acoustic field
- `dof_map::DOFMap`: DOF mapping with `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function compute_r⁰_z⁰!(r⁰, z⁰, lhs_mat, input_data, mesh, dof_map, quad)
    rhs_vec = similar(r⁰)
    factorized_lhs_mat = cholesky(lhs_mat)

    scale = mesh.Δx[1] / 2
    W_basisP = quad.W_ϕP

    assembly_rhs_1d!(rhs_vec, input_data.r₀, scale, W_basisP, mesh, dof_map, quad.xP)
    ldiv!(r⁰, factorized_lhs_mat, rhs_vec)

    assembly_rhs_1d!(rhs_vec, input_data.z₀, scale, W_basisP, mesh, dof_map, quad.xP)
    ldiv!(z⁰, factorized_lhs_mat, rhs_vec)

    return nothing
end

"""
    projection_H01_2d!(uₕ_coefs, ∂ₓu, ∂ᵧu, factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)

Compute the H₀¹ projection of a function onto a 2D FE subspace given its gradient.

Solves: find uₕ ∈ Vₕ such that (∇uₕ, ∇v) = (∇u, ∇v) for all v ∈ Vₕ.

# Arguments
- `uₕ_coefs::AbstractVector{T}`: FEM coefficients of the projection (in-place, length `dof_map.m`)
- `∂ₓu::F1`: Function `(x, y) → T` for ∂u/∂x
- `∂ᵧu::F2`: Function `(x, y) → T` for ∂u/∂y
- `factorized_lhs_mat::F3`: Pre-factorized stiffness matrix (e.g., Cholesky)
- `mesh::CartesianMesh{2}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
- `vec₁::AbstractVector{T}`: Work vector (in-place, length `dof_map.m`)
- `vec₂::AbstractVector{T}`: Work vector (in-place, length `dof_map.m`)
"""
function projection_H01_2d!(
        uₕ_coefs::AbstractVector{T},
        ∂ₓu::F1,
        ∂ᵧu::F2,
        factorized_lhs_mat::F3,
        mesh::CartesianMesh{2},
        dof_map::DOFMap,
        quad::QuadratureSetup,
        vec₁::AbstractVector{T},
        vec₂::AbstractVector{T}
) where {T, F1, F2, F3}
    Δx, Δy = mesh.Δx

    assembly_rhs_2d!(vec₁, ∂ₓu, Δy / 2, quad.W_∂φ∂ξP, mesh, dof_map, quad.xP, quad.yP)
    assembly_rhs_2d!(vec₂, ∂ᵧu, Δx / 2, quad.W_∂φ∂ηP, mesh, dof_map, quad.xP, quad.yP)
    @. vec₁ += vec₂

    ldiv!(uₕ_coefs, factorized_lhs_mat, vec₁)

    return nothing
end