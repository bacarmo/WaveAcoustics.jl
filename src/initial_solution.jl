"""
    projection_H01_2d!(uₕ_coefs, ∂ₓu, ∂ᵧu, factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)

Compute the H₀¹ projection of a function onto a finite element subspace given its gradient components.

Solves the weak formulation (∇uₕ, ∇v) = (∇u, ∇v) for all v ∈ Vₕ, where uₕ is the projection and Vₕ is the finite element space.

# Arguments
- `uₕ_coefs::AbstractVector{T}`: Output vector for FEM coefficients of the projection (modified in-place, length dof_map.m)
- `∂ₓu::F1`: Partial derivative ∂u/∂x with signature `∂ₓu(x, y) → T`
- `∂ᵧu::F2`: Partial derivative ∂u/∂y with signature `∂ᵧu(x, y) → T`
- `factorized_lhs_mat::F3`: Factorized matrix (e.g., Cholesky factorization)
- `mesh::CartesianMesh{2}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
- `vec₁::AbstractVector{T}`: Work vector for RHS assembly (modified in-place, length `dof_map.m`)
- `vec₂::AbstractVector{T}`: Work vector for RHS assembly (modified in-place, length `dof_map.m`)
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

"""
    compute_v⁰_d⁰!(v⁰, d⁰, lhs_mat, input_data, mesh, dof_map, quad)

Compute initial velocity and displacement fields via H₀¹ projection from gradient data.

# Arguments
- `v⁰::AbstractVector`: Output vector for initial velocity coefficients (modified in-place, length `dof_map.m`)
- `d⁰::AbstractVector`: Output vector for initial displacement coefficients (modified in-place, length `dof_map.m`)
- `lhs_mat`: Matrix to be factorized
- `input_data`: Structure containing gradient fields `∂ₓv₀`, `∂ᵧv₀`, `∂ₓu₀`, `∂ᵧu₀`
- `mesh::CartesianMesh{2}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function compute_v⁰_d⁰!(v⁰, d⁰, lhs_mat, input_data, mesh, dof_map, quad)
    vec₁ = similar(v⁰)
    vec₂ = similar(v⁰)
    factorized_lhs_mat = cholesky(lhs_mat)

    projection_H01_2d!(v⁰, input_data.common.∂ₓv₀, input_data.common.∂ᵧv₀,
        factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)
    projection_H01_2d!(d⁰, input_data.common.∂ₓu₀, input_data.common.∂ᵧu₀,
        factorized_lhs_mat, mesh, dof_map, quad, vec₁, vec₂)

    return nothing
end

"""
    compute_r⁰_z⁰!(r⁰, z⁰, lhs_mat, input_data, mesh, dof_map, quad)

Compute initial velocity and displacement fields via L2 projection.

# Arguments
- `r⁰::AbstractVector`: Output vector for initial velocity coefficients (modified in-place, length `dof_map.m`)
- `z⁰::AbstractVector`: Output vector for initial displacement coefficients (modified in-place, length `dof_map.m`)
- `lhs_mat`: Matrix to be factorized
- `input_data`: Structure containing fields `r₀`, `z₀`
- `mesh::CartesianMesh{1}`: 1D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function compute_r⁰_z⁰!(r⁰, z⁰, lhs_mat, input_data, mesh, dof_map, quad)
    rhs_vec = similar(r⁰)
    factorized_lhs_mat = cholesky(lhs_mat)

    scale = mesh.Δx[1] / 2
    W_basisP = quad.W_ϕP

    assembly_rhs_1d!(rhs_vec, input_data.common.r₀, scale, W_basisP, mesh, dof_map, quad.xP)
    ldiv!(r⁰, factorized_lhs_mat, rhs_vec)

    assembly_rhs_1d!(rhs_vec, input_data.common.z₀, scale, W_basisP, mesh, dof_map, quad.xP)
    ldiv!(z⁰, factorized_lhs_mat, rhs_vec)

    return nothing
end