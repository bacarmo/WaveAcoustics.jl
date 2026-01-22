"""
    assembly_local_matrix_ϕxϕ(mesh, family)

Assemble the local mass matrix ∫_{Ωₑ} ϕₐᵉ ϕᵦᵉ dx for an arbitrary element Ωₑ.

# Arguments
- `mesh::CartesianMesh{Dim}`: Uniform Cartesian mesh defining element geometry
- `family::FEFamily`: Finite element family

# Returns
- `SMatrix{N,N}`: Local mass matrix where `N = num_local_dof(family)`
"""
function assembly_local_matrix_ϕxϕ end

function assembly_local_matrix_ϕxϕ(mesh::CartesianMesh{1}, family::FEFamily)
    Δx = mesh.Δx[1]
    deg = polynomial_degree(family)
    Npg = deg + 1
    P, W = legendre(Npg)

    N = num_local_dof(family)
    M = zeros(N, N)
    jac = Δx / 2

    for i in 1:Npg
        ϕ = basis_functions(family, P[i])
        for b in 1:N, a in 1:N
            M[a, b] += W[i] * ϕ[a] * ϕ[b] * jac
        end
    end

    return SMatrix{N, N}(M)
end

function assembly_local_matrix_ϕxϕ(mesh::CartesianMesh{2}, family::FEFamily)
    Δx, Δy = mesh.Δx
    deg = polynomial_degree(family)
    Npg = deg + 1
    P, W = legendre(Npg)

    N = num_local_dof(family)
    M = zeros(N, N)
    jac = Δx * Δy / 4

    for i in 1:Npg, j in 1:Npg
        φ = basis_functions(family, P[i], P[j])
        w = W[i] * W[j]
        for b in 1:N, a in 1:N
            M[a, b] += w * φ[a] * φ[b] * jac
        end
    end

    return SMatrix{N, N}(M)
end

"""
    assembly_local_matrix_∇ϕx∇ϕ(mesh, family)

Assemble the local stiffness matrix ∫_{Ωₑ} ∇ϕₐᵉ · ∇ϕᵦᵉ dx for an arbitrary element Ωₑ.

# Arguments
- `mesh::CartesianMesh{Dim}`: Uniform Cartesian mesh defining element geometry
- `family::FEFamily`: Finite element family

# Returns
- `SMatrix{N,N}`: Local stiffness matrix where `N = num_local_dof(family)`
"""
function assembly_local_matrix_∇ϕx∇ϕ end

function assembly_local_matrix_∇ϕx∇ϕ(mesh::CartesianMesh{1}, family::FEFamily)
    Δx = mesh.Δx[1]
    deg = polynomial_degree(family)
    Npg = deg
    P, W = legendre(Npg)

    N = num_local_dof(family)
    K = zeros(N, N)
    scale = 2 / Δx  # (2 / Δx)^2 * (Δx / 2) 

    for i in 1:Npg
        dϕ = basis_functions_derivatives(family, P[i])
        for b in 1:N, a in 1:N
            K[a, b] += W[i] * dϕ[a] * dϕ[b] * scale
        end
    end

    return SMatrix{N, N}(K)
end

function assembly_local_matrix_∇ϕx∇ϕ(mesh::CartesianMesh{2}, family::FEFamily)
    Δx, Δy = mesh.Δx
    deg = polynomial_degree(family)
    Npg = deg + 1
    P, W = legendre(Npg)

    N = num_local_dof(family)
    K = zeros(N, N)
    scale_x = Δy / Δx   # (2 / Δx)^2 * (Δx * Δy / 4))
    scale_y = Δx / Δy   # (2 / Δy)^2 * (Δx * Δy / 4))

    for i in 1:Npg, j in 1:Npg
        ∂φ∂ξ, ∂φ∂η = basis_functions_derivatives(family, P[i], P[j])
        w = W[i] * W[j]
        for b in 1:N, a in 1:N
            K[a, b] += w * (∂φ∂ξ[a] * ∂φ∂ξ[b] * scale_x +
                            ∂φ∂η[a] * ∂φ∂η[b] * scale_y)
        end
    end

    return SMatrix{N, N}(K)
end

"""
    assembly_local_matrix_DG!(DG, ∂ₛg, v, m, eq, xeP, W_ϕPϕP, ϕP)

DGₐᵦ = ∫ ϕₐ(ξ) * ϕᵦ(ξ) * ∂ₛg(x(ξ), Vₕ(x(ξ))) dx over Ω = (-1,1), with Vₕ(x(ξ)) = Σ v[eq[j]] ϕⱼ(ξ).

# Arguments
- `DG::AbstractMatrix{T}`: Local matrix (nb × nb), zeroed and filled in-place **only in upper triangle**
- `∂ₛg::Fun`: Callable ∂ₛg(x, s) → T
- `v::AbstractVector{T}`: Coefficient vector for Vₕ, length `m`
- `m::I`: Number of active DOFs
- `eq::SVector{nb,I}`: Local-to-global DOF mapping for element `e` (EQoLG[e])
- `xeP::SVector{Npg,T}`: Physical quadrature points (`xᵉ(P) = (Δx/2)*(P + 1) + x_start + (e-1)*Δx`)
- `quad::QuadratureSetup`: Precomputed quadrature data

# Type Parameters
- `T`: Floating point type
- `I`: Integer type
- `nb`: Number of basis functions per element
- `Npg`: Number of quadrature points per element
- `Fun`: Function type

# Notes
- Scaling factor and Jacobian are NOT applied here
"""
function assembly_local_matrix_DG!(
        DG::AbstractMatrix{T},
        ∂ₛg::Fun,
        v::AbstractVector{T},
        m::I,
        eq::SVector{nb, I},
        xeP::SVector{Npg, T},
        quad::QuadratureSetup
) where {T <: AbstractFloat, I <: Integer, nb, Npg, Fun}
    fill!(DG, zero(T))

    for j in 1:Npg
        # Compute Vₕ(x) at quadrature point j
        Vₕx = zero(T)
        ϕPⱼ = quad.ϕP[j]
        for a in 1:nb
            ia = eq[a]
            ia > m && continue
            Vₕx = muladd(v[ia], ϕPⱼ[a], Vₕx)
        end

        # Evaluate ∂ₛg at current point
        g_val = ∂ₛg(xeP[j], Vₕx)

        # Accumulate contributions to local matrix
        W_ϕϕ = quad.W_ϕPϕP[j]
        @inbounds for b in 1:nb, a in 1:b  # Upper triangle: a ≤ b
            DG[a, b] = muladd(W_ϕϕ[a, b], g_val, DG[a, b])
        end
    end

    return nothing
end