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