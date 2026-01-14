"""
    L2_error_1d(u, uₕ_coefs, mesh, dof_map, quad)

Compute the L2 error norm ||u - uₕ||_L2 between exact and FEM solutions.

# Arguments
- `u::F`: Exact solution function with signature `u(x) → T`
- `uₕ_coefs::AbstractVector{T}`: DOF coefficients of the FEM solution (length = `dof_map.m`)
- `mesh::CartesianMesh{1}`: 1D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data

# Returns
- `T`: L2 norm ||u - uₕ||_L2
"""
function L2_error_1d(
        u::Fun,
        uₕ_coefs::AbstractVector{T},
        mesh::CartesianMesh{1},
        dof_map::DOFMap,
        quad::QuadratureSetup
) where {T <: Real, Fun}
    EQoLG = dof_map.EQoLG
    m = dof_map.m

    Δx = mesh.Δx[1]
    Nx = mesh.Nx[1]

    xP = quad.xP
    ϕP = quad.ϕP
    W = quad.W
    Npg = length(xP)
    num_local_dof = length(ϕP[1])

    result = zero(T)

    for e in 1:Nx
        xeP = @. muladd(e - 1, Δx, xP)
        global_indices = EQoLG[e]

        for j in 1:Npg
            ϕPⱼ = ϕP[j]
            uh_at_x = zero(T)

            for a in 1:num_local_dof
                ia = global_indices[a]
                ia > m && continue
                uh_at_x = muladd(uₕ_coefs[ia], ϕPⱼ[a], uh_at_x)
            end

            err = u(xeP[j]) - uh_at_x
            result += W[j] * err^2
        end
    end

    return sqrt(result * Δx / 2)
end

"""
    L2_error_2d(u, uₕ_coefs, mesh, dof_map, quad)

Compute the L2 error norm ||u - uₕ||_L2 between exact and FEM solutions.

# Arguments
- `u::F`: Exact solution function with signature `u(x, y) → T`
- `uₕ_coefs::AbstractVector{T}`: DOF coefficients of the FEM solution (length = `dof_map.m`)
- `mesh::CartesianMesh{2}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data

# Returns
- `T`: L2 norm ||u - uₕ||_L2
"""
function L2_error_2d(
        u::Fun,
        uₕ_coefs::AbstractVector{T},
        mesh::CartesianMesh{2},
        dof_map::DOFMap,
        quad::QuadratureSetup
) where {T <: Real, Fun}
    EQoLG = dof_map.EQoLG
    m = dof_map.m

    Δx, Δy = mesh.Δx
    Nx, Ny = mesh.Nx

    xP = quad.xP
    yP = quad.yP
    φP = quad.φP
    W = quad.W
    Npg = length(xP)
    num_local_dof = length(φP[1, 1])

    result = zero(T)

    for ey in 1:Ny
        yeP = @. muladd(ey - 1, Δy, yP)

        for ex in 1:Nx
            xeP = @. muladd(ex - 1, Δx, xP)
            e = ex + (ey - 1) * Nx
            global_indices = EQoLG[e]

            for j in 1:Npg, i in 1:Npg
                φ_ij = φP[i, j]
                uh_at_xy = zero(T)

                for a in 1:num_local_dof
                    ia = global_indices[a]
                    ia > m && continue
                    uh_at_xy = muladd(uₕ_coefs[ia], φ_ij[a], uh_at_xy)
                end

                err = u(xeP[i], yeP[j]) - uh_at_xy
                result += W[i] * W[j] * err^2
            end
        end
    end

    return sqrt(result * Δx * Δy / 4)
end