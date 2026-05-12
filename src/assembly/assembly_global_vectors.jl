"""
    assembly_rhs_1d!(F, f, scale, W_basisP, mesh, dof_map, xP)

Assemble global RHS vector for 1D FEM by integrating f(x) against basis functions.

# Arguments
- `F::AbstractVector{T}`: Global RHS vector (modified in-place, length `dof_map.m`)
- `f::Fun`: Source function `f(x)` returning type `T`
- `scale::T`: Scaling factor (typically `Δx/2` for F[i] = ∫Ω f(x) ϕᵢ(x) dx)
- `W_basisP::SVector{Npg,SVector{nb,T}}`: Precomputed weighted basis evaluations at quadrature points
- `mesh::CartesianMesh{1,I}`: 1D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `xP::SVector{Npg,T}`: Precomputed fixed part of physical quad points: (Δx/2)*(P+1) + x_start

# Type Parameters
- `T`: Floating point type for computations
- `Npg`: Number of quadrature points per element
- `nb`: Number of local basis functions per element
- `I`: Integer type for mesh indexing
- `Fun`: Function type for source term
"""
function assembly_rhs_1d!(
        F::AbstractVector{T},
        f::Fun,
        scale::T,
        W_basisP::SVector{Npg, SVector{nb, T}},
        mesh::CartesianMesh{1, I},
        dof_map::DOFMap,
        xP::SVector{Npg, T}
) where {T <: Real, Npg, nb, I <: Integer, Fun}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx = mesh.Δx[1]

    for e in eachindex(EQoLG)
        global_indices = EQoLG[e]
        xeP = @. muladd(e - 1, Δx, xP)

        for k in 1:Npg
            fx = f(xeP[k])
            Wₖ_basisPₖ = W_basisP[k]

            for a in 1:nb
                ia = global_indices[a]
                ia > m && continue
                F[ia] = muladd(Wₖ_basisPₖ[a], fx, F[ia])
            end
        end
    end

    lmul!(scale, F)
    return nothing
end

"""
    assembly_rhs_2d!(F, f, scale, W_basisP, mesh, dof_map, xP, yP)

Assemble global RHS vector for 2D FEM by integrating f(x,y) against basis functions.

# Arguments
- `F::AbstractVector{T}`: Global RHS vector (modified in-place, length `dof_map.m`)
- `f::Fun`: Source function `f(x,y)` returning type `T`
- `scale::T`: Scaling factor (typically `Δx*Δy/4` for F[i] = ∫Ω f(x,y) φᵢ(x,y) dx dy)
- `W_basisP::SMatrix{Npg,Npg,SVector{nb,T}}`: Precomputed weighted basis evaluations at quadrature points
- `mesh::CartesianMesh{2,I}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `xP::SVector{Npg,T}`: Precomputed fixed part of physical quad points: (Δx/2)*(P+1) + x_start
- `yP::SVector{Npg,T}`: Precomputed fixed part of physical quad points: (Δy/2)*(P+1) + y_start

# Type Parameters
- `T`: Floating point type for computations
- `Npg`: Number of quadrature points per element
- `nb`: Number of local basis functions per element
- `I`: Integer type for mesh indexing
- `Fun`: Function type for source term
"""
function assembly_rhs_2d!(
        F::AbstractVector{T},
        f::Fun,
        scale::T,
        W_basisP::SMatrix{Npg, Npg, SVector{nb, T}},
        mesh::CartesianMesh{2, I},
        dof_map::DOFMap,
        xP::SVector{Npg, T},
        yP::SVector{Npg, T}
) where {T <: Real, Npg, nb, I <: Integer, Fun}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx, Δy = mesh.Δx
    Nx, Ny = mesh.Nx

    for ey in 1:Ny
        yeP = @. muladd(ey - 1, Δy, yP)

        for ex in 1:Nx
            xeP = @. muladd(ex - 1, Δx, xP)
            e = ex + (ey - 1) * Nx
            global_indices = EQoLG[e]

            Fe = zero(SVector{nb, T})
            for j in 1:Npg, i in 1:Npg
                Fe = muladd(f(xeP[i], yeP[j]), W_basisP[i, j], Fe)
            end

            for a in 1:nb
                ia = global_indices[a]
                ia <= m && (F[ia] += Fe[a])
            end
        end
    end

    lmul!(scale, F)
    return nothing
end

"""
    assembly_nonlinearity_F!(F, scale, f, d, mesh, dof_map, quad)

Fᵢ = scale * ∬ φᵢ(x,y) * f(Uₕ(x,y)) dx dy over Ω, with Uₕ(x,y) = Σ d[j] φⱼ(x,y).

# Arguments
- `F::AbstractVector{T}`: Global vector (zeroed and filled in-place), length `dof_map.m`
- `scale::T`: Scaling factor applied to final result
- `f::Fun`: Nonlinearity function `f(u)` returning type `T`
- `d::AbstractVector{T}`: Coefficient vector for Uₕ, length `dof_map.m`
- `mesh::CartesianMesh{2}`: 2D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function assembly_nonlinearity_F!(
        F::AbstractVector{T},
        scale::T,
        f::Fun,
        d::AbstractVector{T},
        mesh::CartesianMesh{2},
        dof_map::DOFMap,
        quad::QuadratureSetup
) where {T <: AbstractFloat, Fun}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx, Δy = mesh.Δx

    W_φP = quad.W_φP  # SMatrix{Npg,Npg,SVector{num_local_dof,T}}
    φP = quad.φP      # SMatrix{Npg,Npg,SVector{num_local_dof,T}}
    Npg = length(quad.P)
    num_local_dof = length(φP[1, 1])

    for e in eachindex(EQoLG)
        global_indices = EQoLG[e]

        for j in 1:Npg, i in 1:Npg
            uh_at_xy = zero(T)
            φPᵢⱼ = φP[i, j]
            for a in 1:num_local_dof
                ia = global_indices[a]
                ia > m && continue
                uh_at_xy = muladd(d[ia], φPᵢⱼ[a], uh_at_xy)
            end

            fuh = f(uh_at_xy)
            W_φPᵢⱼ = W_φP[i, j]
            for a in 1:num_local_dof
                ia = global_indices[a]
                ia > m && continue
                F[ia] = muladd(W_φPᵢⱼ[a], fuh, F[ia])
            end
        end
    end

    scale_jacobian = scale * (Δx * Δy / 4)
    lmul!(scale_jacobian, F)

    return nothing
end

"""
    assembly_nonlinearity_G!(G, scale, g, v, mesh, dof_map, quad)

Gᵢ = scale * ∫ ϕᵢ(x) * g(x, Vₕ(x)) dx over Ω = (0,1), with Vₕ(x) = Σ v[j] ϕⱼ(x).

# Arguments
- `G::AbstractVector{T}`: Global vector (zeroed and filled in-place), length `dof_map.m`
- `scale::T`: Scaling factor applied to final result
- `g::Fun`: Callable g(x, v) → scalar
- `v::AbstractVector{T}`: Coefficient vector for Vₕ, length `dof_map.m`
- `mesh::CartesianMesh{1}`: 1D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data
"""
function assembly_nonlinearity_G!(
        G::AbstractVector{T},
        scale::T,
        g::Fun,
        v::AbstractVector{T},
        mesh::CartesianMesh{1},
        dof_map::DOFMap,
        quad::QuadratureSetup
) where {T <: AbstractFloat, Fun}
    fill!(G, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx = mesh.Δx[1]

    W_ϕP = quad.W_ϕP  # SVector{Npg,SVector{num_local_dof,T}}
    ϕP = quad.ϕP      # SVector{Npg,SVector{num_local_dof,T}}
    xP = quad.xP
    Npg = length(xP)
    num_local_dof = length(ϕP[1])

    for e in eachindex(EQoLG)
        xeP = @. muladd(e - 1, Δx, xP)
        global_indices = EQoLG[e]

        for j in 1:Npg
            Vₕx = zero(T)
            ϕPⱼ = ϕP[j]
            for a in 1:num_local_dof
                ia = global_indices[a]
                ia > m && continue
                Vₕx = muladd(v[ia], ϕPⱼ[a], Vₕx)
            end

            gxVₕx = g(xeP[j], Vₕx)
            W_ϕPⱼ = W_ϕP[j]
            for a in 1:num_local_dof
                ia = global_indices[a]
                ia > m && continue
                G[ia] = muladd(W_ϕPⱼ[a], gxVₕx, G[ia])
            end
        end
    end

    scale_jacobian = scale * (Δx / 2)
    lmul!(scale_jacobian, G)

    return nothing
end