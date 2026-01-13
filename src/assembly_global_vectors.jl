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