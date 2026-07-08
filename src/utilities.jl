# ==============================================================================
# SystemMatrices
# ==============================================================================

"""
    SystemMatrices{T, I}

Global FEM matrices for the PDE system.

# Fields
- `_2M_m₁xm₁`: mass matrix for the ``m_1``-space.
- `M_m₂xm₂`: mass matrix for the ``m_2``-space.
- `K_m₁xm₁`: stiffness matrix for the ``m_1``-space.`.
"""
struct SystemMatrices{T <: AbstractFloat, I <: Integer}
    _2M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}}
    M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}}
    K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}}
end

"""
    SystemMatrices(fe2D, element_side_lengths, dof_map_m₁, dof_map_m₂, a, τ)

Construct the global FEM matrices for the PDE system.

# Arguments
- `fe1D`: FE basis with polynomial degree `Deg` and spatial dimension `1`
- `fe2D`: FE basis with polynomial degree `Deg` and spatial dimension `2`
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension.
- `dof_map_m₁`: DOF map for the ``m_1``-space (left, right, and top boundaries prescribed).
- `dof_map_m₂`: DOF map for the ``m_2``-space (all boundaries prescribed).
"""
function SystemMatrices(
        fe1D::AbstractFEBasis{Deg, 1},
        fe2D::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T},
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap
) where {T, Deg}
    Δx, Δy = element_side_lengths

    # Local matrices — 1D
    Me_m₂xm₂ = Symmetric(assembly_local_matrix_ϕxϕ(fe1D, (Δx,)))

    # Local matrices, 2D
    _2Me = 2 * Symmetric(assembly_local_matrix_ϕxϕ(fe2D, element_side_lengths))
    Ke = Symmetric(assembly_local_matrix_∇ϕx∇ϕ(fe2D, element_side_lengths))

    return SystemMatrices(
        # Mass matrices
        assembly_global_matrix(_2Me, dof_map_m₁),    # 2*M_m₁xm₁
        assembly_global_matrix(Me_m₂xm₂, dof_map_m₂),# M_m₂xm₂
        # Stiffness matrices
        assembly_global_matrix(Ke, dof_map_m₁)       # K_m₁xm₁
    )
end

# ==============================================================================
# QuadratureSetup
# ==============================================================================

"""
    QuadratureSetup{T, Npg, Npg², nb, nb², nb⁴}

Precomputed quadrature data for finite element assembly using Gauss-Legendre quadrature.

## Type Parameters
- `T`: Floating-point type
- `Npg`: Number of Gauss-Legendre quadrature points per dimension
- `Npg²`: `Npg^2`, total number of 2D quadrature points
- `nb`: Number of 1D basis functions
- `nb²`: `nb^2`, number of 2D basis functions
- `nb⁴`: `nb^4`, the number of entries in a `nb² × nb²` matrix

## Fields
- `P::SVector{Npg}`: Quadrature points in [-1,1]
- `W::SVector{Npg}`: Quadrature weights for the interval [-1,1]
- `ϕP::SVector{Npg, SVector{nb}}`: 1D basis functions evaluated at quadrature points; `ϕP[i] = ϕ(P[i])`
- `W_ϕP::SVector{Npg, SVector{nb}}`: `W_ϕP[i] = W[i] * ϕP[i]`
- `W_ϕPϕP::SVector{Npg, SMatrix{nb, nb}}`: `W_ϕPϕP[j][a,b] = W[j] * ϕP[j][a] * ϕP[j][b]`
- `φP::SMatrix{Npg, Npg, SVector{nb²}}`: 2D basis functions at quadrature points; `φP[i,j] = φ(P[i], P[j])`
- `W_φP::SMatrix{Npg, Npg, SVector{nb²}}`: `W_φP[i,j] = W[i]*W[j] * φP[i,j]`
- `W_φPφP::SMatrix{Npg, Npg, SMatrix{nb², nb²}}`: `W_φPφP[i,j][a,b] = W[i]*W[j] * φP[i,j][a] * φP[i,j][b]`
- `W_∂φ∂ξP::SMatrix{Npg, Npg, SVector{nb²}}`: `W_∂φ∂ξP[i,j] = W[i]*W[j] * ∂φ/∂ξ(P[i],P[j])`
- `W_∂φ∂ηP::SMatrix{Npg, Npg, SVector{nb²}}`: `W_∂φ∂ηP[i,j] = W[i]*W[j] * ∂φ/∂η(P[i],P[j])`
- `xP::SVector{Npg}`: Fixed part of physical x-coordinates; `xP = (Δx/2)*(P+1) + xmin`
- `yP::SVector{Npg}`: Fixed part of physical y-coordinates; `yP = (Δy/2)*(P+1) + ymin`

## Notes
- For element at indices `(ex, ey)`, physical quadrature coordinates are:
```julia
xeP = @. xP + (ex - 1) * Δx
yeP = @. yP + (ey - 1) * Δy
```
"""
struct QuadratureSetup{T <: AbstractFloat, Npg, Npg², nb, nb², nb⁴}
    P::SVector{Npg, T}
    W::SVector{Npg, T}
    ϕP::SVector{Npg, SVector{nb, T}}
    W_ϕP::SVector{Npg, SVector{nb, T}}
    W_ϕPϕP::SVector{Npg, SMatrix{nb, nb, T, nb²}}
    φP::SMatrix{Npg, Npg, SVector{nb², T}, Npg²}
    W_φP::SMatrix{Npg, Npg, SVector{nb², T}, Npg²}
    W_φPφP::SMatrix{Npg, Npg, SMatrix{nb², nb², T, nb⁴}, Npg²}
    W_∂φ∂ξP::SMatrix{Npg, Npg, SVector{nb², T}, Npg²}
    W_∂φ∂ηP::SMatrix{Npg, Npg, SVector{nb², T}, Npg²}
    xP::SVector{Npg, T}
    yP::SVector{Npg, T}
end

# Return the number of local degrees of freedom for the finite element `fe`.
num_local_dof(::Lagrange{Deg, Dim}) where {Dim, Deg} = Val((Deg + 1)^Dim)

"""
    QuadratureSetup(fe1D, fe2D, element_side_lengths, pmin)

Construct a `QuadratureSetup` for given 1D and 2D finite element basis.
"""
function QuadratureSetup(
        fe1D::AbstractFEBasis{Deg, 1},
        fe2D::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T},
        pmin::NTuple{2, T},
        ::Val{nb} = num_local_dof(fe1D),
        ::Val{nb²} = num_local_dof(fe2D)
) where {Deg, T, nb, nb²}
    Δx, Δy = element_side_lengths
    Npg = Deg + 4
    P_raw, W_raw = legendre(T, Npg)
    P = SVector{Npg, T}(P_raw)
    W = SVector{Npg, T}(W_raw)

    ϕP = SVector{Npg}([basis_functions(fe1D, P[i]) for i in 1:Npg])
    W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])
    W_ϕPϕP = SVector{Npg}([SMatrix{nb, nb, T}(W[j] * ϕP[j][a] * ϕP[j][b]
                           for a in 1:nb, b in 1:nb)
                           for j in 1:Npg])

    φP = SMatrix{Npg, Npg}([basis_functions(fe2D, P[i], P[j]) for i in 1:Npg, j in 1:Npg])
    W_φP = SMatrix{Npg, Npg}([W[i] * W[j] * φP[i, j] for i in 1:Npg, j in 1:Npg])
    W_φPφP = SMatrix{Npg, Npg}([SMatrix{nb², nb², T}(
                                    W[i] * W[j] * φP[i, j][a] * φP[i, j][b]
                                for a in 1:nb², b in 1:nb²)
                                for i in 1:Npg, j in 1:Npg])

    ∂φP = SMatrix{Npg, Npg}([basis_functions_derivatives(fe2D, P[i], P[j])
                             for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ξP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][1] for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ηP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][2] for i in 1:Npg, j in 1:Npg])

    xP = (Δx / 2) .* (P .+ one(T)) .+ pmin[1]
    yP = (Δy / 2) .* (P .+ one(T)) .+ pmin[2]

    return QuadratureSetup(
        P, W,
        ϕP, W_ϕP, W_ϕPϕP,
        φP, W_φP, W_φPφP,
        W_∂φ∂ξP, W_∂φ∂ηP,
        xP, yP
    )
end
