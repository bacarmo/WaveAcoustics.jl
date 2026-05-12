# ========================================
# QuadratureSetup
# ========================================

"""
    QuadratureSetup{Npg, Npg2, Nb, Nb2, Nb4, T}

Precomputed quadrature data for finite element assembly using Gauss-Legendre quadrature.

## Type Parameters
- `Npg`: Number of Gauss–Legendre quadrature points per dimension
- `Npg2`: `Npg^2` —  total number of 2D quadrature points
- `Nb`: Number of 1D basis functions
- `Nb2`: `Nb^2` — number of 2D basis functions
- `Nb4`: `Nb^4` — the number of entries in a `Nb2×Nb2` matrix
- `T`: Floating-point type

## Fields
- `P::SVector{Npg}`: Quadrature points in [-1,1]
- `W::SVector{Npg}`: Quadrature weights for the interval [-1,1]
- `ϕP::SVector{Npg, SVector{Nb}}`: 1D basis functions evaluated at quadrature points; `ϕP[i] = ϕ(P[i])`
- `W_ϕP::SVector{Npg, SVector{Nb}}`: `W_ϕP[i] = W[i] * ϕP[i]`
- `W_ϕPϕP::SVector{Npg, SMatrix{Nb,Nb}}`: `W_ϕPϕP[j][a,b] = W[j] * ϕP[j][a] * ϕP[j][b]`
- `φP::SMatrix{Npg,Npg, SVector{Nb2}}`: 2D basis functions at quadrature points; `φP[i,j] = φ(P[i], P[j])`
- `W_φP::SMatrix{Npg,Npg, SVector{Nb2}}`: `W_φP[i,j] = W[i]*W[j] * φP[i,j]`
- `W_φPφP::SMatrix{Npg,Npg, SMatrix{Nb2,Nb2}}`: `W_φPφP[i,j][a,b] = W[i]*W[j] * φP[i,j][a] * φP[i,j][b]`
- `W_∂φ∂ξP::SMatrix{Npg,Npg, SVector{Nb2}}`: `W_∂φ∂ξP[i,j] = W[i]*W[j] * ∂φ/∂ξ(P[i],P[j])`
- `W_∂φ∂ηP::SMatrix{Npg,Npg, SVector{Nb2}}`: `W_∂φ∂ηP[i,j] = W[i]*W[j] * ∂φ/∂η(P[i],P[j])`
- `xP::SVector{Npg}`: Fixed part of physical x-coordinates; `xP = (Δx/2)*(P+1) + xmin`
- `yP::SVector{Npg}`: Fixed part of physical y-coordinates; `yP = (Δy/2)*(P+1) + ymin`

## Notes
- For element at indices `(ex, ey)`, physical quadrature coordinates are:
```julia
xeP = @. xP + (ex - 1) * Δx
yeP = @. yP + (ey - 1) * Δy
```
"""
struct QuadratureSetup{Npg, Npg2, Nb, Nb2, Nb4, T <: AbstractFloat}
    P::SVector{Npg, T}
    W::SVector{Npg, T}
    ϕP::SVector{Npg, SVector{Nb, T}}
    W_ϕP::SVector{Npg, SVector{Nb, T}}
    W_ϕPϕP::SVector{Npg, SMatrix{Nb, Nb, T, Nb2}}
    φP::SMatrix{Npg, Npg, SVector{Nb2, T}, Npg2}
    W_φP::SMatrix{Npg, Npg, SVector{Nb2, T}, Npg2}
    W_φPφP::SMatrix{Npg, Npg, SMatrix{Nb2, Nb2, T, Nb4}, Npg2}
    W_∂φ∂ξP::SMatrix{Npg, Npg, SVector{Nb2, T}, Npg2}
    W_∂φ∂ηP::SMatrix{Npg, Npg, SVector{Nb2, T}, Npg2}
    xP::SVector{Npg, T}
    yP::SVector{Npg, T}
end

"""
    QuadratureSetup(fe1D, fe2D, Δx, pmin, [Val(Npg)])

Construct a `QuadratureSetup` for given 1D and 2D finite element families.

## Arguments
- `fe1D::DimensionalFEFamily`: 1D element family (e.g. `LagrangeElement{1,1}()`)
- `fe2D::DimensionalFEFamily`: 2D element family (e.g. `LagrangeElement{2,1}()`)
- `Δx::NTuple{2,T}`: Element sizes `(Δx, Δy)`
- `pmin::NTuple{2,T}`: Bottom-left corner of the domain `(xmin, ymin)`
- `::Val{Npg}`: Number of Gauss–Legendre points per dimension (default: `Val(4)`)
- `::Val{Nb}`: Number of local DOFs per element in 1D; inferred automatically from `fe1D`. Do not pass explicitly.
- `::Val{Nb2}`: Number of local DOFs per element in 2D; inferred automatically from `fe2D`. Do not pass explicitly.

## Examples
```jldoctest
julia> using WaveAcoustics: QuadratureSetup, LagrangeElement

julia> quad = QuadratureSetup(LagrangeElement{1,1}(), LagrangeElement{2,1}(), (0.1, 0.2), (0.0, 0.0));

julia> length(quad.xP)
4
```
"""
function QuadratureSetup(
        fe1D::DimensionalFEFamily,
        fe2D::DimensionalFEFamily,
        Δx::NTuple{2, T},
        pmin::NTuple{2, T},
        ::Val{Npg} = Val(4),
        ::Val{Nb} = num_local_dof_static(fe1D),
        ::Val{Nb2} = num_local_dof_static(fe2D)
) where {T <: Real, Npg, Nb, Nb2}
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg, T}(P_raw)
    W = SVector{Npg, T}(W_raw)

    Nb^2 == Nb2 || throw(ArgumentError(
        "fe2D must be the Cartesian product of fe1D: expected Nb2=$(Nb^2), got $Nb2"
    ))

    ϕP = SVector{Npg}([basis_functions(fe1D, P[i]) for i in 1:Npg])
    W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])
    W_ϕPϕP = SVector{Npg}([SMatrix{Nb, Nb, T}(W[j] * ϕP[j][a] * ϕP[j][b]
                           for a in 1:Nb, b in 1:Nb)
                           for j in 1:Npg])

    φP = SMatrix{Npg, Npg}([basis_functions(fe2D, P[i], P[j]) for i in 1:Npg, j in 1:Npg])
    W_φP = SMatrix{Npg, Npg}([W[i] * W[j] * φP[i, j] for i in 1:Npg, j in 1:Npg])
    W_φPφP = SMatrix{Npg, Npg}([SMatrix{Nb2, Nb2, T}(
                                    W[i] * W[j] * φP[i, j][a] * φP[i, j][b]
                                for a in 1:Nb2, b in 1:Nb2)
                                for i in 1:Npg, j in 1:Npg])

    ∂φP = SMatrix{Npg, Npg}([basis_functions_derivatives(fe2D, P[i], P[j])
                             for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ξP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][1] for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ηP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][2] for i in 1:Npg, j in 1:Npg])

    xP = (Δx[1] / 2) .* (P .+ one(T)) .+ pmin[1]
    yP = (Δx[2] / 2) .* (P .+ one(T)) .+ pmin[2]

    return QuadratureSetup(
        P, W,
        ϕP, W_ϕP, W_ϕPϕP,
        φP, W_φP, W_φPφP,
        W_∂φ∂ξP, W_∂φ∂ηP,
        xP, yP
    )
end