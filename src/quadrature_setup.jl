"""
    QuadratureSetup{T<:Real}

Precomputed quadrature data for finite element assembly using Gauss-Legendre quadrature.

## Fields
- `P, W`: Quadrature points and weights in [-1,1].
- `ϕP, W_ϕP, W_ϕPϕP`: 1D basis functions and weighted products.
- `φP, W_φP, W_φPφP`: 2D basis functions and weighted products
- `W_∂φ∂ξP, W_∂φ∂ηP`: Weighted basis function derivatives
- `xP, yP`:  Precomputed fixed part of physical quad points: `(Δx/2)*(P+1) + xmin`, `(Δy/2)*(P+1) + ymin`

## Notes
- Currently specialized for `Npg = 4` Gauss–Legendre points per dimension
- Currently specialized for linear Lagrange basis functions
- For element at indices `(ex, ey)`, physical quadrature coordinates are:
```julia
xeP = @. xP + (ex - 1) * Δx
yeP = @. yP + (ey - 1) * Δy
```
"""
struct QuadratureSetup{T <: Real}
    P::SVector{4, T}
    W::SVector{4, T}
    ϕP::SVector{4, SVector{2, T}}
    W_ϕP::SVector{4, SVector{2, T}}
    W_ϕPϕP::SVector{4, SMatrix{2, 2, T, 4}}
    φP::SMatrix{4, 4, SVector{4, T}, 16}
    W_φP::SMatrix{4, 4, SVector{4, T}, 16}
    W_φPφP::SMatrix{4, 4, SMatrix{4, 4, T, 16}, 16}
    W_∂φ∂ξP::SMatrix{4, 4, SVector{4, T}, 16}
    W_∂φ∂ηP::SMatrix{4, 4, SVector{4, T}, 16}
    xP::SVector{4, T}
    yP::SVector{4, T}
end

"""
    QuadratureSetup(Δx, pmin)

Construct quadrature setup with `4` Gauss-Legendre points per dimension.

## Arguments
- `Δx::NTuple{2,T}`: Element sizes per direction
- `pmin::NTuple{2,T}`: Domain bottom-left corner `(xmin, ymin)`

## Examples
```jldoctest
julia> quad = WaveAcoustics.QuadratureSetup((0.1, 0.2), (0.0, 0.0));

julia> length(quad.xP)
4
```
"""
function QuadratureSetup(Δx::NTuple{2, T}, pmin::NTuple{2, T}) where {T <: Real}
    P_raw, W_raw = legendre(4)
    P = SVector{4, T}(P_raw)
    W = SVector{4, T}(W_raw)

    ϕ(ξ) = basis_functions(Lagrange{1, 1}(), ξ)
    φ(ξ, η) = basis_functions(Lagrange{2, 1}(), ξ, η)

    ϕP = @SVector [ϕ(P[i]) for i in 1:4]
    W_ϕP = @SVector [W[i] * ϕP[i] for i in 1:4]
    W_ϕPϕP = @SVector [@SMatrix [W[j] * ϕP[j][a] * ϕP[j][b] for a in 1:2, b in 1:2]
                       for j in 1:4]

    φP = @SMatrix [φ(P[i], P[j]) for i in 1:4, j in 1:4]
    W_φP = @SMatrix [W[i] * W[j] * φP[i, j] for i in 1:4, j in 1:4]
    W_φPφP = @SMatrix [@SMatrix [W[i] * W[j] * φP[i, j][a] * φP[i, j][b]
                                 for a in 1:4, b in 1:4]
                       for i in 1:4, j in 1:4]

    W_∂φ∂ξP = @SMatrix [begin
                            ∂φ∂ξ, _ = basis_functions_derivatives(
                                Lagrange{2, 1}(), P[i], P[j]
                            )
                            W[i] * W[j] * ∂φ∂ξ
                        end
                        for i in 1:4, j in 1:4]

    W_∂φ∂ηP = @SMatrix [begin
                            _, ∂φ∂η = basis_functions_derivatives(
                                Lagrange{2, 1}(), P[i], P[j]
                            )
                            W[i] * W[j] * ∂φ∂η
                        end
                        for i in 1:4, j in 1:4]

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