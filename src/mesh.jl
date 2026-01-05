"""
    AbstractMesh{Dim}

Abstract supertype for meshes in dimension `Dim`.
"""
abstract type AbstractMesh{Dim} end

"""
    CartesianMesh{Dim,I} <: AbstractMesh{Dim}

Uniform Cartesian mesh in dimension `Dim` with integer type `I`.

# Fields
- `pmin::NTuple{Dim,Float64}`: Domain lower corner
- `pmax::NTuple{Dim,Float64}`: Domain upper corner
- `Nx::NTuple{Dim,I}`: Number of elements per direction
- `Δx::NTuple{Dim,Float64}`: Element sizes per direction
"""
struct CartesianMesh{Dim, I} <: AbstractMesh{Dim}
    pmin::NTuple{Dim, Float64}
    pmax::NTuple{Dim, Float64}
    Nx::NTuple{Dim, I}
    Δx::NTuple{Dim, Float64}
end

"""
    CartesianMesh(pmin, pmax, Nx)

Construct uniform Cartesian mesh partitioning `[pmin, pmax]` (1D) or 
`[pmin[1], pmax[1]] × [pmin[2], pmax[2]]` (2D) into `Nx` elements per direction.

# Arguments
- `pmin::NTuple{Dim,Float64}`: Domain lower corner
- `pmax::NTuple{Dim,Float64}`: Domain upper corner
- `Nx::NTuple{Dim,I}`: Number of elements per direction

# Examples
```jldoctest
julia> mesh1d = WaveAcoustics.CartesianMesh((0.0,), (1.0,), (10,));

julia> mesh1d.Δx
(0.1,)

julia> mesh2d = WaveAcoustics.CartesianMesh((0.0, 0.0), (1.0, 2.0), (10, 20));

julia> mesh2d.Δx
(0.1, 0.1)
```
"""
function CartesianMesh(pmin::NTuple{Dim, Float64}, pmax::NTuple{Dim, Float64},
        Nx::NTuple{Dim, I}) where {Dim, I <: Integer}
    Δx = ntuple(d -> (pmax[d] - pmin[d]) / Nx[d], Dim)
    return CartesianMesh{Dim, I}(pmin, pmax, Nx, Δx)
end