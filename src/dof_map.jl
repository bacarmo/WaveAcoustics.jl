
"""
    DOFMap{T <: AbstractVector, I <: Integer}

Local-to-global DOF mapping with homogeneous Dirichlet BCs enforced on the FE approximation subspace.

# Fields
- `EQoLG::T`: Element connectivity. `EQoLG[e][a]` gives global index of local DOF `a` in element `e`
- `m::I`: Number of free DOFs after homogeneous Dirichlet BC enforcement on the FE approximation subspace

# Indexing Convention
- Global functions in the approximation subspace: indices `1, 2, ..., m`
- Global functions NOT in the approximation subspace: sentinel value `m+1`
"""
struct DOFMap{T <: AbstractVector, I <: Integer}
    EQoLG::T
    m::I
end

"""
    DOFMap(mesh::CartesianMesh, family::FEFamily, sides::DirichletSides)

Construct DOF map for given mesh, element family, and Dirichlet boundary conditions.

# Arguments
- `mesh`: Cartesian mesh
- `family`: Finite element family (e.g., `Lagrange{2,3}()`)
- `sides`: Dirichlet boundary condition specification (e.g., `LeftRight()`)

# Returns
`DOFMap` containing element connectivity and number of free DOFs.

# Examples
```jldoctest
julia> mesh = WaveAcoustics.CartesianMesh((0.0,), (1.0,), (4,));

julia> dofmap = WaveAcoustics.DOFMap(mesh, WaveAcoustics.Lagrange{1,1}(), WaveAcoustics.LeftRight());

julia> dofmap.EQoLG
4-element Vector{StaticArraysCore.SVector{2, Int64}}:
 [4, 1]
 [1, 2]
 [2, 3]
 [3, 4]

julia> dofmap.m
3
```
"""
function DOFMap(mesh::CartesianMesh{Dim, I}, family::FEFamily,
        sides::DirichletSides) where {Dim, I <: Integer}
    LG = build_LG(mesh, family)
    EQ, m = build_EQ(mesh.Nx, family, sides)

    n_local_dof = num_local_dof(family)
    Ne = prod(mesh.Nx)
    EQoLG = Vector{SVector{n_local_dof, I}}(undef, Ne)

    apply_EQ!(EQoLG, LG, EQ)

    return DOFMap(EQoLG, m)
end

# ================================================================
# Internal Functions
# ================================================================

"""
    build_LG(mesh, family)

Build local-to-global DOF map before BC enforcement.

Returns vector `LG` where `LG[e]` contains global DOF indices for element `e`.
Uses tensor product ordering: DOFs numbered left-to-right, bottom-to-top.
"""
function build_LG end

function build_LG(mesh::CartesianMesh{1, I}, ::Lagrange{1, Deg}) where {I <: Integer, Deg}
    Ne = Int(mesh.Nx[1])
    num_local_dof = Deg + 1
    LG = Vector{SVector{num_local_dof, I}}(undef, Ne)

    for e in 1:Ne
        start = (e - 1) * Deg
        LG[e] = SVector{num_local_dof, I}(ntuple(k -> I(start + k), num_local_dof))
    end

    return LG
end

function build_LG(mesh::CartesianMesh{2, I}, ::Lagrange{2, Deg}) where {I <: Integer, Deg}
    Nx, Ny = mesh.Nx
    nx = Deg * Int64(Nx) + 1  # Total DOFs in x-direction
    ny = Deg * Int64(Ny) + 1  # Total DOFs in y-direction

    num_dof = nx * ny
    @assert num_dof≤typemax(I) "Integer type too small: need $num_dof ≤ $(typemax(I))"

    num_local_dof = (Deg + 1)^2
    Ne = Nx * Ny
    LG = Vector{SVector{num_local_dof, I}}(undef, Ne)

    # First element DOFs (left-to-right, bottom-to-top)
    first_element = SVector{num_local_dof, I}(
        (I(i + (j - 1) * nx) for i in 1:(Deg + 1), j in 1:(Deg + 1))...
    )

    # Element-to-element shifts in global DOF numbering
    horizontal_shift = I(Deg)
    vertical_shift = I(Deg * nx)

    # Loop over elements (left to right, bottom to top) 
    for j in 1:Ny
        # Global indices of the first element of the j-th layer in the y-direction
        base_row = first_element .+ vertical_shift * I(j - 1)
        for i in 1:Nx
            e = (j - 1) * Nx + i
            LG[e] = base_row .+ horizontal_shift * I(i - 1)
        end
    end

    return LG
end

function build_LG(mesh::CartesianMesh{1, I}, ::Hermite{1, 3}) where {I <: Integer}
    error("build_LG not yet implemented for Hermite{1,3}")
end

function build_LG(mesh::CartesianMesh{2, I}, ::Hermite{2, 3}) where {I <: Integer}
    error("build_LG not yet implemented for Hermite{2,3}")
end

"""
    build_EQ(Nx, family, sides)

Build equation numbering array that enforces homogeneous Dirichlet BCs.

# Returns
- `EQ`: Array mapping global DOF index (before BCs) to free DOF index (after BCs)
- `m`: Number of free DOFs

# Indexing Convention
- Global functions in the approximation subspace: indices `1, 2, ..., m`
- Global functions NOT in the approximation subspace: sentinel value `m+1`
"""
function build_EQ end

function build_EQ(
        Nx::NTuple{1, I}, ::Lagrange{1, Deg}, ::LeftRight) where {I <: Integer, Deg}
    Ne = Int(Nx[1])
    num_dof = Deg * Ne + 1
    m = convert(I, num_dof - 2)
    sentinel = m + one(I)

    EQ = Vector{I}(undef, num_dof)

    EQ[1] = sentinel
    for i in 2:(num_dof - 1)
        EQ[i] = I(i - 1)
    end
    EQ[num_dof] = sentinel

    return EQ, m
end

function build_EQ(Nx::NTuple{2, I}, ::Lagrange{2, Deg},
        ::LeftRightBottomTop) where {I <: Integer, Deg}
    nx = Deg * Int64(Nx[1]) + 1         # Total DOFs in x-direction
    ny = Deg * Int64(Nx[2]) + 1         # Total DOFs in y-direction
    num_dof = nx * ny                   # Total DOFs

    m = num_dof - 2 * ny - 2 * (nx - 2) # Free DOFs 
    EQ = fill(I(m + 1), num_dof)

    # Re-enumerate interior functions
    for j in 2:(ny - 1)
        cst1 = (j - 1) * nx
        cst2 = (j - 2) * (nx - 2) - 1
        for i in 2:(nx - 1)
            @inbounds EQ[cst1 + i] = I(cst2 + i)
        end
    end

    return EQ, m
end

function build_EQ(Nx::NTuple{2, I}, ::Lagrange{2, Deg},
        ::LeftRightTop) where {I <: Integer, Deg}
    nx = Deg * Int64(Nx[1]) + 1     # Total DOFs in x-direction
    ny = Deg * Int64(Nx[2]) + 1     # Total DOFs in y-direction
    num_dof = nx * ny               # Total DOFs

    m = num_dof - 2 * ny - (nx - 2) # Free DOFs 
    EQ = fill(I(m + 1), num_dof)

    # Re-enumerate interior and bottom boundary functions (excluding corners)
    for j in 1:(ny - 1)
        cst1 = (j - 1) * nx
        cst2 = (j - 1) * (nx - 2) - 1
        for i in 2:(nx - 1)
            @inbounds EQ[cst1 + i] = I(cst2 + i)
        end
    end

    return EQ, m
end

function build_EQ(Nx::NTuple{1, I}, ::Hermite{1, 3}, ::LeftRight) where {I <: Integer}
    error("build_EQ not yet implemented for Hermite{1,3}")
end

function build_EQ(Nx::NTuple{2, I}, ::Hermite{2, 3},
        ::LeftRightBottomTop) where {I <: Integer}
    error("build_EQ not yet implemented for Hermite{2,3} with LeftRightBottomTop")
end

function build_EQ(Nx::NTuple{2, I}, ::Hermite{2, 3},
        ::LeftRightTop) where {I <: Integer}
    error("build_EQ not yet implemented for Hermite{2,3} with LeftRightTop")
end

"""
    apply_EQ!(EQoLG, LG, EQ)

Apply equation numbering to local-to-global map (in-place).

Transforms `LG` (before BCs) into `EQoLG` (after BCs) using `EQ` mapping.
"""
function apply_EQ!(EQoLG::Vector{<:SVector}, LG::Vector{<:SVector},
        EQ::Vector{I}) where {I <: Integer}
    @assert length(EQoLG)==length(LG) "EQoLG and LG length mismatch"

    num_dof_local = length(LG[1])
    for e in eachindex(LG)
        LGe = LG[e]
        EQoLG[e] = SVector{num_dof_local, I}(ntuple(a -> EQ[LGe[a]], num_dof_local))
    end

    return nothing
end