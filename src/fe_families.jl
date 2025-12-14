"""
    FEFamily

Abstract supertype for finite element families.

Concrete subtypes must specify spatial dimension `Dim` and polynomial degree `Deg`.
"""
abstract type FEFamily end

"""
    Lagrange{Dim,Deg} <: FEFamily

Lagrange finite element of dimension `Dim` and degree `Deg`.
"""
struct Lagrange{Dim, Deg} <: FEFamily end

"""
    Hermite{Dim,Deg} <: FEFamily

Hermite finite element of dimension `Dim` and degree `Deg`.
"""
struct Hermite{Dim, Deg} <: FEFamily end

"""
    num_local_dof(family::FEFamily)

Return number of local DOFs for the finite element family.

# Examples
```jldoctest
julia> num_local_dof(Lagrange{2,3}())
16

julia> num_local_dof(Hermite{1,3}())
4
```
"""
function num_local_dof end

num_local_dof(::Lagrange{Dim, Deg}) where {Dim, Deg} = (Deg + 1)^Dim
num_local_dof(::Hermite{1, 3}) = 4
num_local_dof(::Hermite{2, 3}) = 16