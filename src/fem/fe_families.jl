# ========================================
# Abstract types
# ========================================

"""
    FEFamily

Abstract type for finite element families (dimension-agnostic).
Use `specialize` function to obtain a dimension-aware element.

Concrete subtypes: [`Lagrange{Deg}`](@ref), [`Hermite{Deg}`](@ref).
"""
abstract type FEFamily end

"""
    DimensionalFEFamily <: FEFamily

Abstract type for dimension-aware finite element families.
Constructed via `specialize` function. Not part of the public API.

Concrete subtypes: `LagrangeElement{Dim,Deg}`, `HermiteElement{Dim,Deg}`.
"""
abstract type DimensionalFEFamily <: FEFamily end

# ========================================
# User-facing: dimension-agnostic families
# ========================================

"""
    Lagrange{Deg} <: FEFamily

Lagrange finite element family of polynomial degree `Deg`.
Supported degrees: 1, 2, 3.
```jldoctest
julia> Lagrange{1}()
Lagrange{1}()
```
"""
struct Lagrange{Deg} <: FEFamily end

"""
    Hermite{Deg} <: FEFamily

Hermite finite element family of polynomial degree `Deg`.
Supported degree: 3.
```jldoctest
julia> Hermite{3}()
Hermite{3}()
```
"""
struct Hermite{Deg} <: FEFamily end

# ========================================
# Internal: dimension-aware elements
# ========================================

"""
    LagrangeElement{Dim, Deg} <: DimensionalFEFamily

Lagrange element of spatial dimension `Dim` and degree `Deg`.
"""
struct LagrangeElement{Dim, Deg} <: DimensionalFEFamily end

"""
    HermiteElement{Dim, Deg} <: DimensionalFEFamily

Hermite element of spatial dimension `Dim` and degree `Deg`.
"""
struct HermiteElement{Dim, Deg} <: DimensionalFEFamily end

# ========================================
# specialize
# ========================================

"""
    specialize(fe::FEFamily, ::Val{Dim}) → DimensionalFEFamily

Specialize a dimension-agnostic family to spatial dimension `Dim`.
```jldoctest
julia> using WaveAcoustics: specialize, LagrangeElement, HermiteElement

julia> specialize(Lagrange{1}(), Val(1))
LagrangeElement{1, 1}()

julia> specialize(Hermite{3}(), Val(2))
HermiteElement{2, 3}()
```
"""
specialize(::Lagrange{Deg}, ::Val{Dim}) where {Deg, Dim} = LagrangeElement{Dim, Deg}()
specialize(::Hermite{Deg}, ::Val{Dim}) where {Deg, Dim} = HermiteElement{Dim, Deg}()

# ========================================
# Interface
# ========================================

"""
    num_local_dof(fe::DimensionalFEFamily) → Int

Return the number of local degrees of freedom for the finite element `fe`.
```jldoctest
julia> using WaveAcoustics: num_local_dof, specialize

julia> num_local_dof(specialize(Lagrange{3}(), Val(2)))
16

julia> num_local_dof(specialize(Hermite{3}(), Val(1)))
4
```
"""
function num_local_dof end

num_local_dof(::LagrangeElement{Dim, Deg}) where {Dim, Deg} = (Deg + 1)^Dim
num_local_dof(::HermiteElement{Dim, 3}) where {Dim} = 4^Dim
num_local_dof_static(::LagrangeElement{Dim, Deg}) where {Dim, Deg} = Val((Deg + 1)^Dim)
num_local_dof_static(::HermiteElement{Dim, 3}) where {Dim} = Val(4^Dim)

"""
    polynomial_degree(family::FEFamily)

Return polynomial degree for the finite element family.

# Examples
```jldoctest
julia> using WaveAcoustics: polynomial_degree, LagrangeElement, HermiteElement

julia> polynomial_degree(LagrangeElement{1,3}())
3

julia> polynomial_degree(HermiteElement{1,3}())
3

julia> polynomial_degree(Hermite{3}())
3
```
"""
function polynomial_degree end

polynomial_degree(::LagrangeElement{Dim, Deg}) where {Dim, Deg} = Deg
polynomial_degree(::HermiteElement{Dim, Deg}) where {Dim, Deg} = Deg

polynomial_degree(::Lagrange{Deg}) where {Deg} = Deg
polynomial_degree(::Hermite{Deg}) where {Deg} = Deg