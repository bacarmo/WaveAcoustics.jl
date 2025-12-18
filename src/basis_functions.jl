# ============================================================================
# LAGRANGE 1D — Reference element [-1, 1]
# ============================================================================

"""
    basis_functions(::Lagrange{1,1}, ξ)

Linear Lagrange basis functions on the reference interval [-1, 1].

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{2,T}`: Values of the two basis functions [ϕ₁(ξ), ϕ₂(ξ)]

# Node Layout
```
1 --- 2
```
"""
@inline function basis_functions(::Lagrange{1, 1}, ξ::T) where {T <: Real}
    half = T(0.5)
    SVector(
        half * (1 - ξ),
        half * (1 + ξ)
    )
end

"""
    basis_functions_derivatives(::Lagrange{1,1}, ξ)

Derivatives of linear Lagrange basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{2,T}`: Derivatives [dϕ₁/dξ, dϕ₂/dξ]

"""
@inline function basis_functions_derivatives(::Lagrange{1, 1}, ξ::T) where {T <: Real}
    half = T(0.5)
    SVector(-half, half)
end

"""
    basis_functions(::Lagrange{1,2}, ξ)

Quadratic Lagrange basis functions on the reference interval [-1, 1].

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{3,T}`: Values of the three basis functions [ϕ₁(ξ), ϕ₂(ξ), ϕ₃(ξ)]

# Node Layout
```
1 --- 2 --- 3
```
"""
@inline function basis_functions(::Lagrange{1, 2}, ξ::T) where {T <: Real}
    half = T(0.5)
    SVector(
        half * ξ * (ξ - 1),
        (1 - ξ) * (1 + ξ),
        half * ξ * (1 + ξ)
    )
end

"""
    basis_functions_derivatives(::Lagrange{1,2}, ξ)

Derivatives of quadratic Lagrange basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{3,T}`: Derivatives [dϕ₁/dξ, dϕ₂/dξ, dϕ₃/dξ]
"""
@inline function basis_functions_derivatives(::Lagrange{1, 2}, ξ::T) where {T <: Real}
    half = T(0.5)
    SVector(
        ξ - half,
        -2 * ξ,
        ξ + half
    )
end

"""
    basis_functions(::Lagrange{1,3}, ξ)

Cubic Lagrange basis functions on the reference interval [-1, 1].

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Values of the four basis functions [ϕ₁(ξ), ϕ₂(ξ), ϕ₃(ξ), ϕ₄(ξ)]

# Node Layout
```
1 --- 2 --- 3 --- 4
```
"""
@inline function basis_functions(::Lagrange{1, 3}, ξ::T) where {T <: Real}
    c = T(0.0625)
    SVector(
        c * (3 * ξ + 1) * (3 * ξ - 1) * (1 - ξ),
        c * (27 * ξ - 9) * (ξ + 1) * (ξ - 1),
        c * (27 * ξ + 9) * (ξ + 1) * (1 - ξ),
        c * (3 * ξ + 1) * (3 * ξ - 1) * (ξ + 1)
    )
end

"""
    basis_functions_derivatives(::Lagrange{1,3}, ξ)

Derivatives of cubic Lagrange basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Derivatives [dϕ₁/dξ, dϕ₂/dξ, dϕ₃/dξ, dϕ₄/dξ]
"""
@inline function basis_functions_derivatives(::Lagrange{1, 3}, ξ::T) where {T <: Real}
    c = T(0.0625)
    SVector(
        c * (ξ * (18 - 27 * ξ) + 1),
        c * (-27 - ξ * (18 - 81 * ξ)),
        c * (27 - ξ * (18 + 81 * ξ)),
        c * (ξ * (18 + 27 * ξ) - 1)
    )
end

# ============================================================================
# LAGRANGE 2D — Reference square [-1,1] × [-1,1]
# ============================================================================

"""
    basis_functions(::Lagrange{2,Deg}, ξ, η)

Tensor-product Lagrange basis functions on the reference square [-1,1] × [-1,1].

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- `SVector{(Deg+1)²,T}`: Values of all basis functions

# Node Layout (Deg = 1)
```
3 --- 4
|     |
1 --- 2
```

# Node Layout (Deg = 2)
```
7 --- 8 --- 9
|     |     |
4 --- 5 --- 6
|     |     |
1 --- 2 --- 3
```
"""
@inline function basis_functions(
        ::Lagrange{2, Deg}, ξ::T, η::T) where {Deg, T <: Real}
    φξ = basis_functions(Lagrange{1, Deg}(), ξ)
    φη = basis_functions(Lagrange{1, Deg}(), η)

    N = Deg + 1
    num_local_dof = N * N

    return SVector{num_local_dof}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            φξ[i] * φη[j]
        end, num_local_dof))
end

"""
    basis_functions_derivatives(::Lagrange{2,Deg}, ξ, η)

Derivatives of tensor-product Lagrange basis functions.

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- Tuple `(∂ϕ/∂ξ, ∂ϕ/∂η)` where:
  - `∂ϕ/∂ξ::SVector{(Deg+1)²,T}`: Derivatives ∂ϕᵢ/∂ξ
  - `∂ϕ/∂η::SVector{(Deg+1)²,T}`: Derivatives ∂ϕᵢ/∂η
"""
@inline function basis_functions_derivatives(
        ::Lagrange{2, Deg}, ξ::T, η::T) where {Deg, T <: Real}
    ϕξ = basis_functions(Lagrange{1, Deg}(), ξ)
    ϕη = basis_functions(Lagrange{1, Deg}(), η)
    dϕξ = basis_functions_derivatives(Lagrange{1, Deg}(), ξ)
    dϕη = basis_functions_derivatives(Lagrange{1, Deg}(), η)

    N = Deg + 1
    num_local_dof = N * N

    ∂ϕ_∂ξ = SVector{num_local_dof}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            dϕξ[i] * ϕη[j]
        end, num_local_dof))

    ∂ϕ_∂η = SVector{num_local_dof}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            ϕξ[i] * dϕη[j]
        end, num_local_dof))

    return ∂ϕ_∂ξ, ∂ϕ_∂η
end

# ============================================================================
# HERMITE 1D — Reference element [-1, 1]
# ============================================================================

"""
    basis_functions(::Hermite{1,3}, ξ)

Cubic Hermite basis functions on the reference interval [-1, 1].

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Values [H₁(ξ), H₁'(ξ), H₂(ξ), H₂'(ξ)]

# Node Layout
```
1:2 ---- 3:4
```
Each node has 2 DOFs: (u, du/dξ)
"""
@inline function basis_functions(::Hermite{1, 3}, ξ::T) where {T <: Real}
    quarter = T(0.25)
    SVector(
        (2 + ξ) * (1 - ξ)^2 * quarter,
        (ξ + 1) * (1 - ξ)^2 * quarter,
        (2 - ξ) * (1 + ξ)^2 * quarter,
        (ξ - 1) * (1 + ξ)^2 * quarter
    )
end

"""
    basis_functions_derivatives(::Hermite{1,3}, ξ)

Derivatives of cubic Hermite basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Derivatives [dH₁/dξ, dH₁'/dξ, dH₂/dξ, dH₂'/dξ]
"""
@inline function basis_functions_derivatives(::Hermite{1, 3}, ξ::T) where {T <: Real}
    quarter = T(0.25)
    SVector(
        3 * (-1 + ξ) * (1 + ξ) * quarter,
        (ξ - 1) * (1 + 3 * ξ) * quarter,
        3 * (1 - ξ) * (1 + ξ) * quarter,
        (ξ + 1) * (-1 + 3 * ξ) * quarter
    )
end

# ============================================================================
# HERMITE 2D — Reference square [-1,1] × [-1,1]
# ============================================================================

"""
    basis_functions(::Hermite{2,3}, ξ, η)

Bicubic Hermite basis functions on the reference square [-1,1] × [-1,1].

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- `SVector{16,T}`: Values of all 16 basis functions

# Associate Node Layout
```
 9:12 ---- 13:16
  |          |
  |          |
 1:4  ----  5:8

# Notes
Each corner node has 4 DOFs.
"""
@inline function basis_functions(::Hermite{2, 3}, ξ::T, η::T) where {T <: Real}
    Hξ = basis_functions(Hermite{1, 3}(), ξ)
    Hη = basis_functions(Hermite{1, 3}(), η)

    SVector{16}(ntuple(k -> begin
            i = mod(k - 1, 4) + 1
            j = div(k - 1, 4) + 1
            Hξ[i] * Hη[j]
        end, 16))
end

"""
    basis_functions_derivatives(::Hermite{2,3}, ξ, η)

Derivatives of Bicubic Hermite basis functions.

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- Tuple `(∂ϕ/∂ξ, ∂ϕ/∂η)` where:
  - `∂ϕ/∂ξ::SVector{16,T}`: Derivatives ∂ϕᵢ/∂ξ
  - `∂ϕ/∂η::SVector{16,T}`: Derivatives ∂ϕᵢ/∂η
"""
@inline function basis_functions_derivatives(::Hermite{2, 3}, ξ::T, η::T) where {T <: Real}
    Hξ = basis_functions(Hermite{1, 3}(), ξ)
    Hη = basis_functions(Hermite{1, 3}(), η)

    dHξ = basis_functions_derivatives(Hermite{1, 3}(), ξ)
    dHη = basis_functions_derivatives(Hermite{1, 3}(), η)

    ∂ϕ_∂ξ = SVector{16}(ntuple(k -> begin
            i = mod(k - 1, 4) + 1
            j = div(k - 1, 4) + 1
            dHξ[i] * Hη[j]
        end, 16))
    ∂ϕ_∂η = SVector{16}(ntuple(k -> begin
            i = mod(k - 1, 4) + 1
            j = div(k - 1, 4) + 1
            Hξ[i] * dHη[j]
        end, 16))

    return ∂ϕ_∂ξ, ∂ϕ_∂η
end