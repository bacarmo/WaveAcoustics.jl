"""
    PDECommonData{T<:Real, Fα, Ff, Fdf, Fg, F∂ₛg, Fu₀, F∂ₓu₀, F∂ᵧu₀, Fv₀, F∂ₓv₀, F∂ᵧv₀, Fz₀, Fr₀}

Common configuration for coupled wave-acoustic PDE system.

## Mathematical Specification

Wave equation on rectangular domain Ω = ]xmin,xmax[ × ]ymin,ymax[:
```math
\\frac{∂²u}{∂t²} - α(t)Δu + f(u) = f₁(x,y,t) \\quad \\text{in } Ω
```

with homogeneous boundary conditions on Γ₀ = ∂Ω \\ Γ₁.

Acoustic equation on bottom boundary Γ₁ = {(x,ymin) : x ∈ ]xmin,xmax[}:
```math
q₁\\frac{∂²z}{∂t²} + q₂\\frac{∂z}{∂t} + q₃z + q₄v(x,ymin,t) = f₂(x,t)
```

where v = ∂u/∂t couples the wave velocity at Γ₁ to the acoustic equation.

## Fields

### Domain Configuration
- `pmin::NTuple{2,T}`: Bottom-left corner (xmin, ymin). Default: `(0, 0)`
- `pmax::NTuple{2,T}`: Top-right corner (xmax, ymax). Default: `(1, 1)`
- `t_final::T`: Final simulation time. Default: `1`

### Physical Parameters
- `q₁::T`: Acoustic acceleration coefficient. Default: `1`
- `q₂::T`: Acoustic velocity coefficient. Default: `1`
- `q₃::T`: Acoustic displacement coefficient. Default: `1`
- `q₄::T`: Wave-acoustic coupling strength. Default: `1`

### Coefficient Functions
- `α::Fα`: Time-dependent wave diffusion coefficient α(t)
- `f::Ff`, `df::Fdf`: Nonlinear wave term f(s) and derivative f'(s)
- `g::Fg`, `∂ₛg::F∂ₛg`: Nonlinear coupling function g(x,s) and s-derivative ∂ₛg(x,s)

### Wave Initial Conditions (2D functions on Ω)
- `u₀::Fu₀`, `∂ₓu₀::F∂ₓu₀`, `∂ᵧu₀::F∂ᵧu₀`: Displacement u(x,y,0) and spatial derivatives
- `v₀::Fv₀`, `∂ₓv₀::F∂ₓv₀`, `∂ᵧv₀::F∂ᵧv₀`: Velocity v(x,y,0) = ∂ₜu(x,y,0) and spatial derivatives

### Acoustic Initial Conditions (1D functions on Γ₁)
- `z₀::Fz₀`: Acoustic displacement z(x,0)
- `r₀::Fr₀`: Acoustic velocity r(x,0) = ∂ₜz(x,0)
"""
Base.@kwdef struct PDECommonData{T <: Real, Fα, Ff, Fdf, Fg, F∂ₛg,
    Fu₀, F∂ₓu₀, F∂ᵧu₀, Fv₀, F∂ₓv₀, F∂ᵧv₀,
    Fz₀, Fr₀}
    # Domain 
    pmin::NTuple{2, T} = (zero(T), zero(T))
    pmax::NTuple{2, T} = (one(T), one(T))
    t_final::T = one(T)

    # Constants
    q₁::T = one(T)
    q₂::T = one(T)
    q₃::T = one(T)
    q₄::T = one(T)

    # Functions
    α::Fα
    f::Ff
    df::Fdf
    g::Fg
    ∂ₛg::F∂ₛg

    # Wave initial conditions
    u₀::Fu₀
    ∂ₓu₀::F∂ₓu₀
    ∂ᵧu₀::F∂ᵧu₀
    v₀::Fv₀
    ∂ₓv₀::F∂ₓv₀
    ∂ᵧv₀::F∂ᵧv₀

    # Acoustic initial conditions
    z₀::Fz₀
    r₀::Fr₀
end

"""
    PDEInputData{C<:PDECommonData, Ff₁, Ff₂, Fu, Fv, Fz, Fr}

Complete problem specification including source terms and analytical solutions.

## Fields
- `common::C`: Problem configuration (see [`PDECommonData`](@ref))
- `f₁::Ff₁`: Wave source term f₁(x,y,t) on Ω
- `f₂::Ff₂`: Acoustic source term f₂(x,t) on Γ₁
- `u::Fu`, `v::Fv`: Analytical wave solutions u(x,y,t), v(x,y,t) (if known)
- `z::Fz`, `r::Fr`: Analytical acoustic solutions z(x,t), r(x,t) (if known)
"""
Base.@kwdef struct PDEInputData{C <: PDECommonData, Ff₁, Ff₂, Fu, Fv, Fz, Fr}
    common::C
    f₁::Ff₁
    f₂::Ff₂
    u::Fu
    v::Fv
    z::Fz
    r::Fr
end

"""
    manufactured_solution_case(common, f₁, f₂, u, v, z, r) -> PDEInputData

Construct problem with manufactured solutions for convergence study.

Source terms `f₁` and `f₂` are computed by substituting analytical solutions
into the governing PDEs (Method of Manufactured Solutions).

## Arguments
- `common::PDECommonData`: Problem configuration
- `f₁`: Manufactured wave source f₁(x,y,t)
- `f₂`: Manufactured acoustic source f₂(x,t)
- `u`, `v`: Analytical wave solutions u(x,y,t), v(x,y,t)
- `z`, `r`: Analytical acoustic solutions z(x,t), r(x,t)

## See also
[`zero_source_case`](@ref)
"""
function manufactured_solution_case(common::PDECommonData{T}, f₁, f₂, u, v, z, r) where {T}
    return PDEInputData(
        common = common,
        f₁ = f₁,
        f₂ = f₂,
        u = u,
        v = v,
        z = z,
        r = r
    )
end

"""
    zero_source_case(common) -> PDEInputData

Construct problem with zero source terms (f₁ = f₂ = 0).

No analytical solution available. Dummy functions returning `nothing` are used
for analytical solution fields.

## See also
[`manufactured_solution_case`](@ref)
"""
function zero_source_case(common::PDECommonData{T}) where {T}
    zero_f₁ = (x, y, t) -> zero(T)
    zero_f₂ = (x, t) -> zero(T)
    dummy_2d = (x, y, t) -> nothing
    dummy_1d = (x, t) -> nothing

    return PDEInputData(
        common = common,
        f₁ = zero_f₁,
        f₂ = zero_f₂,
        u = dummy_2d,
        v = dummy_2d,
        z = dummy_1d,
        r = dummy_1d
    )
end

# ============================================================================
# Example 1
# ============================================================================

"""
    example1_common_data([T=Float64]) -> PDECommonData{T}

Ω = ]0,1[², f(s) = s|s|³, g(x,s) = (1+exp(-x²))(sin(s)+2s), u₀(x,y) = (x^2.4 - x) * (y^2.4 - 1) * 4, z₀(x) = sinpi(x).
"""
function example1_common_data(::Type{T} = Float64) where {T}
    return PDECommonData(
        pmin = (zero(T), zero(T)),
        pmax = (one(T), one(T)),
        t_final = one(T),
        q₁ = T(1),
        q₂ = T(1),
        q₃ = T(1),
        q₄ = T(1),
        α = t -> one(T) + exp(-t),
        f = s -> s * abs(s)^3,
        df = s -> 4 * abs(s)^3,
        g = (x, s) -> (1 + exp(-x * x)) * (sin(s) + 2 * s),
        ∂ₛg = (x, s) -> (1 + exp(-x * x)) * (cos(s) + 2),
        u₀ = (x, y) -> (x^2.4 - x) * (y^2.4 - 1) * 4,
        ∂ₓu₀ = (x, y) -> (2.4 * x^1.4 - 1) * (y^2.4 - 1) * 4,
        ∂ᵧu₀ = (x, y) -> (x^2.4 - x) * (2.4 * y^1.4) * 4,
        v₀ = (x, y) -> zero(T),
        ∂ₓv₀ = (x, y) -> zero(T),
        ∂ᵧv₀ = (x, y) -> zero(T),
        z₀ = x -> sinpi(x),
        r₀ = x -> zero(T)
    )
end

"""
    example1_manufactured([T=Float64]) -> PDEInputData{T}

Example 1 with manufactured solutions for convergence study.
"""
function example1_manufactured(::Type{T} = Float64) where {T}
    common = example1_common_data(T)
    ymin = common.pmin[2]

    # Analytical solutions
    u = (x, y, t) -> (x^2.4 - x) * (y^2.4 - 1) * (4 + t^2)
    v = (x, y, t) -> (x^2.4 - x) * (y^2.4 - 1) * (2 * t)
    z = (x, t) -> sinpi(x) +
                  (1 + exp(-x^2)) *
                  ((cos(-(x^2.4 - x) * (2 * t)) - 1) / (2 * (x^2.4 - x)) -
                   (x^2.4 - x) * (2 * t^2))
    r = (x, t) -> (1 + exp(-x^2)) * (sin(-(x^2.4 - x) * (2 * t)) - (x^2.4 - x) * (4 * t))

    # Auxiliary functions for manufactured sources
    @inline ∂ₜₜu(x, y, t) = (x^2.4 - x) * (y^2.4 - 1) * 2
    @inline Δu(x, y, t) = ((2.4 * 1.4 * x^0.4) * (y^2.4 - 1) +
                           (x^2.4 - x) * (2.4 * 1.4 * y^0.4)) * (4 + t^2)
    @inline ∂ₜₜz(x, t) = -2 * (1 + exp(-x^2)) * (x^2.4 - x) * (2 + cos(2 * t * (x^2.4 - x)))

    # Manufactured source terms
    f₁ = (x, y, t) -> ∂ₜₜu(x, y, t) - common.α(t) * Δu(x, y, t) + common.f(u(x, y, t))
    f₂ = (x, t) -> common.q₁ * ∂ₜₜz(x, t) + common.q₂ * r(x, t) +
                   common.q₃ * z(x, t) + common.q₄ * v(x, ymin, t)

    return manufactured_solution_case(common, f₁, f₂, u, v, z, r)
end

"""
    example1_zero_source([T=Float64]) -> PDEInputData{T}

Example 1 with zero source terms for physical simulation.
"""
function example1_zero_source(::Type{T} = Float64) where {T}
    common = example1_common_data(T)
    return zero_source_case(common)
end