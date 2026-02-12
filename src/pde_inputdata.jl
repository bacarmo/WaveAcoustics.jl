const ScalarFunction1D = FunctionWrapper{Float64, Tuple{Float64}}
const ScalarFunction2D = FunctionWrapper{Float64, Tuple{Float64, Float64}}
const ScalarFunction3D = FunctionWrapper{Float64, Tuple{Float64, Float64, Float64}}

"""
    PDEInputData

Input data configuration for coupled wave-acoustic PDE system.

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
- `pmin::NTuple{2,Float64}`: Bottom-left corner (xmin, ymin). Default: `(0.0, 0.0)`
- `pmax::NTuple{2,Float64}`: Top-right corner (xmax, ymax). Default: `(1.0, 1.0)`
- `t_final::Float64`: Final simulation time. Default: `1.0`

### Physical Parameters
- `q₁::Float64`: Acoustic acceleration coefficient. Default: `1.0`
- `q₂::Float64`: Acoustic velocity coefficient. Default: `1.0`
- `q₃::Float64`: Acoustic displacement coefficient. Default: `1.0`
- `q₄::Float64`: Wave-acoustic coupling strength. Default: `1.0`

### Coefficient Functions
- `α::ScalarFunction1D`: Time-dependent wave diffusion coefficient α(t)
- `f::ScalarFunction1D`, `df::ScalarFunction1D`: Nonlinear wave term f(s) and derivative f'(s)
- `g::ScalarFunction2D`, `∂ₛg::ScalarFunction2D`: Nonlinear coupling function g(x,s) and s-derivative ∂ₛg(x,s)

### Wave Initial Conditions (2D functions on Ω)
- `u₀::ScalarFunction2D`, `∂ₓu₀::ScalarFunction2D`, `∂ᵧu₀::ScalarFunction2D`: Displacement u(x,y,0) and spatial derivatives
- `v₀::ScalarFunction2D`, `∂ₓv₀::ScalarFunction2D`, `∂ᵧv₀::ScalarFunction2D`: Velocity v(x,y,0) = ∂ₜu(x,y,0) and spatial derivatives

### Acoustic Initial Conditions (1D functions on Γ₁)
- `z₀::ScalarFunction1D`: Acoustic displacement z(x,0)
- `r₀::ScalarFunction1D`: Acoustic velocity r(x,0) = ∂ₜz(x,0)

### Source Terms
- `f₁::ScalarFunction3D`: Wave source term f₁(x,y,t) on Ω
- `f₂::ScalarFunction2D`: Acoustic source term f₂(x,t) on Γ₁

### Analytical Solutions
For manufactured solution cases, provide analytical solutions for convergence studies:
- `u::Union{Nothing,ScalarFunction3D}`, `v::Union{Nothing,ScalarFunction3D}`: Analytical wave solutions u(x,y,t), v(x,y,t)
- `z::Union{Nothing,ScalarFunction3D}`, `r::Union{Nothing,ScalarFunction2D}`: Analytical acoustic solutions z(x,t), r(x,t)

For physical simulations without known solutions, these should return `nothing`.
"""
struct PDEInputData
    # Domain 
    pmin::NTuple{2, Float64}
    pmax::NTuple{2, Float64}
    t_final::Float64

    # Constants
    q₁::Float64
    q₂::Float64
    q₃::Float64
    q₄::Float64

    # Functions
    α::ScalarFunction1D
    f::ScalarFunction1D
    df::ScalarFunction1D
    g::ScalarFunction2D
    ∂ₛg::ScalarFunction2D

    # Wave initial conditions
    u₀::ScalarFunction2D
    ∂ₓu₀::ScalarFunction2D
    ∂ᵧu₀::ScalarFunction2D
    v₀::ScalarFunction2D
    ∂ₓv₀::ScalarFunction2D
    ∂ᵧv₀::ScalarFunction2D

    # Acoustic initial conditions
    z₀::ScalarFunction1D
    r₀::ScalarFunction1D

    # Source terms and analytical solutions
    f₁::ScalarFunction3D
    f₂::ScalarFunction2D
    u::Union{Nothing, ScalarFunction3D}
    v::Union{Nothing, ScalarFunction3D}
    z::Union{Nothing, ScalarFunction2D}
    r::Union{Nothing, ScalarFunction2D}
end

# ============================================================================
# Example 1
# ============================================================================
"""
    example1_manufactured(a::Float64=2.4) -> PDEInputData

Example 1 with manufactured solutions for convergence study.

# Arguments
- `a::Float64=2.4`: Exponent for function u(x,y,t) = (xᵃ-x)(yᵃ-1)(4+t²)
"""
function example1_manufactured(a::Float64 = 2.4)
    # Constants
    a_minus_1 = a - 1.0
    a_minus_2 = a - 2.0
    axa_minus_1 = a * a_minus_1
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = t -> 1.0 + exp(-t)
    f = s -> s * abs(s)^3
    df = s -> 4 * abs(s)^3
    g = (x, s) -> (1 + exp(-x^2)) * (sin(s) + 2s)
    ∂ₛg = (x, s) -> (1 + exp(-x^2)) * (cos(s) + 2)

    # Analytical solutions
    u = (x, y, t) -> (x^a - x) * (y^a - 1) * (4 + t^2)
    v = (x, y, t) -> (x^a - x) * (y^a - 1) * (2 * t)
    z = (x, t) -> sinpi(x) +
                  (1 + exp(-x^2)) *
                  ((cos(-(x^a - x) * (2 * t)) - 1) / (2 * (x^a - x)) -
                   (x^a - x) * (2 * t^2))
    r = (x, t) -> (1 + exp(-x^2)) * (sin(-(x^a - x) * (2 * t)) - (x^a - x) * (4 * t))

    # Auxiliary functions for manufactured sources
    @inline ∂ₜₜu(x, y, t) = (x^a - x) * (y^a - 1) * 2
    @inline Δu(x, y, t) = ((axa_minus_1 * x^a_minus_2) * (y^a - 1) +
                           (x^a - x) * (axa_minus_1 * y^a_minus_2)) * (4 + t^2)
    @inline ∂ₜₜz(x, t) = -2 * (1 + exp(-x^2)) * (x^a - x) * (2 + cos(2 * t * (x^a - x)))

    # Manufactured source terms
    f₁ = (x, y, t) -> ∂ₜₜu(x, y, t) - α(t) * Δu(x, y, t) + f(u(x, y, t))
    f₂ = (x, t) -> q₁ * ∂ₜₜz(x, t) + q₂ * r(x, t) + q₃ * z(x, t) + q₄ * v(x, ymin, t)

    # Initial conditions
    u₀ = (x, y) -> (x^a - x) * (y^a - 1) * 4
    ∂ₓu₀ = (x, y) -> (a * x^a_minus_1 - 1) * (y^a - 1) * 4
    ∂ᵧu₀ = (x, y) -> (x^a - x) * (a * y^a_minus_1) * 4
    v₀ = (x, y) -> 0.0
    ∂ₓv₀ = (x, y) -> 0.0
    ∂ᵧv₀ = (x, y) -> 0.0
    z₀ = x -> sinpi(x)
    r₀ = x -> 0.0

    return PDEInputData(
        (0.0, ymin),
        (1.0, 1.0),
        1.0,
        q₁, q₂, q₃, q₄,
        ScalarFunction1D(α), ScalarFunction1D(f), ScalarFunction1D(df),
        ScalarFunction2D(g), ScalarFunction2D(∂ₛg),
        ScalarFunction2D(u₀), ScalarFunction2D(∂ₓu₀), ScalarFunction2D(∂ᵧu₀),
        ScalarFunction2D(v₀), ScalarFunction2D(∂ₓv₀), ScalarFunction2D(∂ᵧv₀),
        ScalarFunction1D(z₀), ScalarFunction1D(r₀),
        ScalarFunction3D(f₁), ScalarFunction2D(f₂),
        ScalarFunction3D(u), ScalarFunction3D(v),
        ScalarFunction2D(z), ScalarFunction2D(r)
    )
end

"""
    example1_zero_source(a::Float64=2.4) -> PDEInputData

Example 1 with zero source terms for physical simulation.
No analytical solution available.

# Arguments
- `a::Float64=2.4`: Exponent for function u₀(x,y) = 4(xᵃ-x)(yᵃ-1)
"""
function example1_zero_source(a::Float64 = 2.4)
    # Constants
    a_minus_1 = a - 1.0
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = t -> 1.0 + exp(-t)
    f = s -> s * abs(s)^3
    df = s -> 4 * abs(s)^3
    g = (x, s) -> (1 + exp(-x^2)) * (sin(s) + 2s)
    ∂ₛg = (x, s) -> (1 + exp(-x^2)) * (cos(s) + 2)

    # Zero source terms
    f₁ = (x, y, t) -> 0.0
    f₂ = (x, t) -> 0.0

    # Initial conditions
    u₀ = (x, y) -> (x^a - x) * (y^a - 1) * 4
    ∂ₓu₀ = (x, y) -> (a * x^a_minus_1 - 1) * (y^a - 1) * 4
    ∂ᵧu₀ = (x, y) -> (x^a - x) * (a * y^a_minus_1) * 4
    v₀ = (x, y) -> 0.0
    ∂ₓv₀ = (x, y) -> 0.0
    ∂ᵧv₀ = (x, y) -> 0.0
    z₀ = x -> sinpi(x)
    r₀ = x -> 0.0

    return PDEInputData(
        (0.0, ymin),
        (1.0, 1.0),
        1.0,
        q₁, q₂, q₃, q₄,
        ScalarFunction1D(α), ScalarFunction1D(f), ScalarFunction1D(df),
        ScalarFunction2D(g), ScalarFunction2D(∂ₛg),
        ScalarFunction2D(u₀), ScalarFunction2D(∂ₓu₀), ScalarFunction2D(∂ᵧu₀),
        ScalarFunction2D(v₀), ScalarFunction2D(∂ₓv₀), ScalarFunction2D(∂ᵧv₀),
        ScalarFunction1D(z₀), ScalarFunction1D(r₀),
        ScalarFunction3D(f₁), ScalarFunction2D(f₂),
        nothing,
        nothing,
        nothing,
        nothing
    )
end

# ============================================================================
# Example 2
# ============================================================================
"""
    example2_manufactured(a::Float64=2.4) -> PDEInputData

Example 2 with manufactured solutions for convergence study.

# Arguments
- `a::Float64=2.4`: Exponent for function u(x,y,t) = (xᵃ-x)(yᵃ-1)(4+t²)
"""
function example2_manufactured(a::Float64 = 2.4)
    # Constants
    a_minus_1 = a - 1.0
    a_minus_2 = a - 2.0
    axa_minus_1 = a * a_minus_1
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = t -> 1.0 + exp(-t)
    f = s -> s * abs(s)^3
    df = s -> 4 * abs(s)^3
    g = (x, s) -> (1 + exp(-x * x)) * s
    ∂ₛg = (x, s) -> (1 + exp(-x * x))

    # Analytical solutions
    u = (x, y, t) -> (x^a - x) * (y^a - 1) * (4 + t^2)
    v = (x, y, t) -> (x^a - x) * (y^a - 1) * (2 * t)
    z = (x, t) -> sinpi(x) - (1 + exp(-x * x)) * (x^a - x) * t^2
    r = (x, t) -> -(1 + exp(-x * x)) * (x^a - x) * (2 * t)

    # Auxiliary functions for manufactured sources
    @inline ∂ₜₜu(x, y, t) = (x^a - x) * (y^a - 1) * 2
    @inline Δu(x, y, t) = ((axa_minus_1 * x^a_minus_2) * (y^a - 1) +
                           (x^a - x) * (axa_minus_1 * y^a_minus_2)) * (4 + t^2)
    @inline ∂ₜₜz(x, t) = -2 * (1 + exp(-x^2)) * (x^a - x)

    # Manufactured source terms
    f₁ = (x, y, t) -> ∂ₜₜu(x, y, t) - α(t) * Δu(x, y, t) + f(u(x, y, t))
    f₂ = (x, t) -> q₁ * ∂ₜₜz(x, t) + q₂ * r(x, t) + q₃ * z(x, t) + q₄ * v(x, ymin, t)

    # Initial conditions
    u₀ = (x, y) -> (x^a - x) * (y^a - 1) * 4
    ∂ₓu₀ = (x, y) -> (a * x^a_minus_1 - 1) * (y^a - 1) * 4
    ∂ᵧu₀ = (x, y) -> (x^a - x) * (a * y^a_minus_1) * 4
    v₀ = (x, y) -> 0.0
    ∂ₓv₀ = (x, y) -> 0.0
    ∂ᵧv₀ = (x, y) -> 0.0
    z₀ = x -> sinpi(x)
    r₀ = x -> 0.0

    return PDEInputData(
        (0.0, ymin),
        (1.0, 1.0),
        1.0,
        q₁, q₂, q₃, q₄,
        ScalarFunction1D(α), ScalarFunction1D(f), ScalarFunction1D(df),
        ScalarFunction2D(g), ScalarFunction2D(∂ₛg),
        ScalarFunction2D(u₀), ScalarFunction2D(∂ₓu₀), ScalarFunction2D(∂ᵧu₀),
        ScalarFunction2D(v₀), ScalarFunction2D(∂ₓv₀), ScalarFunction2D(∂ᵧv₀),
        ScalarFunction1D(z₀), ScalarFunction1D(r₀),
        ScalarFunction3D(f₁), ScalarFunction2D(f₂),
        ScalarFunction3D(u), ScalarFunction3D(v),
        ScalarFunction2D(z), ScalarFunction2D(r)
    )
end

"""
    example2_zero_source(a::Float64=2.4) -> PDEInputData

Example 2 with zero source terms for physical simulation.
No analytical solution available.

# Arguments
- `a::Float64=2.4`: Exponent for function u₀(x,y) = 4(xᵃ-x)(yᵃ-1)
"""
function example2_zero_source(a::Float64 = 2.4)
    # Constants
    a_minus_1 = a - 1.0
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = t -> 1.0 + exp(-t)
    f = s -> s * abs(s)^3
    df = s -> 4 * abs(s)^3
    g = (x, s) -> (1 + exp(-x * x)) * s
    ∂ₛg = (x, s) -> (1 + exp(-x * x))

    # Zero source terms
    f₁ = (x, y, t) -> 0.0
    f₂ = (x, t) -> 0.0

    # Initial conditions
    u₀ = (x, y) -> (x^a - x) * (y^a - 1) * 4
    ∂ₓu₀ = (x, y) -> (a * x^a_minus_1 - 1) * (y^a - 1) * 4
    ∂ᵧu₀ = (x, y) -> (x^a - x) * (a * y^a_minus_1) * 4
    v₀ = (x, y) -> 0.0
    ∂ₓv₀ = (x, y) -> 0.0
    ∂ᵧv₀ = (x, y) -> 0.0
    z₀ = x -> sinpi(x)
    r₀ = x -> 0.0

    return PDEInputData(
        (0.0, ymin),
        (1.0, 1.0),
        1.0,
        q₁, q₂, q₃, q₄,
        ScalarFunction1D(α), ScalarFunction1D(f), ScalarFunction1D(df),
        ScalarFunction2D(g), ScalarFunction2D(∂ₛg),
        ScalarFunction2D(u₀), ScalarFunction2D(∂ₓu₀), ScalarFunction2D(∂ᵧu₀),
        ScalarFunction2D(v₀), ScalarFunction2D(∂ₓv₀), ScalarFunction2D(∂ᵧv₀),
        ScalarFunction1D(z₀), ScalarFunction1D(r₀),
        ScalarFunction3D(f₁), ScalarFunction2D(f₂),
        nothing,
        nothing,
        nothing,
        nothing
    )
end