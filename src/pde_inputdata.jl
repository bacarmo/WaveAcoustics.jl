"""
    PDEInputData{Tα, Tf, Tdf, Tg, T∂ₛg, 
                 Tu₀, T∂ₓu₀, T∂ᵧu₀, 
                 Tv₀, T∂ₓv₀, T∂ᵧv₀, 
                 Tz₀, Tr₀, 
                 Tf₁, Tf₂, 
                 Tu, Tv, Tz, Tr}

Input data for a coupled wave-acoustic PDE system on a rectangular domain
``\\Omega = ]x_{\\min}, x_{\\max}[ \\times ]y_{\\min}, y_{\\max}[``,
with acoustic interaction prescribed on the lower boundary
``\\Gamma_1 = ]x_{\\min}, x_{\\max}[ \\times \\{y_{\\min}\\}``,
where ``\\Gamma = \\partial\\Omega`` and ``\\Gamma_0 = \\Gamma \\setminus \\Gamma_1``.

# Mathematical Model
```math
\\begin{align*}
\\begin{aligned}
& u^{\\prime\\prime}(x,y,t) 
- \\alpha(t)\\Delta u(x,y,t) 
+ f\\big(u(x,y,t)\\big) 
= f_1(x,y,t),
\\\\[5pt]
& q_1 z^{\\prime\\prime}(x,t)
+ q_2 z^\\prime(x,t)
+ q_3 z(x,t)
+ q_4 u^\\prime(x,y_{min},t)
= f_2(x,t),
\\\\[5pt]
& -\\frac{\\partial u}{\\partial y}(x,y_{min},t)
= z^\\prime(x,t)
- g\\big(x,u^\\prime(x,y_{min},t)\\big),
\\quad(x,t)\\in]x_{min},x_{max}[\\times(0,+\\infty),
\\\\[5pt]
& u(x,y,t) = 0,\\quad (x,y,t)\\in\\Gamma_0\\times(0,+\\infty),
\\end{aligned}
\\end{align*}
```
with initial conditions
```math
\\begin{align*}
\\begin{aligned}
& u(x,y,0) = u_0(x,y),\\quad u^\\prime(x,y,0)=v_0(x,y),\\quad (x,y)\\in\\Omega,
\\\\
& z(x,0) = z_0(x),\\quad
  z^\\prime(x,0) = r_0(x) \\equiv -\\frac{\\partial u_0}{\\partial y}(x,y_{\\min}) + g\\big(x,v_0(x,y_{\\min})\\big),\\quad x\\in]x_{\\min},y_{\\max}[.
\\end{aligned}
\\end{align*}
```

# Fields

## Problem Identification
- `name::String`: Identifier for the problem instance (e.g., "example1_manufactured(2.4)")
## Domain Geometry
- `pmin::NTuple{2,Float64}`: Bottom-left corner ``(x_{\\min}, y_{\\min})`` of ``Ω``.
- `pmax::NTuple{2,Float64}`: Top-right corner ``(x_{\\max}, y_{\\max})`` of ``Ω``.

## Physical Parameters
- `q₁::Float64`: Acoustic acceleration coefficient
- `q₂::Float64`: Acoustic velocity coefficient
- `q₃::Float64`: Acoustic displacement coefficient
- `q₄::Float64`: Wave-acoustic coupling strength

## Coefficient Functions
- `α::Tα`: Time-dependent wave diffusion coefficient α(t)
- `f::Tf`, `df::Tdf`: Nonlinear wave term f(s) and derivative f'(s)
- `g::Tg`, `∂ₛg::T∂ₛg`: Nonlinear coupling g(x,s) and s-derivative ∂ₛg(x,s)

## Wave Initial Conditions (2D functions on Ω)
- `u₀::Tu₀`: Displacement u(x,y,0)
- `∂ₓu₀::T∂ₓu₀`, `∂ᵧu₀::T∂ᵧu₀`: Spatial derivatives of displacement
- `v₀::Tv₀`: Velocity v(x,y,0) = ∂ₜu(x,y,0)
- `∂ₓv₀::T∂ₓv₀`, `∂ᵧv₀::T∂ᵧv₀`: Spatial derivatives of velocity

## Acoustic Initial Conditions (1D functions on Γ₁)
- `z₀::Tz₀`: Acoustic displacement z(x,0)
- `r₀::Tr₀`: Acoustic velocity r(x,0) = ∂ₜz(x,0)

## Source Terms
- `f₁::Tf₁`: Wave source term f₁(x,y,t) on Ω
- `f₂::Tf₂`: Acoustic source term f₂(x,t) on Γ₁

## Analytical Solutions
For manufactured solution cases, provide analytical solutions for convergence studies:
- `u::Tu`, `v::Tv`: Analytical wave solutions u(x,y,t), v(x,y,t)
- `z::Tz`, `r::Tr`: Analytical acoustic solutions z(x,t), r(x,t)

For physical simulations without known solutions, these fields are `nothing`.
"""
struct PDEInputData{
    Tα, Tf, Tdf, Tg, T∂ₛg,
    Tu₀, T∂ₓu₀, T∂ᵧu₀,
    Tv₀, T∂ₓv₀, T∂ᵧv₀,
    Tz₀, Tr₀,
    Tf₁, Tf₂,
    Tu, Tv, Tz, Tr}
    # Problem Identification
    name::String

    # Domain geometry
    pmin::NTuple{2, Float64}
    pmax::NTuple{2, Float64}

    # Physical parameters
    q₁::Float64
    q₂::Float64
    q₃::Float64
    q₄::Float64

    # Coefficient functions
    α::Tα
    f::Tf
    df::Tdf
    g::Tg
    ∂ₛg::T∂ₛg

    # Wave initial conditions
    u₀::Tu₀
    ∂ₓu₀::T∂ₓu₀
    ∂ᵧu₀::T∂ᵧu₀
    v₀::Tv₀
    ∂ₓv₀::T∂ₓv₀
    ∂ᵧv₀::T∂ᵧv₀

    # Acoustic initial conditions
    z₀::Tz₀
    r₀::Tr₀

    # Source terms
    f₁::Tf₁
    f₂::Tf₂

    # Analytical solutions
    u::Tu
    v::Tv
    z::Tz
    r::Tr
end

# ============================================================================
# Example 1
# ============================================================================
"""
    example1_manufactured(a::Float64=2.4) -> PDEInputData

Manufactured solution with
``g(x, s) = (1 + e^{-x^2})(\\sin(s) + 2s)``:
```math
\\begin{alignat*}{2}
& u(x,y,t)   &&= (x^a - x)(y^a - 1)(4 + t^2), \\\\
& z(x,t)     &&= \\sin(\\pi x) + (1+e^{-x^2})
                  \\left[\\frac{\\cos\\big(-2t(x^a-x)\\big)-1}{2(x^a-x)}
                  - 2t^2(x^a-x)\\right],
\\end{alignat*}
```
where the acoustic displacement ``z(x,t)`` is obtained by integrating
```math
\\frac{∂z}{∂t}(x,t) = -\\frac{∂u}{∂y}(x,y_{\\min},t) + g\\!\\left(x,\\,\\frac{∂u}{∂t}(x,y_{\\min},t)\\right).
```

# Arguments
- `a::Float64=2.4`: Smoothness parameter controlling solution regularity.

# Returns
`PDEInputData` with analytical solution for convergence study.
"""
function example1_manufactured(a::Float64 = 2.4)
    # Precompute exponent-related constants
    a_minus_1 = a - 1.0
    a_minus_2 = a - 2.0
    axa_minus_1 = a * a_minus_1

    # Physical parameters
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = @inline t -> 1.0 + exp(-t)

    f = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return s * s_abs3
    end
    df = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return 4.0 * s_abs3
    end

    g = @inline (x, s) -> (1.0 + exp(-x * x)) * muladd(2.0, s, sin(s))
    ∂ₛg = @inline (x, s) -> (1.0 + exp(-x * x)) * (2.0 + cos(s))

    # Analytical solutions
    u = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * (4.0 + t * t)
    end
    v = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * (2.0 * t)
    end

    z = @inline function (x, t)
        xa = x^a
        xa_minus_x = xa - x
        two_t = 2.0 * t
        exp_term = 1.0 + exp(-x * x)

        return sinpi(x) +
               exp_term * ((cos(-xa_minus_x * two_t) - 1.0) /
                (2.0 * xa_minus_x) - xa_minus_x * (two_t * t))
    end
    r = @inline function (x, t)
        xa = x^a
        xa_minus_x = xa - x
        exp_term = 1.0 + exp(-x * x)

        return exp_term * (sin(-xa_minus_x * (2.0 * t)) - xa_minus_x * (4.0 * t))
    end

    # Auxiliary functions for manufactured source terms
    ∂ₜₜu = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 2.0
    end
    Δu = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        xa_minus_2 = x^a_minus_2
        ya_minus_2 = y^a_minus_2
        time_term = 4.0 + t * t

        return ((axa_minus_1 * xa_minus_2) * (ya - 1.0) +
                (xa - x) * (axa_minus_1 * ya_minus_2)) * time_term
    end
    ∂ₜₜz = @inline function (x, t)
        xa = x^a
        xa_minus_x = xa - x
        exp_term = 1.0 + exp(-x * x)

        return -2.0 * exp_term * xa_minus_x * (2.0 + cos(2.0 * t * xa_minus_x))
    end

    # Manufactured source terms
    f₁ = @inline (x, y, t) -> ∂ₜₜu(x, y, t) - α(t) * Δu(x, y, t) + f(u(x, y, t))
    f₂ = @inline (x, t) -> q₁ * ∂ₜₜz(x, t) + q₂ * r(x, t) + q₃ * z(x, t) +
                           q₄ * v(x, ymin, t)

    # Initial conditions
    u₀ = @inline function (x, y)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 4.0
    end
    ∂ₓu₀ = @inline function (x, y)
        ya = y^a
        xa_minus_1 = x^a_minus_1
        return (a * xa_minus_1 - 1.0) * (ya - 1.0) * 4.0
    end
    ∂ᵧu₀ = @inline function (x, y)
        xa = x^a
        ya_minus_1 = y^a_minus_1
        return (xa - x) * (a * ya_minus_1) * 4.0
    end
    v₀ = @inline (x, y) -> 0.0
    ∂ₓv₀ = @inline (x, y) -> 0.0
    ∂ᵧv₀ = @inline (x, y) -> 0.0
    z₀ = @inline x -> sinpi(x)
    r₀ = @inline x -> 0.0

    return PDEInputData(
        "example1_manufactured($a)",
        (0.0, ymin),
        (1.0, 1.0),
        q₁, q₂, q₃, q₄,
        α, f, df, g, ∂ₛg,
        u₀, ∂ₓu₀, ∂ᵧu₀,
        v₀, ∂ₓv₀, ∂ᵧv₀,
        z₀, r₀,
        f₁, f₂,
        u, v, z, r
    )
end

"""
    example1_zero_source(a::Float64=2.4) -> PDEInputData

Same configuration as `example1_manufactured` but with f₁ = f₂ = 0.
No analytical solution available.

# Arguments
- `a::Float64=2.4`: Smoothness parameter for initial conditions.

# Returns
`PDEInputData` with analytical solutions set to `nothing`.
"""
function example1_zero_source(a::Float64 = 2.4)
    # Precompute exponent-related constants
    a_minus_1 = a - 1.0

    # Physical parameters
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = @inline t -> 1.0 + exp(-t)

    f = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return s * s_abs3
    end
    df = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return 4.0 * s_abs3
    end

    g = @inline (x, s) -> (1.0 + exp(-x * x)) * muladd(2.0, s, sin(s))
    ∂ₛg = @inline (x, s) -> (1.0 + exp(-x * x)) * (2.0 + cos(s))

    # Zero source terms (pure initial-boundary value problem)
    f₁ = @inline (x, y, t) -> 0.0
    f₂ = @inline (x, t) -> 0.0

    # Initial conditions (same as manufactured case at t = 0)
    u₀ = @inline function (x, y)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 4.0
    end
    ∂ₓu₀ = @inline function (x, y)
        ya = y^a
        xa_minus_1 = x^a_minus_1
        return (a * xa_minus_1 - 1.0) * (ya - 1.0) * 4.0
    end
    ∂ᵧu₀ = @inline function (x, y)
        xa = x^a
        ya_minus_1 = y^a_minus_1
        return (xa - x) * (a * ya_minus_1) * 4.0
    end
    v₀ = @inline (x, y) -> 0.0
    ∂ₓv₀ = @inline (x, y) -> 0.0
    ∂ᵧv₀ = @inline (x, y) -> 0.0
    z₀ = @inline x -> sinpi(x)
    r₀ = @inline x -> 0.0

    return PDEInputData(
        "example1_zero_source($a)",
        (0.0, ymin),
        (1.0, 1.0),
        q₁, q₂, q₃, q₄,
        α, f, df, g, ∂ₛg,
        u₀, ∂ₓu₀, ∂ᵧu₀,
        v₀, ∂ₓv₀, ∂ᵧv₀,
        z₀, r₀,
        f₁, f₂,
        nothing, nothing, nothing, nothing
    )
end

# ============================================================================
# Example 2: Linear Coupling Test Case
# ============================================================================
"""
    example2_manufactured(a::Float64=2.4) -> PDEInputData

Manufactured solution with linear coupling ``g(x,s) = (1+e^{-x^2})s``:
```math
\\begin{alignat*}{2}
& u(x,y,t)   &&= (x^a - x)(y^a - 1)(4 + t^2), \\\\
& z(x,t)     &&= \\sin(\\pi x) - (1+e^{-x^2})(x^a-x)t^2,
\\end{alignat*}
```
where the acoustic displacement ``z(x,t)`` is obtained by integrating
```math
\\frac{∂z}{∂t}(x,t) = -\\frac{∂u}{∂y}(x,y_{\\min},t) + g\\!\\left(x,\\,\\frac{∂u}{∂t}(x,y_{\\min},t)\\right).
```

# Arguments
- `a::Float64=2.4`: Smoothness parameter controlling solution regularity.

# Returns
`PDEInputData` with analytical solution for convergence study.
"""
function example2_manufactured(a::Float64 = 2.4)
    # Precompute exponent-related constants
    a_minus_1 = a - 1.0
    a_minus_2 = a - 2.0
    axa_minus_1 = a * a_minus_1

    # Physical parameters
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = @inline t -> 1.0 + exp(-t)

    f = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return s * s_abs3
    end
    df = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return 4.0 * s_abs3
    end

    g = @inline (x, s) -> (1.0 + exp(-x * x)) * s
    ∂ₛg = @inline (x, s) -> 1.0 + exp(-x * x)

    # Analytical solutions
    u = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * (4.0 + t * t)
    end
    v = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * (2.0 * t)
    end

    z = @inline function (x, t)
        xa = x^a
        exp_term = 1.0 + exp(-x * x)
        return sinpi(x) - exp_term * (xa - x) * (t * t)
    end

    r = @inline function (x, t)
        xa = x^a
        exp_term = 1.0 + exp(-x * x)
        return -exp_term * (xa - x) * (2.0 * t)
    end

    # Auxiliary functions for manufactured source terms
    ∂ₜₜu = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 2.0
    end
    Δu = @inline function (x, y, t)
        xa = x^a
        ya = y^a
        xa_minus_2 = x^a_minus_2
        ya_minus_2 = y^a_minus_2
        time_term = 4.0 + t * t

        return ((axa_minus_1 * xa_minus_2) * (ya - 1.0) +
                (xa - x) * (axa_minus_1 * ya_minus_2)) * time_term
    end
    ∂ₜₜz = @inline function (x, t)
        xa = x^a
        exp_term = 1.0 + exp(-x * x)
        return -2.0 * exp_term * (xa - x)
    end

    # Manufactured source terms
    f₁ = @inline (x, y, t) -> ∂ₜₜu(x, y, t) - α(t) * Δu(x, y, t) + f(u(x, y, t))
    f₂ = @inline (x, t) -> q₁ * ∂ₜₜz(x, t) + q₂ * r(x, t) + q₃ * z(x, t) +
                           q₄ * v(x, ymin, t)

    # Initial conditions
    u₀ = @inline function (x, y)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 4.0
    end
    ∂ₓu₀ = @inline function (x, y)
        ya = y^a
        xa_minus_1 = x^a_minus_1
        return (a * xa_minus_1 - 1.0) * (ya - 1.0) * 4.0
    end
    ∂ᵧu₀ = @inline function (x, y)
        xa = x^a
        ya_minus_1 = y^a_minus_1
        return (xa - x) * (a * ya_minus_1) * 4.0
    end
    v₀ = @inline (x, y) -> 0.0
    ∂ₓv₀ = @inline (x, y) -> 0.0
    ∂ᵧv₀ = @inline (x, y) -> 0.0
    z₀ = @inline x -> sinpi(x)
    r₀ = @inline x -> 0.0

    return PDEInputData(
        "example2_manufactured($a)",
        (0.0, ymin),
        (1.0, 1.0),
        q₁, q₂, q₃, q₄,
        α, f, df, g, ∂ₛg,
        u₀, ∂ₓu₀, ∂ᵧu₀,
        v₀, ∂ₓv₀, ∂ᵧv₀,
        z₀, r₀,
        f₁, f₂,
        u, v, z, r
    )
end

"""
    example2_zero_source(a::Float64=2.4) -> PDEInputData

Same configuration as `example2_manufactured` but with f₁ = f₂ = 0.
No analytical solution available.

# Arguments
- `a::Float64=2.4`: Smoothness parameter for initial conditions.

# Returns
`PDEInputData` with analytical solutions set to `nothing`.
"""
function example2_zero_source(a::Float64 = 2.4)
    # Precompute exponent-related constants
    a_minus_1 = a - 1.0

    # Physical parameters
    q₁ = q₂ = q₃ = q₄ = 1.0
    ymin = 0.0

    # Coefficient functions
    α = @inline t -> 1.0 + exp(-t)

    f = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return s * s_abs3
    end
    df = @inline function (s)
        s_abs = abs(s)
        s_abs3 = s_abs * s_abs * s_abs
        return 4.0 * s_abs3
    end

    g = @inline (x, s) -> (1.0 + exp(-x * x)) * s
    ∂ₛg = @inline (x, s) -> 1.0 + exp(-x * x)

    # Zero source terms
    f₁ = @inline (x, y, t) -> 0.0
    f₂ = @inline (x, t) -> 0.0

    # Initial conditions
    u₀ = @inline function (x, y)
        xa = x^a
        ya = y^a
        return (xa - x) * (ya - 1.0) * 4.0
    end
    ∂ₓu₀ = @inline function (x, y)
        ya = y^a
        xa_minus_1 = x^a_minus_1
        return (a * xa_minus_1 - 1.0) * (ya - 1.0) * 4.0
    end
    ∂ᵧu₀ = @inline function (x, y)
        xa = x^a
        ya_minus_1 = y^a_minus_1
        return (xa - x) * (a * ya_minus_1) * 4.0
    end
    v₀ = @inline (x, y) -> 0.0
    ∂ₓv₀ = @inline (x, y) -> 0.0
    ∂ᵧv₀ = @inline (x, y) -> 0.0
    z₀ = @inline x -> sinpi(x)
    r₀ = @inline x -> 0.0

    return PDEInputData(
        "example2_zero_source($a)",
        (0.0, ymin),
        (1.0, 1.0),
        q₁, q₂, q₃, q₄,
        α, f, df, g, ∂ₛg,
        u₀, ∂ₓu₀, ∂ᵧu₀,
        v₀, ∂ₓv₀, ∂ᵧv₀,
        z₀, r₀,
        f₁, f₂,
        nothing, nothing, nothing, nothing
    )
end