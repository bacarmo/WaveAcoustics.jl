function build_cache(::CrankNicolson2, matrices)
    CrankNicolson2Cache(matrices)
end

"""
    CrankNicolson2Cache{T, I, TS, TM}
 
Pre-allocated workspace for the Crank–Nicolson Strategy 2 solver.
 
# Fields
- `vec_m₁_1`, `vec_m₁_2`, `vec_m₁_3`: scratch vectors of length `m₁`.
- `vec_m₂_1`, `vec_m₂_2`: scratch vectors of length `m₂`.
- `Ff₂`: load vector ``\\mathcal{F}^{m_2}(f_2^{n-1/2})``, length `m₂`.
- `M_m₁xm₁_vs2`: matrix ``2 * M_m₁xm₁``
- `Q`: assembled system matrix ``Q \\in \\mathbb{R}^{m_1 \\times m_1}``.
- `L`: right-hand side vector, length `m₁`.
- `d̂ⁿ`: midpoint displacement, length `m₁`.
- `r̂ⁿ`: midpoint acoustic, length `m₂`.
- `X`: Newton iterate ``X=\\hat{v}^n``, length `m₁`.
- `minusH`: residual ``-H(X)``, length `m₁`.
- `JH_upper`: Jacobian upper-triangular assembly workspace.
- `JH`: Jacobian in full sparse format, passed to the KLU solver.
- `map_direct`: maps each entry `k` of `JH_upper.data.nzval` to its position in `JH.nzval`.
- `map_mirror`: maps each entry `k` of `JH_upper.data.nzval` to the mirror position in `JH.nzval`.
- `linsolve`: cached linear solver handle for ``JH \\, \\Delta X = -H``.
- `M_m₂xm₂_chol`: Cholesky factorization of ``M^{m_2 \\times m_2}``.
"""
struct CrankNicolson2Cache{T <: AbstractFloat, I <: Integer, TS, TM}
    vec_m₁_1::Vector{T}
    vec_m₁_2::Vector{T}
    vec_m₁_3::Vector{T}
    vec_m₂_1::Vector{T}
    vec_m₂_2::Vector{T}
    Ff₂::Vector{T}
    M_m₁xm₁_vs2::Symmetric{T, SparseMatrixCSC{T, I}}
    Q::Symmetric{T, SparseMatrixCSC{T, I}}
    L::Vector{T}
    d̂ⁿ::Vector{T}
    r̂ⁿ::Vector{T}
    X::Vector{T}
    minusH::Vector{T}
    JH_upper::Symmetric{T, SparseMatrixCSC{T, I}}
    JH::SparseMatrixCSC{T, I}
    map_direct::Vector{Int}
    map_mirror::Vector{Int}
    linsolve::TS
    M_m₂xm₂_chol::TM
end

"""
    CrankNicolson2Cache(matrices::SystemMatrices{T, I})
 
Allocate workspace, build block index maps, initialize the KLU factorization, and pre-compute the Cholesky factorization of ``M^{m_2 \\times m_2}``.
"""
function CrankNicolson2Cache(
        matrices::SystemMatrices{T, I}
) where {T <: AbstractFloat, I <: Integer}
    m₁ = size(matrices.M_m₁xm₁, 1)
    m₂ = size(matrices.M_m₂xm₂, 1)

    M_m₁xm₁_vs2 = 2 * matrices.M_m₁xm₁
    Q = similar(matrices.M_m₁xm₁)

    minusH = zeros(T, m₁)
    JH_upper = similar(matrices.M_m₁xm₁)
    JH = sparse(matrices.M_m₁xm₁)

    map_direct, map_mirror = build_maps(JH, JH_upper)

    prob = LS.LinearProblem(JH, minusH)
    linsolve = LS.init(
        prob, LS.KLUFactorization(; reuse_symbolic = true, check_pattern = false))

    return CrankNicolson2Cache(
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₁),
        zeros(T, m₂), zeros(T, m₂), zeros(T, m₂),
        M_m₁xm₁_vs2, Q, zeros(T, m₁),
        zeros(T, m₁), zeros(T, m₂), zeros(T, m₁),
        minusH, JH_upper, JH,
        map_direct, map_mirror,
        linsolve,
        cholesky(matrices.M_m₂xm₂)
    )
end