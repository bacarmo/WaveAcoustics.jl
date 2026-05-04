function build_cache(::CrankNicolson1, matrices)
    CrankNicolson1Cache(matrices)
end

"""
    CrankNicolson1Cache{T, I, TS, TM}
 
Pre-allocated workspace for the Crank–Nicolson Strategy 1 solver.
 
# Fields
- `vec_m₁_1`, `vec_m₁_2`, `vec_m₁_3`: scratch vectors of length `m₁`.
- `vec_m₂_1`, `vec_m₂_2`: scratch vectors of length `m₂`.
- `Ff₂`: load vector ``\\mathcal{F}^{m_2}(f_2^{n-1/2})``, length `m₂`.
- `Q`: assembled system matrix ``Q \\in \\mathbb{R}^{m_1 \\times m_1}``.
- `L`: right-hand side vector, length `m₁`.
- `d̂ⁿ`: midpoint displacement, length `m₁`.
- `v̂ⁿ`: midpoint velocity, length `m₁`.
- `r̂ⁿ`: midpoint acoustic, length `m₂`.
- `X`: Newton iterate ``X \\approx v^n``, length `m₁`.
- `minusH`: residual ``-H(X)``, length `m₁`.
- `JH_upper`: Jacobian upper-triangular assembly workspace.
- `JH`: Jacobian in full sparse format, passed to the KLU solver.
- `map_direct`: maps each entry `k` of `JH_upper.data.nzval` to its position in `JH.nzval`.
- `map_mirror`: maps each entry `k` of `JH_upper.data.nzval` to the mirror position in `JH.nzval`.
- `linsolve`: cached linear solver handle for ``JH \\, \\Delta X = -H``.
- `M_m₂xm₂_chol`: Cholesky factorization of ``M^{m_2 \\times m_2}``.
"""
struct CrankNicolson1Cache{T <: AbstractFloat, I <: Integer, TS, TM}
    vec_m₁_1::Vector{T}
    vec_m₁_2::Vector{T}
    vec_m₁_3::Vector{T}
    vec_m₂_1::Vector{T}
    vec_m₂_2::Vector{T}
    Ff₂::Vector{T}
    Q::Symmetric{T, SparseMatrixCSC{T, I}}
    L::Vector{T}
    d̂ⁿ::Vector{T}
    v̂ⁿ::Vector{T}
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
    CrankNicolson1Cache(matrices::SystemMatrices{T, I})
 
Allocate all working arrays, build the `JH_upper` → `JH` index maps,
initialize the KLU factorization handle, and pre-compute the Cholesky
factorization of ``M^{m_2 \\times m_2}``.
"""
function CrankNicolson1Cache(
        matrices::SystemMatrices{T, I}
) where {T <: AbstractFloat, I <: Integer}
    m₁ = size(matrices.M_m₁xm₁, 1)
    m₂ = size(matrices.M_m₂xm₂, 1)

    Q = similar(matrices.M_m₁xm₁)

    minusH = zeros(T, m₁)
    JH_upper = similar(matrices.M_m₁xm₁)
    JH = sparse(matrices.M_m₁xm₁)

    map_direct, map_mirror = build_maps(JH, JH_upper)

    prob = LS.LinearProblem(JH, minusH)
    linsolve = LS.init(
        prob, LS.KLUFactorization(; reuse_symbolic = true, check_pattern = false))

    return CrankNicolson1Cache(
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₁),
        zeros(T, m₂), zeros(T, m₂), zeros(T, m₂),
        Q, zeros(T, m₁),
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₂), zeros(T, m₁),
        minusH, JH_upper, JH,
        map_direct, map_mirror,
        linsolve,
        cholesky(matrices.M_m₂xm₂)
    )
end

"""
    build_maps(JH, JH_upper) -> (map_direct, map_mirror)

Build index maps from the upper-triangular storage of `JH_upper.data` to `JH.nzval`.
For the `k`-th nonzero `(i, j)` of `JH_upper.data`:

- `map_direct[k]`: index into `JH.nzval` for `JH[i, j]`.
- `map_mirror[k]`: index into `JH.nzval` for `JH[j, i]`.

Diagonal entries satisfy `map_direct[k] == map_mirror[k]`.

# Arguments
- `JH::SparseMatrixCSC{T, I}`: Jacobian matrix with full sparsity pattern.
- `JH_upper::Symmetric{T, SparseMatrixCSC{T, I}}`: symmetric matrix (upper-triangle storage).
"""
function build_maps(
        JH::SparseMatrixCSC{T, I},
        JH_upper::Symmetric{T, SparseMatrixCSC{T, I}}
) where {T, I}
    M = JH_upper.data
    nnz_M = nnz(M)
    map_direct = Vector{Int}(undef, nnz_M)
    map_mirror = Vector{Int}(undef, nnz_M)
    for j in 1:size(M, 2)            # Iterate over columns of M
        for kM in nzrange(M, j)      # Iterate over nonzeros of M[:,j]
            i = M.rowval[kM]
            for kJ in nzrange(JH, j) # Iterate over nonzeros of JH[:,j] and find position of JH[i,j]
                if JH.rowval[kJ] == i
                    map_direct[kM] = kJ
                    break
                end
            end
            for kᵀJ in nzrange(JH, i) # Iterate over nonzeros of JH[:,i] and find position of JH[j,i]
                if JH.rowval[kᵀJ] == j
                    map_mirror[kM] = kᵀJ
                    break
                end
            end
        end
    end
    return map_direct, map_mirror
end