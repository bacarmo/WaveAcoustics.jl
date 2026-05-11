function build_cache(::CrankNicolson3, matrices)
    CrankNicolson3Cache(matrices)
end

"""
    CrankNicolson3Cache{T, I, TLS}

Pre-allocated workspace for the Crank-Nicolson Strategy 3 solver.

# Fields
- `vec_m₁_1`, `vec_m₁_2`: scratch vectors of length `m₁`.
- `vec_m₂_1`, `vec_m₂_2`, `vec_m₂_3`: scratch vectors of length `m₂`.
- `Q₁₁`: ``Q_{11} = M^{m_1\\times m_1} + \\frac{\\tau^2}{4}\\alpha^{n-\\frac{1}{2}}\\, K^{m_1\\times m_1}``.
- `Q̄₁₂`: ``Q₁₂ = [\\bar{Q}_{12};\\, 0^{(m_1-m_2)\\times m_2}]`` with ``\\bar{Q}_{12} = -\\frac{\\tau}{2}\\alpha^{n-\\frac{1}{2}}\\, M^{m_2\\times m_2}``.
- `Q̄₂₁`: ``Q₂₁ = [\\bar{Q}_{21}\\, 0^{m_2\\times (m_1-m_2)}]`` with ``\\bar{Q}_{21} = \\frac{\\tau}{2}q_4\\, M^{m_2\\times m_2}``.
- `Q₂₂`: ``Q_{22} = (q_1 + \\frac{\\tau}{2}q_2 + \\frac{\\tau^2}{4}q_3)M^{m_2\\times m_2}``.
- `τFf₁`: scaled load vector ``\\tau\\mathcal{F}^{m_1}(f_1^{n-1/2})``, length `m₁`.
- `τFf₂`: scaled load vector ``\\tau\\mathcal{F}^{m_2}(f_2^{n-1/2})``, length `m₂`.
- `L₁`: right-hand side of block 1, length `m₁`.
- `L₂`: right-hand side of block 2, length `m₂`.
- `d̂ⁿ`: midpoint displacement ``(d^n + d^{n-1})/2``, length `m₁`.
- `v̂ⁿ`: midpoint velocity ``(v^n + v^{n-1})/2``, length `m₁`.
- `X`: Newton iterate ``X = [v^n; r^n]``, length `m₁ + m₂`.
- `vⁿ`: view of `X[1:m₁]`.
- `rⁿ`: view of `X[m₁+1:m₁+m₂]`.
- `G`: nonlinear contribution ``\\tau\\alpha\\, G^{m_2}(\\hat{v}^n)``, length `m₂`.
- `F`: nonlinear contribution ``\\tau\\mathcal{F}^{m_1}(\\hat{d}^n)``, length `m₁`.
- `minusH`: residual ``-H(X)``, length `m₁ + m₂`.
- `JH₁₁`: (1,1) block of the Jacobian matrix ``JH``, same sparsity as `Q₁₁`.
- `JH`: full Jacobian matrix.
- `map_direct_11`, `map_mirror_11`: index maps from `JH₁₁.data.nzval` into `JH.nzval`.
- `map_direct_12`, `map_mirror_12`: index maps from `Q̄₁₂.data.nzval` into `JH.nzval`.
- `map_direct_21`, `map_mirror_21`: index maps from `Q̄₂₁.data.nzval` into `JH.nzval`.
- `map_direct_22`, `map_mirror_22`: index maps from `Q₂₂.data.nzval` into `JH.nzval`.
- `linsolve`: cached linear solver handle for ``JH\\,\\Delta X = -H``.
"""
struct CrankNicolson3Cache{T <: AbstractFloat, I <: Integer, TLS}
    vec_m₁_1::Vector{T}
    vec_m₁_2::Vector{T}
    vec_m₂_1::Vector{T}
    vec_m₂_2::Vector{T}
    vec_m₂_3::Vector{T}
    Q₁₁::Symmetric{T, SparseMatrixCSC{T, I}}
    Q̄₁₂::Symmetric{T, SparseMatrixCSC{T, I}}
    Q̄₂₁::Symmetric{T, SparseMatrixCSC{T, I}}
    Q₂₂::Symmetric{T, SparseMatrixCSC{T, I}}
    τFf₁::Vector{T}
    τFf₂::Vector{T}
    L₁::Vector{T}
    L₂::Vector{T}
    d̂ⁿ::Vector{T}
    v̂ⁿ::Vector{T}
    X::Vector{T}
    vⁿ::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    rⁿ::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    G::Vector{T}
    F::Vector{T}
    minusH::Vector{T}
    JH₁₁::Symmetric{T, SparseMatrixCSC{T, I}}
    JH::SparseMatrixCSC{T, I}
    map_direct_11::Vector{Int}
    map_mirror_11::Vector{Int}
    map_direct_21::Vector{Int}
    map_mirror_21::Vector{Int}
    map_direct_12::Vector{Int}
    map_mirror_12::Vector{Int}
    map_direct_22::Vector{Int}
    map_mirror_22::Vector{Int}
    linsolve::TLS
end

"""
    CrankNicolson3Cache(matrices::SystemMatrices{T, I})

Allocate workspace, build block index maps, and initialize the KLU factorization.
"""
function CrankNicolson3Cache(
        matrices::SystemMatrices{T, I}
) where {T <: AbstractFloat, I <: Integer}
    m₁ = size(matrices.M_m₁xm₁, 1)
    m₂ = size(matrices.M_m₂xm₂, 1)

    Q₁₁ = copy(matrices.M_m₁xm₁)
    Q̄₁₂ = copy(matrices.M_m₂xm₂)
    Q̄₂₁ = copy(matrices.M_m₂xm₂)
    Q₂₂ = copy(matrices.M_m₂xm₂)
    JH₁₁ = copy(matrices.M_m₁xm₁)

    Q₁₂ = [Q̄₁₂; spzeros(m₁ - m₂, m₂)]::SparseMatrixCSC{T, I}
    Q₂₁ = [Q̄₂₁ spzeros(m₂, m₁ - m₂)]::SparseMatrixCSC{T, I}
    JH = [Q₁₁ Q₁₂; Q₂₁ Q₂₂]::SparseMatrixCSC{T, I}
    minusH = zeros(T, m₁ + m₂)

    X = zeros(T, m₁ + m₂)
    vⁿ = view(X, 1:m₁)
    rⁿ = view(X, (m₁ + 1):(m₁ + m₂))

    map_direct_11, map_mirror_11 = build_maps(JH, Q₁₁)
    map_direct_21, map_mirror_21 = build_maps_21(JH, Q̄₂₁)
    map_direct_12, map_mirror_12 = build_maps_12(JH, Q̄₁₂)
    map_direct_22, map_mirror_22 = build_maps_22(JH, Q₂₂)

    prob = LS.LinearProblem(JH, minusH)
    linsolve = LS.init(
        prob, LS.KLUFactorization(; reuse_symbolic = true, check_pattern = false))

    return CrankNicolson3Cache(
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₂), zeros(T, m₂), zeros(T, m₂),
        Q₁₁, Q̄₁₂, Q̄₂₁, Q₂₂,
        zeros(T, m₁), zeros(T, m₂),
        zeros(T, m₁), zeros(T, m₂),
        zeros(T, m₁), zeros(T, m₁),
        X, vⁿ, rⁿ,
        zeros(T, m₂), zeros(T, m₁),
        minusH,
        JH₁₁, JH,
        map_direct_11, map_mirror_11,
        map_direct_21, map_mirror_21,
        map_direct_12, map_mirror_12,
        map_direct_22, map_mirror_22,
        linsolve
    )
end

"""
    build_maps_21(J, Q̄₂₁) -> (map_direct, map_mirror)

Map each entry of `Q̄₂₁.data.nzval` to its position in `J.nzval`, for the direct
entry and its symmetric counterpart within the (2,1) block.
"""
function build_maps_21(J, Q̄₂₁)
    m₂ = size(Q̄₂₁, 1)
    m₁ = size(J, 1) - m₂

    n = nnz(Q̄₂₁.data)
    map_direct = Vector{Int}(undef, n)
    map_mirror = Vector{Int}(undef, n)
    for j in 1:size(Q̄₂₁, 2)
        for kQ in nzrange(Q̄₂₁.data, j)
            i = Q̄₂₁.data.rowval[kQ]
            for kJ in nzrange(J, j)       # find J[i+m₁, j]
                if J.rowval[kJ] == i + m₁
                    map_direct[kQ] = kJ
                    break
                end
            end
            for kᵀJ in nzrange(J, i)      # find J[j+m₁, i] (symmetric entry)
                if J.rowval[kᵀJ] == j + m₁
                    map_mirror[kQ] = kᵀJ
                    break
                end
            end
        end
    end
    return map_direct, map_mirror
end

"""
    build_maps_12(J, Q̄₁₂) -> (map_direct, map_mirror)

Map each entry of `Q̄₁₂.data.nzval` to its position in `J.nzval`, for the direct
entry and its symmetric counterpart within the (1,2) block.
"""
function build_maps_12(J, Q̄₁₂)
    m₂ = size(Q̄₁₂, 1)
    m₁ = size(J, 1) - m₂

    n = nnz(Q̄₁₂.data)
    map_direct = Vector{Int}(undef, n)
    map_mirror = Vector{Int}(undef, n)
    for j in 1:size(Q̄₁₂, 2)
        for kQ in nzrange(Q̄₁₂.data, j)
            i = Q̄₁₂.data.rowval[kQ]
            for kJ in nzrange(J, j + m₁)  # find J[i, j+m₁]
                if J.rowval[kJ] == i
                    map_direct[kQ] = kJ
                    break
                end
            end
            for kᵀJ in nzrange(J, i + m₁) # find J[j, i+m₁]  (symmetric entry)
                if J.rowval[kᵀJ] == j
                    map_mirror[kQ] = kᵀJ
                    break
                end
            end
        end
    end
    return map_direct, map_mirror
end

"""
    build_maps_22(J, Q₂₂) -> (map_direct, map_mirror)

Map each entry of `Q₂₂.data.nzval` to its position in `J.nzval`, for the direct
entry and its symmetric counterpart within the (2,2) block.
"""
function build_maps_22(J, Q₂₂)
    m₂ = size(Q₂₂, 1)
    m₁ = size(J, 1) - m₂

    n = nnz(Q₂₂.data)
    map_direct = Vector{Int}(undef, n)
    map_mirror = Vector{Int}(undef, n)
    for j in 1:size(Q₂₂, 2)
        jJ = j + m₁
        for kQ in nzrange(Q₂₂.data, j)
            i = Q₂₂.data.rowval[kQ]
            iJ = i + m₁
            for kJ in nzrange(J, jJ)  # find J[i+m₁, j+m₁]
                if J.rowval[kJ] == iJ
                    map_direct[kQ] = kJ
                    break
                end
            end
            for kᵀJ in nzrange(J, iJ) # find J[j+m₁, i+m₁]  (symmetric entry)
                if J.rowval[kᵀJ] == jJ
                    map_mirror[kQ] = kᵀJ
                    break
                end
            end
        end
    end
    return map_direct, map_mirror
end