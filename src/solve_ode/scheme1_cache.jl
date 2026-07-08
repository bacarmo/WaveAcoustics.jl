function build_cache(::Scheme1, matrices::SystemMatrices{T, I}) where {T, I}
    Scheme1Cache(matrices)
end

struct Scheme1Cache{T, I, S1, S2}
    # --- scratch vectors ---
    vec_m₁::Vector{T}
    vec_m₂_1::Vector{T}
    vec_m₂_2::Vector{T}
    # --- rhs vectors ---
    τFf₁::Vector{T}
    τFf₂::Vector{T}
    L::Vector{T}
    # --- nonlinearity vectors ---
    G::Vector{T}
    F::Vector{T}
    # --- midpoint unknowns and auxiliary quantities ---
    v̂ⁿ::Vector{T}
    d̂ⁿ::Vector{T}
    r̂ⁿ::Vector{T}
    # --- nonlinear system ---
    minusH::Vector{T}
    Q::Symmetric{T, SparseMatrixCSC{T, I}}
    JH_upper::Symmetric{T, SparseMatrixCSC{T, I}}
    JH::SparseMatrixCSC{T, I}
    # --- vectors for JH_sparse synchronization ---
    map_direct::Vector{I}
    map_mirror::Vector{I}
    # --- LinearSolve.LinearCache ---
    linsolve_m₁::S1
    linsolve_m₂::S2
end

function Scheme1Cache(matrices::SystemMatrices{T, I}) where {T, I}
    m₁ = size(matrices.K_m₁xm₁, 1)
    m₂ = size(matrices.M_m₂xm₂, 1)

    τFf₂ = ones(T, m₂)

    minusH = ones(T, m₁)
    Q = copy(matrices.K_m₁xm₁)
    JH_upper = copy(matrices.K_m₁xm₁)
    JH = sparse(JH_upper)::SparseMatrixCSC{T, I}

    map_direct, map_mirror = build_upper_to_full_maps11(JH, JH_upper)

    # --- Linear system 1 ---
    A1 = JH
    b1 = minusH
    prob1 = LS.LinearProblem(A1, b1)
    linsolve_m₁ = LS.init(prob1, LS.KLUFactorization(; reuse_symbolic = true, check_pattern = false))
    linsolve_m₁.isfresh = true
    LS.solve!(linsolve_m₁)

    # --- Linear system 2 ---
    A2 = matrices.M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}}
    b2 = τFf₂
    prob2 = LS.LinearProblem(A2, b2)
    linsolve_m₂ = LS.init(prob2, LS.CHOLMODFactorization())
    linsolve_m₂.isfresh = true
    LS.solve!(linsolve_m₂)

    return Scheme1Cache(
        # scratch vectors
        zeros(T, m₁), zeros(T, m₂), zeros(T, m₂),
        # rhs vectors
        zeros(T, m₁), τFf₂, zeros(T, m₁),
        # nonlinearity vectors
        zeros(T, m₂), zeros(T, m₁),
        # midpoint unknowns
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₂),
        # nonlinear system
        minusH, Q, JH_upper, JH,
        # vectors for JH synchronization
        map_direct, map_mirror,
        # LinearSolve.LinearCache
        linsolve_m₁, linsolve_m₂
    )
end