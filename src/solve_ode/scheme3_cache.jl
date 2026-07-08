function build_cache(::Scheme3, matrices::SystemMatrices{T, I}) where {T, I}
    Scheme3Cache(matrices)
end

struct Scheme3Cache{T, I, S1, S2}
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
    vᵃⁿ::Vector{T}
    vⁿ⁻²::Vector{T}
    dᵃⁿ::Vector{T}
    dⁿ⁻²::Vector{T}
    r̂ⁿ::Vector{T}
    # --- system ---
    Q_upper::Symmetric{T, SparseMatrixCSC{T, I}}
    Q::SparseMatrixCSC{T, I}
    # --- vectors for Q synchronization ---
    map_direct::Vector{I}
    map_mirror::Vector{I}
    # --- LinearSolve.LinearCache ---
    linsolve_m₁::S1
    linsolve_m₂::S2
end

function Scheme3Cache(matrices::SystemMatrices{T, I}) where {T, I}
    m₁ = size(matrices.K_m₁xm₁, 1)
    m₂ = size(matrices.M_m₂xm₂, 1)

    τFf₂ = ones(T, m₂)
    L = ones(T, m₁)

    Q_upper = copy(matrices.K_m₁xm₁)
    Q = sparse(Q_upper)::SparseMatrixCSC{T, I}

    map_direct, map_mirror = build_upper_to_full_maps11(Q, Q_upper)

    # --- Linear system 1 ---
    A1 = Q
    b1 = L
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

    return Scheme3Cache(
        # scratch vectors
        zeros(T, m₁), zeros(T, m₂), zeros(T, m₂),
        # rhs vectors
        zeros(T, m₁), τFf₂, L,
        # nonlinearity vectors
        zeros(T, m₂), zeros(T, m₁),
        # midpoint unknowns
        zeros(T, m₁), zeros(T, m₁), zeros(T, m₁),
        zeros(T, m₁), zeros(T, m₁),
        zeros(T, m₂),
        # nonlinear system
        Q_upper, Q,
        # vectors for JH synchronization
        map_direct, map_mirror,
        # LinearSolve.LinearCache 
        linsolve_m₁, linsolve_m₂
    )
end