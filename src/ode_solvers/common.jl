# ==============================================================================
# build_maps
# ==============================================================================
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

# ==============================================================================
# solve_newton_linear_system!
# ==============================================================================
"""
    solve_newton_linear_system!(linsolve)
 
Solve the Newton linear system ``JH \\cdot \\Delta X = -H`` using the cached
KLU factorization. Sets `linsolve.isfresh = true` to trigger refactorization.
 
# Arguments
- `linsolve`: `LinearSolve.LinearCache` initialized with `KLUFactorization`.
"""
function solve_newton_linear_system!(linsolve)
    linsolve.isfresh = true
    LS.solve!(linsolve)
    return nothing
end

# ==============================================================================
# sync_JH!
# ==============================================================================
"""
    sync_JH!(JH, JH_upper, map_direct, map_direct)

Scatter `JH_upper.data.nzval` into both triangles of `JH.nzval`
using the pre-computed `map_direct` and `map_mirror` index maps.
"""
function sync_JH!(
        JH::SparseMatrixCSC{T, I},
        JH_upper::Symmetric{T, SparseMatrixCSC{T, I}},
        map_direct::Vector{Int},
        map_mirror::Vector{Int}
) where {T, I}
    nzval_upper = JH_upper.data.nzval
    nzval_JH = JH.nzval
    @inbounds for k in eachindex(nzval_upper)
        nzval_JH[map_direct[k]] = nzval_upper[k]
        nzval_JH[map_mirror[k]] = nzval_upper[k]
    end
    return nothing
end

# ==============================================================================
# compute_r̂ⁿ!
# ==============================================================================

"""
    compute_r̂ⁿ!(r̂ⁿ, v̂ⁿ, rⁿ⁻¹, zⁿ⁻¹, Ff₂, csts, M_m₂xm₂_chol, vec_m₂)

Compute ``\\hat{r}^n`` in-place using a closed-form expression
```math
\\hat{r}^n =
- \\frac{\\tau q_4}{q_5}\\hat{v}^n_{1:m_2}
+ \\frac{2q_1}{q_5}r^{n-1}
- \\frac{\\tau q_3}{q_5}z^{n-1}
+ \\frac{\\tau}{q_5}\\left(M^{m_2 \\times m_2}\\right)^{-1}
  \\mathcal{F}^{m_2}(f_3^{n-\\frac{1}{2}}).
```

# Notes
- The system ``M^{m_2 \\times m_2} x = Ff₂`` is solved using the precomputed Cholesky factorization `M_m₂xm₂_chol`.
- The vector `vec_m₂` is used as workspace to store the solution of this system.
- The constants are provided as ``csts = [-\\tau q_4/q_5,\\; 2q_1/q_5,\\; -\\tau q_3/q_5,\\; \\tau/q_5]``.
"""
function compute_r̂ⁿ!(
        r̂ⁿ::Vector{T},
        v̂ⁿ::Vector{T},
        rⁿ⁻¹::Vector{T},
        zⁿ⁻¹::Vector{T},
        Ff₂::Vector{T},
        csts::SVector{4, T},
        M_m₂xm₂_chol,
        vec_m₂::Vector{T}
) where {T}
    c1, c2, c3, c4 = csts

    ldiv!(vec_m₂, M_m₂xm₂_chol, Ff₂)

    @inbounds for i in eachindex(r̂ⁿ)
        r̂ⁿ[i] = c1 * v̂ⁿ[i] + c2 * rⁿ⁻¹[i] + c3 * zⁿ⁻¹[i] + c4 * vec_m₂[i]
    end

    return nothing
end

# ==============================================================================
# update_state!
# ==============================================================================

"""
    update_state!(state, v̂ⁿ, r̂ⁿ, τ)
 
Advance `state` from time level ``n-1`` to ``n``:
```math
\\begin{aligned}
v^n &= 2\\hat{v}^n - v^{n-1}, \\\\
r^n &= 2\\hat{r}^n - r^{n-1}, \\\\
d^n &= \\tau\\hat{v}^n + d^{n-1}, \\\\
z^n &= \\tau\\hat{r}^n + z^{n-1}.
\\end{aligned}
```
Also increments `state.n` by 1 and advances `state.t` by ``\\tau``.
"""
function update_state!(
        state::FEMState{T},
        v̂ⁿ::Vector{T},
        r̂ⁿ::Vector{T},
        τ::T
) where {T}
    state.n += 1
    state.t += τ
    @. state.v = muladd(2, v̂ⁿ, -state.v)
    @. state.r = muladd(2, r̂ⁿ, -state.r)
    @. state.d = muladd(τ, v̂ⁿ, state.d)
    @. state.z = muladd(τ, r̂ⁿ, state.z)

    return nothing
end