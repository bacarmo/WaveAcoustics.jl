"""
    _muladd!(A, cst, B, C)

Compute in-place `A = cst * B + C` by iterating directly over the nonzero values.
Assumes `A`, `B`, and `C` share the same sparsity pattern.
"""
function _muladd!(
        A::Symmetric{T, SparseMatrixCSC{T, I}},
        cst::T,
        B::Symmetric{T, SparseMatrixCSC{T, I}},
        C::Symmetric{T, SparseMatrixCSC{T, I}}) where {T, I}
    nzval_A = A.data.nzval
    nzval_B = B.data.nzval
    nzval_C = C.data.nzval

    @. nzval_A = muladd(cst, nzval_B, nzval_C)

    return nothing
end

function update_state!(
        state::State{T},
        v̂ⁿ::Vector{T},
        r̂ⁿ::Vector{T},
        τ::T
) where {T}
    state.n += 1
    state.t += τ

    @. state.v = muladd(2, v̂ⁿ, -state.v)
    @. state.d = muladd(τ, v̂ⁿ, state.d)

    @. state.r = muladd(2, r̂ⁿ, -state.r)
    @. state.z = muladd(τ, r̂ⁿ, state.z)

    return nothing
end

# ==============================================================================
# compute_Q!
# ==============================================================================
"""
    compute_Q!(Q, K_m₁xm₁, _2M_m₁xm₁, M_m₂xm₂, cst₁, cst₂)

Compute in-place

    Q = cst₁*K_m₁xm₁ + _2M_m₁xm₁ + cst₂*[M_m₂xm₂ 0_m₂x(m₁-m₂); 0_(m₁-m₂)xm₂ 0_m₂xm₂]

where `cst₁ = τ²⋅α(tₙ₋₁/₂)/2` and `cst₂ = τ²⋅q₄⋅α(tₙ₋₁/₂)/q₅`
"""
function compute_Q!(
        Q::Symmetric{T, SparseMatrixCSC{T, I}},
        K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        _2M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}},
        M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}},
        cst₁::T, cst₂::T) where {T, I}
    vec1 = Q.data.nzval
    vec2 = K_m₁xm₁.data.nzval
    vec3 = _2M_m₁xm₁.data.nzval
    vec4 = M_m₂xm₂.data.nzval

    nnz_m₁ = length(vec1)
    nnz_m₂ = length(vec4)

    for i in 1:nnz_m₂
        tmp = muladd(cst₁, vec2[i], vec3[i])
        vec1[i] = muladd(cst₂, vec4[i], tmp)
    end
    for i in (nnz_m₂ + 1):nnz_m₁
        vec1[i] = muladd(cst₁, vec2[i], vec3[i])
    end

    return nothing
end

# ==============================================================================
# solve_r̂ⁿ!
# ==============================================================================
"""
    compute_r̂ⁿ!(r̂ⁿ, v̂ⁿ, rⁿ⁻¹, zⁿ⁻¹, sol, cst1, cst2, cst3, cst4)

Compute in-place 

    r̂ⁿ = cst1⋅v̂ⁿ[1:m₂] + cst2⋅rⁿ⁻¹ + cst3⋅zⁿ⁻¹ + cst4⋅sol

where `cst1 = -τ⋅q₄/q₅`, `cst2 = 2⋅q₁/q₅`, `cst3 = -τ⋅q₃/q₅`, `cst4 = 1/q₅`, and `sol = inv(M_m₂xm₂)⋅τFf₂`
"""
function compute_r̂ⁿ!(r̂ⁿ, v̂ⁿ, rⁿ⁻¹, zⁿ⁻¹, sol, cst1, cst2, cst3, cst4)
    for i in eachindex(r̂ⁿ)
        r̂ⁿ[i] = cst1 * v̂ⁿ[i] + cst2 * rⁿ⁻¹[i] + cst3 * zⁿ⁻¹[i] + cst4 * sol[i]
    end
    return nothing
end