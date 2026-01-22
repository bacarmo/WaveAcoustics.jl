"""
    assembly_global_matrix(local_matrix, dof_map) -> SparseMatrixCSC

Assemble global FEM matrix from element-local matrix.

# Arguments
- `local_matrix::SMatrix{N,N,T}`: Element matrix (N×N)
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs

# Returns
- `SparseMatrixCSC{T,I}`: Assembled global matrix (m×m), excluding DOFs > m
"""
function assembly_global_matrix(
        local_matrix::SMatrix{N, N, T},
        dof_map::DOFMap{<:AbstractVector, I}
) where {N, T, I <: Integer}
    m = dof_map.m
    EQoLG = dof_map.EQoLG
    Ne = length(EQoLG)

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    capacity = Ne * N * N
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)

    idx = 0
    @inbounds for e in 1:Ne
        global_indices = EQoLG[e]
        for b in 1:N
            jb = global_indices[b]
            jb > m && continue
            for a in 1:N
                ia = global_indices[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b]
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    return sparse(I_rows, J_cols, V_vals, m, m)
end

"""
    assembly_global_matrix(local_matrix, dof_map) -> Symmetric{T, SparseMatrixCSC{T,I}}

Assemble global symmetric FEM matrix from element-local symmetric matrix.
Only stores upper triangle, reducing memory and assembly time.

# Arguments
- `local_matrix::Symmetric{T, <:SMatrix{N,N,T}}`: Symmetric element matrix (N×N)
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m×m)

# Assumptions
Assumes `a ≤ b ⇒ ia ≤ jb` for all local indices, where `ia = EQoLG[e][a]` and 
`jb = EQoLG[e][b]`. This condition holds for structured meshes with consistent 
left-to-right, bottom-to-top numbering.
"""
function assembly_global_matrix(
        local_matrix::Symmetric{T, <:SMatrix{N, N, T}},
        dof_map::DOFMap{<:AbstractVector, I}
) where {N, T, I <: Integer}
    m = dof_map.m
    EQoLG = dof_map.EQoLG
    Ne = length(EQoLG)

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (N * (N + 1)) ÷ 2
    capacity = Ne * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)

    idx = 0
    @inbounds for e in 1:Ne
        global_indices = EQoLG[e]
        for b in 1:N
            jb = global_indices[b]
            jb > m && continue
            for a in 1:b  # Upper triangle: a ≤ b
                ia = global_indices[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b]
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    K_upper = sparse(I_rows, J_cols, V_vals, m, m)
    return Symmetric(K_upper, :U)
end

"""
    assembly_global_matrix_DG(scale, ∂ₛg, v, mesh, dof_map, quad)

DGᵢⱼ = scale * ∫ ϕᵢ(x) * ϕⱼ(x) * ∂ₛg(x, Vₕ(x)) dx over Ω ⊂ ℜ, with Vₕ(x) = Σ v[k] ϕₖ(x).

# Arguments
- `scale::T`: Scaling factor applied to final result
- `∂ₛg::Fun`: Callable ∂ₛg(x, v) → T
- `v::AbstractVector{T}`: Coefficient vector for Vₕ, length `dof_map.m`
- `mesh::CartesianMesh{1,I}`: 1D Cartesian mesh
- `dof_map::DOFMap`: DOF mapping with `EQoLG` connectivity and `m` free DOFs
- `quad::QuadratureSetup`: Precomputed quadrature data

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m×m)
"""
function assembly_global_matrix_DG(
        scale::T,
        ∂ₛg::Fun,
        v::AbstractVector{T},
        mesh::CartesianMesh{1, I},
        dof_map::DOFMap,
        quad::QuadratureSetup
) where {T <: AbstractFloat, I <: Integer, Fun}
    Δx = mesh.Δx[1]
    Nx = mesh.Nx[1]
    m = dof_map.m
    num_local_dof = length(quad.ϕP[1])

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (num_local_dof * (num_local_dof + 1)) ÷ 2
    capacity = Nx * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)
    local_matrix = FixedSizeArray{T}(undef, num_local_dof, num_local_dof)

    scale_jacobian = scale * (Δx / 2)

    idx = 0
    for e in 1:Nx
        eq = dof_map.EQoLG[e]
        xeP = @. muladd(e - 1, Δx, quad.xP)
        assembly_local_matrix_DG!(local_matrix, ∂ₛg, v, m, eq, xeP, quad)

        for b in 1:num_local_dof
            jb = eq[b]
            jb > m && continue

            for a in 1:b # Upper triangle: a ≤ b
                ia = eq[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b] * scale_jacobian
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    DG_upper = sparse(I_rows, J_cols, V_vals, m, m)

    return Symmetric(DG_upper, :U)
end