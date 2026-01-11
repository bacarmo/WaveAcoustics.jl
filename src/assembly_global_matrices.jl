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