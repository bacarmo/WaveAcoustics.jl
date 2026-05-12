# ==============================================================================
# SystemMatrices
# ==============================================================================

"""
    SystemMatrices{T, I}

Global FEM matrices for the coupled wave-acoustic system.
All matrices are assembled once before the time loop and remain constant
throughout the time integration.

# Type Parameters
- `T <: AbstractFloat`: floating-point precision (e.g. `Float64`).
- `I <: Integer`: integer type used by the sparse matrices (e.g. `Int64`).

# Fields
- `M_m₁xm₁`: mass matrix for the ``m_1``-space (wave field).
- `M_m₂xm₂`: mass matrix for the ``m_2``-space (acoustic field).
- `K_m₁xm₁`: stiffness matrix for the ``m_1``-space.
"""
struct SystemMatrices{T <: AbstractFloat, I <: Integer}
    M_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}}
    M_m₂xm₂::Symmetric{T, SparseMatrixCSC{T, I}}
    K_m₁xm₁::Symmetric{T, SparseMatrixCSC{T, I}}
end

"""
    SystemMatrices(mesh1D, mesh2D, fe1D, fe2D, dof_map_m₁, dof_map_m₂)

Construct the global FEM matrices for the coupled thermo-wave-acoustic system.
See [`SystemMatrices`](@ref) for a description of the assembled fields.

# Arguments
- `mesh1D::CartesianMesh{1}`: 1D Cartesian mesh (boundary ``\\Gamma_1``).
- `mesh2D::CartesianMesh{2}`: 2D Cartesian mesh (domain ``\\Omega``).
- `fe1D::DimensionalFEFamily`: 1D finite element family.
- `fe2D::DimensionalFEFamily`: 2D finite element family.
- `dof_map_m₁::DOFMap`: DOF map for the ``m_1``-space (left, right, and top boundaries prescribed).
- `dof_map_m₂::DOFMap`: DOF map for the ``m_2``-space (left and right boundaries prescribed).
"""
function SystemMatrices(
        mesh1D::CartesianMesh{1},
        mesh2D::CartesianMesh{2},
        fe1D::F1,
        fe2D::F2,
        dof_map_m₁::DOFMap,
        dof_map_m₂::DOFMap
) where {F1 <: DimensionalFEFamily, F2 <: DimensionalFEFamily}
    # Local matrices — 2D
    Me_m₁xm₁ = Symmetric(assembly_local_matrix_ϕxϕ(mesh2D, fe2D))
    Ke_m₁xm₁ = Symmetric(assembly_local_matrix_∇ϕx∇ϕ(mesh2D, fe2D))

    # Local matrices — 1D
    Me_m₂xm₂ = Symmetric(assembly_local_matrix_ϕxϕ(mesh1D, fe1D))

    return SystemMatrices(
        # Mass matrices
        assembly_global_matrix(Me_m₁xm₁, dof_map_m₁), # M_m₁xm₁
        assembly_global_matrix(Me_m₂xm₂, dof_map_m₂), # M_m₂xm₂
        # Stiffness matrices
        assembly_global_matrix(Ke_m₁xm₁, dof_map_m₁)  # K_m₁xm₁
    )
end