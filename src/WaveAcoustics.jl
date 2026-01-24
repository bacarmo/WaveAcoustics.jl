module WaveAcoustics

using StaticArrays: SVector, SMatrix, @SVector, @SMatrix
using GaussQuadrature: legendre
using Printf: @printf
using SparseArrays: sparse, spzeros
using LinearAlgebra: Symmetric, lmul!, cholesky, ldiv!, mul!
using FixedSizeArrays: FixedSizeArray

# Exports
export PDECommonData, PDEInputData
export manufactured_solution_case, zero_source_case
export example1_manufactured, example1_zero_source
export Lagrange, Hermite
export pde_solve
export convergence_test, print_convergence_table

# Includes
include("fe_families.jl")
include("boundary_conditions.jl")
include("basis_functions.jl")
include("mesh.jl")
include("dof_map.jl")
include("quadrature_setup.jl")
include("assembly_local_matrices.jl")
include("assembly_global_matrices.jl")
include("assembly_global_vectors.jl")
include("pde_inputdata.jl")
include("initial_solution.jl")
include("error_norms.jl")
include("crank_nicolson.jl")
include("pde_solve.jl")
include("convergence_test.jl")
end