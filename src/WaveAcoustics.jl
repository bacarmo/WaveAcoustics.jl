module WaveAcoustics

using StaticArrays: SVector, SMatrix, @SVector, @SMatrix
using GaussQuadrature: legendre
using Printf: @printf
using SparseArrays: sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: Symmetric, lmul!, cholesky, ldiv!, mul!
using FixedSizeArrays: FixedSizeArray

# Exports
export PDECommonData, PDEInputData
export manufactured_solution_case, zero_source_case
export example1_manufactured, example1_zero_source
export example2_manufactured, example2_zero_source
export Lagrange, Hermite
export pde_solve, CrankNicolson, CrankNicolsonLinearized, ConvergenceStudy, SolutionHistory,
       EnergyHistory
export convergence_test_coupled, convergence_test_spatial, convergence_test_temporal,
       print_convergence_table

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
include("pde_solve.jl")
include("crank_nicolson.jl")
include("crank_nicolson_linearized.jl")
include("convergence_test.jl")
end