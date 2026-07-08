module WaveAcoustics

using FEM: AbstractFEBasis, Lagrange, LeftRightTop, AllSides, DOFMap,
           assembly_local_matrix_ϕxϕ, assembly_local_matrix_∇ϕx∇ϕ, assembly_global_matrix,
           basis_functions, basis_functions_derivatives, projection_H01!, projection_L2!,
           error_L2, assembly_rhs_1d!, assembly_rhs_2d!, assembly_nonlinearity_G!,
           assembly_nonlinearity_F!, assembly_global_matrix_DG, assembly_global_matrix_DF,
           scatter_symmetric!, build_upper_to_full_maps11
using SparseArrays: SparseMatrixCSC, sparse, nnz, nzrange
using LinearAlgebra: Symmetric, cholesky, mul!, dot, ldiv!
using StaticArrays: SVector, SMatrix
using GaussQuadrature: legendre
import LinearSolve as LS
using Printf: @sprintf
using Dates
using InteractiveUtils

# Exports
export PDEInputData, example1_manufactured, example1_zero_source, example2_manufactured,
       example2_zero_source, example3_manufactured, example3_zero_source
export AbstractFEBasis, Lagrange
export L2ErrorCallback, SolutionCallback
export Scheme1, Scheme2, Scheme3
export solve_pde
export run_convergence_study, run_cases, ConvergenceResults

# Includes
include("input_data.jl")
include("utilities.jl")
include("callbacks.jl")
include("solve_ode/solve_ode.jl")
include("solve_pde.jl")
include("run_convergence_studies.jl")
end