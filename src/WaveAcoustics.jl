module WaveAcoustics

using StaticArrays: SVector, SMatrix
using GaussQuadrature: legendre
using Printf: @printf
using SparseArrays: sparse, spzeros, SparseMatrixCSC, nnz, nzrange
using LinearAlgebra: Symmetric, lmul!, cholesky, ldiv!, mul!, dot
import LinearSolve as LS

# Exports
export PDEInputData, example1_manufactured, example1_zero_source, example2_manufactured,
       example2_zero_source
export Lagrange, Hermite
export FEMState, L2ErrorCallback, SolutionCallback
export CrankNicolson1
export pde_solve
export convergence_study_coupled, convergence_study_spatial, convergence_study_temporal,
       print_convergence_table

# Includes
include("pde_inputdata.jl")
include("mesh/mesh.jl")
include("fem/fem.jl")
include("callbacks/callbacks.jl")
include("assembly/assembly.jl")
include("initial_solution.jl")
include("ode_solvers/ode_solvers.jl")
include("pde.solve.jl")
include("postprocessing/postprocessing.jl")
end