module WaveAcoustics

using StaticArrays: SVector, SMatrix, @SVector, @SMatrix
using GaussQuadrature: legendre

# Exports
export PDECommonData, PDEInputData
export manufactured_solution_case, zero_source_case
export example1_manufactured, example1_zero_source
export Lagrange, Hermite
export example1_manufactured, example1_zero_source
export pde_solve

# Includes
include("pde_inputdata.jl")
include("pde_solve.jl")
include("fe_families.jl")
include("boundary_conditions.jl")
include("basis_functions.jl")
include("mesh.jl")
include("dof_map.jl")
include("quadrature_setup.jl")

end