module WaveAcoustics

using StaticArrays: SVector

# Exports
export CartesianMesh
export Lagrange, Hermite
export LeftRight, LeftRightBottomTop, LeftRightTop

# Includes
include("fe_families.jl")
include("boundary_conditions.jl")
include("mesh.jl")
include("dof_map.jl")

end