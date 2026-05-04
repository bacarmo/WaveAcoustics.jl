"""
    DirichletSides

Abstract supertype for domain sides with Dirichlet BCs.
"""
abstract type DirichletSides end

"""
    LeftRight <: DirichletSides

Dirichlet boundary conditions imposed on the left and right sides of a 1D domain.
"""
struct LeftRight <: DirichletSides end

"""
    LeftRightBottomTop <: DirichletSides

Dirichlet boundary conditions imposed on all four sides
(left, right, bottom, and top) of a 2D domain.
"""
struct LeftRightBottomTop <: DirichletSides end

"""
    LeftRightTop <: DirichletSides

Dirichlet boundary conditions imposed on the left, right, and top sides
of a 2D domain.
"""
struct LeftRightTop <: DirichletSides end