"""
    DirichletSides

Abstract supertype for domain sides with Dirichlet BCs.
"""
abstract type DirichletSides end

struct LeftRight <: DirichletSides end              # 1D: left and right
struct LeftRightBottomTop <: DirichletSides end     # 2D: all 4 sides
struct LeftRightTop <: DirichletSides end           # 2D: left, right, top