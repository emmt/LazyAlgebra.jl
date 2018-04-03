#
# LazyAlgebra.jl -
#
# A simple linear algebra system.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

module LazyAlgebra

isdefined(Base, :apply) && import Base: apply

export
    apply,
    apply!,
    contents,
    input_type,
    input_eltype,
    input_size,
    input_ndims,
    output_type,
    output_eltype,
    output_size,
    output_ndims,
    lineartype,
    is_applicable_in_place,
    conjgrad!,
    vcombine!,
    vcombine,
    vcopy!,
    vcopy,
    vcreate,
    vdot,
    vfill!,
    vnorm1,
    vnorm2,
    vnorminf,
    vproduct!,
    vproduct,
    vscale!,
    vscale,
    vswap!,
    vupdate!,
    vzero!,
    Adjoint,
    AdjointInverse,
    Direct,
    GeneralMatrix,
    HalfHessian,
    Identity,
    Inverse,
    InverseAdjoint,
    Linear,
    LinearMapping,
    Mapping,
    Operations,
    Nonlinear,
    NonuniformScalingOperator,
    RankOneOperator,
    Scalar,
    SelfAdjointOperator,
    SingularSystem,
    SymmetricRankOneOperator,
    UniformScalingOperator

# The following constants are to decide whether or not use BLAS routines
# whenever possible.
const USE_BLAS_DOT = true
const USE_BLAS_AXPY = true
const USE_BLAS_GEMV = true

include("types.jl")
include("rules.jl")
include("blas.jl")
include("methods.jl")
include("vectors.jl")
include("mappings.jl")
include("conjgrad.jl")

end
