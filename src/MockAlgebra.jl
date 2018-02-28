#
# MockAlgebra.jl -
#
# A simple linear algebra system.
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

module MockAlgebra

import Base: *, \, inv, convert

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
    is_applicable_in_place,
    conjgrad!,
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
    LinearOperator,
    SelfAdjointOperator,
    Direct,
    Adjoint,
    Inverse,
    AdjointInverse,
    InverseAdjoint,
    GeneralMatrix,
    HalfHessian,
    Identity,
    NonuniformScaling,
    SingularSystem

# The following constants are to decide whether or not use BLAS routines
# whenever possible.
const USE_BLAS_DOT = true
const USE_BLAS_AXPY = true
const USE_BLAS_GEMV = true

include("types.jl")
include("blas.jl")
include("vectors.jl")
include("operators.jl")
include("conjgrad.jl")

end
