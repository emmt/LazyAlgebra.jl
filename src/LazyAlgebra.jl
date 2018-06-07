#
# LazyAlgebra.jl -
#
# A simple linear algebra system.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

module LazyAlgebra

isdefined(Base, :apply) && import Base: apply

export
    apply!,
    apply,
    conjgrad!,
    conjgrad!,
    conjgrad,
    contents,
    diagonaltype,
    inplacetype,
    input_eltype,
    input_ndims,
    input_size,
    input_type,
    is_applicable_in_place,
    lineartype,
    morphismtype,
    output_eltype,
    output_ndims,
    output_size,
    output_type,
    selfadjointtype,
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
    vones,
    vproduct!,
    vproduct,
    vscale!,
    vscale,
    vswap!,
    vupdate!,
    vzero!,
    vzeros,
    Adjoint,
    AdjointInverse,
    DiagonalMapping,
    Direct,
    Endomorphism,
    FFTOperator,
    GeneralMatrix,
    HalfHessian,
    Hessian,
    Identity,
    InPlace,
    Inverse,
    InverseAdjoint,
    Linear,
    LinearMapping,
    Mapping,
    Morphism,
    NonDiagonalMapping,
    NonLinear,
    NonSelfAdjoint,
    NonuniformScalingOperator,
    Operations,
    OutOfPlace,
    RankOneOperator,
    Scalar,
    SelfAdjoint,
    SingularSystem,
    SymmetricRankOneOperator,
    UniformScalingOperator

# The following constant is to decide whether or not use BLAS routines whenever
# possible.
const USE_BLAS = true

include("types.jl")
include("rules.jl")
@static if USE_BLAS
    include("blas.jl")
end
include("methods.jl")
include("vectors.jl")
include("mappings.jl")
include("fft.jl")
import .FFT.FFTOperator
include("conjgrad.jl")

end
