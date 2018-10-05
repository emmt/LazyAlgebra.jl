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

export
    adjoint,
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

# Deal with compatibility issues.
isdefined(Base, :apply) && import Base: apply
@static if isdefined(Base, :adjoint)
    import Base: adjoint
else
    import Base: ctranspose
    const adjoint = ctranspose
end
using Compat
using Compat.Printf
using Compat: @debug, @error, @info, @warn

# Important revision numbers:
#   * 0.7.0-DEV.3204: A_mul_B! is deprecated (as mul! or scale!)
#   * 0.7.0-DEV.3449: LinearAlgebra in the stdlib
#   * 0.7.0-DEV.3563: scale! -> mul1!
#   * 0.7.0-DEV.3665: mul1! -> rmul!

# Define `LinearAlgebra`.
@static if VERSION < v"0.7.0-DEV.3449"
    # LinearAlgebra not in the stdlib
    const LinearAlgebra = Base.LinAlg
else
    import LinearAlgebra
end
const BLAS = LinearAlgebra.BLAS
import .BLAS: libblas, @blasfunc, BlasInt, BlasReal, BlasFloat, BlasComplex

# Import/define `mul!` and `⋅`.
if VERSION < v"0.7.0-DEV.3204"
    # A_mul_B! not deprecated
    import Base: ⋅, A_mul_B!
    const mul! = A_mul_B!
else
    import .LinearAlgebra: ⋅, mul!
end

# Define `rmul!`.
if VERSION < v"0.7.0-DEV.3563"
    # scale! not deprecated
    rmul!(A::AbstractArray, s::Number) = scale!(A, s)
    #export mul!, rmul!
elseif VERSION < v"0.7.0-DEV.3665"
    # scale! -> mul1!
    rmul!(A::AbstractArray, s::Number) = rmul1!(A, s)
else
    import .LinearAlgebra: rmul!
end

import Base: *, ∘, +, -, \, /, inv

include("types.jl")
include("rules.jl")
include("methods.jl")
include("vectors.jl")
include("blas.jl")
include("mappings.jl")
include("fft.jl")
import .FFT: FFTOperator
include("conjgrad.jl")

end
