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
# Copyright (c) 2017-2019 Éric Thiébaut.
#

__precompile__(true)

module LazyAlgebra

export
    Adjoint,
    AdjointInverse,
    Diag,
    DiagonalMapping,
    DiagonalType,
    Direct,
    Endomorphism,
    FFTOperator,
    GeneralMatrix,
    HalfHessian,
    Hessian,
    Identity,
    Inverse,
    InverseAdjoint,
    Linear,
    LinearMapping,
    LinearType,
    Mapping,
    Morphism,
    MorphismType,
    NonDiagonalMapping,
    NonLinear,
    NonSelfAdjoint,
    NonuniformScalingOperator,
    Operations,
    RankOneOperator,
    SelfAdjoint,
    SelfAdjointType,
    SimpleFiniteDifferences,
    SingularSystem,
    SparseOperator,
    SymmetricRankOneOperator,
    SymbolicLinearMapping,
    SymbolicMapping,
    adjoint,
    allindices,
    allof,
    anyof,
    apply!,
    apply,
    conjgrad!,
    conjgrad,
    contents,
    diag,
    input_eltype,
    input_ndims,
    input_size,
    input_type,
    is_diagonal,
    is_endomorphism,
    has_standard_indexing,
    isflatarray,
    StorageType,
    AnyStorage,
    FlatStorage,
    is_linear,
    is_selfadjoint,
    isone,
    iszero,
    lgemm!,
    lgemm,
    lgemv!,
    lgemv,
    multiplier,
    noneof,
    operand,
    operands,
    output_eltype,
    output_ndims,
    output_size,
    output_type,
    reversemap,
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
    vzeros

@static isdefined(Base, :I) || export I

# Deal with compatibility issues.
@static isdefined(Base, :apply) && import Base: apply
@static if isdefined(Base, :adjoint)
    import Base: adjoint
else
    import Base: ctranspose
    const adjoint = ctranspose
end
@static if isdefined(Base, :axes)
    import Base: axes
else
    import Base: indices
    const axes = indices
end
@static isdefined(Base, :isone) && import Base: isone
using Compat
using Compat.Printf
using Compat: @debug, @error, @info, @warn

# Make paths to LinearAlgebra and BLAS available as constants.  To import/using
# from these, prefix the alias with a dot (relative module path).
const LinearAlgebra = Compat.LinearAlgebra
const BLAS = Compat.LinearAlgebra.BLAS

# Import/using from LinearAlgebra and BLAS.
using .LinearAlgebra
import .LinearAlgebra: UniformScaling, diag
using .BLAS: libblas, @blasfunc, BlasInt, BlasReal, BlasFloat, BlasComplex

# Important revision numbers:
#   * 0.7.0-DEV.3204: A_mul_B! is deprecated (as mul! or scale!)
#   * 0.7.0-DEV.3449: LinearAlgebra in the stdlib
#   * 0.7.0-DEV.3563: scale! -> mul1!
#   * 0.7.0-DEV.3665: mul1! -> rmul!

# Import/define `mul!` and `⋅`.
@static if VERSION < v"0.7.0-DEV.3204"
    # A_mul_B! not deprecated
    import Base: ⋅, A_mul_B!
    const mul! = A_mul_B!
else
    import .LinearAlgebra: ⋅, mul!
end

# Define `rmul!`.
@static if VERSION < v"0.7.0-DEV.3563"
    # scale! not deprecated
    rmul!(A::AbstractArray, s::Number) = scale!(A, s)
    #export mul!, rmul!
elseif VERSION < v"0.7.0-DEV.3665"
    # scale! -> mul1!
    rmul!(A::AbstractArray, s::Number) = rmul1!(A, s)
else
    import .LinearAlgebra: rmul!
end

import Base: *, ∘, +, -, \, /, ==, inv,
    show, showerror, convert, eltype, ndims, size, length, stride, strides,
    getindex, setindex!, eachindex, first, last, one, zero, iszero

include("types.jl")
include("utils.jl")
include("methods.jl")
include("vectors.jl")
include("genmult.jl")
import .GenMult: lgemm!, lgemm, lgemv!, lgemv
include("blas.jl")
include("coder.jl")
include("rules.jl")
include("mappings.jl")
include("sparse.jl")
include("finitedifferences.jl")
import .FiniteDifferences: SimpleFiniteDifferences
include("fft.jl")
import .FFT: FFTOperator
include("conjgrad.jl")
include("deprecations.jl")

end
