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
    Adjoint,
    AdjointInverse,
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
    UniformScalingOperator,
    SymbolicLinearMapping,
    SymbolicMapping,
    adjoint,
    apply!,
    apply,
    conjgrad!,
    conjgrad,
    contents,
    input_eltype,
    input_ndims,
    input_size,
    input_type,
    is_diagonal,
    is_endomorphism,
    is_linear,
    is_selfadjoint,
    isone,
    izero,
    multiplier,
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
@static if isdefined(Base, :isone)
    import Base: isone
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
    import Base.LinAlg: UniformScaling
else
    import LinearAlgebra
    import LinearAlgebra: UniformScaling
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

import Base: *, ∘, +, -, \, /, ==, inv,
    show, showerror, convert, eltype, ndims, size, length, stride,
    getindex, setindex!, eachindex, first, last, one, zero, iszero

include("types.jl")
include("methods.jl")
include("vectors.jl")
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

end
