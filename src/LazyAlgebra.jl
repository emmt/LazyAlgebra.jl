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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

module LazyAlgebra

export
    Adjoint,
    AdjointInverse,
    CirculantConvolution,
    CroppingOperator,
    Diag,
    DiagonalMapping,
    DiagonalType,
    Direct,
    Endomorphism,
    FFTOperator,
    GeneralMatrix,
    Gram,
    Id,
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
    NonuniformScaling,
    Operations,
    RankOneOperator,
    SelfAdjoint,
    SelfAdjointType,
    SimpleFiniteDifferences,
    SingularSystem,
    SparseOperator,
    SymbolicLinearMapping,
    SymbolicMapping,
    SymmetricRankOneOperator,
    ZeroPaddingOperator,
    adjoint,
    apply!,
    apply,
    coefficients,
    conjgrad!,
    conjgrad,
    diag,
    gram,
    input_eltype,
    input_ndims,
    input_size,
    input_type,
    is_diagonal,
    is_endomorphism,
    is_linear,
    is_selfadjoint,
    isone,
    iszero,
    lgemm!,
    lgemm,
    lgemv!,
    lgemv,
    multiplier,
    output_eltype,
    output_ndims,
    output_size,
    output_type,
    sparse,
    terms,
    unpack!,
    unscaled,
    unveil,
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

using Printf
using ArrayTools

import Base: *, ∘, +, -, \, /, ==
import Base: Tuple, adjoint, inv, axes,
    show, showerror, convert, eltype, ndims, size, length, stride, strides,
    getindex, setindex!, eachindex, first, last, firstindex, lastindex,
    one, zero, isone, iszero, @propagate_inbounds

# Import/using from LinearAlgebra, BLAS and SparseArrays.
using LinearAlgebra
import LinearAlgebra: UniformScaling, diag, ⋅, mul!, rmul!
using LinearAlgebra.BLAS
using LinearAlgebra.BLAS: libblas, @blasfunc,
    BlasInt, BlasReal, BlasFloat, BlasComplex

using SparseArrays: sparse

include("types.jl")
include("utils.jl")
include("methods.jl")
include("vectors.jl")
include("genmult.jl")
import .GenMult: lgemm!, lgemm, lgemv!, lgemv
include("blas.jl")
include("coder.jl")
include("simplify.jl")
include("rules.jl")
include("mappings.jl")
include("sparse.jl")
import .SparseOperators: SparseOperator, sparse, unpack!
include("cropping.jl")
import .Cropping: CroppingOperator, ZeroPaddingOperator, defaultoffset
include("finitedifferences.jl")
import .FiniteDifferences: SimpleFiniteDifferences
include("fft.jl")
import .FFTs: CirculantConvolution, FFTOperator
include("conjgrad.jl")
include("deprecations.jl")

end
