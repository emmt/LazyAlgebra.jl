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
# Copyright (c) 2017-2025 Éric Thiébaut.
#

module LazyAlgebra

export
    CirculantConvolution,
    CompressedSparseOperator,
    CroppingOperator,
    Diag,
    Diff,
    FFTOperator,
    GeneralMatrix,
    Gram,
    Id,
    Identity,
    Operator,
    NonuniformScaling,
    RankOneOperator,
    SingularSystem,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    SymbolicOperator,
    SymbolicMapping,
    SymmetricRankOneOperator,
    ZeroPaddingOperator,
    adjoint,
    apply!,
    apply,
    coefficients,
    col_size,
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
    is_selfadjoint,
    isone,
    iszero,
    lgemm!,
    lgemm,
    lgemv!,
    lgemv,
    multiplier,
    ncols,
    nnz,
    nonzeros,
    nrows,
    output_eltype,
    output_ndims,
    output_size,
    output_type,
    row_size,
    sparse,
    terms,
    unpack!,
    unscaled,
    unveil,
    vcombine!,
    vcombine,
    vcopy!,
    vcopy,
    vdot,
    vfill!,
    vmul!,
    vmul,
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

using TypeUtils: @public
@public convert_multiplier
@public create_output
@public multiplier_type
@public unsafe_vcopy!
@public unsafe_vdot
@public unsafe_vscale!
@public unsafe_vswap!
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
include("traits.jl")
include("utils.jl")
include("methods.jl")
include("vectors.jl")
#include("genmult.jl")
#import .GenMult: lgemm!, lgemm, lgemv!, lgemv
#include("blas.jl")
#include("rules.jl")
include("mappings.jl")
#include("foundations.jl")

#include("sparse.jl")
#using .SparseOperators
#import .SparseOperators: unpack!
#
#include("cropping.jl")
#import .Cropping: CroppingOperator, ZeroPaddingOperator, defaultoffset
#include("diff.jl")
#import .FiniteDifferences: Diff
#include("fft.jl")
#import .FFTs: CirculantConvolution, FFTOperator
#include("conjgrad.jl")
#include("init.jl")

end
