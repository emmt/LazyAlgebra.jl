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
# Copyright (c) 2017-2021 Éric Thiébaut.
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
    Jacobian,
    LinearMapping,
    Mapping,
    NonuniformScaling,
    RankOneOperator,
    SimpleFiniteDifferences,
    SingularSystem,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    SymbolicLinearMapping,
    SymbolicMapping,
    SymmetricRankOneOperator,
    ZeroPaddingOperator,
    ∇,
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
    is_linear,
    is_selfadjoint,
    isone,
    iszero,
    jacobian,
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
    vcreate,
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
include("genmult.jl")
import .GenMult: lgemm!, lgemm, lgemv!, lgemv
include("blas.jl")
include("coder.jl")
include("rules.jl")
include("mappings.jl")
include("foundations.jl")

include("sparse.jl")
using .SparseOperators
#import .SparseOperators: SparseOperator, sparse, unpack!

include("cropping.jl")
import .Cropping: CroppingOperator, ZeroPaddingOperator, defaultoffset
include("finitedifferences.jl")
import .FiniteDifferences: SimpleFiniteDifferences, Diff
include("fft.jl")
import .FFTs: CirculantConvolution, FFTOperator
include("conjgrad.jl")

end
