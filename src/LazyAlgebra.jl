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
# Copyright (c) 2017-2022 Éric Thiébaut.
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
    NonLinearMapping,
    NonuniformScaling,
    RankOneOperator,
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
    nterms,
    output_eltype,
    output_ndims,
    output_size,
    output_type,
    row_size,
    sparse,
    terms,
    unpack!,
    unscaled,
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
using Unitless

import Base: *, ∘, +, -, \, /, ==
import Base: Tuple, adjoint, inv, axes,
    show, showerror, convert, eltype, ndims, size, length, stride, strides,
    getindex, setindex!, eachindex, first, last, firstindex, lastindex

using Base: @propagate_inbounds

# Import/using from LinearAlgebra, BLAS and SparseArrays.
using LinearAlgebra
using LinearAlgebra: UniformScaling
import LinearAlgebra: diag, ⋅, mul!, rmul!
using LinearAlgebra.BLAS
using LinearAlgebra.BLAS: libblas, @blasfunc,
    BlasInt, BlasReal, BlasFloat, BlasComplex

using SparseArrays: sparse

include("types.jl")
include("traits.jl")
include("utils.jl")
include("methods.jl")
include("vectors.jl")
#=
include("genmult.jl")
import .GenMult: lgemm!, lgemm, lgemv!, lgemv
include("blas.jl")
=#
include("rules.jl")
include("simplify.jl")
include("mappings.jl")
#=
include("foundations.jl")

include("sparse.jl")
using .SparseOperators
import .SparseOperators: unpack!

include("cropping.jl")
import .Cropping: CroppingOperator, ZeroPaddingOperator, default_offset
include("diff.jl")
import .FiniteDifferences: Diff
include("fft.jl")
import .FFTs: CirculantConvolution, FFTOperator
include("conjgrad.jl")
=#
include("init.jl")

end
