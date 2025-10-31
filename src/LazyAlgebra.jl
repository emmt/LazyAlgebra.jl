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

# Macros must be defined earlier, before being exported.
include("macros.jl")

export
    AbstractSparseOperator,
    COO,
    CSC,
    CSR,
    CirculantConvolution,
    CompressedSparseColumn,
    CompressedSparseCoordinate,
    CompressedSparseRow,
    CroppingOperator,
    Diag,
    Diff,
    FFT,
    FlexibleMatrix,
    Gram,
    Id,
    Identity,
    Operator,
    PseudoMatrix,
    RankOneOperator,
    SparseFormat,
    SparseOperatorLike,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    SymbolicOperator,
    SymmetricRankOneOperator,
    ZeroPaddingOperator,
    conjgrad!,
    conjgrad,
    #goodfftdim,
    #goodfftdims,
    #rfftdims,
    #is_diagonal,
    #is_endomorphism,
    #is_selfadjoint,
    #lgemm!,
    #lgemm,
    #lgemv!,
    #lgemv,
    #multiplier,
    simplify,
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
    vnans,
    vnans!,
    vones,
    vones!,
    vproduct!,
    vproduct,
    vscale!,
    vscale,
    vswap!,
    vupdate!,
    vzeros!,
    vzeros,

    # Re-exports from LinearAlgebra:
    diag,

    # Re-exports from TypeUtils:
    adapt_precision,
    get_precision,

    # Re-exports from AbstractFFTs:
    #fftfreq,
    fftshift,
    ifftshift,

    # Re-exports from SparseArrays:
    nnz,
    nonzeros,
    sparse

# Non-exported but public API.
using TypeUtils: @public
@public @callable,
        @dispatch_on_multiplier,
        Adjoint,
        ColumnMajor,
        Conjugate,
        ConjugateGradient,
        HasInputEltype,
        HasInputShape,
        HasOutputEltype,
        HasOutputShape,
        InputEltype,
        InputEltypeUnknown,
        InputShape,
        InputShapeUnknown,
        Inverse,
        InverseAdjoint,
        InverseConjugate,
        InverseTranspose,
        LowerTriangularShape,
        MatrixShape,
        MatrixShapeAny,
        OutputEltype,
        OutputEltypeUnknown,
        OutputShape,
        OutputShapeUnknown,
        Prod,
        RowMajor,
        StorageOrder,
        StorageOrderAny,
        Sum,
        Swapped,
        Transpose,
        TriangularShape,
        UpperTriangularShape,
        check_input_axes,
        check_output_axes,
        col_axes,
        col_index,
        col_indices,
        col_ndims,
        col_size,
        convert_multiplier,
        create_output,
        default_cropping_offset,
        default_cropping_offset,
        default_zeropadding_offset,
        each_col_index,
        each_nz_index,
        each_row_index,
        fast_max,
        fast_min,
        first_nz_index,
        input_axes,
        input_eltype,
        input_ndims,
        input_size,
        is_column_major,
        is_lower_triangular,
        is_row_major,
        is_upper_triangular,
        last_nz_index,
        multiplier,
        ncols,
        nrows,
        offsets,
        ordinal_suffix,
        output_axes,
        output_eltype,
        output_ndims,
        output_size,
        row_axes,
        row_index,
        row_indices,
        row_ndims,
        row_size,
        test_API,
        try_simplify,
        unpack!,
        unsafe_vcombine!,
        unsafe_vcopy!,
        unsafe_vdot,
        unsafe_vmul!,
        unsafe_vproduct!,
        unsafe_vscale!,
        unsafe_vswap!,
        unsafe_vupdate!,
        unscaled

using Printf
using ArrayTools
using LazyMaps
using Neutrals
using Test
using StructuredArrays
using TypeUtils
using Unitful
using Unitful: AbstractQuantity, Quantity, NoDims, unit, ustrip
using ZippedArrays
using AbstractFFTs, FFTW

import SparseArrays
using SparseArrays: SparseMatrixCSC, nonzeros, nnz, sparse

using Base: OneTo, Fix1, Fix2, @propagate_inbounds

# Import/using from LinearAlgebra, BLAS and SparseArrays.
using LinearAlgebra
#using LinearAlgebra.BLAS
#using LinearAlgebra.BLAS: libblas, @blasfunc,
#    BlasInt, BlasReal, BlasFloat, BlasComplex
#

include("types.jl")
include("traits.jl")
include("errors.jl")
include("show.jl")
include("utils.jl")
include("multipliers.jl")
include("vectors.jl")
include("operators.jl")
include("matrices.jl")
include("rules.jl")
include("symbolic.jl")
include("identity.jl")
include("diag.jl")
include("rank1.jl")
include("pseudomatrices.jl")
include("cropping.jl")
include("diff.jl")
include("sparse.jl")
include("fft.jl")
include("simplify.jl")
include("conjgrad.jl")
import .ConjugateGradient: conjgrad, conjgrad!
#include("genmult.jl")
#import .GenMult: lgemm!, lgemm, lgemv!, lgemv
#include("blas.jl")

end
