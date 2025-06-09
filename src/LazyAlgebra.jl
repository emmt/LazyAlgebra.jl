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
    CirculantConvolution,
    CompressedSparseOperator,
    CroppingOperator,
    Diag,
    Diff,
    FFT,
    FlexibleMatrix,
    Id,
    Identity,
    Operator,
    PseudoMatrix,
    RankOneOperator,
    #SingularSystem,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    SymbolicOperator,
    SymmetricRankOneOperator,
    ZeroPaddingOperator,
    conjgrad!,
    conjgrad,
    diag, # re-export from LinearAlgebra
    #fftfreq,
    fftshift, # re-export from AbstractFFTs
    #goodfftdim,
    #goodfftdims,
    ifftshift, # re-export from AbstractFFTs
    #rfftdims,
    get_precision,
    #is_diagonal,
    #is_endomorphism,
    #is_selfadjoint,
    #lgemm!,
    #lgemm,
    #lgemv!,
    #lgemv,
    #multiplier,
    nnz, # re-export from SparseArrays
    nonzeros, # re-export from SparseArrays
    simplify,
    sparse, # re-export from SparseArrays
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
    with_precision

# Non-exported but public API.
using TypeUtils: @public
@public @callable
@public @dispatch_on_multiplier
@public Adjoint
@public HasInputEltype
@public HasInputShape
@public HasOutputEltype
@public HasOutputShape
@public Identity
@public InputEltype
@public InputEltypeUnknown
@public InputShape
@public InputShapeUnknown
@public Inverse
@public InverseAdjoint
@public LazyMap
@public OutputEltype
@public OutputEltypeUnknown
@public OutputShape
@public OutputShapeUnknown
@public Prod
@public Sum
@public check_input_axes
@public check_output_axes
@public col_axes
@public col_index
@public col_indices
@public col_ndims
@public col_size
@public convert_inplace_multiplier
@public convert_multiplier
@public create_output
@public default_cropping_offset
@public default_cropping_offset
@public default_zeropadding_offset
@public dimensionless
@public each_col_index
@public each_nz_index
@public each_row_index
@public first_nz_index
@public inplace_multiplier
@public input_axes
@public input_eltype
@public input_ndims
@public input_size
@public last_nz_index
@public multiplier
@public multiplier_type
@public ncols
@public nrows
@public offsets
@public ordinal_suffix
@public output_axes
@public output_eltype
@public output_ndims
@public output_size
@public row_axes
@public row_index
@public row_indices
@public row_ndims
@public row_size
@public test_API
@public try_simplify
@public unpack!
@public unsafe_vcombine!
@public unsafe_vcopy!
@public unsafe_vdot
@public unsafe_vmul!
@public unsafe_vproduct!
@public unsafe_vscale!
@public unsafe_vswap!
@public unsafe_vupdate!
@public unscaled

using Printf
using ArrayTools
using Neutrals
using Test
using StructuredArrays
using TypeUtils
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
include("map.jl")
include("rank1.jl")
include("pseudomatrices.jl")
include("cropping.jl")
include("diff.jl")
include("sparse.jl")
include("fft.jl")
include("simplify.jl")
include("conjgrad.jl")
#include("genmult.jl")
#import .GenMult: lgemm!, lgemm, lgemv!, lgemv
#include("blas.jl")

end
