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
    #CirculantConvolution,
    #CompressedSparseOperator,
    #CroppingOperator,
    Diag,
    #Diff,
    #FFTOperator,
    #GeneralMatrix,
    Gram,
    Id,
    Identity,
    Operator,
    RankOneOperator,
    #SingularSystem,
    #SparseOperator,
    #SparseOperatorCOO,
    #SparseOperatorCSC,
    #SparseOperatorCSR,
    SymbolicOperator,
    SymmetricRankOneOperator,
    #ZeroPaddingOperator,
    #coefficients,
    #col_size,
    #conjgrad!,
    #conjgrad,
    diag, # re-export from LinearAlgebra
    #gram,
    #is_diagonal,
    #is_endomorphism,
    #is_selfadjoint,
    #lgemm!,
    #lgemm,
    #lgemv!,
    #lgemv,
    #multiplier,
    #ncols,
    #nnz,
    #nonzeros,
    #nrows,
    #row_size,
    #sparse,
    #terms,
    #unpack!,
    #unscaled,
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

using TypeUtils: @public
@public @callable
@public Adjoint Inverse InverseAdjoint
@public Identity
@public input_eltype InputEltype InputEltypeUnknown HasInputEltype
@public input_axes input_ndims InputShape InputShapeUnknown HasInputShape
#@public LazyMap
@public output_eltype OutputEltype OutputEltypeUnknown HasOutputEltype
@public output_axes output_ndims OutputShape OutputShapeUnknown HasOutputShape
@public Sum Prod
@public convert_multiplier multiplier_type
#@public create_output
#@public default_cropping_offset
#@public default_zeropadding_offset
@public unsafe_vcombine! dispatch_vcombine!
@public unsafe_vcopy!
@public unsafe_vdot
@public unsafe_vmul!
@public unsafe_vproduct!
@public unsafe_vscale! dispatch_vscale! dispatch_vscale!
@public unsafe_vswap!
@public unsafe_vupdate!
@public unveil

#using Printf
using ArrayTools
using TypeUtils
#using FFTW

using Base: OneTo, Fix1, Fix2, @propagate_inbounds

# Import/using from LinearAlgebra, BLAS and SparseArrays.
using LinearAlgebra
#using LinearAlgebra.BLAS
#using LinearAlgebra.BLAS: libblas, @blasfunc,
#    BlasInt, BlasReal, BlasFloat, BlasComplex
#
#using SparseArrays: sparse

include("types.jl")
#include("traits.jl")
include("utils.jl")
include("multipliers.jl")
include("vectors.jl")
include("operators.jl")
include("rules.jl")
include("symbolic.jl")
include("identity.jl")
include("diag.jl")
include("map.jl")
include("rank1.jl")
#include("matrices.jl")
#include("cropping.jl")
#include("genmult.jl")
#import .GenMult: lgemm!, lgemm, lgemv!, lgemv
#include("blas.jl")
#include("foundations.jl")

#include("sparse.jl")
#using .SparseOperators
#import .SparseOperators: unpack!
#
#import .Cropping: CroppingOperator, ZeroPaddingOperator, defaultoffset
#include("diff.jl")
#import .FiniteDifferences: Diff
#include("fft.jl")
#import .FFTs: CirculantConvolution, FFTOperator
#include("conjgrad.jl")
#include("init.jl")

end
