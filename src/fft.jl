#
# fft.jl -
#
# Implementation of FFT and circulant convolution operators.
#
#------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (C) 2017-2020, Éric Thiébaut.
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
#

module FFTs

# Be nice with the caller: re-export `fftshift` and `ifftshift` but not `fft`,
# `ifft` etc. as the `FFTOperator` is meant to replace them.
export
    CirculantConvolution,
    FFTOperator,
    fftfreq,
    fftshift,
    goodfftdim,
    goodfftdims,
    ifftshift,
    rfftdims

using ..LazyAlgebra
import ..LazyAlgebra: adjoint, apply!, vcreate, MorphismType, mul!,
    input_size, input_ndims, input_eltype,
    output_size, output_ndims, output_eltype,
    is_same_mapping
using ..LazyAlgebra: _merge_mul

import Base: *, /, \, inv, show

import AbstractFFTs: Plan, fftshift, ifftshift

using FFTW
import FFTW: fftwNumber, fftwReal, fftwComplex, FFTWPlan, cFFTWPlan, rFFTWPlan

# The time needed to allocate temporary arrays is negligible compared to the
# time taken to compute a FFT (e.g., 5µs to allocate a 256×256 array of double
# precision complexes versus 1.5ms to compute its FFT).  We therefore do not
# store any temporary arrays in the FFT operator.  Only the FFT plans are
# cached in the operator.

#------------------------------------------------------------------------------
# Extend LazyAlgebra framework for FFTW plans.
#
# This simplify a lot the implementation of FFT and circulant convolution
# operators without loss of performances.

macro checksize(name, arg, dims)
    return quote
        size($(esc(arg))) == $(esc(dims)) || badsize($(esc(name)), $(esc(dims)))
    end
end

@noinline badsize(name::String, dims::Tuple{Vararg{Integer}}) =
    throw(DimensionMismatch("$name must have dimensions $dims"))

input_size(P::FFTWPlan) = P.sz
output_size(P::FFTWPlan) = P.osz
#input_strides(P::FFTWPlan) = P.istride
#output_strides(P::FFTWPlan) = P.ostride
flags(P::FFTWPlan) = P.flags

destroys_input(A::FFTWPlan) =
    (flags(A) & (FFTW.PRESERVE_INPUT|FFTW.DESTROY_INPUT)) == FFTW.DESTROY_INPUT

preserves_input(A::FFTWPlan) =
    (flags(A) & (FFTW.PRESERVE_INPUT|FFTW.DESTROY_INPUT)) == FFTW.PRESERVE_INPUT

# Create result for an in-place complex-complex forward/backward FFT
# transform.
function vcreate(::Type{Direct},
                 A::cFFTWPlan{Complex{T},K,true,N},
                 x::StridedArray{Complex{T},N},
                 scratch::Bool=false) where {T<:fftwReal,K,N}
    @checksize "argument" x input_size(A)
    return (scratch ? x : Array{Complex{T}}(undef, output_size(A)))
end

# Create result for an out-of-place complex-complex forward/backward FFT
# transform.
function vcreate(::Type{Direct},
                 A::cFFTWPlan{Complex{T},K,false,N},
                 x::StridedArray{Complex{T},N},
                 scratch::Bool=false) where {T<:fftwReal,K,N}
    @checksize "argument" x input_size(A)
    return Array{Complex{T}}(undef, output_size(A))
end

# Create result for a real-complex or a complex-real forward/backward FFT
# transform.  The result is necessarily a new array whatever the `scratch`
# flag.
function vcreate(::Type{Direct},
                 A::rFFTWPlan{T,K,false,N},
                 x::StridedArray{T,N},
                 scratch::Bool=false) where {T<:fftwReal,K,N}
    @checksize "argument" x input_size(A)
    return Array{Complex{T}}(undef, output_size(A))
end

function vcreate(::Type{Direct},
                 A::rFFTWPlan{Complex{T},K,false,N},
                 x::StridedArray{Complex{T},N},
                 scratch::Bool=false) where {T<:fftwReal,K,N}
    @checksize "argument" x input_size(A)
    return Array{T}(undef, output_size(A))
end

# Extend `apply!` for FFTW plans.  We want to compute:
#
#    y = α⋅F⋅x + β⋅y
#
# with as few temporaries as possible.  If β = 0, then there are no needs
# to save the contents of y which can be used directly for the output of
# the transform.  Extra checks are required to make sure the contents x is
# not damaged unless scratch is true.  It tuns out that the implementation
# depends on the type of transform so several versions are coded below.

# Apply in-place complex-complex forward/backward FFT transform.
function apply!(α::Real,
                ::Type{Direct},
                A::cFFTWPlan{Complex{T},K,true,N},
                x::StridedArray{Complex{T},N},
                scratch::Bool,
                β::Real,
                y::StridedArray{Complex{T},N}) where {T<:fftwReal,N,K}
    @checksize "argument" x  input_size(A)
    @checksize "result"   y output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        mul!(y, A, vscale!(y, α, x))
    elseif scratch
        vcombine!(y, α, mul!(x, A, x), β, y)
    else
        z = copyto!(Array{Complex{T},N}(undef, size(x)), x)
        vcombine!(y, α, mul!(z, A, z), β, y)
    end
    return y
end

# Apply out-of-place complex-complex forward/backward FFT transform.
function apply!(α::Real,
                ::Type{Direct},
                A::cFFTWPlan{Complex{T},K,false,N},
                x::StridedArray{Complex{T},N},
                scratch::Bool,
                β::Real,
                y::StridedArray{Complex{T},N}) where {T<:fftwReal,N,K}
    @checksize "argument" x  input_size(A)
    @checksize "result"   y output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        safe_mul!(y, A, x, scratch && x !== y)
        α == 1 || vscale!(y, α)
    else
        vcombine!(y, α, safe_mul(A, x, scratch), β, y)
    end
    return y
end

# Apply real-to-complex forward transform.  The transform is necessarily
# out-of-place.
function apply!(α::Real,
                ::Type{Direct},
                A::rFFTWPlan{T,K,false,N},
                x::StridedArray{T,N},
                scratch::Bool,
                β::Real,
                y::StridedArray{Complex{T},N}) where {T<:fftwReal,K,N}
    @checksize "argument" x  input_size(A)
    @checksize "result"   y output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        safe_mul!(y, A, x, scratch)
        α == 1 || vscale!(y, α)
    else
        vcombine!(y, α, safe_mul(A, x, scratch), β, y)
    end
    return y
end

# Apply complex-to-real (c2r) backward transform. Preserving input is not
# possible for multi-dimensional c2r transforms so we must copy the input
# argument x.
function apply!(α::Real,
                ::Type{Direct},
                A::rFFTWPlan{Complex{T},K,false,N},
                x::StridedArray{Complex{T},N},
                scratch::Bool,
                β::Real,
                y::StridedArray{T,N}) where {T<:fftwReal,K,N}
    @checksize "argument" x  input_size(A)
    @checksize "result"   y output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        safe_mul!(y, A, x, scratch)
        α == 1 || vscale!(y, α)
    else
        vcombine!(y, α, safe_mul(A, x, scratch), β, y)
    end
    return y
end

"""
```julia
safe_mul!(dest, A, src, scratch=false) -> dest
```

overwrite `dest` with the result of applying operator `A` to `src` and
returns `dest`.  Unless `scratch` is true, it is guaranteed that `src` is
preserved which may involve making a temporary copy of it.

See also [`safe_mul`](@ref).

"""
function safe_mul!(dest::StridedArray{Complex{T},N},
                   A::cFFTWPlan{Complex{T},K,inplace,N},
                   src::StridedArray{Complex{T},N},
                   scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    _safe_mul!(dest, A, src, scratch)
end

function safe_mul!(dest::StridedArray{Complex{T},N},
                   A::rFFTWPlan{T,K,inplace,N},
                   src::StridedArray{T,N},
                   scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    _safe_mul!(dest, A, src, scratch)
end

function safe_mul!(dest::StridedArray{T,N},
                   A::rFFTWPlan{Complex{T},K,inplace,N},
                   src::StridedArray{Complex{T},N},
                   scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    _safe_mul!(dest, A, src, scratch)
end

function _safe_mul!(dest::StridedArray, A::FFTWPlan,
                    src::StridedArray{T,N}, scratch::Bool) where {T,N}
    if scratch || preserves_input(A)
        mul!(dest, A, src)
    else
        cpy = copyto!(Array{T,N}(undef, size(src)), src)
        mul!(dest, A, cpy)
    end
    return dest
end

"""
```julia
safe_mul(A, x, scratch=false)
```

yields the result of applying operator `A` to `x`.  Unless `scratch` is
true, it is guaranteed that input `x` is preserved which may involve making
a temporary copy of it.

See also [`safe_mul!`](@ref).

"""
function safe_mul(A::cFFTWPlan{Complex{T},K,inplace,N},
                  x::StridedArray{Complex{T},N},
                  scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    y = Array{Complex{T},N}(undef, output_size(A))
    safe_mul!(y, A, x, scratch)
end

function safe_mul(A::rFFTWPlan{T,K,inplace,N},
                  x::StridedArray{T,N},
                  scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    y = Array{Complex{T},N}(undef, output_size(A))
    safe_mul!(y, A, x, scratch)
end

function safe_mul(A::rFFTWPlan{Complex{T},K,inplace,N},
                  x::StridedArray{Complex{T},N},
                  scratch::Bool = false) where {T<:fftwReal,K,inplace,N}
    y = Array{T,N}(undef, output_size(A))
    safe_mul!(y, A, x, scratch)
end

#------------------------------------------------------------------------------
# FFT operator.

"""
```julia
FFTOperator(A) -> F
```

yields an FFT operator suitable for computing the fast Fourier transform of
arrays similar to `A`.  The operator can also be specified by the
real/complex floating-point type of the elements of the arrays to transform
and their dimensions:

```julia
FFTOperator(T, dims) -> F
```

where `T` is one of `Float64`, `Float32` (for a real-complex FFT),
`Complex{Float64}`, `Complex{Float32}` (for a complex-complex FFT) and
`dims` gives the dimensions of the arrays to transform (by the `Direct` or
`InverseAdjoint` operation).

The interest of creating such an operator is that it caches the ressources
necessary for fast computation of the FFT and can be therefore *much*
faster than calling `fft`, `rfft`, `ifft`, etc.  This is especially true on
small arrays.  Keywords `flags` and `timelimit` may be used to specify
planning options and time limit to create the FFT plans (see
http://www.fftw.org/doc/Planner-Flags.html).  The defaults are
`flags=FFTW.ESTIMATE` and no time limit.

An instance of `FFTOperator` is a linear mapping which can be used as any
other mapping:

```julia
F*x     # yields the FFT of x
F'*x    # yields the adjoint FFT applied to x, that is the backward FFT of x
F\\x     # yields the inverse FFT of x
```

See also: [`fft`](@ref), [`plan_fft`](@ref), [`bfft`](@ref),
          [`plan_bfft`](@ref), [`rfft`](@ref), [`plan_rfft`](@ref),
          [`brfft`](@ref), [`plan_brfft`](@ref).

"""
struct FFTOperator{T<:fftwNumber,  # element type of input
                   N,              # number of dimensions
                   C<:fftwComplex, # element type of output
                   F<:Plan{T},     # type of forward plan
                   B<:Plan{C}      # type of backward plan
                   } <: LinearMapping
    ncols::Int             # number of input elements
    inpdims::NTuple{N,Int} # input dimensions
    outdims::NTuple{N,Int} # output dimensions
    forward::F             # plan for forward transform
    backward::B            # plan for backward transform
end

# Real-to-complex FFT.
function FFTOperator(::Type{T},
                     dims::NTuple{N,Int};
                     timelimit::Real = FFTW.NO_TIMELIMIT,
                     flags::Integer = FFTW.ESTIMATE) where {T<:fftwReal,N}
    # Check arguments and build dimension list of the result of the forward
    # real-to-complex (r2c) transform.
    planning = check_flags(flags)
    ncols = check_dimensions(dims)
    zdims = rfftdims(dims)

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms implemented in FFTW, see
    # http://www.fftw.org/doc/Planner-Flags.html).
    forward = plan_rfft(Array{T}(undef, dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)
    backward = plan_brfft(Array{Complex{T}}(undef, zdims), dims[1];
                          flags = (planning  | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)

    # Build operator.
    F = typeof(forward)
    B = typeof(backward)
    return FFTOperator{T,N,Complex{T},F,B}(ncols, dims, zdims,
                                           forward, backward)
end

# Complex-to-complex FFT.
function FFTOperator(::Type{T},
                     dims::NTuple{N,Int};
                     timelimit::Real = FFTW.NO_TIMELIMIT,
                     flags::Integer = FFTW.ESTIMATE) where {T<:fftwComplex,N}
    # Check arguments.  The input and output of the complex-to-complex
    # transform have the same dimensions.
    planning = check_flags(flags)
    ncols = check_dimensions(dims)
    temp = Array{T}(undef, dims)

    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.
    forward = plan_fft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                        timelimit = timelimit)
    backward = plan_bfft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)

    # Build operator.
    F = typeof(forward)
    B = typeof(backward)
    return FFTOperator{T,N,T,F,B}(ncols, dims, dims, forward, backward)
end

# Constructor for dimensions not specified as a tuple.
FFTOperator(T::Type{<:fftwNumber}, dims::Integer...; kwds...) =
    FFTOperator(T, dims; kwds...)

# The following 2 definitions are needed to avoid ambiguities.
FFTOperator(T::Type{<:fftwReal}, dims::Tuple{Vararg{Integer}}; kwds...) =
    FFTOperator(T, map(Int, dims); kwds...)
FFTOperator(T::Type{<:fftwComplex}, dims::Tuple{Vararg{Integer}}; kwds...) =
    FFTOperator(T, map(Int, dims); kwds...)

# Constructor for transforms applicable to a given array.
FFTOperator(A::DenseArray{T,N}; kwds...) where {T<:fftwNumber,N} =
    FFTOperator(T, size(A); kwds...)

# Traits:
MorphismType(::FFTOperator{<:Complex}) = Endomorphism()

ncols(A::FFTOperator) = A.ncols
ncols(A::Adjoint{<:FFTOperator}) = ncols(operand(A))
ncols(A::Inverse{<:FFTOperator}) = ncols(operand(A))
ncols(A::InverseAdjoint{<:FFTOperator}) = ncols(operand(A))

input_size(A::FFTOperator) = A.inpdims # FIXME: input_size(A.forward)
input_size(A::FFTOperator, i::Integer) = get_dimension(input_size(A), i)
output_size(A::FFTOperator) = A.outdims
output_size(A::FFTOperator, i::Integer) = get_dimension(output_size(A), i)
input_ndims(A::FFTOperator{T,N,C}) where {T,N,C} = N
output_ndims(A::FFTOperator{T,N,C}) where {T,N,C} = N
input_eltype(A::FFTOperator{T,N,C}) where {T,N,C} = T
output_eltype(A::FFTOperator{T,N,C}) where {T,N,C} = C

# 2 FFT operators can be considered the same if they operate on arguments with
# the same element type and the same dimensions.  If the types do not match,
# the matching method is the one which return false, so it is only needed to
# implement the method for two arguments with the same types (omitting the type
# of the plans as it is irrelevant here).
is_same_mapping(A::FFTOperator{T,N,C}, B::FFTOperator{T,N,C}) where {T,N,C} =
    (input_size(A) == input_size(B))

show(io::IO, A::FFTOperator) = print(io, "FFT")

# Impose the following simplifying rules:
#     inv(F) = n\F'
#     ==> F⋅F' = F'⋅F = n⋅I
#     ==> inv(F⋅F') = inv(F'⋅F) = inv(F)⋅inv(F') = inv(F')⋅inv(F) = n\I
*(A::Adjoint{F}, B::F) where {F<:FFTOperator} =
    (is_same_mapping(operand(A), B) ? ncols(A)*I : _merge_mul(A, B))
*(A::F, B::Adjoint{F}) where {F<:FFTOperator} =
    (is_same_mapping(A, operand(B)) ? ncols(A)*I : _merge_mul(A, B))
*(A::InverseAdjoint{F}, B::Inverse{F}) where {F<:FFTOperator} =
    (is_same_mapping(operand(A), operand(B)) ? (1//ncols(A))*I :
     _merge_mul(A, B))
*(A::Inverse{F}, B::InverseAdjoint{F}) where {F<:FFTOperator} =
    (is_same_mapping(operand(A), operand(B)) ? (1//ncols(A))*I :
     _merge_mul(A, B))

function vcreate(P::Type{<:Union{Direct,InverseAdjoint}},
                 A::FFTOperator{T,N,C},
                 x::DenseArray{T,N},
                 scratch::Bool=false) where {T,N,C}
    vcreate(Direct, A.forward, x, scratch)
end

function vcreate(P::Type{<:Union{Adjoint,Inverse}},
                 A::FFTOperator{T,N,C},
                 x::DenseArray{C,N},
                 scratch::Bool=false) where {T,N,C}
    vcreate(Direct, A.backward, x, scratch)
end

#
# In principle, FFTW plans can be applied to strided arrays (StridedArray) but
# this imposes that the arguments have the same strides.  So for now, we choose
# to restrict arguments to arrays with contiguous elements (DenseArray).
#

function apply!(α::Real,
                ::Type{Direct},
                A::FFTOperator{T,N,C},
                x::DenseArray{T,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {T,N,C}
    return apply!(α, Direct, A.forward, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{Adjoint},
                A::FFTOperator{T,N,C},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{T,N}) where {T,N,C}
    return apply!(α, Direct, A.backward, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{Inverse},
                A::FFTOperator{T,N,C},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{T,N}) where {T,N,C}
    return apply!(α/ncols(A), Direct, A.backward, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{InverseAdjoint},
                A::FFTOperator{T,N,C},
                x::DenseArray{T,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {T,N,C}
    return apply!(α/ncols(A), Direct, A.forward, x, scratch, β, y)
end

#------------------------------------------------------------------------------
# Circulant convolution.

struct CirculantConvolution{T<:fftwNumber,N,
                            C<:fftwComplex,
                            F<:Plan{T},
                            B<:Plan{C}} <: LinearMapping
    dims::NTuple{N,Int}  # input/output dimensions
    zdims::NTuple{N,Int} # complex dimensions
    mtf::Array{C,N}      # modulation transfer function
    forward::F           # plan for forward transform
    backward::B          # plan for backward transform
end

# Traits:
MorphismType(::CirculantConvolution) = Endomorphism()

# Basic methods for a linear operator on Julia's arrays.
input_size(H::CirculantConvolution) = H.dims
output_size(H::CirculantConvolution) = H.dims
input_size(H::CirculantConvolution, i::Integer) = get_dimension(H.dims, i)
output_size(H::CirculantConvolution, i::Integer) = get_dimension(H.dims, i)
input_ndims(H::CirculantConvolution{T,N}) where {T,N} = N
output_ndims(H::CirculantConvolution{T,N}) where {T,N} = N
input_eltype(H::CirculantConvolution{T,N}) where {T,N} = T
output_eltype(H::CirculantConvolution{T,N}) where {T,N} = T

# Basic methods for an array.
Base.eltype(H::CirculantConvolution{T,N}) where {T,N} = T
Base.size(H::CirculantConvolution{T,N}) where {T,N} =
    ntuple(i -> H.dims[(i ≤ N ? i : i - N)], 2*N)
Base.size(H::CirculantConvolution{T,N}, i::Integer) where {T,N} =
    H.dims[(i ≤ N ? i : i - N)]
Base.ndims(H::CirculantConvolution{T,N}) where {T,N} = 2*N

"""
# Circulant convolution operator

The circulant convolution operator `H` is defined by:

```julia
H  = (1/n)*F'*Diag(mtf)*F
```

with `n` the number of elements, `F` the discrete Fourier transform operator
and `mtf` the modulation transfer function.

The operator `H` can be created by:

```julia
H = CirculantConvolution(psf; flags=FFTW.ESTIMATE, timelimit=Inf, shift=false)
```

where `psf` is the point spread function (PSF).  Note that the PSF is assumed
to be centered according to the convention of the discrete Fourier transform.
You may use `ifftshift` or the keyword `shift` if the PSF is geometrically
centered:

```julia
H = CirculantConvolution(ifftshift(psf))
H = CirculantConvolution(psf, shift=true)
```

The following keywords can be specified:

* `shift` (`false` by default) indicates whether to apply `ifftshift` to `psf`.

* `normalize` (`false` by default) indicates whether to divide `psf` by the sum
  of its values.  This keyword is only available for real-valued PSF.

* `flags` is a bitwise-or of FFTW planner flags, defaulting to `FFTW.ESTIMATE`.
  If the operator is to be used many times (as in iterative methods), it is
  recommended to use at least `flags=FFTW.MEASURE` which generally yields
  faster transforms compared to the default `flags=FFTW.ESTIMATE`.

* `timelimit` specifies a rough upper bound on the allowed planning time, in
  seconds.

The operator can be used as a regular linear operator: `H(x)` or `H*x` to
compute the convolution of `x` and `H'(x)` or `H'*x` to apply the adjoint of
`H` to `x`.

For a slight improvement of performances, an array `y` to store the result of
the operation can be provided:

```julia
apply!(y, [P=Direct,] H, x) -> y
apply!(y, H, x)
apply!(y, H', x)
```

If provided, `y` must be at a different memory location than `x`.

""" CirculantConvolution

# Create a circular convolution operator for real arrays.
function CirculantConvolution(psf::AbstractArray{T,N};
                              flags::Integer = FFTW.ESTIMATE,
                              normalize::Bool = false,
                              shift::Bool = false,
                              kwds...) where {T<:fftwReal,N}
    # Check arguments and compute dimensions.
    planning = check_flags(flags)
    n = length(psf)
    dims = size(psf)
    zdims = ntuple(i -> (i == 1 ? div(dims[i],2) + 1 : dims[i]), N)

    # Allocate temporary array for the scaled MTF and, if needed, a scratch
    # array for planning which may destroy its input.
    mtf = Array{Complex{T}}(undef, zdims)
    if planning == FFTW.ESTIMATE || planning == FFTW.WISDOM_ONLY
        tmp = psf
    else
        tmp = Array{T}(undef, dims)
    end

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms).
    forward = plan_rfft(tmp; flags = (planning | FFTW.PRESERVE_INPUT), kwds...)
    backward = plan_brfft(mtf, dims[1]; flags = (planning | FFTW.DESTROY_INPUT),
                          kwds...)

    # Compute the scaled MTF.
    mul!(mtf, forward, (shift ? ifftshift(psf) : psf))
    if normalize
        sum = real(mtf[1])
        if sum != 1
            if sum <= 0
                throw(ArgumentError("cannot normalize: sum(PSF) ≤ 0"))
            end
            vscale!(mtf, 1/sum)
        end
    end

    # Build operator.
    F = typeof(forward)
    B = typeof(backward)
    CirculantConvolution{T,N,Complex{T},F,B}(dims, zdims, mtf,
                                             forward, backward)
end

# Create a circular convolution operator for complex arrays (see
# docs/convolution.md for explanations).
function CirculantConvolution(psf::AbstractArray{Complex{T},N};
                              flags::Integer = FFTW.ESTIMATE,
                              shift::Bool = false,
                              kwds...) where {T<:fftwReal,N}
    # Check arguments and get dimensions.
    planning = check_flags(flags)
    n = length(psf)
    dims = size(psf)

    # Allocate array for the scaled MTF, will also be used
    # as a scratch array for planning which may destroy its input.
    mtf = Array{Complex{T}}(undef, dims)

    # Compute the plans with FFTW flags suitable for out-of-place forward
    # transform and in-place backward transform.
    forward = plan_fft(mtf; flags = (planning | FFTW.PRESERVE_INPUT), kwds...)
    backward = plan_bfft!(mtf; flags = (planning | FFTW.DESTROY_INPUT), kwds...)

    # Compute the MTF.
    mul!(mtf, forward, (shift ? ifftshift(psf) : psf))

    # Build the operator.
    F = typeof(forward)
    B = typeof(backward)
    CirculantConvolution{Complex{T},N,Complex{T},F,B}(dims, dims, mtf,
                                                      forward, backward)
end

function vcreate(::Type{<:Operations},
                 H::CirculantConvolution{T,N},
                 x::AbstractArray{T,N},
                 scratch::Bool = false) where {T<:fftwNumber,N}
    return Array{T,N}(undef, H.dims)
end

function apply!(α::Real,
                P::Type{<:Union{Direct,Adjoint}},
                H::CirculantConvolution{Complex{T},N,Complex{T}},
                x::AbstractArray{Complex{T},N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Complex{T},N}) where {T<:fftwReal,N}
    @assert !Base.has_offset_axes(x, y)
    if α == 0
        @assert size(y) == H.dims
        vscale!(y, β)
    else
        n = length(x)
        if β == 0
            # Use y as a workspace.
            mul!(y, H.forward, x) # out-of-place forward FFT of x in y
            _apply!(y, α/n, P, H.mtf) # in-place multiply y by mtf/n
            mul!(y, H.backward, y) # in-place backward FFT of y
        else
            # Must allocate a workspace.
            z = Array{Complex{T}}(undef, H.zdims) # allocate temporary
            mul!(z, H.forward, x) # out-of-place forward FFT of x in z
            _apply!(z, α/n, P, H.mtf) # in-place multiply z by mtf/n
            mul!(z, H.backward, z) # in-place backward FFT of z
            vcombine!(y, 1, z, β, y)
        end
    end
    return y
end

function apply!(α::Real,
                P::Type{<:Union{Direct,Adjoint}},
                H::CirculantConvolution{T,N,Complex{T}},
                x::AbstractArray{T,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{T,N}) where {T<:fftwReal,N}
    @assert !Base.has_offset_axes(x, y)
    if α == 0
        @assert size(y) == H.dims
        vscale!(y, β)
    else
        n = length(x)
        z = Array{Complex{T}}(undef, H.zdims) # allocate temporary
        mul!(z, H.forward, x) # out-of-place forward FFT of x in z
        _apply!(z, α/n, P, H.mtf) # in-place multiply z by mtf/n
        if β == 0
            mul!(y, H.backward, z) # out-of-place backward FFT of z in y
        else
            w = Array{T}(undef, H.dims) # allocate another temporary
            mul!(w, H.backward, z) # out-of-place backward FFT of z in y
            vcombine!(y, 1, w, β, y)
        end
    end
    return y
end

# Manage to have `H(x)` works as `H*x`:
(H::CirculantConvolution{T,N})(x::AbstractArray{T,N}) where {T,N} = H*x
(H::Adjoint{CirculantConvolution{T,N}})(x::AbstractArray{T,N}) where {T,N} = H*x

"""
```julia
_apply!(arr, α, P, mtf)
```

stores in `arr` the elementwise multiplication of `arr` by `α*mtf` if `P` is
`Direct` or by `α*conj(mtf)` if `P` is `Adjoint`.  An error is thrown if the
arrays do not have the same dimensions.  It is assumed that `α ≠ 0`.

"""
function _apply!(arr::AbstractArray{Complex{T},N},
                 α::Real, ::Type{Direct},
                 mtf::AbstractArray{Complex{T},N}) where {T,N}
    @assert axes(arr) == axes(mtf)
    if α == 1
        @inbounds @simd for i in eachindex(arr, mtf)
            arr[i] *= mtf[i]
        end
    else
        alpha = convert(T, α)
        @inbounds @simd for i in eachindex(arr, mtf)
            arr[i] *= alpha*mtf[i]
        end
    end
end

function _apply!(arr::AbstractArray{Complex{T},N},
                 α::Real, ::Type{Adjoint},
                 mtf::AbstractArray{Complex{T},N}) where {T,N}
    @assert axes(arr) == axes(mtf)
    if α == 1
        @inbounds @simd for i in eachindex(arr, mtf)
            arr[i] *= conj(mtf[i])
        end
    else
        alpha = convert(T, α)
        @inbounds @simd for i in eachindex(arr, mtf)
            arr[i] *= alpha*conj(mtf[i])
        end
    end
end

#------------------------------------------------------------------------------
# Utilities.

"""

`check_flags(flags)` checks whether `flags` is an allowed bitwise-or
combination of FFTW planner flags (see
http://www.fftw.org/doc/Planner-Flags.html) and returns the filtered flags.

"""
function check_flags(flags::Integer)
    planning = flags & (FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT |
                        FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY)
    flags == planning ||
        throw(ArgumentError("only FFTW planning flags can be specified"))
    return UInt32(planning)
end

"""

`check_dimensions(dims)` checks whether the list of dimensions `dims` is
correct and returns the corresponding total number of elements.

"""
function check_dimensions(dims::NTuple{N,Int}) where {N}
    number = 1
    for i in 1:length(dims)
        dim = dims[i]
        dim ≥ 1 || throw(ArgumentError("invalid dimension(s)"))
        number *= dim
    end
    return number
end

"""

`get_dimension(dims, i)` yields the `i`-th dimension in tuple of integers
`dims`.  Like for broadcasting rules, it is assumed that the length of
all dimensions after the last one are equal to 1.

"""
get_dimension(dims::NTuple{N,Int}, i::Integer) where {N} =
    (i < 1 ? error("invalid dimension index") : i ≤ N ? dims[i] : 1)
# FIXME: should be in ArrayTools

"""
```julia
goodfftdim(len)
```

yields the smallest integer which is greater or equal `len` and which is a
multiple of powers of 2, 3 and/or 5.  If argument is an array dimesion list
(i.e. a tuple of integers), a tuple of good FFT dimensions is returned.

Also see: [`goodfftdims`](@ref), [`rfftdims`](@ref), [`FFTOperator`](@ref).

"""
goodfftdim(len::Integer) = goodfftdim(Int(len))
goodfftdim(len::Int) = nextprod([2,3,5], len)

"""
```julia
goodfftdims(dims)
```

yields a list of dimensions suitable for computing the FFT of arrays whose
dimensions are `dims` (a tuple or a vector of integers).

Also see: [`goodfftdim`](@ref), [`rfftdims`](@ref), [`FFTOperator`](@ref).

"""
goodfftdims(dims::Integer...) = map(goodfftdim, dims)
goodfftdims(dims::Union{AbstractVector{<:Integer},Tuple{Vararg{Integer}}}) =
    map(goodfftdim, dims)

"""
```julia
rfftdims(dims)
```

yields the dimensions of the complex array produced by a real-complex FFT of a
real array of size `dims`.

Also see: [`goodfftdim`](@ref), [`FFTOperator`](@ref).

"""
rfftdims(dims::Integer...) = rfftdims(dims)
rfftdims(dims::NTuple{N,Integer}) where {N} =
    ntuple(d -> (d == 1 ? (Int(dims[d]) >> 1) + 1 : Int(dims[d])), Val(N))

"""
### Generate Discrete Fourier Transform frequency indexes or frequencies

Syntax:

```julia
k = fftfreq(dim)
f = fftfreq(dim, step)
```

With a single argument, the function returns a vector of `dim` values set with
the frequency indexes:

```
k = [0, 1, 2, ..., n-1, -n, ..., -2, -1]   if dim = 2*n
k = [0, 1, 2, ..., n,   -n, ..., -2, -1]   if dim = 2*n + 1
```

depending whether `dim` is even or odd.  These rules are compatible to what is
assumed by `fftshift` (which to see) in the sense that:

```
fftshift(fftfreq(dim)) = [-n, ..., -2, -1, 0, 1, 2, ...]
```

With two arguments, `step` is the sample spacing in the direct space and the
result is a floating point vector with `dim` elements set with the frequency
bin centers in cycles per unit of the sample spacing (with zero at the start).
For instance, if the sample spacing is in seconds, then the frequency unit is
cycles/second.  This is equivalent to:

```
fftfreq(dim)/(dim*step)
```

See also: [`FFTOperator`](@ref), [`fftshift`](@ref).

"""
function fftfreq(_dim::Integer)
    dim = Int(_dim)
    n = div(dim, 2)
    f = Array{Int}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = k - 1
        end
        for k in dim-n+1:dim
            f[k] = k - (1 + dim)
        end
    end
    return f
end

function fftfreq(_dim::Integer, step::Real)
    dim = Int(_dim)
    scl = Cdouble(1/(dim*step))
    n = div(dim, 2)
    f = Array{Cdouble}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = (k - 1)*scl
        end
        for k in dim-n+1:dim
            f[k] = (k - (1 + dim))*scl
        end
    end
    return f
end

end # module
