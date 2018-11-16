#
# fft.jl -
#
# Implement FFT operator.
#
#------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
#

module FFT

# Be nice with the caller: re-export `fftshift` and `ifftshift` but not `fft`,
# `ifft` etc. as the `FFTOperator` is meant to replace them.
export
    FFTOperator,
    fftfreq,
    fftshift,
    goodfftdim,
    goodfftdims,
    ifftshift,
    rfftdims

using Compat

using ..LazyAlgebra
import ..LazyAlgebra: adjoint, apply!, vcreate, MorphismType, mul!,
    input_size, input_ndims, input_eltype,
    output_size, output_ndims, output_eltype,
    is_same_mapping
using ..LazyAlgebra: _merge_mul

import Base: *, /, \, inv, show

import AbstractFFTs: Plan, fftshift, ifftshift

using FFTW
import FFTW: fftwNumber, fftwReal, fftwComplex

abstract type Direction end
struct Forward  <: Direction end
struct Backward <: Direction end

# The time needed to allocate temporary arrays is negligible compared to the
# time taken to computate a FFT (e.g., 5µs to allocate a 256×256 array of
# double precision complexes versus 1.5ms to compute its FFT).  We therefore do
# not store any temporary arrays in the FFT operator.  Only the FFT plans are
# cached in the operator.

"""
```julia
FFTOperator(A) -> F
```

yields an FFT operator suitable for computing the fast Fourier transform of
arrays similar to `A`.  The operator can also be specified by the real/complex
floating-point type of the elements of the arrays to transform and their
dimensions:

```julia
FFTOperator(T, dims) -> F
```

where `T` is one of `Float64`, `Float32` (for a real-complex FFT),
`Complex{Float64}`, `Complex{Float32}` (for a complex-complex FFT) and `dims`
gives the dimensions of the arrays to transform (by the `Direct` or
`InverseAdjoint` operation).

The interest of creating such an operator is that it caches the ressources
necessary for fast computation of the FFT and can be therefore *much* faster
than calling `fft`, `rfft`, `ifft`, etc.  This is especially true on small
arrays.  Keywords `flags` and `timelimit` may be used to specify planning
options and time limit to create the FFT plans (see
http://www.fftw.org/doc/Planner-Flags.html).

Another advantage is that the returned object is a linear mapping which can be
used as any other mapping:

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
                   C<:fftwComplex, # element type of output
                   N,              # number of dimensions
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
                     flags::Integer = FFTW.ESTIMATE,
                     timelimit::Real = FFTW.NO_TIMELIMIT) where {T<:fftwReal,N}
    # Check arguments and build dimension list of the result of the forward
    # real-to-complex (r2c) transform.
    planning = check_flags(flags)
    ncols = check_dimensions(dims)
    zdims = rfftdims(dims)

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms).
    forward = plan_rfft(Array{T}(undef, dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)
    backward = plan_brfft(Array{Complex{T}}(undef, zdims), dims[1];
                          flags = (planning  | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)

    # Build operator.
    F = typeof(forward)
    B = typeof(backward)
    return FFTOperator{T,Complex{T},N,F,B}(ncols, dims, zdims,
                                           forward, backward)
end

# Complex-to-complex FFT.
function FFTOperator(::Type{Complex{T}},
                     dims::NTuple{N,Int};
                     flags::Integer = FFTW.ESTIMATE,
                     timelimit::Real = FFTW.NO_TIMELIMIT) where {T<:fftwReal,N}
    # Check arguments.  The input and output of the complex-to-complex
    # transform have the same dimensions.
    planning = check_flags(flags)
    ncols = check_dimensions(dims)
    temp = Array{Complex{T}}(undef, dims)

    # Compute the plans with suitable FFTW flags.  The forward and backward
    # transforms must preserve their input.
    forward = plan_fft(temp; flags = (planning | FFTW.PRESERVE_INPUT),
                       timelimit = timelimit)
    backward = plan_bfft(temp; flags = (planning | FFTW.PRESERVE_INPUT),
                         timelimit = timelimit)

    # Build operator.
    F = typeof(forward)
    B = typeof(backward)
    return FFTOperator{Complex{T},Complex{T},N,F,B}(ncols, dims, dims,
                                                    forward, backward)
end

FFTOperator(T::Type{<:fftwReal}, dims::Tuple{Vararg{Integer}}; kwds...) =
    FFTOperator(T, map(Int, dims); kwds...)

FFTOperator(T::Type{Complex{<:fftwReal}}, dims::Tuple{Vararg{Integer}}; kwds...) =
    FFTOperator(T, map(Int, dims); kwds...)

FFTOperator(arr::Array{T,N}; kwds...) where {T<:fftwNumber,N} =
    FFTOperator(eltype(arr), size(arr); kwds...)

# Traits:
MorphismType(::FFTOperator{<:Complex}) = Endomorphism

ncols(A::FFTOperator) = A.ncols
ncols(A::Adjoint{<:FFTOperator}) = ncols(operand(A))
ncols(A::Inverse{<:FFTOperator}) = ncols(operand(A))
ncols(A::InverseAdjoint{<:FFTOperator}) = ncols(operand(A))

input_size(A::FFTOperator) = A.inpdims
input_size(A::FFTOperator, d) = A.inpdims[d]
output_size(A::FFTOperator) = A.outdims
output_size(A::FFTOperator, d) = A.outdims[d]
input_ndims(A::FFTOperator{T,C,N}) where {T,C,N} = N
output_ndims(A::FFTOperator{T,C,N}) where {T,C,N} = N
input_eltype(A::FFTOperator{T,C,N}) where {T,C,N} = T
output_eltype(A::FFTOperator{T,C,N}) where {T,C,N} = C

# 2 FFT operators can be considered the same if they operate on arguments with
# the same element type and the same dimensions.  If the types do not match,
# the matching method is the one which return false, so it is only needed to
# implement the method for two arguments with the same types (omitting the type
# of the plans as it is irrelevant here).
is_same_mapping(A::FFTOperator{T,C,N}, B::FFTOperator{T,C,N}) where {T,C,N} =
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

macro checksize(name, arg, dims)
    return quote
        size($(esc(arg))) == $(esc(dims)) || badsize($(esc(name)), $(esc(dims)))
    end
end

@noinline badsize(name::String, dims::Tuple{Vararg{Integer}}) =
    throw(DimensionMismatch("$name must have dimensions $dims"))

function vcreate(P::Type{<:Union{Forward,Direct,InverseAdjoint}},
                 A::FFTOperator{T,C,N},
                 x::DenseArray{T,N},
                 scratch::Bool=false) where {T,C,N}
    @checksize "argument" x input_size(A)
    return (scratch && T === C ? x : Array{C}(undef, output_size(A)))
end

function vcreate(P::Type{<:Union{Backward,Adjoint,Inverse}},
                 A::FFTOperator{T,C,N},
                 x::DenseArray{C,N},
                 scratch::Bool=false) where {T,C,N}
    @checksize "argument" x output_size(A)
    return (scratch && T === C ? x : Array{T}(undef, input_size(A)))
end

function apply!(α::Real,
                ::Type{Direct},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {T,C,N}
    return apply!(α, Forward, A, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{Adjoint},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{T,N}) where {T,C,N}
    return apply!(α, Backward, A, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{Inverse},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{T,N}) where {T,C,N}
    return apply!(α/A.ncols, Backward, A, x, scratch, β, y)
end

function apply!(α::Real,
                ::Type{InverseAdjoint},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {T,C,N}
    return apply!(α/A.ncols, Forward, A, x, scratch, β, y)
end

# We want to compute:
#
#    y = α⋅F^P⋅x + β⋅y
#
# with as few temporaries as possible.  If β = 0, then there are no needs to
# save the contents of y which can be used directly for the output of the
# transform.

# Apply forward transform.
function apply!(α::Real,
                ::Type{Forward},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {T,C,N}
    @checksize "argument" x  input_size(A)
    @checksize "result"   y output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        mul!(y, A.forward, x)
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Forward, A, x, scratch)
        mul!(z, A.forward, x)
        vcombine!(y, α, z, β, y)
    end
    return y
end

# Apply backward complex-to-complex transform.
function apply!(α::Real,
                ::Type{Backward},
                A::FFTOperator{C,C,N},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{C,N}) where {C<:fftwComplex,N}
    @checksize "argument" x output_size(A)
    @checksize "result"   y  input_size(A)
    size(x) == output_size(A) || bad_src_size()
    size(y) ==  input_size(A) || bad_dst_size()
    if α == 0
        vscale!(y, β)
    elseif β == 0
        mul!(y, A.backward, x)
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Backward, A, x, scratch)
        mul!(z, A.backward, x)
        vcombine!(y, α, z, β, y)
    end
    return y
end

# Apply backward complex-to-real (c2r) transform. Preserving input is not
# possible for multi-dimensional c2r transforms so we must copy the input
# argument x.
function apply!(α::Real,
                ::Type{Backward},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{T,N}) where {T<:fftwReal,C,N}
    @checksize "argument" x output_size(A)
    @checksize "result"   y  input_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        mul!(y, A.backward, (scratch ? x : vcopy(x)))
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Backward, A, x)
        mul!(z, A.backward, (scratch ? x : vcopy(x)))
        vcombine!(y, α, z, β, y)
    end
    return y
end

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
rfftdims(dims::NTuple{N,Int}) where {N} =
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
