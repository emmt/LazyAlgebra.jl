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
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# Copyright (C) 2017-2018, Éric Thiébaut.
#

module FFT

export FFTOperator

using ..LazyAlgebra
import ..LazyAlgebra: apply!, vcreate,
    input_size, input_ndims, input_eltype,
    output_size, output_ndims, output_eltype

import Base.FFTW: fftwNumber, fftwReal, fftwComplex

abstract type Direction end
struct Forward  <: Direction end
struct Backward <: Direction end

# The time needed to allocate temporary arrays is negligible compared to the
# time taken to computate a FFT (e.g. 20µs to allocate a 256×256 array of
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
F'*x    # yields the adjoint FFT of x
F\\x     # yields the inverse FFT of x
```

See also: [`fft`](@ref), [`plan_fft`](@ref), [`bfft`](@ref),
          [`plan_bfft`](@ref), [`rfft`](@ref), [`plan_rfft`](@ref),
          [`brfft`](@ref), [`plan_brfft`](@ref).

"""
struct FFTOperator{T<:fftwNumber,C<:fftwComplex,N,F,B} <: LinearMapping
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
    zdims = ntuple(i -> (i == 1 ? (dims[i] >> 1) + 1 : dims[i]), Val{N})

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms).
    forward = plan_rfft(Array{T}(dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)
    backward = plan_brfft(Array{Complex{T}}(zdims), dims[1];
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
                     flags::Integer=FFTW.ESTIMATE,
                     timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:fftwReal,N}
    # Check arguments.  The input and output of the complex-to-complex
    # transformhave the same dimensions.
    planning = check_flags(flags)
    ncols = check_dimensions(dims)
    temp = Array{Complex{T}}(dims)

    # Compute the plans with suitable FFTW flags.  The forward and backward
    # transform must preserve their input.
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

FFTOperator(arr::Array{T,N}; kwds...) where {T<:fftwNumber,N} =
    FFTOperator(eltype(arr), size(arr); kwds...)

# Traits:
morphismtype(::FFTOperator{<:Complex}) = Endomorphism

input_size(A::FFTOperator) = A.inpdims
input_size(A::FFTOperator, i::Integer) = A.inpdims[i]
output_size(A::FFTOperator) = A.outdims
output_size(A::FFTOperator, i::Integer) = A.outdims[i]
input_ndims(A::FFTOperator{T,C,N}) where {T,C,N} = N
output_ndims(A::FFTOperator{T,C,N}) where {T,C,N} = N
input_eltype(A::FFTOperator{T,C,N}) where {T,C,N} = T
output_eltype(A::FFTOperator{T,C,N}) where {T,C,N} = C

function vcreate(::Type{P},
                 A::FFTOperator{T,C,N},
                 x::DenseArray{T,N}) where {P<:Union{Forward,Direct,
                                                     InverseAdjoint},T,C,N}
    return Array{C}(output_size(A))
end

function vcreate(::Type{P},
                 A::FFTOperator{T,C,N},
                 x::DenseArray{C,N}) where {P<:Union{Backward,Adjoint,Inverse},
                                            T,C,N}
    return Array{T}(input_size(A))
end

function apply!(α::Scalar,
                ::Type{Direct},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                β::Scalar,
                y::DenseArray{C,N}) where {T,C,N}
    return apply!(α, Forward, A, x, β, y)
end

function apply!(α::Scalar,
                ::Type{Adjoint},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                β::Scalar,
                y::DenseArray{T,N}) where {T,C,N}
    return apply!(α, Backward, A, x, β, y)
end

function apply!(α::Scalar,
                ::Type{Inverse},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                β::Scalar,
                y::DenseArray{T,N}) where {T,C,N}
    return apply!(Scalar(α/A.ncols), Backward, A, x, β, y)
end

function apply!(α::Scalar,
                ::Type{InverseAdjoint},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                β::Scalar,
                y::DenseArray{C,N}) where {T,C,N}
    return apply!(Scalar(α/A.ncols), Forward, A, x, β, y)
end

# We want to compute:
#
#    y = α⋅F^P⋅x + β⋅y
#
# with as few temporaries as possible.  If β = 0, then there are no needs to
# save the contents of y which can be used directly for the output of the
# transform.

# Apply forward transform.
function apply!(α::Scalar,
                ::Type{Forward},
                A::FFTOperator{T,C,N},
                x::DenseArray{T,N},
                β::Scalar,
                y::DenseArray{C,N}) where {T,C,N}
    @assert size(x) == input_size(A)
    @assert size(y) == output_size(A)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        A_mul_B!(y, A.forward, x)
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Forward, A, x)
        A_mul_B!(z, A.forward, x)
        vcombine!(y, α, z, β, y)
    end
    return y
end

# Apply backward complex-to-complex transform.
function apply!(α::Scalar,
                ::Type{Backward},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                β::Scalar,
                y::DenseArray{T,N}) where {T<:fftwComplex,C,N}
    if α == 0
        vscale!(y, β)
    elseif β == 0
        A_mul_B!(y, A.backward, x)
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Backward, A, x)
        A_mul_B!(z, A.backward, x)
        vcombine!(y, α, z, β, y)
    end
    return y
end

# Apply backward complex-to-real (c2r) transform. Preserving input is not
# possible for multi-dimensional c2r transforms so we must copy the input
# argument x.
function apply!(α::Scalar,
                ::Type{Backward},
                A::FFTOperator{T,C,N},
                x::DenseArray{C,N},
                β::Scalar,
                y::DenseArray{T,N};
                overwriteinput::Bool=false) where {T<:fftwReal,C,N}
    if α == 0
        vscale!(y, β)
    elseif β == 0
        A_mul_B!(y, A.backward, (overwriteinput ? x : vcopy(x)))
        α == 1 || vscale!(y, α)
    else
        z = vcreate(Backward, A, x)
        A_mul_B!(z, A.backward, (overwriteinput ? x : vcopy(x)))
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
    if flags != planning
        throw(ArgumentError("only FFTW planning flags can be specified"))
    end
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
        if dim < 1
            error("invalid dimension(s)")
        end
        number *= dim
    end
    return number
end

end # module
