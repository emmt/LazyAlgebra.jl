# Implementation of FFT and circulant convolution operators.
#
#-------------------------------------------------------------------------- FFT OPERATOR -

"""
    F = FFT(forward)
    F = FFT(forward, backward)

builds a fast Fourier transform (FFT) operator based on the given `forward` and `backward`
FFT plans. If not specified, the `backward` plan is automatically built from the `forward`
plan.

Another possibility is:

    F = FFT(x; kwds...)

which builds an FFT operator suitable for computing the FFT of arrays similar to `x`. The
operator can also be specified by the real/complex floating-point type of the elements of
the arrays to transform and their dimensions:

   F =  FFT{T}(shape...; kwds...)

where `T` is one of `Float64`, `Float32` (for a real-complex FFT), `Complex{Float64}`, or
`Complex{Float32}` (for a complex-complex FFT) and `shape...` are the dimensions or axes
of the arrays to transform (by the forward FFT).

Keywords `flags` and `timelimit` may be used to specify planning options and time limit to
create the FFT plans (see http://www.fftw.org/doc/Planner-Flags.html). The defaults are
`flags=FFTW.MEASURE` and no time limit.

The interest of creating such an operator is that it caches the resources necessary for
fast computation of the FFT and can be therefore *much* faster than calling `fft`, `rfft`,
`ifft`, etc. This is especially true on small arrays.

An instance of `FFT` behaves as any other linear operator of `LazyAlgebra`:

```julia
F*x     # yields the FFT of x
F'*x    # yields the adjoint FFT applied to x, that is the backward FFT of x
F\\x     # yields the inverse FFT of x
```

See also [`vmul`](@ref), [`vmul!`](@ref), and [`LazyAlgebra.Operator`](@ref).

"""
FFT(forward::FFTW.FFTWPlan) = FFT(forward, inv(forward).p)

function FFT{T}(dims::Dims{N};
                timelimit::Real = FFTW.NO_TIMELIMIT,
                flags::Integer = FFTW.MEASURE) where {T<:FFTW.fftwNumber,N}
    # Get planning flags.
    flags = check_fftw_flags(flags)
    temp = Array{T}(undef, dims)
    if T <: Complex
        # Compute the plans with suitable FFTW flags for a complex-to-complex FFT
        # operator. For maximum efficiency, the transforms are applied in-place and thus
        # cannot preserve their inputs.
        forward = plan_fft!(temp; flags = (flags | FFTW.DESTROY_INPUT),
                            timelimit = timelimit)
        backward = plan_bfft!(temp; flags = (flags | FFTW.DESTROY_INPUT),
                              timelimit = timelimit)
    else
        # Compute the plans with suitable FFTW flags for a real-to-complex FFT operator.
        # The forward transform (r2c) shall preserve its input, while the backward
        # transform (c2r) may destroy it (in fact there are no input-preserving algorithms
        # for multi-dimensional c2r transforms implemented in FFTW, see
        # http://www.fftw.org/doc/Planner-Flags.html).
        forward = plan_rfft(temp; flags = (flags | FFTW.PRESERVE_INPUT),
                            timelimit = timelimit)
        backward = plan_brfft(Array{Complex{T}}(undef, rfftdims(dims)), dims[1];
                              flags = (flags | FFTW.DESTROY_INPUT),
                              timelimit = timelimit)
    end
    return FFT(forward, backward)
end

FFT{T}(shape::eltype(ArrayShape)...; kwds...) where {T<:FFTW.fftwNumber} =
    FFT{T}(shape; kwds...)

FFT{T}(shape::ArrayShape; kwds...) where {T<:FFTW.fftwNumber} =
    FFT{T}(as_array_size(shape); kwds...)

FFT(A::AbstractArray; kwds...) = FFT{float(eltype(A))}(axes(A); kwds...)

# Conversion constructors.
FFT(A::FFT) = A
FFT{T}(A::FFT{T}) where {T<:FFTW.fftwNumber} = A
function FFT{T}(A::FFT) where {T<:FFTW.fftwNumber}
    (T <: Complex) == (input_eltype(A) <: Complex) || throw(ArgumentError(
        "input element types must be both real or both complex"))
    flags = get_plan(A).flags & ~(FFTW.PRESERVE_INPUT|FFTW.DESTROY_INPUT)
    dims = input_size(A)
    return FFT{T}(dims; flags = flags)
end

# Accessors and LazyAlgebra operator API for FFT operators.
OutputShape(::Type{<:FFT{T,C,N}}) where {T,C,N} = HasOutputShape{N}()
output_size(A::FFT) = output_size(get_plan(A))
output_axes(A::FFT) = map(Base.OneTo, output_size(A))

InputShape(::Type{<:FFT{T,C,N}}) where {T,C,N} = HasInputShape{N}()
input_size(A::FFT) = input_size(get_plan(A))
input_axes(A::FFT) = map(Base.OneTo, input_size(A))

InputEltype(::Type{<:FFT}) = HasInputEltype()
input_eltype(::Type{<:Union{F,InverseAdjoint{F}}}) where {T,C,N,F<:FFT{T,C,N}} = T
input_eltype(::Type{<:Union{Adjoint{F},Inverse{F}}}) where {T,C,N,F<:FFT{T,C,N}} = C

OutputEltype(::Type{<:FFT}) = HasOutputEltype()
output_eltype(::Type{<:Union{F,InverseAdjoint{F}}}) where {T,C,N,F<:FFT{T,C,N}} = C
output_eltype(::Type{<:Union{Adjoint{F},Inverse{F}}}) where {T,C,N,F<:FFT{T,C,N}} = T

# Other accessors.
get_plan(A::FFT) = getfield(A, :forward)
get_plan(A::InverseAdjoint{<:FFT}) = get_plan(A[][])
get_plan(A::Union{Adjoint{F},Inverse{F}}) where {F<:FFT} = getfield(A[], :backward)

fft_length(A::Union{F,InverseAdjoint{F}}) where {F<:FFT} = ncols(A)
fft_length(A::Union{Adjoint{F},Inverse{F}}) where {F<:FFT} = nrows(A)

# Precision.
get_precision(::Type{<:FFT{T}}) where {T} = real(T)
_with_precision(::Type{T}, A::FFT{<:Union{T,Complex{T}}}) where {T<:AbstractFloat} = A
_with_precision(::Type{T}, A::FFT{<:Real}) where {T<:AbstractFloat} =
    FFT{T}(A)
_with_precision(::Type{T}, A::FFT{<:Complex}) where {T<:AbstractFloat} =
    FFT{Complex{T}}(A)

function unsafe_vmul!(α::Number, A::Union{F,Adjoint{F}},
                      x::AbstractArray{<:Any,N}, β::Number, y::AbstractArray{<:Any,N},
                      scratch::Bool = false) where {T,C,N,F<:FFT{T,C,N}}
    return unsafe_vmul!(α, get_plan(A), x, β, y, scratch)
end

function unsafe_vmul!(α::Number, A::Union{Inverse{F},InverseAdjoint{F}},
                      x::AbstractArray{<:Any,N}, β::Number, y::AbstractArray{<:Any,N},
                      scratch::Bool = false) where {T,C,N,F<:FFT{T,C,N}}
    λ = with_precision(get_precision(A), divide(α, fft_length(A)))
    return unsafe_vmul!(λ, get_plan(A), x, β, y, scratch)
end

# 2 FFT operators yield the same result if they operate on arguments with the same element
# type and the same dimensions. If the types do not match, the matching method is the one
# which return false, so it is only needed to implement the method for two arguments with
# the same types (omitting the type of the plans as it is irrelevant here).
Base.:(==)(A::FFT{T,C,N}, B::FFT{T,C,N}) where {T,C,N} =
    (input_size(A) == input_size(B))

brief(io::IO, A::FFT) = write(io, "FFT")
Base.show(io::IO, ::MIME"text/plain", A::FFT) = show(io, A)
function Base.show(io::IO, A::FFT)
    print(io, "FFT{", input_eltype(A), "}(")
    print_shape(io, input_axes(A))
    print(io, "; flags = ")
    print_fftw_flags(io, get_plan(A).flags)
    print(io, ")")
    nothing
end


# Impose the following simplifying rules:
#     inv(F) = n\F'
#     ==> F⋅F' = F'⋅F = n⋅Id
#     ==> inv(F⋅F') = inv(F'⋅F) = inv(F)⋅inv(F') = inv(F')⋅inv(F) = n\Id

try_simplify((A,B)::Prod{Adjoint{F},F}) where {F<:FFT} =
    isequal(A[], B) ? fft_length(A)*Id : nothing

try_simplify((A,B)::Prod{F,Adjoint{F}}) where {F<:FFT} =
    isequal(A, B[]) ? fft_length(A)*Id : nothing

try_simplify((A,B)::Prod{InverseAdjoint{F},Inverse{F}}) where {F<:FFT} =
    isequal(A[][], B[]) ? (1//fft_length(A))*Id : nothing

try_simplify((A,B)::Prod{Inverse{F},InverseAdjoint{F}}) where {F<:FFT} =
    isequal(A[], B[][]) ? (1//fft_length(A))*Id : nothing

#---------------------------------------------------------------------------- FFTW PLANS -

fft_type(::FFTW.cFFTWPlan{<:Complex}) = "c2c"
fft_type(::FFTW.rFFTWPlan{<:Complex}) = "c2r"
fft_type(::FFTW.rFFTWPlan{<:Real}) = "r2c"

function check_fftw_plans(forward::FFTW.FFTWPlan{Tf,Kf},
                          backward::FFTW.FFTWPlan{Tb,Kb}) where {Tf,Kf,Tb,Kb}
    Kb == -Kf || throw(ArgumentError(
        "forward and backward FFT plans have the same \"direction\""))
    input_size(backward) == output_size(forward) &&
        output_size(backward) == input_size(forward) || throw(DimensionMismatch(
            "forward and backward FFT plans have incompatible dimensions"))
    real(Tf) === real(Tb) || throw(ArgumentError(
        "forward and backward FFT plans have different floating-point types"))
    forward isa FFTW.cFFTWPlan && backward isa FFTW.cFFTWPlan && return nothing
    forward isa FFTW.rFFTWPlan{<:Complex} && backward isa FFTW.rFFTWPlan{<:Real} && return nothing
    forward isa FFTW.rFFTWPlan{<:Real} && backward isa FFTW.rFFTWPlan{<:Complex} && return nothing
    throw(ArgumentError(
            "$(fft_type(forward)) forward FFT is not compatible with $(fft_type(backward)) backward FFT"))
end

for P in (:cFFTWPlan, :rFFTWPlan)
    @eval begin
        InputShape(::Type{<:FFTW.$P{T,K,inplace,N,G}}) where {T,K,inplace,N,G} =
            HasInputShape{N}()
        OutputShape(::Type{<:FFTW.$P{T,K,inplace,N,G}}) where {T,K,inplace,N,G} =
            HasOutputShape{N}()
    end
end

input_size(A::FFTW.FFTWPlan) = A.sz
input_axes(A::FFTW.FFTWPlan) = map(Base.OneTo, input_size(A))

output_size(A::FFTW.FFTWPlan) = A.osz
output_axes(A::FFTW.FFTWPlan) = map(Base.OneTo, output_size(A))

InputEltype(::Type{<:FFTW.FFTWPlan}) = HasInputEltype()
input_eltype(A::FFTW.FFTWPlan) = input_eltype(typeof(A))
input_eltype(::Type{<:FFTW.FFTWPlan{T}}) where {T} = T

OutputEltype(::Type{<:FFTW.FFTWPlan}) = HasOutputEltype()
output_eltype(A::FFTW.FFTWPlan) = output_eltype(typeof(A))
output_eltype(::Type{<:FFTW.FFTWPlan{T}}) where {T} = T
output_eltype(::Type{<:FFTW.rFFTWPlan{T}}) where {T<:Real} = Complex{T}
output_eltype(::Type{<:FFTW.rFFTWPlan{Complex{T}}}) where {T<:Real} = T

# Unfortunately, Julia interface to FFTW only records the flags passed to the FFTW
# library, not the actual flags. So we must be conservative.
does_not_destroy_input(A::FFTW.FFTWPlan) = !iszero(A.flags & FFTW.PRESERVE_INPUT)

# Extend `unsafe_vmul!` for FFTW plans.
#
# For  FFTW plan `P`, we want to compute:
#
#    y = α⋅F⋅x + β⋅y
#
# with as few temporaries as possible. If `β = 0`, then there are no needs to save the
# contents of `y` which can be used directly for the output of the transform. Extra checks
# are required to make sure the contents `x` is not damaged unless `scratch` is true. It
# turns out that the implementation depends on the type of transform so several versions
# are coded below.
#
# NOTE The machinery of FFTW plans is quite involved with many different possible types of
#      plans, the adjoint of a plan is a specific object of type
#      `AbstractFFTs.AdjointPlan~, an inverse-FFT plan is a scaled plan of type
#      `AbstractFFTs.ScaledPlan`, the inverse of a plan is cached in the plan, etc. We
#      therefore only extend LazyAlgebra Operator API for a definite subset of FFTW plans
#      used by `FFT`.
#
# NOTE In principle, FFTW plans can be applied to strided arrays (StridedArray) but this
#      imposes that the arguments have the same strides. So for now, we choose to restrict
#      arguments to arrays with contiguous elements (DenseArray).
#
# For a complex-to-complex (c2c) transform, the type of plan returned by `plan_fft`,
# `plan_bfft`, `plan_fft!`, and `plan_bfft!` is:
#
#     FFTW.cFFTWPlan{Complex{T}, K, inplace, N, ...}
#
# with `T` the floating-point type, `K` is `-1` for the forward transform and `+1` for the
# backward transform, `inplace` indicates whether the transform is in-place or
# out-of-place (true with the `!` suffix, false otherwise), and `N` the number of dimensions.
#
function unsafe_vmul!(α::Number, A::FFTW.cFFTWPlan{Complex{T},K,inplace,N},
                      x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N},
                      scratch::Bool = false) where {T,N,inplace,K}
    if y isa DenseArray{Complex{T},N} && iszero(β)
        # `y` can be used as the output of the transform.
        if inplace
            x === y || unsafe_vscale!(y, α, x) # copy and scale without checking axes nor dispatching
            mul!(y, A, y) # in-place transform
            x === y || isone(α) || unsafe_vscale!(y, α) # scale without dispatching if needed
        elseif x isa DenseArray{Complex{T},N} && (scratch || does_not_destroy_input(A)) && !Base.mightalias(x, y)
            # `x` can be used as the input of the out-of-place transform.
            mul!(y, A, x) # out-of-place transform
            isone(α) || unsafe_vscale!(y, α) # scale without dispatching if needed
        else
            # `x` is copied (and scaled) in a temporary array.
            w = Array{Complex{T}}(undef, size(x))
            unsafe_vscale!(w, α, x) # copy and scale without checking axes nor dispatching
            mul!(y, A, w)
        end
    else
        # A temporary array must be used for the output of the transform.
        z = Array{Complex{T}}(undef, size(y))
        if inplace
            unsafe_vcopy!(z, x)
            mul!(z, A, z)
        elseif x isa DenseArray{Complex{T},N} && (scratch || does_not_destroy_input(A))
            # `x` can be used as the input of the transform.
            mul!(z, A, x)
        else
            # `x` is copied in a temporary array.
            w = Array{Complex{T}}(undef, size(x))
            unsafe_vcopy!(w, x)
            mul!(z, A, w)
        end
        unsafe_vcombine!(y, α, z, β, y) # combine without checking axes nor dispatching
    end
    return y
end
#
# For a real-to-complex (r2c) and complex-to-real (c2r) transforms, the types of plan
# returned by `plan_rfft` and by `plan_brfft` is respectively:
#
#     FFTW.rFFTWPlan{T, -1, false, N, ...}
#     FFTW.rFFTWPlan{Complex{T}, 1, false, N, ...}
#
# In FFTW, the r2c and c2r transforms are always out-of-place and may destroy their input;
# the multi-dimensional c2r always destroys its input.
#
function unsafe_vmul!(α::Number, A::FFTW.rFFTWPlan{<:Any,K,false,N},
                      x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N},
                      scratch::Bool = false) where {K,N}
    I = input_eltype(A)
    O = output_eltype(A)
    if x isa DenseArray{I,N} && (scratch || does_not_destroy_input(A))
        # `x` can be used as input to the transform.
        if y isa DenseArray{O,N} && iszero(β) # FIXME: && !Base.mightalias(x, y)
            # `y` can be used for the output of the transform.
            mul!(y, A, x)
            isone(α) || unsafe_vscale!(y, α)
        else
            # Use a temporary array for the output of the transform.
            z = Array{O,N}(undef, size(y))
            unsafe_vcombine!(y, α, mul!(z, A, x), β, y)
        end
    else
        # `x` is not directly suitable as input to the transform.
        w = Array{I,N}(undef, size(x))
        unsafe_vcopy!(w, x)
        unsafe_vmul!(α, A, w, β, y, true)
    end
    return y
end

#----------------------------------------------------------------- CIRCULANT CONVOLUTION -

# Basic methods for a linear operator on Julia's arrays.

InputShape(::Type{<:CirculantConvolution{T,N}}) where {T,N} = HasInputShape{N}()
input_size(H::CirculantConvolution) = H.dims
input_axes(H::CirculantConvolution) = map(Base.OneTo, input_size(H))

OutputShape(::Type{<:CirculantConvolution{T,N}}) where {T,N} = HasOutputShape{N}()
output_size(H::CirculantConvolution) = input_size(H)
output_axes(H::CirculantConvolution) = input_axes(H)

InputEltype(::Type{<:CirculantConvolution}) = HasInputEltype()
input_eltype(::Type{<:CirculantConvolution{T,N}}) where {T,N} = T

OutputEltype(::Type{<:CirculantConvolution}) = HasOutputEltype()
output_eltype(::Type{<:CirculantConvolution{T,N}}) where {T,N} = T

# Basic methods for an array.
Base.eltype(::Type{<:CirculantConvolution{T,N}}) where {T,N} = T
Base.ndims( ::Type{<:CirculantConvolution{T,N}}) where {T,N} = 2N
Base.size(H::CirculantConvolution{T,N}) where {T,N} =
    ntuple(i -> H.dims[(i ≤ N ? i : i - N)], 2N)
Base.size(H::CirculantConvolution{T,N}, i::Integer) where {T,N} =
    (i < 1 ? bad_dimension_index() : i ≤ N ? H.dims[i] : i ≤ 2N ? H.dims[i-N] : 1)

"""
# Circulant convolution operator

The circulant convolution operator `H` is defined by:

```julia
H = (1/n)*F'*Diag(mtf)*F
```

with `n` the number of elements, `F` the discrete Fourier transform operator and `mtf` the
*Modulation Transfer Function*.

The operator `H` can be created by:

```julia
H = CirculantConvolution(psf; flags=FFTW.MEASURE, timelimit=Inf, shift=false)
```

where `psf` is the point spread function (PSF). Note that the PSF is assumed to be
centered according to the convention of the discrete Fourier transform. You may use
`ifftshift` or the keyword `shift` if the PSF is geometrically centered:

```julia
H = CirculantConvolution(ifftshift(psf))
H = CirculantConvolution(psf, shift=true)
```

The following keywords can be specified:

* `shift` (`false` by default) indicates whether to apply `ifftshift` to `psf`.

* `normalize` (`false` by default) indicates whether to divide `psf` by the sum of its
  values. This keyword is only available for real-valued PSF.

* `flags` is a bitwise-or of FFTW planner flags, defaulting to `FFTW.MEASURE`. If the
  operator is to be used many times (as in iterative methods), it is recommended to use at
  least `flags=FFTW.MEASURE` (the default) which generally yields faster transforms
  compared to `flags=FFTW.ESTIMATE`.

* `timelimit` specifies a rough upper bound on the allowed planning time, in seconds.

The operator can be used as a regular linear operator: `H(x)` or `H*x` to compute the
convolution of `x` and `H'(x)` or `H'*x` to apply the adjoint of `H` to `x`.

For a slight improvement of performances, an array `y` to store the result of the
operation can be provided:

```julia
vmul!(y, H, x) -> y
vmul!(y, inv(H), x) -> y
vmul!(y, H', x) -> y
```

If provided, `y` must be at a different memory location than `x`.

"""
function CirculantConvolution(psf::AbstractArray; kwds...)
    T = float(eltype(psf))
    T <: FFTW.fftwNumber || throw(ArgumentError(
        "unsupported element type `$(eltype(psf))` for the PSF"))
    return CirculantConvolution(convert(Array{T}, psf); kwds...)
end

function CirculantConvolution(psf::AbstractArray{<:Union{T,Complex{T}},N};
                              flags::Integer = FFTW.MEASURE,
                              normalize::Bool = false,
                              shift::Bool = false,
                              kwds...) where {T<:FFTW.fftwReal,N}
    flags = check_fftw_flags(flags)

    # Allocate array for the scaled MTF, this array also serves as a workspace for
    # planning operations which may destroy their input.
    dims = size(psf)
    mtf = Array{Complex{T}}(undef, eltype(psf) <: Real ? rfftdims(dims) : dims)

    if eltype(psf) <: Real
        # Build an operator for arrays of reals.
        #
        # Compute the plans with suitable FFTW flags. The forward transform (r2c) must
        # preserve its input, while the backward transform (c2r) may destroy it (in fact
        # there are no input-preserving algorithms for multi-dimensional c2r transforms).
        # However if the planning flags do not prevent it, the input of `plan_rfft` may be
        # overwritten to find the best strategy, so we use a temporary array here. The
        # `mtf` array is not yet instantiated, so its contents may be modified with no
        # problem by `plan_brfft`.
        F = plan_rfft(Array{T}(undef, dims); flags = (flags | FFTW.PRESERVE_INPUT), kwds...)
        B = plan_brfft(mtf, dims[1]; flags = (flags | FFTW.DESTROY_INPUT), kwds...)
    else
        # Build an operator for arrays of complexes.
        #
        # Compute the plans with FFTW flags suitable for out-of-place forward
        # transform and in-place backward transform.
        F = plan_fft(mtf; flags = (flags | FFTW.PRESERVE_INPUT), kwds...)
        B = plan_bfft!(mtf; flags = (flags | FFTW.DESTROY_INPUT), kwds...)
    end

    # Compute the scaled MTF *after* computing the plans.
    mul!(mtf, F, (shift ? ifftshift(psf) : psf))
    if normalize
        eltype(psf) <: Real || throw(ArgumentError(
            "normalizing a complex PSF makes no sense"))
        s = mtf[1] # FIXME: keep imaginary part?
        isone(s) || vscale!(mtf, inv(s))
    end

    # Build the operator.
    return CirculantConvolution(mtf, F, B)
end

# Unsafe assumptions:
#   1. dimensions (axes) of arguments have been checked;
#   2. multipliers `α` and `β` have suitable precision;
#   3. `α` is nonzero.
function unsafe_vmul!(α::Number,
                      H::Union{F,Adjoint{<:F}},
                      x::AbstractArray{Complex{T},N},
                      β::Number,
                      y::AbstractArray{Complex{T},N}) where {T<:FFTW.fftwReal,N,
                                                             F<:CirculantConvolution{
                                                                 Complex{T},Complex{T},N}}
    n = length(x)
    d = H isa Adjoint ? Diag(H.mtf)' : Diag(H.mtf)
    if F <: CirculantConvolution{<:Complex}
        if β == 𝟘
            # Use `y` as a workspace.
            mul!(y, H.forward, x) # out-of-place forward FFT of x in y
            vmul!(Stage(1), y, α/n, d) # in-place multiply y by mtf/n
            mul!(y, H.backward, y) # in-place backward FFT of y
        else
            # Must allocate a workspace.
            z = Array{Complex{T}}(undef, H.zdims) # allocate temporary
            mul!(z, H.forward, x) # out-of-place forward FFT of x in z
            vmul!(Stage(1), z, α/n, d) # in-place multiply z by mtf/n
            mul!(z, H.backward, z) # in-place backward FFT of z
            vcombine!(y, 𝟙, z, β, y)
        end
    else
        z = Array{Complex{T}}(undef, H.zdims) # allocate temporary
        mul!(z, H.forward, x) # out-of-place forward FFT of x in z
        vmul!(Stage(1), z, α/n, d) # in-place multiply z by mtf/n
        if β == 𝟘
            mul!(y, H.backward, z) # out-of-place backward FFT of z in y
        else
            w = Array{T}(undef, H.dims) # allocate another temporary
            mul!(w, H.backward, z) # out-of-place backward FFT of z in y
            vcombine!(y, 1, w, β, y)
        end
    end
    return y
end

#----------------------------------------------------------------------------- UTILITIES -

const FFTW_PLANNING = (FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT | FFTW.EXHAUSTIVE |
    FFTW.WISDOM_ONLY)

"""

`check_fftw_flags(flags)` checks whether `flags` is an allowed bitwise-or combination of
FFTW planner flags (see http://www.fftw.org/doc/Planner-Flags.html) and returns the
filtered flags.

"""
function check_fftw_flags(flags::Integer)
    planning = flags & FFTW_PLANNING
    flags == planning || throw_bad_argument("only FFTW planning flags can be specified")
    return UInt32(planning)
end

function print_fftw_flags(io::IO, flags::Integer)
    n = 0
    for (bit, key) in (FFTW.ESTIMATE       => :ESTIMATE,
                       FFTW.MEASURE        => :MEASURE,
                       FFTW.PATIENT        => :PATIENT,
                       FFTW.EXHAUSTIVE     => :EXHAUSTIVE,
                       FFTW.WISDOM_ONLY    => :WISDOM_ONLY)
        if iszero(bit) ? iszero(flags & FFTW_PLANNING) : !iszero(bit & flags)
            n > 0 && print(io, " | ")
            print(io, "FFTW.", key)
            n += 1
        end
    end
    bits = FFTW.DESTROY_INPUT | FFTW.PRESERVE_INPUT
    for (bit, key) in (FFTW.DESTROY_INPUT  => :DESTROY_INPUT,
                       FFTW.PRESERVE_INPUT => :PRESERVE_INPUT)
        if (flags & bits) == bit
            print(io, " #= FFTW.", key, " =#")
        end
    end
    nothing
end

"""
```julia
goodfftdim(len)
```

yields the smallest integer which is greater or equal `len` and which is a
multiple of powers of 2, 3 and/or 5.  If argument is an array dimesion list
(i.e. a tuple of integers), a tuple of good FFT dimensions is returned.

Also see: [`goodfftdims`](@ref), [`rfftdims`](@ref), [`FFT`](@ref).

"""
goodfftdim(len::Integer) = goodfftdim(Int(len))
goodfftdim(len::Int) = nextprod([2,3,5], len)

"""
```julia
goodfftdims(dims)
```

yields a list of dimensions suitable for computing the FFT of arrays whose
dimensions are `dims` (a tuple or a vector of integers).

Also see: [`goodfftdim`](@ref), [`rfftdims`](@ref), [`FFT`](@ref).

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

Also see: [`goodfftdim`](@ref), [`FFT`](@ref).

"""
rfftdims(dims::Integer...) = rfftdims(dims)
rfftdims(dims::NTuple{N,Integer}) where {N} =
    ntuple(d -> (d == 1 ? (Int(dims[d]) >>> 1) + 1 : Int(dims[d])), Val(N))
# Note: The above version is equivalent but much faster than
#     ((dims[1] >>> 1) + 1, dims[2:end]...)
# which is not optimized out by the compiler.

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

See also: [`FFT`](@ref), [`fftshift`](@ref).

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
