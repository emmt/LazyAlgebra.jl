#
# methods.jl -
#
# Implement non-specific methods for mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

function unimplemented(::Type{P}, ::Type{T}) where {P<:Operations, T<:Mapping}
    throw(UnimplementedOperation("unimplemented operation `$P` for mapping $T"))
end

function unimplemented(func::Union{AbstractString,Symbol},
                       ::Type{T}) where {T<:Mapping}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

"""
```julia
@callable T
```

makes concrete type `T` callable as a regular mapping that is `A(x)` yields
`apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = apply(A, x)
    end
end

for T in (Adjoint, Inverse, InverseAdjoint, Scaled, Sum, Composition, Hessian)
    @eval (A::$T)(x) = apply(A, x)
end

show(io::IO, ::MIME"text/plain", A::Mapping) = show(io, A)
show(io::IO, A::Identity) = print(io, "I")
show(io::IO, A::Scaled) =
    (multiplier(A) ≠ -one(multiplier(A)) ?
     print(io, multiplier(A), "⋅", operand(A)) :
     print(io, "-", operand(A)))
show(io::IO, A::Scaled{<:Sum}) =
    (multiplier(A) ≠ -one(multiplier(A)) ?
     print(io, multiplier(A), "⋅(", operand(A), ")") :
     print(io, "-(", operand(A), ")"))

show(io::IO, A::Adjoint{<:Mapping}) = print(io, operand(A), "'")
show(io::IO, A::Adjoint{T}) where {T<:Union{Scaled,Composition,Sum}} =
    print(io, "(", operand(A), ")'")
show(io::IO, A::Inverse{<:Mapping}) = print(io, "inv(", operand(A), ")")
show(io::IO, A::InverseAdjoint{<:Mapping}) = print(io, "inv(", operand(A), "')")

function show(io::IO, A::Sum{N}) where {N}
    for i in 1:N
        B = A[i]
        if i > 1
            if isa(B, Scaled) && multiplier(B) < 0
                B = -1*B
                print(io, " - ")
            else
                print(io, " + ")
            end
        end
        if isa(B, Sum)
            print(io, "(", B, ")")
        else
            print(io, B)
        end
    end
end

function show(io::IO, A::Composition{N}) where {N}
    for i in 1:N
        B = A[i]
        if i > 1
            print(io, "⋅")
        end
        if isa(B, Sum) || isa(B, Scaled)
            print(io, "(", B, ")")
        else
            print(io, B)
        end
    end
end

"""
```julia
operands(A)
```

yields the list (as a tuple) of operands that compose operand `A`.
If `A` is a sum or a composition, the list of operands is returned;
otherwise, the single-element tuple `(A,)` is returned.

!!! note
    The [`operand`](@ref) method (without an "s") has a different meaning.

"""
operands(A::Mapping) = (A,)
operands(A::Sum) = A.ops
operands(A::Composition) = A.ops

"""
The calls:

```julia
operand(A)
```

and

```julia
multiplier(A)
```

respectively yield the operand `M` and the multiplier `λ` if `A = λ*M` is a
scaled operand; yield `A` and `1` otherwise.

!!! note
    The [`operands`](@ref) method (with an "s") has a different meaning.

See also: [`Scaled`](@ref).

"""
operand(A::Scaled) = A.M
operand(A::Adjoint) = A.op
operand(A::Inverse) = A.op
operand(A::InverseAdjoint) = A.op

multiplier(A::Scaled) = A.λ
multiplier(A::Mapping) = 1 # FIXME: should never be used!
@doc @doc(operand) multiplier

# Extend base methods to simplify the code for reducing expressions.
first(A::Mapping) = A
last(A::Mapping) = A
first(A::Union{Sum,Composition}) = A.ops[1]
last(A::Union{Sum{N},Composition{N}}) where {N} = A.ops[N]
length(A::Union{Sum{N},Composition{N}}) where {N} = N
getindex(A::Union{Sum,Composition}, i) = A.ops[i]

# To complement first() and last() when applied to tuples.
tail(A::NTuple{N}) where {N} = A[2:N]
tail(A::Tuple) = A[2:end]
head(A::NTuple{N}) where {N} = A[1:N-1]
head(A::Tuple) = A[1:end-1]

"""
```julia
input_type([P=Direct,] A)
output_type([P=Direct,] A)
```

yield the (preferred) types of the input and output arguments of the operation
`P` with mapping `A`.  If `A` operates on Julia arrays, the element type,
list of dimensions, `i`-th dimension and number of dimensions for the input and
output are given by:

    input_eltype([P=Direct,] A)          output_eltype([P=Direct,] A)
    input_size([P=Direct,] A)            output_size([P=Direct,] A)
    input_size([P=Direct,] A, i)         output_size([P=Direct,] A, i)
    input_ndims([P=Direct,] A)           output_ndims([P=Direct,] A)

Only `input_size(A)` and `output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearMapping`](@ref),
[`Operations`](@ref).

"""
function input_type end

for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input)

    fn1 = Symbol(pfx, "_", sfx)

    for P in (Direct, Adjoint, Inverse, InverseAdjoint)

        fn2 = Symbol(P == Adjoint || P == Inverse ?
                     (pfx == :output ? :input : :output) : pfx, "_", sfx)

        T = (P == Adjoint || P == InverseAdjoint ? LinearMapping : Mapping)

        # Provide basic methods for the different operations and for tagged
        # mappings.
        @eval begin

            if $(P != Direct)
                $fn1(A::$P{<:$T}) = $fn2(A.op)
            end

            $fn1(::Type{$P}, A::$T) = $fn2(A)

            if $(sfx == :size)
                if $(P != Direct)
                    $fn1(A::$P{<:$T}, dim...) = $fn2(A.op, dim...)
                end
                $fn1(::Type{$P}, A::$T, dim...) = $fn2(A, dim...)
            end
        end
    end

    # Link documentation for the basic methods.
    @eval begin
        if $(fn1 != :input_type)
            @doc @doc(:input_type) $fn1
        end
    end

end

# Provide default methods for `$(sfx)_size(A, dim...)` and `$(sfx)_ndims(A)`.
for pfx in (:input, :output)
    pfx_size = Symbol(pfx, "_size")
    pfx_ndims = Symbol(pfx, "_ndims")
    @eval begin

        $pfx_ndims(A::Mapping) = length($pfx_size(A))

        $pfx_size(A::Mapping, dim) = $pfx_size(A)[dim]

        function $pfx_size(A::Mapping, dim...)
            dims = $pfx_size(A)
            ntuple(i -> dims[dim[i]], length(dim))
        end

    end
end

for f in (:input_eltype, :output_eltype, :input_size, :output_size)
    @eval $f(::T) where {T<:Mapping} = unimplemented($(string(f)), T)
end

"""
```julia
convert_multiplier(α, T...)
```

converts the scalar `α` in a suitable type for operations involving arguments
of types `T...`.  In general, `T...` is a single type and is the element type
of the variables to be multiplied by `α`.

!!! note
    For now, complex-valued multipliers are not supported.  The type of the
    multiplier `α` must be integer or floating-point.  If `α` and the real part
    of all types `T...` are integers, the returned value is an integer;
    otherwise, the returned value is a floating-point.

See also: [`convert`](@ref) and [`promote_type`](@ref).

"""
convert_multiplier(α::Real, T::Type{<:Number}, args::Type{<:Number}...) =
    convert_multiplier(α, promote_type(T, args...))

# Sub-types of Number are: Complex and Real.
convert_multiplier(α::Real, ::Type{Complex{T}}) where {T<:Real} =
    convert_multiplier(α, T)

# Sub-types of Real are: AbstractFloat, AbstractIrrational, Integer and
# Rational.
convert_multiplier(α::Integer, T::Type{<:Integer}) = convert(T, α)
convert_multiplier(α::Real, T::Type{<:AbstractFloat}) = convert(T, α)
convert_multiplier(α::Real, T::Type{<:Real}) = convert(float(T), α)

"""
```julia
checkmapping(y, A, x) -> (v1, v2, v1 - v2)
```

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a
linear mapping, `y` a "vector" of the output space of `A` and `x` a "vector"
of the input space of `A`.  In principle, the two inner products should be the
same whatever `x` and `y`; otherwise the mapping has a bug.

Simple linear mappings operating on Julia arrays can be tested on random
"vectors" with:

```julia
checkmapping([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)
```

with `outdims` and `outdims` the dimensions of the output and input "vectors"
for `A`.  Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`,
`output_eltype` and `output_size` have been specialized for `A`, then:

```julia
checkmapping(A) -> (v1, v2, v1 - v2)
```

is sufficient to check `A` against automatically generated random arrays.

See also: [`vdot`](@ref), [`vcreate`](@ref), [`apply!`](@ref),
          [`input_type`](@ref).

"""
function checkmapping(y::Ty, A::Mapping, x::Tx) where {Tx, Ty}
    is_linear(A) ||
        throw(ArgumentError("expecting a linear map"))
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function checkmapping(::Type{T},
                      outdims::Tuple{Vararg{Int}},
                      A::Mapping,
                      inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    checkmapping(randn(T, outdims), A, randn(T, inpdims))
end

function checkmapping(outdims::Tuple{Vararg{Int}},
                      A::Mapping,
                      inpdims::Tuple{Vararg{Int}})
    checkmapping(Float64, outdims, A, inpdims)
end

checkmapping(A::LinearMapping) =
    checkmapping(randn(output_eltype(A), output_size(A)), A,
                 randn(input_eltype(A), input_size(A)))

"""
```julia
is_same_mutable_object(a, b)
```

yields whether `a` and `b` are references to the same object.  This function
can be used to check whether [`vcreate`](@ref) returns the same object as the
input variables.

This function is very fast, it takes a few nanoseonds on my laptop.

"""
is_same_mutable_object(a, b) =
    (! isimmutable(a) && ! isimmutable(b) &&
     pointer_from_objref(a) === pointer_from_objref(b))

"""
```julia
is_same_mapping(A, B)
```

yields whether `A` and `B` are the same mappings in the sense that their
effects will always be the same.  This method is used to perform some
simplifications and optimizations.

!!! note
    The returned result may be true although `A` and `B` are not necessarily
    the same objects.  For instance, if `A` and `B` are two sparse matrices
    whose coefficients and indices are stored in the same vectors (as can be
    tested with [`is_same_mutable_object`](@ref)) this method should return
    `true` because the two operators will behave identically (any changes in
    the coefficients or indices of `A` will be reflected in `B`).  If any of
    the vectors storing the coefficients or the indices are not the same
    objects, then `is_same_mapping(A,B)` must return `false` even though the
    stored values may be the same because it is possible, later, to change one
    operator without affecting identically the other.

"""
is_same_mapping(::Mapping, ::Mapping) = false  # always false by default
