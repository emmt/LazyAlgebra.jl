#
# methods.jl -
#
# Implement non-specific methods for mappings.
#

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
    @eval $f(::T) where {T<:Mapping} = unimplemented_method($(string(f)), T)
end

unimplemented_method(func::Union{AbstractString,Symbol}, ::Type{T}) where {T} =
    error("method `$func` not implemented by this mapping ($T)")

"""
```julia
is_applicable_in_place([P,] A, x)
```

yields whether mapping `A` is applicable *in-place* for performing operation
`P` with argument `x`, that is with the result stored into the argument `x`.
This can be used to spare allocating ressources.

See also: [`LinearMapping`](@ref), [`apply!`](@ref).

"""
is_applicable_in_place(::Type{<:Operations}, A::Mapping, x) = false
is_applicable_in_place(A::Mapping, x) =
    is_applicable_in_place(Direct, A, x)

"""

`promote_scalar(T1, [T2, ...] α)` yields scalar `α` converted to
`promote_type(T1, T2, ...)`.

"""
promote_scalar(::Type{T1}, alpha::Real) where {T1<:AbstractFloat} =
    convert(T1, alpha)

function promote_scalar(::Type{T1}, ::Type{T2},
                        alpha::Real) where {T1<:AbstractFloat,
                                            T2<:AbstractFloat}
    return convert(promote_type(T1, T2), alpha)
end

function promote_scalar(::Type{T1}, ::Type{T2}, ::Type{T3},
                        alpha::Real) where {T1<:AbstractFloat,
                                            T2<:AbstractFloat,
                                            T3<:AbstractFloat}
    return convert(promote_type(T1, T2, T3), alpha)
end

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
function checkmapping(y::Ty, A::LinearMapping, x::Tx) where {Tx, Ty}
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function checkmapping(::Type{T},
                      outdims::Tuple{Vararg{Int}},
                      A::LinearMapping,
                      inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    checkmapping(randn(T, outdims), A, randn(T, inpdims))
end

function checkmapping(outdims::Tuple{Vararg{Int}},
                      A::LinearMapping,
                      inpdims::Tuple{Vararg{Int}})
    checkmapping(Float64, outdims, A, inpdims)
end

checkmapping(A::LinearMapping) =
    checkmapping(randn(output_eltype(A), output_size(A)), A,
                 randn(input_eltype(A), input_size(A)))


