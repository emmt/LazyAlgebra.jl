#
# diag.jl -
#
# Implement non-uniform scaling, a.k.a. diagonal operator, in LazyAlgebra.
#
#-----------------------------------------------------------------------------------------

"""
    A = Diag(w)

yields a non-uniform scaling linear mapping whose effect is to apply elementwise
multiplication of its argument by the scaling factors `w`. This operator can be thought as
a generalized *diagonal* operator.

The `LinearAlgebra.diag` method (exported by `using LazyAlgebra`) can be called to
retrieve the scaling factors:

    using LinearAlgebra
    W = Diag(A)
    diag(W) === A  # this is true

!!! note
    Beware of the differences between the [`Diag`](@ref) (with an uppercase 'D') and
    `diag` (with an lowercase 'd') methods.

""" Diag

# MIME"text/plain" is for the REPL.
Base.show(io::IO, ::MIME"text/plain", A::Diag) = print(io, "Diag(…)")
function Base.show(io::IO, A::Diag)
    print(io, "Diag(")
    show(io, typeof(diag(A)))
    print(io, "(…))")
end

# Accessors.
LinearAlgebra.diag(A::Diag) = A.diag
LinearAlgebra.diag(A::Adjoint{<:Diag}) = lazymap(conj, diag(A[]))
LinearAlgebra.diag(A::Inverse{<:Diag}) = lazymap(inv, diag(A[]))
LinearAlgebra.diag(A::InverseAdjoint{<:Diag}) = lazymap(inv∘conj, diag(A[][]))

# Conversion constructors. The rationale is that `Diag(A) -> A` if `A` behaves as a
# diagonal operator and implements `diag(A)`.
Diag(A::DiagonalOperator) = A

# Constructors for identity and uniform scaling.
Diag(A::Identity) = A
Diag(A::Prod{<:Number,<:Identity}) = A
Diag(A::UniformScaling) = Operator(A)

# API for operators.
Base.eltype(::Type{<:Diag{D}}) where {D} = eltype(D)

OutputShape(::Type{<:Diag{<:AbstractArray{T,N}}}) where {T,N} = HasOutputShape{N}()
output_axes(A::Diag) = axes(diag(A))

InputShape(::Type{<:Diag{<:AbstractArray{T,N}}}) where {T,N} = HasInputShape{N}()
input_axes(A::Diag) = axes(diag(A))

# Testing for equality.
for cmp in (:(==), :isequal)
    @eval begin
        Base.$cmp(A::Diag, B::Diag) = A === B || $cmp(diag(A), diag(B))
    end
end

conj_mul(w, x) = conj(w)*x
conj_ldiv(w, x) = conj(w)\x
for (T, B, f) in ((:(                 Diag ), :(              A),   :(*)),
                  (:(Adjoint{       <:Diag}), :(       parent(A)),  :conj_mul),
                  (:(Inverse{       <:Diag}), :(       parent(A)),  :(\)),
                  (:(InverseAdjoint{<:Diag}), :(parent(parent(A))), :conj_ldiv))
    @eval begin
        function unsafe_vmul!(α::Number, A::$T, x::AbstractArray,
                              β::Number, y::AbstractArray)
            # Axes have been checked, `α` and `β` have been converted, and `α` is not
            # zero, so we can directly call `unsafe_vmap!`.
            return unsafe_vmap!(α, $f, diag($B), x, β, y)
        end
    end
end

# Precision for diagonal operators.
TypeUtils.get_precision(::Type{A}) where {A<:Diag} = get_precision(eltype(A))
TypeUtils.adapt_precision(::Type{T}, A::Diag) where {T<:TypeUtils.Precision} =
    Diag(adapt_precision(T, diag(A)))
