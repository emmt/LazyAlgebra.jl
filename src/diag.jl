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

Base.show(io::IO, A::Diag) = print(io, "Diag(…)")

# Accessors.
LinearAlgebra.diag(A::Diag) = A.diag
LinearAlgebra.diag(A::Adjoint{<:Diag}) = LazyMap(conj, diag(A[]))
LinearAlgebra.diag(A::Inverse{<:Diag}) = LazyMap(inv, diag(A[]))
LinearAlgebra.diag(A::InverseAdjoint{<:Diag}) = LazyMap(inv∘conj, diag(A[][]))

# API for operators.
Base.eltype(::Type{<:Union{A,Adjoint{A}}}) where {D,A<:Diag{D}} = eltype(D)
Base.eltype(::Type{<:Union{Inverse{A},InverseAdjoint{A}}}) where {D,A<:Diag{D}} =
    float(eltype(D))

OutputShape(::Type{<:Diag{<:AbstractArray{T,N}}}) where {T,N} = HasOutputShape{N}()
output_axes(A::Diag) = axes(diag(A))

InputShape(::Type{<:Diag{<:AbstractArray{T,N}}}) where {T,N} = HasInputShape{N}()
input_axes(A::Diag) = axes(diag(A))

conj_mul(w, x) = conj(w)*x
conj_ldiv(w, x) = conj(w)\x
for (T, B, f) in ((:(                 Diag ), :(              A),   :(*)),
                  (:(Adjoint{       <:Diag}), :(       parent(A)),  :conj_mul),
                  (:(Inverse{       <:Diag}), :(       parent(A)),  :(\)),
                  (:(InverseAdjoint{<:Diag}), :(parent(parent(A))), :conj_ldiv))
    @eval begin
        function unsafe_vmul!(α::Number, A::$T, x::AbstractArray,
                              β::Number, y::AbstractArray)
            # Call `vmap!` at stage 1 to dispatch on the multipliers because
            # axes of array arguments have already been checked.
            return vmap!(Stage(1), α, $f, diag($B), x, β, y)
        end
    end
end

# Precision for diagonal operators.
get_precision(::Type{T}) where {T<:Diag} = get_precision(eltype(T))
_with_precision(::Type{T}, A::Diag) where {T<:AbstractFloat} =
    Diag(_with_precision(T, diag(A)))
