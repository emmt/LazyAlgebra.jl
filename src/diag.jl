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
for (T, f) in ((:(                 Diag ), :(*)),
               (:(Adjoint{       <:Diag}), :conj_mul),
               (:(Inverse{       <:Diag}), :(\)),
               (:(InverseAdjoint{<:Diag}), :conj_ldiv))
    @eval begin
        # FIXME function unsafe_vmul!(y::AbstractArray, α::Number, A::$T, x::AbstractArray)
        # FIXME     return dispatch_vmap!(y, α, $f, diag(unveil(A)), x)
        # FIXME end
        function unsafe_vmul!(α::Number, A::$T, x::AbstractArray,
                              β::Number, y::AbstractArray)
            # Call `dispatch_vmap!`, not `unsafe_vmap!` directly, because `β` may be zero
            # although `α` should be non-zero.
            return dispatch_vmap!(α, $f, diag(unveil(A)), x, β, y)
        end
    end
end
