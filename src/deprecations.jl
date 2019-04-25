# Deprecations in LazyAlgebra module.

import Base: @deprecate

@deprecate(makedims(args...), dimensions(args...))
@deprecate(densearray(args...), flatarray(args...))
@deprecate(densevector(args...), flatvector(args...))
@deprecate(densematrix(args...), flatmatrix(args...))
@deprecate(is_flat_array(args...), isflatarray(args...))
@deprecate(has_oneto_axes(args...), has_standard_indexing(args...))
@deprecate(SparseOperator(rowdims::Union{Integer,Tuple{Vararg{Integer}}},
                          coldims::Union{Integer,Tuple{Vararg{Integer}}},
                          C::AbstractVector,
                          I::AbstractVector{<:Integer},
                          J::AbstractVector{<:Integer}),
           SparseOperator(I, J, C, rowdims, coldims))
