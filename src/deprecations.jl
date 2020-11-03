# Deprecations in LazyAlgebra module.

@deprecate operands terms
@deprecate operand unveil
@deprecate NonuniformScalingOperator NonuniformScaling

@deprecate(SparseOperator(rowdims::Union{Integer,Tuple{Vararg{Integer}}},
                          coldims::Union{Integer,Tuple{Vararg{Integer}}},
                          C::AbstractVector,
                          I::AbstractVector{<:Integer},
                          J::AbstractVector{<:Integer}),
           SparseOperator(I, J, C, rowdims, coldims))
