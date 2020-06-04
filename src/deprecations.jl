# Deprecations in LazyAlgebra module.

@deprecate(contents(args...), coefficients(args...))
@deprecate(convert_multiplier(λ::Number, T::Type),
           promote_multiplier(λ, T))
@deprecate(convert_multiplier(λ::Number, T::Type, ::Type),
           promote_multiplier(λ, T))
@deprecate(makedims(args::Integer...), to_size(args))
@deprecate(makedims(args::Tuple{Vararg{Integer}}), to_size(args))
@deprecate(densearray(args...), to_flat_array(args...))
@deprecate(densevector(args...), to_flat_array(args...))
@deprecate(densematrix(args...), to_flat_array(args...))
@deprecate(has_oneto_axes(args...), has_standard_indexing(args...))
@deprecate(SparseOperator(rowdims::Union{Integer,Tuple{Vararg{Integer}}},
                          coldims::Union{Integer,Tuple{Vararg{Integer}}},
                          C::AbstractVector,
                          I::AbstractVector{<:Integer},
                          J::AbstractVector{<:Integer}),
           SparseOperator(I, J, C, rowdims, coldims))
