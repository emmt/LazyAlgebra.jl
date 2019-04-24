# Deprecations in LazyAlgebra module.

import Base: @deprecate

@deprecate(densearray(args...),  flatarray(args...))
@deprecate(densevector(args...), flatvector(args...))
@deprecate(densematrix(args...), flatmatrix(args...))
@deprecate(is_flat_array(args...), isflatarray(args...))
@deprecate(has_oneto_axes(args...), has_standard_indexing(args...))
