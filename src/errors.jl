"""
    LazyAlgebra.throw_bad_argument(args...)

Throw an `ArgumentError` exception with a textual message given by `args...` converted into
a string. This function is not in-lined.

"""
@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))
@noinline throw_bad_argument(mesg::AbstractString) = throw(ArgumentError(mesg))

@noinline throw_not_fast_array(id) =
    throw_bad_argument(id, " does not implement fast indexing")

@noinline throw_non_linear_indexing(id) =
    throw_bad_argument(id, " does not implement linear indexing")

@noinline throw_non_standard_indexing(id) =
    throw_bad_argument(id, " has non-standard indexing")

"""
    LazyAlgebra.throw_assertion_error(args...)

Throw an `AssertionError` exception with a textual message given by `args...` converted into
a string. This function is not in-lined.

"""
@noinline throw_assertion_error(args...) = throw_assertion_error(string(args...))
@noinline throw_assertion_error(mesg::AbstractString) = throw(AssertionError(mesg))

"""
    LazyAlgebra.throw_dimension_mismatch(args...)

Throw a `DimensionMismatch` exception with a textual message given by `args...` converted
into a string. This function is not in-lined.

"""
@noinline throw_dimension_mismatch(args...) = throw_dimension_mismatch(string(args...))
@noinline throw_dimension_mismatch(mesg::AbstractString) = throw(DimensionMismatch(mesg))

@noinline throw_incompatible_dimensions(id) =
    throw_dimension_mismatch(id, " has incompatible dimensions")

@noinline throw_incompatible_dimensions() =
    throw_dimension_mismatch("incompatible dimensions")

@noinline throw_incompatible_number_of_dimensions() =
    throw_dimension_mismatch("incompatible number of dimensions")

@noinline throw_incompatible_number_of_elements() =
    throw_dimension_mismatch("incompatible number of elements")

"""
    LazyAlgebra.throw_bounds_error(A, i)

Throw an `BoundsError` exception for array `A` and index `i`. This function is not in-lined.

"""
@noinline throw_bounds_error(args...) = throw(BoundsError(args...))
