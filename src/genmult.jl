#
# genmult.jl -
#
# Generalized dot product by grouping consecutive dimensions.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

module GenMult

export
    lgemm!,
    lgemm,
    lgemv!,
    lgemv

using Compat
using ..LazyAlgebra # for `allindices`, `is_flat_array`, etc.
using ..LazyAlgebra: Complexes, Floats, Reals, axes, convert_multiplier,
    libblas, @blasfunc, BlasInt, BlasReal, BlasFloat, BlasComplex

# Make paths to LinearAlgebra and BLAS available (see notes in LazyAlgebra.jl).
const LinearAlgebra = LazyAlgebra.LinearAlgebra
const BLAS = LazyAlgebra.BLAS

"""

```julia
Implementation(Val(:alg), args...)`
```

is used to quickly determine the most efficient implementation of the code to
use for algorithm `alg` with arguments `args...`.  The returned value is one of
four possible singletons:

- `Blas()` when highly optimized BLAS code can be used.  This is the preferred
  implementation as it is assumed to be the fastest.

- `Basic()` when *vector* and *matrix* arguments have respectively one and two
  dimensions.

- `Linear()` when *vector* and *matrix* arguments can be efficiently indexed
  by, respectively, one and two linear indices.

- `Generic()` to use generic implementation which can accomodate from any type
  of arguments and of multi-dimensional indices.  This implementation should be
  always safe to use and should provide the reference implementation of the
  algorithm `alg`.

Whenever possible, the best implementation is automatically determined at
compilation time by calling this method.

"""
abstract type Implementation end

for S in (:Basic, :Blas, :Linear, :Generic)
    @eval begin
        struct $S <: Implementation end
        @doc @doc(Implementation) $S
    end
end

incompatible_dimensions() =
    throw(DimensionMismatch("incompatible dimensions"))

invalid_transpose_character() =
    throw(ArgumentError("invalid transpose character"))

include("lgemv.jl")
include("lgemm.jl")

end # module
