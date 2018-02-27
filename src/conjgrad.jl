#
# conjgrad.jl -
#
# Linear conjugate-gradient.
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
conjgrad!(x, A, b, x0 [, p, q, r]) -> x
```

solves the linear system `A⋅x = b` starting at `x0` by means of the iterative
conjugate gradient method.  The result is stored in `x` which is returned.
Note that `b` and `x` and `x0` can be the same (*e.g.*, to save memory).
Argument `A` implements a symmetric positive definite linear operator and is
used as:

```julia
apply!(q, A, p)
```

to overwrite `q` with `q = A⋅p`.  This method can be extended to be specialized
for the specific type of the linear operator `A`.

Optional arguments `p`, `q` and `r` are writable workspace *vectors*.  On
return, `p` is the last search direction, `q = A⋅p` and `r = b - A.xp` with
`xp` the previous or last solution.  If provided, these workspaces must be
distinct.  All *vectors* must have the same sizes.  If all workspace vectors
are provided, no other memory allocation is necessary.

Keyword `maxiter` can be set with the maximum number of iterations to perform.

Keyword `strict` can be set to a boolean value (default is `true`) to specify
whether non-positive definite operator `A` throws a `NonPositiveDefinite`
exception or just returns the best solution found so far (with a warning if
`quiet` is false).

Keyword `quiet` can be set to a boolean value (default is `false`) to specify
whether or not to print warning messages.

"""
function conjgrad!(x::AbstractArray{T,N},
                   A,
                   b::AbstractArray{T,N},
                   x0::AbstractArray{<:Real,N},
                   p::AbstractArray{T,N} = similar(x),
                   q::AbstractArray{T,N} = similar(x),
                   r::AbstractArray{T,N} = similar(x);
                   maxiter::Integer = min(50, length(b)),
                   quiet::Bool = false,
                   strict::Bool = true
                   ) where {T<:AbstractFloat,N}
    # Check that all arrays have the same size (for x0 this is done by vcopy!).
    @assert indices(x) == indices(b) == indices(p) == indices(q) == indices(r)

    # Initialization.
    vcopy!(x, x0)
    if maxiter < 1
        return x
    end
    xnorm2 = vdot(x, x) # FIXME: use countnz(x)?
    if xnorm2 > 0
        apply!(r, A, x)
        @inbounds @simd for i in eachindex(b, r)
            r[i] = b[i] - r[i]
        end
    else
        copy!(r, b)
    end
    rho = zero(xnorm2) # to make sure the type of rho is stable
    k = 1
    while true
        rhoprev = rho
        rho = vdot(r, r)
        if rho ≤ 0
            break
        end
        if k == 1
            copy!(p, r)
        else
            beta = convert(T, rho/rhoprev)
            @inbounds @simd for i in eachindex(p, r)
                p[i] = r[i] + beta*p[i]
            end
        end
        apply!(q, A, p)
        gamma = vdot(p, q)
        if gamma ≤ 0
            strict && throw(NonPositiveDefinite("in conjugate gradient"))
            quiet || warn("matrix is not positive definite")
            break
        end
        alpha = rho/gamma
        vupdate!(x,  alpha, p)
        if k ≥ maxiter
            break
        end
        vupdate!(r, -alpha, q)
        k += 1
    end
    return x
end
