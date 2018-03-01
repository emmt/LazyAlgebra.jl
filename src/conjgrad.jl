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
function conjgrad!(x, A, b, x0,
                   p = vcreate(x), q = vcreate(x), r = vcreate(x);
                   maxiter::Integer = min(50, length(b)),
                   quiet::Bool = false,
                   strict::Bool = true)
    # Declare typed local scalars to make sure their type is stable.
    local alpha::Float64, beta::Float64, gamma::Float64, epsilon::Float64
    local rho::Float64, rhoprev::Float64

    # Initialization.
    vcopy!(x, x0)
    if maxiter < 1
        return x
    end
    if vnorm2(x) > 0
        # Compute r = b - A⋅x.
        vcombine!(r, 1, b, -1, apply!(r, A, x))
    else
        # Save applying A since x = 0.
        vcopy!(r, b)
    end
    epsilon = 0.0
    rho = 0.0
    k = 1
    while true
        rhoprev = rho
        rho = vdot(r, r)
        if rho ≤ epsilon
            break
        end
        if k == 1
            vcopy!(p, r)
        else
            beta = rho/rhoprev
            vcombine!(p, beta, p, +1, r)
        end
        apply!(q, A, p)
        gamma = vdot(p, q)
        if gamma ≤ 0.0
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
