# Installation

LazyAlgebra is not yet an [offical Julia package](https://pkg.julialang.org/)
but it is easy to install it from Julia as explained below.  Note that
LazyAlgebra requires the [ArrayTools](https://github.com/emmt/ArrayTools.jl)
package.


## Using the package manager


At the [REPL of
Julia](https://docs.julialang.org/en/stable/manual/interacting-with-julia/),
hit the `]` key to switch to the package manager REPL (you should get a
`... pkg>` prompt) and type:

```julia
pkg> add https://github.com/emmt/ArrayTools.jl
pkg> add https://github.com/emmt/StructuredArrays.jl
pkg> add https://github.com/emmt/ZippedArrays.jl
pkg> add https://github.com/emmt/LazyAlgebra.jl
```

where `pkg>` represents the package manager prompt and `https` protocol has
been assumed; if `ssh` is more suitable for you, then type:

```julia
pkg> add git@github.com:emmt/ArrayTools.jl
pkg> add git@github.com:emmt/LazyAlgebra.jl
```

instead.  To check whether the LazyAlgebra package works correctly, type:

```julia
pkg> test LazyAlgebra
```

Later, to update to the last version (and run tests), you can type:

```julia
pkg> update LazyAlgebra
pkg> build LazyAlgebra
pkg> test LazyAlgebra
```

If something goes wrong, it may be because you already have an old version of
LazyAlgebra.  Uninstall LazyAlgebra as follows:

```julia
pkg> rm LazyAlgebra
pkg> gc
pkg> add https://github.com/emmt/LazyAlgebra.jl
```

before re-installing.

To revert to Julia's REPL, hit the `Backspace` key at the `... pkg>` prompt.


## Installation in scripts

To install LazyAlgebra in a Julia script, write:

```julia
if VERSION >= v"0.7.0-"; using Pkg; end
Pkg.add(PackageSpec(url="https://github.com/emmt/LazyAlgebra.jl", rev="master"));
```

or with `url="git@github.com:emmt/LazyAlgebra.jl"` if you want to use `ssh`.

This also works from the Julia REPL.
