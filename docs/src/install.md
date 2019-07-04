# Installation


## Installation with the package manager

LazyAlgebra.jl is not yet an [offical Julia
package](https://pkg.julialang.org/) but it is easy to install it from Julia.
At the [REPL of
Julia](https://docs.julialang.org/en/stable/manual/interacting-with-julia/),
hit the `]` key to switch to the package manager REPL (you should get a
`... pkg>` prompt) and type:

```julia
pkg> add https://github.com/emmt/LazyAlgebra.jl.git
```

where `pkg>` represents the package manager prompt and `https` protocol has
been assumed; if `ssh` is more suitable for you, then type:

```julia
pkg> add git@github.com:emmt/LazyAlgebra.jl.git
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
pkg> add https://github.com/emmt/LazyAlgebra.jl.git
```

before re-installing.

To revert to Julia's REPL, hit the `Backspace` key at the `... pkg>` prompt.


## Installation in Julia scripts

To install LazyAlgebra in a Julia script, write:

```julia
if VERSION >= v"0.7.0-"; using Pkg; end
Pkg.add(PackageSpec(url="https://github.com/emmt/LazyAlgebra.jl", rev="master"));
```

or with `url="git@github.com:emmt/LazyAlgebra.jl.git"` if you want to use `ssh`.


This also works from the Julia REPL.
