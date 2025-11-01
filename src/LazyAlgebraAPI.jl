"""
    using LazyAlgebra.LazyAlgebraAPI

Make all the public symbols of `LazyAlgebra` available in the current namespace. This
includes the symbols exported by `LazyAlgebra` and made available by `using LazyAlgebra`
but also the non-exported public symbols of `LazyAlgebra`. This is meant for developers
and foreign packages who want to extend `LazyAlgebra`.

"""
module LazyAlgebraAPI

using ..LazyAlgebra

for name in names(LazyAlgebra)
    @eval begin
        import LazyAlgebra: $name
        export $name
    end
end

end # module
