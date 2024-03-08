#
# foundations.jl -
#
# Sub-module exporting types and methods needed to extend or implement
# LazyAlgebra mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2022 Éric Thiébaut.
#

"""
    using LazyAlgebra.Foundations

makes directly available types and methods that may be useful to extend or
implement `LazyAlgebra` mappings.

"""
module Foundations

using ..LazyAlgebra

for sym in [Symbol("@callable"),
            :Adjoint,
            :AdjointInverse,
            :DiagonalMapping,
            :DiagonalType,
            :Endomorphism,
            :Inverse,
            :InverseAdjoint,
            :Linear,
            :LinearType,
            :Morphism,
            :MorphismType,
            :NonDiagonalMapping,
            :NonLinear,
            :NonSelfAdjoint,
            :SelfAdjoint,
            :SelfAdjointType,
            :axpby_yields_zero,
            :axpby_yields_y,
            :axpby_yields_my,
            :axpby_yields_by,
            :axpby_yields_x,
            :axpby_yields_xpy,
            :axpby_yields_xmy,
            :axpby_yields_xpby,
            :axpby_yields_mx,
            :axpby_yields_ymx,
            :axpby_yields_mxmy,
            :axpby_yields_bymx,
            :axpby_yields_ax,
            :axpby_yields_axpy,
            :axpby_yields_axmy,
            :axpby_yields_axpby,
            :multiplier_type,
            :multiplier_floatingpoint_type,
            :promote_multiplier,
            :types_of_terms]
    @eval begin
        import ..LazyAlgebra: $sym
        export $sym
    end
end

end # module Foundations
