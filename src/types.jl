# Definition of types and constants for `LazyAlgebra`.

# Union of multiplier types that are left unchanged by `convert_multiplier`.
const StaticMultiplier{v} = Union{Neutral{v},AbstractQuantity{Neutral{v}}}

"""
    LazyAlgebra.default_precision

is the floating-point type used by defauts in `LazyAlgebra`.

"""
const default_precision = Float64

# Lightweight structure to keep track of some algorithm stage.
# Can be used to jump or branch to a given stage.
#
# For example:
# • 0 -> 1 check indices;
# • 1 -> 2 dispatch on 1st multiplier;
# • 2 -> 3 dispatch on 2nd multiplier;
# • etc. and eventually call the `unsafe_*` method.
struct Stage{N}
    # The inner constructor is to restrict the type of the parameter value.
    Stage(N::Int) = new{N}()
end

"""
    Operator

is the abstract type representing any linear function between two variables spaces in
`LazyAlgebra`. Using upper-case Latin letters to denote operators, lower-case Latin
letters to denote variables, and Greek letters to denote scalars, then:

* `A*x` or `A(x)` yields the result of applying the operator `A` to `x`;

* `A'*x` yields the result of applying the adjoint of `A` to `x`;

* `inv(A)*x` and `A\\x` yield the result of applying the inverse of `A` to `x`;

* `A'\\x`, `inv(A')*x`, and `inv(A)'*x` yield the result of applying the inverse-adjoint
  of the operator `A` to `x`.

Constructions are allowed for any kind of operators and can be used to create new
operators which behave as above. Compared to matrices in Julia, operators are not limited
to be 2-dimensional. Moreover, combining operators does not (generally) involve explicitly
computing the coefficients of the resulting operator but, instead, yields a composite
operator storing the structure and the components of the combination. This combination can
be manipulated (e.g. simplified), re-used in another combination, or applied to an
argument. For example:

* `B = A'` and `B = adjoint(A)` build an operator `B` such that `B*x` yields the same
  result as `A'*x`.

* `B = inv(A)`, `B = Id/A`, and `B = A\\Id` (where [`Id`](@ref) represents the *universal
  identity*) , build an operator `B` such that `B*x` yields the same result as `inv(A)*x`.

* `B = α*A` (where `α` is a real) builds an operator `B` which behaves as `A` times `α`;
  that is `B*x` yields the same result as `α*(A*x)`.

* `C = A + B + ...` builds an operator `C` which behaves as the sum of the operators `A`,
  `B`, ...; that is `C*x` yields the same result as `A*x + B*x + ...`.

* `C = A*B` builds an operator `C` which behaves as the composition of the operators `A`
  and `B`; that is `C*x` yields the same result as `A*(B*x)`. As for the sum of operators,
  there may be an arbitrary number of operators in a composition; for example, if `D =
  A*B*C` then `D*x` yields the same result as `A*(B*(C*x))`.

* `C = A\\B` yields an operator `C` such that `C*x` yields the same result as `A\\(B*x)`.

* `C = A/B` yields an operator `C` such that `C*x` yields the same result as `A*(B\\x)`.

These constructions can be combined to build up more complex operators. For example:

* `D = A*(B + C)` yields an operator `D` such that `D*x` yields the same result as `A*(B*x
  + C*x)`.

See also [`vmul`](@ref) and [`vmul!`](@ref).

"""
abstract type Operator end

if VERSION ≥ v"1.3.0"
    # Only since Julia 1.3, methods can be added to an abstract type.
    @callable Operator
end

"""
    B = A'
    B = adjoint(A)
    B = LazyAlgebra.Adjoint(A)

yield a linear operator `B` representing the *adjoint* (conjugate transpose) of the linear
operator `A`.

Taking the adjoint of `B` yields back `A`, that is `B' === A` holds. Calling
`Base.parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = A'`.

Also see [`LazyAlgebra.Inverse`](@ref).

"""
struct Adjoint{T<:Operator} <: Operator
    parent::T
end

@callable Adjoint

"""
    B = inv(A)
    B = A\\Id
    B = Id/A
    B = LazyAlgebra.Inverse(A)

yield a linear operator `B` representing the *inverse* of the linear operator `A`
regardless whether this inverse exists or not.

Taking the inverse of `B` yields back `A`, that is `inv(B)' === A` holds. Calling
`Base.parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = inv(A)`.

Also see [`LazyAlgebra.Adjoint`](@ref).

"""
struct Inverse{T<:Operator} <: Operator
    parent::T
end

@callable Inverse

"""
    LazyAlgebra.InverseAdjoint{A}

is an alias for the type of an operator that is the inverse adjoint (or adjoint inverse)
of an operator of type `A`.

See also [`LazyAlgebra.Adjoint`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseAdjoint{A} = Union{Inverse{Adjoint{A}},Adjoint{Inverse{A}}}

# Any of A, A', inv(A), inv(A'), or inv(A)'.
const AnyVariant{A} = Union{A,Adjoint{A},Inverse{A},InverseAdjoint{A}}

"""
    B = Gram(A)
    B = simplify(A'*A)

yield a linear operator `B` representing the composition `A'*A` for the linear operator
`A`. Applying this construction may be optimized for some kind of operators like the
finite difference [`Diff`](@ref).

Calling `Base.parent(B)` or `B[]` reveals the bare linear operator `A` embedded in `B`.

"""
struct Gram{T<:Operator} <: Operator
    parent::T
end

@callable Gram

"""
    C = A + B
    C = LazyAlgebra.Sum(A::Operator, B::Operator)

yields a linear operator `C` representing the sum of the linear operators `A` and `B`.

If `C` is an instance of `LazyAlgebra.Sum`, then `C[1]` and `C[2]` respectively yield the
left and right operands of `C`. However, due to simplifications that may occur, these are
not necessarily `A` and `B`.

"""
struct Sum{L<:Operator,R<:Operator} <: Operator
    operands::Tuple{L,R}
    Sum(left::L, right::R) where {L<:Operator,R<:Operator} = new{L,R}((left, right))
end

@callable Sum

# Type of operands in a product.
const Operand = Union{Number,Operator}

"""
    C = A*B
    C = LazyAlgebra.Prod(A::Union{Number,Operator}, B::Union{Number,Operator})

yield the result of multiplying operand `A` by operand `B`. If both operands are numbers,
the result is a number; otherwise, if at least one of `A` or `B` is a linear operator, an
instance of `Prod` is returned. If both operands are linear operators, `A∘B` and `A*B`
yield the same result.

If `C` is an instance of `LazyAlgebra.Prod`, then `C[1]` and `C[2]` respectively yield the
left and right operands of `C`. However, due to simplifications that may occur, these are
not necessarily `A` and `B`.

When composing instances of `Prod` whose operands are operators, right associativity is
applied so as to keep the operands in suitable order when applying the composite operator:

```julia
A*B*C   -> Prod(A, Prod(B, C))
(A*B)*C -> Prod(A, Prod(B, C))
A*(B*C) -> Prod(A, Prod(B, C))
```

"""
struct Prod{L<:Operand,R<:Operator} <: Operator
    # In a `Prod` object, only the left operand can be a scalar, the right operand must be
    # an operator. This is to force factorization of scalar multipliers to the left of
    # products.
    operands::Tuple{L,R}
    Prod(left::L, right::R) where {L<:Operand,R<:Operator} = new{L,R}((left, right))
end

@callable Prod

# Alias representing `λ*A`, the linear operator `A` multiplied by a scalar `λ`.
# Call [`LazyAlgebra.multiplier(B)`](@ref) and [`unscaled(B)`](@ref) with a scaled
# operator `B = λ*A` to retrieve `λ` and `A` respectively.
const Scaled{L<:Number,R} = Prod{L,R}

abstract type InputShape end
struct InputShapeUnknown <: InputShape end
struct HasInputShape{N}  <: InputShape end

abstract type OutputShape end
struct OutputShapeUnknown <: OutputShape end
struct HasOutputShape{N}  <: OutputShape end

abstract type InputEltype end
struct InputEltypeUnknown <: InputEltype end
struct HasInputEltype     <: InputEltype end

abstract type OutputEltype end
struct OutputEltypeUnknown <: OutputEltype end
struct HasOutputEltype     <: OutputEltype end

@callable struct Identity{I} <: Operator
    shape::I
    # A private constructor is needed to "filter" the input shape.
    global _Identity
    _Identity(shape::I) where {I<:Union{Colon,Dims,ArrayAxes}} = new{I}(shape)
end

"""
    Id

is the *universal identity* operator in `LazyAlgebra`; it is a *singleton* of type
[`Identity{Colon}`](@ref LazyAlgebra.Identity).

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but in the sense of a matrix). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `I/A` and `A\\I` all yield `inv(A)` for any `LazyAlgebra` operator `A`.

"""
const Id = _Identity(:)
const UniversalIdentity = Identity{Colon}
const ShapedIdentity{N} = Identity{<:Union{Dims{N},ArrayAxes{N}}}

@callable struct Diag{D<:AbstractArray} <: Operator
    diag::D
end

struct LazyMap{T,N,L,F,A<:AbstractArray{<:Any,N}} <: AbstractArray{T,N}
    func::F
    arr::A
    LazyMap{T}(func::F, arr::A) where {T,N,F<:Function,A<:AbstractArray{<:Any,N}} =
        new{T,N,IndexStyle(A)==IndexLinear(),F,A}(func, arr)
end

abstract type AbstractRankOneOperator{U<:AbstractArray,V<:AbstractArray} <: Operator end

@callable struct RankOneOperator{U,V} <: AbstractRankOneOperator{U,V}
    u::U
    v::V
end

@callable struct SymmetricRankOneOperator{U} <: AbstractRankOneOperator{U,U}
    u::U
end

@callable struct PseudoMatrix{T, # element type
                              M, # number of row dimensions or Colon
                              N, # total number of dimensions (rows + columns)
                              A<:AbstractArray{T,N}} <: Operator
    parent::A

    PseudoMatrix{T,Colon}(arr::A) where {T,N,A<:AbstractArray{T,N}} =
        new{T,Colon,N,A}(arr)

    function PseudoMatrix{T,M}(arr::A) where {T,M,N,A<:AbstractArray{T,N}}
        M isa Int || throw(ArgumentError("number of row dimensions must be an `Int`"))
        0 ≤ M ≤ N || throw(ArgumentError("out of range number of row dimensions"))
        return new{T,M,N,A}(arr)
    end
end

const FlexibleMatrix{T,N,A} = PseudoMatrix{T,Colon,N,A}

@callable struct CroppingOperator{N,I<:ArrayAxes{N},J<:ArrayAxes{N}} <: Operator
    i::I # output (cropped) axes
    j::J # input axes
    k::CartesianIndex{N} # offset of cropped region w.r.t. input array
    # Inner constructor to check arguments.
    function CroppingOperator(i::I, j::J, k::CartesianIndex{N}) where {N,
                                                                       I<:ArrayAxes{N},
                                                                       J<:ArrayAxes{N}}
        @inbounds for d in 1:N
            check_cropping_axis(i[d], j[d], k[d])
        end
        return new{N,I,J}(i, j, k)
    end
end

# A zero-padding operator is implemented as the adjoint of a cropping operator.
const ZeroPaddingOperator{N,I,J} = Adjoint{CroppingOperator{N,J,I}}

# Finite difference operator with `L` the order of differentiation and `D` the list of
# dimensions along which to compute the differences ( `Colon` for all, a tuple of `Int`s
# or a single `Int`).
@callable struct Diff{L,D} <: Operator
    # Inner constructor to avoid building with unchecked parameters.
    function Diff{L,D}() where {L,D}
        L isa Int || throw(ArgumentError("finite difference order `L` must be an `Int`"))
        D === Colon || D isa Int || D isa Tuple{Vararg{Int}} || D === :any || throw(
            ArgumentError("invalid dimension(s) of differentiation"))
        return new{L,D}()
    end
end

"""
    SparseOperator{T,M,N}

is the abstract type inherited by sparse operator types. Parameter `T` is the type of the
structural non-zeros. Parameters `M` and `N` are the number of dimensions of the *rows*
and of the *columns* respectively. Sparse operators are a generalization of sparse
matrices in the sense that they implement linear operators which can be applied to
`N`-dimensional arguments to produce `M`-dimensional results (as explained below). See
[`PseudoMatrix`](@ref) for a similar generalization but for *dense* matrices.

See [`CompressedSparseOperator`](@ref) for usage of sparse operators implementing
compressed storage formats.

"""
abstract type SparseOperator{T,M,N} <: Operator end

abstract type CompressedSparseOperator{F,T,M,N} <: SparseOperator{T,M,N} end

@callable struct SparseOperatorCSR{T,M,N,
                                   V<:AbstractVector{T},
                                   J<:AbstractVector{Int},
                                   K<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:CSR,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    cols::J         # linear column indices of entries
    offs::K         # row offsets in arrays of entries and column indices
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not check
    # whether arguments are correct.
    global _SparseOperatorCSR
    function _SparseOperatorCSR(m::Integer, n::Integer,
                                vals::V, cols::J, offs::K,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        J<:AbstractVector{Int},
                                                        K<:AbstractVector{Int}}
        new{T,M,N,V,J,K}(m, n, vals, cols, offs, rowsiz, colsiz)
    end
end

@callable struct SparseOperatorCSC{T,M,N,
                                   V<:AbstractVector{T},
                                   I<:AbstractVector{Int},
                                   K<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:CSC,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    rows::I         # linear row indices of entries
    offs::K         # columns offsets in arrays of entries and row indices
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not check
    # whether arguments are correct.
    global _SparseOperatorCSC
    function _SparseOperatorCSC(m::Int, n::Int,
                                vals::V, rows::I, offs::K,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        I<:AbstractVector{Int},
                                                        K<:AbstractVector{Int}}
        new{T,M,N,V,I,K}(m, n, vals, rows, offs, rowsiz, colsiz)
    end
end

@callable struct SparseOperatorCOO{T,M,N,
                                   V<:AbstractVector{T},
                                   I<:AbstractVector{Int},
                                   J<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:COO,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    rows::I         # linear row indices of entries
    cols::J         # linear column indices of entries
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple outer
    # constructor, it is not meant to be called directly as it does not check whether
    # arguments are correct.
    global _SparseOperatorCOO
    function _SparseOperatorCOO(m::Integer, n::Integer,
                                vals::V, rows::I, cols::J,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        I<:AbstractVector{Int},
                                                        J<:AbstractVector{Int}}
        new{T,M,N,V,I,J}(m, n, vals, rows, cols, rowsiz, colsiz)
    end
end
