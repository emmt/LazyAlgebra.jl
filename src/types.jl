# Definition of types and constants for `LazyAlgebra`.

# Union of types acceptable to define array size.
const ArraySize = Union{Integer,Tuple{Vararg{Integer}}}

# Union of multiplier types that are left unchanged by `convert_multiplier`.
const StaticMultiplier{v} = Union{Neutral{v},AbstractQuantity{Neutral{v}}}

"""
    Operator

Abstract type representing any linear function between two variables spaces in
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

* `B = α*A` (where `α` is a number) builds an operator `B` which behaves as `A` times `α`;
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

Build a linear operator `B` lazily representing the *adjoint* (conjugate transpose) of the
linear operator `A`.

Taking the adjoint of `B` yields back `A`, that is `B' === A` holds. Calling `parent(B)`
or `B[]` also reveals the linear operator `A` embedded in `B = A'`.

Also see [`LazyAlgebra.Transpose`](@ref), [`LazyAlgebra.Conjugate`](@ref) and
[`LazyAlgebra.Swapped`](@ref).

"""
struct Adjoint{T<:Operator} <: Operator
    parent::T
end

@callable Adjoint

"""
    B = transpose(A)
    B = LazyAlgebra.Transpose(A)

Build a linear operator `B` lazily representing the *transpose* of the linear operator
`A`.

Taking the transpose of `B` yields back `A`, that is `transpose(B) === A` holds. Calling
`parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = transpose(A)`.

Also see [`LazyAlgebra.Adjoint`](@ref) and [`LazyAlgebra.Swapped`](@ref).

"""
struct Transpose{T<:Operator} <: Operator
    parent::T
end

@callable Transpose

"""
    B = conj(A)
    B = LazyAlgebra.Conjugate(A)

Build a linear operator `B` lazily representing the *conjugate* of the linear operator
`A`.

Taking the conjugate of `B` yields back `A`, that is `conj(B) === A` holds. Calling
`parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = conj(A)`.

Also see [`LazyAlgebra.Adjoint`](@ref).

"""
struct Conjugate{T<:Operator} <: Operator
    parent::T
end

@callable Conjugate

"""
    LazyAlgebra.Swapped{T} = Union{LazyAlgebra.Adjoint{T},
                                   LazyAlgebra.Transpose{T}}

Union of types of linear operators similar to `T` but whose row and column indices are
swapped.

Also see [`LazyAlgebra.Unswapped`](@ref), [`LazyAlgebra.Adjoint`](@ref), and
[`LazyAlgebra.Transpose`](@ref).

"""
const Swapped{T<:Operator} = Union{Adjoint{T},Transpose{T}}

"""
    LazyAlgebra.Unswapped{T} = Union{T, LazyAlgebra.Conjugate{T}}

Union of types of linear operators similar to `T` but whose row and column indices are
*not* swapped.

Also see [`LazyAlgebra.Swapped`](@ref) and [`LazyAlgebra.Conjugate`](@ref).

"""
const Unswapped{T<:Operator} = Union{T,Conjugate{T}}

"""
    B = inv(A)
    B = A\\Id
    B = Id/A
    B = LazyAlgebra.Inverse(A)

yield a linear operator `B` representing the *inverse* of the linear operator `A`
regardless whether this inverse exists or not.

Taking the inverse of `B` yields back `A`, that is `inv(B)' === A` holds. Calling
`parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = inv(A)`.

"""
struct Inverse{T<:Operator} <: Operator
    parent::T
end

@callable Inverse

"""
    LazyAlgebra.InverseAdjoint{T} ≡ LazyAlgebra.Inverse{LazyAlgebra.Adjoint{T}}

Alias for the type of an operator that is the inverse adjoint of an operator of type `T`.

!!! note
    Construction rules imply that `inv(A)'` is always built as `inv(A')`. In other words,
    adjoint inverse is always automatically converted into an inverse adjoint.

See also [`LazyAlgebra.Adjoint`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseAdjoint{T} = Inverse{Adjoint{T}}

"""
    LazyAlgebra.InverseTranspose{T} ≡ LazyAlgebra.Inverse{LazyAlgebra.Transpose{T}}

Alias for the type of an operator that is the inverse transpose of an operator of type `T`.

!!! note
    Construction rules imply that `transpose(inv(A))` is always built as
    `inv(transpose(A))`. In other words, transpose inverse is always automatically
    converted into an inverse transpose.

See also [`LazyAlgebra.Transpose`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseTranspose{T} = Inverse{Transpose{T}}

"""
    LazyAlgebra.InverseConjugate{T} ≡ LazyAlgebra.Inverse{LazyAlgebra.Conjugate{T}}

Alias for the type of an operator that is the inverse conjugate of an operator of type `T`.

!!! note
    Construction rules imply that `conj(inv(A))` is always built as `inv(conj(A))`. In
    other words, conjugate inverse is always automatically converted into an inverse
    conjugate.

See also [`LazyAlgebra.Conjugate`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseConjugate{T} = Inverse{Conjugate{T}}

# Union of types for any of A, A', transpose(A), inv(A), inv(A'), inv(A)',
# inv(transpose(A)), or transpose(inv(A)).
const AnyVariant{T} = Union{T, Adjoint{T}, Transpose{T}, Inverse{T},
                            InverseAdjoint{T}, InverseTranspose{T}}

@callable struct Gram{T<:Operator} <: Operator
    parent::T
end

"""
    C = A + B + ...
    C = LazyAlgebra.Sum(A::Operator, B::Operator, ...)

Return a linear operator `C` representing the sum of linear operators `A`, `B`, etc.

If `C` is an instance of `LazyAlgebra.Sum`, then `C[i]` yields the `i`-th term of the sum
represented by `C`.

"""
struct Sum{T<:Tuple{Vararg{Operator}}} <: Operator
    terms::T
end

@callable Sum

# Type of a simple sum of 2 operators.
const TwoSum{A<:Operator,B<:Operator} = Sum{Tuple{A,B}}

"""
    C = A*B*...
    C = A∘B∘...
    C = LazyAlgebra.Prod(A::Operator, B::Operator, ...)

Return a linear operator `C` representing the composition of operators `A`, `B`, etc.

If `C` is an instance of `LazyAlgebra.Prod`, then `C[i]` yields the `i`-th term of the
composition represented by `C`.

"""
struct Prod{T<:Tuple{Vararg{Operator}}} <: Operator
    terms::T
end

@callable Prod

# Type of a simple composition of 2 operators.
const TwoProd{A<:Operator,B<:Operator} = Prod{Tuple{A,B}}

"""
    B = λ*A
    B = LazyAlgebra.Scaled(λ::Number, A::Operator)

Return an operator representing the result of multiplying number `λ` by operator `A`.

If `B` is an instance of `LazyAlgebra.Prod`, then `B[1]` (or
[`LazyAlgebra.multiplier(B)`](@ref)) and `B[2]` ([`unscaled(B)`](@ref)) respectively yield
the multiplier and the operator in `B`. However, due to simplifications that may occur,
these are not necessarily `λ` and `A`.

"""
struct Scaled{L<:Number,R<:Operator} <: Operator
    terms::Tuple{L,R}
    Scaled(left::L, right::R) where {L<:Number,R<:Operator} = new{L,R}((left, right))
end

@callable Scaled

# Alias representing `A` or `λ*A`, the linear operator `A` multiplied by a scalar `λ`.
const MaybeScaled{A<:Operator} = Union{A,Scaled{<:Number,A}}

#---------------------------------------------------------------------------------- Traits -

abstract type StorageOrder end
struct StorageOrderAny <: StorageOrder end
struct RowMajor <: StorageOrder end
struct ColumnMajor <: StorageOrder end

abstract type MatrixShape end
struct MatrixShapeAny <: MatrixShape end
struct LowerTriangularShape <: MatrixShape end
struct UpperTriangularShape <: MatrixShape end
const TriangularShape = Union{UpperTriangularShape,LowerTriangularShape}

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

#-------------------------------------------------------------------------------------------

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
const UniversalIdentity = typeof(Id)
const ShapedIdentity{N} = Identity{<:Union{Dims{N},ArrayAxes{N}}}

@callable struct Diag{D<:AbstractArray} <: Operator
    diag::D
end

const DiagonalOperator = Union{Diag,Adjoint{<:Diag},Inverse{<:Diag},InverseAdjoint{<:Diag}}

abstract type AbstractRankOneOperator{U<:AbstractArray,V<:AbstractArray} <: Operator end

@callable struct RankOneOperator{U,V} <: AbstractRankOneOperator{U,V}
    u::U
    v::V
end

@callable struct SymmetricRankOneOperator{U} <: AbstractRankOneOperator{U,U}
    u::U
end

@callable struct PseudoMatrix{T, # element type
                              L, # number of leading (row) dimensions or `:`
                              P<:AbstractArray{T}} <: Operator
    parent::P

    function PseudoMatrix{T,L}(A::P) where {T,L,P<:AbstractArray{T}}
        if L isa Int
            0 ≤ L ≤ ndims(P) || throw(ArgumentError(
                "out of range number of leading dimensions"))
        elseif !(L isa Colon)
            throw(ArgumentError("number of leading dimensions must be an `Int` or `:`"))
        end
        return new{T,L,P}(A)
    end
end

const FlexibleMatrix{T,P} = PseudoMatrix{T,:,P}

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

abstract type SparseFormat end
struct CompressedSparseRow        <: SparseFormat end
struct CompressedSparseColumn     <: SparseFormat end
struct CompressedSparseCoordinate <: SparseFormat end

const CSR = CompressedSparseRow
const CSC = CompressedSparseColumn
const COO = CompressedSparseCoordinate

"""
    SparseOperator{F,T,M,N}

Abstract type inherited by sparse operator types. Parameter `F` is the storage format of
the structural non-zeros of the sparse operator (see [`SparseFormat`](@ref)). Parameter
`T` is the type of the structural non-zeros. Parameters `M` and `N` are the respective
number of dimensions of the *rows* and of the *columns* of the operator. Sparse operators
are a generalization of sparse matrices in the sense that they implement linear operators
which can be applied to `N`-dimensional arguments to produce `M`-dimensional results (see
[`PseudoMatrix`](@ref) for a similar generalization but for *dense* matrices).

"""
abstract type SparseOperator{F<:SparseFormat,T,M,N} <: Operator end

"""
    LazyAlgebra.SparseOperatorLike{F,F′,T,M,N}

Union of types of linear operators that keep the same structure as a sparse operator. This
includes bare sparse factors (of type in union [`BareSparseOperator`](@doc)), their adjoint,
transpose, or conjugate, but not their inverse. Parameters are the direct format `F`, the
transposed format `F′`, the element type `T`, and the respective numbers `M` and `N` of
dimensions of the equivalent rows and columns of the operator.

The structural non-zeros of an object `A` of this type can be accessed with the methods of
the sparse operators API, like `A[k]` to get or set the `k`-th structural non-zero.

!!! note
    It is needed to explicitly specify the transposed format `F′` because there cannot be
    deferred expressions, like `transpose(F)`, in the right hand-side of a `const`
    definition.

"""
const SparseOperatorLike{F,F′,T,M,N} = Union{SparseOperator{F,T,M,N},
                                             Conjugate{<:SparseOperator{F,T,M,N}},
                                             Adjoint{  <:SparseOperator{F′,T,N,M}},
                                             Transpose{<:SparseOperator{F′,T,N,M}}}

# Unions of compressed sparse operators that can be considered as being in a given storage
# format. Whatever the format, `T` is the element type, `M` is the number of row
# dimensions, and `N` is the number of column dimensions.
const AnySparseCSR{T,M,N} = SparseOperatorLike{CSR,CSC,T,M,N}
const AnySparseCSC{T,M,N} = SparseOperatorLike{CSC,CSR,T,M,N}
const AnySparseCOO{T,M,N} = SparseOperatorLike{COO,COO,T,M,N}

@callable struct SparseOperatorCSR{T,M,N,
                                   V<:AbstractVector{T},
                                   J<:AbstractVector{Int},
                                   K<:AbstractVector{Int}
                                   } <: SparseOperator{CSR,T,M,N}
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
                                   } <: SparseOperator{CSC,T,M,N}
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
                                   } <: SparseOperator{COO,T,M,N}
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

# Union of basic concrete sparse operators.
const BareSparseOperator = Union{SparseOperatorCSR,SparseOperatorCSC,SparseOperatorCOO}

# The time needed to allocate temporary arrays is negligible compared to the time taken to
# compute a FFT (e.g., 5µs to allocate a 256×256 array of double precision complexes
# versus 1.5ms to compute its FFT). We therefore do not store any temporary arrays in the
# FFT operator. Only the FFT plans are cached in the operator.
@callable struct FFT{T<:FFTW.fftwNumber,  # input element type
                     C<:FFTW.fftwComplex, # output element type
                     N,                   # number of input or output dimensions
                     F<:FFTW.FFTWPlan{T},
                     B<:FFTW.FFTWPlan{C}} <: Operator
    forward::F     # plan for forward transform
    backward::B    # plan for backward transform
    function FFT(forward::F, backward::B) where {T<:FFTW.fftwNumber,
                                                 C<:FFTW.fftwComplex,
                                                 F<:FFTW.FFTWPlan{T},
                                                 B<:FFTW.FFTWPlan{C}}
        check_fftw_plans(forward, backward)
        N = input_ndims(F)
        return new{T,C,N,F,B}(forward, backward)
    end
end

@callable struct CirculantConvolution{T <: FFTW.fftwNumber,
                                      C <: FFTW.fftwComplex,
                                      N,
                                      F <: FFTW.FFTWPlan{T},
                                      B <: FFTW.FFTWPlan{C}} <: Operator
    mtf::Array{C,N} # modulation transfer function
    forward::F      # plan for forward transform
    backward::B     # plan for backward transform

    # Inner constructor to check the consistency of the arguments.
    function CirculantConvolution(mtf::Array{C,N},
                                  forward::F,
                                  backward::B) where {T <: FFTW.fftwNumber,
                                                      C <: FFTW.fftwComplex, N,
                                                      F <: FFTW.FFTWPlan{T},
                                                      B <: FFTW.FFTWPlan{C}}
        check_fftw_plans(forward, backward)
        size(mtf) == output_size(forward) || throw(
            DimensionMismatch("incompatible dimensions of MTF and forward FFT plan"))
        return new{T,C,N,F,B}(mtf, forward, backward)
    end

end
