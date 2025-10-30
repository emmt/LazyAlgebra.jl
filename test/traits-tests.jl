module TestingLazyAlgebraTraits

using LazyAlgebra
using Test
using LinearAlgebra
using Neutrals
using SparseArrays

using LazyAlgebra:
    Adjoint,
    ColumnMajor,
    Inverse,
    LowerTriangularShape,
    MatrixShape,
    MatrixShapeAny,
    Prod,
    RowMajor,
    Scaled,
    StorageOrder,
    StorageOrderAny,
    Sum,
    Transpose,
    TriangularShape,
    UpperTriangularShape,
    divide,
    inverse,
    is_column_major,
    is_complex,
    is_lower_triangular,
    is_row_major,
    is_upper_triangular

struct SingleTraitOperator{T} <: Operator end
LazyAlgebra.MatrixShape(::Type{SingleTraitOperator{T}}) where {T} = MatrixShape(T)
LazyAlgebra.StorageOrder(::Type{SingleTraitOperator{T}}) where {T} = StorageOrder(T)

@testset "Traits" begin
    @testset "Numbers" begin
        @test @inferred(is_complex(1)) == false
        @test @inferred(is_complex(1 - 0im)) == true
        @test @inferred(is_complex("hello")) == false
        @test @inferred(is_complex(Int)) == false
        @test @inferred(is_complex(AbstractString)) == false
        @test @inferred(is_complex(Complex)) == true
        @test @inferred(is_complex(Complex{Float32})) == true
    end

    @testset "Equivalent Shape" begin
        # Shape of equivalent shape instances.
        @test @inferred(MatrixShape(MatrixShapeAny())) === MatrixShapeAny()
        @test @inferred(is_lower_triangular(MatrixShapeAny())) === false
        @test @inferred(is_upper_triangular(MatrixShapeAny())) === false

        @test @inferred(MatrixShape(LowerTriangularShape())) === LowerTriangularShape()
        @test @inferred(is_lower_triangular(LowerTriangularShape())) === true
        @test @inferred(is_upper_triangular(LowerTriangularShape())) === false

        @test @inferred(MatrixShape(UpperTriangularShape())) === UpperTriangularShape()
        @test @inferred(is_lower_triangular(UpperTriangularShape())) === false
        @test @inferred(is_upper_triangular(UpperTriangularShape())) === true

        # Shape of equivalent shape types.
        @test @inferred(MatrixShape(MatrixShapeAny)) === MatrixShapeAny()
        @test @inferred(is_lower_triangular(MatrixShapeAny)) === false
        @test @inferred(is_upper_triangular(MatrixShapeAny)) === false

        @test @inferred(MatrixShape(LowerTriangularShape)) === LowerTriangularShape()
        @test @inferred(is_lower_triangular(LowerTriangularShape)) === true
        @test @inferred(is_upper_triangular(LowerTriangularShape)) === false

        @test @inferred(MatrixShape(UpperTriangularShape)) === UpperTriangularShape()
        @test @inferred(is_lower_triangular(UpperTriangularShape)) === false
        @test @inferred(is_upper_triangular(UpperTriangularShape)) === true

        # Transposition of equivalent shape.
        @test @inferred(transpose(MatrixShapeAny())) === MatrixShapeAny()
        @test @inferred(transpose(LowerTriangularShape())) === UpperTriangularShape()
        @test @inferred(transpose(UpperTriangularShape())) === LowerTriangularShape()

        # Inverse of equivalent shape.
        @test @inferred(inv(MatrixShapeAny())) === MatrixShapeAny()
        @test @inferred(inv(LowerTriangularShape())) === LowerTriangularShape()
        @test @inferred(inv(UpperTriangularShape())) === UpperTriangularShape()

        # Unknown equivalent shape.
        for A in (SymbolicOperator(:A), SingleTraitOperator{MatrixShapeAny}())
            @test @inferred(MatrixShape(A)) === MatrixShapeAny()
            @test @inferred(MatrixShape(A')) === MatrixShapeAny()
            @test @inferred(MatrixShape(transpose(A))) === MatrixShapeAny()
            @test @inferred(MatrixShape(conj(A))) === MatrixShapeAny()
            @test @inferred(MatrixShape(2*A)) === MatrixShapeAny()

            @test @inferred(is_lower_triangular(A)) === false
            @test @inferred(is_lower_triangular(A')) === false
            @test @inferred(is_lower_triangular(transpose(A))) === false
            @test @inferred(is_lower_triangular(conj(A))) === false
            @test @inferred(is_lower_triangular(2*A)) === false

            @test @inferred(is_upper_triangular(A)) === false
            @test @inferred(is_upper_triangular(A')) === false
            @test @inferred(is_upper_triangular(transpose(A))) === false
            @test @inferred(is_upper_triangular(conj(A))) === false
            @test @inferred(is_upper_triangular(2*A)) === false
        end

        # Equivalent shape for lower triangular operator.
        for A in (SingleTraitOperator{LowerTriangularShape}(),)
            @test @inferred(MatrixShape(A)) === LowerTriangularShape()
            @test @inferred(MatrixShape(A')) === UpperTriangularShape()
            @test @inferred(MatrixShape(transpose(A))) === UpperTriangularShape()
            @test @inferred(MatrixShape(conj(A))) === LowerTriangularShape()
            @test @inferred(MatrixShape(3*A)) === LowerTriangularShape()

            @test @inferred(is_lower_triangular(A)) === true
            @test @inferred(is_lower_triangular(A')) === false
            @test @inferred(is_lower_triangular(transpose(A))) === false
            @test @inferred(is_lower_triangular(conj(A))) === true
            @test @inferred(is_lower_triangular(3*A)) === true

            @test @inferred(is_upper_triangular(A)) === false
            @test @inferred(is_upper_triangular(A')) === true
            @test @inferred(is_upper_triangular(transpose(A))) === true
            @test @inferred(is_upper_triangular(conj(A))) === false
            @test @inferred(is_upper_triangular(3*A)) === false
        end

        # Equivalent shape for upper triangular operators.
        for A in (Diff(), SingleTraitOperator{UpperTriangularShape}())
            @test @inferred(MatrixShape(A)) === UpperTriangularShape()
            @test @inferred(MatrixShape(A')) === LowerTriangularShape()
            @test @inferred(MatrixShape(transpose(A))) === LowerTriangularShape()
            @test @inferred(MatrixShape(conj(A))) === UpperTriangularShape()
            @test @inferred(MatrixShape(3*A)) === UpperTriangularShape()

            @test @inferred(is_lower_triangular(A)) === false
            @test @inferred(is_lower_triangular(A')) === true
            @test @inferred(is_lower_triangular(transpose(A))) === true
            @test @inferred(is_lower_triangular(conj(A))) === false
            @test @inferred(is_lower_triangular(3*A)) === false

            @test @inferred(is_upper_triangular(A)) === true
            @test @inferred(is_upper_triangular(A')) === false
            @test @inferred(is_upper_triangular(transpose(A))) === false
            @test @inferred(is_upper_triangular(conj(A))) === true
            @test @inferred(is_upper_triangular(3*A)) === true
        end

        # Equivalent shape of Julia arrays.
        let Adjoint = LinearAlgebra.Adjoint, Transpose = LinearAlgebra.Transpose
            A = reshape(1:9, 3, 3)
            L = LinearAlgebra.LowerTriangular(A)
            U = LinearAlgebra.UpperTriangular(A)

            @test @inferred(MatrixShape(A)) === MatrixShapeAny()
            @test @inferred(MatrixShape(adjoint(A))) === MatrixShapeAny()
            @test @inferred(MatrixShape(transpose(A))) === MatrixShapeAny()
            @test @inferred(is_lower_triangular(A)) === false
            @test @inferred(is_lower_triangular(adjoint(A))) === false
            @test @inferred(is_lower_triangular(transpose(A))) === false
            @test @inferred(is_upper_triangular(A)) === false
            @test @inferred(is_upper_triangular(adjoint(A))) === false
            @test @inferred(is_upper_triangular(transpose(A))) === false

            @test @inferred(MatrixShape(L)) === LowerTriangularShape()
            @test @inferred(MatrixShape(adjoint(L))) === UpperTriangularShape()
            @test @inferred(MatrixShape(transpose(L))) === UpperTriangularShape()
            @test @inferred(MatrixShape(inv(L))) === LowerTriangularShape()
            @test @inferred(is_lower_triangular(L)) === true
            @test @inferred(is_lower_triangular(adjoint(L))) === false
            @test @inferred(is_lower_triangular(transpose(L))) === false
            @test @inferred(is_upper_triangular(L)) === false
            @test @inferred(is_upper_triangular(adjoint(L))) === true
            @test @inferred(is_upper_triangular(transpose(L))) === true

            @test @inferred(MatrixShape(U)) === UpperTriangularShape()
            @test @inferred(MatrixShape(adjoint(U))) === LowerTriangularShape()
            @test @inferred(MatrixShape(transpose(U))) === LowerTriangularShape()
            @test @inferred(MatrixShape(inv(U))) === UpperTriangularShape()
            @test @inferred(is_lower_triangular(U)) === false
            @test @inferred(is_lower_triangular(adjoint(U))) === true
            @test @inferred(is_lower_triangular(transpose(U))) === true
            @test @inferred(is_upper_triangular(U)) === true
            @test @inferred(is_upper_triangular(adjoint(U))) === false
            @test @inferred(is_upper_triangular(transpose(U))) === false
        end

    end

    @testset "Storage Order" begin
        # Storage order of storage order instances.
        @test @inferred(StorageOrder(StorageOrderAny())) === StorageOrderAny()
        @test @inferred(is_row_major(StorageOrderAny())) === false
        @test @inferred(is_column_major(StorageOrderAny())) === false

        @test @inferred(StorageOrder(ColumnMajor())) === ColumnMajor()
        @test @inferred(is_row_major(ColumnMajor())) === false
        @test @inferred(is_column_major(ColumnMajor())) === true

        @test @inferred(StorageOrder(RowMajor())) === RowMajor()
        @test @inferred(is_row_major(RowMajor())) === true
        @test @inferred(is_column_major(RowMajor())) === false

        # Storage order of storage order types.
        @test @inferred(StorageOrder(StorageOrder)) === StorageOrderAny()
        @test @inferred(is_row_major(StorageOrder)) === false
        @test @inferred(is_column_major(StorageOrder)) === false

        @test @inferred(StorageOrder(StorageOrderAny)) === StorageOrderAny()
        @test @inferred(is_row_major(StorageOrderAny)) === false
        @test @inferred(is_column_major(StorageOrderAny)) === false

        @test @inferred(StorageOrder(ColumnMajor)) === ColumnMajor()
        @test @inferred(is_row_major(ColumnMajor)) === false
        @test @inferred(is_column_major(ColumnMajor)) === true

        @test @inferred(StorageOrder(RowMajor)) === RowMajor()
        @test @inferred(is_row_major(RowMajor)) === true
        @test @inferred(is_column_major(RowMajor)) === false

        # Transposition of storage order.
        @test @inferred(transpose(StorageOrderAny())) === StorageOrderAny()
        @test @inferred(transpose(RowMajor())) === ColumnMajor()
        @test @inferred(transpose(ColumnMajor())) === RowMajor()

        # Unknown storage order.
        A = SymbolicOperator(:A)

        @test @inferred(StorageOrder(A)) === StorageOrderAny()
        @test @inferred(StorageOrder(A')) === StorageOrderAny()
        @test @inferred(StorageOrder(transpose(A))) === StorageOrderAny()

        @test @inferred(StorageOrder(typeof(A))) === StorageOrderAny()
        @test @inferred(StorageOrder(typeof(A'))) === StorageOrderAny()
        @test @inferred(StorageOrder(typeof(transpose(A)))) === StorageOrderAny()

        @test @inferred(is_row_major(A)) === false
        @test @inferred(is_row_major(A')) === false
        @test @inferred(is_row_major(transpose(A))) === false

        @test @inferred(is_column_major(A)) === false
        @test @inferred(is_column_major(A')) === false
        @test @inferred(is_column_major(transpose(A))) === false

        # Storage order of sparse operator in COO format.
        A = SparseOperatorCOO([-1.0, 2.0, 0.0, 4.0, 7.0], [1, 1, 2, 3, 3], [1, 2, 1, 2, 4], (3,), (4,))

        @test @inferred(StorageOrder(A)) === StorageOrderAny()
        @test @inferred(StorageOrder(A')) === StorageOrderAny()
        @test @inferred(StorageOrder(transpose(A))) === StorageOrderAny()

        @test @inferred(StorageOrder(typeof(A))) === StorageOrderAny()
        @test @inferred(StorageOrder(typeof(A'))) === StorageOrderAny()
        @test @inferred(StorageOrder(typeof(transpose(A)))) === StorageOrderAny()

        @test @inferred(is_row_major(A)) === false
        @test @inferred(is_row_major(A')) === false
        @test @inferred(is_row_major(transpose(A))) === false

        @test @inferred(is_column_major(A)) === false
        @test @inferred(is_column_major(A')) === false
        @test @inferred(is_column_major(transpose(A))) === false

        # Storage order of sparse operator in CSC format.
        C = SparseOperatorCSC(A)

        @test @inferred(StorageOrder(C)) === ColumnMajor()
        @test @inferred(StorageOrder(C')) === RowMajor()
        @test @inferred(StorageOrder(transpose(C))) === RowMajor()

        @test @inferred(StorageOrder(typeof(C))) === ColumnMajor()
        @test @inferred(StorageOrder(typeof(C'))) === RowMajor()
        @test @inferred(StorageOrder(typeof(transpose(C)))) === RowMajor()

        @test @inferred(is_row_major(C)) === false
        @test @inferred(is_row_major(C')) === true
        @test @inferred(is_row_major(transpose(C))) === true

        @test @inferred(is_column_major(C)) === true
        @test @inferred(is_column_major(C')) === false
        @test @inferred(is_column_major(transpose(C))) === false

        # Storage order of sparse operator in CSR format.
        R = SparseOperatorCSR(A)

        @test @inferred(StorageOrder(R)) === RowMajor()
        @test @inferred(StorageOrder(R')) === ColumnMajor()
        @test @inferred(StorageOrder(transpose(R))) === ColumnMajor()

        @test @inferred(StorageOrder(typeof(R))) === RowMajor()
        @test @inferred(StorageOrder(typeof(R'))) === ColumnMajor()
        @test @inferred(StorageOrder(typeof(transpose(R)))) === ColumnMajor()

        @test @inferred(is_row_major(R)) === true
        @test @inferred(is_row_major(R')) === false
        @test @inferred(is_row_major(transpose(R))) === false

        @test @inferred(is_column_major(R)) === false
        @test @inferred(is_column_major(R')) === true
        @test @inferred(is_column_major(transpose(R))) === true

        let Adjoint = LinearAlgebra.Adjoint, Transpose = LinearAlgebra.Transpose
            # Storage order of Julia regular arrays.
            @test @inferred(StorageOrder(Array)) === ColumnMajor()
            @test @inferred(StorageOrder(Adjoint{<:Any,Array})) === RowMajor()
            @test @inferred(StorageOrder(Transpose{<:Any,Array})) === RowMajor()

            # Storage order of Julia sparse matrices.
            @test @inferred(StorageOrder(SparseMatrixCSC)) === ColumnMajor()
            @test @inferred(StorageOrder(Adjoint{<:Any,SparseMatrixCSC})) === RowMajor()
            @test @inferred(StorageOrder(Transpose{<:Any,SparseMatrixCSC})) === RowMajor()

            # Storage order of Julia abstract arrays.
            @test @inferred(StorageOrder(AbstractArray)) === StorageOrderAny()
            @test @inferred(StorageOrder(Adjoint{<:Any,AbstractArray})) === StorageOrderAny()
            @test @inferred(StorageOrder(Transpose{<:Any,AbstractArray})) === StorageOrderAny()
        end
    end
end

end # module

nothing
