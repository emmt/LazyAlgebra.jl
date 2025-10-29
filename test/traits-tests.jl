using LazyAlgebra
using Test
using LinearAlgebra
using Neutrals
using SparseArrays

using LazyAlgebra:
    Adjoint,
    ColumnMajor,
    Inverse,
    Prod,
    RowMajor,
    Scaled,
    StorageOrder,
    StorageOrderUnknown,
    Sum,
    Transpose,
    divide,
    inverse,
    is_column_major,
    is_row_major

@testset "Traits" begin
    @testset "Storage Order" begin
        # Storage order or storage order instances.
        @test @inferred(StorageOrder(StorageOrderUnknown())) === StorageOrderUnknown()
        @test @inferred(is_row_major(StorageOrderUnknown())) === false
        @test @inferred(is_column_major(StorageOrderUnknown())) === false

        @test @inferred(StorageOrder(ColumnMajor())) === ColumnMajor()
        @test @inferred(is_row_major(ColumnMajor())) === false
        @test @inferred(is_column_major(ColumnMajor())) === true

        @test @inferred(StorageOrder(RowMajor())) === RowMajor()
        @test @inferred(is_row_major(RowMajor())) === true
        @test @inferred(is_column_major(RowMajor())) === false

        # Storage order or storage order types.
        @test @inferred(StorageOrder(StorageOrder)) === StorageOrderUnknown()
        @test @inferred(is_row_major(StorageOrder)) === false
        @test @inferred(is_column_major(StorageOrder)) === false

        @test @inferred(StorageOrder(StorageOrderUnknown)) === StorageOrderUnknown()
        @test @inferred(is_row_major(StorageOrderUnknown)) === false
        @test @inferred(is_column_major(StorageOrderUnknown)) === false

        @test @inferred(StorageOrder(ColumnMajor)) === ColumnMajor()
        @test @inferred(is_row_major(ColumnMajor)) === false
        @test @inferred(is_column_major(ColumnMajor)) === true

        @test @inferred(StorageOrder(RowMajor)) === RowMajor()
        @test @inferred(is_row_major(RowMajor)) === true
        @test @inferred(is_column_major(RowMajor)) === false

        # Transposition of storage order.
        @test @inferred(transpose(StorageOrderUnknown())) === StorageOrderUnknown()
        @test @inferred(transpose(RowMajor())) === ColumnMajor()
        @test @inferred(transpose(ColumnMajor())) === RowMajor()

        # Unknown storage order.
        A = SymbolicOperator(:A)

        @test @inferred(StorageOrder(A)) === StorageOrderUnknown()
        @test @inferred(StorageOrder(A')) === StorageOrderUnknown()
        @test @inferred(StorageOrder(transpose(A))) === StorageOrderUnknown()

        @test @inferred(StorageOrder(typeof(A))) === StorageOrderUnknown()
        @test @inferred(StorageOrder(typeof(A'))) === StorageOrderUnknown()
        @test @inferred(StorageOrder(typeof(transpose(A)))) === StorageOrderUnknown()

        @test @inferred(is_row_major(A)) === false
        @test @inferred(is_row_major(A')) === false
        @test @inferred(is_row_major(transpose(A))) === false

        @test @inferred(is_column_major(A)) === false
        @test @inferred(is_column_major(A')) === false
        @test @inferred(is_column_major(transpose(A))) === false

        # Storage order of sparse operator in COO format.
        A = SparseOperatorCOO([-1.0, 2.0, 0.0, 4.0, 7.0], [1, 1, 2, 3, 3], [1, 2, 1, 2, 4], (3,), (4,))

        @test @inferred(StorageOrder(A)) === StorageOrderUnknown()
        @test @inferred(StorageOrder(A')) === StorageOrderUnknown()
        @test @inferred(StorageOrder(transpose(A))) === StorageOrderUnknown()

        @test @inferred(StorageOrder(typeof(A))) === StorageOrderUnknown()
        @test @inferred(StorageOrder(typeof(A'))) === StorageOrderUnknown()
        @test @inferred(StorageOrder(typeof(transpose(A)))) === StorageOrderUnknown()

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
            @test @inferred(StorageOrder(AbstractArray)) === StorageOrderUnknown()
            @test @inferred(StorageOrder(Adjoint{<:Any,AbstractArray})) === StorageOrderUnknown()
            @test @inferred(StorageOrder(Transpose{<:Any,AbstractArray})) === StorageOrderUnknown()
        end
    end
end

nothing
