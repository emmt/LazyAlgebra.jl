| Proposed new API          | `SparseArrays`       | Old `LazyAlgebra` API | Old `ASAP` API        | `SparseOperators` |
|:--------------------------|:---------------------|:----------------------|:----------------------|:------------------|
| `nnz(A)`                  | `nnz(A)`             | `nnz(A)`              | `nnz(A)`              |                   |
| `nonzeros(A)`             | `nonzeros(A)`        | `get_vals(A)`         | `get_vals(A)`         |                   |
| `offsets(A)`              | `getcolptr(A)`       | `get_offs(A)`         | `get_offs(A)`         |                   |
| `row_indices(A)`          | `rowvals(A)`         | `get_rows(A)`         | `get_rows(A)`         |                   |
| `col_indices(A)`          |                      | `get_cols(A)`         | `get_cols(A)`         |                   |
| `permutations(A)`         |                      |                       | `get_perm(A)`         |                   |
| `ranks(A)`                |                      |                       | `get_rank(A)`         |                   |
| `each_nz_index(A,...)`    | `nzrange(A, j)`      | `each_nz(A[, ij])`    | `each_nz(A, r)`       |                   |
| `each_row_index(A)`       |                      | `each_row(A)`         |                       |                   |
| `each_col_index(A)`       |                      | `each_col(A)`         |                       |                   |
| `each_rank(A)`            |                      |                       | `each_rank(A)`        |                   |
| `row_index(A, rk)`        |                      |                       | `get_row(A, r)`       |                   |
| `col_index(A, rk)`        |                      |                       | `get_col(A, r)`       |                   |
| `A[k]`                    | `nonzeros(A)[k]`     | `get_val(A, k)`       | `get_val(A, k)`       |                   |
| `A[k] = v`                | `nonzeros(A)[k] = v` | `set_val!(A, k, v)`   | `set_val!(A, k, v)`   |                   |
| `diag_nz_index(A, r)`     |                      |                       | `diag_nz(A, r)`       |                   |
| `split_nz_range(A, r)`    |                      |                       | `split_nz(A,)`        |                   |
| `rank_to_index(A, r)`     |                      |                       | `rank_to_index(A, r)` |                   |
| `index_to_rank(A, ij)`    |                      |                       | `get_rank(A, ij)`     |                   |
| `valid_nodes(A)`          |                      |                       | `valid_nodes(A)`      |                   |
| `copy(nonzeros(A))`       |                      | `copy_vals(A)`        |                       |                   |
| `collect(row_indices(A))` |                      | `copy_rows(A)`        |                       |                   |
| `collect(col_indices(A))` |                      | `copy_cols(A)`        |                       |                   |

Notation:

- `A` is the sparse matrix (Julia `SparseArrays`) or operator (`LazyAlgebra` or `ASAP`);
- `i` is a row index;
- `j` is a column index;
- `ij` is a row or column index depending on the storage format of `A`;
- `r` is a rank index into the permutation indices;
- `k` is an index into the array of nonzeros;
- `m` is the number of rows;
- `n` is the number of columns;

The new API avoids some confusions like `get_row` or `get_col` which, in `LazyAlgebra` and
`ASAP`, used to yield a row or column index, while, `getrow` or `getcol`, in base Julia,
yield slices of matrix values along a row or a column.

In the proposed API and unlike for Julia sparse matrices, sparse operators are not meant
to be seen as abstract matrices, so the syntax `A[i,j]` to access the value at row `i` and
column `j` is not implemented. Instead, `A[k]` is used to directly access the `k`-th
nonzero and is a shortcut to `nonzeros(A)[k]` which is valid for sparse operators and
sparse matrices.

```
Format
|- RowWise
|   |- RowWiseLower
|   `- RowWiseUpper
`- ColumnWise
    |- ColumnWiseLower
    `- ColumnWiseUpper

LowerTriangularFormat = Union{RowWiseLower,ColumnWiseLower}
UpperTriangularFormat = Union{RowWiseUpper,ColumnWiseUpper}

AbstractSparseFactor{F<:Format,T,N}
|- BareSparseFactor{F<:Format,T,N}
|  `- SparseFactor{F,T,N}
`- Wrapped{F<:Format,T,N}
   |- Swapped{F,T,N}
   |  |- Transpose
   |  `- Adjoint
   `- Conjugate

Inverse
Gram
```
