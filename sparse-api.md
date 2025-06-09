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

Example (adapted from the doc.) of use with `SparseArrays` API:

```julia
A = sparse(I,J,V)
rows = rowvals(A)
vals = nonzeros(A)
m, n = size(A)
@inbounds for j = 1:n
    for k in nzrange(A, j)
        i = rows[k]
        Aᵢⱼ = vals[k]
        # perform sparse wizardry...
    end
end
```

With the proposed API, above example becomes:

```julia
A = sparse(I,J,V)
vals = nonzeros(A)
@inbounds for j = each_col_index(A)
    for k in each_nz_index(A, j)
        i = row_index(A, k)
        Aᵢⱼ = vals[k]
        # perform sparse wizardry...
    end
end
```

It may be noted that Julia's sparse matrices are stored in *Compressed Sparse Column*
(CSC) format.

Same example for a `LazyAlgebra` sparse operator in CSC format:

```julia
# A is in CSC format
@inbounds for j in each_col_index(A)
    for k in each_nz_index(A, j)
        i = row_index(A, k)
        Aᵢⱼ = A[k]
        # perform sparse wizardry...
    end
end
```

which is very similar to accessing a Julia sparse matrix.

Same example for a `LazyAlgebra` sparse operator in *Compressed Sparse Row* (CSR) format:

```julia
# A is in CSR format
@inbounds for i in each_row_index(A)
    for k in each_nz_index(A, i)
        j = col_index(A, k)
        Aᵢⱼ = A[k]
        # perform sparse wizardry...
    end
end
```

Same example for a `LazyAlgebra` sparse operator in *Compressed Coordinates* (COO) format:

```julia
# A is in COO format
@inbounds for k in each_nz_index(A)
    i = row_index(A, k)
    j = col_index(A, k)
    Aᵢⱼ = A[k]
    # perform sparse wizardry...
end
```
