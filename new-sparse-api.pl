#! /usr/bin/perl -pi.orig
s/\bget_vals\b/nonzeros/g;
s/\bget_offs\b/offsets/g;
s/\bget_rows\b/row_indices/g;
s/\bget_row\b/row_index/g;
s/\beach_row\b/each_row_index/g;
s/\bget_cols\b/col_indices/g;
s/\bget_col\b/col_index/g;
s/\beach_nz\b/each_nz_index/g;
s/\beach_col\b/each_col_index/g;
s/\bcopy_rows\b/copy_row_indices/g;
s/\bcopy_cols\b/copy_col_indices/g;
s/\bcopy_vals\b/copy_nonzeros/g;

# ASAP
# s/\bget_perm\b/permutations/g;
# s/\bget_rank\( *(\w+) *\)\b/ranks(\1)/g; # FIXME
# s/\bget_rank\( *(\w+) *, (\w+) *\)\b/index_to_rank(\1, \2)/g; # FIXME
#
# s/\bdiag_nz\b/diag_nz_index/g;
# s/\bsplit_nz\b/split_nz_range/g;
