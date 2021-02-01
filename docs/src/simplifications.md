# Simplification and optimization of combinations of mappings

LazyAlgebra provides mappings and structures to store arbitrarily complex
combinations of mappings.  When constructing such combinations, a number of
simplifications are automatically performed.  These simplifications follow a
number of rules which are explained below.


## Rationale for simplification rules

A distinction must be made between:

- *basic* mappings which are the simplest mappings and are the building blocks
  of more complex constructions,
- *scaled* or *decorated* mappings (see [`LazyAlgebra.DecoratedMapping`](@ref))
  which are simple structures wrapped around a mapping and which are almost
  costless to build,
- *combinations* of mappings which are sums or compositions of mappings.

As general guidelines, simplification that are automatically performed at
construction time should:

- require as few computations as possible,
- avoid creating new mappings with mutable contents,
- be type stable if possible,
- preserve basic mappings embedded in more complex constructions to
  make it more likely to realize that two mappings be identical by their types.

Imposing *type stability* is not a strict requirement as it would prevent a lot
of worthwile simplifications like:

- `λ*A` yields `A` if `λ = 1`, a scaled mapping otherwise,

- `A + B` yields `2*A` if `B ≡ A`, a sum of mappings otherwise;

There are some cases where the rules to follow are not obvious:

* Multiplying a sum of mappings, say `λ*(A + B + C)`, can be constructed as:
  1. a sum of scaled mappings, `λ*A + λ*B + λ*C`;
  2. a scaled sum.

* Taking the adjoint of a sum of mappings, say `(A + B + C)'`, can be:
  1. simplified into a sum of adjoints, `A' + B' + C'`;
  2. constructed as the adjoint of the sum.

* Taking the adjoint of a composition of mappings, say `(A*B*C)'`, can be:
  1. simplified into a composition of adjoints, `C'*B'*A'`;
  2. constructed as the adjoint of the composition.

* Taking the inverse of a composition of mappings, say `inv(A*B*C)'`, can be:
  1. simplified into a composition of inverses, `inv(C)*inv(B)*inv(A)`;
  2. constructed as the inverse of the composition.

For all these cases, the second solution requires that (i) the `apply` method
be specialized to be applicable to the resulting construction and, for
consistency, that (ii) the other possible expressions be automatically
recognized and constructed in the same way (for instance ``C'*B'*A'` should be
automatically simplified as `(A*B*C)'`.

In the case of the scaling of a sum, it is more efficient to use the second
form because is factorize the multiplication at the end of calculus.

On the one hand, the first solution is more simple to implement.  On the other
hand, with the second solution, it is easier to write simplification rules that
apply automatically.

The curent rules in LazyAlgebra (see the next section) implement the second
solution for the multiplication of a sum by a scalar and the first solution in
all other case.  This is expected to change in the future.


## Implemented simplification rules:

- Multipliers are factorized to the left as possible.

- Adjoint of a sum (or a composition) of terms is rewritten as the sum
  (respectively composition) of the adjoint of the terms.

- Adjoint of a scaled mapping is rewritten as a scaled adjoint of the
  mapping.  Similarly, inverse of a scaled mapping is rewritten as a scaled
  inverse of the mapping, if the mapping is linear, or as the inverse of the
  mapping times a scaled identity otherwise.

- Adjoint of the inverse is rewritten as inverse of the adjoint.

- Inner constructors are fully qualified but check arguments.  Un-qualified
  outer constructors just call the inner constructors with the suitable
  parameters.

- To simplify a sum, the terms corresponding to identical mappings (possibly
  scaled) are first grouped to produce a single mapping (possibly scaled)
  per group, the resulting terms are sorted (so that all equivalent
  expressions yield the same result) and the "zeros" eliminated (if all
  terms are "zero", the sum simplifies to the first one).  For now, the
  sorting is not absolutely perfect as it is based on `objectid()` hashing
  method.  The odds of having the same identifier for two different things
  are however extremely low.

- To simplify a composition, a fusion algorithm is applied and "ones" are
  eliminated.  It is assumed that composition is non-commutative so the
  ordering of terms is left unchanged.  Thanks to this, simplification rules
  for simple compositions (made of two non-composition mappings) can be
  automatically performed by proper dispatching rules.  Calling the fusion
  algorithm is only needed for more complex compositions.

The simplication algorithm is not perfect (LazyAlgebra is not intended to be
for symbolic computations) but do a reasonnable job.  In particular complex
mappings built using the same sequences should be simplified in the same way
and thus be correctly identified as being identical.

Since applying a construction of mappings will result in applying its
components, it can

```julia
+(A::Adjoint, B::Adjoint) = (A + B)'
*(A::Adjoint, B::Adjoint) = (B*A)'
*(A::Inverse, B::Inverse) = inv(B*A)
*(A::Inverse{T}, B::T) where {T<:Mapping} =
    (identical(unveil(A), B) ? Id : Composition(A,B))
*(A::T, B::Inverse{T}) where {T<:Mapping} =
    (identical(A, unveil(B)) ? Id : Composition(A,B))
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)
adjoint(A::Adjoint) = unveil(A)
adjoint(A::AdjointInverse) = inv(A)
adjoint(A::Inverse) = AdjointInverse(A)
inv(A::Inverse) = unveil(A)
inv(A::Adjoint) = AdjointInverse(A)
inv(A::AdjointInverse) = A'
```


## Coding recommendations

To make the code easier to maintain and avoid incosistencies, there are a few
recommendations to follow.  This is especially true for coding the
simplification rules that are automatically performed.

* Simplification rules are initiated by specializing for mapping arguments the
  operators (addition, multiplication, adjoint, etc.) used in Julia expressions.
  Hence simplifications are automatically performed when such expressions
  appears in the code.

* More complex rules may require calling auxiliary helper functions like
  `simplify`.  But the entry point for a simplification is always a simple
  expression so that the end user shall not have to call `simplify` directly.

* To avoid building non-simplified constructions, the `Adjoint`, `Inverse`,
  `AdjointInverse`, `Scaled`, `Sum`, `Composition`, and `Gram` constructors
  should not be directly called by a end user who should use expressions like
  `A'` to construct the adjoint, `inv(A)` for the inverse, etc.  To discourage
  calling constructors for combining mappings, these constructors are not
  exported by LazyAlgebra.

* Trust the optimizer and resist to the tendency of writing very specialized
  rules to deal with complex cases in favor of writing more simpler and more
  general rules that, applied together, yield the correct answer.

  For instance, the following rules would be sufficient to implement the
  right-multiplication and the right-division of a mapping by a scalar:

  ```julia
  *(A::Mapping, α::Number) = (is_linear(A) ? α*A : A*(α*Id))
  /(A::Mapping, α::Number) = A*inv(α)
  ```

  Only the right-multiplication is in charge of deciding whether the operation
  is commutative and these two methods returns their result as an expression to
  delegate the construction of the result to the methods implementing the
  left-multiplication of a mapping by a scalar, the composition of two
  mappings, etc.
