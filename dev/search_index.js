var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Home-1",
    "page": "Home",
    "title": "Home",
    "category": "section",
    "text": "This is the documentation of the LazyAlgebra package for Julia. The sources are here."
},

{
    "location": "#Contents-1",
    "page": "Home",
    "title": "Contents",
    "category": "section",
    "text": "Pages = [\"install.md\", \"introduction.md\", \"vectors.md\", \"sparse.md\", \"mappings.md\"]"
},

{
    "location": "#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "install/#",
    "page": "Installation",
    "title": "Installation",
    "category": "page",
    "text": ""
},

{
    "location": "install/#Installation-1",
    "page": "Installation",
    "title": "Installation",
    "category": "section",
    "text": "LazyAlgebra.jl is not yet an offical Julia package but it is easy to install it from Julia. At the REPL of Julia, hit the ] key to switch to the package manager REPL (you should get a ... pkg> prompt) and type:pkg> add https://github.com/emmt/LazyAlgebra.jl.gitwhere pkg> represents the package manager prompt and https protocol has been assumed; if ssh is more suitable for you, then type:pkg> add git@github.com:emmt/LazyAlgebra.jl.gitinstead.  To check whether the LazyAlgebra package works correctly, type:pkg> test LazyAlgebraLater, to update to the last version (and run tests), you can type:pkg> update LazyAlgebra\npkg> build LazyAlgebra\npkg> test LazyAlgebraIf something goes wrong, it may be because you already have an old version of LazyAlgebra.  Uninstall LazyAlgebra as follows:pkg> rm LazyAlgebra\npkg> gc\npkg> add https://github.com/emmt/LazyAlgebra.jl.gitbefore re-installing.To revert to Julia\'s REPL, hit the Backspace key at the ... pkg> prompt."
},

{
    "location": "introduction/#",
    "page": "Lazy algebra framework",
    "title": "Lazy algebra framework",
    "category": "page",
    "text": ""
},

{
    "location": "introduction/#Lazy-algebra-framework-1",
    "page": "Lazy algebra framework",
    "title": "Lazy algebra framework",
    "category": "section",
    "text": "LazyAlgebra is a Julia package to generalize the notion of matrices and vectors used in linear algebra.Many numerical methods (e.g. in numerical optimization or digital signal processing) involve essentially linear operations on the considered variables.  LazyAlgebra provides a framework to implement these kind of numerical methods independently of the specific type of the variables. This is exploited in OptimPackNextGen package, an attempt to provide most optimization algorithms of OptimPack in pure Julia.LazyAlgebra also provides a flexible and extensible framework for creating complex mappings and linear mappings to operate on the variables.A few concepts are central to LazyAlgebra:vectors represent the variables of interest and can be anything providing a few methods are implemented for their specific type;\nmappings are any functions between such vectors;\nlinear mappings (a.k.a. linear operators) behave linearly with respect to their arguments.There are several reasons to have special methods for basic vector operations rather than relying on Julia linear algebra methods.  First, the notion of vector is different, in Julia a mono-dimensional array is a vector while, here any object with embedded values can be assumed to be a vector providing a subset of methods are specialized for this type of object.  For instance, LazyAlgebra provides such methods specialized for real-valued and complex-valued (with real components) arrays of any dimensionality.  Second, the meaning of the methods may have to be different.  For instance, only real-valued functions can be minimized (or maximized) and for this task, complex-valued variables can just be considered as real-valued variables (each complex value being equivalent to a pair of reals)."
},

{
    "location": "introduction/#Mappings-1",
    "page": "Lazy algebra framework",
    "title": "Mappings",
    "category": "section",
    "text": "LazyAlgebra features:flexible and extensible framework for creating complex mappings;\nlazy evaluation of the mappings;\nlazy assumptions when combining mappings;\nefficient memory allocation by avoiding temporaries."
},

{
    "location": "introduction/#General-mappings-1",
    "page": "Lazy algebra framework",
    "title": "General mappings",
    "category": "section",
    "text": "A Mapping can be any function between two variables spaces.  Using Householder-like notation (that is upper case Latin letters denote mappings, lower case Latin letters denote variables, and Greek letters denote scalars), then:A(x), A*x or A⋅x yields the result of applying the mapping A to x;\nA\\x yields the result of applying the inverse of A to x;Simple constructions are allowed for any kind of mappings and can be used to create new instances of mappings which behave correctly.  For instance:B = α*A (where α is a number) is a mapping which behaves as A times α; that is B(x) yields the same result as α*(A(x)).\nC = A + B + ... is a mapping which behaves as the sum of the mappings A, B, ...; that is C(x) yields the same result as A(x) + B(x) + ....\nC = A*B, C = A∘B or C = A⋅B is a mapping which behaves as the composition of the mappings A and B; that is C⋅x yields the same result as A(B(x)).  As for the sum of mappings, there may be an arbitrary number of mappings in a composition; for example, if D = A*B*C then D(x) yields the same result as A(B(C(x))).\nC = A\\B is a mapping such that C(x) yields the same result as inv(A)(B(x)).\nC = A/B is a mapping such that C(x) yields the same result as A(inv(B)(x)).These constructions can be combined to build up more complex mappings.  For example:D = A*(B + 3C) is a mapping such that D⋅x yields the same result as A(B(x) + 3*C(x))."
},

{
    "location": "introduction/#Linear-mappings-1",
    "page": "Lazy algebra framework",
    "title": "Linear mappings",
    "category": "section",
    "text": "A LinearMapping can be any linear mapping between two spaces.  This abstract subtype of Mapping is introduced to extend the notion of matrices and vectors.  Assuming the type of A inherits from LinearMapping, then:for linear mappings A and B, A⋅B is the same as A∘B or A*B which yields the composition of A and B whose effect is to apply B and then A;\nA\'⋅x and A\'*x yields the result of applying the adjoint of the mapping A to x;\nA\'\\x yields the result of applying the adjoint of the inverse of mapping A to x.\nB = A\' is a mapping such that B⋅x yields the same result as A\'⋅x.note: Note\nBeware that, due to the priority of operators in Julia, A*B(x) is the same as A(B(x)) not (A*B)(x)."
},

{
    "location": "introduction/#Automatic-simplifications-1",
    "page": "Lazy algebra framework",
    "title": "Automatic simplifications",
    "category": "section",
    "text": "An important feature of LazyAlgebra framework for mappings is that a number of simplifications are automatically made at contruction time.  For instance, assuming A is a mapping:B = A\'\nC = B\'yields C which is just a reference to A. In other words, adjoint(adjoint(A)) -> A holds.  LikelyD = inv(A)\nE = inv(D)yields E which is another reference to A.  In other words, inv(inv(A)) -> A holds assuming by default that A is invertible.  This follows the principles of laziness.  It is however, possible to prevent this by extending the Base.inv method so as to throw an exception when applied to the specific type of A:Base.inv(::SomeNonInvertibleMapping) = error(\"non-invertible mapping\")where SomeNonInvertibleMapping <: Mapping is the type of A.Other example of simplifications:B = 3A\nC = 7B\'where mappings B and C are such that B*x ≡ 3*(A*x) and C*x ≡ 21*(A*x) for any vector x.  That is C*x is evaluated as 21*(A*x) not as 7*(3*(A*x)) thanks to simplifications occurring while the mapping C is constructed.Using the ≡ to denote in the right-hand side the actual construction made by LazyAlgebra for the expression in the left-hand side and assuming A, B and C are linear mappings, the following simplications will occur:(A + C + B + 3C)\' ≡ A\' + B\' + 4C\'\n(A*B*3C)\'         ≡ 3C\'*B\'*A\'\ninv(A*B*3C)       ≡ 3\\inv(C)*inv(B)*inv(A)However, if M is a non-linear mapping, then:inv(A*B*3M) ≡ inv(M)*(3\\inv(B))*inv(A)which can be compared to inv(A*B*3C) when all operands are linear mappings.note: Note\nDue to the associative rules applied by Julia, parentheses are needed around constructions like 3*C if it has to be interpreted as 3C in all contexes.  Otherwise, A*B*(3*C) is equivalent to A*B*3C while A*B*3*C is interpreted as ((A*B)*3)*C; that is, compose A and B, apply A*B to 3 and right multiply the result by C."
},

{
    "location": "introduction/#Creating-new-mappings-1",
    "page": "Lazy algebra framework",
    "title": "Creating new mappings",
    "category": "section",
    "text": "LazyAlgebra provides a number of simple mappings.  Creating new primitive mapping types (not by combining existing mappings as explained above) which benefit from the LazyAlgebra framework is as simple as declaring a new mapping subtype of Mapping (or one of its abstract subtypes) and extending two methods vcreate and apply! specialized for the new mapping type.  For mode details, see here."
},

{
    "location": "vectors/#",
    "page": "Methods for vectors",
    "title": "Methods for vectors",
    "category": "page",
    "text": ""
},

{
    "location": "vectors/#Methods-for-vectors-1",
    "page": "Methods for vectors",
    "title": "Methods for vectors",
    "category": "section",
    "text": "A vector is that which has the algebra of a vector space (Peano 1888, van der Waerden 1931).  See talk by Jiahao Chen: Taking Vector Transposes Seriously at JuliaCon 2017."
},

{
    "location": "vectors/#Vectorized-methods-1",
    "page": "Methods for vectors",
    "title": "Vectorized methods",
    "category": "section",
    "text": "Most necessary operations on the variables of interest are linear operations. Hence variables (whatever their specific type and size) are just called vectors in LazyAlgebra.  Numerical methods based on LazyAlgebra manipulate the variables via a small number of vectorized methods:vdot([T,][w,]x,y) yields the inner product of x and y; that is, the sum of conj(x[i])*y[i] or, if w is specified, the sum of w[i]*conj(x[i])*y[i], for all indices i.  Optional argument T is the type of the result; for real valued vectors, T is a floating-point type; for complex valued vectors, T can be a complex type (with floating-point parts) or a floating-point type to compute only the real part of the inner product.  vdot([T,]sel,x,y) yields the sum of x[i]*y[i] for all i ∈ sel where sel is a selection of indices.\nvnorm1([T,]x) yields the L-1 norm of x, that is the sum of the absolute values of the components of x.  Optional argument T is the floating-point type of the result.\nvnorm2([T,]x) yields the Euclidean (or L-2) norm of x, that is the square root of sum of the squared values of the components of x.  Optional argument T is the floating-point type of the result.\nvnorminf([T,]x) L-∞ norm of x, that is the maximal absolute values of the components of x.  Optional argument T is the floating-point type of the result\nvcreate(x) yields a new variable instance similar to x.  If x is an array, the element type of the result is a floating-point type.\nvcopy!(dst,src) copies the contents of src into dst and returns dst.\nvcopy(x) yields a fresh copy of the vector x.\nvswap!(x,y) exchanges the contents of x and y (which must have the same type and size if they are arrays).\nvfill!(x,α) sets all elements of x with the scalar value α and return x.\nvzero!(x)fills x with zeros and returns it.\nvscale!(dst,α,src) overwrites dst with α*src and returns dst.  The convention is that, if α = 0, then dst is filled with zeros whatever the contents of src.\nvscale!(x,α) and vscale!(α,x) overwrite x with α*x and returns x. The convention is that, if α = 0, then x is filled with zeros whatever its prior contents.\nvscale(α,x) and vscale(x,α) yield a new vector whose elements are those of x multiplied by the scalar α.\nvproduct!(dst,[sel,]x,y) overwrites dst with the elementwise multiplication of x by y.  Optional argument sel is a selection of indices to consider.\nvproduct(x,y) yields the elementwise multiplication of x by y.\nvupdate!(y,[sel,]α,x) overwrites y with α*x + y and returns y. Optional argument sel is a selection of indices to which apply the operation (if an index is repeated, the operation will be performed several times at this location).\nvcombine(α,x,β,y) yields the linear combination α*x or α*x + β*y.\nvcombine!(dst,α,x,β,y) overwrites dst with the linear combination dst = α*x or dst = α*x + β*y and returns dst.Note that the names of these methods all start with a v (for vector) as the conventions used by these methods may be particular.  For instance, compared to copy! and when applied to arrays, vcopy! imposes that the two arguments have exactly the same dimensions.  Another example is the vdot method which has a slightly different semantics than Julia dot method.LazyAlgebra already provides implementations of these methods for Julia arrays with floating-point type elements.  This implementation assumes that an array is a valid vector providing it has suitable type and dimensions."
},

{
    "location": "vectors/#Implementing-a-new-vector-type-1",
    "page": "Methods for vectors",
    "title": "Implementing a new vector type",
    "category": "section",
    "text": "To have a numerical method based on LazyAlgebra be applicable to a new given type of variables, it is sufficient to implement a subset of these basic methods specialized for this kind of variables.The various operations that should be implemented for a vector are:compute the inner product of two vectors of the same kind (vdot(x,y) method);\ncreate a vector of a given kind (vcreate(x) method);\ncopy a vector (vcopy!(dst,src));\nfill a vector with a given value (vfill!(x,α) method);\nexchange the contents of two vectors (vswap!(x,y) method);\nlinearly combine several vectors (vcombine!(dst,α,x,β,y) method).Derived methods are:compute the Euclidean norm of a vector (vnorm2 method, based on vdot by default);\nmultiply a vector by a scalar: vscale!(dst,α,src) and/or vscale!(x,α) methods (based on vcombine! by default);\nupdate a vector by a scaled step: vupdate!(y,α,x) method (based on vcombine! by default) and, for some constrained optimization methods, vupdate!(y,sel,α,x) method;\nerase a vector: vzero!(x) method (based on vfill! by default);\nvscale and vcopy methods are implemented with vcreate and respectivelyvscale! and vcopy!.Other methods which may be required by some packages:compute the L-1 norm of a vector: vnorm1(x) method;\ncompute the L-∞ norm of a vector: vnorminf(x) method;Methods that must be implemented (V represent the vector type):vdot(::Type{T}, x::Tx, y::Ty) :: T where {T<:AbstractFloat,Tx,Ty}vscale!(dst::V, alpha::Real, src::V) -> dstmethods that may be implemented:vscale!(alpha::Real, x::V) -> xFor mappings and linear operators (see Implementation of new mappings for details), implement:apply!(α::Scalar, P::Type{<:Operations}, A::Ta, x::Tx, β::Scalar, y::Ty) -> yandvcreate(P::Type{P}, A::Ta, x::Tx) -> yfor Ta<:Mapping and the supported operations P<:Operations."
},

{
    "location": "sparse/#",
    "page": "Sparse operators",
    "title": "Sparse operators",
    "category": "page",
    "text": ""
},

{
    "location": "sparse/#Sparse-operators-1",
    "page": "Sparse operators",
    "title": "Sparse operators",
    "category": "section",
    "text": "A sparse operator (SparseOperator) in LazyAlgebra is the generalization of a sparse matrix.  Like a GeneralMatrix, rows and columns may be multi-dimensional."
},

{
    "location": "sparse/#Construction-1",
    "page": "Sparse operators",
    "title": "Construction",
    "category": "section",
    "text": "A sparse operator can be built as follows:SparseOperator(I, J, C, rowdims, coldims)where I and J are row and column indices of the non-zero coefficients whose values are specified by C and with rowdims and coldims the dimensions of the rows and of the columns.  Appart from the fact that the rows and columns may be multi-dimensional, this is very similar to the sparse method in SparseArrays standard Julia module.Another possibility is to build a sparse operator from an array or from a sparse matrix:S = SparseOperator(A)where A is an array or a sparse matrix (of type SparseMatrixCSC and provided by the SparseArrays standard Julia module).  If A is an array with more than 2 dimensions, the number n of dimensions corresponding to the rows of the operator can be specified:S = SparseOperator(A, n)If not specified, n=1 is assumed.A sparse operator can be converted to a regular array, to a regular matrix or to a sparse matrix.  Assuming S is a SparseOperator, convertions to other representations are done by:A = Array(S)     # convert S to an array\nM = Matrix(S)    # convert S to a matrix\nsp = sparse(S)   # convert S to a sparse matrixnote: Note\nIf the sparse operator S has multi-dimensional columns/rows, these dimensions are preserved when S is converted to an array but are silently flatten when S is converted to a matrix or to a sparse matrix.Package LinearInterpolators provides a SparseInterpolator which is a LazyAlgebra LinearMapping and which can also be converted to a SparseOperator (sse the documentation of this package)."
},

{
    "location": "sparse/#Usage-1",
    "page": "Sparse operators",
    "title": "Usage",
    "category": "section",
    "text": "A sparse operator can be used as any other LazyAlgebra linear mapping, e.g., S*x yields the result of applying the sparse operator S to x (unless x is a scalar, see below).A sparse operator can be reshaped:reshape(S, rowdims, coldims)where rowdims and coldims are the new list of dimensions for the rows and the columns, their product must be equal to the product of the former lists of dimensions.  The reshaped sparse operator and S share the arrays of non-zero coefficients and corresponding row and column indices.Left or right composition of a by a sparse operator by NonuniformScalingOperator (with comptable dimensions) yields another sparse operator whith same row and column indices but scaled coefficients.  A similar simplification is performed when a sparse operator is left or right multiplied by a scalar.The non-zero coefficients of a sparse operator S can be unpacked into a provided array A:unpack!(A, S) -> Awhere A must have the same element type as the coefficients of S and the same number of elements as the the products of the row and of the column dimensions of S.  Unpacking is perfomed by adding the non-zero coefficients of S to the correponding element of A (or using the | operator for boolean elements).  Hence unpacking into an array of zeros with appropriate dimensions yields the same result as Array(S)."
},

{
    "location": "mappings/#",
    "page": "Methods for mappings",
    "title": "Methods for mappings",
    "category": "page",
    "text": ""
},

{
    "location": "mappings/#Methods-for-mappings-1",
    "page": "Methods for mappings",
    "title": "Methods for mappings",
    "category": "section",
    "text": "LazyAlgebra provides a number of mappings and linear operators.  To create new primitive mapping types (not by combining existing mappings) and benefit from the LazyAlgebra infrastruture, you have to:Create a new type derived from Mapping or one of its abstract subtypes such as LinearMapping.\nImplement at least two methods apply! and vcreate specialized for the new mapping type.  Applying the mapping is done by the former method.  The latter method is called to create a new output variable suitable to store the result of applying the mapping (or one of its variants) to some input variable."
},

{
    "location": "mappings/#The-vcreate-method-1",
    "page": "Methods for mappings",
    "title": "The vcreate method",
    "category": "section",
    "text": "The signature of the vcreate method is:vcreate(::Type{P}, A::Ta, x::Tx, scratch::Bool=false) -> ywhere A is the mapping, x its argument and P is one of Direct, Adjoint, Inverse and/or InverseAdjoint (or equivalently AdjointInverse) and indicates how A is to be applied:Direct to apply A to x, e.g. to compute A⋅x;\nAdjoint to apply the adjoint of A to x, e.g. to compute A\'⋅x;\nInverse to apply the inverse of A to x, e.g. to compute A\\x;\nInverseAdjoint or AdjointInverse to apply the inverse of A\' to x, e.g. to compute A\'\\x.The result returned by vcreate is a new output variables suitable to store the result of applying the mapping A (or one of its variants as indicated by P) to the input variables x.The optional argument scratch is a boolean to let the caller indicate whether the input variable x may be re-used to store the result.  If scratch is true and if that make sense, the value returned by vcreate may be x.  The default value of scratch must be false.  Calling vcreate with scratch=true can be used to limit the allocation of resources when possible. Having scratch=true is only indicative and a specific implementation of vcreate may legitimately always assume scratch=false and return a new variable whatever the value of this argument (e.g. because applying the considered mapping in-place is not possible or because the considered mapping is not an endomorphism).  Of course, the opposite behavior (i.e., assuming that scratch=true while the method was called with scratch=false) is forbidden."
},

{
    "location": "mappings/#The-apply!-method-1",
    "page": "Methods for mappings",
    "title": "The apply! method",
    "category": "section",
    "text": "The signature of the apply! method is:apply!(α::Real, ::Type{P}, A::Ta, x::Tx, scratch::Bool=false, β::Real, y::Ty) -> yThis method shall overwrites the contents of output variables y with the result of α*P(A)⋅x + β*y where P is one of Direct, Adjoint, Inverse and/or InverseAdjoint (or equivalently AdjointInverse) and shall return y.  The convention is that the prior contents of y is not used at all if β = 0 so the contents of y does not need to be initialized in that case.Not all operations P must be implemented, only the supported ones.  For iterative resolution of (inverse) problems, it is generally needed to implement at least the Direct and Adjoint operations for linear operators.  However nonlinear mappings are not supposed to implement the Adjoint and derived operations.Argument scratch is a boolean to let the caller indicate whether the contents of the input variable x may be overwritten during the operations.  If scratch=false, the apply! method shall not modify the contents of x."
},

{
    "location": "mappings/#Example-1",
    "page": "Methods for mappings",
    "title": "Example",
    "category": "section",
    "text": "The following example implements a simple sparse linear operator which is able to operate on multi-dimensional arrays (the so-called variables):using LazyAlgebra\nimport LazyAlgebra: vcreate, apply!, input_size, output_size\n\nstruct SparseOperator{T<:AbstractFloat,M,N} <: LinearMapping\n    outdims::NTuple{M,Int}\n    inpdims::NTuple{N,Int}\n    A::Vector{T}\n    I::Vector{Int}\n    J::Vector{Int}\nend\n\ninput_size(S::SparseOperator) = S.inpdims\noutput_size(S::SparseOperator) = S.outdims\n\nfunction vcreate(::Type{Direct}, S::SparseOperator{Ts,M,N},\n                 x::DenseArray{Tx,N}) where {Ts<:Real,Tx<:Real,M,N}\n    @assert size(x) == input_size(S)\n    Ty = promote_type(Ts, Tx)\n    return Array{Ty}(undef, output_size(S))\nend\n\nfunction vcreate(::Type{Adjoint}, S::SparseOperator{Ts,M,N},\n                 x::DenseArray{Tx,M}) where {Ts<:Real,Tx<:Real,M,N}\n    @assert size(x) == output_size(S)\n    Ty = promote_type(Ts, Tx)\n    return Array{Ty}(undef, input_size(S))\nend\n\nfunction apply!(alpha::Real,\n                ::Type{Direct},\n                S::SparseOperator{Ts,M,N},\n                x::DenseArray{Tx,N},\n                beta::Real,\n                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}\n    @assert size(x) == input_size(S)\n    @assert size(y) == output_size(S)\n    beta == 1 || vscale!(y, beta)\n    if alpha != 0\n        A, I, J = S.A, S.I, S.J\n        _alpha_ = convert(promote_type(Ts,Tx,Ty), alpha)\n        @assert length(I) == length(J) == length(A)\n        for k in 1:length(A)\n            i, j = I[k], J[k]\n            y[i] += _alpha_*A[k]*x[j]\n        end\n    end\n    return y\nend\n\nfunction apply!(alpha::Real, ::Type{Adjoint},\n                S::SparseOperator{Ts,M,N},\n                x::DenseArray{Tx,M},\n                beta::Real,\n                y::DenseArray{Ty,N}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}\n    @assert size(x) == output_size(S)\n    @assert size(y) == input_size(S)\n    beta == 1 || vscale!(y, beta)\n    if alpha != 0\n        A, I, J = S.A, S.I, S.J\n        _alpha_ = convert(promote_type(Ts,Tx,Ty), alpha)\n        @assert length(I) == length(J) == length(A)\n        for k in 1:length(A)\n            i, j = I[k], J[k]\n            y[j] += _alpha_*A[k]*x[i]\n        end\n    end\n    return y\nendRemarks:In our example, arrays are restricted to be dense so that linear indexing is efficient.  For the sake of clarity, the above code is intended to be correct although there are many possible optimizations.\nIf alpha = 0 there is nothing to do except scale y by beta.\nThe call to vscale!(beta, y) is to properly initialize y.  Remember the convention that the contents of y is not used at all if beta = 0 so y does not need to be properly initialized in that case, it will simply be zero-filled by the call to vscale!.  The statements\nbeta == 1 || vscale!(y, beta)\nare equivalent to:\nif beta != 1\n    vscale!(y, beta)\nend\nwhich may be simplified to just calling vscale! unconditionally:\nvscale!(y, beta)\nas vscale!(y, beta) does nothing if beta = 1.\n@inbounds could be used for the loops but this would require checking that all indices are whithin the bounds.  In this example, only k is guaranteed to be valid, i and j have to be checked."
},

]}
