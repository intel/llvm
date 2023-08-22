# SYCL Index Space Representation

Identifying a work-item in the SYCL index space (using its global id or the
combination of its work-group id and its local id within a work-group) is
necessary to perform memory access analysis on SYCL device code, so representing
these SYCL index space identifiers is required to enable such analysis through
the SYCL dialect.

**Note**: Through this document, we assume `memref` and `llvm.ptr` are used for
pointer types, `llvm.struct` for structured types and `sycl.cast` is the
operation used to perform a casting from a SYCL type to its base class (or one
of its base classes in case of multiple inheritance).

## SYCL ND-range operations

These operations query ND-range components such as the number of work items or
the global id. These can be ND `id` or `range` instances or `i32` scalars,
depending on the operation.

Note `i32` scalars will only be returned by operations encoding
sub-group-related information:

- `sycl.num_sub_groups`;
- `sycl.sub_group_size`;
- `sycl.sub_group_id`;
- `sycl.sub_group_local_id`;
- `sycl.sub_group_max_size`.

This decision was made as, looking [at the SYCL specification for the
`sub_group`
class](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sub-group-class),
we can find homologous member functions for the first four operations returning
`uint32_t` (represented as `i32` in our dialect). Also, for
`sycl.sub_group_max_size`, an homologous kernel descriptor of `uint32_t` type
can be found [in the SYCL
spec](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_kernel_information_descriptors).

For flavours of `sub_group` member functions returning an `id<1>` or `range<1>`,
a conversion between `i32` and the target type will be needed.

See the SYCL dialect documentation for more information on each operation.

### Lowering

The ND-range operations lower to platform-specific built-ins, e.g., SPIR-V
built-ins and the necessary casts. If the array subscript order differs, this is
taken into account.

### Design decisions

#### Not providing versions querying a particular dimension of the ND component

There are two reasons why these were omitted:

1. There is no correspondence with the SYCL specification: The SYCL spec points
   out the compiler should have tools to build kernel-input `id`, `item` and
   `nd_item`, (these operations are our representation of that), but it does not
   specify that there should be any user-callable function querying this, so no
   need for that flavor of these operations;
2. As per the lowering section, a "flip" of the components might be needed when
   lowering. In order to do so, we need to know the **dimensionality** of the
   kernel. As this wouldn't be possible in the general case, unless that is
   attached to the operation as an attribute, correct lowering of that flavor of
   these operations taking into account different array subscript orders would
   be impossible.

## `SYCLMethodOpInterface`

The `SYCLMethodOpInterface` operation interface is defined as the interface
describing an operation representing a member function call of a SYCL type. This
interface provides the following functions:

| Function                                                       | Description                                                           |
|:---------------------------------------------------------------|:----------------------------------------------------------------------|
| `static mlir::TypeID getTypeID();`                             | Return the ID of the type this method is a member of.                 |
| `static llvm::ArrayRef<llvm::StringLiteral> getMethodNames();` | Return the list of the method names to be replaced by this operation. |

The non-member functions `getTypeID` and `getMethodNames` return respectively
the ID of the type defined in the SYCL specification of which the function is a
member of and a set of names which must contain `FunctionName`. Note that
`getTypeID` does not refer to the base type present in the implementation from
which the SYCL type is derived, but to the type present in the SYCL
dialect. This decision removes the need of updating operations definitions if
implementation details change or a different implementation is to be supported.

See the SYCL dialect documentation for more information on each operation.

### `SYCLMethodOpInterfaceImpl`

To implement this interface consistently and avoid code repetition, the TableGen
`SYCLMethodOpInterfaceImpl` class is provided. Every SYCL dialect operation
implementing the aforementioned interface must be derived from this class.

### Registering operations implementing `SYCLMethodOpInterface`

The `SYCLDialect::addOperations()` function is defined, overriding
`Dialect::addOperations()`. This function calls the base implementation, but
also **registers** operations as `SYCLMethodOpInterface` instances. In order to
do so, a `MethodRegistry` member variable is added to the dialect. This holds a
`(TypeID, FunctionName) -> OperationName` table so that operations can be
registered as "methods" calling `MethodRegistry::registerMethod(mlir::TypeID,
llvm::StringRef, llvm::StringRef)`, where the first two parameters correspond to
the keys of the map (`getTypeID()` and `getMethodNames()`) and the last value,
to the value of the entry (the name of the operation).

This process will require no action from the users of the dialect. As this
mechanism is implemented in the `SYCLDialect::addOperations()` function,
defining a new function implementing `SYCLMethodOpInterface` in the TableGen
file also generates call to register it with no further action.

### Looking up operations implementing `SYCLMethodOpInterface`

In `cgeist` codegen, right before generating a `sycl.call` operation, we check
whether there is an operation implementing `SYCLMethodOpInterface` homologous to
that function call. In order to do that,
`SYCLDialect::lookupMethod(mlir::TypeID, llvm::StringRef)` can be used, passing
the ID of the SYCL type the function is a member of and the function name. After
looking up the name in its registry, `llvm::None` is returned if no operation
can be used to replace the `sycl.call`; otherwise, the name of the operation is
returned, which can be used to create a different operation instead.

Note that, in case the function is a member of a base type, the `sycl.cast`
operation used to cast to the base class is abstracted beforehand.

### Lowering

Lowering of these operations mimics the homologous function implementations in
[the SYCL headers](../../../sycl/include). These operations are lowered to
operations of different dialects (`llvm`, `arith`, `memref`, `spirv`, and
`vector`).

Note each SYCL implementation may need a different lowering. Currently only
DPC++ is supported. In order to support a new target, an homologous value should
be added to `mlir::sycl::LoweringTarget` and new patterns should be
implemented. See
[SYCLToLLVM.cpp](../../lib/Conversion/SYCLToLLVM/SYCLToLLVM.cpp) and
[DPCPP.cpp](../../lib/Conversion/SYCLToLLVM/DPCPP.cpp) for reference.

### Verification

Four different traits are introduced: `SYCLGetID`, `SYCLGetComponent`,
`SYCLGetRange`, `SYCLGetGroup`:

- `SYCLGetID`: This trait describes a SYCLMethodOpInterface that returns an ID
  if called with a single argument and a size_t if called with two arguments.
- `SYCLGetComponent`: This trait describes a SYCLMethodOpInterface that returns
  a range if called with a single argument and a size_t if called with two
  arguments.
- `SYCLGetRange`: This trait describes a SYCLMethodOpInterface that returns a
  range if called with a single argument and a size_t if called with two
  arguments.
- `SYCLGetGroup`: This trait describes an SYCLMethodOpInterface that returns a
  group if called with a single argument and a size_t if called with two
  arguments.

### Design decisions

#### Changing the type of some arguments w.r.t. the implementing function

This decision can be useful for two aspects:

1. It abstracts implementation details: no need to change operations definitions
   if implementation changes or a new SYCL implementation is to be supported;
2. Easier canonicalization, e.g., if the type of `this` and the type of the
   first argument matched, as the type of the operand would be `memref`, the
   canonicalization pass would need to abstract several cast operations
   (potentially: `polygeist.memref2pointer`, `polygeist.pointer2memref` and
   `sycl.cast`).

#### Not defining new types for SYCL index space identifiers

Defining new types for these would require additional logic to support
operations taking these types as arguments, e.g., having a `global_id_0` type
representing the first component of a global id (currently `i64` type) would
require adding logic to perform arithmetic operations with this types (including
canonicalization, operating with other `i64` values, etc.). Given the complexity
of this task, we chose not to implement this change and resort to using the
already present SYCL types, like `id`, `range` or `group`.

#### Using reference semantics in operation signatures

In order to represent some member function semantics, e.g., `size_t
&sycl::id::operator[](int)`, reference semantics, i.e., using `memref` for the
`this` operator, are needed. Otherwise, it would not be possible to track
aliasing and it would be harder to represent mutation of SYCL types.
