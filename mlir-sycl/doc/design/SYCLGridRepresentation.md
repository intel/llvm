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

## The `sycl.call` operation

The SYCL dialect includes a `sycl.call` operation representing function
calls. As these SYCL index space identifiers are obtained through member
function calls, e.g., `sycl::nd_item::get_global_id()`, `sycl.call` is the first
candidate that comes to mind for this task. However, this operation is too
low-level to do the work for us. If we look at this example:

```MLIR
%0 = sycl.call() {FunctionName = @get_global_id, MangledFunctionName = @..., TypeName = @nd_item} : (!memref<?xsycl_nd_item_1_>) -> !sycl_id_1_
%1 = sycl.call() {FunctionName = @get_local_id, MangledFunctionName = @..., TypeName = @nd_item} : (!memref<?xsycl_nd_item_1_>) -> !sycl_id_1_
```

We can see how, even though `%0` and `%1` have a very different nature, they
have the same type and defining operation, in fact, they only differ on the
`FunctionName` (and `MangledFunctionName`) attribute of their defining
operations. As the target is to make memory access analysis easier, we propose
index space identifiers of different nature should differ at least on their
defining operation.

Also, in order to perform memory access analysis, these operations should be
presented in a canonical form. A canonicalization pattern checking every
`sycl.call` operation in the MLIR module and replacing it with different
`sycl.call` operations can be a bit cumbersome, thus we propose these operations
to have different defining operations.

## `SYCLMethodOpInterface`

As these operations are really similar in terms of structure, the
`SYCLMethodOpInterface` operation interface is defined as the interface
describing an operation representing a member function call of a SYCL type. This
interface provides the following functions:

| Function                                                       | Description                                                            |
|:---------------------------------------------------------------|:-----------------------------------------------------------------------|
| `static mlir::TypeID getTypeID();`                             | Return the ID of the type this method is a member of.                  |
| `static llvm::ArrayRef<llvm::StringLiteral> getMethodNames();` | Return the list of the method names to be replaced by this operation.  |
| `llvm::ArrayRef<mlir::Type> getFunctionArgTypes();`            | Return the argument types of the function implementing this operation. |
| `llvm::StringRef getFunctionName();`                           | Return the name of the function implementing this operation.           |
| `llvm::StringRef getTypeName();`                               | Return the original name of the type this method is implemented in.    |

As we can see, the attributes `FunctionName` and `TypeName` are analogous to the
attributes present in the `sycl.call` operation, as this interface also
represents function calls. On the other hand, the `FunctionArgTypes` attribute
contains the argument types of the member function to call. E.g., in an
implementation in which `nd_item` was derived from a base class and the generic
SPIR-V address space was to be used for pointer arguments (an implementation
different from the one corresponding to the example above), a function call to
`nd_item::get_global_id()` would have `{!llvm.ptr<struct<...>, 4>}` as
`FunctionArgTypes`, `@get_global_id` as `FunctionName` and `@nd_item` as
`TypeName`.

The non-member functions `getTypeID` and `getMethodNames` return respectively
the ID of the type defined in the SYCL specification of which the function is a
member of and a set of names which must contain `FunctionName`. Note that
`getTypeID` does not refer to the base type present in the implementation from
which the SYCL type is derived, but to the type present in the SYCL
dialect. This decision removes the need of updating operations definitions if
implementation details change or a different implementation is to be supported.

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

In the codegen side, right before generating a `sycl.call` operation, we check
whether there is an operation implementing `SYCLMethodOpInterface` homologous to
that function call. In order to do that, `SYCLDialect::findMethod(mlir::TypeID,
llvm::StringRef)` can be used, passing the ID of the SYCL type the function is a
member of and the function name. After looking up the name in its registry,
`llvm::None` is returned if no operation can be used to replace the `sycl.call`;
otherwise, the name of the operation is returned, which can be used to create a
different operation instead.

Note that, in case the function is a member of a base type, the `sycl.cast`
operation used to cast to the base class is abstracted beforehand.

### Canonicalization

To help the analysis, canonicalization rules are introduced in such a way that
the total number of operations is minimized, i.e., we reduce the number of
operations by replacing combinations of operations by equivalent
single-operations, e.g., the following code:

```MLIR
%0 = sycl.nd_item.get_global_id(%nd) {FunctionArgTypes = {!llvm.ptr<struct<...>, 4>}, FunctionName = @get_global_id, TypeName = @nd_item} : (!sycl_nd_item_1_) -> !sycl_id_1_
%1 = sycl.id.get(%0, %idx) {FunctionArgTypes = {memref<?x!sycl_id_1_, 4>, i32}, FunctionName = @get, TypeName = @id} : (!sycl_id_1_, i32) -> i64
```

would be simplified to:

```MLIR
%1 = sycl.nd_item.get_global_id(%nd, %idx) {FunctionArgTypes = {!llvm.ptr<struct<...>, 4>, i32}, FunctionName = @get_global_id, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64
```

### Function definition register

After the canonicalization pass, we might obtain a module in which a new
operation implementing `SYCLMethodOpInterface` is introduced, but no
corresponding definition is present in the module. To avoid this situation, the
SYCL dialect provides a function definition register which corresponds to a map
`(FunctionName, FunctionType) -> Definition`. Definitions are inserted using
`SYCLDialect::addMethodDefinition(llvm::StringRef, mlir::func::FuncOp)` and
retrieved using `SYCLDialect::lookupMethodDefinition(llvm::StringRef,
mlir::FunctionType)`.

In the codegen side, when code is generated for a function in the `sycl`
namespace (an `mlir::func::FuncOp` is obtained), if this corresponds to an
operation, it is inserted in the SYCL dialect. After this insertions takes
place, only declarations can be overriden by functions of the same name.

### Lowering

All operations implementing `SYCLMethodOpInterface` are lowered to a `sycl.call`
operation. In order to do so, the corresponding attributes of the operation are
used to construct the `sycl.call`. Regarding the `MangledFunctionName`
attribute, this was not needed in the original operation as the lowering process
makes use of the definition register to obtain the function implementing the
operation, insert it in the module if it is not present yet, and use the name of
this function to build the attribute.

In order to create a legal function call, the `FunctionArgTypes` attribute is
used to perform the necessary transformations beforehand:

1. Build a `memref.alloca()` of type `memref<1xTy>`, being `Ty` the type of the
   first argument;
2. Store the value there;
3. If the argument type is a `memref` and the shapes differ, introduce a
   `memref.cast` operation;
4. If the memory spaces differ, use a `memref.memory_space_cast` to cast
   to the required one;
5. If needed, cast to the base type using `sycl.cast` (or other operation we
   might have in the future).

In case the operation to be lowered does not have a definition in the register,
it will be "decanonicalized", following the inverse process to the 
canonicalization process explained above and each operation being generated will
be lowered following the same process.

If an operation does not count with a definition in the register and cannot be
split into further operations, the lowering process will fail, as the user would
have failed to provide a definition for an operation.

#### Lowering example

The following `sycl.nd_item.get_global_id` operation:

```MLIR
%1 = sycl.nd_item.get_global_id(%nd, %idx) {FunctionArgTypes = {!llvm.ptr<struct<...>, 4>}, FunctionName = @get_global_id, TypeName = @nd_item} : (!sycl_nd_item_1_, i32) -> i64
```

Would be lowered to:

```MLIR
%1 = memref.alloca() : memref<1x!sycl_nd_item_1_>
memref.store %nd, %1[0] : memref<1x!sycl_nd_item_1_>
%2 = memref.memory_space_cast %1 : memref<1x!sycl_nd_item_1_> to memref<1x!sycl_nd_item_1_, 4>
%3 = sycl.cast(%2) : (memref<1x!sycl_nd_item_1_, 4>) -> !llvm.ptr<struct<...>, 4>
%4 = sycl.call(%3, %idx) {FunctionName = @get_global_id, MangledFunctionName = <MFN>, TypeName = @nd_item} : (!llvm.ptr<struct<...>, 4>, i32) -> i64
```

Being `<MFN>` the name of the function to call, retrieved by calling
`SYCLDialect::lookupMethodDefinition`.

### Verification

For all operations implementing this interface, the `FunctionName` attribute
must be a member of the list returned by `getMethodNames()`.

In addition to this generic verification, four different traits are introduced:
`SYCLGetID`, `SYCLGetComponent`, `SYCLGetRange`, `SYCLGetGroup`:

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

#### Dropping `MangledFunctionName`

The `MangledFunctionName` attribute is not used in these operations as it is no
longer needed. Canonicalization adds the requirement of a definition register,
so the mangled function name can be retrieved from this at the only point in
which it is needed: lowering.

Thus, we further separate the SYCL dialect from the underlying SYCL
implementation, moving a implementation-specific detail to the lowering, where
it belongs.

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
