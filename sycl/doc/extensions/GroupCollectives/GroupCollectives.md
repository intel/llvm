# SYCL(TM) Proposals: Group Collectives for NDRange Parallelism

**IMPORTANT**: This specification is a draft.

**NOTE**: Khronos(R) is a registered trademark and SYCL(TM) is a trademark of the Khronos Group, Inc. OpenCL(TM) is a trademark of Apple Inc. used by permission by Khronos.

This proposal exposes the work-group functions from OpenCL 2.0 (any, all, broadcast, reductions and scans) to the NDRange variant of `parallel_for`, and does not address hierarchical parallelism.

The new functions are added to the `cl::sycl::group` class, and guarded by the `__SYCL_INTEL_GROUP_COLLECTIVES__` macro.

## Alignment with OpenCL vs C++

Where a feature is common to both OpenCL and C++, this proposal opts for C++-like naming:
- Collective operators are named as in `<functional>` (e.g. `plus` instead of `sum`) and to avoid clashes with names in `<algorithm>` (e.g. `minimum` instead of `min`).
- Scan operations are named as in `<algorithm>` (e.g. `inclusive_scan` instead of `scan_inclusive`).

## Data Types

All functions are supported for the fundamental scalar types supported by SYCL and instances of the SYCL `vec` class.  The fundamental scalar types (as defined in Section 6.5 of the SYCL 1.2.1 specification) are: `bool`, `char`, `signed char`, `unsigned char`, `short int`, `unsigned short int`, `int`, `unsigned int`, `long int`, `unsigned long int`, `long long int`, `unsigned long long int`, `size_t`, `float`, `double`, `half`.

Functions with arguments of type `vec<T,N>` are applied component-wise: they are semantically equivalent to `N` calls to a scalar function of type `T`.

## Function Objects

A number of function objects are provided in the `cl::sycl::intel` namespace that are equivalent to those found in the `<functional>` header from the C++ standard library.  These function objects are used for all interfaces requiring an operator to be specified.

The parameter types and return type for all function objects will be deduced if `T` is not specified.

|Function object|Description|
|----------------|-----------|
|`template <typename T=void> struct plus;`|`T operator(const T&, const T&) const` calls `operator+` on its arguments.|
|`template <typename T=void> struct minimum;`|`T operator(const T&, const T&) const` applies `std::less` to its arguments, in the same order, then returns the lesser argument unchanged.|
|`template <typename T=void> struct maximum;`|`T operator(const T&, const T&) const` applies `std::greater` to its arguments, in the same order, then returns the greater argument unchanged.|

# Functions

The member functions of the `group` class described in this section act as a work-group barrier, and it is undefined behavior for these functions to be invoked within a `parallel_for_work_group` or `parallel_for_work_item` context.

## Vote / Ballot

|Member functions|Description|
|----------------|-----------|
| `bool any(bool predicate) const` | Return `true` if `predicate` evaluates to `true` for any work-item in the work-group.|
| `bool all(bool predicate) const` | Return `true` if `predicate` evaluates to `true` for all work-items in the work-group.|

## Collectives

|Member functions|Description|
|----------------|-----------|
|`template <typename T>T broadcast(T x, id<1> local_id) const` | Broadcast the value of `x` from the work-item with the specified id to all work-items within the work-group. The value of `local_id` must be the same for all work-items in the work-group.|
|`template <typename T, class BinaryOp> T reduce(T x, BinaryOp binary_op) const;`|Combine the values of `x` from all work-items in the work-group using the specified operator, which must be one of `plus`, `minimum` or `maximum`.|
|`template <typename T, class BinaryOp> T reduce(T x, T init, BinaryOp binary_op) const;`|Combine the values of `x` from all work-items in the work-group using an initial value of `init` and the specified operator, which must be one of `plus`, `minimum` or `maximum`.|
|`template <typename T, class BinaryOp> T exclusive_scan(T x, BinaryOp binary_op) const;`|Perform an exclusive scan over the values of `x` from all work-items in the work-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`.  The value returned on work-item `i` is the exclusive scan of the first `i` work items in the work-group and the `init` value. For multi-dimensional work-groups, the order of work-items in the group is determined by their linear id. The initial value is the identity value of the operator.|
|`template <typename T, class BinaryOp> T exclusive_scan(T x, T init, BinaryOp binary_op) const;`|Perform an exclusive scan over the values of `x` from all work-items in the work-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`.  The value returned on work-item `i` is the exclusive scan of the first `i` work items in the work-group and the `init` value. For multi-dimensional work-groups, the order of work-items in the group is determined by their linear id. The initial value is specified by `init`.|
|`template <typename T, class BinaryOp> T inclusive_scan(T x, BinaryOp binary_op) const;`|Perform an inclusive scan over the values of `x` from all work-items in the work-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`.  The value returned on work-item `i` is the inclusive scan of the first `i` work items in the work-group. For multi-dimensional work-groups, the order of work-items in the group is determined by their linear id.|
|`template <typename T, class BinaryOp> T inclusive_scan(T x, BinaryOp binary_op, T init) const;`|Perform an inclusive scan over the values of `x` from all work-items in the work-group using the specified operator, which must be one of: `plus`, `minimum` or `maximum`.  The value returned on work-item `i` is the inclusive scan of the first `i` work items in the work-group and the `init` value. For multi-dimensional work-groups, the order of work-items in the group is determined by their linear id.|
