# Group sort algorithm

Group sorting algorithms are needed to sort data without calling additional kernels
They are described by SYCL 2020 Extension specification:
[direct link to the specification's extension][group_sort_spec].

[group_sort_spec]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_group_sort.asciidoc

Example usage:

```cpp
#include <sycl/sycl.hpp>

namespace oneapi_exp = sycl::ext::oneapi::experimental;
sycl::range<1> local_range{256};
// predefine radix_sorter to calculate local memory size
using RSorter = oneapi_exp::radix_sorter<T, oneapi_exp::sorting_order::descending>;
// calculate required local memory size
size_t temp_memory_size =
    RSorter::memory_required(sycl::memory_scope::work_group, local_range);
q.submit([&](sycl::handler& h) {
  auto acc = sycl::accessor(buf, h);
  auto scratch = sycl::local_accessor<std::byte, 1>( {temp_memory_size}, h);
  h.parallel_for(
    sycl::nd_range<1>{ local_range, local_range },
    [=](sycl::nd_item<1> id) {
      acc[id.get_local_id()] =
        oneapi_exp::sort_over_group(
          id.get_group(),
          acc[id.get_local_id()],
          RSorter(sycl::span{scratch.get_pointer(), temp_memory_size})
      );
    });
  });
...
```

## Design objectives

In DPC++ Headers/DPC++ RT we don't know which sorting algorithm is better for
different architectures. Backends have more capability to optimize the sorting algorithm
using low-level instructions.

Data types that should be supported by backends: arithmetic types
(https://en.cppreference.com/w/c/language/arithmetic_types), `sycl::half`.

Comparators that should be supported by backends: `std::less`, `std::greater`,
custom comparators

## Design

Overall, for backend support we need to have the following:
- Fallback implementation of sorting algorithms for user's types, comparators and/or sorters.

- Backend implementation for types, comparators and/or sorters
  that can be optimized using backend specific instructions.

  **NOTE**: It was decided that `radix_sorter` will be implemented only in DPC++ Headers since
it's difficult to support such algorithm at backends' level.

- Fallback implementation in case if backends don't have more optimized implementations yet.

- Level Zero extension for `memory_required` functions.

The following should be implemented:

- [x] Sorter classes and their `operator()` including sorting algorithms
  - [x] Default sorter.
  - [x] Radix sorter.
- [x] `joint_sort` and `sort_over_group` functions.
- [x] Traits to distinguish interfaces with `Compare` and `Sorter` parameters.
- [x] Checks when radix sort is applicable (arithmetic types only).
- [x] The `radix_order` enum class.
- [x] `group_with_scratchpad` predefined group helper.
- [x] `SYCL_EXT_ONEAPI_GROUP_SORT` feature macro.
- [ ] `sort_over_group` with `span`-based parameters.
- [ ] Level Zero extension for `memory_required` functions
  - [ ] Specification.
  - [ ] Implementation.
- [ ] Backend support for sorting algorithms.
  - [ ] Default sorter
- [ ] Fallback library if device doesn't implement functions.

**Note**: The "tick" means that corresponding feature is implemented.

Sections below describe each component in more details.

### DPC++ Headers

DPC++ Headers contain the following:
- required definitions of `joint_sort`, `sort_over_group` functions, `radix_order` enum class,
  `default_sorter`, `radix_sorter` classes with corresponding `operator()`
  as well as other classes and methods.

- Checks if radix sort is applicable for provided data types.

- Traits to distinguish interfaces with `Compare` and `Sorter` parameters.

- Fallback solution for user's types, user's comparators and/or user's sorters.

### Level Zero

To implement `memory_required` methods for sorters we need to calculate
how much temporary memory is needed.
However, we don't have an information how much memory is needed by backend compiler.
That's why we need a Level Zero function that calls a function from the backend and
provide actual value to the SYCL code.

Required interfaces:
```cpp
    // Returns whether default work-group or sub-group sort is present in builtins
    virtual bool DefaultGroupSortSupported(GroupSortMemoryScope::MemoryScope_t scope,
                                           GroupSortKeyType::KeyType_t keyType,
                                           bool isKeyValue,
                                           bool isJointSort) const;

    // Returns required amount of memory for default joint work-group or sub-group sort
    // devicelib builtin function in bytes per workgroup (or sub-group), >= 0
    // or -1 if the algorithm for the specified parameters is not implemented
    //
    // totalItems -- number of elements to sort
    // rangeSize -- work-group or sub-group size respectively
    //
    // For key-only sort pass valueTypeSizeInBytes = 0
    virtual long DefaultGroupJointSortMemoryRequired(GroupSortMemoryScope::MemoryScope_t scope,
                                                     long totalItems,
                                                     long rangeSize,
                                                     long keyTypeSizeInBytes,
                                                     long valueTypeSizeInBytes) const;

    // Returns required amount of memory for default private memory work-group or sub-group sort
    // devicelib builtin function in bytes per workgroup (or sub-group), >= 0
    // or -1 if the algorithm for the specified parameters is not implemented
    //
    // itemsPerWorkItem -- number of elements in private array to sort
    // rangeSize -- work-group or sub-group size respectively
    //
    // For key-only sort pass valueTypeSizeInBytes = 0
    virtual long DefaultGroupPrivateSortMemoryRequired(GroupSortMemoryScope::MemoryScope_t scope,
                                                       long itemsPerWorkItem,
                                                       long rangeSize,
                                                       long keyTypeSizeInBytes,
                                                       long valueTypeSizeInBytes) const;
```

### Fallback SPIR-V library

If backend compilers can generate optimized implementations based on low-level instructions,
we need a function that they can take and optimize.

If there are no implementations in a backend yet,
implementations from the fallback library will be called.

Interface for the library and backends:

```cpp
// for default sorting algorithm
void __devicelib_default_work_group_joint_sort_ascending_<encoded_param_types>(T* first, uint n, byte* scratch);

void __devicelib_default_work_group_joint_sort_descending_<encoded_param_types>(T* first, uint n, byte* scratch);

// for fixed-size arrays
void __devicelib_default_work_group_private_sort_close_ascending_<encoded_param_types>(T* first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_close_descending_<encoded_param_types>(T* first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_spread_ascending_<encoded_param_types>(T* first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_spread_descending_<encoded_param_types>(T* first, uint n, byte* scratch);

// for sub-groups
T __devicelib_default_sub_group_private_sort_ascending_<encoded_scalar_param_type>(T value);

T __devicelib_default_sub_group_private_sort_descending_<encoded_scalar_param_type>(T value);

// for key value sorting using the default algorithm
void __devicelib_default_work_group_joint_sort_ascending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

void __devicelib_default_work_group_joint_sort_descending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

// for key value sorting using fixed-size arrays
void __devicelib_default_work_group_private_sort_close_ascending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_close_descending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_spread_ascending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

void __devicelib_default_work_group_private_sort_spread_descending_<encoded_param_types>(T* keys_first, U* values_first, uint n, byte* scratch);

```

Notes:
- `T`, `U` are from the following list `i8`, `i16`,
  `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`.
- `encoded_param_types` is `T` prepended with `p1` for global/private address
  space and `p3` for shared local memory.
- `first` is a pointer to the actual data for sorting.
- The type of `n` (number of elements) is u32.
- `keys_first` points to "keys" for key-value sorting.
  "Keys" are comparing and moving during the sorting.
- `scratch` is a temporary storage (local or global) that can be used by backends.
  The type of `scratch` is always `byte*`.
- `values_first` points to "values" for key-value sorting. "Keys" are only moving
  corresponding the "keys" order during the sorting.

Examples:
```cpp
void __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p3i8(int* first, uint n, byte* scratch);
void __devicelib_default_work_group_joint_sort_descending_p1u32_u32_p1i8(uint* first, uint n, byte* scratch);
void __devicelib_default_work_group_joint_sort_ascending_p3u32_p3u32_u32_p1i8(uint* first_keys, uint* first_values, uint n, byte* scratch);
void __devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(uint* first_keys, uint* first_values, uint n, byte* scratch);
double __devicelib_default_sub_group_private_sort_ascending_f64(double value);
```

## Alternative Design

If it's proved that no specific improvements can be done at backends' level (e.g. special
instructions, hardware dispatch) comparing to high-level SYCL code then implementations
of sorting functions can be placed in DPC++ Headers
(no hardware backends, no Level Zero support will be needed in such cases).
