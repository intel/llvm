# Group sort algorithm

Group sorting algorithms are needed to sort data without calling additional kernels
They are described by SYCL 2020 Extension specification:
[direct link to the specification's extension][group_sort_spec].

[group_sort_spec]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/GroupAlgorithms/SYCL_INTEL_group_sort.asciidoc

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

The following should be implemented:

1. Sorter classes and their `operator()` including sorting algorithms.

2. `joint_sort` and `sort_over_group` functions.

3. Traits to distinguish interfaces with `Compare` and `Sorter` parameters.

4. Checks when radix sort is applicable (arithmetic types only).

5. The `radix_order` enum class.

6. `default_sorter` and `radix_sorter`.

7. `group_with_scratchpad` predefined group helper.

8. Backend support for sorting algorithms.

9. `SYCL_EXT_ONEAPI_GROUP_SORT` feature macro.

Data types that should be supported by backends: arithmetic types
(https://en.cppreference.com/w/c/language/arithmetic_types), `sycl::half`.

Comparators that should be supported by backends: `std::less`, `std::greater`,
custom comparators

## Design

Overall, for backend support we need to have the following:
- Fallback implementation of sorting algorithms for user's types, comparators and/or sorters.

- Backend implementation for types, comparators and/or sorters
that can be optimized using backend specific instructions.

- Fallback implementation in case if backends don't have more optimized implementations yet.

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

To implement `memory_reuired` methods for sorters we need to calculate
how much temporary memory is needed.
However, we don't have an information how much memory is needed by backend compiler.
That's why we need a Level Zero function that calls a function from the backend and
provide actual value to the SYCL code.

### Fallback SPIR-V library

If backend compilers can generate optimized implementations based on low-level instructions,
we need a function that they can take and optimize.

If there are no implementations in a backend yet,
implementations from the fallback library will be called.

Interface for the library and backends:

```cpp
// for default sorting algorithm
void __devicelib_default_work_group_sort_ascending(T* first, T* last, T* temp_memory);

void __devicelib_default_work_group_sort_descending(T* first, T* last, T* temp_memory);

void __devicelib_default_sub_group_sort_ascending(T* first, T* last, T* temp_memory);

void __devicelib_default_sub_group_sort_descending(T* first, T* last, T* temp_memory);

// for key value sorting using the default algorithm
void __devicelib_default_work_group_sort_ascending(T* keys_first, T* keys_last,
                                                   U* values_first, U* values_last,
                                                   T* temp_memory);

void __devicelib_default_work_group_sort_descending(T* keys_first, T* keys_last,
                                                    U* values_first, U* values_last,
                                                    T* temp_memory);

void __devicelib_default_sub_group_sort_ascending(T* keys_first, T* keys_last,
                                                  U* values_first, U* values_last,
                                                  T* temp_memory);

void __devicelib_default_sub_group_sort_descending(T* keys_first, T* keys_last,
                                                   U* values_first, U* values_last,
                                                   T* temp_memory);
```

Notes:
- `T`, `U` are arithmetic types or `sycl::half`.
- `first`, `last` describe the range of actual data for sorting.
- `keys_first`, `keys_last` describe the range of "keys" for key-value sorting. "Keys" are comparing
and moving during the sorting.
- `temp_memory` is a temporary storage (local or global) that can be used by backends
- `values_first`, `values_last` describe the range of "values" for key-value sorting. "Keys" are
only moving corresponding the "keys" order during the sorting.
