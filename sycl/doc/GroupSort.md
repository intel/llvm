# Group sort algorithm

Group sorting algorithms are needed to sort data without calling additional kernels
They are described by SYCL 2020 Extension specification:
[direct link to the specification's extension][group_sort_spec].

[group_sort_spec]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/GroupAlgorithms/SYCL_INTEL_group_sort.asciidoc

Example usage:

```cpp
#include <sycl/sycl.hpp>

using namespace sycl;
void do_sort(buffer<int> buf) {
  queue myQueue;

  myQueue.submit([&](handler &cgh) {
    accessor in_acc { in, cgh };

    cgh.parallel_for(
        nd_range(in.get_range(), 256), [=](nd_item item_id) {
            auto idx = item_id.get_global_id();

            in_acc[item_id.get_global_id()] =
                ext::oneapi::sort_over_group(
                    item_id.get_group(),
                    in_acc[item_id.get_global_id()],
                    ext::oneapi::radix_sorter<int, ext::oneapi::radix_order::descending>());
        });
  });

  myQueue.wait();
}
```

## Design objectives

The following should be implemented:

1. Sorter classes and their `operator()` including sorting algorithms.

2. `joint_sort` and `sort_over_group` functions for `sycl::ext::oneapi` namespace.

3. Traits to distinguish interfaces with `Compare` and `Sorter` parameters.

4. Checks when radix sort is applicable (arithmetic types only).

5. The `radix_order` enum class.

6. `default_sorter` and `radix_sorter` for `sycl::ext::oneapi` namespace.

7. Backend support for sorting algorithms.

8. `SYCL_EXT_ONEAPI_GROUP_SORT` feature macro.

Data types that should be supported by backends: arithmetic types
(https://en.cppreference.com/w/c/language/arithmetic_types), `sycl::half`.

Comparators that should be supported by backends: `std::less`, `std::greater`.

## Design

In DPC++ Headers/DPC++ RT we don't know which sorting algorithm is better for
different architectures. Backends have more capability to optimize the sorting algorithm
using low-level instructions.

Overall, we need to have the following:
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

### Fallback SPIR-V library

If backend compilers can generate optimized implementations based on low-level instructions,
we need a function that they can take and optimize.

If there are no implementations in a backend yet,
implementations from the fallback library will be called.

Interface for the library and backends:

```cpp
// for default sorting algorithm
void __devicelib_default_work_group_sort(T* first, T* last,
                                         bool is_ascending_or_descending);

void __devicelib_default_sub_group_sort(T* first, T* last,
                                        bool is_ascending_or_descending);

// for key value sorting using the default algorithm
void __devicelib_default_work_group_sort(T* keys_first, T* keys_last,
                                         U* values_first, U* values_last, bool is_ascending_or_descending);
void __devicelib_default_sub_group_sort(T* keys_first, T* keys_last,
                                        U* values_first, U* values_last, bool is_ascending_or_descending);
```

Notes:
- `T`, `U` are arithmetic types or `sycl::half`.
- `first`, `last` describe the range of actual data for sorting.
- `is_ascending_or_descending` equals `1` when ascending sort is requested, equals `0`
when descending sort is requested.
- `keys_first`, `keys_last` describe the range of "keys" for key-value sorting. "Keys" are comparing
and moving during the sorting.
- `values_first`, `values_last` describe the range of "values" for key-value sorting. "Keys" are
only moving corresponding the "keys" order during the sorting.
- `BitsPerPass` describes how much bits of the value are taken into account
- `radix_iter` describes the iteration for radix sorting.
