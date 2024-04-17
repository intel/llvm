# ESIMD methods and functions

This document describes ESIMD methods and functions, their semantics,
restrictions and hardware dependencies.
See more general ESIMD documentation [here](./sycl_ext_intel_esimd.md).

## Table of contents
- [Compile-time properties](#compile-time-properties)
  - [Cache-hint properties and restrictions depending on the usage context](#cache-hint-properties)
- [Stateless/stateful memory mode](#statelessstateful-memory-mode)
- [block_load(...) - fast load from a contiguous memory block](#block_load---fast-load-from-a-contiguous-memory-block)
- [block_store(...) - fast store to a contiguous memory block](#block-store---fast-store-to-a-contiguous-memory-block)
- [gather(...)](#gather---load-from-memory-locations-addressed-by-a-vector-of-offsets)
- [scatter(...)](#scatter---store-to-memory-locations-addressed-by-a-vector-of-offsets)
- [load_2d(...) - load 2D block](#load_2d---load-2d-block)
- [prefetch_2d(...) - prefetch 2D block](#prefetch_2d---prefetch-2d-block)
- [store_2d(...) - store 2D block](#store_2d---store-2d-block)
- [atomic_update(...)](#atomic_update)
- [prefetch(...)](#prefetch)
- [fence(...) - set the memory read/write order](#fence---set-the-memory-readwrite-order)
- [Examples](#examples)

## Other content:
* [General ESIMD documentation](./sycl_ext_intel_esimd.md)
* [ESIMD API/doxygen reference](https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd.html)
* [Examples](./examples/README.md)
* [ESIMD LIT tests - working code examples](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/ESIMD/)

---
## Stateless/stateful memory mode
ESIMD functions have two memory assumption modes: `stateful` and `stateless`.
`Stateless` read/write/prefetch uses a pointer to global memory,
which also may be adjusted by a scalar/vector 64-bit offset.  
`Stateful` read/write/prefetch accesses memory using a
`surface-index` and a `32-bit` scalar/vector offset.

The `-fsycl-esimd-force-stateless-mem` compilation option (which is ON by default)
forces the translation of ESIMD memory API functions to `stateless` accesses.
In this mode the ESIMD functions that accept a byte-offset argument accept it as
any integral type scalar/vector.  
The `-fno-sycl-esimd-force-stateless-mem` compilation option may be used to translate
ESIMD functions accepting a `SYCL device accessor` to `stateful` accesses. In this case
the corresponding ESIMD functions accept only 32-bit scalar/vector byte offsets.


## Compile time properties
```C++
namespace sycl::ext::intel::esimd {
template <typename PropertiesT> class properties;
template <int K> inline constexpr alignment_key::value_t<K> alignment;
template <cache_hint Hint> inline constexpr cache_hint_L1_key::value_t<Hint> cache_hint_L1;
template <cache_hint Hint> inline constexpr cache_hint_L2_key::value_t<Hint> cache_hint_L2;
template <cache_hint Hint> inline constexpr cache_hint_L3_key::value_t<Hint> cache_hint_L3;
}
```
Many ESIMD functions have an optional argument `alignment`, `L1-cache-hint`, `L2-cache-hint`,
`L3-cache-hint`(reserved). The list may be extended in the future. The properties may be added
to the properties list in any order. See an example of using the properties below:
```C++
using namespace sycl::ext::intel::esimd;

// f32_ptr is aligned by 16-bytes, no cache-hints passed.
auto vec_a = block_load<float, 16>(f32_ptr, properties{alignment<16>});

// (f32_ptr + 1) is aligned by 4-bytes only, L1=uncached, L2=cached.
properties props{cache_hint_L1<cache_hint::uncached>, alignment<4> cache_hint_L1<cache_hint::cached>};
auto vec_b = block_load<float, 16>(f32_ptr + 1, props);
```
### Cache-hint properties
Cache-hint properties (if passed) currently add a restriction on the target-device, it must be a Intel® Arc Series (aka DG2) or Intel® Data Center GPU Max Series (aka PVC).  
The valid combinations of L1/L2 cache-hints depend on the usage context.. There are 4 contexts:
* load: `block_load()`, `load_2d()`, `gather()` functions;
* prefetch: `prefetch()` and `prefetch_2d()` functions;
* store: `block_store()`, `store_2d()`, `scatter()` functions;
* atomic_update: `atomic_update()` functions.

#### Valid combinations of `L1` and `L2` cache-hints for `load` functions:
| `L1` | `L2` |
|-|-|
| none | none |
| uncached | uncached |
| uncached | cached |
| cached | uncached |
| cached | cached |
| streaming | uncached |
| streaming | cached |
| read_invalidate | cached |

#### Valid combinations of `L1` and `L2` cache-hints for `prefetch` functions:
| `L1` | `L2` |
|-|-|
| uncached | cached |
| cached | uncached |
| cached | cached |
| streaming | uncached |
| streaming | cached |

#### Valid combinations of `L1` and `L2` cache-hints for `store` functions:
| `L1` | `L2` |
|-|-|
| none | none |
| uncached | uncached |
| uncached | write_back |
| write_through | uncached |
| write_through | write_back |
| streaming | uncached |
| streaming | write_back |
| write_back | write_back |

#### Valid combinations of `L1` and `L2` cache-hints for `atomic_update` functions:
| `L1` | `L2` |
|-|-|
| none | none |
| uncached | uncached |
| uncached | write_back |


## block_load(...) - fast load from a contiguous memory block
```C++
// block load from USM memory.
namespace sycl::ext::intel::esimd {
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*usm-bl-1*/ simd<T, N> block_load(const T* ptr, PropertyListT props={});
/*usm-bl-2*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, PropertyListT props={});
/*usm-bl-3*/ simd<T, N> block_load(const T* ptr, simd_mask<1> pred, PropertyListT props={});
/*usm-bl-4*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred, PropertyListT props={});
/*usm-bl-5*/ simd<T, N> block_load(const T* ptr, simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props={});
/*usm-bl-6*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props={});

// block load from device accessor. OffsetT is uint64_t in `stateless` mode(default), and uint32_t in `stateful`
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*acc-bl-1*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, props = {});
/*acc-bl-2*/ simd<T, N> block_load(AccessorT acc, props = {});
/*acc-bl-3*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, props = {});
/*acc-bl-4*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred, props = {});
/*acc-bl-5*/ simd<T, N> block_load(AccessorT acc, simd_mask<1> pred, simd<T, N> pass_thru, props = {});
/*acc-bl-6*/ simd<T, N> block_load(AccessorT acc, simd_mask<1> pred, props = {});

// block load from local accessor (SLM).
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-bl-1*/ simd<T, N> block_load(AccessorT lacc, uint32_t byte_offset, PropertyListT props={});
/*lacc-bl-2*/ simd<T, N> block_load(AccessorT lacc, PropertyListT props={});
/*lacc-bl-3*/ simd<T, N> block_load(AccessorT lacc, uint32_t byte_offset, simd_mask<1> pred, PropertyListT props={});
/*lacc-bl-4*/ simd<T, N> block_load(AccessorT lacc, simd_mask<1> pred, PropertyListT props={});
/*lacc-bl-5*/ simd<T, N> block_load(AccessorT lacc, uint32_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props={});
/*lacc-bl-6*/ simd<T, N> block_load(AccessorT lacc, simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props={});

// block load from SLM (Shared Local Memory).
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*slm-bl-1*/ simd<T, N> slm_block_load(uint32_t byte_offset, PropertyListT props={});
/*slm-bl-2*/ simd<T, N> slm_block_load(uint32_t byte_offset, simd_mask<1> pred, PropertyListT props={});
/*slm-bl-3*/ simd<T, N> slm_block_load(uint32_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props={});
} // end namespace sycl::ext::intel::esimd
```
### Description
`(usm-bl-*)`: Loads a contiguous memory block from global memory referenced by the USM pointer `ptr` optionally adjusted by `byte_offset`.  
`(acc-bl-*)`, `(lacc-bl-*)`: Loads a contiguous memory block from the memory referenced by the accessor optionally adjusted by `byte_offset`.  
`(slm-bl-*)`: Loads a contiguous memory block from the shared local memory referenced by `byte_offset`.  
The optional parameter `byte_offset` has a scalar integer 64-bit type for `(usm-bl-*)`, 32-bit type for `(lacc-bl-*)` and `(slm-bl-*)`, 32-bit for `(acc-bl-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit for `(acc-bl-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a 1-element `simd_mask`. If zero mask is passed, then the load is skipped and the `pass_thru` value is returned.  
If `pred` is zero and the `pass_thru` operand was not passed, then the function returns an undefined value.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-bl-*)` and `(slm-bl-*)` functions.

### Restrictions/assumptions:
`Alignment` - if not specified by the `props` param, then `assumed` alignment is used. If the actual memory reference has a smaller alignment than the `assumed`, then it must be explicitly passed in `props` argument.

`Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-load-functions) for `load` functions.

| `Function` | `Assumed` alignment   | `Minimally required` alignment |
|-|-|-|
| `(usm-bl-*)`  | `max(4, sizeof(T))` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |
| `(acc-bl-*)`  | `max(4, sizeof(T))` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |
| `(lacc-bl-*)`, `(slm-bl-*)` | `16` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |

`N` - the valid values may depend on usage of cache-hints or passing of the `pred` argument:

| `Function` | `Condition` | Requirement for `N` | Required Intel GPU |
|-|-|-|-|
| `(usm-bl-*)` | (no cache-hints) and (`pred` is not passed) | `N` is any positive number | Any Intel GPU |
| `(usm-bl-*)` | (cache-hints) or (`pred` is passed) | `N` must be from [Table1 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(acc-bl-*)` | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-bl-*)` - Read the corresponding `(usm-*)` line above | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above |
| `(acc-bl-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and (no cache-hints) and (`pred` is not passed) and (sizeof(`T`) * `N` == 16,32,64,128) | sizeof(`T`)*`N` == 16,32,64,128 | Any Intel GPU |
| `(acc-bl-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and ((cache-hints) or (`pred` is passed)) or (sizeof(`T`) * `N` != 16,32,64,128) | `N` must be from [Table1 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(lacc-bl-1,2)`, `(slm-bl-1)` | `pred` is not passed | `N` is any positive number | Any Intel GPU |
| `(lacc-bl-3,4,5,6)`, `(slm-bl-2,3)`  | `pred` is passed | `N` must be from [Table1 below] | Any Intel GPU |


#### Table1 - Valid values of `N` if cache-hints are used or `pred` parameter is passed:
| sizeof(`T`) | Valid values of `N` | Special case - PVC only - the maximal `N`: requires bigger alignment: 8 or more |
|---|--------------------------------|-----|
| 1 | 4, 8, 12, 16, 32, 64, 128, 256 | 512 |
| 2 | 2, 4, 6, 8, 16, 32, 64, 128 | 256 |
| 4 | 1, 2, 3, 4, 8, 16, 32, 64 | 128 |
| 8 | 1, 2, 3, 4, 8, 16, 32 | 64 |

## block store(...) - fast store to a contiguous memory block
```C++
namespace sycl::ext::intel::esimd {
// block store to USM memory.
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*usm-bs-1*/ void block_store(T* ptr, simd<T, N> vals, PropertyListT props={});
/*usm-bs-2*/ void block_store(T* ptr, size_t byte_offset, simd<T, N> vals, PropertyListT props={});
/*usm-bs-3*/ void block_store(T* ptr, simd<T, N> vals, simd_mask<1> pred, PropertyListT props={});
/*usm-bs-4*/ void block_store(T* ptr, size_t byte_offset, simd<T, N> vals, simd_mask<1> pred, PropertyListT props={});

// block store to device accessor. OffsetT is uint64_t in `stateless` mode(default), and uint32_t in `stateful`
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*acc-bs-1*/ void block_store(AccessorT acc, OffsetT byte_offset, simd<T, N> vals, PropertyListT props = {});
/*acc-bs-2*/ void block_store(AccessorT acc, simd<T, N> vals, PropertyListT props = {});
/*acc-bs-3*/ void block_store(AccessorT acc, OffsetT byte_offset, simd<T, N> vals, simd_mask<1> pred, PropertyListT props = {});
/*acc-bs-4*/ void block_store(AccessorT acc, simd<T, N> vals, simd_mask<1> pred, PropertyListT props = {});

// block store to local accessor (SLM).
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-bs-1*/ void block_store(AccessorT lacc, uint32_t byte_offset, simd<T, N> vals, PropertyListT props={});
/*lacc-bs-2*/ void block_store(AccessorT lacc, simd<T, N> vals, PropertyListT props={});
/*lacc-bs-3*/ void block_store(AccessorT lacc, uint32_t byte_offset, simd<T, N> vals, simd_mask<1> pred, PropertyListT props={});
/*lacc-bs-4*/ void block_store(AccessorT lacc, simd<T, N> vals, PropertyListT props={});
void block_store(AccessorT lacc, simd<T, N> vals, simd_mask<1> pred, PropertyListT props={});

// block store to SLM (Shared Local Memory).
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*slm-bs-1*/ void slm_block_store(uint32_t byte_offset, simd<T, N> vals, simd_mask<1> pred, PropertyListT props={});
/*slm-bs-2*/ void slm_block_store(uint32_t byte_offset, simd<T, N> vals, PropertyListT props={});
} // end namespace sycl::ext::intel::esimd
```
### Description
`(usm-bs-*)`: Stores `vals` to a contiguous global memory block referenced by the USM pointer `ptr` optionally adjusted by `byte_offset`.  
`(acc-bs-*)`, `(lacc-bs-*)`: Stores `vals` to a contiguous memory block referenced by the accessor optionally adjusted by `byte_offset`.  
`(slm-bs-*)`: Stores `vals` to a contiguous shared-local-memory block referenced by `byte_offset`.  
The optional parameter `byte_offset` has a scalar integer 64-bit type for `(usm-bs-*)`, 32-bit type for `(lacc-bs-*)` and `(slm-bs-*)`, 32-bit for `(acc-bs-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit for `(acc-bs-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a 1-element `simd_mask`. If zero mask is passed, then the store is skipped.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-bs-*)` and `(slm-bs-*)` functions.

### Restrictions/assumptions:
`Alignment` - if not specified by the `props` param, then `assumed` alignment is used. If the actual memory reference requires a smaller alignment than the `assumed`, then it must be explicitly passed in `props` argument.

`Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-store-functions) for `store` functions.

| `Function` | Condition | `Assumed` alignment   | `Minimally required` alignment |
|-|-|-|-|
| `(usm-bs-*)`  | (no cache-hints) and (`pred` is not passed). | `16` | `sizeof(T))` |
| `(usm-bs-*)`  | (cache-hints) or (`pred` is passed). | `max(4, sizeof(T))` | `max(4, sizeof(T))` |
| `(acc-bs-*)`  | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-bs-*)` - Read the corresponding `(usm-bs-*)` line above | Lowered to `(usm-bs-*)` - Read the corresponding `(usm-bs-*)` line above |
| `(acc-bs-*)`  | [Stateful](#statelessstateful-memory-mode) memory mode and (no cache-hints) and (`pred` is not passed) and (`sizeof(T) * N` == {16,32,64,128}) | `16` | `max(4, sizeof(T))` |
| `(acc-bs-*)`  | [Stateful](#statelessstateful-memory-mode) memory mode and ((cache-hints) or (`pred` is passed) or (`sizeof(T) * N` != {16,32,64,128})) | `max(4, sizeof(T))` | `max(4, sizeof(T))` |
| `(lacc-bs-1,2)`, `(slm-bs-2)` | `pred` is not passed | `16` | `sizeof(T)` |
| `(lacc-bs-3,4)`, `(slm-bs-1)` | `pred` is passed  | `max(4, sizeof(T))` | `max(4, sizeof(T))` |

`N` - the valid values may depend on usage of cache-hints or passing of the `pred` argument:

| `Function` | `Condition` | Requirement for `N` | Required Intel GPU |
|-|-|-|-|
| `(usm-bs-*)` | (no cache-hints) and (`pred` is not passed) | `N` is any positive number | Any Intel GPU |
| `(usm-bs-*)` | (cache-hints) or (`pred` is passed) | `N` must be from [Table2 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(acc-bs-*)` | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-bs-*)` - Read the corresponding `(usm-bs-*)` line above | Lowered to `(usm-bs-*)` - Read the corresponding `(usm-bs-*)` line above |
| `(acc-bs-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and (no cache-hints) and (`pred` is not passed) and (sizeof(`T`) * `N` == 16,32,64,128) | sizeof(`T`)*`N` == 16,32,64,128 | Any Intel GPU |
| `(acc-bs-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and ((cache-hints) or (`pred` is passed)) or (sizeof(`T`) * `N` != 16,32,64,128) | `N` must be from [Table2 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(lacc-bs-1,2)`, `(slm-bs-2)`  | `pred` is not passed | `N` is any positive number | Any Intel GPU |
| `(lacc-bs-3,4)`, `(slm-bs-1)`  | `pred` is passed | `N` must be from [Table2 below] | DG2 or PVC |


#### Table2 - Valid values of `N` if cache-hints used or `pred` parameter is passed:
| sizeof(`T`) | Valid values of `N` | Special case - PVC only - the maximal `N`: requires bigger alignment: 8 or more |
|---|--------------------------------|-----|
| 1 | 4, 8, 12, 16, 32, 64, 128, 256 | 512 |
| 2 | 2, 4, 6, 8, 16, 32, 64, 128 | 256 |
| 4 | 1, 2, 3, 4, 8, 16, 32, 64 | 128 |
| 8 | 1, 2, 3, 4, 8, 16, 32 | 64 |

## gather(...) - load from memory locations addressed by a vector of offsets
```C++
namespace sycl::ext::intel::esimd {
// gather from USM memory - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-ga-1*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                               simd<T, N> pass_thru, PropertyListT props = {});
/*usm-ga-2*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                               PropertyListT props = {});
/*usm-ga-3*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
                               PropertyListT props = {});

// gather from USM - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-ga-4*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                               PropertyListT props = {});
/*usm-ga-5*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
                               PropertyListT props = {});
/*usm-ga-6*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
                               PropertyListT props = {});

// gather from USM - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1,
          typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*usm-ga-7*/ simd <T, N> gather(const T *p, OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*usm-ga-8*/ simd <T, N> gather(const T *p, OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-ga-9*/ simd <T, N> gather(const T *p, OffsetSimdViewT byte_offsets,
                                PropertyListT props = {});


// gather from memory accessed via device-accessor - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-ga-1*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                               simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*acc-ga-2*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                               simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-ga-3*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                               PropertyListT props = {});

// gather from memory accessed via device-accessor - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-ga-4*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                               simd_mask<N> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*acc-ga-5*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                               simd_mask<N> mask, PropertyListT props = {});
/*acc-ga-6*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                               PropertyListT props = {});

// gather from memory accessed via device-accessor - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename AccessorT,
          typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*acc-ga-7*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*acc-ga-8*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-ga-9*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                PropertyListT props = {});


// gather from memory accessed via local-accessor/SLM - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-ga-1*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                                simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*lacc-ga-2*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                                simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-ga-3*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                                PropertyListT props = {});

// gather from memory accessed via local-accessor/SLM - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-ga-4*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                                PropertyListT props = {});
/*lacc-ga-5*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
                                PropertyListT props = {});
/*lacc-6*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                             PropertyListT props = {});

// gather from memory accessed via local-accessor/SLM - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*lacc-ga-7*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                 simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*lacc-ga-8*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                 simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-ga-9*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                                 PropertyListT props = {});


// gather from shared local memmory (SLM) - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename PropertyListT = empty_properties_t>
/*slm-ga-1*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets,
                               simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*slm-ga-2*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets,
                               simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-ga-3*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets,
                               PropertyListT props = {});

// gather from shared local memory (SLM) - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*slm-ga-4*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets,
                               simd_mask<N> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*slm-ga-5*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets,
                               simd_mask<N> mask, PropertyListT props = {});
/*slm-ga-6*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets,
                               PropertyListT props = {});

// gather from shared local memory (SLM) - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*slm-ga-7*/ simd <T, N> gather(OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*slm-ga-8*/ simd <T, N> gather(OffsetSimdViewT byte_offsets,
                                simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-ga-9*/ simd <T, N> gather(OffsetSimdViewT byte_offsets,
                                PropertyListT props = {});
} // end namespace sycl::ext::intel::esimd
```

### Description
`(usm-ga-*)`: Loads ("gathers") elements of the type `T` from global memory locations addressed by the base USM pointer `p` and byte-offsets `byte_offsets`.  
`(acc-ga-*)`, `(lacc-ga-*)`: Loads ("gathers") elements of the type `T` from memory locations addressed by the accessor and byte-offsets `byte_offsets`.  
`(slm-ga-*)`: Loads ("gathers") elements of the type `T` from shared local memory locations addressed by `byte_offsets`.  
The parameter `byte_offset` is a vector of any integral type elements for `(usm-ga-*)`, 32-bit integer elements for `(lacc-ga-*)` and `(slm-ga-*)`, any integral type integer elements for `(acc-ga-*)` in [stateless](#statelessstateful-memory-mode) mode(default),
and up-to-32-bit integer elements for `(acc-ga-*)` in [stateful](#statelessstateful-memory-mode) mode.  
The optional parameter `mask` provides a `simd_mask`. If some element in `mask` is zero, then the load of the corresponding memory location is skipped and the element of the result is copied from `pass_thru` (if it is passed) or it is undefined (if `pass_thru` is omitted).  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-*)` and `(slm-*)` functions.  
The template parameter `N` can be any positive number.  
The optional template parameter `VS` must be one of `{1, 2, 3, 4, 8, 16, 32, 64}` values. It specifies how many conseсutive elements are loaded per each element in `byte_offsets`.   
### Example
```C++
simd<int64_t, 4> offsets(0, 100); // 0, 100, 200, 300 - offsets in bytes
// loads and returns a vector {ptr[0], ptr[100/4], ptr[200/4], ptr[300/4]};
simd<float, 4> vec4 = gather<float, 4>(ptr, offsets);

// VS = 2, loads and returns a vector {ptr[0], ptr[100/4],     ptr[200/4],     ptr[300/4],
//                                     ptr[1], ptr[100/4 + 1], ptr[200/4 + 1], ptr[300/4 + 1]};
simd<float, 8> vec8 = gather<float, 8, 2>(ptr, offsets);
```

### Restrictions

`Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-load-functions) for `load` functions.

| `Function` | `Condition` | Required Intel GPU |
|-|-|-|
| `(usm-ga-1,4,7)`,`(acc-ga-1,4,7)` | true (`pass_thru` arg is passed) | DG2 or PVC |
| `(usm-ga-2,3,8,9)`, `(acc-ga-2,3,8,9)` | !(cache-hints) and (`VS` == 1) and (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(usm-ga-2,3,8,9)`, `(acc-ga-2,3,8,9)` | (cache-hints) or (`VS` > 1) or (`N` != 1,2,4,8,16,32) | DG2 or PVC |
| `(usm-ga-5,6)`, `(acc-ga-5,6)` | !(cache-hints) and (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(usm-ga-5,6)`, `(acc-ga-5,6)` | (cache-hints) or (`N` != 1,2,4,8,16,32) | DG2 or PVC |
| The next 5 lines are similar to the previous 5 lines. They are for SLM gather and the only difference is that SLM gathers ignore cache-hints|||
| `(slm-ga-1,4,7)`,`(lacc-ga-1,4,7)` | true (`pass_thru` is passed) | DG2 or PVC |
| `(slm-ga-2,3,8,9)`, `(lacc-ga-2,3,8,9)` | (`VS` == 1) and (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(slm-ga-2,3,8,9)`, `(lacc-ga-2,3,8,9)` | (`VS` > 1) or (`N` != 1,2,4,8,16,32) | DG2 or PVC |
| `(slm-ga-5,6)`, `(lacc-ga-5,6)` | (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(slm-ga-5,6)`, `(lacc-ga-5,6)` | (`N` != 1,2,4,8,16,32) | DG2 or PVC |



## scatter(...) - store to memory locations addressed by a vector of offsets
```C++
namespace sycl::ext::intel::esimd {
// scatter to USM memory.
template <typename T, int N, int VS = 1, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-sc-1*/ void scatter(T *p, simd<OffsetT, N / VS> byte_offsets,
                          simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-sc-2*/ void scatter(T *p, simd<OffsetT, N / VS> byte_offsets,
                          simd<T, N> vals, PropertyListT props = {});

// scatter to USM memory - similar to (usm-sc-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*usm-sc-3*/ void scatter(T *p, OffsetSimdViewT byte_offsets,
                          simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-sc-4*/ void scatter(T *p, OffsetSimdViewT byte_offsets,
                          simd<T, N> vals, PropertyListT props = {});


// scatter to memory accessed via device-accessor.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-sc-1*/ void scatter(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                          simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-sc-2*/ void scatter(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                          simd<T, N> vals, PropertyListT props = {});

// scatter to memory accessed via device-accessor - similar to (acc-sc-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*acc-sc-3*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets,
                          simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-sc-4*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
                          PropertyListT props = {});


// scatter to shared local memory accessed via local-accessor.
template <typename T, int N, int VS = 1, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-sc-1*/ void scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
                           simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-sc-2*/ void scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
                           simd<T, N> vals, PropertyListT props = {});

// scatter to shared local memory accessed via local-accessor - similar to (lacc-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*lacc-sc-3*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets,
                           simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-sc-4*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets,
                           simd<T, N> vals, PropertyListT props = {});

// scatter to shared local memory.
template <typename T, int N, int VS = 1, typename PropertyListT = empty_properties_t>
/*slm-sc-1*/ void scatter(simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
                          simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-sc-2*/ void scatter(simd<uint32_t, N / VS> byte_offsets,
                          simd<T, N> vals, PropertyListT props = {});

// scatter to shared local memory  - similar to (slm-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*slm-sc-3*/ void scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
                          simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-sc-4*/ void scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
                          PropertyListT props = {});
} // end namespace sycl::ext::intel::esimd
```

### Description
`(usm-sc-*)`: Stores ("scatters") the vector `vals` to global memory locations addressed by the base USM pointer `p` and byte-offsets `byte_offsets`.  
`(acc-sc-*)`, `(lacc-sc-*)`: Stores ("scatters") the vector `vals` to memory locations addressed by the the accessor and byte-offsets `byte_offsets`.  
`(slm-sc-*)`: Stores ("scatters") the vector `vals` to shared local memory locations addressed by `byte_offsets`.  
The parameter `byte_offset` is a vector of any integral type elements for `(usm-sc-*)`, 32-bit integer elements for `(lacc-sc-*)` and `(slm-sc-*)`, any integral type integer elements for `(acc-sc-*)` in [stateless](#statelessstateful-memory-mode) mode(default),
and up-to-32-bit integer elements for `(acc-sc-*)` in [stateful](#statelessstateful-memory-mode) mode.  
The optional parameter `mask` provides a `simd_mask`. If some element in `mask` is zero, then the store to the corresponding memory location is skipped.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-sc-*)` and `(slm-sc-*)` functions.
The template parameter `N` can be any positive number.  
The optional template parameter `VS` must be one of `{1, 2, 3, 4, 8, 16, 32, 64}` values. It specifies how many conseсutive elements are written per each element in `byte_offsets`.   
### Example
```C++
simd<int64_t, 4> offsets4(0, 100); // 0, 100, 200, 300 - offsets in bytes
simd<float, 4> vec4;
// stores the elements of vec4 to memory locations {ptr[0], ptr[100/4], ptr[200/4], ptr[300/4]};
scatter(ptr, offsets4, vec4);

// VS = 2, stores the elements of vec8 to memory locations {ptr[0], ptr[100/4],     ptr[200/4],     ptr[300/4],
//                                                          ptr[1], ptr[100/4 + 1], ptr[200/4 + 1], ptr[300/4 + 1]};
simd<float, 8> vec8;
scatter<float, 8, 2>(ptr, offsets4);
```

### Restrictions

`Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-store-functions) for `store` functions.


| `Function` | `Condition` | Required Intel GPU |
|-|-|-|
| `(usm-sc-*)`, `(acc-sc-*)` | !(cache-hints) and (`VS` == 1) and (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(usm-sc-*)`, `(acc-sc-*)` | (cache-hints) or (`VS` > 1) or (`N` != 1,2,4,8,16,32) | DG2 or PVC |
| The next 2 lines are similar to the previous 2 lines. They are for SLM gather and the only difference is that SLM scatters ignore cache-hints|||
| `(slm-sc-*)`, `(lacc-sc-*)` | !(cache-hints) and (`VS` == 1) and (`N` == 1,2,4,8,16,32) | Any Intel GPU |
| `(slm-sc-*)`, `(lacc-sc-*)` | (cache-hints) or (`VS` > 1) or (`N` != 1,2,4,8,16,32) | DG2 or PVC |

## load_2d(...) - load 2D block
```C++
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          int N = detail::get_lsc_block_2d_data_size<T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>(),
          typename PropertyListT = empty_properties_t>
simd<T, N> load_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
                   unsigned SurfacePitch, int X, int Y, PropertyListT props = {});
```
### Description
Loads and returns a vector `simd<T, N>` where `N` is `BlockWidth * BlockHeight * NBlocks`.  
`T` is element type.  
`BlockWidth` - the block width in number of elements.  
`BlockHeight` - the block height in number of elements.  
`NBlocks` - the number of blocks.  
`Transposed` - the transposed version or not.  
`Transformed` - apply VNNI transform or not.  
`N` - (automatically deduced) the size of the returned vector in elements.  
`Ptr` - the surface base address for this operation.  
`SurfaceWidth` - the surface width minus 1 in bytes.  
`SurfaceHeight` - the surface height minus 1 in rows.  
`SurfacePitch` - the surface pitch minus 1 in bytes.  
`X` - zero based X-coordinate of the left upper rectangle corner in number of elements.  
`Y` - zero based Y-coordinate of the left upper rectangle corner in rows.  
`props` - The optional compile-time properties. Only cache hint properties are used.

### Restrictions
* This function is available only for Intel® Data Center GPU Max Series (aka PVC).
* `Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-load-functions) for `load` functions.
* `Transformed` and `Transposed` cannot be set to true at the same time.
* `BlockWidth` * `BlockHeight` * `NBlocks` * sizeof(`T`) must not exceed 2048.
* If `Transposed` is `true` then:
  * sizeof(`T`) must be 4- or 8-byte (`dwords` or `qwords`).
  * `NBlocks` must be 1.
  * `BlockHeight` must be 8 for `qwords` and be in range [`1`..`32`] for `dwords`.
  * `BlockWidth` must be 1,2,4 for `qwords` and be in range [`1`..`8`] for `dwords`.
* If `Transformed` is `true` then:
  * sizeof(`T`) must be 1- or 2-byte (`bytes` or `words`).
  * `NBlocks` must be 1,2,4.
  * `BlockHeight` must be in range [4..32] for `bytes` and [2..32] for `words`.
  * `BlockWidth` must be in range [4..16] for `bytes` and [2..16] for `words`.
  * `BlockWidth` * `NBlocks` must not exceed 64 for `bytes` and 32 for `words`.
* If `Transposed` and `Transformed` are both set to `false` then:
  * `NBlocks` must be {1,2,4} for `bytes` and `words`, {1,2} for `dwords`, 1 for `qwords`.
  * `BlockHeight` must not exceed 32.
  * `BlockWidth` must be 4 or more for `bytes`, 2 or more for `words`, 1 or more for `dwords` and `qwords`.
  * `BlockWidth` * `NBlocks` must not exceed 64 for `bytes`, 32 for `words`, 16 for `dwords`, and 8 for `qwords`.


## prefetch_2d(...) - prefetch 2D block
```C++
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          int N = detail::get_lsc_block_2d_data_size<T, NBlocks, BlockHeight, BlockWidth, false /*Transposed*/, false /*Transformed*/>(),
          typename PropertyListT = empty_properties_t>
void prefetch_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
                 unsigned SurfacePitch, int X, int Y, PropertyListT props = {});
```
### Description
Prefetches elements from a memory block of the size `BlockWidth * BlockHeight * NBlocks` to cache.  
`T` is element type.  
`BlockWidth` - the block width in number of elements.  
`BlockHeight` - the block height in number of elements.  
`NBlocks` - the number of blocks.  
`N` - (automatically deduced) the size of the returned vector in elements.  
`Ptr` - the surface base address for this operation.  
`SurfaceWidth` - the surface width minus 1 in bytes.  
`SurfaceHeight` - the surface height minus 1 in rows.  
`SurfacePitch` - the surface pitch minus 1 in bytes.  
`X` - zero based X-coordinate of the left upper rectangle corner in number of elements.  
`Y` - zero based Y-coordinate of the left upper rectangle corner in rows.  
`props` - The compile-time properties, which must specify cache-hints.

### Restrictions
* This function is available only for Intel® Data Center GPU Max Series (aka PVC).
* `Cache-hint` properties must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-prefetch-functions) for `prefetch` functions.
* `BlockWidth` * `BlockHeight` * `NBlocks` * sizeof(`T`) must not exceed 2048.
* `NBlocks` must be {1,2,4} for `bytes` and `words`, {1,2} for `dwords`, 1 for `qwords`.
* `BlockHeight` must not exceed 32.
* `BlockWidth` must be 4 or more for `bytes`, 2 or more for `words`, 1 or more for `dwords` and `qwords`.
* `BlockWidth` * `NBlocks` must not exceed 64 for `bytes`, 32 for `words`, 16 for `dwords`, and 8 for `qwords`.

## store_2d(...) - store 2D block
```C++
template <typename T, int BlockWidth, int BlockHeight = 1,
          int N = detail::get_lsc_block_2d_data_size<T, 1u, BlockHeight, BlockWidth, false /*Transposed*/, false /*Transformed*/>(),
          typename PropertyListT = empty_properties_t>
void store_2d(T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
              unsigned SurfacePitch, int X, int Y, simd<T, N> Vals, PropertyListT props = {});

```
### Description
Stores the vector `Vals` of the type `simd<T, N>` to 2D memory block where `N` is `BlockWidth * BlockHeight`.  
`T` is element type of the values to be stored to memory.  
`BlockWidth` - the block width in number of elements.  
`BlockHeight` - the block height in number of elements.  
`N` - (automatically deduced) the size of the vector to be stored.  
`Ptr` - the surface base address for this operation.  
`SurfaceWidth` - the surface width minus 1 in bytes.  
`SurfaceHeight` - the surface height minus 1 in rows.  
`SurfacePitch` - the surface pitch minus 1 in bytes.  
`X` - zero based X-coordinate of the left upper rectangle corner in number of elements.  
`Y` - zero based Y-coordinate of the left upper rectangle corner in rows.  
`props` - The optional compile-time properties. Only cache hint properties are used.

### Restrictions
* This function is available only for Intel® Data Center GPU Max Series (aka PVC).
* `Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-store-functions) for `store` functions.
* `BlockWidth` * `BlockHeight` * sizeof(`T`) must not exceed 512.
* `BlockHeight` must not exceed 8.
* `BlockWidth` must be 4 or more for `bytes`, 2 or more for `words`, 1 or more for `dwords` and `qwords`.
* `BlockWidth` must not exceed 64 for `bytes`, 32 for `words`, 16 for `dwords`, and 8 for `qwords`.

## atomic_update(...)

### atomic_update() with 0 operands (inc, dec, load)
```C++
namespace sycl::ext::intel::esimd {
// Atomic update the USM memory locations - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename PropertyListT = empty_properties_t>
/*usm-au0-1*/ simd<T, N> atomic_update(T *p, simd<Toffset, N> byte_offset, simd_mask<N> mask, props = {});
/*usm-au0-2*/ simd<T, N> atomic_update(T *p, simd<Toffset, N> byte_offset,props = {});

// Similar to (usm-au0-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename OffsetSimdViewT,
          typename PropertyListT = detail::empty_properties_t>
/*usm-au0-3*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset, simd_mask<N> mask, props = {});
/*usm-au0-4*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset, props = {});


// Atomic update the memory locations referenced by device-accessor - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-au0-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd_mask<N> mask, props = {});
/*acc-au0-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       props = {});

// Similar to (acc-au0-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename AccessorT, typename OffsetSimdViewT,
          typename PropertyListT = empty_properties_t>
/*acc-au0-3*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       simd_mask<N> mask, props = {});
/*acc-au0-4*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       props = {});


// Atomic update the memory locations referenced by local-accessor (SLM) - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-au0-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd_mask<1> mask = 1);

// Atomic update the shared local memory (SLM) - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N>
/*slm-au0-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd_mask<N> mask = 1);
```

### atomic_update() with 1 operands (*add, *sub, *min, *max, bit_or/xor/and, store, xchg)
```C++
// Atomic update the USM memory locations - 1 operand (add, max, etc.).
/*usm-au1-1*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*usm-au1-2*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, props = {});
// Similar to (usm-au1-1,2), but `byte_offset` is `simd_view`.
/*usm-au1-3*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*usm-au1-4*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, props = {});


// Atomic update the memory locations referenced by device-accessor - 1 operand (add, max, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-au1-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*acc-au1-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, props = {});

// Similar to (acc-au1-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename AccessorT,
          typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*acc-au1-3*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*acc-au1-4*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, props = {});

// Atomic update the memory locations referenced by local-accessor (SLM) - one operand (add, max, etc.).
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-au1-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd<T, N> src0, simd_mask<1> mask = 1);

// Atomic update the shared local memory (SLM) - one operand (add, max etc.).
template <atomic_op Op, typename T, int N>
/*slm-au1-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd<T, N> src0, simd_mask<N> mask = 1);
```
### atomic_update() with 2 operands (cmpxchg, fcmpxchg)
```C++
// Atomic update the USM memory locations - 2 operand: cmpxchg, fcmpxchg.
/*usm-au2-1*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*usm-au2-2*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});
// Similar to (usm-au2-1,2), but `byte_offset` is `simd_view`.
/*usm-au2-3*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*usm-au2-4*/ simd<T, N> atomic_update(T *p, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});


// Atomic update the memory locations referenced by device-accessor - 2 operands: cmpxchg, fcmpxchg.
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-au2-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*acc-au2-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});

// Similar to (acc-au2-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename AccessorT,
          typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*acc-au2-3*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*acc-au2-4*/ simd<T, N> atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});

// Atomic update the memory locations referenced by local-accessor (SLM) - two operands: cmpxchg, fcmpxchg.
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-au2-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd<T, N> src0, simd<T, N> src1, simd_mask<1> mask = 1);

// Atomic update the shared local memory (SLM) - two operands: cmpxchg, fcmpxchg.
template <atomic_op Op, typename T, int N>
/*slm-au2-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask = 1);
} // end namespace sycl::ext::intel::esimd
```
### Description
`(usm-*)`: Atomically updates the global memory locations addressed by the base USM pointer `ptr` and byte-offsets `byte_offset`.  
`(acc-*)`, `(lacc-*)`: Atomically updates the memory locations addressed by the the accessor and byte-offsets `byte_offset`.  
`(slm-*)`: Atomically updates the shared memory locations addressed by `byte_offset`.  
The parameter `byte_offset` is a vector of any integral type elements for `(usm-*)`, 32-bit integer elements for `(lacc-*)` and `(slm-*)`, any integral type integer elements for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode(default),
and up-to-32-bit integer elements for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode.  
The optional parameter `mask` provides a `simd_mask`. If some element in `mask` is zero, then the corresponding memory location is not updated.  
`(usm-*)`, `(acc-*)`: The optional [compile-time properties](#compile-time-properties) list `props` may specify `cache-hints`.  
The template parameter `Op` specifies the atomic operation applied to the memory.  
The template parameter `T` specifies the type of the elements used in the atomic_update operation. Only 2,4,8-byte types are supported.  
The template parameter `N` is the number of elements being atomically updated.

### Restrictions
`Cache-hint` properties, if passed, must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-atomic_update-functions) for `atomic_update` functions.

| `Function` | `Condition` | Required Intel GPU |
|-|-|-|
| `(usm-au0-*)`, `(acc-au0-*)` | !(cache-hints) and (`N` == 1,2,4,8,16,32) and (sizeof(T) >= 4) | Any Intel GPU |
| `(usm-au0-*)`, `(acc-au0-*)` | (cache-hints) or (`N` != 1,2,4,8,16,32) or (sizeof(T) == 2) | DG2 or PVC |
| `(usm-au1-*)`, `(acc-au1-*)`, `(usm-au2-*)`, `(acc-au2-*)`  | !(cache-hints) and (`N` == 1,2,4,8,16,32) and (sizeof(T) >= 4) and (`Op` is integral operation) | Any Intel GPU |
| `(usm-au1-*)`, `(acc-au1-*)`, `(usm-au2-*)`, `(acc-au2-*)` | (cache-hints) or (`N` != 1,2,4,8,16,32) or (sizeof(T) == 2) or (`Op` is FP operation) | DG2 or PVC |
|-|-|-|
| `(slm-au0-*)`, `(lacc-au0-*)` | (`N` == 1,2,4,8,16,32) and (sizeof(T) == 4) | Any Intel GPU |
| `(slm-au0-*)`, `(lacc-au0-*)` | (`N` != 1,2,4,8,16,32) or (sizeof(T) == 2) or (sizeof(T) == 8)| DG2 or PVC |
| `(slm-au1-*)`, `(lacc-au1-*)`, `(slm-au2-*)`, `(lacc-au2-*)` | (`N` == 1,2,4,8,16,32) and (sizeof(T) == 4) and (`Op` is integral operation) | Any Intel GPU |
| `(slm-au1-*)`, `(lacc-au1-*)`, `(slm-au2-*)`, `(lacc-au2-*)` | (`N` != 1,2,4,8,16,32) or (sizeof(T) == 2) or (sizeof(T) == 8) or (`Op` is FP operation)| DG2 or PVC |


## prefetch(...)
```C++
namespace sycl::ext::intel::esimd {
template <typename T, int N, int VS, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-pf-1*/ void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
                           simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-pf-2*/ void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
                           PropertyListT props = {});

// The next 2 are similar to (usm-pf-1,2). `VS` parameter is set to 1,
// which allows callers to omit explicit specification of `T` and `N` params.
template <typename T, int N, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-pf-3*/ void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
                           simd_mask<N> mask, PropertyListT props = {});
/*usm-pf-4*/ void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
                           PropertyListT props = {});

// The next 2 are similar to (usm-1,2), added to handle `byte_offsets` in `simd_view` form.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*usm-pf-5*/ void prefetch(const T *p, OffsetSimdViewT byte_offsets,
                           simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-pf-6*/ void prefetch(const T *p, OffsetSimdViewT byte_offsets,
                           PropertyListT props = {});

// The next functions perform transposed 1-channel prefetch - prefetch linear block of code.
template <typename T, int VS = 1, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-pf-7*/ void prefetch(const T *p, OffsetT byte_offset,
                           simd_mask<1> mask, PropertyListT props = {});
/*usm-pf-8*/ void prefetch(const T *p, OffsetT byte_offset,
                           PropertyListT props = {});
template <typename T, int VS = 1, typename PropertyListT = empty_properties_t>
/*usm-pf-9*/ void prefetch(const T *p, simd_mask<1> mask, PropertyListT props = {});
/*usm-pf-10*/ void prefetch(const T *p, PropertyListT props = {});


template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-pf-1*/ void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                           simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-pf-2*/ void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                           PropertyListT props = {});

// The next 2 are similar (acc-pf-1,2). `VS` parameter is set to 1,
// which allows callers to omit explicit specification of `T` and `N` params.
template <typename T, int N, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-pf-3*/ void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
                           simd_mask<N> mask, PropertyListT props = {});
/*acc-pf-4*/ void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
                           PropertyListT props = {});

// The next 2 are similar to (acc-1,2), added to handle `byte_offsets` in `simd_view` form.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
          typename AccessorT, typename PropertyListT = empty_properties_t>
/*acc-pf-5*/ void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
                           simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-pf-6*/ void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
                           PropertyListT props = {});

/// The next functions perform transposed 1-channel prefetch.
template <typename T, int VS = 1, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-pf-7*/ void prefetch(AccessorT acc, OffsetT byte_offset, simd_mask<1> mask,
                           PropertyListT props = {});
/*acc-pf-8*/ void prefetch(AccessorT acc, OffsetT byte_offset,
                           PropertyListT props = {});
template <typename T, int VS = 1, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-pf-9*/ void prefetch(AccessorT acc, simd_mask<1> mask, PropertyListT props = {});
/*acc-pf-10*/ void prefetch(AccessorT acc, PropertyListT props = {});
} // end namespace sycl::ext::intel::esimd
```
### Description
`(usm-pf-1,2,3,4,5,6)`: Prefetches the memory locations addressed by the base USM pointer `ptr` and the vector of any integral type byte-offsets `byte_offsets`.

`(acc-pf-1,2,3,4,5,6)`: Prefetches the memory locations addressed by the accessor `acc` and the vector of byte-offsets `byte_offsets`.
The `byte_offsets` is a vector of any integral type elements, limited in [stateful](#statelessstateful-memory-mode) mode by 32-bits maximum.

`(usm-pf-7,8,9,10)`: Prefetches a linear block of memory addressed by the base USM pointer `ptr` and the optional 64-bit `byte-offset`.

`(acc-pf-7,8,9,10)`: Prefetches a linear block of memory addressed by the accessor `acc` and the optional `byte-offset` parameter, which is 64-bit in [stateless](#statelessstateful-memory-mode) mode(default), and 32-bit in [stateful](#statelessstateful-memory-mode) mode.

`(usm-pf-1,2,3,4,5,6)`, `(acc-pf-1,2,3,4,5,6)`: The optional parameter `mask` provides a `simd_mask`. If some element in `mask` is zero, then the corresponding memory location is not prefetched.  
`(usm-pf-7,8,9,10)`, `(acc-pf-7,8,9,10)`: The optional parameter `mask` provides 1-element
`simd_mask`. If it is zero, then the whole prefetch operation is skipped.

`(usm-pf-*)`, `(acc-pf-*)`: The [compile-time properties](#compile-time-properties) list `props` must specify `cache-hints`.

### Restrictions

* This function is available only for Intel® Arc Series (aka DG2) or Intel® Data Center GPU Max Series (aka PVC).
* `Cache-hint` properties must follow the [rules](#valid-combinations-of-l1-and-l2-cache-hints-for-prefetch-functions) for `prefetch` functions.



## fence(...) - set the memory read/write order
```C++
namespace sycl::ext::intel::esimd {
enum fence_mask : uint8_t {
  /// “Commit enable” - wait for fence to complete before continuing.
  global_coherent_fence = 0x1,
  /// Flush the instruction cache.
  l2_flush_instructions = 0x2,
  /// Flush sampler (texture) cache.
  l2_flush_texture_data = 0x4,
  /// Flush constant cache.
  l2_flush_constant_data = 0x8,
  /// Flush constant cache.
  l2_flush_rw_data = 0x10,
  /// Issue SLM memory barrier only. If not set, the memory barrier is global.
  local_barrier = 0x20,
  /// Flush L1 read - only data cache.
  l1_flush_ro_data = 0x40
};
/*fence-1*/template <uint8_t ctrl_mask> void fence();


/// The target memory kind for fence() operation.
enum class memory_kind : uint8_t {
  global = 0, /// untyped global memory
  image = 2, /// image (also known as typed global memory)
  local = 3, /// shared local memory
};
/// The cache flush operation to apply to caches after fence() is complete.
enum class fence_flush_op : uint8_t {
  none = 0,       /// no operation;
  evict = 1,      /// R/W: evict dirty lines; R/W and RO: invalidate clean lines
  invalidate = 2, /// R/W and RO: invalidate all clean lines;
  clean = 4 /// R/W: dirty lines are written to memory, but retained in
            /// cache in clean state; RO: no effect.
};
/// The scope that fence() operation should apply to.
enum class fence_scope : uint8_t {
  /// Wait until all previous memory transactions from this thread are observed
  /// within the local thread-group.
  group = 0,
  /// Wait until all previous memory transactions from this thread are observed
  /// within the local sub-slice.
  local = 1,
  /// Wait until all previous memory transactions from this thread are observed
  /// in the local tile.
  tile = 2,
  /// Wait until all previous memory transactions from this thread are observed
  /// in the local GPU.
  gpu = 3,
  /// Wait until all previous memory transactions from this thread are observed
  /// across all GPUs in the system.
  gpus = 4,
  /// Global memory data-port only: wait until all previous memory transactions
  /// from this thread are observed at the "system" level.
  system = 5,
  /// Global memory data-port only: for GPUs that do not follow
  /// PCIe Write ordering for downstream writes targeting device memory,
  /// this op will commit to device memory all downstream and peer writes that
  /// have reached the device.
  system_acquire = 6
};

/*fence-2*/template <memory_kind Kind = memory_kind::global,
                     fence_flush_op FenceOp = fence_flush_op::none,
                     fence_scope Scope = fence_scope::group> void fence();
} // end namespace sycl::ext::intel::esimd
```
### Description
`(fence-1)`: Sets the memory read/write order. This function has pretty limited functionality compared to `(fence-2)`. It accepts an 8-bit `ctrl_mask` containing one or more `fence_mask` enum values in it. It can be used for any Intel GPU.

`(fence-2)`: Sets the memory read/write order. This function provide a bit more flexible controls comparing to `(fence-1)`, but requires `Intel® Arc Series` (aka `DG2`) or `Intel® Data Center GPU Max Series` (aka `PVC`) to run.

## Examples
```C++
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  namespace esimd_ex = sycl::ext::intel::experimental::esimd;
  ...
  // Case1: load <float, 128>, specify the alignment = 16:
  // old/obsolete experimental function - not recommended
  auto vec = esimd_ex::lsc_block_load<float, 128>(fptr, overaligned_tag<16>{});
  // new API described in this document:
  auto vec = block_load<float, 128>(fptr, properties{alignment<16>});


  // Case2: load <float, 16> using L1 and L2 cache-hints:
  // old/obsolete experimental function - not recommended
  auto vec2 =
      esimd_ex::lsc_block_load<float, 16, esimd_ex::lsc_data_size::default_size,
                               esimd_ex::cache_hint::uncached /*L1*/,
                               esimd_ex::cache_hint::cached /*L2*/>(fptr);
  // new API described in this document:
  properties props{cache_hint_L1<cache_hint::uncached>, cache_hint_L2<cache_hint::cached>};
  auto vec2 = block_load<float, 16>(fptr, props);

  // Case3 store `vec2` to global memory using L1/L2 cache-hints:
  // old/obsolete experimental function - not recommended
  esimd_ex::lsc_block_store<float, 16, esimd_ex::lsc_data_size::default_size,
                            esimd_ex::cache_hint::write_back /*L1*/,
                            esimd_ex::cache_hint::write_back /*L2*/>(fptr, vec2);
  // new API described in this document:
  block_store(fptr, properties{cache_hint_L1<cache_hint::write_back>,
                               cache_hint_L2<cache_hint::write_back>});

  auto vec2_view = vec2.select<16,1>();
  // Passing the simd_view currently requires specifying `T` and `N` parameters:
  block_store(fptr, vec2_view); // Compilation error: cannot match simd_view operand to simd
  block_store<float, 16>(fptr, vec2_view); // This works well.
```
