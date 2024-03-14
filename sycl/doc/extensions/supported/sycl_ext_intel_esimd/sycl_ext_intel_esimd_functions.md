# ESIMD methods and functions

This document describes the ESIMD methods and functions, their semantics,
restrictions and hardware dependencies.
Look for more general ESIMD documentation [here](./sycl_ext_intel_esimd.md).

## Table of content
- [Compile-time properties](#compile-time-properties)
- [Stateless/stateful memory mode](#statelessstateful-memory-mode)
- [block_load(...) - fast load from a contiguous memory block](#block_load---fast-load-from-a-contiguous-memory-block)
- [block_store(...) - fast store to a contiguous memory block](#block-store---fast-store-to-a-contiguous-memory-block)
- [gather(...)](#gather---load-from-memory-locations-addressed-by-a-vector-of-offsets)
- [scatter(...)](#scatter---store-to-memory-locations-addressed-by-a-vector-of-offsets)
- [atomic_update(...)](#atomic_update)
- [prefetch(...)](#prefetch)
- [Examples](#examples)

## Other content:
* [General ESIMD documentation](./sycl_ext_intel_esimd.md)
* [ESIMD API/doxygen reference](https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd.html)
* [Examples](./examples/README.md)
* [ESIMD LIT tests - working code examples](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/ESIMD/)

---
## Stateless/stateful memory mode
ESIMD functions may assume `stateful` and `stateless` access to the memory.  
`Stateless` read/write/prefetch is such that uses USM pointer to global memory,
which also may be adjusted by a scalar/vector 64-bit offset.  
'Stateful' read/write/prefetch is the access to memory that uses a
`surface-index` and `32-bit` scalar/vector offset. Originally the ESIMD
functions accepting `device accessors` were translated into `stateful`
read/write/prefetch instructions. After the recent enabling of the `stateless`
memory enforcement mode was turned ON by default, the device-accessor-based APIs
started being translated into `stateless` access instructions too.  
The `-fsycl-esimd-force-stateless-mem` compilation option (it is ON by default)
specifies how to handle the functions accepting `device-accessors`. In the default
mode the functions are translated into `stateless` memory accesses and the functions also
may accept 64-bit scalar/vector offset.  
With `-fno-sycl-esimd-force-stateless-mem` compilation switch the functions
accepting `device accessors` are translated into `stateful` memory accesses and
the functions may accept only 32-bit scalar/vector offsets.

Some of ESIMD memory API are considered `stateless`. Those are the functions that accept USM pointer
as a reference to memory. Such functions address 64-bit add

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
Many of ESIMD functions have an optional argument `alignment`, `L1-cache-hint`, `L2-cache-hint`,
`L3-cache-hint`(reserved). The list may be extended in future. The properties may be added
to the properties list in any order. See the example of using the properties below:
```C++
using namespace sycl::ext::intel::esimd;

// f32_ptr is promised to be aligned by 16-bytes, no cache-hints passed.
auto vec_a = block_load<float, 16>(f32_ptr, properties{alignment<16>});

// f32_ptr is promised to be aligned by 4-bytes only, L1=uncached, L2=cached.
properties props{cache_hint_L1<cache_hint::uncached>, alignment<4> cache_hint_L1<cache_hint::cached>};
auto vec_b = block_load<float, 16>(f32_ptr + 1, props);
```

Cache-hint properties (if passed) currently add the restriction to the target-device, it must be Intel® Arc Series (aka DG2) or Intel® Data Center GPU Max Series (aka PVC).

## block_load(...) - fast load from a contiguous memory block
```C++
namespace sycl::ext::intel::esimd {
template <typename T, int N, typename PropertyListT = empty_properties_t>
// block load from USM memory.
/*usm-1*/ simd<T, N> block_load(const T* ptr, props={});
/*usm-2*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, props={});
/*usm-3*/ simd<T, N> block_load(const T* ptr, simd_mask<1> pred, props={});
/*usm-4*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred, props={});
/*usm-5*/ simd<T, N> block_load(const T* ptr, simd_mask<1> pred, simd<T, N> pass_thru, props={});
/*usm-6*/ simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, props={});

// block load from device accessor.
/*acc-1*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, props = {});
/*acc-2*/ simd<T, N> block_load(AccessorT acc, props = {});
/*acc-3*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, props = {});
/*acc-4*/ simd<T, N> block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred, props = {});
/*acc-5*/ simd<T, N> block_load(AccessorT acc, simd_mask<1> pred, simd<T, N> pass_thru, props = {});
/*acc-6*/ simd<T, N> block_load(AccessorT acc, simd_mask<1> pred, props = {});

// block load from local accessor (SLM).
/*lacc-1*/ simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset, props={});
/*lacc-2*/ simd<T, N> block_load(local_accessor lacc, props={});
/*lacc-3*/ simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset, simd_mask<1> pred, props={});
/*lacc-4*/ simd<T, N> block_load(local_accessor lacc, simd_mask<1> pred, props={});
/*lacc-5*/ simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, props={});
/*lacc-6*/ simd<T, N> block_load(local_accessor lacc, simd_mask<1> pred, simd<T, N> pass_thru, props={});

// block load from SLM (Shared Local Memory).
/*slm-1*/ simd<T, N> slm_block_load(uint32_t byte_offset, props={});
/*slm-2*/ simd<T, N> slm_block_load(uint32_t byte_offset, simd_mask<1> pred, props={});
/*slm-3*/ simd<T, N> slm_block_load(uint32_t byte_offset, simd_mask<1> pred, simd<T, N> pass_thru, props={});
}
```
### Description
`(usm-*)`: Loads a contiguous memory block from the global memory referenced by the USM pointer `ptr` optionally adjusted by `byte_offset`.  
`(acc-*)`, `(lacc-*)`: Loads a contiguous memory block from the memory  referenced referenced by the accessor optionally adjusted by `byte_offset`.  
`(slm-*)`: Loads a contiguous memory block from the shared local memory  referenced by `byte_offset`.  
The optional parameter `byte_offset` has a scalar integer 64-bit type for `(usm-*)`, 32-bit type for `(lacc-*)` and `(slm-*)`, 32-bit for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a 1-element `simd_mask`. If zero mask is passed, then the load is skipped and the `pass_thru` value is returned.  
If `pred` is zero and `pass_thru` operand was not passed, then the function returns an undefined value.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-*)` and `(slm-*)` functions.

### Restrictions/assumptions:
`Alignment` - if not specified by the `props` param, then `assumed` alignment is used. If the actual memory reference requires a smaller alignment than the `assumed`, then it must be explicitly passed in `props` argument.

| `Function` | `Assumed` alignment   | `Minimally required` alignment |
|-|-|-|
| `(usm-*)`  | `max(4, sizeof(T))` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |
| `(acc-*)`  | `max(4, sizeof(T))` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |
| `(lacc-*)`, `(slm-*)` | `16` | `sizeof(T)` if no cache-hints, otherwise it is `max(4, sizeof(T))` |

`N` - the valid values may depend on usage of cache-hints or passing of the `pred` argument:

| `Function` | `Condition` | Requirement for `N` | Required/supported Intel GPU |
|-|-|-|-|
| `(usm-*)` | (no cache-hints) and (`pred` is not passed) | `N` is any positive number | Any Intel GPU |
| `(usm-*)` | (cache-hints) or (`pred` is passed) | `N` must be from [Table1 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(acc-*)` | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above |
| `(acc-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and (no cache-hints) and (`pred` is not passed) and (sizeof(`T`) * `N` == 16,32,64,128) | sizeof(`T`)*`N` == 16,32,64,128 | Any Intel GPU |
| `(acc-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and ((cache-hints) or (`pred` is passed)) or (sizeof(`T`) * `N` != 16,32,64,128) | `N` must be from [Table1 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(lacc-1,2)`, `(slm-1)` | `pred` is not passed | `N` is any positive number | Any Intel GPU |
| `(lacc-3,4,5,6)`, `(slm-2,3)`  | `pred` is passed | `N` must be from [Table1 below] | Any Intel GPU |


#### Table1 - Valid values of `N` if cache-hints used or `pred` parameter is passed:
| sizeof(`T`) | Valid values of `N` | Special case - PVC only - the maximal `N`: requires bigger alignment: 8 or more |
|---|--------------------------------|-----|
| 1 | 4, 8, 12, 16, 32, 64, 128, 256 | 512 |
| 2 | 2, 4, 6, 8, 16, 32, 64, 128 | 256 |
| 4 | 1, 2, 3, 4, 8, 16, 32, 64 | 128 |
| 8 | 1, 2, 3, 4, 8, 16, 32 | 64 |

## block store(...) - fast store to a contiguous memory block
```C++
namespace sycl::ext::intel::esimd {
template <typename T, int N, typename PropertyListT = empty_properties_t>
// block store to USM memory.
/*usm-1*/ void block_store(T* ptr, simd<T, N> vals, props={});
/*usm-2*/ void block_store(T* ptr, size_t byte_offset, simd<T, N> vals, props={});
/*usm-3*/ void block_store(T* ptr, simd<T, N> vals, simd_mask<1> pred, props={});
/*usm-4*/ void block_store(T* ptr, size_t byte_offset, simd<T, N> vals, simd_mask<1> pred, props={});

// block store to device accessor.
/*acc-1*/ void block_store(AccessorT acc, OffsetT byte_offset, simd<T, N> vals, props = {});
/*acc-2*/ void block_store(AccessorT acc, simd<T, N> vals, props = {});
/*acc-3*/ void block_store(AccessorT acc, OffsetT byte_offset, simd<T, N> vals, simd_mask<1> pred, props = {});
/*acc-4*/ void block_store(AccessorT acc, simd<T, N> vals, simd_mask<1> pred, props = {});

// block store to local accessor (SLM).
/*lacc-1*/ void block_store(local_accessor lacc, uint32_t byte_offset, simd<T, N> vals, props={});
/*lacc-2*/ void block_store(local_accessor lacc, simd<T, N> vals, props={});
/*lacc-3*/ void block_store(local_accessor lacc, uint32_t byte_offset, simd<T, N> vals, simd_mask<1> pred, props={});
/*lacc-4*/ void block_store(local_accessor lacc, simd<T, N> vals, props={});
void block_store(local_accessor lacc, simd<T, N> vals, simd_mask<1> pred, props={});

// block store to SLM (Shared Local Memory).
/*slm-1*/ void slm_block_store(uint32_t byte_offset, simd<T, N> vals, simd_mask<1> pred, props={});
/*slm-2*/ void slm_block_store(uint32_t byte_offset, simd<T, N> vals, props={});
}
```
### Description
`(usm-*)`: Stores `vals` to a contiguous global memory block referenced by the USM pointer `ptr` optionally adjusted by `byte_offset`.  
`(acc-*)`, `(lacc-*)`: Stores `vals` to a contiguous global memory block referenced by the accessor optionally adjusted by `byte_offset`.  
`(slm-*)`: Stores `vals` to a contiguous shared-local-memory block referenced by `byte_offset`.  
The optional parameter `byte_offset` has a scalar integer 64-bit type for `(usm-*)`, 32-bit type for `(lacc-*)` and `(slm-*)`, 32-bit for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a 1-element `simd_mask`. If zero mask is passed, then the store is skipped.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-*)` and `(slm-*)` functions.

### Restrictions/assumptions:
`Alignment` - if not specified by the `props` param, then `assumed` alignment is used. If the actual memory reference requires a smaller alignment than the `assumed`, then it must be explicitly passed in `props` argument.

| `Function` | Condition | `Assumed` alignment   | `Minimally required` alignment |
|-|-|-|-|
| `(usm-*)`  | (no cache-hints) and (`pred` is not passed). | `16` | `sizeof(T))` |
| `(usm-*)`  | (cache-hints) or (`pred` is passed). | `max(4, sizeof(T))` | `max(4, sizeof(T))` |
| `(acc-*)`  | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above |
| `(acc-*)`  | [Stateful](#statelessstateful-memory-mode) memory mode and (no cache-hints) and (`pred` is not passed) and (`sizeof(T) * N` == {16,32,64,128}) | `16` | `max(4, sizeof(T))` |
| `(acc-*)`  | [Stateful](#statelessstateful-memory-mode) memory mode and ((cache-hints) or (`pred` is passed) or (`sizeof(T) * N` != {16,32,64,128})) | `max(4, sizeof(T))` | `max(4, sizeof(T))` |
| `(lacc-1,2)`, `(slm-2)` | `pred` is not passed | `16` | `sizeof(T)` |
| `(lacc-3,4)`, `(slm-1)` | `pred` is passed  | `max(4, sizeof(T))` | `max(4, sizeof(T))` |

`N` - the valid values may depend on usage of cache-hints or passing of the `pred` argument:

| `Function` | `Condition` | Requirement for `N` | Required/supported Intel GPU |
|-|-|-|-|
| `(usm-*)` | (no cache-hints) and (`pred` is not passed) | `N` is any positive number | Any Intel GPU |
| `(usm-*)` | (cache-hints) or (`pred` is passed) | `N` must be from [Table2 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(acc-*)` | [Stateless](#statelessstateful-memory-mode) memory mode (default) | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above | Lowered to `(usm-*)` - Read the corresponding `(usm-*)` line above |
| `(acc-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and (no cache-hints) and (`pred` is not passed) and (sizeof(`T`) * `N` == 16,32,64,128) | sizeof(`T`)*`N` == 16,32,64,128 | Any Intel GPU |
| `(acc-*)` | ([Stateful](#statelessstateful-memory-mode) memory mode: `-fno-sycl-esimd-force-stateless-mem`) and ((cache-hints) or (`pred` is passed)) or (sizeof(`T`) * `N` != 16,32,64,128) | `N` must be from [Table2 below](#table1---valid-values-of-n-if-cache-hints-used-or-pred-parameter-is-passed) | DG2 or PVC |
| `(lacc-1,2)`, `(slm-2)`  | `pred` is not passed | `N` is any positive number | Any Intel GPU |
| `(lacc-3,4)`, `(slm-1)`  | `pred` is passed | `N` must be from [Table2 below] | DG2 or PVC |


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
/*usm-1*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                            simd<T, N> pass_thru, PropertyListT props = {});
/*usm-2*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                            PropertyListT props = {});
/*usm-3*/ simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
                            PropertyListT props = {});

// gather from USM - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-4*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                            PropertyListT props = {});
/*usm-5*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
                            PropertyListT props = {});
/*usm-6*/ simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
                            PropertyListT props = {});

// gather from USM - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT, typename PropertyListT = empty_props_t>
/*usm-7*/ simd <T, N> gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*usm-8*/ simd <T, N> gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-9*/ simd <T, N> gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             PropertyListT props = {});


// gather from memory accessed via device-accessor - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-1*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                            simd<T, N> pass_thru, PropertyListT props = {});
/*acc-2*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                            PropertyListT props = {});
/*acc-3*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                            PropertyListT props = {});

// gather from memory accessed via device-accessor - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-4*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                            PropertyListT props = {});
/*acc-5*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
                            PropertyListT props = {});
/*acc-6*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                            PropertyListT props = {});

// gather from memory accessed via device-accessor - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetObjT,
          typename OffsetRegionT, typename PropertyListT = empty_props_t>
/*acc-7*/ simd <T, N> gather(AccessorT acc, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*acc-8*/ simd <T, N> gather(AccessorT acc, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-9*/ simd <T, N> gather(AccessorT acc, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
                             PropertyListT props = {});


// gather from memory accessed via local-accessor/SLM - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-1*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                             simd<T, N> pass_thru, PropertyListT props = {});
/*lacc-2*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
                             PropertyListT props = {});
/*lacc-3*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                             PropertyListT props = {});

// gather from memory accessed via local-accessor/SLM - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-4*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                             PropertyListT props = {});
/*lacc-5*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
                             PropertyListT props = {});
/*lacc-6*/ simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
                             PropertyListT props = {});

// gather from memory accessed via local-accessor/SLM - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*lacc-7*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                              simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*lacc-8*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                              simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-9*/ simd <T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
                              PropertyListT props = {});


// gather from shared local memmory (SLM) - general form (must specify T, N, VS parameters).
template <typename T, int N, int VS, typename PropertyListT = empty_properties_t>
/*slm-1*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask,
                            simd<T, N> pass_thru, PropertyListT props = {});
/*slm-2*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-3*/ simd<T, N> gather(simd<uint32_t, N / VS> byte_offsets, PropertyListT props = {});

// gather from shared local memmory (SLM) - convenience/short form (VS = 1; T and N can also be omitted)
template <typename T, int N, typename PropertyListT = empty_properties_t>
/*slm-4*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets, simd_mask<N> mask, simd<T, N> pass_thru,
                            PropertyListT props = {});
/*slm-5*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets, simd_mask<N> mask, PropertyListT props = {});
/*slm-6*/ simd<T, N> gather(simd<uint32_t, N> byte_offsets, PropertyListT props = {});

// gather from shared local memory (SLM) - general form accepting offsets as simd_view
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_props_t>
/*slm-7*/ simd <T, N> gather(OffsetSimdViewT byte_offsets,
                             simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {});
/*slm-8*/ simd <T, N> gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-9*/ simd <T, N> gather(OffsetSimdViewT byte_offsets, PropertyListT props = {});
```

### Description
`(usm-*)`: Loads ("gathers") elements of the type `T` from global memory locations addressed by the base USM pointer `p` and byte-offsets `byte_offsets`.  
`(acc-*)`, `(lacc-*)`: Loads ("gathers") elements of the type `T` from memory locations addressed by the the accessor and byte-offsets `byte_offsets`.  
`(slm-*)`: Loads ("gathers") elements of the type `T` from shared local memory locations addressed by `byte_offsets`.  
The parameter `byte_offsets` has is a vector of integer 64-bit type elements for `(usm-*)`, 32-bit integer elements for `(lacc-*)` and `(slm-*)`, 32-bit integer elements for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit integer elements for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a `simd_mask`. If some element in `pred` is zero, then the load of the corresponding memory location is skipped and the element of the result is copied from `pass_thru` (if it is passed) or it is undefined (if `pass_thru` is omitted).  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-*)` and `(slm-*)` functions.

## scatter(...) - store to memory locations addressed by a vector of offsets
```C++
namespace sycl::ext::intel::esimd {
// scatter to USM memory.
template <typename T, int N, int VS = 1, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-1*/ void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-2*/ void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals, PropertyListT props = {});

// scatter to USM memory - similar to (usm-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*usm-3*/ void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-4*/ void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals, PropertyListT props = {});


// scatter to memory accessed via device-accessor.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetT, typename PropertyListT = empty_properties_t>
/*acc-1*/ void scatter(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-2*/ void scatter(AccessorT acc, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals, PropertyListT props = {});

// scatter to memory accessed via device-accessor - similar to (acc-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*acc-3*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals, simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-4*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals, PropertyListT props = {});


// scatter to shared local memory accessed via local-accessor.
template <typename T, int N, int VS = 1, typename AccessorT, typename PropertyListT = empty_properties_t>
/*lacc-1*/ void scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-2*/ void scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
                        simd<T, N> vals, PropertyListT props = {});

// scatter to shared local memory accessed via local-accessor - similar to (lacc-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*lacc-3*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*lacc-4*/ void scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
                        PropertyListT props = {});

// scatter to shared local memory.
template <typename T, int N, int VS = 1, typename PropertyListT = empty_properties_t>
/*slm-1*/ void scatter(simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
                       simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-2*/ void scatter(simd<uint32_t, N / VS> byte_offsets,
                       simd<T, N> vals, PropertyListT props = {});

// scatter to shared local memory  - similar to (slm-1,2), but the `byte_offsets` is `simd_view`.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*slm-3*/ void scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
                       simd_mask<N / VS> mask, PropertyListT props = {});
/*slm-4*/ void scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
                       PropertyListT props = {});
```

### Description
`(usm-*)`: Stores ("scatters") the vector `vals` to global memory locations addressed by the base USM pointer `p` and byte-offsets `byte_offsets`.  
`(acc-*)`, `(lacc-*)`: Stores ("scatters") the vector `vals` to memory locations addressed by the the accessor and byte-offsets `byte_offsets`.  
`(slm-*)`: Stores ("scatters") the vector `vals` to shared local memory locations addressed by `byte_offsets`.  
The parameter `byte_offsets` has is a vector of integer 64-bit type elements for `(usm-*)`, 32-bit integer elements for `(lacc-*)` and `(slm-*)`, 32-bit integer elements for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit integer elements for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a `simd_mask`. If some element in `pred` is zero, then the store to the corresponding memory location is skipped.  
The optional [compile-time properties](#compile-time-properties) list `props` may specify `alignment` and/or `cache-hints`. The cache-hints are ignored for `(lacc-*)` and `(slm-*)` functions.

## atomic_update(...)

### atomic_update() with 0 operands (inc, dec, load)
```C++
// Atomic update the USM memory locations - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename PropertyListT = empty_properties_t>
/*usm-0op-1*/ simd<T, N> atomic_update(T *p, simd<Toffset, N> byte_offset, simd_mask<N> mask, props = {});
/*usm-0op-2*/ simd<T, N> atomic_update(T *p, simd<Toffset, N> byte_offset,props = {});

// Similar to (usm-0op-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename OffsetObjT, typename RegionT,
          typename PropertyListT = detail::empty_properties_t>
/*usm-0op-3*/ simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, RegionT> byte_offset, simd_mask<N> mask, props = {});
/*usm-0op-4*/simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, RegionT> byte_offset, props = {});


// Atomic update the memory locations referenced by device-accessor - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-0op-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd_mask<N> mask, props = {});
/*acc-0op-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       props = {});

// Similar to (acc-0op-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename OffsetObjT, typename AccessorT, typename RegionT,
          typename PropertyListT = empty_properties_t>
/*acc-0op-3*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset,
                                       simd_mask<N> mask, props = {});
/*acc-0op-4*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset,
                                       props = {});


// Atomic update the memory locations referenced by local-accessor (SLM) - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-0op-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd_mask<1> pred = 1);

// Atomic update the shared local memory (SLM) - zero operands (dec, load, etc.).
template <atomic_op Op, typename T, int N>
/*slm-0op-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd_mask<N> mask = 1);
```

### atomic_update() with 1 operands (*add, *sub, *min, *max, bit_or/xor/and, store, xchg)
```C++
// Atomic update the USM memory locations - 1 operand (add, max, etc.).
/*usm-1op-1*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*usm-1op-2*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, props = {});
// Similar to (usm-1op-1,2), but `byte_offset` is `simd_view`.
/*usm-1op-3*/ simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*usm-1op-4*/ simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
                                       simd<T, N> src0, props = {});


// Atomic update the memory locations referenced by device-accessor - 1 operand (add, max, etc.).
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-1op-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*acc-1op-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, props = {});

// Similar to (acc-1op-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename OffsetObjT, typename AccessorT,
          typename RegionT, typename PropertyListT = empty_properties_t>
/*acc-1op-3*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset,
                                       simd<T, N> src0, simd_mask<N> mask, props = {});
/*acc-1op-4*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset,
                                       simd<T, N> src0, props = {});

// Atomic update the memory locations referenced by local-accessor (SLM) - one operand (add, max, etc.).
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-1op-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd<T, N> src0, simd_mask<1> pred = 1);

// Atomic update the shared local memory (SLM) - one operand (add, max etc.).
template <atomic_op Op, typename T, int N>
/*slm-1op-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd<T, N> src0, simd_mask<N> mask = 1);
```
### atomic_update() with 2 operands (cmpxchg, fcmpxchg)
```C++
// Atomic update the USM memory locations - 2 operand: *cmpxchg.
/*usm-2op-1*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*usm-2op-2*/ simd<T, N> atomic_update(T *ptr, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});
// Similar to (usm-2op-1,2), but `byte_offset` is `simd_view`.
/*usm-2op-3*/ simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*usm-2op-4*/ simd<T, N> atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});


// Atomic update the memory locations referenced by device-accessor - 2 operands: *cmpxchg.
template <atomic_op Op, typename T, int N, typename Toffset, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-2op-1*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*acc-2op-2*/ simd<T, N> atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});

// Similar to (acc-2op-1,2), but `byte_offset` is `simd_view`.
template <atomic_op Op, typename T, int N, typename OffsetObjT, typename AccessorT,
          typename RegionT, typename PropertyListT = empty_properties_t>
/*acc-2op-3*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset, 
                                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask, props = {});
/*acc-2op-4*/ simd<T, N> atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionT> byte_offset,
                                       simd<T, N> src0, simd<T, N> src1, props = {});

// Atomic update the memory locations referenced by local-accessor (SLM) - two operands: *cmpxchg.
template <atomic_op Op, typename T, int N, typename AccessorT>
/*lacc-2op-1*/ simd<T, N> atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
                                        simd<T, N> src0, simd<T, N> src1, simd_mask<1> pred = 1);

// Atomic update the shared local memory (SLM) - two operands: *cmpxchg.
template <atomic_op Op, typename T, int N>
/*slm-2op-1*/ simd<T, N> slm_atomic_update(simd<uint32_t, N> byte_offset,
                                           simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask = 1);
```
### Description
`(usm-*)`: Atomically updates the global memory locations addressed by the base USM pointer `ptr` and byte-offsets `byte_offset`.  
`(acc-*)`, `(lacc-*)`: Atomically updates the memory locations addressed by the the accessor and byte-offsets `byte_offset`.  
`(slm-*)`: Atomically updates the shared memory locations addressed by `byte_offset`.  
The parameter `byte_offset` has is a vector of integer 64-bit type elements for `(usm-*)`, 32-bit integer elements for `(lacc-*)` and `(slm-*)`, 32-bit integer elements for `(acc-*)` in [stateful](#statelessstateful-memory-mode) mode, and 64-bit integer elements for `(acc-*)` in [stateless](#statelessstateful-memory-mode) mode.  
The optional parameter `pred` provides a `simd_mask`. If some element in `pred` is zero, then the corresponding memory location is not updated.  
`(usm-*)`, `(acc-*)`: The optional [compile-time properties](#compile-time-properties) list `props` may specify `cache-hints`.

## prefetch(...)
```C++
template <typename T, int N, int VS, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-1*/ void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-2*/ void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
                        PropertyListT props = {});

// The next 2 are similar (usm-1,2). `VS` parameter is set to 1,
// which allows callers to omit explicit specification of `T` and `N` params.
template <typename T, int N, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-3*/ void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
                   simd_mask<N> mask, PropertyListT props = {});
/*usm-4*/ void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
                        PropertyListT props = {});

// The next 2 are similar to (usm-1,2), added to handle `byte_offsets` in `simd_view` form.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/*usm-5*/ void prefetch(const T *p, OffsetSimdViewT byte_offsets,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*usm-6*/ void prefetch(const T *p, OffsetSimdViewT byte_offsets,
                        PropertyListT props = {});

// The next functions perform transposed 1-channel prefetch - prefetch linear block of code.
template <typename T, int VS = 1, typename OffsetT, typename PropertyListT = empty_properties_t>
/*usm-7*/ void prefetch(const T *p, OffsetT byte_offset,
                        simd_mask<1> mask, PropertyListT props = {});
/*usm-8*/ void prefetch(const T *p, OffsetT byte_offset,
                        PropertyListT props = {});
template <typename T, int VS = 1, typename PropertyListT = empty_properties_t>
/*usm-9*/ void prefetch(const T *p, simd_mask<1> mask, PropertyListT props = {});
/*usm-10*/ void prefetch(const T *p, PropertyListT props = {});


template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-1*/ void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-2*/ void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
                        PropertyListT props = {});

// The next 2 are similar (acc-1,2). `VS` parameter is set to 1,
// which allows callers to omit explicit specification of `T` and `N` params.
template <typename T, int N, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-3*/ void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
                        simd_mask<N> mask, PropertyListT props = {});
/*acc-4*/ void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
                        PropertyListT props = {});

// The next 2 are similar to (acc-1,2), added to handle `byte_offsets` in `simd_view` form.
template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
          typename AccessorT, typename PropertyListT = empty_properties_t>
/*acc-5*/ void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
                        simd_mask<N / VS> mask, PropertyListT props = {});
/*acc-6*/ void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
                        PropertyListT props = {});

/// The next functions perform transposed 1-channel prefetch.
template <typename T, int VS = 1, typename AccessorT, typename OffsetT,
          typename PropertyListT = empty_properties_t>
/*acc-7*/ void prefetch(AccessorT acc, OffsetT byte_offset, simd_mask<1> mask,
                        PropertyListT props = {});
/*acc-8*/ void prefetch(AccessorT acc, OffsetT byte_offset,
                        PropertyListT props = {});
template <typename T, int VS = 1, typename AccessorT,
          typename PropertyListT = empty_properties_t>
/*acc-9*/ void prefetch(AccessorT acc, simd_mask<1> mask, PropertyListT props = {});
/*acc-10*/ void prefetch(AccessorT acc, PropertyListT props = {});
```
### Description
`(usm-1,2,3,4,5,6)`: Prefetches the memory locations addressed by the base USM pointer `ptr` or the accessor `acc` and the vector of 64-bit byte-offsets `byte_offsets`.

`(acc-1,2,3,4,5,6)`: Prefetches the memory locations addressed by the base USM pointer `ptr` or the accessor `acc` and the vector of byte-offsets `byte_offsets`.
The `byte_offsets` is a vector of 32-bit integers elements in [stateful](#statelessstateful-memory-mode) mode, and it is a vector of 64-bit integer elements for in [stateless](#statelessstateful-memory-mode) mode.

`(usm-7,8,9,10)`: Prefetches a linear block of memory addressed by the base USM pointer `ptr` or the accessor `acc` and the optional 64-bit `byte-offset`.

`(acc-7,8,9,10)`: Prefetches a linear block of memory addressed by the base USM pointer `ptr` or the accessor `acc` and the optional `byte-offset` parameter, which is 32-bit in [stateful](#statelessstateful-memory-mode) mode, and 32-bit in [stateless](#statelessstateful-memory-mode) mode.


`(usm-1,2,3,4,5,6)`, `(acc-1,2,3,4,5,6)`: The optional parameter `mask` provides a `simd_mask`. If some element in `mask` is zero, then the corresponding memory location is not prefetched.  
`(usm-7,8,9,10)`, `(acc-7,8,9,10)`: The optional parameter `mask` provides 1-element
`simd_mask`. If it is zero, then the whole prefetch operation is skipped.

`(usm-*)`, `(acc-*)`: The [compile-time properties](#compile-time-properties) list `props` must specify `cache-hints`.


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
