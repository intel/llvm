# SYCLcompat

SYCLcompat is a header-only library that intends to help developers familiar
with other heterogeneous programming models (such as OpenMP, CUDA or HIP) to
familiarize themselves with the SYCL programming API while porting their
existing codes. Compatibility tools can also benefit from the reduced API size
when converting legacy codebases.

SYCLcompat provides:

* A high-level API that provides closer semantics to other programming models,
simplifying line by line conversions.
* Alternative submission APIs that encapusulate SYCL-specific "queue" and
"event" APIs for easier reference.
* Ability to gradually introduce other SYCL concepts as the user familiarises
themselves with the core SYCL API.
* Clear distinction between core SYCL API and the compatibility interface via
separate namespaces.

## Important Disclaimer

SYCLcompat state is experimental. Its functionalities have been implemented but
are not assured to remain consistent in the future. The API is subject to
potential disruptions with new updates, so exercise caution when using it.

## Notice

Copyright © 2023-2023 Codeplay Software Limited. All rights reserved.

Khronos(R) is a registered trademark and SYCL(TM) and SPIR(TM) are trademarks of
The Khronos Group Inc. OpenCL(TM) is a trademark of Apple Inc. used by
permission by Khronos.

## Support

SYCLcompat depends on specific oneAPI DPC++ compiler extensions that may not be
available to all the SYCL 2020 specification implementations.

Specifically, this library depends on the following SYCL extensions:

* [sycl_ext_oneapi_local_memory](
    ../extensions/supported/sycl_ext_oneapi_local_memory.asciidoc)
* [sycl_ext_oneapi_complex](
    ../extensions/experimental/sycl_ext_oneapi_complex.asciidoc)
* [sycl_ext_oneapi_free_function_queries](
    ../extensions/experimental/sycl_ext_oneapi_free_function_queries.asciidoc)
* [sycl_ext_oneapi_assert](
    ../extensions/supported/sycl_ext_oneapi_assert.asciidoc)
* [sycl_ext_oneapi_enqueue_barrier](
    ../extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc)
* [sycl_ext_oneapi_usm_device_read_only](../extensions/supported/sycl_ext_oneapi_usm_device_read_only.asciidoc)

## Usage

All functionality is available under the `syclcompat::` namespace, imported
through the main header, `syclcompat.hpp`. Note that `syclcompat.hpp` does not
import the <sycl/sycl.hpp> header.

``` cpp
#include <syclcompat.hpp>
```

This document presents the public API under the [Features](#features) section,
and provides a working [Sample code](#sample-code) using this library. Refer to
those to learn to use the library.

## Features

### dim3

SYCLcompat provides a `dim3` class akin to that of CUDA or HIP programming
models. `dim3` encapsulates other languages iteration spaces that are
represented with coordinate letters (x, y, z).

```cpp
namespace syclcompat {

class dim3 {
public:
  const size_t x, y, z;
  constexpr dim3(const sycl::range<3> &r);
  constexpr dim3(const sycl::range<2> &r);
  constexpr dim3(const sycl::range<1> &r);
  constexpr dim3(size_t x, size_t y = 1, size_t z = 1);

  constexpr size_t size();

  operator sycl::range<3>();
  operator sycl::range<2>();
  operator sycl::range<1>();
};

// Element-wise operators
dim3 operator*(const dim3 &a, const dim3 &b);
dim3 operator+(const dim3 &a, const dim3 &b);
dim3 operator-(const dim3 &a, const dim3 &b);

} // syclcompat
```

In SYCL, the fastest-moving dimension is the one with the highest index, e.g. in
a SYCL 2D range iteration space, there are two dimensions, 0 and 1, and 1 will
be the one that "moves faster". The compatibility headers for SYCL offer a
number of convenience functions that help the mapping between xyz-based
coordinates to SYCL iteration spaces in the different scopes available. In
addition to the global range, the following helper functions are also provided:

``` c++
namespace syclcompat {

namespace local_id {
size_t x();
size_t y();
size_t z();
} // namespace local_id

namespace local_range {
size_t x();
size_t y();
size_t z();
} // namespace local_range

namespace work_group_id {
size_t x();
size_t y();
size_t z();
} // namespace work_group_id

namespace work_group_range {
size_t x();
size_t y();
size_t z();
} // namespace work_group_range

namespace global_range {
size_t x();
size_t y();
size_t z();
} // namespace global_range

namespace global_id {
size_t x();
size_t y();
size_t z();
} // namespace global_id

} // syclcompat
```

These translate any kernel dimensions from one convention to the other. An
example of an equivalent SYCL call for a 3D kernel using `compat` is
`syclcompat::global_id::x() == get_global_id(2)`.

### Local Memory

When using `compat` functions, there are two distinct interfaces to allocate
device local memory. The first interface uses the _sycl_ext_oneapi_local_memory_
extension to leverage local memory defined at compile time.
_sycl_ext_oneapi_local_memory_ is accessed through the following wrapper:

``` c++
namespace syclcompat {

template <typename AllocT> auto *local_mem();

} // syclcompat
```

`syclcompat::local_mem<AllocT>()` can be used as illustrated in the example
below.

```c++
// Sample kernel
using namespace syclcompat;
template <int BLOCK_SIZE>
void local_mem_2d(int *d_A) {
  // Local memory extension wrapper, size defined at compile-time
  auto As = local_mem<int[BLOCK_SIZE][BLOCK_SIZE]>();
  int id_x = local_id::x();
  int id_y = local_id::y();
  As[id_y][id_x] = id_x * BLOCK_SIZE + id_y;
  wg_barrier();
  int val = As[BLOCK_SIZE - id_y - 1][BLOCK_SIZE - id_x - 1];
  d_A[global_id::y() * BLOCK_SIZE + global_id::x()] = val;
}
```

The second interface allows users to allocate device local memory at runtime.
SYCLcompat provides this functionality through its kernel launch interface,
`launch<function>`, defined in the following section.

### launch<function>

SYCLcompat provides a kernel `launch` interface which accepts a function that
executes on the device (a.k.a "kernel") instead of a lambda/functor. It can be
called either by using a pair of "teams"/"blocks" and "threads", from
OpenMP/CUDA terminology, or using a `sycl::nd_range`. The interface accepts a
device _function_ with the use of an `auto F` template parameter, and a variadic
`Args` for the function's arguments.

Various overloads for `launch<function>` exist to permit the user to launch on a
specific `queue`, or to define dynamically sized device local memory.

``` c++
namespace syclcompat {

template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads, Args... args);

template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range, Args... args);

template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range,
                   sycl::queue q, Args... args);

template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads,
                   sycl::queue q, Args... args);

template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range, size_t mem_size,
                   sycl::queue q, Args... args);

template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range, size_t mem_size,
                   Args... args);

template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads,
                   size_t mem_size, sycl::queue q, Args... args);

template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads,
                   size_t mem_size, Args... args);

} // syclcompat
```

For example, if the user had an existing function named `vectorAdd` to execute
on a device such as follows:

``` c++
void vectorAdd(const float *A, const float *B, float *C, int n);
```

using SYCLcompat, the user can call it as follows:

``` c++
syclcompat::launch<vectorAdd>(blocksPerGrid, threadsPerBlock, d_A, d_B, d_C, n);
```

which would be equivalent to the following call using a `sycl::nd_range`:

``` c++
auto range = sycl::nd_range<3>{blocksPerGrid * threadsPerBlock,
                               threadsPerBlock};
syclcompat::launch<vectorAdd>(range, d_A, d_B, d_C, n);
```

For dynamic local memory allocation, `launch<function>` injects a pointer to a
local `char *` accessor of `mem_size` as the last argument of the kernel
function. For example, the previous function named `vectorAdd` can be modified
with the following signature, which adds a `char *` pointer to access local
memory inside the kernel:

``` c++
void vectorAdd(const float *A, const float *B, float *C, int n,
               char *local_mem);
```

Then, `vectorAdd` can be launched like this:

``` c++
syclcompat::launch<vectorAdd>(blocksPerGrid, threadsPerBlock, mem_size, d_A,
                              d_B, d_C, n);
```

or this:

``` c++
auto range = sycl::nd_range<3>{globalSize, localSize};
syclcompat::launch<vectorAdd>(range, mem_size, d_A, d_B, d_C, n);
```

This `launch` interface allows users to define an internal memory pool, or
scratchpad, that can then be reinterpreted as the datatype required by the user
within the kernel function.

### Utilities

SYCLcompat introduces a set of utility functions designed to streamline the
usage of the library and its `launch<function>` mechanism.

The first utility function is `syclcompat::wg_barrier()`, which provides a
concise work-group barrier. `syclcompat::wg_barrier()` uses the
_SYCL_INTEL_free_function_queries_ extension to provide this functionality.

The second utility function, `syclcompat::compute_nd_range`, ensures that the
provided global size and work group sizes are appropriate for a given
dimensionality, and that global size is rounded up to a multiple of the work
group size in each dimension.

```c++
namespace syclcompat {

void wg_barrier();

template <int Dim>
sycl::nd_range<Dim> compute_nd_range(sycl::range<Dim> global_size_in,
                                     sycl::range<Dim> work_group_size);
sycl::nd_range<1> compute_nd_range(int global_size_in, int work_group_size);

} // syclcompat
```

### Queues

The design for this library assumes _in-order_ queues
(`sycl::property::queue::in_order()`).

Many of the APIs accept an optional `queue` parameter, and this can be an
out-of-order queue, either created manually or retrieved via a call to
`syclcompat::create_queue()`, specifying `false` for the `in_order` parameter.

```c++
namespace syclcompat {

sycl::queue create_queue(bool print_on_async_exceptions = false,
                         bool in_order = true);

} // syclcompat
```

However, SYCLcompat does not implement any mechanisms to deal with this case.
The rationale for this is that a user wanting the full power of SYCL's
dependency management shouldn't be using the this library. As such, support for
out-of-order queues is very limited. The only way to safely use an out-of-order
queue at present is to explicitly `q.wait()` or `e.wait()` where `e` is the
`sycl::event` returned through a `syclcompat::async` API.

To facilitate machine translation from other heterogeneous programming models to
SYCL, SYCLcompat provides the following pointer aliases for `sycl::event` and
`sycl::queue`, and the function `destroy_event` which destroys an `event_ptr`
allocated on the heap.

``` c++
namespace syclcompat {

using event_ptr = sycl::event *;

using queue_ptr = sycl::queue *;

static void destroy_event(event_ptr event);

} // syclcompat
```

### Memory Allocation

This library provides interfaces to allocate memory to be accessed within kernel
functions and on the host. The `syclcompat::malloc` function allocates device
USM memory, the `syclcompat::malloc_host` function allocates host USM memory,
and the `syclcompat::malloc_shared` function allocates shared USM memory.

In each case we provide a template and non-templated interface for allocating
memory, taking the number of elements or number of bytes respectively.

The interface includes both synchronous and asynchronous `malloc`, `memcpy`,
`memset`, `fill`, and `free` operations.

There is a helper class `pointer_attributes` to query allocation type for memory
pointers using SYCLcompat, through `sycl::usm::alloc` and
`sycl::get_pointer_device`.

``` c++
namespace syclcompat {

// Expects number of elements
template <typename T>
T *malloc(size_t count, sycl::queue q = get_default_queue());
template <typename T>
T *malloc_host(size_t count, sycl::queue q = get_default_queue());
template <typename T>
T *malloc_shared(size_t count, sycl::queue q = get_default_queue());

// Expects size of the memory in bytes
void *malloc(size_t num_bytes, sycl::queue q = get_default_queue());
void *malloc_host(size_t num_bytes, sycl::queue q = get_default_queue());
void *malloc_shared(size_t num_bytes, sycl::queue q = get_default_queue());

// 2D, 3D memory allocation wrappers
void *malloc(size_t &pitch, size_t x, size_t y,
             sycl::queue q = get_default_queue())
pitched_data malloc(sycl::range<3> size, sycl::queue q = get_default_queue());

// Blocking memcpy
void memcpy(void *to_ptr, const void *from_ptr, size_t size,
            sycl::queue q = get_default_queue());
void memcpy(T *to_ptr, const T *from_ptr, size_t count,
            sycl::queue q = get_default_queue());
void memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
            size_t from_pitch, size_t x, size_t y,
            sycl::queue q = get_default_queue()); // 2D matrix
void memcpy(pitched_data to, sycl::id<3> to_pos,
            pitched_data from, sycl::id<3> from_pos,
            sycl::range<3> size,
            sycl::queue q = get_default_queue()); // 3D matrix

// Non-blocking memcpy
sycl::event memcpy_async(void *to_ptr, const void *from_ptr, size_t size,
                         sycl::queue q = get_default_queue());
template <typename T>
sycl::event memcpy_async(T *to_ptr, T void *from_ptr, size_t count,
                         sycl::queue q = get_default_queue());
sycl::event memcpy_async(void *to_ptr, size_t to_pitch,
                         const void *from_ptr, size_t from_pitch,
                         size_t x, size_t y,
                         sycl::queue q = get_default_queue()); // 2D matrix
sycl::event memcpy_async(pitched_data to, sycl::id<3> to_pos,
                         pitched_data from, sycl::id<3> from_pos,
                         sycl::range<3> size,
                         sycl::queue q = get_default_queue()); // 3D matrix

// Fill
template <class T>
void fill(void *dev_ptr, const T &pattern, size_t count,
          sycl::queue q = get_default_queue());
template <typename T>
sycl::event fill_async(void *dev_ptr, const T &pattern,
                       size_t count, sycl::queue q = get_default_queue());

// Memset
void memset(void *dev_ptr, int value, size_t size,
                   sycl::queue q = get_default_queue());
void memset(void *ptr, size_t pitch, int val, size_t x, size_t y,
            sycl::queue q = get_default_queue()); // 2D matrix
void memset(pitched_data pitch, int val, sycl::range<3> size,
                          sycl::queue q = get_default_queue()); // 3D matrix
sycl::event memset_async(void *dev_ptr, int value, size_t size,
                         sycl::queue q = get_default_queue());
sycl::event memset_async(void *ptr, size_t pitch, int val,
                         size_t x, size_t y,
                         sycl::queue q = get_default_queue()); // 2D matrix
sycl::event memset_async(pitched_data pitch, int val,
                         sycl::range<3> size,
                         sycl::queue q = get_default_queue()); // 3D matrix

void free(void *ptr, sycl::queue q = get_default_queue());
sycl::event free_async(const std::vector<void *> &pointers,
                       const std::vector<sycl::event> &events,
                       sycl::queue q = get_default_queue());

// Queries pointer allocation type
class pointer_attributes {
public:
  void init(const void *ptr, sycl::queue q = get_default_queue());
  sycl::usm::alloc get_memory_type();
  const void *get_device_pointer();
  const void *get_host_pointer();
  bool is_memory_shared();
  unsigned int get_device_id();
};

} // syclcompat
```

Finally, the class `pitched_data`, which manages memory allocation for 3D
spaces, padded to avoid uncoalesced memory accesses.

```c++
namespace syclcompat {

class pitched_data {
public:
  pitched_data();
  pitched_data(void *data, size_t pitch, size_t x, size_t y);

  void *get_data_ptr();
  size_t get_pitch();
  size_t get_x();
  size_t get_y();

  void set_data_ptr(void *data);
  void set_pitch(size_t pitch);
  void set_x(size_t x);
  void set_y(size_t y);
};

} // syclcompat
```

There are various helper classes and aliases defined within SYCLcompat to
encapsulate and define memory operations and objects. These classes and aliases
are primarily designed to assist with machine translation from other
heterogeneous programming models.

The wrapper class `device_memory` provides a unified representation for device
memory in various regions. The class provides methods to allocate memory for the
object (`init()`) and access the underlying memory in various ways (`get_ptr()`,
`get_access()`, `operator[]`). Aliases for global and USM shared specializations
are provided.

The `memory_traits` class is provided as a traits helper for `device_memory`.
The `accessor` class template provides a 2D or 3D `sycl::accessor`-like wrapper
around raw pointers.

```c++
namespace syclcompat {

enum class memory_region {
  global = 0, // device global memory
  constant,   // device read-only memory
  local,      // device local memory
  usm_shared, // memory which can be accessed by host and device
};

using byte_t = uint8_t;

enum class target { device, local };

template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr sycl::access::address_space asp =
      (Memory == memory_region::local)
          ? sycl::access::address_space::local_space
          : sycl::access::address_space::global_space;
  static constexpr target target =
      (Memory == memory_region::local)
          ? target::local
          : target::device;
  static constexpr sycl::access_mode mode =
      (Memory == memory_region::constant)
          ? sycl::access_mode::read
          : sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional_t<Memory == constant, const T, T>;
  using value_t = typename std::remove_cv_t<T>;
  template <size_t Dimension = 1>
  using accessor_t = typename std::conditional_t<
      target == target::local,
      sycl::local_accessor<T, Dimension>,
      sycl::accessor<T, Dimension, mode>>;
  using pointer_t = T *;
};

template <class T, memory_region Memory, size_t Dimension> class device_memory {
public:
  using accessor_t =
      typename memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename memory_traits<Memory, T>::value_t;
  using syclcompat_accessor_t =
      syclcompat::accessor<T, Memory, Dimension>;

  device_memory();

  device_memory(const sycl::range<Dimension> &in_range,
                std::initializer_list<value_t> &&init_list);

  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list);

  device_memory(const sycl::range<Dimension> &range_in);

  // Variadic constructor taking 1, 2 or 3 integers to be interpreted as a
  // sycl::range<Dim>.
  template <class... Args>
  device_memory(Args... Arguments);

  ~device_memory();

  // Allocate memory with default queue, and init memory if has initial value.
  void init();
  // Allocate memory with specified queue, and init memory if has initial
  // value.
  void init(sycl::queue q);

  // The variable is assigned to a device pointer.
  void assign(value_t *src, size_t size);

  // Get memory pointer of the memory object, which is virtual pointer when
  // usm is not used, and device pointer when usm is used.
  value_t *get_ptr();
  // Get memory pointer of the memory object, which is virtual pointer when
  // usm is not used, and device pointer when usm is used.
  value_t *get_ptr(sycl::queue q);

  // Get the device memory object size in bytes.
  size_t get_size();

  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index);

  // Get accessor with dimension info for the device memory object
  // when usm is used and dimension is greater than 1.
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, syclcompat_accessor_t>::type
  get_access(sycl::handler &cgh);
};


template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
  using accessor_t =
      typename memory_traits<Memory, T>::template accessor_t<0>;
  device_memory(const value_t &val);
  device_memory();
};

template <class T, size_t Dimension>
using global_memory = device_memory<T, memory_region::global, Dimension>;
template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = device_memory<T, memory_region::usm_shared, Dimension>;


template <class T, memory_region Memory, size_t Dimension> class accessor;

template <class T, memory_region Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;

  accessor(pointer_t data, const sycl::range<3> &in_range);
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != memory_region::local,
                                   const accessor_t>::type &acc);
  accessor(const accessor_t &acc, const sycl::range<3> &in_range);

  accessor<T, Memory, 2> operator[](size_t index) const;

  pointer_t get_ptr() const;

};

template <class T, memory_region Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;

  accessor(pointer_t data, const sycl::range<2> &in_range);
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != memory_region::local,
                                   const accessor_t>::type &acc);
  accessor(const accessor_t &acc, const sycl::range<2> &in_range);

  pointer_t operator[](size_t index);

  pointer_t get_ptr() const;
};

} // syclcompat
```

### Device Information

`sycl::device` properties are encapsulated using the `device_info` helper class.
The class is meant to be constructed and used through the extended device
implemented in SYCLcompat.

This is the synopsis of `device_info`:

```c++
class device_info {
public:
  const char *get_name();
  char *get_name();
  template <typename WorkItemSizesTy = sycl::id<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() const;

  template <typename WorkItemSizesTy = sycl::id<3>,
          std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                std::is_same_v<WorkItemSizesTy, int *>,
                            int> = 0>
  auto get_max_work_item_sizes() const;
  int get_major_version() const;
  int get_minor_version() const;
  int get_integrated() const;
  int get_max_clock_frequency() const;
  int get_max_compute_units() const;
  int get_max_work_group_size() const;
  int get_max_sub_group_size() const;
  int get_max_work_items_per_compute_unit() const;
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size() const;
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size();
  size_t get_global_mem_size() const;
  size_t get_local_mem_size() const;

void set_name(const char *name);
  void set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes);
  void set_major_version(int major);
  void set_minor_version(int minor);
  void set_integrated(int integrated);
  void set_max_clock_frequency(int frequency);
  void set_max_compute_units(int max_compute_units);
  void set_global_mem_size(size_t global_mem_size);
  void set_local_mem_size(size_t local_mem_size);
  void set_max_work_group_size(int max_work_group_size);
  void set_max_sub_group_size(int max_sub_group_size);
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit);
  void set_max_nd_range_size(int max_nd_range_size[]);
};
```

### Device Management

Multiple SYCL functionalities are exposed through utility functions to manage
the current `sycl::device`, `sycl::queue`, and `sycl::context`, exposed as
follows:

```c++
namespace syclcompat {

// Util function to create a new queue for the current device
sycl::queue create_queue(bool print_on_async_exceptions = false,
                         bool in_order = true);

// Util function to get the default queue of current device in
// device manager.
sycl::queue get_default_queue();

// Util function to wait for the queued kernels.
void wait(sycl::queue q = get_default_queue());

// Util function to wait for the queued kernels and throw unhandled errors.
void wait_and_throw(sycl::queue q = get_default_queue());

// Util function to get the id of current device in
// device manager.
unsigned int get_current_device_id();

// Util function to get the current device.
device_ext &get_current_device();

// Util function to get a device by id.
device_ext &get_device(unsigned int id);

// Util function to get the context of the default queue of current
// device in device manager.
sycl::context get_default_context();

// Util function to get a CPU device.
device_ext &cpu_device();

// Util function to select a device by its id
unsigned int select_device(unsigned int id);

} // syclcompat
```

The exposed functionalities include creation and destruction of queues, through
`syclcompat::create_queue` and `syclcompat::destroy_queue`, and providing the
ability to wait for submitted kernels using `syclcompat::wait` or
`syclcompat::wait_and_throw`. Any async errors will be output to `stderr` if
`print_on_async_exceptions`, and will have the default behavior otherwise, which calls `std:terminate`. Synchronous exceptions have to be managed
by users independently of what is set in this parameter.

Devices are managed through a helper class, `device_ext`. The `device_ext` class
associates a vector of `sycl::queues` with its `sycl::device`. The `device_ext`
destructor waits on a set of `sycl::event` which can be added to via
`add_event`. This is used, for example, to implement `syclcompat::free_async` to
schedule release of memory after a kernel or `mempcy`. SYCL device properties
can be queried through `device_ext` as well.

The class is exposed as follows:

```c++
namespace syclcompat {

class device_ext : public sycl::device {
  device_ext();
  device_ext(const sycl::device &base);
  ~device_ext();

  bool is_native_host_atomic_supported();
  int get_major_version();
  int get_minor_version();
  int get_max_compute_units();
  int get_max_clock_frequency();
  int get_integrated();
  void get_device_info(device_info &out);

  device_info get_device_info();
  void reset();

  sycl::queue *default_queue();
  void queues_wait_and_throw();
  sycl::queue *create_queue(bool print_on_async_exceptions = false,
                            bool in_order = true);
  void destroy_queue(sycl::queue *&queue);
  void set_saved_queue(sycl::queue *q);
  sycl::queue *get_saved_queue();
  sycl::context get_context();
};

} // syclcompat
```

#### Multiple devices

SYCLcompat allows you to manage multiple devices through
`syclcompat::select_device` and `syclcompat::create_queue`. The library uses the
default SYCL device (i.e. the device returned by `sycl::default_selector_v`) as
the default device, and exposes all other devices available on the system
through the `syclcompat::select_device(unsigned int id)` member function.

The interface uses the `syclcompat::device_ext::get_current_device_id()` to get
the current CPU thread, and returns the associated device stored internally as a
map with that thread. The map is constructed using calls to
`syclcompat::select_device(unsigned int id)`. Any thread which hasn't used this
member function to select a device will be given the default device. Note that
this implies multiple threads on a single device by default.

Be aware that targetting multiple devices may lead to unintended behavior caused
by developers, as SYCLcompat does not implement a mechanism to warn when the
wrong queue is used as an argument in any of the member functions of the
`syclcompat` namespace.

#### Atomic Operations

SYCLcompat provides an interface for common atomic operations (`add`, `sub`,
`and`, `or`, `xor`, `min`, `max`, `inc`, `dec`, `exchange`, `compare_exchange`).
While SYCL exposes atomic operations through member functions of
`sycl::atomic_ref`, this library provides access via functions taking a standard
pointer argument. Template arguments control the `sycl::memory_scope`,
`sycl::memory_order` and `sycl::access::address_space` of these atomic
operations. SYCLcompat also exposes overloads for these atomic functions which
take a runtime memoryScope argument. Every atomic operation is implemented via
an API function taking a raw pointer as the target. Additional overloads for
`syclcompat::compare_exchange_strong` are provided which take a
`sycl::multi_ptr` instead of a raw pointer. The type of the operand for most
atomic operations is defined as `syclcompat::type_identity_t<T>` to avoid
template deduction issues when an operand of a different type (e.g. double
literal) is supplied. Atomic addition and subtraction free functions make use of
`syclcompat::arith_t<T>` to differentiate between numeric and pointer
arithmetics.

The available operations are exposed as follows:

``` c++
namespace syclcompat {

template <class T> struct type_identity {
  using type = T;
};
template <class T> using type_identity_t = typename type_identity<T>::type;

template <typename T> struct arith {
  using type = std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
};
template <typename T> using arith_t = typename arith<T>::type;

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_add(T *addr, arith_t<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_sub(T *addr, arith_t<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_and(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_or(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_xor(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_min(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_fetch_max(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                      unsigned int operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
unsigned int atomic_fetch_compare_dec(unsigned int *addr,
                                      unsigned int operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_exchange(T *addr, type_identity<T> operand);

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, sycl::access::address_space::generic_space> addr,
    T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed);
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed);

} // namespace syclcompat
```

SYCLcompat also provides an atomic class with the `store`, `load`, `exchange`,
`compare_exchange_weak`, `fetch_add`, and `fetch_sub` operations. The atomic
class wrapper supports int, unsigned int, long, unsigned long, long long,
unsigned long long, float, double and pointer datatypes.

```cpp
namespace syclcompat {

template <typename T,
          sycl::memory_scope DefaultScope = sycl::memory_scope::system,
          sycl::memory_order DefaultOrder = sycl::memory_order::seq_cst,
          sycl::access::address_space Space =
              sycl::access::address_space::generic_space>
class atomic {
  static constexpr sycl::memory_order default_read_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope,
                       Space>::default_read_order;
  static constexpr sycl::memory_order default_write_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope,
                       Space>::default_write_order;
  static constexpr sycl::memory_scope default_scope = DefaultScope;
  static constexpr sycl::memory_order default_read_modify_write_order =
      DefaultOrder;

  constexpr atomic() noexcept = default;

  constexpr atomic(T d) noexcept;

  void store(T operand, sycl::memory_order memoryOrder = default_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept;

  T load(sycl::memory_order memoryOrder = default_read_order,
         sycl::memory_scope memoryScope = default_scope) const noexcept;

  T exchange(T operand,
             sycl::memory_order memoryOrder = default_read_modify_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept;

  bool compare_exchange_weak(
      T &expected, T desired, sycl::memory_order success,
      sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept;

  bool compare_exchange_weak(
      T &expected, T desired,
      sycl::memory_order memoryOrder = default_read_modify_write_order,
      sycl::memory_scope memoryScope = default_scope) noexcept;

  bool compare_exchange_strong(
      T &expected, T desired, sycl::memory_order success,
      sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept;

  bool compare_exchange_strong(
      T &expected, T desired,
      sycl::memory_order memoryOrder = default_read_modify_write_order,
      sycl::memory_scope memoryScope = default_scope) noexcept;

  T fetch_add(arith_t<T> operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope memoryScope = default_scope) noexcept;

  T fetch_sub(arith_t<T> operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope memoryScope = default_scope) noexcept;
};

} // namespace syclcompat
```

### Compatibility Utilities

This library provides a number of small compatibility utilities which exist to
facilitate machine translation of code from other programming models to SYCL.
These functions are part of the public API, but they are not expected to be
useful to developers writing their own code.

Functionality is provided to represent a pair of integers as a `double`.
`cast_ints_to_double(int, int)` returns a `double` containing the given integers
in the high & low 32-bits respectively. `cast_double_to_int` casts the high or
low 32-bits back into an integer.

`syclcompat::fast_length` provides a wrapper to SYCL's
`fast_length(sycl::vec<float,N>)` that accepts arguments for a C++ array and a
length.

`vectorized_max` and `vectorized_min` are binary operations returning the
max/min of two arguments, where each argument is treated as a `sycl::vec` type.
`vectorized_isgreater` performs elementwise `isgreater`, treating each argument
as a vector of elements, and returning `0` for vector components for which
`isgreater` is false, and `-1` when true.

`reverse_bits` reverses the bits of a 32-bit unsigned integer, `ffs` returns the
position of the first least significant set bit in an integer.
`byte_level_permute` returns a byte-permutation of two input unsigned integers,
with bytes selected according to a third unsigned integer argument.

There is also an `experimental::logical_group` class which allows
`sycl::sub_group`s to be further subdivided into 'logical' groups to perform
sub-group level operations. This class provides methods to get the local & group
id and range. The functions `select_from_sub_group`, `shift_sub_group_left`,
`shift_sub_group_right` and `permute_sub_group_by_xor` provide equivalent
functionality to `sycl::select_from_group`, `sycl::shift_group_left`,
`sycl::shift_group_right` and `sycl::permute_group_by_xor`, respectively.
However, they provide an optional argument to represent the `logical_group` size
(default 32).

The functions `cmul`,`cdiv`,`cabs`, and `conj` define complex math operations
which accept `sycl::vec<T,2>` arguments representing complex values.

```c++
namespace syclcompat {

inline int cast_double_to_int(double d, bool use_high32 = true);

inline double cast_ints_to_double(int high32, int low32);

inline float fast_length(const float *a, int len);

template <typename S, typename T> inline T vectorized_max(T a, T b);

template <typename S, typename T> inline T vectorized_min(T a, T b);

template <typename S, typename T> inline T vectorized_isgreater(T a, T b);

template <>
inline unsigned vectorized_isgreater<sycl::ushort2, unsigned>(unsigned a,
                                                              unsigned b);

template <typename T> inline T reverse_bits(T a);

inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s);

template <typename T> inline int ffs(T a);

template <typename T>
T select_from_sub_group(sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32);

template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32);

template <typename T>
T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32);

template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32);

template <typename T>
sycl::vec<T, 2> cmul(sycl::vec<T, 2> x, sycl::vec<T, 2> y);

template <typename T>
sycl::vec<T, 2> cdiv(sycl::vec<T, 2> x, sycl::vec<T, 2> y);

template <typename T> T cabs(sycl::vec<T, 2> x);

template <typename T> sycl::vec<T, 2> conj(sycl::vec<T, 2> x);

} // namespace syclcompat
```

The function `experimental::nd_range_barrier` synchronizes work items from all
work groups within a SYCL kernel. This is not officially supported by the SYCL
spec, and so should be used with caution.

```c++
namespace syclcompat {
namespace experimental {

template <int dimensions = 3>
inline void nd_range_barrier(
    sycl::nd_item<dimensions> item,
    sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter);

template <>
inline void nd_range_barrier(
    sycl::nd_item<1> item,
    sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter);

class logical_group {
public:
  logical_group(sycl::nd_item<3> item, sycl::group<3> parent_group,
                uint32_t size);
  uint32_t get_local_linear_id() const;
  uint32_t get_group_linear_id() const;
  uint32_t get_local_linear_range() const;
  uint32_t get_group_linear_range() const;
};

} // namespace experimental
} // namespace syclcompat
```

To assist machine translation, helper aliases are provided for inlining and
alignment attributes. The class template declarations `sycl_compat_kernel_name`
and `sycl_compat_kernel_scalar` are used to assist automatic generation of
kernel names during machine translation.

`get_sycl_language_version` returns an integer representing the version of the
SYCL spec supported by the current SYCL compiler.

``` c++
namespace syclcompat {

#define __sycl_compat_align__(n) __attribute__((aligned(n)))
#define __sycl_compat_inline__ __inline__ __attribute__((always_inline))

#define __sycl_compat_noinline__ __attribute__((noinline))

template <class... Args> class sycl_compat_kernel_name;
template <int Arg> class sycl_compat_kernel_scalar;

int get_sycl_language_version();

} // namespace syclcompat
```

#### Kernel Helper Functions

Kernel helper functions provide a structure `kernel_function_info` to keep SYCL
kernel information, and provide a utility function `get_kernel_function_info()`
to get the kernel information. Overloads are provided to allow either returning
a `kernel_function_info` object, or to return by pointer argument. In the
current version, `kernel_function_info` describes only maximum work-group size.

``` c++
namespace syclcompat {

struct kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function);
static kernel_function_info get_kernel_function_info(const void *function);
} // namespace syclcompat
```

## Sample Code

Below is a simple linear algebra sample, which computes `y = mx + b` implemented
using this library:

``` c++
#include <cassert>
#include <iostream>

#include <syclcompat.hpp>
#include <sycl/sycl.hpp>

/**
 * Slope intercept form of a straight line equation: Y = m * X + b
 */
template <int BLOCK_SIZE>
void slope_intercept(float *Y, float *X, float m, float b, size_t n) {

  // Block index
  size_t bx = syclcompat::work_group_id::x();
  // Thread index
  size_t tx = syclcompat::local_id::x();

  size_t i = bx * BLOCK_SIZE + tx;
  // or  i = syclcompat::global_id::x();
  if (i < n)
    Y[i] = m * X[i] + b;
}

void check_memory(void *ptr, std::string msg) {
  if (ptr == nullptr) {
    std::cerr << "Failed to allocate memory: " << msg << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * Program main
 */
int main(int argc, char **argv) {
  std::cout << "Simple Kernel example" << std::endl;

  constexpr size_t n_points = 32;
  constexpr float m = 1.5f;
  constexpr float b = 0.5f;

  int block_size = 32;
  if (block_size > syclcompat::get_current_device()
                       .get_info<sycl::info::device::max_work_group_size>())
    block_size = 16;

  std::cout << "block_size = " << block_size << ", n_points = " << n_points
            << std::endl;

  // Allocate host memory for vectors X and Y
  size_t mem_size = n_points * sizeof(float);
  float *h_X = (float *)syclcompat::malloc_host(mem_size);
  float *h_Y = (float *)syclcompat::malloc_host(mem_size);
  check_memory(h_X, "h_X allocation failed.");
  check_memory(h_Y, "h_Y allocation failed.");

  // Alternative templated allocation for the expected output
  float *h_expected = syclcompat::malloc_host<float>(n_points);
  check_memory(h_expected, "Not enough for h_expected.");

  // Initialize host memory & expected output
  for (size_t i = 0; i < n_points; i++) {
    h_X[i] = i + 1;
    h_expected[i] = m * h_X[i] + b;
  }

  // Allocate device memory
  float *d_X = (float *)syclcompat::malloc(mem_size);
  float *d_Y = (float *)syclcompat::malloc(mem_size);
  check_memory(d_X, "d_X allocation failed.");
  check_memory(d_Y, "d_Y allocation failed.");

  // copy host memory to device
  syclcompat::memcpy(d_X, h_X, mem_size);

  size_t threads = block_size;
  size_t grid = n_points / block_size;

  std::cout << "Computing result using SYCL Kernel... ";
  if (block_size == 16) {
    syclcompat::launch<slope_intercept<16>>(grid, threads, d_Y, d_X, m, b,
                                        n_points);
  } else {
    syclcompat::launch<slope_intercept<32>>(grid, threads, d_Y, d_X, m, b,
                                        n_points);
  }
  syclcompat::wait();
  std::cout << "DONE" << std::endl;

  // Async copy result from device to host
  syclcompat::memcpy_async(h_Y, d_Y, mem_size).wait();

  // Check output
  for (size_t i = 0; i < n_points; i++) {
    assert(h_Y[i] - h_expected[i] < 1e6);
  }

  // Clean up memory
  syclcompat::free(h_X);
  syclcompat::free(h_Y);
  syclcompat::free(h_expected);
  syclcompat::free(d_X);
  syclcompat::free(d_Y);

  return 0;
}
```

## Maintainers

To report problems with this library, please open a new issue with the [COMPAT]
tag at:

<https://github.com/intel/llvm/issues>

## Contributors

Alberto Cabrera, Codeplay \
Gordon Brown, Codeplay \
Joe Todd, Codeplay \
Pietro Ghiglio, Codeplay \
Ruyman Reyes, Codeplay/Intel

## Contributions

This library is licensed under the Apache 2.0 license. If you have an idea for a
new sample, different build system integration or even a fix for something that
is broken, please get in contact.
