# SYCLcompat

SYCLcompat is a header-only library that intends to help developers familiar
with other heterogeneous programming models (such as OpenMP, CUDA or HIP) to
familiarize themselves with the SYCL programming API while porting their
existing codes. Compatibility tools can also benefit from the reduced API size
when converting legacy codebases.

SYCLcompat provides:

* A high-level API that provides closer semantics to other programming models,
simplifying line by line conversions.
* Alternative submission APIs that encapsulate SYCL-specific "queue" and
"event" APIs for easier reference.
* Ability to gradually introduce other SYCL concepts as the user familiarizes
themselves with the core SYCL API.
* Clear distinction between core SYCL API and the compatibility interface via
separate namespaces.

## Notice

Copyright © 2023-2024 Codeplay Software Limited. All rights reserved.

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
    ../extensions/supported/sycl_ext_oneapi_free_function_queries.asciidoc)
* [sycl_ext_oneapi_assert](
    ../extensions/supported/sycl_ext_oneapi_assert.asciidoc)
* [sycl_ext_oneapi_enqueue_barrier](
    ../extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc)
* [sycl_ext_oneapi_usm_device_read_only](../extensions/supported/sycl_ext_oneapi_usm_device_read_only.asciidoc)

If available, the following extensions extend SYCLcompat functionality:

* [sycl_ext_intel_device_info](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_device_info.md) \[Optional\]
* [sycl_ext_oneapi_bfloat16_math_functions](../extensions/experimental/sycl_ext_oneapi_bfloat16_math_functions.asciidoc) \[Optional\]
* [sycl_ext_oneapi_max_work_group_query](
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_max_work_group_query.md)
  \[Optional\]

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

## Versioning

SYCLcompat adopts [semantic versioning](https://semver.org/)
(`major.minor.patch`) in a manner which aligns with oneAPI releases. Each oneAPI
product release has an associated SYCLcompat release. Between oneAPI releases,
there will be at most one `major` or `minor` bump. In other words, if a given
oneAPI release has SYCLcompat version `1.0.0`, the next release will have either
`1.1.0` or, if breaking changes have been made, `2.0.0`. This guarantee has
implications for code merged to the `sycl` branch, described below.

Between release cycles, ongoing updates to SYCLcompat (including possibly
breaking changes) are merged into DPC++ via PRs to the
[`sycl`](https://github.com/intel/llvm/tree/sycl) branch. If a PR introduces the
*first* breaking changes since the last release, that PR must bump to the next
`major` version. Otherwise, if the PR introduces *new functionality* and neither
the `major` nor `minor` have been bumped since the last release, it must bump to
the next `minor` release. If a PR introduces important bugfixes to existing
functionality, `patch` should be bumped, and there are no limits to how many
`patch` bumps can occur between release cycles.

### Release Process

Once all changes planned for a release have been merged, the release process is
defined as:

1. Check the `major.minor` version associated with the *previous* release.
2. Confirm the version bump process outlined above has been followed.
3. If no version bump has occurred since previous release, bump to next `minor`.
4. oneAPI release is delivered.
5. Tag the SYCLcompat release on DPC++ repo: `SYCLcompat-major.minor.0`.

### Deprecation Process/Breaking Changes

As outlined above, SYCLcompat may sometimes make API breaking changes, indicated
with a `major` version bump. Advanced notice (at least one major oneAPI release)
will be provided via a deprecation warning on the relevant APIs, indicating to
the user which alternative API should be used instead.

Note that SYCLcompat is currently in pre-release, and until version `1.0.0` we
do not consider our API to be stable, and may change it with shorter notice.

### Changelog

Since SYCLcompat releases are aligned with oneAPI product releases, the changelog for SYCLcompat is incorporated into [SYCL's Release Notes](https://github.com/intel/llvm/blob/sycl/sycl/ReleaseNotes.md).

### Experimental Namespace

SYCLcompat provides some new experimental features in the `syclcompat::experimental` namespace. This serves as a testing ground for new features which are expected to migrate to `syclcompat::` in time, but the developers do not guarantee either API stability or continued existence of these features; they may be modified or removed without notice. When features are migrated from `syclcompat::experimental` to `syclcompat::`, this will be treated as a `minor` version bump.

## Features

### dim3

SYCLcompat provides a `dim3` class akin to that of CUDA or HIP programming
models. `dim3` encapsulates other languages iteration spaces that are
represented with coordinate letters (x, y, z). In SYCL, the fastest-moving
dimension is the one with the highest index, e.g. in a SYCL 2D range iteration
space, there are two dimensions, 0 and 1, and 1 will be the one that "moves
faster". For CUDA/HIP, the convention is reversed: `x` is the dimension which
moves fastest. `syclcompat::dim3` follows this convention, so that
`syclcompat::dim3(32, 4)` is equivalent to `sycl::range<2>(4, 32)`, and
`syclcompat::dim3(32, 4, 2)` is equivalent to `sycl::range<3>(2, 4, 32)`.

```cpp
namespace syclcompat {

class dim3 {
public:
  unsigned int x, y, z;
  dim3(const sycl::range<3> &r);
  dim3(const sycl::range<2> &r);
  dim3(const sycl::range<1> &r);
  constexpr dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);

  constexpr size_t size();

  operator sycl::range<3>();
  operator sycl::range<2>();
  operator sycl::range<1>();
};

// Element-wise operators
inline dim3 operator*(const dim3 &a, const dim3 &b);
inline dim3 operator+(const dim3 &a, const dim3 &b);
inline dim3 operator-(const dim3 &a, const dim3 &b);

} // syclcompat
```

The compatibility headers for SYCL offer a number of convenience functions that
help the mapping between xyz-based coordinates to SYCL iteration spaces in the
different scopes available. In addition to the global range, the following
helper functions are also provided:

``` c++
namespace syclcompat {

namespace local_id {
inline size_t x();
inline size_t y();
inline size_t z();
} // namespace local_id

namespace local_range {
inline size_t x();
inline size_t y();
inline size_t z();
} // namespace local_range

namespace work_group_id {
inline size_t x();
inline size_t y();
inline size_t z();
} // namespace work_group_id

namespace work_group_range {
inline size_t x();
inline size_t y();
inline size_t z();
} // namespace work_group_range

namespace global_range {
inline size_t x();
inline size_t y();
inline size_t z();
} // namespace global_range

namespace global_id {
inline size_t x();
inline size_t y();
inline size_t z();
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

To launch a kernel with a specified sub-group size, overloads similar to above
`launch` functions are present in the `syclcompat::experimental` namespace,
which accept SubgroupSize as a template parameter and can be called as
`launch<Function, SubgroupSize>`

```cpp

template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(sycl::nd_range<3> launch_range, std::size_t local_memory_size,
       sycl::queue queue, Args... args);

template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(sycl::nd_range<Dim> launch_range, std::size_t local_memory_size,
       Args... args);

template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(::syclcompat::dim3 grid_dim, ::syclcompat::dim3 block_dim,
       std::size_t local_memory_size, Args... args);


template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(sycl::nd_range<3> launch_range, sycl::queue queue, 
       Args... args);

template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(sycl::nd_range<Dim> launch_range,
       Args... args);

template <auto F, int SubgroupSize, typename... Args>
sycl::event launch(::syclcompat::dim3 grid_dim, ::syclcompat::dim3 block_dim,
       Args... args);

```

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

inline void wg_barrier();

template <int Dim>
inline sycl::nd_range<Dim> compute_nd_range(sycl::range<Dim> global_size_in,
                                            sycl::range<Dim> work_group_size);
inline sycl::nd_range<1> compute_nd_range(int global_size_in, 
                                          int work_group_size);

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

inline sycl::queue create_queue(bool print_on_async_exceptions = false,
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

### Memory Operations

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

// Free
void wait_and_free(void *ptr, sycl::queue q = get_default_queue());
void free(void *ptr, sycl::queue q = get_default_queue());
sycl::event enqueue_free(const std::vector<void *> &pointers,
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

The `syclcompat::experimental` namespace contains currently unsupported `memcpy` overloads which take a `syclcompat::experimental::memcpy_parameter` argument. These are included for forwards compatibility and currently throw a `std::runtime_error`.

```cpp
namespace syclcompat {
namespace experimental {
// Forward declarations for types relating to unsupported memcpy_parameter API:

enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
class image_mem_wrapper;
#endif
class image_matrix;

/// Memory copy parameters for 2D/3D memory data.
struct memcpy_parameter {
  struct data_wrapper {
    pitched_data pitched{};
    sycl::id<3> pos{};
#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
    experimental::image_mem_wrapper *image_bindless{nullptr};
#endif
    image_matrix *image{nullptr};
  };
  data_wrapper from{};
  data_wrapper to{};
  sycl::range<3> size{};
  syclcompat::detail::memcpy_direction direction{syclcompat::detail::memcpy_direction::automatic};
};

/// [UNSUPPORTED] Synchronously copies 2D/3D memory data specified by \p param .
/// The function will return after the copy is completed.
///
/// \param param Memory copy parameters.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void memcpy(const memcpy_parameter &param,
                          sycl::queue q = get_default_queue());

/// [UNSUPPORTED] Asynchronously copies 2D/3D memory data specified by \p param
/// . The return of the function does NOT guarantee the copy is completed.
///
/// \param param Memory copy parameters.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void memcpy_async(const memcpy_parameter &param,
                                sycl::queue q = get_default_queue());

} // namespace experimental
} // namespace syclcompat
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
  template <typename WorkItemSizesTy = sycl::range<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() const;

  template <typename WorkItemSizesTy = sycl::range<3>,
          std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                std::is_same_v<WorkItemSizesTy, int *>,
                            int> = 0>
  auto get_max_work_item_sizes() const;
  bool get_host_unified_memory() const;
  int get_major_version() const;
  int get_minor_version() const;
  int get_integrated() const;
  int get_max_clock_frequency() const;
  int get_max_compute_units() const;
  int get_max_work_group_size() const;
  int get_max_sub_group_size() const;
  int get_max_work_items_per_compute_unit() const;
  int get_max_register_size_per_work_group() const;
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

  unsigned int get_memory_clock_rate() const;
  unsigned int get_memory_bus_width() const;
  uint32_t get_device_id() const;
  std::array<unsigned char, 16> get_uuid() const;
  unsigned int get_global_mem_cache_size() const;
  int get_image1d_max() const;
  auto get_image2d_max() const;
  auto get_image2d_max();
  auto get_image3d_max() const;
  auto get_image3d_max();

  void set_name(const char *name);
  void set_max_work_item_sizes(const sycl::range<3> max_work_item_sizes);
  [[deprecated]] void
  set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes);
  void set_host_unified_memory(bool host_unified_memory);
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
  void set_max_nd_range_size(sycl::id<3> max_nd_range_size);
  void set_memory_clock_rate(unsigned int memory_clock_rate);
  void set_memory_bus_width(unsigned int memory_bus_width);
  void 
  set_max_register_size_per_work_group(int max_register_size_per_work_group);
  void set_device_id(uint32_t device_id);
  void set_uuid(std::array<unsigned char, 16> uuid);
  void set_global_mem_cache_size(unsigned int global_mem_cache_size);
  void set_image1d_max(size_t image_max_buffer_size);
  void set_image2d_max(size_t image_max_width_buffer_size,
                       size_t image_max_height_buffer_size);
  void set_image3d_max(size_t image_max_width_buffer_size,
                       size_t image_max_height_buffer_size,
                       size_t image_max_depth_buffer_size);
};
```

### Device Management

Multiple SYCL functionalities are exposed through utility functions to manage
the current `sycl::device`, `sycl::queue`, and `sycl::context`, exposed as
follows:

```c++
namespace syclcompat {

// Util function to create a new queue for the current device
static inline sycl::queue create_queue(bool print_on_async_exceptions = false,
                                       bool in_order = true);

// Util function to get the default queue of current device in
// device manager.
static inline sycl::queue get_default_queue();

// Util function to set the default queue of the current device in the
// device manager.
// If the device extension saved queue is the default queue, 
// the previous saved queue will be overwritten as well.
// This function will be blocking if there are submitted kernels in the
// previous default queue.
static inline void set_default_queue(const sycl::queue &q);

// Util function to wait for the queued kernels.
static inline void wait(sycl::queue q = get_default_queue());

// Util function to wait for the queued kernels and throw unhandled errors.
static inline void wait_and_throw(sycl::queue q = get_default_queue());

// Util function to get the id of current device in
// device manager.
static inline unsigned int get_current_device_id();

// Util function to get the current device.
static inline device_ext &get_current_device();

// Util function to get a device by id.
static inline device_ext &get_device(unsigned int id);

// Util function to get the context of the default queue of current
// device in device manager.
static inline sycl::context get_default_context();

// Util function to get a CPU device.
static inline device_ext &cpu_device();

/// Filter out devices; only keep the device whose name contains one of the
/// subname in \p dev_subnames.
/// May break device id mapping and change current device. It's better to be
/// called before other SYCLcompat or SYCL APIs.
static inline void filter_device(const std::vector<std::string> &dev_subnames);

/// Print all the devices (and their IDs) in the dev_mgr
static inline void list_devices();

// Util function to select a device by its id
static inline unsigned int select_device(unsigned int id);

// Util function to get the device id from a device
static inline unsigned int get_device_id(const sycl::device &dev);

// Util function to get the number of available devices
static inline unsigned int device_count();

} // syclcompat
```

The exposed functionalities include creation and destruction of queues, through
`syclcompat::create_queue` and `syclcompat::destroy_queue`, and providing the
ability to wait for submitted kernels using `syclcompat::wait` or
`syclcompat::wait_and_throw`. Any async errors will be output to `stderr` if
`print_on_async_exceptions`, and will have the default behavior otherwise, which
calls `std:terminate`. Synchronous exceptions have to be managed by users
independently of what is set in this parameter.

Devices are managed through a helper class, `device_ext`. The `device_ext` class
associates a vector of `sycl::queues` with its `sycl::device`. The `device_ext`
destructor waits on a set of `sycl::event` which can be added to via
`add_event`. This is used, for example, to implement `syclcompat::enqueue_free` to
schedule release of memory after a kernel or `mempcy`. SYCL device properties
can be queried through `device_ext` as well.
`device_ext` also provides the `has_capability_or_fail` member function, which
throws a `sycl::exception` if the device does not have the specified list of
`sycl::aspect`.

Devices can be listed and filtered using `syclcompat::list_devices()` and
`syclcompat::filter_device()`. If `SYCLCOMPAT_VERBOSE` is defined at compile
time, the available SYCL devices are printed to the standard output both at
initialization time, and when the device list is filtered using
`syclcompat::filter_device`.

Users can manage queues through the `syclcompat::set_default_queue(sycl::queue
q)` free function, and the `device_ext` `set_saved_queue`, `set_default_queue`,
and `get_saved_queue` member functions.
`set_default_queue` is blocking, and overwrites the previous default queue with
a user defined one, waiting for any submitted kernels to finish.
The `device_ext` automatically sets the saved queue to the default queue.
Therefore, it's important to note that if the previous default queue was the
device's saved queue, setting a new default queue will update the reference of
the saved queue to the new default one to keep the state of the class
consistent.

The class is exposed as follows:

```c++
namespace syclcompat {

class device_ext : public sycl::device {
  device_ext();
  device_ext(const sycl::device &base, bool print_on_async_exceptions = false,
             bool in_order = true);
  ~device_ext();

  bool is_native_host_atomic_supported();
  int get_major_version() const;
  int get_minor_version() const;
  int get_max_compute_units() const;
  int get_max_clock_frequency() const;
  int get_integrated() const;
  int get_max_sub_group_size() const;
  int get_max_register_size_per_work_group() const;
  int get_max_work_group_size() const;
  int get_mem_base_addr_align() const;
  size_t get_global_mem_size() const;
  void get_memory_info(size_t &free_memory, size_t &total_memory) const;

  void get_device_info(device_info &out) const;
  device_info get_device_info() const;
  void reset(bool print_on_async_exceptions = false, bool in_order = true);

  sycl::queue *default_queue();
  void set_default_queue(const sycl::queue &q);
  void queues_wait_and_throw();
  sycl::queue *create_queue(bool print_on_async_exceptions = false,
                            bool in_order = true);
  void destroy_queue(sycl::queue *&queue);
  void set_saved_queue(sycl::queue *q);
  sycl::queue *get_saved_queue();
  sycl::context get_context();

  void
  has_capability_or_fail(const std::initializer_list<sycl::aspect> &props) const;
};

} // syclcompat
```

Free functions are provided for querying major and minor version directly from a `sycl::device`, equivalent to the methods of `device_ext` described above:

```c++
static int get_major_version(const sycl::device &dev);
static int get_minor_version(const sycl::device &dev);
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
arithmetic.

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
    sycl::multi_ptr<T, addressSpace> addr, type_identity_t<T> expected,
    type_identity_t<T> desired,
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

`reverse_bits` reverses the bits of a 32-bit unsigned integer, `ffs` returns the
position of the first least significant set bit in an integer.
`byte_level_permute` returns a byte-permutation of two input unsigned integers,
with bytes selected according to a third unsigned integer argument.
`match_all_over_sub_group` and `match_any_over_sub_group` allows comparison of
values across work-items within a sub-group.

The functions `select_from_sub_group`, `shift_sub_group_left`,
`shift_sub_group_right` and `permute_sub_group_by_xor` provide equivalent
functionality to `sycl::select_from_group`, `sycl::shift_group_left`,
`sycl::shift_group_right` and `sycl::permute_group_by_xor`, respectively.
However, they provide an optional argument to represent the `logical_group` size
(default 32).

`int_as_queue_ptr` helps with translation of code by reinterpret casting an
address to `sycl::queue *`, or returning a pointer to SYCLcompat's default queue
if the address is <= 2.
`args_selector` is a helper class for extracting arguments from an array of
pointers to arguments or buffer of arguments to pass to a kernel function.
The class allows users to exclude parameters such as `sycl::nd_item`.
Experimental support for masked versions of `select_from_sub_group`,
`shift_sub_group_left`, `shift_sub_group_right` and `permute_sub_group_by_xor`
is provided only for SPIRV or CUDA devices.

As part of the compatibility utilities to facilitate machine translation to
SYCL, two aliases for errors are provided, `err0` and `err1`.

```c++
namespace syclcompat {

inline int cast_double_to_int(double d, bool use_high32 = true);

inline double cast_ints_to_double(int high32, int low32);

inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s);

template <typename ValueT> inline int ffs(ValueT a);

template <typename T>
unsigned int match_any_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value);

template <typename T>
unsigned int match_all_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value, int *pred);

template <typename ValueT>
ValueT select_from_sub_group(sycl::sub_group g, ValueT x, int remote_local_id,
                        int logical_sub_group_size = 32);

template <typename ValueT>
ValueT shift_sub_group_left(sycl::sub_group g, ValueT x, unsigned int delta,
                       int logical_sub_group_size = 32);

template <typename ValueT>
ValueT shift_sub_group_right(sycl::sub_group g, ValueT x, unsigned int delta,
                        int logical_sub_group_size = 32);

template <typename ValueT>
ValueT permute_sub_group_by_xor(sycl::sub_group g, ValueT x, unsigned int mask,
                           int logical_sub_group_size = 32);

namespace experimental {

template <typename ValueT>
ValueT select_from_sub_group(unsigned int member_mask, sycl::sub_group g, ValueT x,
                             int remote_local_id, int logical_sub_group_size = 32);

template <typename ValueT>
ValueT shift_sub_group_left(unsigned int member_mask, sycl::sub_group g, ValueT x,
                            unsigned int delta, int logical_sub_group_size = 32);

template <typename ValueT>
ValueT shift_sub_group_right(unsigned int member_mask, sycl::sub_group g, ValueT x,
                             unsigned int delta, int logical_sub_group_size = 32);

template <typename ValueT>
ValueT permute_sub_group_by_xor(unsigned int member_mask, sycql::sub_group g, ValueT x,
                                unsigned int mask, int logical_sub_group_size = 32);

} // namespace experimental

inline sycl::queue *int_as_queue_ptr(uintptr_t x);

using err0 = detail::generic_error_type<struct err0_tag, int>;
using err1 = detail::generic_error_type<struct err1_tag, int>;

template <int n_nondefault_params, int n_default_params, typename T>
class args_selector;

template <int n_nondefault_params, int n_default_params, typename R,
          typename... Ts>
class args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
public:
  // Get the type of the ith argument of R(Ts...)
  template <int i>
  using arg_type =
      std::tuple_element_t<account_for_default_params<i>(), std::tuple<Ts...>>;

  // If kernel_params is nonnull, then args_selector will
  // extract arguments from kernel_params. Otherwise, it
  // will extract them from extra.
  args_selector(void **kernel_params, void **extra)
      : kernel_params(kernel_params), args_buffer(get_args_buffer(extra)) {}

  // Get a reference to the i-th argument extracted from kernel_params
  // or extra.
  template <int i> arg_type<i> &get();
};

} // namespace syclcompat
```

The function `experimental::nd_range_barrier` synchronizes work items from all
work groups within a SYCL kernel. This is not officially supported by the SYCL
spec, and so should be used with caution.
`experimental::calculate_max_active_wg_per_xecore` and
`experimental::calculate_max_potential_wg` are used for occupancy calculation.
There is also an `experimental::logical_group` class which allows
`sycl::sub_group`s to be further subdivided into 'logical' groups to perform
sub-group level operations. This class provides methods to get the local & group
id and range. `experimental::group_type`, `experimental::group` and
`experimental::group_base` are helper classes to manage the supported group
types.

```c++
namespace syclcompat {
namespace experimental {

#if defined(__AMDGPU__) || defined(__NVPTX__)
// seq_cst currently not working for AMD nor Nvidia
constexpr sycl::memory_order barrier_memory_order = sycl::memory_order::acq_rel;
#else
constexpr sycl::memory_order barrier_memory_order = sycl::memory_order::seq_cst;
#endif

template <int dimensions = 3>
inline void nd_range_barrier(
    sycl::nd_item<dimensions> item,
    sycl::atomic_ref<unsigned int, barrier_memory_order,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter);

template <>
inline void nd_range_barrier(
    sycl::nd_item<1> item,
    sycl::atomic_ref<unsigned int, barrier_memory_order,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter);

template <int dimensions = 3> class logical_group {
public:
  logical_group(sycl::nd_item<dimensions> item, sycl::group<dimensions> parent_group,
                uint32_t size);
  uint32_t get_local_linear_id() const;
  uint32_t get_group_linear_id() const;
  uint32_t get_local_linear_range() const;
  uint32_t get_group_linear_range() const;
};

inline int calculate_max_active_wg_per_xecore(int *num_wg, int wg_size,
                                              int slm_size = 0,
                                              int sg_size = 32,
                                              bool used_barrier = false,
                                              bool used_large_grf = false);

inline int calculate_max_potential_wg(int *num_wg, int *wg_size,
                                      int max_wg_size_for_device_code,
                                      int slm_size = 0, int sg_size = 32,
                                      bool used_barrier = false,
                                      bool used_large_grf = false);
// Supported group types
enum class group_type { work_group, sub_group, logical_group, root_group };

// The group_base will dispatch the function call to the specific interface
// based on the group type.
template <int dimensions = 3> class group_base {
public:
  group_base(sycl::nd_item<dimensions> item);

  // Returns the number of work-items in the group.
  size_t get_local_linear_range();
  // Returns the index of the work-item within the group.
  size_t get_local_linear_id();

  // Wait for all the elements within the group to complete their execution
  // before proceeding.
  void barrier();
};

// Container type that can store supported group_types.
template <typename GroupT, int dimensions = 3>
class group : public group_base<dimensions> {
public:
  group(GroupT g, sycl::nd_item<dimensions> item);
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

The `SYCLCOMPAT_CHECK_ERROR` macro encapsulates an error-handling mechanism for
expressions that might throw `sycl::exception` and `std::runtime_error`. If no
exceptions are thrown, it returns `syclcompat::error_code::SUCCESS`. If a
`sycl::exception` is caught, it returns `syclcompat::error_code::BACKEND_ERROR`.
If a `std::runtime_error` exception is caught,
`syclcompat::error_code::DEFAULT_ERROR` is returned instead. For both cases, it
prints the error message to the standard error stream.

``` c++
namespace syclcompat {

template <class... Args> class syclcompat_kernel_name;
template <int Arg> class syclcompat_kernel_scalar;

#if defined(_MSC_VER)
#define __syclcompat_align__(n) __declspec(align(n))
#define __syclcompat_inline__ __forceinline
#else
#define __syclcompat_align__(n) __attribute__((aligned(n)))
#define __syclcompat_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __syclcompat_noinline__ __declspec(noinline)
#else
#define __syclcompat_noinline__ __attribute__((noinline))
#endif

#define SYCLCOMPAT_COMPATIBILITY_TEMP (600)

#ifdef _WIN32
#define SYCLCOMPAT_EXPORT __declspec(dllexport)
#else
#define SYCLCOMPAT_EXPORT
#endif


namespace syclcompat {
enum error_code { SUCCESS = 0, BACKEND_ERROR = 1, DEFAULT_ERROR = 999 };
}

#define SYCLCOMPAT_CHECK_ERROR(expr)

int get_sycl_language_version();

} // namespace syclcompat
```

### Kernel Helper Functions

Kernel helper functions provide a structure `kernel_function_info` to keep SYCL
kernel information, and provide a utility function `get_kernel_function_info()`
to get the kernel information. Overloads are provided to allow either returning
a `kernel_function_info` object, or to return by pointer argument. In the
current version, `kernel_function_info` describes only maximum work-group size.

SYCLcompat also provides the `kernel_library` and `kernel_function` classes.
`kernel_library` facilitates the loading and unloading of kernel libraries.
`kernel_function` represents a specific kernel function within a loaded library
and can be invoked with specified arguments.
`load_kernel_library`, `load_kernel_library_mem`, and `unload_kernel_library`
are free functions to handle the loading and unloading of `kernel_library`
objects. `get_kernel_function`, and `invoke_kernel_function` offer a similar
functionality for `kernel_function` objects.

``` c++
namespace syclcompat {

struct kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function);
static kernel_function_info get_kernel_function_info(const void *function);

class kernel_library {
  constexpr kernel_library();
  constexpr kernel_library(void *ptr);
  operator void *() const;
};

static kernel_library load_kernel_library(const std::string &name);
static kernel_library load_kernel_library_mem(char const *const image);
static void unload_kernel_library(const kernel_library &library);

class kernel_function {
    constexpr kernel_function();
    constexpr kernel_function(kernel_functor ptr);
    operator void *() const;
    void operator()(sycl::queue &q, const sycl::nd_range<3> &range,
                    unsigned int local_mem_size, void **args, void **extra);
};

static kernel_function get_kernel_function(kernel_library &library,
                                           const std::string &name);
static void invoke_kernel_function(kernel_function &function,
                                   sycl::queue &queue,
                                   sycl::range<3> group_range,
                                   sycl::range<3> local_range,
                                   unsigned int local_mem_size,
                                   void **kernel_params, void **extra);

} // namespace syclcompat
```

### Math Functions

The `funnelshift_*` APIs perform a concatenate-shift operation on two 32-bit
values, and return a 32-bit result. The two unsigned integer arguments (`low`
and `high`) are concatenated to a 64-bit value which is then shifted left or
right by `shift` bits. The functions then return either the least- or
most-significant 32 bits. The `_l*` variants shift *left* and return the *most*
significant 32 bits, while the `_r*` variants shift *right* and return the
*least* significant 32 bits. The `_l`/`_r` APIs differ from the `_lc`/`_rc` APIs
in how they clamp the `shift` argument: `funnelshift_l` and `funnelshift_r`
shift the result by `shift & 31` bits, whereas `funnelshift_lc` and
`funnelshift_rc` shift the result by `min(shift, 32)` bits.

`syclcompat::fast_length` provides a wrapper to SYCL's
`fast_length(sycl::vec<float,N>)` that accepts arguments for a C++ array and a
length. `syclcompat::length` provides a templated version that wraps over
`sycl::length`. There are wrappers for `clamp`, `isnan`, `cbrt`, `min`, `max`,
`fmax_nan`, `fmin_nan`, and `pow`, as well as an implementation of `relu`
saturation is also provided.

`compare`, `unordered_compare`, `compare_both`, `unordered_compare_both`,
`compare_mask`, and `unordered_compare_mask`, handle both ordered and unordered
comparisons.

`vectorized_max` and `vectorized_min` are binary operations returning the
max/min of two arguments, where each argument is treated as a `sycl::vec` type.
`vectorized_isgreater` performs elementwise `isgreater`, treating each argument
as a vector of elements, and returning `0` for vector components for which
`isgreater` is false, and `-1` when true.
`vectorized_sum_abs_diff` calculates the absolute difference for two values
without modulo overflow for vector types.

The functions `cmul`,`cdiv`,`cabs`, `cmul_add`, and `conj` define complex math
operations which accept `sycl::vec<T,2>` arguments representing complex values.

The `dp4a` function returns the 4-way 8-bit dot product accumulate for unsigned
and signed 32-bit integer values. The `dp2a_lo` and `dp2a_hi` functions return the
two-way 16-bit to 8-bit dot product using the second and first 16 bits of the
second operand, respectively. These three APIs return a single 32-bit value with
the accumulated result, which is unsigned if both operands are `uint32_t` and
signed otherwise.

```cpp
inline unsigned int funnelshift_l(unsigned int low, unsigned int high,
                                  unsigned int shift); 

inline unsigned int funnelshift_lc(unsigned int low, unsigned int high,
                                   unsigned int shift); 

inline unsigned int funnelshift_r(unsigned int low, unsigned int high,
                                  unsigned int shift);

inline unsigned int funnelshift_rc(unsigned int low, unsigned int high,
                                   unsigned int shift);

inline float fast_length(const float *a, int len);

template <typename ValueT>
inline ValueT length(const ValueT *a, const int len);

inline ValueT clamp(ValueT val, ValueT min_val, ValueT max_val);

// Determine whether 2 element value is NaN.
template <typename ValueT>
inline std::enable_if_t<ValueT::size() == 2, ValueT> isnan(const ValueT a);

// cbrt function wrapper.
template <typename ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        ValueT>
cbrt(ValueT val) {
  return sycl::cbrt(static_cast<ValueT>(val));
}

// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
// sycl::half supported as well.
template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b);
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b);

sycl::half min(sycl::half a, sycl::half b);

template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b);
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b);

sycl::half max(sycl::half a, sycl::half b);

// Performs 2 elements comparison and returns the bigger one. If either of
// inputs is NaN, then return NaN.
template <typename ValueT, typename ValueU>
inline std::common_type_t<ValueT, ValueU> fmax_nan(const ValueT a,
                                                   const ValueU b);

template <typename ValueT, typename ValueU>
inline sycl::vec<std::common_type_t<ValueT, ValueU>, 2>
fmax_nan(const sycl::vec<ValueT, 2> a, const sycl::vec<ValueU, 2> b);

// Performs 2 elements comparison and returns the smaller one. If either of
// inputs is NaN, then return NaN.
template <typename ValueT, typename ValueU>
inline std::common_type_t<ValueT, ValueU> fmin_nan(const ValueT a,
                                                   const ValueU b);
template <typename ValueT, typename ValueU>
inline sycl::vec<std::common_type_t<ValueT, ValueU>, 2>
fmin_nan(const sycl::vec<ValueT, 2> a, const sycl::vec<ValueU, 2> b);

inline float pow(const float a, const int b) { return sycl::pown(a, b); }
inline double pow(const double a, const int b) { return sycl::pown(a, b); }

template <typename ValueT, typename ValueU>
inline typename std::enable_if_t<std::is_floating_point_v<ValueT>, ValueT>
pow(const ValueT a, const ValueU b);

// Requires aspect::fp64, as it casts to double internally.
template <typename ValueT, typename ValueU>
inline typename std::enable_if_t<!std::is_floating_point_v<ValueT>, double>
pow(const ValueT a, const ValueU b);

template <typename ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        ValueT>
relu(const ValueT a);

template <class ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        sycl::vec<ValueT, 2>>
relu(const sycl::vec<ValueT, 2> a);

template <class ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        sycl::marray<ValueT, 2>>
relu(const sycl::marray<ValueT, 2> a);

// The following definition is enabled when BinaryOperation(ValueT, ValueT) returns bool
// std::enable_if_t<std::is_same_v<std::invoke_result_t<BinaryOperation, ValueT, ValueT>, bool>, bool>
template <typename ValueT, class BinaryOperation>
inline bool 
compare(const ValueT a, const ValueT b, const BinaryOperation binary_op);
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, ValueT>
compare(const ValueT a, const ValueT b, const BinaryOperation binary_op);

// The following definition is enabled when BinaryOperation(ValueT, ValueT) returns bool
// std::enable_if_t<std::is_same_v<std::invoke_result_t<BinaryOperation, ValueT, ValueT>, bool>, bool>
template <typename ValueT, class BinaryOperation>
inline bool
unordered_compare(const ValueT a, const ValueT b,
                  const BinaryOperation binary_op);
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, ValueT>
unordered_compare(const ValueT a, const ValueT b,
                  const BinaryOperation binary_op);

template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, bool>
compare_both(const ValueT a, const ValueT b, const BinaryOperation binary_op);
template <typename ValueT, class BinaryOperation>

inline std::enable_if_t<ValueT::size() == 2, bool>
unordered_compare_both(const ValueT a, const ValueT b,
                       const BinaryOperation binary_op);

template <typename ValueT, class BinaryOperation>
inline unsigned compare_mask(const sycl::vec<ValueT, 2> a,
                             const sycl::vec<ValueT, 2> b,
                             const BinaryOperation binary_op);

template <typename ValueT, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::vec<ValueT, 2> a,
                                       const sycl::vec<ValueT, 2> b,
                                       const BinaryOperation binary_op);

template <typename S, typename T> inline T vectorized_max(T a, T b);

template <typename S, typename T> inline T vectorized_min(T a, T b);

template <typename S, typename T> inline T vectorized_isgreater(T a, T b);

template <>
inline unsigned vectorized_isgreater<sycl::ushort2, unsigned>(unsigned a,
                                                              unsigned b);

template <typename VecT>
inline unsigned vectorized_sum_abs_diff(unsigned a, unsigned b);

template <typename T>
sycl::vec<T, 2> cmul(sycl::vec<T, 2> x, sycl::vec<T, 2> y);

template <typename T>
sycl::vec<T, 2> cdiv(sycl::vec<T, 2> x, sycl::vec<T, 2> y);

template <typename T> T cabs(sycl::vec<T, 2> x);

template <typename ValueT>
inline sycl::vec<ValueT, 2> cmul_add(const sycl::vec<ValueT, 2> a,
                                     const sycl::vec<ValueT, 2> b,
                                     const sycl::vec<ValueT, 2> c);

template <typename ValueT>
inline sycl::marray<ValueT, 2> cmul_add(const sycl::marray<ValueT, 2> a,
                                        const sycl::marray<ValueT, 2> b,
                                        const sycl::marray<ValueT, 2> c);

template <typename T> sycl::vec<T, 2> conj(sycl::vec<T, 2> x);

template <typename ValueT> inline ValueT reverse_bits(ValueT a);


template <typename T1, typename T2>
using dot_product_acc_t =
    std::conditional_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
                       uint32_t, int32_t>;

template <typename T1, typename T2>
inline dot_product_acc_t<T1, T2> dp2a_lo(T1 a, T2 b,
                                         dot_product_acc_t<T1, T2> c);

template <typename T1, typename T2>
inline dot_product_acc_t<T1, T2> dp2a_hi(T1 a, T2 b,
                                         dot_product_acc_t<T1, T2> c);

template <typename T1, typename T2>
inline dot_product_acc_t<T1, T2> dp4a(T1 a, T2 b,
                                      dot_product_acc_t<T1, T2> c);
```

`vectorized_binary` computes the `BinaryOperation` for two operands,
with each value treated as a vector type. `vectorized_unary` offers the same
interface for operations with a single operand.
The implemented `BinaryOperation`s are `abs_diff`, `add_sat`, `rhadd`, `hadd`,
`maximum`, `minimum`, and `sub_sat`.

```cpp
namespace syclcompat {
  
template <typename VecT, class UnaryOperation>
inline unsigned vectorized_unary(unsigned a, const UnaryOperation unary_op);

// A sycl::abs wrapper functor.
struct abs {
  template <typename ValueT> auto operator()(const ValueT x) const;
};

template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b,
                                  const BinaryOperation binary_op);

// A sycl::abs_diff wrapper functor.
struct abs_diff {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::add_sat wrapper functor.
struct add_sat {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::rhadd wrapper functor.
struct rhadd {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::hadd wrapper functor.
struct hadd {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::max wrapper functor.
struct maximum {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::min wrapper functor.
struct minimum {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};
// A sycl::sub_sat wrapper functor.
struct sub_sat {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const;
};

} // namespace syclcompat
```

The math header provides a set of functions to extend 32-bit operations
to 33 bit, and handle sign extension internally. There is support for `add`,
`sub`, `absdiff`, `min` and `max` operations. Each operation provides overloads
to include a second, separate, `BinaryOperation` after the first, and include
the `_sat` variation, determines if the returning value is saturated or not.

```cpp
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add(AT a, BT b, CT c, BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add_sat(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub(AT a, BT b, CT c, BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub_sat(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff(AT a, BT b, CT c,
                                     BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff_sat(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff_sat(AT a, BT b, CT c,
                                         BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min(AT a, BT b, CT c, BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min_sat(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max(AT a, BT b, CT c, BinaryOperation second_op);

template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max_sat(AT a, BT b);

template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op);
```

Another set of vectorized extend 32-bit operations is provided in the math 
header.These APIs treat each of the 32-bit operands as 2-elements vector 
(16-bits each) while handling sign extension to 17-bits internally. There is 
support for `add`, `sub`, `absdiff`, `min`, `max` and `avg` binary operations. 
Each operation provides has a `_sat` variat which determines if the returning 
value is saturated or not, and a `_add` variant that computes the binary sum 
of the the initial operation outputs and a third operand. 

```cpp
/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2(AT a, BT b, RetT c);

/// Compute vectorized addition of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_add(AT a, BT b, RetT c);

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_sat(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_add(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_sat(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_add(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_sat(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_add(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_sat(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_add(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_sat(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend average maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2_add(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2_sat(AT a, BT b, RetT c);
```

Similarly, a set of vectorized extend 32-bit operations is provided in the math 
header treating each of the 32-bit operands as 4-elements vector (8-bits each) 
while handling sign extension to 9-bits internally. There is support for `add`,
`sub`, `absdiff`, `min`, `max` and `avg` binary operations. 
Each operation provides has a `_sat` variat which determines if the returning 
value is saturated or not, and a `_add` variant that computes the binary sum 
of the the initial operation outputs and a third operand. 

```cpp
/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4(AT a, BT b, RetT c);

/// Compute vectorized addition of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_add(AT a, BT b, RetT c);

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_sat(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_add(AT a, BT b, RetT c);

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_sat(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_add(AT a, BT b, RetT c);

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_sat(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_add(AT a, BT b, RetT c);

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_sat(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_add(AT a, BT b, RetT c);

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_sat(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized average of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4_add(AT a, BT b, RetT c);

/// Compute vectorized average of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4_sat(AT a, BT b, RetT c);
```

Vectorized comparison APIs also provided in the math header behave similarly 
and support a `std` comparison operator parameter which can be `greater`, 
`less`, `greater_equal`, `less_equal`, `equal_to` or `not_equal_to`. These APIs 
cover both the 2-elements *(16-bits each)* and 4-elements *(8-bits each)* 
variants, as well as an additional `_add` variant that computes the sum of the 
2/4 output elements.

```cpp
/// Extend \p a and \p b to 33 bit and vectorized compare input values using
/// specified comparison \p cmp .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values.
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare2(AT a, BT b, BinaryOperation cmp);

/// Extend Inputs to 33 bit, and vectorized compare input values using specified
/// comparison \p cmp , then add the result with \p c .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values, and add the
/// result with \p c .
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare2_add(AT a, BT b, unsigned c,
                                               BinaryOperation cmp);

/// Extend \p a and \p b to 33 bit and vectorized compare input values using
/// specified comparison \p cmp .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values.
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare4(AT a, BT b, BinaryOperation cmp);

/// Extend Inputs to 33 bit, and vectorized compare input values using specified
/// comparison \p cmp , then add the result with \p c .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values, and add the
/// result with \p c .
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare4_add(AT a, BT b, unsigned c,
                                               BinaryOperation cmp);
```

The math header file provides APIs for bit-field insertion (`bfi_safe`) and
bit-field extraction (`bfe_safe`). These are bounds-checked variants of
underlying `detail` APIs (`detail::bfi`, `detail::bfe`) which, in future
releases, will be exposed to the user.

```c++

/// Bitfield-insert with boundary checking.
///
/// Align and insert a bit field from \param x into \param y . Source \param
/// bit_start gives the starting bit position for the insertion, and source
/// \param num_bits gives the bit field length in bits.
///
/// \tparam T The type of \param x and \param y , must be an unsigned integer.
/// \param x The source of the bitfield.
/// \param y The source where bitfield is inserted.
/// \param bit_start The position to start insertion.
/// \param num_bits The number of bits to insertion.
template <typename T>
inline T bfi_safe(const T x, const T y, const uint32_t bit_start,
                  const uint32_t num_bits);

/// Bitfield-extract with boundary checking.
///
/// Extract bit field from \param source and return the zero or sign-extended
/// result. Source \param bit_start gives the bit field starting bit position,
/// and source \param num_bits gives the bit field length in bits.
///
/// The result is padded with the sign bit of the extracted field. If `num_bits`
/// is zero, the  result is zero. If the start position is beyond the msb of the
/// input, the result is filled with the replicated sign bit of the extracted
/// field.
///
/// \tparam T The type of \param source value, must be an integer.
/// \param source The source value to extracting.
/// \param bit_start The position to start extracting.
/// \param num_bits The number of bits to extracting.
template <typename T>
inline T bfe_safe(const T source, const uint32_t bit_start,
                  const uint32_t num_bits);
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
    assert(h_Y[i] - h_expected[i] < 1e-6);
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
