
# FPGA lsu

The Intel FPGA `lsu` class is implemented in `CL/sycl/INTEL/fpga_lsu.hpp` which
is included in `CL/sycl/INTEL/fpga_extensions.hpp`.

The class `cl::sycl::INTEL::lsu` allows users to explicitly request that the
implementation of a global memory access is configured in a certain way. The
class has two member functions, `load()` and `store()` which allow loading from
and storing to a `multi_ptr`, respectively, and is templated on the following
4 optional paremeters:

1.  **`cl::sycl::INTEL::burst_coalesce<B>`, where `B` is a boolean**: request,
to the extent possible, that a dynamic burst coalescer be implemented when
`load` or `store` are called. The default value of this parameter is `false`.
2. **`cl::sycl::INTEL::cache<N>`, where `N` is an integer greater or equal to
0**: request, to the extent possible, that a read-only cache of the specified
size in bytes be implemented when when `load` is called. It is not allowed to
use that parameter for `store`. The default value of this parameter is `0`.
3. **`cl::sycl::INTEL::statically_coalesce<B>`, where `B` is a boolean**:
request, to the extent possible, that `load` or `store` accesses, is allowed to
be statically coalesced with other memory accesses at compile time. The default
value of this parameter is `true`.
4. **`cl::sycl::INTEL::prefetch<B>`, where `B` is a boolean**: request, to the
extent possible, that a prefetcher be implemented when `load` is called. It is
not allowed to use that parameter for `store`. The default value of this
parameter is `false`.

Currently, not every combination of parameters is allowed due to limitations in
the backend. The following rules apply:
1. For `store`, `cl::sycl::INTEL::cache` must be `0` and
`cl::sycl::INTEL::prefetch` must be `false`.
2. For `load`, if `cl::sycl::INTEL::cache` is set to a value greater than `0`,
then `cl::sycl::INTEL::burst_coalesce` must be set to `true`.
3. For `load`, exactly one of `cl::sycl::INTEL::prefetch` and
`cl::sycl::INTEL::burst_coalesce` is allowed to be `true`.
4. For `load`, exactly one of `cl::sycl::INTEL::prefetch` and
`cl::sycl::INTEL::cache` is allowed to be `true`.

## Implementation

The implementation relies on the Clang built-in `__builtin_intel_fpga_mem` when
parsing the SYCL device code. The built-in uses the LLVM `ptr.annotation`
intrinsic under the hood to annotate the pointer that is being accessed.
```c++
template <class... mem_access_params> class lsu final {
public:
  lsu() = delete;

  template <typename _T, access::address_space _space>
  static _T load(sycl::multi_ptr<_T, _space> Ptr) {
    check_space<_space>();
    check_load();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    return *__builtin_intel_fpga_mem((_T *)Ptr,
                                     _burst_coalesce | _cache |
                                         _dont_statically_coalesce | _prefetch,
                                     _cache_val);
#else
    return *Ptr;
#endif
  }

  template <typename _T, access::address_space _space>
  static void store(sycl::multi_ptr<_T, _space> Ptr, _T Val) {
    check_space<_space>();
    check_store();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    *__builtin_intel_fpga_mem((_T *)Ptr,
                              _burst_coalesce | _cache |
                                  _dont_statically_coalesce | _prefetch,
                              _cache_val) = Val;
#else
    *Ptr = Val;
#endif
  }
  ...
}
```

## Usage

```c++
#include <CL/sycl/INTEL/fpga_extensions.hpp>
...
cl::sycl::buffer<int, 1> output_buffer(output_data, 1);
cl::sycl::buffer<int, 1> input_buffer(input_data, 1);

Queue.submit([&](cl::sycl::handler &cgh) {
  auto output_accessor = output_buffer.get_access<cl::sycl::access::mode::write>(cgh);
  auto input_accessor = input_buffer.get_access<cl::sycl::access::mode::read>(cgh);

  cgh.single_task<class kernel>([=] {
    auto input_ptr = input_accessor.get_pointer();
    auto output_ptr = output_accessor.get_pointer();

    using PrefetchingLSU =
        cl::sycl::INTEL::lsu<cl::sycl::INTEL::prefetch<true>,
                             cl::sycl::INTEL::statically_coalesce<false>>;

    using BurstCoalescedLSU =
        cl::sycl::INTEL::lsu<cl::sycl::INTEL::burst_coalesce<false>,
                             cl::sycl::INTEL::statically_coalesce<false>>;

    using CachingLSU =
        cl::sycl::INTEL::lsu<cl::sycl::INTEL::burst_coalesce<true>,
                             cl::sycl::INTEL::cache<1024>,
                             cl::sycl::INTEL::statically_coalesce<true>>;

    using PipelinedLSU = cl::sycl::INTEL::lsu<>;

    int X = PrefetchingLSU::load(input_ptr); // int X = input_ptr[0]
    int Y = CachingLSU::load(input_ptr + 1); // int Y = input_ptr[1]

    BurstCoalescedLSU::store(output_ptr, X); // output_ptr[0] = X
    PipelinedLSU::store(output_ptr + 1, Y);  // output_ptr[1] = Y
  });
});
...
```
