
# FPGA lsu

The Intel FPGA `lsu` class is implemented in `sycl/ext/intel/fpga_lsu.hpp` which
is included in `sycl/ext/intel/fpga_extensions.hpp`.

The class `sycl::ext::intel::lsu` allows users to explicitly request that the
implementation of a global memory access is configured in a certain way. The
class has two member functions, `load()` and `store()` which allow loading from
and storing to a `multi_ptr`, respectively, and is templated on the following
4 optional parameters:

1.  **`sycl::ext::intel::burst_coalesce<B>`, where `B` is a boolean**: request,
to the extent possible, that a dynamic burst coalescer be implemented when
`load` or `store` are called. The default value of this parameter is `false`.
2. **`sycl::ext::intel::cache<N>`, where `N` is an integer greater or equal to
0**: request, to the extent possible, that a read-only cache of the specified
size in bytes be implemented when when `load` is called. It is not allowed to
use that parameter for `store`. The default value of this parameter is `0`.
3. **`sycl::ext::intel::statically_coalesce<B>`, where `B` is a boolean**:
request, to the extent possible, that `load` or `store` accesses, is allowed to
be statically coalesced with other memory accesses at compile time. The default
value of this parameter is `true`.
4. **`sycl::ext::intel::prefetch<B>`, where `B` is a boolean**: request, to the
extent possible, that a prefetcher be implemented when `load` is called. It is
not allowed to use that parameter for `store`. The default value of this
parameter is `false`.

Currently, not every combination of parameters is allowed due to limitations in
the backend. The following rules apply:
1. For `store`, `sycl::ext::intel::cache` must be `0` and
`sycl::ext::intel::prefetch` must be `false`.
2. For `load`, if `sycl::ext::intel::cache` is set to a value greater than `0`,
then `sycl::ext::intel::burst_coalesce` must be set to `true`.
3. For `load`, exactly one of `sycl::ext::intel::prefetch` and
`sycl::ext::intel::burst_coalesce` is allowed to be `true`.
4. For `load`, exactly one of `sycl::ext::intel::prefetch` and
`sycl::ext::intel::cache` is allowed to be `true`.

Member functions `load()` or `store()` can take in an ext::oneapi::properties as argument,
which contains the following two properties of latency control:

1. **`sycl::ext::intel::latency_anchor_id<N>`, where `N` is an integer**:
represents ID of the current function call when it performs as an anchor. The ID
must be unique within the application, with a diagnostic required if that
condition is not met.
2. **`sycl::ext::intel::latency_constraint<A, B, C>`** contains control
parameters when the current function performs as a non-anchor, where:
    - **`A` is an integer**: The ID of the target anchor defined on a different
    instruction through a `latency_anchor_id` property.
    - **`B` is an enum value**: The type of control from the set
    {`type::exact`, `type::min`, `type::max`}.
    - **`C` is an integer**: The relative clock cycle difference between the
    target anchor and the current function call, that the constraint should
    infer subject to the type of the control (exact, min, max).

## Implementation

The implementation relies on the Clang built-in `__builtin_intel_fpga_mem` when
parsing the SYCL device code. The built-in uses the LLVM `ptr.annotation`
intrinsic under the hood to annotate the pointer that is being accessed.
```c++
namespace sycl::ext::oneapi {

struct latency_anchor_id {
  template<int32_t ID>
  using value_t = property_value<latency_anchor_id,
                                 std::integral_constant<int32_t, ID>>;
};

struct latency_constraint {
  enum class type {
    none,
    exact,
    max,
    min
  };
  template <int32_t Target, type Type, int32_t Cycle>
  using value_t = property_value<latency_constraint,
                                 std::integral_constant<int32_t, Target>,
                                 std::integral_constant<type, Type>,
                                 std::integral_constant<int32_t, Cycle>>;
};

template<int32_t Target, type Type, int32_t Cycle>
struct property_value<latency_constraint,
                      std::integral_constant<int32_t, Target>,
                      std::integral_constant<latency_constraint::type, Type>,
                      std::integral_constant<int32_t, Cycle>> {
  static constexpr int32_t target = Target;
  static constexpr latency_constraint::type type = Type;
  static constexpr int32_t cycle = Cycle;
};

} // namespace sycl::ext::oneapi

namespace sycl::ext::intel {

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
                                     _cache_val,
                                     _reserved_default_anchor_id, 0, 0, 0);
#else
    return *Ptr;
#endif
  }

  // Added in version 2 of this extension.
  template <typename _T, access::address_space _space, typename Properties>
  static _T load(sycl::multi_ptr<_T, _space> Ptr, Properties p) {
    check_space<_space>();
    check_load();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    statuc constexpr int32_t _target = p.get_property<latency_constraint>().target;

    static constexpr sycl::ext::oneapi::latency_constraint::type _type_enum = p.get_property<latency_constraint>().type;
    int32_t _type;
    if (_type_enum == sycl::ext::oneapi::latency_constraint::type::none) {
      _type = 0;
    } else if (_type_enum == sycl::ext::oneapi::latency_constraint::type::exact) {
      _type = 1;
    } else if (_type_enum == sycl::ext::oneapi::latency_constraint::type::max) {
      _type = 2;
    } else { // _type_enum == sycl::ext::oneapi::latency_constraint::type::min
      _type = 3;
    }

    statuc constexpr int32_t _cycle = p.get_property<latency_constraint>().cycle;

    return *__builtin_intel_fpga_mem((_T *)Ptr,
                                     _burst_coalesce | _cache |
                                     _dont_statically_coalesce | _prefetch,
                                     _cache_val,
                                     p.get_property<latency_anchor_id>().value,
                                     _target, _type, _cycle);
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
                              _cache_val,
                              _reserved_default_anchor_id, 0, 0, 0) = Val;
#else
    *Ptr = Val;
#endif
  }

  // Added in version 2 of this extension.
  template <typename _T, access::address_space _space, typename Properties>
  static void store(sycl::multi_ptr<_T, _space> Ptr, _T Val, Properties p) {
    check_space<_space>();
    check_store();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    statuc constexpr int32_t _target = p.get_property<latency_constraint>().target;

    static constexpr sycl::ext::oneapi::latency_constraint::type _type_enum = p.get_property<latency_constraint>().type;
    int32_t _type;
    if (_type_enum == sycl::ext::oneapi::latency_constraint::type::none) {
      _type = 0;
    } else if (_type_enum == sycl::ext::oneapi::latency_constraint::type::exact) {
      _type = 1;
    } else if (_type_enum == sycl::ext::oneapi::latency_constraint::type::max) {
      _type = 2;
    } else { // _type_enum == sycl::ext::oneapi::latency_constraint::type::min
      _type = 3;
    }

    statuc constexpr int32_t _cycle = p.get_property<latency_constraint>().cycle;

    *__builtin_intel_fpga_mem((_T *)Ptr,
                              _burst_coalesce | _cache |
                              _dont_statically_coalesce | _prefetch,
                              _cache_val,
                              p.get_property<latency_anchor_id>().value,
                              _target, _type, _cycle) = Val;
#else
    *Ptr = Val;
#endif
  }
  ...
}

} // namespace sycl::ext::intel
```

## Usage

```c++
#include <sycl/ext/intel/fpga_extensions.hpp>
...
sycl::buffer<int, 1> output_buffer(output_data, 1);
sycl::buffer<int, 1> input_buffer(input_data, 1);

Queue.submit([&](sycl::handler &cgh) {
  auto output_accessor = output_buffer.get_access<sycl::access::mode::write>(cgh);
  auto input_accessor = input_buffer.get_access<sycl::access::mode::read>(cgh);

  cgh.single_task<class kernel>([=] {
    auto input_ptr = input_accessor.get_pointer();
    auto output_ptr = output_accessor.get_pointer();

    using PrefetchingLSU =
        sycl::ext::intel::lsu<sycl::ext::intel::prefetch<true>,
                             sycl::ext::intel::statically_coalesce<false>>;

    using BurstCoalescedLSU =
        sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<false>,
                             sycl::ext::intel::statically_coalesce<false>>;

    using CachingLSU =
        sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                             sycl::ext::intel::cache<1024>,
                             sycl::ext::intel::statically_coalesce<true>>;

    using PipelinedLSU = sycl::ext::intel::lsu<>;

    int X = PrefetchingLSU::load(input_ptr); // int X = input_ptr[0]
    int Y = CachingLSU::load(input_ptr + 1); // int Y = input_ptr[1]

    BurstCoalescedLSU::store(output_ptr, X); // output_ptr[0] = X
    PipelinedLSU::store(output_ptr + 1, Y);  // output_ptr[1] = Y

    // Latency controls. Added in version 2 of this extension.
    // Load is anchor 1
    int Z = PrefetchingLSU::load(input_ptr + 2,
                                 sycl::ext::oneapi::properties{latency_anchor_id<1>});
    // Store occurs 5 cycles after the anchor 1 load
    BurstCoalescedLSU::store(output_ptr + 2, Z,
                             sycl::ext::oneapi::properties{latency_constraint<1,
                                                                              sycl::ext::oneapi::latency_constraint::type::exact,
                                                                              5>});
  });
});
...
```

## Feature Test Macro

This extension provides a feature-test macro as described in the core SYCL
specification section 6.3.3 "Feature test macros". Therefore, an implementation
supporting this extension must predefine the macro `SYCL_EXT_INTEL_FPGA_LSU`
to one of the values defined in the table below. Applications can test for the
existence of this macro to determine if the implementation supports this
feature, or applications can test the macro’s value to determine which of the
extension’s APIs the implementation supports.

|Value |Description|
|:---- |:---------:|
|1     |Initial extension version. Base features are supported.|
|2     |Add latency control feature to member functions.|
