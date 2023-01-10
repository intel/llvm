
# FPGA lsu

The Intel FPGA `lsu` class is implemented in `sycl/ext/intel/fpga_lsu.hpp` which
is included in `sycl/ext/intel/fpga_extensions.hpp`.

The class `sycl::ext::intel::lsu` allows users to explicitly request that the
implementation of a global memory access is configured in a certain way. The
class has two member functions, `load()` and `store()` which allow loading from
and storing to a `multi_ptr`, respectively, and is templated on the following
4 optional paremeters:

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

## Implementation

The implementation relies on the Clang built-in `__builtin_intel_fpga_mem` when
parsing the SYCL device code. The built-in uses the LLVM `ptr.annotation`
intrinsic under the hood to annotate the pointer that is being accessed.
```c++
template <class... MemAccessParams> class lsu final {
public:
  lsu() = delete;

  template <typename T, access::address_space Space>
  static T load(sycl::multi_ptr<T, Space> Ptr) {
    check_space<Space>();
    check_load();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    return *__builtin_intel_fpga_mem((T *)Ptr,
                                     _burst_coalesce | _cache |
                                         _dont_statically_coalesce | _prefetch,
                                     _cache_val);
#else
    return *Ptr;
#endif
  }

  template <typename T, access::address_space Space>
  static void store(sycl::multi_ptr<T, Space> Ptr, T Val) {
    check_space<Space>();
    check_store();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    *__builtin_intel_fpga_mem((T *)Ptr,
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
  });
});
...
```

## Experimental APIs

**NOTE**: The APIs described in this section are experimental. Future versions of
this extension may change these APIs in ways that are incompatible with the
versions described here.

In the experimental API version, member functions `load()` and `store()` take
in a property list as function argument, which can contain the latency control
properties `latency_anchor_id` and/or `latency_constraint`.

1. **`sycl::ext::intel::experimental::latency_anchor_id<N>`, where `N` is an integer**:
represents ID of the current function call when it performs as an anchor. The ID
must be unique within the application, with a diagnostic required if that
condition is not met.
2. **`sycl::ext::intel::experimental::latency_constraint<A, B, C>`** contains control
parameters when the current function performs as a non-anchor, where:
    - **`A` is an integer**: The ID of the target anchor defined on a different
    instruction through a `latency_anchor_id` property.
    - **`B` is an enum value**: The type of control from the set
    {`latency_control_type::exact`, `latency_control_type::max`, `latency_control_type::min`}.
    - **`C` is an integer**: The relative clock cycle difference between the
    target anchor and the current function call, that the constraint should
    infer subject to the type of the control (exact, max, min).

### Synopsis
```c++
// Added in version 2 of this extension.
namespace sycl::ext::intel::experimental {
enum class latency_control_type {
  none, // default
  exact,
  max,
  min
};

struct latency_anchor_id_key {
  template <int Anchor>
  using value_t =
      oneapi::experimental::property_value<latency_anchor_id_key,
                                           std::integral_constant<int, Anchor>>;
};

struct latency_constraint_key {
  template <int Target, latency_control_type Type, int Cycle>
  using value_t = oneapi::experimental::property_value<
      latency_constraint_key, std::integral_constant<int, Target>,
      std::integral_constant<latency_control_type, Type>,
      std::integral_constant<int, Cycle>>;
};

template <int Anchor>
inline constexpr latency_anchor_id_key::value_t<Anchor> latency_anchor_id;

template <int Target, latency_control_type Type, int Cycle>
inline constexpr latency_constraint_key::value_t<Target, Type, Cycle>
    latency_constraint;

template <class... MemAccessParams> class lsu final {
  template <typename T, access::address_space Space>
  static T load(sycl::multi_ptr<T, Space> Ptr);

  template <typename T, access::address_space Space, typename PropertiesT>
  static T load(sycl::multi_ptr<T, Space> Ptr, PropertiesT Properties);

  template <typename T, access::address_space Space>
  static void store(sycl::multi_ptr<T, Space> Ptr, T Val);

  template <typename T, access::address_space Space, typename PropertiesT>
  static void store(sycl::multi_ptr<T, Space> Ptr, T Val,
                    PropertiesT Properties);
};
} // namespace sycl::ext::intel::experimental
```

### Usage
```c++
// Added in version 2 of this extension.
#include <sycl/ext/intel/fpga_extensions.hpp>
...
sycl::buffer<int, 1> output_buffer(output_data, 1);
sycl::buffer<int, 1> input_buffer(input_data, 1);
Queue.submit([&](sycl::handler &cgh) {
  auto output_accessor =
      output_buffer.get_access<sycl::access::mode::write>(cgh);
  auto input_accessor = input_buffer.get_access<sycl::access::mode::read>(cgh);

  cgh.single_task<class kernel>([=] {
    auto input_ptr = input_accessor.get_pointer();
    auto output_ptr = output_accessor.get_pointer();

    using ExpPrefetchingLSU = sycl::ext::intel::experimental::lsu<
        sycl::ext::intel::experimental::prefetch<true>,
        sycl::ext::intel::experimental::statically_coalesce<false>>;

    using ExpBurstCoalescedLSU = sycl::ext::intel::experimental::lsu<
        sycl::ext::intel::experimental::burst_coalesce<false>,
        sycl::ext::intel::experimental::statically_coalesce<false>>;

    // The following load is anchor 1
    int Z = ExpPrefetchingLSU::load(
        input_ptr + 2,
        sycl::ext::oneapi::experimental::properties(latency_anchor_id<1>));

    // The following store occurs exactly 5 cycles after the anchor 1 read
    ExpBurstCoalescedLSU::store(
        output_ptr + 2, Z,
        sycl::ext::oneapi::experimental::properties(
            latency_constraint<1, latency_control_type::exact, 5>));
  });
});
...
} // namespace sycl::ext::intel::experimental
```

## Feature Test Macro

This extension provides a feature-test macro as described in the core SYCL
specification section 6.3.3 "Feature test macros". Therefore, an implementation
supporting this extension must predefine the macro `SYCL_EXT_INTEL_FPGA_LSU`
to one of the values defined in the table below. Applications can test for the
existence of this macro to determine if the implementation supports this
feature, or applications can test the macro's value to determine which of the
extension's APIs the implementation supports.

|Value |Description|
|:---- |:---------:|
|1     |Initial extension version. Base features are supported.|
|2     |Add experimental latency control API.|
