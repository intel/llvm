
# FPGA lsu

**IMPORTANT:** This is a proposed update to an existing extension.  The APIs
described in this document are not yet implemented and cannot be used in
application code.  See [here][1] for the existing extension, which is
implemented.

[1]: <../supported/SYCL_EXT_INTEL_FPGA_LSU.md>

The Intel FPGA `lsu` class is implemented in `sycl/ext/intel/fpga_lsu.hpp` which
is included in `sycl/ext/intel/fpga_extensions.hpp`.

The class `cl::sycl::ext::intel::lsu` allows users to explicitly request that the
implementation of a global memory access is configured in a certain way. The
class has two member functions, `load()` and `store()` which allow loading from
and storing to a `multi_ptr`, respectively, and is templated on the following
4 optional paremeters:

1.  **`cl::sycl::ext::intel::burst_coalesce<B>`, where `B` is a boolean**: request,
to the extent possible, that a dynamic burst coalescer be implemented when
`load` or `store` are called. The default value of this parameter is `false`.
2. **`cl::sycl::ext::intel::cache<N>`, where `N` is an integer greater or equal to
0**: request, to the extent possible, that a read-only cache of the specified
size in bytes be implemented when when `load` is called. It is not allowed to
use that parameter for `store`. The default value of this parameter is `0`.
3. **`cl::sycl::ext::intel::statically_coalesce<B>`, where `B` is a boolean**:
request, to the extent possible, that `load` or `store` accesses, is allowed to
be statically coalesced with other memory accesses at compile time. The default
value of this parameter is `true`.
4. **`cl::sycl::ext::intel::prefetch<B>`, where `B` is a boolean**: request, to the
extent possible, that a prefetcher be implemented when `load` is called. It is
not allowed to use that parameter for `store`. The default value of this
parameter is `false`.

Currently, not every combination of parameters is allowed due to limitations in
the backend. The following rules apply:
1. For `store`, `cl::sycl::ext::intel::cache` must be `0` and
`cl::sycl::ext::intel::prefetch` must be `false`.
2. For `load`, if `cl::sycl::ext::intel::cache` is set to a value greater than `0`,
then `cl::sycl::ext::intel::burst_coalesce` must be set to `true`.
3. For `load`, exactly one of `cl::sycl::ext::intel::prefetch` and
`cl::sycl::ext::intel::burst_coalesce` is allowed to be `true`.
4. For `load`, exactly one of `cl::sycl::ext::intel::prefetch` and
`cl::sycl::ext::intel::cache` is allowed to be `true`.

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
#include <sycl/ext/intel/fpga_extensions.hpp>
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
        cl::sycl::ext::intel::lsu<cl::sycl::ext::intel::prefetch<true>,
                             cl::sycl::ext::intel::statically_coalesce<false>>;

    using BurstCoalescedLSU =
        cl::sycl::ext::intel::lsu<cl::sycl::ext::intel::burst_coalesce<false>,
                             cl::sycl::ext::intel::statically_coalesce<false>>;

    using CachingLSU =
        cl::sycl::ext::intel::lsu<cl::sycl::ext::intel::burst_coalesce<true>,
                             cl::sycl::ext::intel::cache<1024>,
                             cl::sycl::ext::intel::statically_coalesce<true>>;

    using PipelinedLSU = cl::sycl::ext::intel::lsu<>;

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
template arguments, which can contain the latency control properties
`latency_anchor_id` and/or `latency_constraint`.

1. **`sycl::ext::intel::experimental::latency_anchor_id<N>`, where `N` is an integer**:
represents ID of the current function call when it performs as an anchor. The ID
must be unique within the application, with a diagnostic required if that
condition is not met.
2. **`sycl::ext::intel::experimental::latency_constraint<A, B, C>`** contains control
parameters when the current function performs as a non-anchor, where:
    - **`A` is an integer**: The ID of the target anchor defined on a different
    instruction through a `latency_anchor_id` property.
    - **`B` is an enum value**: The type of control from the set
    {`type::exact`, `type::max`, `type::min`}.
    - **`C` is an integer**: The relative clock cycle difference between the
    target anchor and the current function call, that the constraint should
    infer subject to the type of the control (exact, max, min).

The template arguments above don't have to be specified if user doesn't want to
apply latency controls. The template arguments can be passed in arbitrary order.

### Implementation
```c++
// Added in version 2 of this extension.
namespace sycl::ext::intel::experimental {
enum class type {
  none, // default
  exact,
  max,
  min
};

template <int32_t _N> struct latency_anchor_id {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = -1;
};

template <int32_t _N1, type _N2, int32_t _N3> struct latency_constraint {
  static constexpr std::tuple<int32_t, type, int32_t> value = {_N1, _N2, _N3};
  static constexpr std::tuple<int32_t, type, int32_t> default_value = {
      0, type::none, 0};
};

template <class... mem_access_params> class lsu final {
public:
  lsu() = delete;

  template <class... _Params, typename _T, access::address_space _space>
  static _T load(sycl::multi_ptr<_T, _space> Ptr) {
    check_space<_space>();
    check_load();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    static constexpr auto _anchor_id =
        __GetValue<int, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        __GetValue3<int, type, int, latency_constraint, _Params...>::value;

    static constexpr int _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int _cycle = std::get<2>(_constraint);
    int _type;
    if (_control_type == type::none) {
      _type = 0;
    } else if (_control_type == type::exact) {
      _type = 1;
    } else if (_control_type == type::max) {
      _type = 2;
    } else { // _control_type == type::min
      _type = 3;
    }

    return *__latency_control_mem_wrapper((_T *)Ptr, _anchor_id, _target_anchor,
                                          _type, _cycle);
#else
    return *Ptr;
#endif
  }

  template <class... _Params, typename _T, access::address_space _space>
  static void store(sycl::multi_ptr<_T, _space> Ptr, _T Val) {
    check_space<_space>();
    check_store();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    static constexpr auto _anchor_id =
        __GetValue<int, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        __GetValue3<int, type, int, latency_constraint, _Params...>::value;

    static constexpr int _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int _cycle = std::get<2>(_constraint);
    int _type;
    if (_control_type == type::none) {
      _type = 0;
    } else if (_control_type == type::exact) {
      _type = 1;
    } else if (_control_type == type::max) {
      _type = 2;
    } else { // _control_type == type::min
      _type = 3;
    }

    *__latency_control_mem_wrapper((_T *)Ptr, _anchor_id, _target_anchor, _type,
                                   _cycle) = Val;
#else
    *Ptr = Val;
#endif
  }
  ...
private:
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
  template <typename _T>
  static _T *__latency_control_mem_wrapper(_T * Ptr, int32_t AnchorID,
                                           int32_t TargetAnchor, int32_t Type,
                                           int32_t Cycle) {
    return __builtin_intel_fpga_mem(Ptr,
                                    _burst_coalesce | _cache |
                                    _dont_statically_coalesce | _prefetch,
                                    _cache_val);
  }
#endif
  ...
}
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

    // latency controls
    using ExpPrefetchingLSU = sycl::ext::intel::experimental::lsu<
        sycl::ext::intel::experimental::prefetch<true>,
        sycl::ext::intel::experimental::statically_coalesce<false>>;

    using ExpBurstCoalescedLSU = sycl::ext::intel::experimental::lsu<
        sycl::ext::intel::experimental::burst_coalesce<false>,
        sycl::ext::intel::experimental::statically_coalesce<false>>;

    // The following load is anchor 1
    int Z = ExpPrefetchingLSU::load<
        sycl::ext::intel::experimental::latency_anchor_id<1>>(input_ptr + 2);

    // The following store occurs exactly 5 cycles after the anchor 1 read
    ExpBurstCoalescedLSU::store<
        sycl::ext::intel::experimental::latency_constraint<
            1, sycl::ext::intel::experimental::type::exact, 5>>(output_ptr + 2,
                                                                Z);
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
feature, or applications can test the macro's value to determine which of the
extension's APIs the implementation supports.

|Value |Description|
|:---- |:---------:|
|1     |Initial extension version. Base features are supported.|
|2     |Add experimental latency control API.|
