//==-------------- fpga_lsu.hpp --- SYCL FPGA LSU Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "fpga_utils.hpp"
#include <sycl/detail/defines.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/pointers.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

constexpr uint8_t BURST_COALESCE = 0x1;
constexpr uint8_t CACHE = 0x2;
constexpr uint8_t STATICALLY_COALESCE = 0x4;
constexpr uint8_t PREFETCH = 0x8;

template <int32_t _N> struct burst_coalesce_impl {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = 0;
};

template <int32_t _N> struct cache {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = 0;
};

template <int32_t _N> struct prefetch_impl {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = 0;
};

template <int32_t _N> struct statically_coalesce_impl {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = 1;
};

template <bool _B> using burst_coalesce = burst_coalesce_impl<_B>;
template <bool _B> using prefetch = prefetch_impl<_B>;
template <bool _B> using statically_coalesce = statically_coalesce_impl<_B>;

template <class... _mem_access_params> class lsu final {
public:
  lsu() = delete;

  template <typename _T, access::address_space _space,
            access::decorated _Is_decorated, typename _propertiesT>
  static _T load(sycl::multi_ptr<_T, _space, _Is_decorated> Ptr,
                 _propertiesT Properties) {
    check_space<_space>();
    check_load();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    _T *P = Ptr;
    if constexpr (!std::is_same_v<_propertiesT,
                                  oneapi::experimental::empty_properties_t>) {
      detail::AnnotatedMemberValue<_T *, _propertiesT> annotated_wrapper(Ptr);
      P = annotated_wrapper.MValue;
    }
    return *__builtin_intel_fpga_mem(
        P, _burst_coalesce | _cache | _dont_statically_coalesce | _prefetch,
        _cache_val);
#else
    (void)Properties;
    return *Ptr;
#endif
  }

  template <typename _T, access::address_space _space,
            access::decorated _Is_decorated>
  static _T load(sycl::multi_ptr<_T, _space, _Is_decorated> Ptr) {
    return load<_T, _space>(Ptr, oneapi::experimental::properties{});
  }

  template <typename _T, access::address_space _space,
            access::decorated _Is_decorated, typename _propertiesT>
  static void store(sycl::multi_ptr<_T, _space, _Is_decorated> Ptr, _T Val,
                    _propertiesT Properties) {
    check_space<_space>();
    check_store();
#if defined(__SYCL_DEVICE_ONLY__) && __has_builtin(__builtin_intel_fpga_mem)
    _T *P = Ptr;
    if constexpr (!std::is_same_v<_propertiesT,
                                  oneapi::experimental::empty_properties_t>) {
      detail::AnnotatedMemberValue<_T *, _propertiesT> annotated_wrapper(Ptr);
      P = annotated_wrapper.MValue;
    }
    *__builtin_intel_fpga_mem(
        P, _burst_coalesce | _cache | _dont_statically_coalesce | _prefetch,
        _cache_val) = Val;
#else
    (void)Properties;
    *Ptr = Val;
#endif
  }

  template <typename _T, access::address_space _space,
            access::decorated _Is_decorated>
  static void store(sycl::multi_ptr<_T, _space, _Is_decorated> Ptr, _T Val) {
    store<_T, _space>(Ptr, Val, oneapi::experimental::properties{});
  }

private:
  static constexpr int32_t _burst_coalesce_val =
      detail::_GetValue<burst_coalesce_impl, _mem_access_params...>::value;
  static constexpr uint8_t _burst_coalesce =
      _burst_coalesce_val == 1 ? BURST_COALESCE : 0;

  static constexpr int32_t _cache_val =
      detail::_GetValue<cache, _mem_access_params...>::value;
  static constexpr uint8_t _cache = (_cache_val > 0) ? CACHE : 0;

  static constexpr int32_t _statically_coalesce_val =
      detail::_GetValue<statically_coalesce_impl, _mem_access_params...>::value;
  static constexpr uint8_t _dont_statically_coalesce =
      _statically_coalesce_val == 0 ? STATICALLY_COALESCE : 0;

  static constexpr int32_t _prefetch_val =
      detail::_GetValue<prefetch_impl, _mem_access_params...>::value;
  static constexpr uint8_t _prefetch = _prefetch_val ? PREFETCH : 0;

  static_assert(_cache_val >= 0, "cache size parameter must be non-negative");

  template <access::address_space _space> static void check_space() {
    static_assert(
        _space == access::address_space::global_space ||
            _space == access::address_space::ext_intel_global_device_space ||
            _space == access::address_space::ext_intel_global_host_space,
        "lsu controls are only supported for global_ptr, "
        "device_ptr, and host_ptr objects");
  }

  static void check_load() {
    static_assert(_cache == 0 || _burst_coalesce == BURST_COALESCE,
                  "unable to implement a cache without a burst coalescer");
    static_assert(_prefetch == 0 || _burst_coalesce == 0,
                  "unable to implement a prefetcher and a burst coalescer "
                  "simulataneously");
    static_assert(
        _prefetch == 0 || _cache == 0,
        "unable to implement a prefetcher and a cache simulataneously");
  }
  static void check_store() {
    static_assert(_cache == 0, "unable to implement a store LSU with a cache.");
    static_assert(_prefetch == 0,
                  "unable to implement a store LSU with a prefetcher.");
  }
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
