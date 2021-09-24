//==-------------- fpga_lsu.hpp --- SYCL FPGA LSU Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "fpga_utils.hpp"
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/pointers.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
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

private:
  static constexpr int32_t _burst_coalesce_val =
      _GetValue<burst_coalesce_impl, _mem_access_params...>::value;
  static constexpr uint8_t _burst_coalesce =
      _burst_coalesce_val == 1 ? BURST_COALESCE : 0;

  static constexpr int32_t _cache_val =
      _GetValue<cache, _mem_access_params...>::value;
  static constexpr uint8_t _cache = (_cache_val > 0) ? CACHE : 0;

  static constexpr int32_t _statically_coalesce_val =
      _GetValue<statically_coalesce_impl, _mem_access_params...>::value;
  static constexpr uint8_t _dont_statically_coalesce =
      _statically_coalesce_val == 0 ? STATICALLY_COALESCE : 0;

  static constexpr int32_t _prefetch_val =
      _GetValue<prefetch_impl, _mem_access_params...>::value;
  static constexpr uint8_t _prefetch = _prefetch_val ? PREFETCH : 0;

  static_assert(_cache_val >= 0, "cache size parameter must be non-negative");

  template <access::address_space _space> static void check_space() {
    static_assert(_space == access::address_space::global_space ||
                      _space == access::address_space::global_device_space ||
                      _space == access::address_space::global_host_space,
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
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") INTEL {
  using namespace ext::intel;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
