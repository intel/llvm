//==-------------- fpga_lsu.hpp --- SYCL FPGA Reg Extensions ---------------==//
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
namespace intel {
constexpr uint8_t BURST_COALESCE = 0x1;
constexpr uint8_t CACHE = 0x2;
constexpr uint8_t DONT_STATICALLY_COALESCE = 0x4;
constexpr uint8_t PREFETCH = 0x8;

template <int32_t N> struct burst_coalesce {
  static constexpr int32_t value = N;
  static constexpr int32_t default_value = false;
};

template <int32_t N> struct cache {
  static constexpr int32_t value = N;
  static constexpr int32_t default_value = 0;
};

template <int32_t N> struct prefetch {
  static constexpr int32_t value = N;
  static constexpr int32_t default_value = false;
};

template <int32_t N> struct dont_statically_coalesce {
  static constexpr int32_t value = N;
  static constexpr int32_t default_value = 0;
};

template <class... mem_access_params> class lsu final {
public:
  lsu() = delete;

  template <typename T> static T &load(sycl::global_ptr<T> Ptr) {
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

  template <typename T> static void store(sycl::global_ptr<T> Ptr, T Val) {
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

private:
  static constexpr int32_t _burst_coalesce_val =
      GetValue<burst_coalesce, mem_access_params...>::value;
  static constexpr uint8_t _burst_coalesce =
      _burst_coalesce_val == 1 ? BURST_COALESCE : 0;

  static constexpr int32_t _cache_val =
      GetValue<cache, mem_access_params...>::value;
  static constexpr uint8_t _cache = (_cache_val > 0) ? CACHE : 0;

  static constexpr int32_t _dont_statically_coalesce_val =
      GetValue<dont_statically_coalesce, mem_access_params...>::value;
  static constexpr uint8_t _dont_statically_coalesce =
      _dont_statically_coalesce_val == 1 ? DONT_STATICALLY_COALESCE : 0;

  static constexpr int32_t _prefetch_val =
      GetValue<prefetch, mem_access_params...>::value;
  static constexpr uint8_t _prefetch = _prefetch_val ? PREFETCH : 0;

  static_assert(_burst_coalesce_val == 0 || _burst_coalesce_val == 1,
                "burst_coalesce parameter must be 0 or 1");
  static_assert(_cache_val >= 0, "cache size parameter must be non-negative");
  static_assert(_dont_statically_coalesce_val == 0 ||
                    _dont_statically_coalesce_val == 1,
                "dont_statically_coalesce parameter must be 0 or 1");
  static_assert(_prefetch_val == 0 || _prefetch_val == 1,
                "prefetch parameter must be 0 or 1");

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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
