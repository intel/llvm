//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/kernel_arg_mask.hpp>
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>

#include <mutex>
#include <optional>

#include <boost/unordered/unordered_flat_map.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
using FastKernelCacheKeyT = std::pair<ur_device_handle_t, ur_context_handle_t>;
using FastKernelCacheValT =
    std::tuple<ur_kernel_handle_t, std::mutex *, const KernelArgMask *,
               ur_program_handle_t>;
using FastKernelSubcacheMapT =
    ::boost::unordered_flat_map<FastKernelCacheKeyT, FastKernelCacheValT>;

using FastKernelSubcacheMutexT = SpinLock;
using FastKernelSubcacheReadLockT = std::lock_guard<FastKernelSubcacheMutexT>;
using FastKernelSubcacheWriteLockT = std::lock_guard<FastKernelSubcacheMutexT>;

struct FastKernelSubcacheT {
  FastKernelSubcacheMapT Map;
  FastKernelSubcacheMutexT Mutex;
};

struct KernelNameBasedCacheT {
  FastKernelSubcacheT FastKernelSubcache;
  std::optional<bool> UsesAssert;
  // Implicit local argument position is represented by an optional int, this
  // uses another optional on top of that to represent lazy initialization of
  // the cached value.
  std::optional<std::optional<int>> ImplicitLocalArgPos;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
