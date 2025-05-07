//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/kernel_arg_mask.hpp>
#include <sycl/detail/ur.hpp>

#include <mutex>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered_map.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
using FastKernelCacheKeyT = std::pair<ur_device_handle_t, ur_context_handle_t>;
using FastKernelCacheValT =
    std::tuple<ur_kernel_handle_t, std::mutex *, const KernelArgMask *,
               ur_program_handle_t>;
using FastKernelSubcacheT =
    ::boost::unordered_flat_map<FastKernelCacheKeyT, FastKernelCacheValT>;

struct KernelNameBasedCacheT {
  FastKernelSubcacheT *FastKernelSubcachePtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
