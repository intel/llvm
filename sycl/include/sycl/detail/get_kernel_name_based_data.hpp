//==--------------------- get_kernel_name_based_data.hpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/detail/kernel_name_str_t.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
class KernelNameBasedCacheT;
__SYCL_EXPORT KernelNameBasedCacheT *createKernelNameBasedCache();
#endif

class DeviceKernelInfo;

__SYCL_EXPORT DeviceKernelInfo *
getKernelNameBasedDataImpl(detail::ABINeutralKernelNameStrRefT KernelName);

// Retrieves and caches a data pointer to avoid kernel name based lookup
// overhead.
template <typename KernelNameT>
DeviceKernelInfo *
getKernelNameBasedData(detail::ABINeutralKernelNameStrRefT KernelName) {
  static DeviceKernelInfo *Instance = getKernelNameBasedDataImpl(KernelName);
  return Instance;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
