//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/kernel_name_str_t.hpp>

#include <ur_api.h>

#include <memory>
#include <tuple>
#include <vector>

namespace sycl {
inline namespace _V1 {
class event;

namespace detail {
class CGExecKernel;
class queue_impl;
class RTDeviceBinaryImage;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
void waitEvents(std::vector<sycl::event> DepEvents);
#endif

std::tuple<const RTDeviceBinaryImage *, ur_program_handle_t>
retrieveKernelBinary(queue_impl &Queue, KernelNameStrRefT KernelName,
                     CGExecKernel *CGKernel = nullptr);
} // namespace detail
} // namespace _V1
} // namespace sycl
