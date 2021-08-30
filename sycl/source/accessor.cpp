//==------------ accessor.cpp - SYCL standard source file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/accessor.hpp>
#include <detail/queue_impl.hpp>

__SYCL_OPEN_NS() {
namespace detail {
device getDeviceFromHandler(handler &CommandGroupHandlerRef) {
  return CommandGroupHandlerRef.MQueue->get_device();
}
} // namespace detail
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
