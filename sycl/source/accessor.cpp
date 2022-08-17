//==------------ accessor.cpp - SYCL standard source file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/accessor.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
device getDeviceFromHandler(handler &CommandGroupHandlerRef) {
  return CommandGroupHandlerRef.MQueue->get_device();
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
