//==------------ accessor.cpp - SYCL standard source file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/accessor.hpp>
#include <detail/queue_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
device getDeviceFromHandler(handler &CommandGroupHandlerRef) {
  return CommandGroupHandlerRef.MQueue->get_device();
}
context getContextFromHandler(handler &CommandGroupHandlerRef) {
  return CommandGroupHandlerRef.MQueue->get_context();
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
