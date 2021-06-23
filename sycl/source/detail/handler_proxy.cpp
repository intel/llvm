//==--------- handler_proxy.cpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/handler_proxy.hpp>

#include <sycl/__impl/handler.hpp>

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace detail {

void associateWithHandler(handler &CGH, AccessorBaseHost *Acc,
                          access::target Target) {
  CGH.associateWithHandler(Acc, Target);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
