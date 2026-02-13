//==--------- handler_proxy.cpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/handler_proxy.hpp>

#include <sycl/handler.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

void associateWithHandler(handler &CGH, AccessorBaseHost *Acc,
                          access::target Target) {
  CGH.associateWithHandler(Acc, Target);
}

void associateWithHandler(handler &CGH, UnsampledImageAccessorBaseHost *Acc,
                          image_target Target) {
  CGH.associateWithHandler(Acc, Target);
}

void associateWithHandler(handler &CGH, SampledImageAccessorBaseHost *Acc,
                          image_target Target) {
  CGH.associateWithHandler(Acc, Target);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
