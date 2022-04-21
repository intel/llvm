//===-- util.cpp - Shared SYCL runtime utilities impl ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/util.hpp>
#include <detail/global_handler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

Sync &Sync::getInstance() { return GlobalHandler::instance().getSync(); }

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
