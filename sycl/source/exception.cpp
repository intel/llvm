//==---------------- exception.cpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <CL/sycl/context.hpp>
#include <CL/sycl/exception.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

const char *exception::what() const noexcept { return MMsg.c_str(); }

bool exception::has_context() const { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw invalid_object_error();

  return *MContext;
}

cl_int exception::get_cl_code() const { return MCLErr; }

const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return {static_cast<int>(Err), sycl_category()};
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
