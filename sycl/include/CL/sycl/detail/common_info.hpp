//==------- common_info.hpp ----- Common SYCL info methods------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

std::vector<std::string> __SYCL_EXPORT split_string(const std::string &str,
                                                    char delimeter);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
