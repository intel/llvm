//==------- common_info.hpp ----- Common SYCL info methods------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/__impl/detail/export.hpp>
#include <sycl/__impl/stl.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace detail {

vector_class<string_class> __SYCL_EXPORT split_string(const string_class &str,
                                                      char delimeter);

} // namespace detail
} // namespace sycl
} // namespace __sycl_internal
