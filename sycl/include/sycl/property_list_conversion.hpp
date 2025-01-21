//==---- property_list_conversion.hpp --- SYCL property list conversion ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This file contains conversion routines from property_list to
// accessor_property_list. A separate file helps to avoid cyclic dependencies
// between header files.

#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/property_list.hpp>

namespace sycl {
inline namespace _V1 {
template <typename... T>
inline property_list::operator ext::oneapi::accessor_property_list<T...>() {
  return ext::oneapi::accessor_property_list<T...>(MDataLessProps,
                                                   MPropsWithData);
}
} // namespace _V1
} // namespace sycl
