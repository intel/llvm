//==---- property_list_conversion.hpp --- SYCL property list conversion ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains conversion routines from property_list to
// accessor_property_list. A separate file helps to avoid cyclic dependencies
// between header files.

#include <sycl/__impl/ONEAPI/accessor_property_list.hpp>
#include <sycl/__impl/property_list.hpp>

namespace __sycl_internal {
inline namespace __v1 {
template <typename... T>
inline property_list::operator ONEAPI::accessor_property_list<T...>() {
  return ONEAPI::accessor_property_list<T...>(MDataLessProps, MPropsWithData);
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
#endif
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
}
#endif
