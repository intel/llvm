//==----------- context_properties.hpp --- SYCL context properties ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/property_helper.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace property {
namespace context {
namespace cuda {
class use_primary_context
    : public detail::DataLessProperty<detail::UsePrimaryContext> {};
} // namespace cuda
} // namespace context
} // namespace property
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
