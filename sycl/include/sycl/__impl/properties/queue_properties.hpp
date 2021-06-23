//==----------- queue_properties.hpp --- SYCL queue properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/property_helper.hpp>

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace property {
namespace queue {
class in_order : public detail::DataLessProperty<detail::InOrder> {};
class enable_profiling
    : public detail::DataLessProperty<detail::QueueEnableProfiling> {};
} // namespace queue
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifndef __SYCL_ENABLE_SYCL121_NAMESPACE
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#endif
