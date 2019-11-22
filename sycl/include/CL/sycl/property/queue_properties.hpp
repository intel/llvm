//==--------- queue_properties.hpp --- SYCL queue properties ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_base.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace queue {

class enable_profiling : public detail::property_base {};

class in_order : public detail::property_base {};

} // namespace queue
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
