//==----------- context_properties.hpp --- SYCL context properties ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/property_helper.hpp>

__SYCL_OPEN_NS() {
namespace property {
namespace context {
namespace cuda {
class use_primary_context
    : public detail::DataLessProperty<detail::UsePrimaryContext> {};
} // namespace cuda
} // namespace context
} // namespace property
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
