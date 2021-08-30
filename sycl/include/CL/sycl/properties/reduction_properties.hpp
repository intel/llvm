//==------- reduction_properties.hpp --- SYCL reduction properties ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/property_helper.hpp>

__SYCL_OPEN_NS {
namespace property {
namespace reduction {
class initialize_to_identity
    : public detail::DataLessProperty<detail::InitializeToIdentity> {};
} // namespace reduction
} // namespace property
} __SYCL_CLOSE_NS
