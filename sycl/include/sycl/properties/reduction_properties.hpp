//==------- reduction_properties.hpp --- SYCL reduction properties ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/detail/property_helper.hpp>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace property::reduction {
class initialize_to_identity
    : public detail::DataLessProperty<detail::InitializeToIdentity> {};
} // namespace property::reduction

// Reduction property trait specializations
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
