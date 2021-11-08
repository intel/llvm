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
#include <CL/sycl/properties/property_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace reduction {
class initialize_to_identity
    : public detail::DataLessProperty<detail::InitializeToIdentity> {};
} // namespace reduction
} // namespace property

// Reduction property trait specializations
template <>
struct is_property<property::reduction::initialize_to_identity>
    : std::true_type {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
