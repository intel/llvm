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
#include <CL/sycl/properties/property_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace context {
namespace cuda {
class use_primary_context
    : public detail::DataLessProperty<detail::UsePrimaryContext> {};
} // namespace cuda
} // namespace context
} // namespace property

// Forward declaration
class context;

// Context property trait specializations
template <>
struct is_property<property::context::cuda::use_primary_context>
    : std::true_type {};

template <>
struct is_property_of<property::context::cuda::use_primary_context, context>
    : std::true_type {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
