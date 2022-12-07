//==----------- context_properties.hpp --- SYCL context properties ---------==//
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
namespace ext::oneapi::cuda::property::context {
class use_primary_context : public ::sycl::detail::DataLessProperty<
                                ::sycl::detail::UsePrimaryContext> {};
} // namespace ext::oneapi::cuda::property::context

namespace property::context {
namespace __SYCL2020_DEPRECATED(
    "use 'sycl::ext::oneapi::cuda::property::context' instead") cuda {
class use_primary_context
    : public ::sycl::ext::oneapi::cuda::property::context::use_primary_context {
};
// clang-format off
} // namespace cuda
// clang-format on
} // namespace property::context

// Forward declaration
class context;

// Context property trait specializations
template <>
struct is_property_of<property::context::cuda::use_primary_context, context>
    : std::true_type {};

template <>
struct is_property_of<ext::oneapi::cuda::property::context::use_primary_context,
                      context> : std::true_type {};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
