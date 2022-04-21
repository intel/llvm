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
namespace ext {
namespace oneapi {
namespace cuda {
namespace property {
namespace context {
class use_primary_context : public ::cl::sycl::detail::DataLessProperty<
                                ::cl::sycl::detail::UsePrimaryContext> {};
} // namespace context
} // namespace property
} // namespace cuda
} // namespace oneapi
} // namespace ext

namespace property {
namespace context {
namespace __SYCL2020_DEPRECATED(
    "use 'sycl::ext::oneapi::cuda::property::context' instead") cuda {
  class use_primary_context : public ::cl::sycl::ext::oneapi::cuda::property::
                                  context::use_primary_context {};
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

template <>
struct is_property<ext::oneapi::cuda::property::context::use_primary_context>
    : std::true_type {};

template <>
struct is_property_of<ext::oneapi::cuda::property::context::use_primary_context,
                      context> : std::true_type {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
