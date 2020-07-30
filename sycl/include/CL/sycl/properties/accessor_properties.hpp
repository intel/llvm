//==----------- accessor_properties.hpp --- SYCL accessor properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_helper.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {

class noinit : public detail::DataLessProperty<detail::NoInit> {};

} // namespace property

#if __cplusplus > 201402L

inline constexpr property::noinit noinit;

#else

namespace {

constexpr const auto &noinit =
    sycl::detail::InlineVariableHelper<property::noinit>::value;
}

#endif

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
