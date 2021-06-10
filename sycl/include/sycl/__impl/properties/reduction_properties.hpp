//==------- reduction_properties.hpp --- SYCL reduction properties ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/property_helper.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace property {
namespace reduction {
class initialize_to_identity
    : public detail::DataLessProperty<detail::InitializeToIdentity> {};
} // namespace reduction
} // namespace property
} // namespace sycl
} // namespace __sycl_internal
