//==------- property_base.hpp --- SYCL property base class -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace detail {

// Base class for SYCL properties. This is used by property_list for
// identifying SYCL properties to avoid ambiguities in constructors taking
// a property_list parameter.
class property_base {
public:
  virtual ~property_base() = default;
};

} // namespace detail
} // namespace property
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
