//==---- specialization_id.hpp -- SYCL standard header file ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>      // for __SYCL_TYPE
#include <sycl/kernel_bundle_enums.hpp> // for bundle_state

#include <utility> // for forward

namespace sycl {
inline namespace _V1 {

/// Declaring a specialization constant
///
/// \ingroup sycl_api
template <typename T> class __SYCL_TYPE(specialization_id) specialization_id {
public:
  using value_type = T;

  template <class... Args>
  explicit constexpr specialization_id(Args &&...args)
      : MDefaultValue(std::forward<Args>(args)...) {}

  specialization_id(const specialization_id &rhs) = delete;
  specialization_id(specialization_id &&rhs) = delete;
  specialization_id &operator=(const specialization_id &rhs) = delete;
  specialization_id &operator=(specialization_id &&rhs) = delete;

private:
  template <bundle_state State> friend class kernel_bundle;
  T getDefaultValue() const noexcept { return MDefaultValue; }

  T MDefaultValue;
};

} // namespace _V1
} // namespace sycl
