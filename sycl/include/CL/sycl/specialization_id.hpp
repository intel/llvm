//==---- specialization_id.hpp -- SYCL standard header file ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

/// Declaring a specialization constant
///
/// \ingroup sycl_api
template <typename T> class specialization_id {
public:
  using value_type = T;

  template <class... Args>
  explicit constexpr specialization_id(Args &&... args)
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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
