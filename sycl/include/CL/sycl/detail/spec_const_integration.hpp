//==---------------------- spec_const_integration.hpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header file must not be included to any DPC++ headers.
// This header file should only be included to integration footer.

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#if __cplusplus >= 201703L
// Translates SYCL 2020 `specialization_id` to a unique symbolic identifier
// which is used internally by the toolchain
template <auto &SpecName> const char *get_spec_constant_symbolic_ID() {
  return get_spec_constant_symbolic_ID_impl<SpecName>();
}
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
