//===--- level_zero_ownership.hpp - Level Zero Ownership -- ------*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero {

// Since Level-Zero is not doing any reference counting itself, we have to
// be explicit about the ownership of the native handles used in the
// interop functions below.
//
enum class ownership { transfer, keep };

} // namespace ext::oneapi::level_zero
} // namespace _V1
} // namespace sycl
