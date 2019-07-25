//==-------------- usm_enums.hpp - SYCL USM Enums --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

namespace cl {
namespace sycl {
namespace usm {

enum class alloc { host, device, shared, unknown };

} // namespace usm
} // namespace sycl
} // namespace cl
