//===---- SPIR.h - Declare SPIR and SPIR-V target interfaces ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace clang {
namespace targets {

// Used by both the SPIR and SPIR-V targets. Code of the generic address space
// for the target
constexpr unsigned SPIR_GENERIC_AS = 4u;

} // namespace targets
} // namespace clang
