//==-------- llvm_error_stub.cpp - Stub for ErrorInfoBase typeinfo --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// LLVM is built with LLVM_ENABLE_RTTI=OFF, which means typeinfo symbols are
// not generated. However, when Expected<T> destructor runs on an error value,
// it needs to delete the ErrorInfoBase pointer, which requires typeinfo.
// This file provides a minimal stub to satisfy the linker.

#include <typeinfo>

namespace llvm {
class ErrorInfoBase {
public:
  virtual ~ErrorInfoBase() = default;
};
} // namespace llvm
