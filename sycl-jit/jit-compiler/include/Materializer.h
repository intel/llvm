//===- Materializer.h - Public interface for spec constant materializer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "JITBinaryInfo.h"
#include "Macros.h"
#include "Options.h"
#include "View.h"
#include "sycl/detail/string.hpp"

#include <cassert>

namespace jit_compiler {

/// Result type for specialization constant materializer: Wraps either a binary
/// descriptor of the device image compiled with concrete values, or an error
/// message, without using std::variant.
class SCMResult {
public:
  /// Constructs a result indicating failure.
  explicit SCMResult(const char *ErrorMessage)
      : Failed{true}, BinaryInfo{}, ErrorMessage{ErrorMessage} {}

  /// Constructs a result indicating success.
  explicit SCMResult(const JITBinaryInfo &BinaryInfo)
      : Failed{false}, BinaryInfo(BinaryInfo), ErrorMessage{} {}

  bool failed() const noexcept { return Failed; }

  const char *getErrorMessage() const noexcept {
    assert(failed() && "No error message present");
    return ErrorMessage.c_str();
  }

  const JITBinaryInfo &getBinaryInfo() const noexcept {
    assert(!failed() && "No binary info");
    return BinaryInfo;
  }

private:
  const bool Failed;
  const JITBinaryInfo BinaryInfo;
  const sycl::detail::string ErrorMessage;
};

/// Obtains an LLVM module from the \p BinaryInfo descriptor, replaces loads of
/// specialization constants in kernel \p KernelName with the values encoded in
/// \p SpecConstBlob, and finally translates the module to \p TargetFormat.
JIT_EXPORT_SYMBOL SCMResult materializeSpecConstants(
    const char *KernelName, const jit_compiler::JITBinaryInfo &BinaryInfo,
    jit_compiler::BinaryFormat TargetFormat, View<unsigned char> SpecConstBlob);

} // namespace jit_compiler
