//==--- Materializer.h - Public interface for spec constant materializer ---==//
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

class SCMResult {
public:
  explicit SCMResult(const char *ErrorMessage)
      : Failed{true}, BinaryInfo{}, ErrorMessage{ErrorMessage} {}

  explicit SCMResult(const JITBinaryInfo &BinaryInfo)
      : Failed{false}, BinaryInfo(BinaryInfo), ErrorMessage{} {}

  bool failed() const { return Failed; }

  const char *getErrorMessage() const {
    assert(failed() && "No error message present");
    return ErrorMessage.c_str();
  }

  const JITBinaryInfo &getBinaryInfo() const {
    assert(!failed() && "No binary info");
    return BinaryInfo;
  }

private:
  bool Failed;
  JITBinaryInfo BinaryInfo;
  sycl::detail::string ErrorMessage;
};

extern "C" {

JIT_EXPORT_SYMBOL SCMResult materializeSpecConstants(
    const char *KernelName, const jit_compiler::JITBinaryInfo &BinaryInfo,
    View<unsigned char> SpecConstBlob);

} // end of extern "C"

} // namespace jit_compiler
