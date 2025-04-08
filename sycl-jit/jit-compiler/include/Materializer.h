//==--- Materializer.h - Public interface for spec constant materializer ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32
#define SCM_EXPORT_SYMBOL __declspec(dllexport)
#else
#define SCM_EXPORT_SYMBOL
#endif

#include "Kernel.h"
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

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif // __clang__

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif // _MSC_VER

SCM_EXPORT_SYMBOL SCMResult materializeSpecConstants(
    const char *KernelName, const jit_compiler::JITBinaryInfo &BinaryInfo,
    View<unsigned char> SpecConstBlob);

/// Clear all previously set options.
SCM_EXPORT_SYMBOL void resetJITConfiguration();

/// Add an option to the configuration.
SCM_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt);

} // end of extern "C"

} // namespace jit_compiler
