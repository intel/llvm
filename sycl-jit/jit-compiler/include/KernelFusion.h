//==- KernelFusion.h - Public interface of JIT compiler for kernel fusion --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
#define SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H

#include "Kernel.h"
#include "Options.h"
#include "Parameter.h"
#include "View.h"
#include "sycl/detail/string.hpp"

#include <cassert>

namespace jit_compiler {

class JITResult {
public:
  explicit JITResult(const char *ErrorMessage)
      : Type{JITResultType::FAILED}, KernelInfo{}, ErrorMessage{ErrorMessage} {}

  explicit JITResult(const SYCLKernelInfo &KernelInfo, bool Cached = false)
      : Type{(Cached) ? JITResultType::CACHED : JITResultType::NEW},
        KernelInfo(KernelInfo), ErrorMessage{} {}

  bool failed() const { return Type == JITResultType::FAILED; }

  bool cached() const { return Type == JITResultType::CACHED; }

  const char *getErrorMessage() const {
    assert(failed() && "No error message present");
    return ErrorMessage.c_str();
  }

  const SYCLKernelInfo &getKernelInfo() const {
    assert(!failed() && "No kernel info");
    return KernelInfo;
  }

private:
  enum class JITResultType { FAILED, CACHED, NEW };

  JITResultType Type;
  SYCLKernelInfo KernelInfo;
  sycl::detail::string ErrorMessage;
};

extern "C" {

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif // __clang__
JITResult fuseKernels(View<SYCLKernelInfo> KernelInformation,
                      const char *FusedKernelName,
                      View<ParameterIdentity> Identities,
                      BarrierFlags BarriersFlags,
                      View<ParameterInternalization> Internalization,
                      View<jit_compiler::JITConstant> JITConstants);

JITResult materializeSpecConstants(const char *KernelName,
                                   jit_compiler::SYCLKernelBinaryInfo &BinInfo,
                                   View<unsigned char> SpecConstBlob);

JITResult compileSYCL(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
                      View<const char *> UserArgs);

/// Clear all previously set options.
void resetJITConfiguration();

/// Add an option to the configuration.
void addToJITConfiguration(OptionStorage &&Opt);

} // end of extern "C"

} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
