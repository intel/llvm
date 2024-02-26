//==- KernelFusion.h - Public interface of JIT compiler for kernel fusion --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
#define SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H

#include "DynArray.h"
#include "Kernel.h"
#include "Options.h"
#include "Parameter.h"
#include "View.h"

#include <cassert>

namespace jit_compiler {

class FusionResult {
public:
  explicit FusionResult(const char *ErrorMessage)
      : Type{FusionResultType::FAILED}, KernelInfo{},
        ErrorMessage{ErrorMessage} {}

  explicit FusionResult(const SYCLKernelInfo &KernelInfo, bool Cached = false)
      : Type{(Cached) ? FusionResultType::CACHED : FusionResultType::NEW},
        KernelInfo(KernelInfo), ErrorMessage{} {}

  bool failed() const { return Type == FusionResultType::FAILED; }

  bool cached() const { return Type == FusionResultType::CACHED; }

  const char *getErrorMessage() const {
    assert(failed() && "No error message present");
    return ErrorMessage.c_str();
  }

  const SYCLKernelInfo &getKernelInfo() const {
    assert(!failed() && "No kernel info");
    return KernelInfo;
  }

private:
  enum class FusionResultType { FAILED, CACHED, NEW };

  FusionResultType Type;
  SYCLKernelInfo KernelInfo;
  DynString ErrorMessage;
};

class KernelFusion {

public:
  static FusionResult
  fuseKernels(View<SYCLKernelInfo> KernelInformation,
              const char *FusedKernelName, View<ParameterIdentity> Identities,
              BarrierFlags BarriersFlags,
              View<ParameterInternalization> Internalization,
              View<jit_compiler::JITConstant> JITConstants);

  /// Clear all previously set options.
  static void resetConfiguration();

  /// Set \p Opt to the value built in-place by \p As.
  template <typename Opt, typename... Args> static void set(Args &&...As) {
    set(new Opt{std::forward<Args>(As)...});
  }

private:
  /// Take ownership of \p Option and include it in the current configuration.
  static void set(OptionPtrBase *Option);
};

} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
