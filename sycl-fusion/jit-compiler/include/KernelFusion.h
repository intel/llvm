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
#include <cassert>
#include <string>
#include <variant>
#include <vector>

namespace jit_compiler {

class FusionResult {
public:
  explicit FusionResult(std::string &&ErrorMessage)
      : Type{FusionResultType::FAILED}, Value{std::move(ErrorMessage)} {}

  explicit FusionResult(SYCLKernelInfo KernelInfo, bool Cached = false)
      : Type{(Cached) ? FusionResultType::CACHED : FusionResultType::NEW},
        Value{std::forward<SYCLKernelInfo>(KernelInfo)} {}

  bool failed() const { return Type == FusionResultType::FAILED; }

  bool cached() const { return Type == FusionResultType::CACHED; }

  const std::string &getErrorMessage() const {
    assert(failed() && std::holds_alternative<std::string>(Value) &&
           "No error message present");
    return std::get<std::string>(Value);
  }

  const SYCLKernelInfo &getKernelInfo() const {
    assert(!failed() && std::holds_alternative<SYCLKernelInfo>(Value) &&
           "No kernel info");
    return std::get<SYCLKernelInfo>(Value);
  }

private:
  enum class FusionResultType { FAILED, CACHED, NEW };
  FusionResultType Type;

  std::variant<std::string, SYCLKernelInfo> Value;
};

class KernelFusion {

public:
  static FusionResult fuseKernels(
      Config &&JITConfig, const std::vector<SYCLKernelInfo> &KernelInformation,
      const char *FusedKernelName, jit_compiler::ParamIdentList &Identities,
      BarrierFlags BarriersFlags,
      const std::vector<jit_compiler::ParameterInternalization>
          &Internalization,
      const std::vector<jit_compiler::JITConstant> &JITConstants);
};

} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
