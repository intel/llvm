//==--- SYCLCP.h - Pass for constant propagation as part of kernel fusion --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_SYCLCP_H
#define SYCL_FUSION_PASSES_SYCLCP_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>

///
/// Pass to promote JIT constants. Replaces each value of the given
/// argument by a constant.
namespace llvm {
class SYCLCP : public PassInfoMixin<SYCLCP> {
public:
  constexpr static StringLiteral Key{"sycl.kernel.constants"};

  PreservedAnalyses run(Module &F, ModuleAnalysisManager &FM);
};
} // namespace llvm

#endif // SYCL_FUSION_PASSES_SYCLCP_H
