//==---- Internalization.h - Pass to internalize global memory accesses ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_INTERNALIZATION_H
#define SYCL_FUSION_PASSES_INTERNALIZATION_H

#include <llvm/IR/PassManager.h>

namespace llvm {
/// Performs private and local promotion of the selected arguments. Local
/// promotion consists on casting to the local address space (3); private
/// promotion, on performing an allocation (address space 0) and replacing each
/// usage of the parameter by such allocation.
class SYCLInternalizer : public PassInfoMixin<SYCLInternalizer> {
public:
  constexpr static StringLiteral Key{"sycl.kernel.promote"};
  constexpr static StringLiteral LocalSizeKey{"sycl.kernel.promote.localsize"};
  constexpr static StringLiteral ElemSizeKey{"sycl.kernel.promote.elemsize"};

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // SYCL_FUSION_PASSES_INTERNALIZATION_H
