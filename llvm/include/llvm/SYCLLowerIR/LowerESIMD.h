//===---- LowerESIMD.h - lower Explicit SIMD (ESIMD) constructs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers CM-specific LLVM IR constructs coming out of the front-end. These are:
// - ESIMD intrinsics, e.g.:
//     template <typename Ty, int N, int M, int VStride, int Width,
//       int Stride, int ParentWidth = 0>
//       sycl::intel::gpu::vector_type_t<Ty, M>
//       __esimd_rdregion(sycl::intel::gpu::vector_type_t<Ty, N> Input,
//         uint16_t Offset);
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_LOWERESIMD_H
#define LLVM_SYCLLOWERIR_LOWERESIMD_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// SPIRV (ESIMD) target specific pass to transform ESIMD specific constructs
/// like intrinsics to a form parsable by the ESIMD-aware SPIRV translator.
class SYCLLowerESIMDPass : public PassInfoMixin<SYCLLowerESIMDPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &,
                        SmallPtrSet<Type *, 4> &GVTS);
};

FunctionPass *createSYCLLowerESIMDPass();
void initializeSYCLLowerESIMDLegacyPassPass(PassRegistry &);

FunctionPass *createESIMDLowerLoadStorePass();
void initializeESIMDLowerLoadStorePass(PassRegistry &);

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_LOWERESIMD_H
