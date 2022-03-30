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
//       sycl::ext::intel::experimental::esimd::vector_type_t<Ty, M>
//       __esimd_rdregion(sycl::ext::intel::experimental::esimd::vector_type_t<Ty,
//       N> Input,
//         uint16_t Offset);
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_LOWERESIMD_H
#define LLVM_SYCLLOWERIR_LOWERESIMD_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;
class ModulePass;
class PassRegistry;

/// SPIRV (ESIMD) target specific pass to transform ESIMD specific constructs
/// like intrinsics to a form parsable by the ESIMD-aware SPIRV translator.
class SYCLLowerESIMDPass : public PassInfoMixin<SYCLLowerESIMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  size_t runOnFunction(Function &F, SmallPtrSet<Type *, 4> &);
};

ModulePass *createSYCLLowerESIMDPass();
void initializeSYCLLowerESIMDLegacyPassPass(PassRegistry &);

class ESIMDLowerLoadStorePass : public PassInfoMixin<ESIMDLowerLoadStorePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

FunctionPass *createESIMDLowerLoadStorePass();
void initializeESIMDLowerLoadStorePass(PassRegistry &);

// Pass converts simd* function parameters and globals to
// llvm's first-class vector* type.
class ESIMDLowerVecArgPass : public PassInfoMixin<ESIMDLowerVecArgPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  DenseMap<GlobalVariable *, GlobalVariable *> OldNewGlobal;

  Function *rewriteFunc(Function &F);
  Type *getSimdArgPtrTyOrNull(Value *arg);
  void fixGlobals(Module &M);
  void removeOldGlobals();
};

ModulePass *createESIMDLowerVecArgPass();
void initializeESIMDLowerVecArgLegacyPassPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_LOWERESIMD_H
