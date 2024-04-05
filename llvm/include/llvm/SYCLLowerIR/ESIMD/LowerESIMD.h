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
  SYCLLowerESIMDPass(bool ModuleContainsScalar = true)
      : ModuleContainsScalarCode(ModuleContainsScalar) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  bool prepareForAlwaysInliner(Module &M);
  size_t runOnFunction(Function &F, SmallPtrSetImpl<Type *> &);
  bool ModuleContainsScalarCode;
};

ModulePass *createSYCLLowerESIMDPass();
void initializeSYCLLowerESIMDLegacyPassPass(PassRegistry &);

class ESIMDLowerLoadStorePass : public PassInfoMixin<ESIMDLowerLoadStorePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

FunctionPass *createESIMDLowerLoadStorePass();
void initializeESIMDLowerLoadStorePass(PassRegistry &);

// - Converts simd* function parameters and return values passed by pointer to
// pass-by-value
//   (where possible)
// - Converts globals of type simd* to simd::raw_vector_t* globals (llvm vector
// type pointer)
class ESIMDOptimizeVecArgCallConvPass
    : public PassInfoMixin<ESIMDOptimizeVecArgCallConvPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

// Lowers calls __esimd_slm_alloc, __esimd_slm_free and __esimd_slm_init APIs.
// See more details in the .cpp file.
size_t lowerSLMReservationCalls(Module &M);

// Lowers calls to __esimd_set_kernel_properties
class SYCLLowerESIMDKernelPropsPass
    : public PassInfoMixin<SYCLLowerESIMDKernelPropsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

// Fixes ESIMD Kernel attributes for wrapper functions for ESIMD kernels
class SYCLFixupESIMDKernelWrapperMDPass
    : public PassInfoMixin<SYCLFixupESIMDKernelWrapperMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

class ESIMDRemoveHostCodePass : public PassInfoMixin<ESIMDRemoveHostCodePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

class ESIMDRemoveOptnoneNoinlinePass
    : public PassInfoMixin<ESIMDRemoveOptnoneNoinlinePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_LOWERESIMD_H
