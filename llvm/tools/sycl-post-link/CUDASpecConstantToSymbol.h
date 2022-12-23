//===---- CUDASpecConstantToSymbol.h - CUDA Spec Constants To Symbol Pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass creates a global variable to be used for spec constants, rewrites
// the signature of the kernel to remove the spec constant argument and fixes
// up all accesses. The pi_cuda plugin uses CUDA API to copy the value of spec
// constants to the symbol.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class CUDASpecConstantToSymbolPass
    : public PassInfoMixin<CUDASpecConstantToSymbolPass> {
public:
  CUDASpecConstantToSymbolPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

private:
  // Communicate to the plugin that the spec constant implicit arg has no uses
  // and doesn't need to be removed from the kernel, hence no need for the
  // symbol setup.
  //
  // @param M [in] Module to work on.
  void setUpPlaceholderEntries(Module &M);

  // For each kernel loop over its specialization constants and accumulate
  // their size, then create a global variable of that size.
  //
  // @param MD [in] Root node containing all the kernels making use of
  // specialization constants.
  void allocatePerKernelGlobals(NamedMDNode *MD);

  // Drop the implicit argument from the kernel signature and replace all uses
  // of it with accesses to the global variable.
  //
  // @param MD [in] Root node containing all the kernels making use of
  // specialization constants.
  void rewriteKernelSignature(NamedMDNode *MD);

  // Loop over all uses of Argument A (the, now defunct, implicit argument) and
  // fix them up with accesses to the global variable.
  //
  // @param A [in] Argument which uses are replaced.
  // @param KernelName [in] The name of the kernel.
  void fixupSpecConstantUses(Argument *A, const StringRef KernelName);

private:
  // NOTE: GlobalNamePrefix is read by pi_cuda and must not be changed.
  const char *GlobalNamePrefix = "sycl_specialization_constants_kernel_";
  // A mapping between kernel names and global variables allocated to store
  // spec constants.
  DenseMap<StringRef, GlobalVariable *> SpecConstGlobals;
};

} // namespace llvm
