//===---- LowerESIMDKernelAttrs - lower __esimd_set_kernel_attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Finds and adds  sycl_explicit_simd attributes to wrapper functions that wrap
// ESIMD kernel functions

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "LowerESIMDKernelAttrs"

using namespace llvm;

namespace llvm {

// Filter function for graph traversal when propagating ESIMD attribute.
// While traversing the call graph, non-call use of the traversed function is
// not added to the graph. The reason is that it is impossible to gurantee
// correct inference of use of that function, in particular to determine if that
// function is used as an argument for invoke_simd. As a result, any use of
// function pointers requires explicit marking of the functions as
// ESIMD_FUNCTION if needed.
bool filterInvokeSimdUse(const Instruction *I, const Function *F) {
  return false;
}

PreservedAnalyses
SYCLFixupESIMDKernelWrapperMDPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Modified = false;
  for (Function &F : M) {
    if (llvm::esimd::isESIMD(F)) {
      // TODO: Keep track of traversed functions to avoid repeating traversals
      // over same function.
      sycl::utils::traverseCallgraphUp(
          &F,
          [&](Function *GraphNode) {
            if (!llvm::esimd::isESIMD(*GraphNode)) {
              GraphNode->setMetadata(
                  llvm::esimd::ESIMD_MARKER_MD,
                  llvm::MDNode::get(GraphNode->getContext(), {}));
              Modified = true;
            }
          },
          false, filterInvokeSimdUse);
    }
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
} // namespace llvm
