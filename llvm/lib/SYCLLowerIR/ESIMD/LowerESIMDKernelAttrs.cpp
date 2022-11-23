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

#define DEBUG_TYPE "LowerESIMDKernelAttrs"

using namespace llvm;

namespace llvm {

// Checks if Call Instruction corresponds to InvokeSimd function call.
bool isInvokeSimdBuiltinCall(const CallInst *CI) {
  Function *F = CI->getCalledFunction();

  return F && F->getName().startswith(esimd::INVOKE_SIMD_PREF);
}

// Checks the use of a function address being stored in a memory.
// Returns false if the function address is used as an argument for
// invoke_simd function call, true otherwise.
bool checkFunctionAddressUse(const Value *address) {
  if (address == nullptr)
    return true;

  SmallPtrSet<const Use *, 4> Uses;
  llvm::esimd::collectUsesLookThroughCasts(address, Uses);

  for (const Use *U : Uses) {
    Value *V = U->getUser();

    if (auto *StI = dyn_cast<StoreInst>(V)) {
      if (U != &StI->getOperandUse(StoreInst::getPointerOperandIndex()))
        return false; // this is double indirection - not supported

      V = esimd::stripCasts(StI->getPointerOperand());
      if (!isa<AllocaInst>(V))
        return false; // unsupported case of data flow through non-local memory

      if (auto *LI = dyn_cast<LoadInst>(V)) {
        // A value loaded from another address is stored at this address -
        // recurse into the other address
        if (!checkFunctionAddressUse(LI->getPointerOperand()))
          return false;
      }
    } else if (const auto *CI = dyn_cast<CallInst>(V)) {
      // if __builtin_invoke_simd uses the pointer, do not traverse the function
      if (isInvokeSimdBuiltinCall(CI))
        return false;

    } else if (isa<LoadInst>(V)) {
      if (!checkFunctionAddressUse(V))
        return false;

    } else
      return false;
  }

  return true;
}

// Filter function for graph traverse to filter out cases when a function
// is used as an argument for InvokeSimd call
bool filterInvokeSimdUse(const Instruction *I) {
  // if the instruction is to store address of a function, check if it is later
  // used by InvokeSimd.
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    const Value *Addr = SI->getPointerOperand();
    return checkFunctionAddressUse(Addr);
  }
  return true;
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
