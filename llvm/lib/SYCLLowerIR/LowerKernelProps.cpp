//===---- LowerKernelProps.cpp - lower __sycl_set_kernel_properties ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Finds and lowers __sycl_set_kernel_properties calls: converts the call to
// function attributes and adds those attributes to all kernels which can
// potentially call this intrinsic.

#include "llvm/SYCLLowerIR/LowerKernelProps.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "LowerKernelProps"

using namespace llvm;

namespace {

constexpr char SET_KERNEL_PROPS_FUNC_NAME[] =
    "_Z28__sycl_set_kernel_propertiesi";

// Kernel property identifiers. Should match ones in
// sycl/include/sycl/ext/intel/experimental/kernel_properties.hpp
enum property_ids { use_large_grf = 0 };

void processSetKernelPropertiesCall(CallInst &CI) {
  auto F = CI.getFunction();
  auto *ArgV = CI.getArgOperand(0);

  if (!isa<ConstantInt>(ArgV)) {
    llvm::report_fatal_error(
        llvm::Twine(__FILE__ " ") +
        "integral constant is expected for set_kernel_properties");
  }
  uint64_t PropID = cast<llvm::ConstantInt>(ArgV)->getZExtValue();

  switch (PropID) {
  case property_ids::use_large_grf:
    // TODO: Keep track of traversed functions to avoid repeating traversals
    // over same function.
    llvm::sycl::utils::traverseCallgraphUp(F, [](Function *GraphNode) {
      GraphNode->addFnAttr(::sycl::kernel_props::ATTR_LARGE_GRF);
    });
    break;
  default:
    assert(false && "Invalid property id");
  }
}

} // namespace

namespace llvm {
PreservedAnalyses SYCLLowerKernelPropsPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  Function *F = M.getFunction(SET_KERNEL_PROPS_FUNC_NAME);

  if (!F) {
    return PreservedAnalyses::all();
  }
  bool Modified = false;
  SmallVector<User *, 4> Users(F->users());

  for (User *Usr : Users) {
    // a call can be the only use of the __sycl_set_kernel_properties built-in
    CallInst *CI = cast<CallInst>(Usr);
    processSetKernelPropertiesCall(*CI);
    CI->eraseFromParent();
    Modified = true;
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
} // namespace llvm
