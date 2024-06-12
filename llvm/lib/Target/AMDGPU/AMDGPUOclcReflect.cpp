//===- AMDGPUOclcReflect.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass searches for occurences of the AMDGPU_OCLC_REFLECT function, and
// replaces the calls with some val dependent on the operand of the func. This
// can be used to reflect across different implementations of functions at
// compile time based on a compiler flag or some other means. This pass
// currently supports use cases:
//
// 1. Choose a safe or unsafe version of atomic_xor at compile time, which can
//    be chosen at compile time by setting the flag
//    --amdgpu-oclc-unsafe-int-atomics=true.
// 2. Choose a safe or unsafe version of atomic_fadd at compile time, which can
//    be chosen at compile time by setting the flag
//    --amdgpu-oclc-unsafe-fp-atomics=true.
//
// This pass is similar to the NVPTX pass NVVMReflect.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define AMDGPU_OCLC_REFLECT "__oclc_amdgpu_reflect"

static cl::opt<bool>
    AMDGPUReflectEnabled("amdgpu-oclc-reflect-enable", cl::init(true),
                         cl::Hidden,
                         cl::desc("AMDGPU reflection, enabled by default"));
static cl::opt<bool> AMDGPUUnsafeIntAtomicsEnable(
    "amdgpu-oclc-unsafe-int-atomics", cl::init(false), cl::Hidden,
    cl::desc("Should unsafe int atomics be chosen. Disabled by default."));
static cl::opt<bool> AMDGPUUnsafeFPAtomicsEnable(
    "amdgpu-oclc-unsafe-fp-atomics", cl::init(false), cl::Hidden,
    cl::desc("Should unsafe fp atomics be chosen. Disabled by default."));

PreservedAnalyses AMDGPUOclcReflectPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  if (!AMDGPUReflectEnabled)
    return PreservedAnalyses::all();

  if (F.getName() == AMDGPU_OCLC_REFLECT) {
    assert(F.isDeclaration() &&
           "__oclc_amdgpu_reflect function should not have a body");
    return PreservedAnalyses::all();
  }

  SmallVector<CallInst *, 4> ToRemove;

  for (Instruction &I : instructions(F)) {
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call)
      continue;
    if (Function *Callee = Call->getCalledFunction();
        !Callee || Callee->getName() != AMDGPU_OCLC_REFLECT)
      continue;

    assert(Call->arg_size() == 1 &&
           "Wrong number of operands to __oclc_amdgpu_reflect function");

    ToRemove.push_back(Call);
  }

  if (!ToRemove.size())
    return PreservedAnalyses::all();

  for (CallInst *Call : ToRemove) {
    const Value *Str = Call->getArgOperand(0);
    const Value *Operand = cast<Constant>(Str)->getOperand(0);
    StringRef ReflectArg = cast<ConstantDataSequential>(Operand)->getAsString();
    ReflectArg = ReflectArg.drop_back(1);

    if (ReflectArg == "AMDGPU_OCLC_UNSAFE_INT_ATOMICS") {
      int ReflectVal = AMDGPUUnsafeIntAtomicsEnable ? 1 : 0;
      Call->replaceAllUsesWith(ConstantInt::get(Call->getType(), ReflectVal));
    } else if (ReflectArg == "AMDGPU_OCLC_UNSAFE_FP_ATOMICS") {
      int ReflectVal = AMDGPUUnsafeFPAtomicsEnable ? 1 : 0;
      Call->replaceAllUsesWith(ConstantInt::get(Call->getType(), ReflectVal));
    } else {
      report_fatal_error("Invalid arg passed to __oclc_amdgpu_reflect");
    }
    Call->eraseFromParent();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
