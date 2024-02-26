//=-- ESIMDRemoveOptnoneNoinline.cpp - remove optnone/noinline for ESIMD --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=------------------------------------------------------------------------=//
// The GPU backend for ESIMD does not support debugging and we often see crashes
// or wrong answers if we do not optimize. Remove optnone and noinline from
// ESIMD functions so users at least can run their programs and get the
// right answer.
//=------------------------------------------------------------------------=//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::esimd;

cl::opt<bool> AllowOptnoneNoinline(
    "esimd-allow-optnone-noinline", llvm::cl::Optional, llvm::cl::Hidden,
    llvm::cl::desc("Allow optnone and noinline."), llvm::cl::init(false));

PreservedAnalyses ESIMDRemoveOptnoneNoinlinePass::run(Module &M,
                                                      ModuleAnalysisManager &) {
  if (AllowOptnoneNoinline)
    return PreservedAnalyses::all();

  // TODO: Remove this pass once VC supports debugging.
  bool Modified = false;
  for (auto &F : M.functions()) {
    if (!isESIMD(F) || F.hasFnAttribute("CMGenxSIMT"))
      continue;
    F.removeFnAttr(Attribute::OptimizeNone);
    F.removeFnAttr(Attribute::NoInline);
    Modified = true;
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
