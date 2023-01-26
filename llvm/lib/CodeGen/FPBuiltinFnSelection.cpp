//===- FPBuiltinFnSelection.cpp - Pre-ISel intrinsic lowering pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements alternate math library implementation selection for
// llvm.fpbuiltin.* intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FPBuiltinFnSelection.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

#define DEBUG_TYPE "fpbuiltin-fn-selection"

static bool replaceWithAltMathFunction(FPBuiltinIntrinsic &BuiltinCall,
                                       const StringRef ImplName) {
  Module *M = BuiltinCall.getModule();

  Function *OldFunc = BuiltinCall.getCalledFunction();

  // Check if the alt math library function is already declared in this module,
  // otherwise insert it.
  Function *ImplFunc = M->getFunction(ImplName);
  if (!ImplFunc) {
    ImplFunc = Function::Create(OldFunc->getFunctionType(),
                                Function::ExternalLinkage, ImplName, *M);
    // TODO: Copy non-builtin attributes ImplFunc->copyAttributesFrom(OldFunc);
  }

  // Replace the call to the fpbuiltin intrinsic with a call
  // to the corresponding function from the alternate math library.
  IRBuilder<> IRBuilder(&BuiltinCall);
  SmallVector<Value *> Args(BuiltinCall.args());
  // Preserve the operand bundles.
  SmallVector<OperandBundleDef, 1> OpBundles;
  BuiltinCall.getOperandBundlesAsDefs(OpBundles);
  CallInst *Replacement = IRBuilder.CreateCall(ImplFunc, Args, OpBundles);
  assert(OldFunc->getFunctionType() == ImplFunc->getFunctionType() &&
         "Expecting function types to be identical");
  BuiltinCall.replaceAllUsesWith(Replacement);
  // TODO: fpbuiltin.sincos won't be reported as an FPMathOperator
  //       Do we need to do anything about that?
  if (isa<FPMathOperator>(Replacement)) {
    // Preserve fast math flags for FP math.
    Replacement->copyFastMathFlags(&BuiltinCall);
  }

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `"
                    << OldFunc->getName() << "` with call to `" << ImplName
                    << "`.\n");
  return true;
}

static bool selectFnForFPBuiltinCalls(const TargetLibraryInfo &TLI,
                                      FPBuiltinIntrinsic &BuiltinCall) {
  LLVM_DEBUG({
    dbgs() << "Selecting an implementation for "
           << BuiltinCall.getCalledFunction()->getName() << " with accuracy = ";
    if (BuiltinCall.getRequiredAccuracy() == std::nullopt)
      dbgs() << "(none)\n";
    else
      dbgs() << BuiltinCall.getRequiredAccuracy().value() << "\n";
  });

//  if (BuiltinCall->hasUnecognizedFPAttrs.push_back();
  StringSet RecognizedAttrs = { FPBuiltinIntrinsic::FP_MAX_ERROR };
  if (BuiltinCall.hasUnrecognizedFPAttrs(RecognizedAttrs)) {
    report_fatal_error(
        Twine(BuiltinCall.getCalledFunction()->getName()) +
            Twine(" was called with unrecognized floating-point attributes.\n"),
        false);
    return false;
  }



  /// Call TLI to select a function implementation to call
  StringRef ImplName = TLI.selectFPBuiltinImplementation(&BuiltinCall);
  if (ImplName.empty()) {
    LLVM_DEBUG(dbgs() << "No matching implementation found!\n");
    std::string RequiredAccuracy;
    if (BuiltinCall.getRequiredAccuracy() == std::nullopt)
      RequiredAccuracy = "(none)";
    else
      RequiredAccuracy =
          formatv("{0}", BuiltinCall.getRequiredAccuracy().value());

    report_fatal_error(Twine(BuiltinCall.getCalledFunction()->getName()) +
                       Twine(" was called with required accuracy = ") +
                       Twine(RequiredAccuracy) + 
                       Twine(" but no suitable implementation was found.\n"),
                       false);
    return false;
  }

  LLVM_DEBUG(dbgs() << "Selected " << ImplName << "\n");

  return replaceWithAltMathFunction(BuiltinCall, ImplName);
}

static bool runImpl(const TargetLibraryInfo &TLI, Function &F) {
  bool Changed = false;
  SmallVector<FPBuiltinIntrinsic *> ReplacedCalls;
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<FPBuiltinIntrinsic>(&I)) {
      if (selectFnForFPBuiltinCalls(TLI, *CI)) {
        ReplacedCalls.push_back(CI);
        Changed = true;
      }
    }
  }
  // Erase the calls to the intrinsics that have been replaced
  // with calls to the alternate math library.
  for (auto *CI : ReplacedCalls) {
    CI->eraseFromParent();
  }
  return Changed;
}

namespace {

class FPBuiltinFnSelectionLegacyPass : public FunctionPass {
public:
  static char ID;

  FPBuiltinFnSelectionLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);

    return runImpl(*TLI, F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<TargetLibraryInfoWrapperPass>();
  }
};

} // end anonymous namespace

char FPBuiltinFnSelectionLegacyPass::ID;

INITIALIZE_PASS_BEGIN(FPBuiltinFnSelectionLegacyPass, DEBUG_TYPE,
                      "FPBuiltin Function Selection", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(FPBuiltinFnSelectionLegacyPass, DEBUG_TYPE,
                    "FPBuiltin Function Selection", false, false)

FunctionPass *llvm::createFPBuiltinFnSelectionPass() {
  return new FPBuiltinFnSelectionLegacyPass;
}

PreservedAnalyses FPBuiltinFnSelectionPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  bool Changed = runImpl(TLI, F);
  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<TargetLibraryAnalysis>();
    return PA;
  } else {
    // The pass did not replace any calls, hence it preserves all analyses.
    return PreservedAnalyses::all();
  }
}
