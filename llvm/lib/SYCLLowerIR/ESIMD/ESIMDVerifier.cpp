//===---------- ESIMDVerifier.cpp - ESIMD-specific IR verification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ESIMD specific IR verification pass. So far it only
// detects invalid API calls in ESIMD context.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDVerifier.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

using namespace llvm;

#define DEBUG_TYPE "esimd-verifier"

// A list of unsupported functions in ESIMD context.
static const char *IllegalFunctions[] = {
    "^cl::sycl::multi_ptr<.+> cl::sycl::accessor<.+>::get_pointer<.+>\\(\\) "
    "const"};

namespace {

class ESIMDVerifierImpl {
  const Module &M;

public:
  ESIMDVerifierImpl(const Module &M) : M(M) {}

  void verify() {
    SmallPtrSet<const Function *, 8u> Visited;
    SmallVector<const Function *, 8u> Worklist;

    auto Add2Worklist = [&Worklist, &Visited](const Function *F) {
      if (Visited.insert(F).second)
        Worklist.push_back(F);
    };

    // Start with adding all ESIMD functions to the work list.
    for (const Function &F : M)
      if (F.hasMetadata("sycl_explicit_simd"))
        Add2Worklist(&F);

    // Then check ESIMD functions and all functions called from ESIMD context
    // for invalid calls.
    while (!Worklist.empty()) {
      const Function *F = Worklist.pop_back_val();
      for (const Instruction &I : instructions(F)) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            continue;

          // Demangle called function name and check if it matches any illegal
          // function name. Report an error if there is a match.
          std::string DemangledName = demangle(Callee->getName().str());
          for (const char *Name : IllegalFunctions) {
            Regex NameRE(Name);
            assert(NameRE.isValid() && "invalid function name regex");
            if (NameRE.match(DemangledName)) {
              std::string ErrorMsg = std::string("function '") + DemangledName +
                                     "' is not supported in ESIMD context";
              F->getContext().emitError(&I, ErrorMsg);
            }
          }

          // Add callee to the list to be analyzed if it is not a declaration.
          if (!Callee->isDeclaration())
            Add2Worklist(Callee);
        }
      }
    }
  }
};

} // end anonymous namespace

PreservedAnalyses ESIMDVerifierPass::run(Module &M, ModuleAnalysisManager &AM) {
  ESIMDVerifierImpl(M).verify();
  return PreservedAnalyses::all();
}

namespace {

struct ESIMDVerifier : public ModulePass {
  static char ID;

  ESIMDVerifier() : ModulePass(ID) {
    initializeESIMDVerifierPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    ESIMDVerifierImpl(M).verify();
    return false;
  }
};

} // end anonymous namespace

char ESIMDVerifier::ID = 0;

INITIALIZE_PASS_BEGIN(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                      false, false)
INITIALIZE_PASS_END(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                    false, false)

ModulePass *llvm::createESIMDVerifierPass() { return new ESIMDVerifier(); }
