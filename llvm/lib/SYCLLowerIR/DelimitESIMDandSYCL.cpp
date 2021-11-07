
#include "llvm/SyclLowerIR/DelimitEsimdandSycl.h"

#include "llvm/Transforms/Utils/Cloning.h"

//#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
//#include "llvm/ADT/StringSwitch.h"
//#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
//#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
//#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
//#include "llvm/Support/raw_ostream.h"

#include <iostream>
//#include <cctype>
//#include <cstring>
//#include <unordered_map>

#define DEBUG_TYPE "delimit-esimd-and-sycl"

using namespace llvm;

namespace {
SmallPtrSet<Type *, 4> collectGenXVolatileTypes(Module &);
void generateKernelMetadata(Module &);

class DelimitESIMDandSYCLLegacyPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  DelimitESIMDandSYCLLegacyPass() : ModulePass(ID) {
    initializeDelimitESIMDandSYCLLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // run the DelimitESIMDandSYCL pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  DelimitESIMDandSYCLPass Impl;
};
} // namespace

char DelimitESIMDandSYCLLegacyPass::ID = 0;
INITIALIZE_PASS(DelimitESIMDandSYCLLegacyPass, "DelimitESIMDandSYCL",
                "Delimit ESIMD and SYCL code in a module", false, false)

// Public interface to the DelimitESIMDandSYCLPass.
ModulePass *llvm::createDelimitESIMDandSYCLPass() {
  return new DelimitESIMDandSYCLLegacyPass();
}

namespace {

constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";
using FuncPtrSet = SmallPtrSetImpl<Function*>;

template <class ActOnCallF>
void traverseCalls(Function *F, ActOnCallF Action) {
  for (const auto &I : instructions(F)) {
    if (const CallBase *CB = dyn_cast<CallBase>(&I)) {
      if (Function *CF = CB->getCalledFunction()) {
        if (!CF->isDeclaration())
          Action(CF);
      } else {
        llvm_unreachable_internal("Unsupported call form", __FILE__, __LINE__);
      }
    }
  }
}

void prnfset(const char *msg, FuncPtrSet &S) {
  std::cout << msg << ":\n";
  for (const auto *F : S) {
    std::cout << "  " << F->getName().data() << "\n";
  }
}

template <class CfgARootTestF>
void divideModuleCFGs(Module &M, FuncPtrSet &A, FuncPtrSet &AB, CfgARootTestF IsRootInA) {
  SmallPtrSet<Function*, 32> B;
  {
    SmallVector<Function*, 32> Workq;

    // Collect CFG roots and populate work queue with them.
    for (Function &F : M) {
      if (IsRootInA(&F)) {
        A.insert(&F);
        Workq.push_back(&F);
      } else {
        B.insert(&F); // all non-A-roots go to B for now, clean up below
      }
    }
    prnfset("A", A);
    prnfset("B", B);
    // Build and traverse the CFGs.
    while (Workq.size() > 0) {
      Function *F = Workq.pop_back_val();
      B.erase(F); // cleanup B: F is reached from A, then it can't be part of B
      traverseCalls(F, [&A, &Workq](Function *F1) {
        if (A.count(F1) == 0) {
          A.insert(F1);
          Workq.push_back(F1);
        }
      });
    }
  }
  prnfset("A1", A);
  prnfset("B1", B);

  // Now B contains only functions not reacheable from A, but some of A
  // functions can be also reacheable from B - identify them, remove from A and
  // add to AB.
  {
    SmallVector<Function*, 32> Workq(B.begin(), B.end());

    while (Workq.size() > 0) {
      Function *F = Workq.pop_back_val();

      traverseCalls(F, [&A, &B, &AB, &Workq](Function *F1) {
        if (B.count(F1) == 0 && AB.count(F1) == 0) {
          if (A.erase(F1)) {
            // F1 is reacheable both from A and B
            AB.insert(F1);
            Workq.push_back(F1);
          }
        }
      });
    }
  }
  prnfset("A2", A);
  prnfset("AB", AB);
}

Function* clone(Function *F, Twine suff) {
  ValueToValueMapTy VMap;
  Function *Res = CloneFunction(F, VMap);
  Res->setName(F->getName() + "." + suff);
  Res->copyAttributesFrom(F);
  Res->setLinkage(F->getLinkage());
  Res->setVisibility(F->getVisibility());
  return Res;
}
} // namespace

PreservedAnalyses DelimitESIMDandSYCLPass::run(Module &M, ModuleAnalysisManager &) {
  SmallPtrSet<Function*, 32> EsimdOnlyFuncs; // called only from ESIMD CFGs
  SmallPtrSet<Function*, 32> CommonFuncs; // called both from Esimd and SYCL

  // Collect the 3 sets of functions based on CFG: 
  divideModuleCFGs(M, EsimdOnlyFuncs, CommonFuncs, [](Function *F) {
    return F->getMetadata(ESIMD_MARKER_MD) != nullptr;
  });
  if (EsimdOnlyFuncs.size() == 0)
    return PreservedAnalyses::all();

  DenseMap<Value*, Value*> Sycl2Esimd; // map common function to its Esimd clone

  // Clone common functions:
  for (auto *F : CommonFuncs) {
    Function *EsimdF = clone(F, "esimd");
    EsimdOnlyFuncs.erase(F);
    EsimdOnlyFuncs.insert(EsimdF);
    Sycl2Esimd[F] = EsimdF;
  }
  bool Modified = false;

  // Mark all Esimd functions with proper metadata:
  for (auto *F : EsimdOnlyFuncs) {
    if (!F->getMetadata(ESIMD_MARKER_MD)) {
      F->setMetadata(ESIMD_MARKER_MD, llvm::MDNode::get(F->getContext(), {}));
      Modified = true;
    }
  }
  // Now replace common functions usages within Esimd CFGs with the clones.
  // TODO now the "usage" means only calls, function pointers are not supported.
  for (auto *F : CommonFuncs) {
    auto *EsimdF = Sycl2Esimd[F];
    F->replaceUsesWithIf(EsimdF, [F, &Modified, &EsimdOnlyFuncs](Use &U) -> bool {
      if (const CallBase *CB = dyn_cast<const CallBase>(U.getUser())) {
        Function *CF = CB->getCalledFunction();
        if (CF != F)
          llvm_unreachable_internal("Unsupported call form", __FILE__, __LINE__);
        // see if the call happens within a function from the ESIMD call graph:
        bool CalledFromEsimd = EsimdOnlyFuncs.count(CB->getFunction()) > 0;
        Modified |= CalledFromEsimd;
        return CalledFromEsimd;
      }
      llvm_unreachable_internal("Unsupported use of function", __FILE__, __LINE__);
      return false;
    });
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
