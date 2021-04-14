//===-- ESIMDVecArgPass.cpp - lower Close To Metal (CM) constructs --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Change in function parameter type from simd* to native llvm vector type for
// cmc compiler to generate correct code for subroutine parameter passing and
// globals:
//
// Old IR:
// ======
//
// Parameter %0 is of type simd*
// define dso_local spir_func void @_Z3fooPiN2cm3gen4simdIiLi16EEE(i32
// addrspace(4)* %C,
//      "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" * %0)
//      local_unnamed_addr #2 {
//
// New IR:
// ======
//
// Translate simd* parameter (#1) to vector <16 x 32>* type and insert bitcast.
// All users of old parameter will use result of the bitcast.
//
// define dso_local spir_func void @_Z3fooPiN2cm3gen4simdIiLi16EEE(i32
// addrspace(4)* %C,
//      <16 x i32>* %0) local_unnamed_addr #2 {
// entry:
// % 1 = bitcast<16 x i32> * % 0 to %
// "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" *
//
// It is OK not to rewrite a function (for example, when its address is taken)
// since it does not affect correctness. But that may lead to vector backend
// not being able to hold the value in GRF and generate memory references.
//
// Change in global variables:
//
// Old IR:
// ======
// @vc = global %"class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd"
//          zeroinitializer, align 64 #0
//
// % call.cm.i.i = tail call<16 x i32> @llvm.genx.vload.v16i32.p4v16i32(
//    <16 x i32> addrspace(4) * getelementptr(
//    % "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd",
//    % "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" addrspace(4) *
//    addrspacecast(% "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" * @vc to
//    % "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" addrspace(4) *), i64 0,
//    i32 0))
//
// New IR:
// ======
//
// @0 = dso_local global <16 x i32> zeroinitializer, align 64 #0 <-- New Global
// Variable
//
// % call.cm.i.i = tail call<16 x i32> @llvm.genx.vload.v16i32.p4v16i32(
//        <16 x i32> addrspace(4) * getelementptr(
//        % "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd",
//        % "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" addrspace(4) *
//        addrspacecast(% "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" *
//        bitcast(<16 x i32> * @0 to
//        %"class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" *) to %
//        "class._ZTSN2cm3gen4simdIiLi16EEE.cm::gen::simd" addrspace(4) *),
//        i64 0, i32 0))
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LowerESIMD.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "ESIMDLowerVecArg"

namespace {
class ESIMDLowerVecArgLegacyPass : public ModulePass {
public:
  static char ID;
  ESIMDLowerVecArgLegacyPass() : ModulePass(ID) {
    initializeESIMDLowerVecArgLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    Impl.run(M, MAM);
    return true;
  }

  bool doInitialization(Module &M) override { return false; }

private:
  ESIMDLowerVecArgPass Impl;
};
} // namespace

char ESIMDLowerVecArgLegacyPass::ID = 0;
INITIALIZE_PASS(ESIMDLowerVecArgLegacyPass, "ESIMDLowerVecArg",
                "Translate simd ptr to native vector type", false, false)

// Public interface to VecArgPass
ModulePass *llvm::createESIMDLowerVecArgPass() {
  return new ESIMDLowerVecArgLegacyPass();
}

// Return ptr to first-class vector type if Value is a simd*, else return
// nullptr.
Type *ESIMDLowerVecArgPass::getSimdArgPtrTyOrNull(Value *arg) {
  auto ArgType = dyn_cast<PointerType>(arg->getType());
  if (!ArgType || !ArgType->getElementType()->isStructTy())
    return nullptr;
  auto ContainedType = ArgType->getElementType();
  if ((ContainedType->getStructNumElements() != 1) ||
      !ContainedType->getStructElementType(0)->isVectorTy())
    return nullptr;
  return PointerType::get(ContainedType->getStructElementType(0),
                          ArgType->getPointerAddressSpace());
}

// F may have multiple arguments of type simd*. This
// function updates all parameters along with call
// call sites of F.
Function *ESIMDLowerVecArgPass::rewriteFunc(Function &F) {
  FunctionType *FTy = F.getFunctionType();
  Type *RetTy = FTy->getReturnType();
  SmallVector<Type *, 8> ArgTys;

  for (unsigned int i = 0; i != F.arg_size(); i++) {
    auto Arg = F.getArg(i);
    Type *NewTy = getSimdArgPtrTyOrNull(Arg);
    if (NewTy) {
      // Copy over byval type for simd* type
      ArgTys.push_back(NewTy);
    } else {
      // Transfer all non-simd ptr arguments
      ArgTys.push_back(Arg->getType());
    }
  }

  FunctionType *NFTy = FunctionType::get(RetTy, ArgTys, false);

  // Create new function body and insert into the module
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getName());
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);

  SmallVector<ReturnInst *, 8> Returns;
  SmallVector<BitCastInst *, 8> BitCasts;
  ValueToValueMapTy VMap;
  for (unsigned int I = 0; I != F.arg_size(); I++) {
    auto Arg = F.getArg(I);
    Type *newTy = getSimdArgPtrTyOrNull(Arg);
    if (newTy) {
      // bitcast vector* -> simd*
      auto BitCast = new BitCastInst(NF->getArg(I), Arg->getType());
      BitCasts.push_back(BitCast);
      VMap.insert(std::make_pair(Arg, BitCast));
      continue;
    }
    VMap.insert(std::make_pair(Arg, NF->getArg(I)));
  }

  llvm::CloneFunctionInto(NF, &F, VMap,
                          CloneFunctionChangeType::LocalChangesOnly, Returns);

  // insert bitcasts in new function only if its a definition
  for (auto &B : BitCasts) {
    if (!F.isDeclaration())
      NF->begin()->getInstList().push_front(B);
    else
      delete B;
  }

  NF->takeName(&F);

  // Fix call sites
  SmallVector<std::pair<Instruction *, Instruction *>, 10> OldNewInst;
  for (auto &use : F.uses()) {
    // Use must be a call site
    SmallVector<Value *, 10> Params;
    auto Call = cast<CallInst>(use.getUser());
    // Variadic functions not supported
    assert(!Call->getFunction()->isVarArg() &&
           "Variadic functions not supported");
    for (unsigned int I = 0; I < Call->getNumArgOperands(); I++) {
      auto SrcOpnd = Call->getOperand(I);
      auto NewTy = getSimdArgPtrTyOrNull(SrcOpnd);
      if (NewTy) {
        auto BitCast = new BitCastInst(SrcOpnd, NewTy, "", Call);
        Params.push_back(BitCast);
      } else {
        if (SrcOpnd != &F)
          Params.push_back(SrcOpnd);
        else
          Params.push_back(NF);
      }
    }
    // create new call instruction
    auto NewCallInst = CallInst::Create(NFTy, NF, Params, "");
    NewCallInst->setCallingConv(F.getCallingConv());
    OldNewInst.push_back(std::make_pair(Call, NewCallInst));
  }

  for (auto InstPair : OldNewInst) {
    auto OldInst = InstPair.first;
    auto NewInst = InstPair.second;
    ReplaceInstWithInst(OldInst, NewInst);
  }

  F.eraseFromParent();

  return NF;
}

// This function creates new global variables of type vector* type
// when old one is of simd* type.
void ESIMDLowerVecArgPass::fixGlobals(Module &M) {
  for (auto &G : M.getGlobalList()) {
    auto NewTy = getSimdArgPtrTyOrNull(&G);
    if (NewTy && !G.user_empty()) {
      // Peel off ptr type that getSimdArgPtrTyOrNull applies
      NewTy = NewTy->getPointerElementType();
      auto InitVal =
          G.hasInitializer() && isa<UndefValue>(G.getInitializer())
              ? static_cast<ConstantData *>(UndefValue::get(NewTy))
              : static_cast<ConstantData *>(ConstantAggregateZero::get(NewTy));
      auto NewGlobalVar =
          new GlobalVariable(NewTy, G.isConstant(), G.getLinkage(), InitVal, "",
                             G.getThreadLocalMode(), G.getAddressSpace());
      NewGlobalVar->setExternallyInitialized(G.isExternallyInitialized());
      NewGlobalVar->copyAttributesFrom(&G);
      NewGlobalVar->takeName(&G);
      NewGlobalVar->copyMetadata(&G, 0);
      M.getGlobalList().push_back(NewGlobalVar);
      OldNewGlobal.insert(std::make_pair(&G, NewGlobalVar));
    }
  }

  removeOldGlobals();
}

// Remove old global variables from the program.
void ESIMDLowerVecArgPass::removeOldGlobals() {
  for (auto &G : OldNewGlobal) {
    auto OldGlob = G.first;
    auto NewGlobal = G.second;
    OldGlob->replaceAllUsesWith(
        ConstantExpr::getBitCast(NewGlobal, OldGlob->getType()));
    OldGlob->eraseFromParent();
  }
}

PreservedAnalyses ESIMDLowerVecArgPass::run(Module &M,
                                            ModuleAnalysisManager &) {
  fixGlobals(M);

  SmallVector<Function *, 10> functions;
  for (auto &F : M) {
    // Skip functions that are used through function pointers.
    if (!F.hasAddressTaken())
      functions.push_back(&F);
  }

  for (auto F : functions) {
    for (unsigned int I = 0; I != F->arg_size(); I++) {
      auto Arg = F->getArg(I);
      if (getSimdArgPtrTyOrNull(Arg)) {
        rewriteFunc(*F);
        break;
      }
    }
  }
  return PreservedAnalyses::none();
}
