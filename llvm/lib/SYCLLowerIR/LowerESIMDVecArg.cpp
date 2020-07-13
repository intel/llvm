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

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "ESIMDLowerVecArg"

namespace llvm {

// Forward declarations
void initializeESIMDLowerVecArgLegacyPassPass(PassRegistry&);
ModulePass *createESIMDLowerVecArgPass();

// Pass converts simd* function parameters and globals to
// llvm's first-class vector* type.
class ESIMDLowerVecArgPass {
public:
  bool run(Module &M);

private:
  DenseMap<GlobalVariable *, GlobalVariable *> OldNewGlobal;

  Function *rewriteFunc(Function &F);
  Type *argIsSimdPtr(Value *arg);
  void fixGlobals(Module &M);
  bool hasGlobalConstExpr(Value *V, GlobalVariable *&Global);
  void replaceConstExprWithGlobals(Module &M);
  ConstantExpr *createNewConstantExpr(GlobalVariable *newGlobalVar,
                                      Type *oldGlobalType, Value *old);
  void removeOldGlobals();
};

} // namespace llvm

namespace {
class ESIMDLowerVecArgLegacyPass : public ModulePass {
public:
  static char ID;
  ESIMDLowerVecArgLegacyPass() : ModulePass(ID) {
    initializeESIMDLowerVecArgLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    auto Modified = Impl.run(M);
    return Modified;
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
Type *ESIMDLowerVecArgPass::argIsSimdPtr(Value *arg) {
  auto ArgType = arg->getType();
  if (ArgType->isPointerTy()) {
    auto containedType = ArgType->getPointerElementType();
    if (containedType->isStructTy()) {
      if (containedType->getStructNumElements() == 1 &&
          containedType->getStructElementType(0)->isVectorTy()) {
        return PointerType::get(containedType->getStructElementType(0),
                                ArgType->getPointerAddressSpace());
      }
    }
  }
  return nullptr;
}

// F may have multiple arguments of type simd*. This
// function updates all parameters along with call
// call sites of F.
Function *ESIMDLowerVecArgPass::rewriteFunc(Function &F) {
  FunctionType *FTy = F.getFunctionType();
  Type *RetTy = FTy->getReturnType();
  SmallVector<Type *, 8> ArgTys;
  AttributeList AttrVec;
  const AttributeList &PAL = F.getAttributes();
  // Argument, result of load
  DenseMap<Argument *, Value *> ToModify;
  auto &Context = F.getContext();

  for (unsigned int i = 0; i != F.arg_size(); i++) {
    auto Arg = F.getArg(i);
    Type *NewTy = argIsSimdPtr(Arg);
    if (NewTy) {
      // Copy over byval type for simd* type
      ArgTys.push_back(NewTy);
    } else {
      // Transfer all non-simd ptr arguments
      ArgTys.push_back(Arg->getType());
      AttributeSet Attrs = PAL.getParamAttributes(i);
      if (Attrs.hasAttributes()) {
        AttrBuilder B(Attrs);
        AttrVec = AttrVec.addParamAttributes(Context, i, B);
      }
    }
  }

  FunctionType *NFTy = FunctionType::get(RetTy, ArgTys, false);

  // Add any function attributes
  AttributeSet FnAttrs = PAL.getFnAttributes();
  if (FnAttrs.hasAttributes()) {
    AttrBuilder B(FnAttrs);
    AttrVec = AttrVec.addAttributes(Context, AttributeList::FunctionIndex, B);
  }

  auto RetAttrs = PAL.getRetAttributes();
  if (RetAttrs.hasAttributes()) {
    AttrBuilder B(RetAttrs);
    AttrVec = AttrVec.addAttributes(Context, AttributeList::ReturnIndex, B);
  }

  // Create new function body and insert into the module
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getName());
  NF->copyAttributesFrom(&F);
  NF->setCallingConv(F.getCallingConv());

  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->takeName(&F);
  NF->setSubprogram(F.getSubprogram());

  // Now to splice the body of the old function into the new function
  NF->getBasicBlockList().splice(NF->begin(), F.getBasicBlockList());

  for (unsigned int I = 0; I != F.arg_size(); I++) {
    auto Arg = F.getArg(I);
    Type *newTy = argIsSimdPtr(Arg);
    if (newTy) {
      // Insert bitcast
      // bitcast vector* -> simd*
      auto BitCast = new BitCastInst(NF->getArg(I), Arg->getType());
      NF->begin()->getInstList().push_front(BitCast);
      ToModify.insert(std::make_pair(Arg, nullptr));
      Arg->replaceAllUsesWith(BitCast);
    }
  }

  // Loop over the argument list, transferring uses of the old arguments to the
  // new arguments, also tranferring over the names as well
  Function::arg_iterator I2 = NF->arg_begin();
  unsigned int ArgNo = 0;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E;
       ++I, ++I2, ArgNo++) {
    auto ArgIt = ToModify.find(I);
    if (ArgIt == ToModify.end()) {
      // Transfer old arguments as is
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
    }
  }

  // Fix call sites
  SmallVector<std::pair<Instruction *, Instruction *>, 10> OldNewInst;
  for (auto &use : F.uses()) {
    // Use must be a call site
    SmallVector<Value *, 10> Params;
    auto User = use.getUser();
    if (isa<CallInst>(User)) {
      auto Call = cast<CallInst>(User);
      for (unsigned int I = 0,
                        NumOpnds = cast<CallInst>(Call)->getNumArgOperands();
           I != NumOpnds; I++) {
        auto SrcOpnd = Call->getOperand(I);
        auto NewTy = argIsSimdPtr(SrcOpnd);
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
  }

  for (auto InstPair : OldNewInst) {
    auto OldInst = InstPair.first;
    auto NewInst = InstPair.second;
    ReplaceInstWithInst(OldInst, NewInst);
  }

  F.eraseFromParent();

  return NF;
}

bool ESIMDLowerVecArgPass::hasGlobalConstExpr(Value *V, GlobalVariable *&Global) {
  if (isa<GlobalVariable>(V)) {
    Global = cast<GlobalVariable>(V);
    return true;
  }

  if (isa<ConstantExpr>(V)) {
    auto FirstOpnd = cast<ConstantExpr>(V)->getOperand(0);
    return hasGlobalConstExpr(FirstOpnd, Global);
  }

  return false;
}

// Replace ConstantExpr if it contains old global variable.
ConstantExpr *
ESIMDLowerVecArgPass::createNewConstantExpr(GlobalVariable *NewGlobalVar,
                                         Type *OldGlobalType, Value *Old) {
  ConstantExpr *NewConstantExpr = nullptr;

  if (isa<GlobalVariable>(Old)) {
    NewConstantExpr = cast<ConstantExpr>(
        ConstantExpr::getBitCast(NewGlobalVar, OldGlobalType));
    return NewConstantExpr;
  }

  auto InnerMost = createNewConstantExpr(
      NewGlobalVar, OldGlobalType, cast<ConstantExpr>(Old)->getOperand(0));

  NewConstantExpr = cast<ConstantExpr>(
      cast<ConstantExpr>(Old)->getWithOperandReplaced(0, InnerMost));

  return NewConstantExpr;
}

// Globals are part of ConstantExpr. This loop iterates over
// all such instances and replaces them with a new ConstantExpr
// consisting of new global vector* variable.
void ESIMDLowerVecArgPass::replaceConstExprWithGlobals(Module &M) {
  for (auto &F : M) {
    for (auto BB = F.begin(), BBEnd = F.end(); BB != BBEnd; ++BB) {
      DenseMap<Instruction *, Instruction *> OldNewInst;
      for (auto OI = BB->begin(), OE = BB->end(); OI != OE; ++OI) {
        SmallVector<Value *, 6> Operands;
        bool HasGlobals = false;
        auto &Inst = (*OI);
        for (unsigned int OP = 0, OPE = Inst.getNumOperands(); OP != OPE;
             ++OP) {
          auto opnd = Inst.getOperand(OP);
          if (!isa<ConstantExpr>(opnd)) {
            Operands.push_back(opnd);
            continue;
          }
          GlobalVariable *OldGlobal = nullptr;
          auto OldGlobalVar = hasGlobalConstExpr(opnd, OldGlobal);
          if (OldGlobalVar && OldNewGlobal.find(OldGlobal) != OldNewGlobal.end()) {
            HasGlobals = true;
            auto NewGlobal = OldNewGlobal[OldGlobal];
            assert(NewGlobal && "Didnt find new global");
            Operands.push_back(
                createNewConstantExpr(NewGlobal, OldGlobal->getType(), opnd));
          } else {
            Operands.push_back(opnd);
          }
        }

        if (HasGlobals) {
          Instruction *NewInst = nullptr;
          if (isa<CallInst>(&Inst)) {
            assert(isa<CallInst>(&Inst) && "Expecting call instruction");
            // pop last parameter which is function declaration
            auto CallI = cast<CallInst>(&Inst);
            Operands.pop_back();
            NewInst =
                CallInst::Create(CallI->getFunctionType(),
                                 CallI->getCalledFunction(), Operands, "");
            cast<CallInst>(NewInst)->setTailCallKind(CallI->getTailCallKind());
          } else if (isa<StoreInst>(&Inst)) {
            auto StoreI = cast<StoreInst>(&Inst);
            NewInst = new StoreInst(Operands[0], Operands[1],
                                    StoreI->isVolatile(), StoreI->getAlign());
          } else if (isa<LoadInst>(&Inst)) {
            auto LoadI = cast<LoadInst>(&Inst);
            NewInst = new LoadInst(Inst.getType(), Operands[0],
                                   LoadI->getName(), LoadI->isVolatile(),
                                   LoadI->getAlign());
          } else {
            assert(false && "Not expecting this instruction with global");
          }
          OldNewInst[&Inst] = NewInst;
          NewInst->copyMetadata(Inst);
        }
      }

      for (auto Replace : OldNewInst) {
        ReplaceInstWithInst(Replace.first, Replace.second);
      }
    }
  }
}

// This function creates new global variables of type vector* type
// when old one is of simd* type.
void ESIMDLowerVecArgPass::fixGlobals(Module &M) {
  for (auto &G : M.getGlobalList()) {
    auto NewTy = argIsSimdPtr(&G);
    if (NewTy && !G.user_empty()) {
      // Peel off ptr type that argIsSimdPtr applies
      NewTy = NewTy->getPointerElementType();
      auto ZeroInit = new APInt(32, 0);
      auto NewGlobalVar =
          new GlobalVariable(NewTy, G.isConstant(), G.getLinkage(),
                             Constant::getIntegerValue(NewTy, *ZeroInit));
      NewGlobalVar->setExternallyInitialized(G.isExternallyInitialized());
      NewGlobalVar->copyAttributesFrom(&G);
      NewGlobalVar->takeName(&G);
      M.getGlobalList().push_back(NewGlobalVar);
      SmallVector<DIGlobalVariableExpression *, 5> GVs;
      G.getDebugInfo(GVs);
      for (auto md : GVs) {
        NewGlobalVar->addDebugInfo(md);
      }
      OldNewGlobal.insert(std::make_pair(&G, NewGlobalVar));
    }
  }

  replaceConstExprWithGlobals(M);

  removeOldGlobals();
}

// Remove old global variables from the program.
void ESIMDLowerVecArgPass::removeOldGlobals() {
  for (auto &G : OldNewGlobal) {
    G.first->removeDeadConstantUsers();
    G.first->eraseFromParent();
  }
}

bool ESIMDLowerVecArgPass::run(Module &M) {
  fixGlobals(M);

  SmallVector<Function *, 10> functions;
  for (auto &F : M) {
    functions.push_back(&F);
  }

  for (auto F : functions) {
    for (unsigned int I = 0; I != F->arg_size(); I++) {
      auto Arg = F->getArg(I);
      if (argIsSimdPtr(Arg)) {
        rewriteFunc(*F);
        break;
      }
    }
  }

  return true;
}
