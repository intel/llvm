//===----- SpecConstants.cpp - SYCL Specialization Constants Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "SpecConstants.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

// __sycl* intrinsic names are Itanium ABI-mangled; this is common prefix for
// all mangled names of __sycl_getSpecConstantValue intrinsics, which differ by
// the template type parameter and the specialization constant value type.
constexpr char SYCL_GET_SPEC_CONST_VAL[] = "_Z27__sycl_getSpecConstantValue";
// Unmangled base name of all __spirv_SpecConstant intrinsics which differ by
// the value type.
constexpr char SPIRV_GET_SPEC_CONST_VAL[] = "__spirv_SpecConstant";
// Metadata ID string added to calls to __spirv_SpecConstant to record the
// original symbolic spec constant ID.
constexpr char SPEC_CONST_SYM_ID_MD_STRING[] = "SYCL_SPEC_CONST_SYM_ID";

static void AssertRelease(bool Cond, const char *Msg) {
  if (!Cond)
    report_fatal_error((Twine("SpecConstants.cpp: ") + Msg).str().c_str());
}

StringRef getStringLiteralArg(const CallInst *CI, unsigned ArgNo,
                              SmallVectorImpl<Instruction *> &DelInsts,
                              GlobalVariable *&SymGlob) {
  Value *V = CI->getArgOperand(ArgNo)->stripPointerCasts();

  if (auto *L = dyn_cast<LoadInst>(V)) {
    // Must be a
    // vvvvvvvvvvvvvvvvvvvv
    // @.str = private unnamed_addr constant[10 x i8] c"SpecConst\00", align 1
    // ...
    // %TName = alloca i8 addrspace(4)*, align 8
    // ...
    // store i8 addrspace(4)* addrspacecast(
    //    i8* getelementptr inbounds([10 x i8], [10 x i8] * @.str, i32 0, i32 0)
    //    to i8 addrspace(4)*), i8 addrspace(4)** %TName, align 8, !tbaa !10
    // %1 = load i8 addrspace(4)*, i8 addrspace(4)** %TName, align 8, !tbaa !10
    // %call = call spir_func zeroext
    //   i1 @_Z27__sycl_getSpecConstantValueIbET_PKc(i8 addrspace(4)* %1)
    // ^^^^^^^^^^^^^^^^^^^^
    // sequence, w/o any intervening stores and calls between the store and load
    // so that %1 is trivially known to be the address of the @.str literal.

    AllocaInst *TmpPtr =
        cast<AllocaInst>(L->getPointerOperand()->stripPointerCasts());
    // find the store of the literal address into TmpPtr
    StoreInst *Store = nullptr;

    for (User *U : TmpPtr->users()) {
      if (StoreInst *St = dyn_cast<StoreInst>(U)) {
        AssertRelease(!Store, "single store expected");
        Store = St;
#ifndef NDEBUG
        break;
#endif // NDEBUG
      }
    }
    AssertRelease(Store, "unexpected spec const IR pattern 0");
    DelInsts.push_back(Store);
#ifndef NDEBUG
    // verify there are no intervening stores/calls
    AssertRelease(L->getParent() == Store->getParent(), "same BB expected");

    for (const Instruction *I = Store->getNextNode(); I; I = I->getNextNode()) {
      if (I == L) {
        DelInsts.push_back(L);
        L = nullptr; // mark as met
        break;
      }
      AssertRelease(!I->mayHaveSideEffects(),
                    "unexpected spec const IR pattern 1");
    }
    AssertRelease(!L, "load not met after the store");
#endif // NDEBUG
    AssertRelease(Store, "store not met");
    V = Store->getValueOperand()->stripPointerCasts();
  }
  const Constant *Init = cast<GlobalVariable>(V)->getInitializer();
  SymGlob = cast<GlobalVariable>(V);
  StringRef Res = cast<ConstantDataArray>(Init)->getAsString();
  if (Res.size() > 0 && Res[Res.size() - 1] == '\0')
    Res = Res.substr(0, Res.size() - 1);
  return Res;
}

// TODO support spec constant types other than integer or
// floating-point.
Value *genDefaultValue(Type *T, Instruction *At) {
  if (T->isIntegerTy())
    return ConstantInt::get(T, 0);
  if (T->isFloatingPointTy())
    return ConstantFP::get(T, 0.0);
  llvm_unreachable("non-numeric specialization constants are NYI");
  return nullptr;
}

std::string manglePrimitiveType(Type *T) {
  if (T->isFloatTy())
    return "f";
  if (T->isDoubleTy())
    return "d";
  assert(T->isIntegerTy() &&
         "unsupported spec const type, must've been guarded in headers");
  switch (T->getIntegerBitWidth()) {
  case 1:
    return "b";
  case 8:
    return "a";
  case 16:
    return "s";
  case 32:
    return "i";
  case 64:
    return "x";
  default:
    llvm_unreachable("unsupported spec const integer type");
  }
  return "";
}

// This is a very basic mangler which can mangle non-templated and non-member
// functions with primitive types in the signature.
std::string mangleFuncItanium(StringRef BaseName, FunctionType *FT) {
  std::string Res =
      (Twine("_Z") + Twine(BaseName.size()) + Twine(BaseName)).str();
  for (unsigned I = 0; I < FT->getNumParams(); ++I)
    Res += manglePrimitiveType(FT->getParamType(I));
  return Res;
}

void setSpecConstMetadata(Instruction *I, StringRef SymID, int IntID) {
  LLVMContext &Ctx = I->getContext();
  MDString *SymV = MDString::get(Ctx, SymID);
  ConstantAsMetadata *IntV =
      ConstantAsMetadata::get(ConstantInt::get(Ctx, APInt(32, IntID)));
  MDNode *Entry = MDNode::get(Ctx, {SymV, IntV});
  I->setMetadata(SPEC_CONST_SYM_ID_MD_STRING, Entry);
}

std::pair<StringRef, unsigned> getSpecConstMetadata(Instruction *I) {
  const MDNode *N = I->getMetadata(SPEC_CONST_SYM_ID_MD_STRING);
  if (!N)
    return std::make_pair("", 0);
  const auto *MDSym = cast<MDString>(N->getOperand(0));
  const auto *MDInt = cast<ConstantAsMetadata>(N->getOperand(1));
  unsigned ID = static_cast<unsigned>(
      cast<ConstantInt>(MDInt->getValue())->getValue().getZExtValue());
  return std::make_pair(MDSym->getString(), ID);
}

static Value *getDefaultCPPValue(Type *T) {
  if (T->isIntegerTy())
    return Constant::getIntegerValue(T, APInt(T->getScalarSizeInBits(), 0));
  if (T->isFloatingPointTy())
    return ConstantFP::get(T, 0);
  llvm_unreachable("unsupported spec const type");
  return nullptr;
}

} // namespace

PreservedAnalyses SpecConstantsPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  int NextID = 0;
  StringMap<unsigned> IDMap;

  // Iterate through all calls to
  // template <typename T> T __sycl_getSpecConstantValue(const char *ID)
  // intrinsic and lower them depending on the SetValAtRT setting (see below).
  bool IRModified = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 32> SCIntrCalls;

    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      Function *Callee = nullptr;
      if (!CI || CI->isIndirectCall() || !(Callee = CI->getCalledFunction()))
        continue;
      StringRef Name = Callee->getName();

      if (!Name.startswith(SYCL_GET_SPEC_CONST_VAL))
        continue;
      SCIntrCalls.push_back(CI);
    }
    IRModified = IRModified || (SCIntrCalls.size() > 0);

    for (auto *CI : SCIntrCalls) {
      // 1. Find the symbolic ID (string literal) passed as the actual argument
      // to the intrinsic - this should always be possible, as only string
      // literals are passed to it in the SYCL RT source code, and application
      // code can't use this intrinsic directly.
      SmallVector<Instruction *, 3> DelInsts;
      DelInsts.push_back(CI);
      GlobalVariable *SymGlob = nullptr;
      StringRef SymID = getStringLiteralArg(CI, 0, DelInsts, SymGlob);
      Type *SCTy = CI->getType();

      if (SetValAtRT) {
        // 2. Spec constant value will be set at run time - then add the literal
        // to a "spec const string literal ID" -> "integer ID" map, uniquing
        // the integer ID if this is new literal
        auto Ins = IDMap.insert(std::make_pair(SymID, 0));
        if (Ins.second)
          Ins.first->second = NextID++;
        //  3. Transform to spirv intrinsic _Z*__spirv_SpecConstant*.
        LLVMContext &Ctx = F.getContext();
        // Generate arguments needed by the SPIRV version of the intrinsic
        // - integer constant ID:
        Value *ID = ConstantInt::get(Type::getInt32Ty(Ctx), NextID - 1);
        // - default value:
        Value *Def = genDefaultValue(SCTy, CI);
        // ... Now replace the call with SPIRV intrinsic version.
        Value *Args[] = {ID, Def};
        constexpr size_t NArgs = sizeof(Args) / sizeof(Args[0]);
        Type *ArgTys[NArgs] = {nullptr};
        for (unsigned int I = 0; I < NArgs; ++I)
          ArgTys[I] = Args[I]->getType();
        FunctionType *FT = FunctionType::get(SCTy, ArgTys, false /*isVarArg*/);
        Module &M = *F.getParent();
        std::string SPIRVName = mangleFuncItanium(SPIRV_GET_SPEC_CONST_VAL, FT);
        FunctionCallee FC = M.getOrInsertFunction(SPIRVName, FT);
        assert(FC.getCallee() && "SPIRV intrinsic creation failed");
        CallInst *SPIRVCall =
            CallInst::Create(FT, FC.getCallee(), Args, "", CI);
        CI->replaceAllUsesWith(SPIRVCall);
        // Mark the instruction with <symbolic_id, int_id> pair for later
        // recollection by collectSpecConstantMetadata method.
        setSpecConstMetadata(SPIRVCall, SymID, NextID - 1);
        // Example of the emitted call when spec constant is integer:
        // %6 = call i32 @_Z20__spirv_SpecConstantii(i32 0, i32 0), \
        //                                          !SYCL_SPEC_CONST_SYM_ID !22
      } else {
        // 2a. Spec constant must be resolved at compile time - just replace
        // the intrinsic with default C++ value for the spec constant type.
        CI->replaceAllUsesWith(getDefaultCPPValue(SCTy));
      }
      for (auto *I : DelInsts) {
        assert(I->getNumUses() == 0 && "removing live instruction");
        I->removeFromParent();
        I->deleteValue();
      }
      // Don't delete SymGlob here, as it may be referenced from multiple
      // functions if __sycl_getSpecConstantValue is inlined.
    }
  }
  return IRModified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool SpecConstantsPass::collectSpecConstantMetadata(
    Module &M, std::map<StringRef, unsigned> &IDMap) {
  bool Met = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 32> SCIntrCalls;

    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      Function *Callee = nullptr;
      if (!CI || CI->isIndirectCall() || !(Callee = CI->getCalledFunction()))
        continue;
      std::pair<StringRef, unsigned> Res = getSpecConstMetadata(CI);

      if (!Res.first.empty()) {
        IDMap[Res.first] = Res.second;
        Met = true;
      }
    }
  }
  return Met;
}
