//===- SPIRVRegularizeLLVM.cpp - Regularize LLVM for SPIR-V ------- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements regularization of LLVM module for SPIR-V.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvregular"

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/ADT/StringExtras.h" // llvm::isDigit
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h" // expandMemSetAsLoop()

#include <set>
#include <vector>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

static bool SPIRVDbgSaveRegularizedModule = false;
static std::string RegularizedModuleTmpFile = "regularized.bc";

class SPIRVRegularizeLLVMBase {
public:
  SPIRVRegularizeLLVMBase() : M(nullptr), Ctx(nullptr) {}

  bool runRegularizeLLVM(Module &M);
  // Lower functions
  bool regularize();

  /// Erase cast inst of function and replace with the function.
  /// Assuming F is a SPIR-V builtin function with op code \param OC.
  void lowerFuncPtr(Function *F, Op OC);
  void lowerFuncPtr(Module *M);

  /// Some LLVM intrinsics that have no SPIR-V counterpart may be wrapped in
  /// @spirv.llvm_intrinsic_* function. During reverse translation from SPIR-V
  /// to LLVM IR we can detect this @spirv.llvm_intrinsic_* function and
  /// replace it with @llvm.intrinsic.* back.
  void lowerIntrinsicToFunction(IntrinsicInst *Intrinsic);

  /// No SPIR-V counterpart for @llvm.fshl.*(@llvm.fshr.*) intrinsic. It will be
  /// lowered to a newly generated @spirv.llvm_fshl_*(@spirv.llvm_fshr_*)
  /// function.
  ///
  /// Conceptually, FSHL (FSHR):
  /// 1. concatenates the ints, the first one being the more significant;
  /// 2. performs a left (right) shift-rotate on the resulting doubled-sized
  /// int;
  /// 3. returns the most (least) significant bits of the shift-rotate result,
  ///    the number of bits being equal to the size of the original integers.
  /// If FSHL (FSHR) operates on a vector type instead, the same operations are
  /// performed for each set of corresponding vector elements.
  ///
  /// The actual implementation algorithm will be slightly different for
  /// simplification purposes.
  void lowerFunnelShift(IntrinsicInst *FSHIntrinsic);

  void lowerUMulWithOverflow(IntrinsicInst *UMulIntrinsic);
  void buildUMulWithOverflowFunc(Function *UMulFunc);

  // For some cases Clang emits VectorExtractDynamic as:
  // void @_Z28__spirv_VectorExtractDynamic(<Ty>* sret(<Ty>), jointMatrix, idx);
  // Instead of:
  // <Ty> @_Z28__spirv_VectorExtractDynamic(JointMatrix, Idx);
  // And VectorInsertDynamic as:
  // @_Z27__spirv_VectorInsertDynamic(jointMatrix, <Ty>* byval(<Ty>), idx);
  // Instead of:
  // @_Z27__spirv_VectorInsertDynamic(jointMatrix, <Ty>, idx)
  // Need to add additional GEP, store and load instructions and mutate called
  // function to avoid translation failures
  void expandSYCLHalfUsing(Module *M);
  void expandVEDWithSYCLHalfSRetArg(Function *F);
  void expandVIDWithSYCLHalfByValComp(Function *F);

  static std::string lowerLLVMIntrinsicName(IntrinsicInst *II);
  void adaptStructTypes(StructType *ST);
  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
};

class SPIRVRegularizeLLVMPass
    : public llvm::PassInfoMixin<SPIRVRegularizeLLVMPass>,
      public SPIRVRegularizeLLVMBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runRegularizeLLVM(M) ? llvm::PreservedAnalyses::none()
                                : llvm::PreservedAnalyses::all();
  }
};

class SPIRVRegularizeLLVMLegacy : public ModulePass,
                                  public SPIRVRegularizeLLVMBase {
public:
  SPIRVRegularizeLLVMLegacy() : ModulePass(ID) {
    initializeSPIRVRegularizeLLVMLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  static char ID;
};

char SPIRVRegularizeLLVMLegacy::ID = 0;

bool SPIRVRegularizeLLVMLegacy::runOnModule(Module &Module) {
  return runRegularizeLLVM(Module);
}

std::string SPIRVRegularizeLLVMBase::lowerLLVMIntrinsicName(IntrinsicInst *II) {
  Function *IntrinsicFunc = II->getCalledFunction();
  assert(IntrinsicFunc && "Missing function");
  std::string FuncName = IntrinsicFunc->getName().str();
  std::replace(FuncName.begin(), FuncName.end(), '.', '_');
  FuncName = "spirv." + FuncName;
  return FuncName;
}

void SPIRVRegularizeLLVMBase::lowerIntrinsicToFunction(
    IntrinsicInst *Intrinsic) {
  // For @llvm.memset.* intrinsic cases with constant value and length arguments
  // are emulated via "storing" a constant array to the destination. For other
  // cases we wrap the intrinsic in @spirv.llvm_memset_* function and expand the
  // intrinsic to a loop via expandMemSetAsLoop() from
  // llvm/Transforms/Utils/LowerMemIntrinsics.h
  if (auto *MSI = dyn_cast<MemSetInst>(Intrinsic))
    if (isa<Constant>(MSI->getValue()) && isa<ConstantInt>(MSI->getLength()))
      return; // To be handled in LLVMToSPIRV::transIntrinsicInst

  std::string FuncName = lowerLLVMIntrinsicName(Intrinsic);
  if (Intrinsic->isVolatile())
    FuncName += ".volatile";
  // Redirect @llvm.intrinsic.* call to @spirv.llvm_intrinsic_*
  Function *F = M->getFunction(FuncName);
  if (F) {
    // This function is already linked in.
    Intrinsic->setCalledFunction(F);
    return;
  }
  // TODO copy arguments attributes: nocapture writeonly.
  FunctionCallee FC =
      M->getOrInsertFunction(FuncName, Intrinsic->getFunctionType());
  auto IntrinsicID = Intrinsic->getIntrinsicID();
  Intrinsic->setCalledFunction(FC);

  F = dyn_cast<Function>(FC.getCallee());
  assert(F && "must be a function!");

  switch (IntrinsicID) {
  case Intrinsic::memset: {
    auto *MSI = static_cast<MemSetInst *>(Intrinsic);
    Argument *Dest = F->getArg(0);
    Argument *Val = F->getArg(1);
    Argument *Len = F->getArg(2);
    Argument *IsVolatile = F->getArg(3);
    Dest->setName("dest");
    Val->setName("val");
    Len->setName("len");
    IsVolatile->setName("isvolatile");
    IsVolatile->addAttr(Attribute::ImmArg);
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *MemSet = IRB.CreateMemSet(Dest, Val, Len, MSI->getDestAlign(),
                                    MSI->isVolatile());
    IRB.CreateRetVoid();
    expandMemSetAsLoop(cast<MemSetInst>(MemSet));
    MemSet->eraseFromParent();
    break;
  }
  case Intrinsic::bswap: {
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *BSwap = IRB.CreateIntrinsic(Intrinsic::bswap, Intrinsic->getType(),
                                      F->getArg(0));
    IRB.CreateRet(BSwap);
    IntrinsicLowering IL(M->getDataLayout());
    IL.LowerIntrinsicCall(BSwap);
    break;
  }
  default:
    break; // do nothing
  }

  return;
}

void SPIRVRegularizeLLVMBase::lowerFunnelShift(IntrinsicInst *FSHIntrinsic) {
  // Get a separate function - otherwise, we'd have to rework the CFG of the
  // current one. Then simply replace the intrinsic uses with a call to the new
  // function.
  // Expected LLVM IR for the function: i* @spirv.llvm_fsh?_i* (i* %a, i* %b, i*
  // %c)
  FunctionType *FSHFuncTy = FSHIntrinsic->getFunctionType();
  Type *FSHRetTy = FSHFuncTy->getReturnType();
  const std::string FuncName = lowerLLVMIntrinsicName(FSHIntrinsic);
  Function *FSHFunc =
      getOrCreateFunction(M, FSHRetTy, FSHFuncTy->params(), FuncName);

  if (!FSHFunc->empty()) {
    FSHIntrinsic->setCalledFunction(FSHFunc);
    return;
  }
  auto *RotateBB = BasicBlock::Create(M->getContext(), "rotate", FSHFunc);
  IRBuilder<> Builder(RotateBB);
  Type *Ty = FSHFunc->getReturnType();
  // Build the actual funnel shift rotate logic.
  // In the comments, "int" is used interchangeably with "vector of int
  // elements".
  FixedVectorType *VectorTy = dyn_cast<FixedVectorType>(Ty);
  Type *IntTy = VectorTy ? VectorTy->getElementType() : Ty;
  unsigned BitWidth = IntTy->getIntegerBitWidth();
  ConstantInt *BitWidthConstant = Builder.getInt({BitWidth, BitWidth});
  Value *BitWidthForInsts =
      VectorTy ? Builder.CreateVectorSplat(VectorTy->getNumElements(),
                                           BitWidthConstant)
               : BitWidthConstant;
  auto *RotateModVal =
      Builder.CreateURem(/*Rotate*/ FSHFunc->getArg(2), BitWidthForInsts);
  Value *FirstShift = nullptr, *SecShift = nullptr;
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr)
    // Shift the less significant number right, the "rotate" number of bits
    // will be 0-filled on the left as a result of this regular shift.
    FirstShift = Builder.CreateLShr(FSHFunc->getArg(1), RotateModVal);
  else
    // Shift the more significant number left, the "rotate" number of bits
    // will be 0-filled on the right as a result of this regular shift.
    FirstShift = Builder.CreateShl(FSHFunc->getArg(0), RotateModVal);

  // We want the "rotate" number of the more significant int's LSBs (MSBs) to
  // occupy the leftmost (rightmost) "0 space" left by the previous operation.
  // Therefore, subtract the "rotate" number from the integer bitsize...
  auto *SubRotateVal = Builder.CreateSub(BitWidthForInsts, RotateModVal);
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr)
    // ...and left-shift the more significant int by this number, zero-filling
    // the LSBs.
    SecShift = Builder.CreateShl(FSHFunc->getArg(0), SubRotateVal);
  else
    // ...and right-shift the less significant int by this number, zero-filling
    // the MSBs.
    SecShift = Builder.CreateLShr(FSHFunc->getArg(1), SubRotateVal);

  // A simple binary addition of the shifted ints yields the final result.
  auto *FunnelShiftRes = Builder.CreateOr(FirstShift, SecShift);
  Builder.CreateRet(FunnelShiftRes);

  FSHIntrinsic->setCalledFunction(FSHFunc);
}

void SPIRVRegularizeLLVMBase::buildUMulWithOverflowFunc(Function *UMulFunc) {
  if (!UMulFunc->empty())
    return;

  BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", UMulFunc);
  IRBuilder<> Builder(EntryBB);
  // Build the actual unsigned multiplication logic with the overflow
  // indication.
  auto *FirstArg = UMulFunc->getArg(0);
  auto *SecondArg = UMulFunc->getArg(1);

  // Do unsigned multiplication Mul = A * B.
  // Then check if unsigned division Div = Mul / A is not equal to B.
  // If so, then overflow has happened.
  auto *Mul = Builder.CreateNUWMul(FirstArg, SecondArg);
  auto *Div = Builder.CreateUDiv(Mul, FirstArg);
  auto *Overflow = Builder.CreateICmpNE(FirstArg, Div);

  // umul.with.overflow intrinsic return a structure, where the first element
  // is the multiplication result, and the second is an overflow bit.
  auto *StructTy = UMulFunc->getReturnType();
  auto *Agg = Builder.CreateInsertValue(UndefValue::get(StructTy), Mul, {0});
  auto *Res = Builder.CreateInsertValue(Agg, Overflow, {1});
  Builder.CreateRet(Res);
}

void SPIRVRegularizeLLVMBase::lowerUMulWithOverflow(
    IntrinsicInst *UMulIntrinsic) {
  // Get a separate function - otherwise, we'd have to rework the CFG of the
  // current one. Then simply replace the intrinsic uses with a call to the new
  // function.
  FunctionType *UMulFuncTy = UMulIntrinsic->getFunctionType();
  Type *FSHLRetTy = UMulFuncTy->getReturnType();
  const std::string FuncName = lowerLLVMIntrinsicName(UMulIntrinsic);
  Function *UMulFunc =
      getOrCreateFunction(M, FSHLRetTy, UMulFuncTy->params(), FuncName);
  buildUMulWithOverflowFunc(UMulFunc);
  UMulIntrinsic->setCalledFunction(UMulFunc);
}

void SPIRVRegularizeLLVMBase::expandVEDWithSYCLHalfSRetArg(Function *F) {
  auto Attrs = F->getAttributes();
  Attrs = Attrs.removeParamAttribute(F->getContext(), 0, Attribute::StructRet);
  std::string Name = F->getName().str();
  CallInst *OldCall = nullptr;
  mutateFunction(
      F,
      [=, &OldCall](CallInst *CI, std::vector<Value *> &Args, Type *&RetTy) {
        Args.erase(Args.begin());
        auto *SRetPtrTy = cast<PointerType>(CI->getOperand(0)->getType());
        auto *ET = SRetPtrTy->getPointerElementType();
        RetTy = cast<StructType>(ET)->getElementType(0);
        OldCall = CI;
        return Name;
      },
      [=, &OldCall](CallInst *NewCI) {
        IRBuilder<> Builder(OldCall);
        auto *SRetPtrTy = cast<PointerType>(OldCall->getOperand(0)->getType());
        auto *ET = SRetPtrTy->getPointerElementType();
        Value *Target = Builder.CreateStructGEP(ET, OldCall->getOperand(0), 0);
        return Builder.CreateStore(NewCI, Target);
      },
      nullptr, &Attrs, true);
}

void SPIRVRegularizeLLVMBase::expandVIDWithSYCLHalfByValComp(Function *F) {
  auto Attrs = F->getAttributes();
  Attrs = Attrs.removeParamAttribute(F->getContext(), 1, Attribute::ByVal);
  std::string Name = F->getName().str();
  mutateFunction(
      F,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        auto *CompPtrTy = cast<PointerType>(CI->getOperand(1)->getType());
        auto *ET = CompPtrTy->getPointerElementType();
        Type *HalfTy = cast<StructType>(ET)->getElementType(0);
        IRBuilder<> Builder(CI);
        auto *Target = Builder.CreateStructGEP(ET, CI->getOperand(1), 0);
        Args[1] = Builder.CreateLoad(HalfTy, Target);
        return Name;
      },
      nullptr, &Attrs, true);
}

void SPIRVRegularizeLLVMBase::expandSYCLHalfUsing(Module *M) {
  std::vector<Function *> ToExpandVEDWithSYCLHalfSRetArg;
  std::vector<Function *> ToExpandVIDWithSYCLHalfByValComp;

  for (auto &F : *M) {
    if (F.getName().startswith("_Z28__spirv_VectorExtractDynamic") &&
        F.hasStructRetAttr()) {
      auto *SRetPtrTy = cast<PointerType>(F.getArg(0)->getType());
      if (isSYCLHalfType(SRetPtrTy->getPointerElementType()))
        ToExpandVEDWithSYCLHalfSRetArg.push_back(&F);
      else
        llvm_unreachable("The return type of the VectorExtractDynamic "
                         "instruction cannot be a structure other than SYCL "
                         "half.");
    }
    if (F.getName().startswith("_Z27__spirv_VectorInsertDynamic") &&
        F.getArg(1)->getType()->isPointerTy()) {
      auto *CompPtrTy = cast<PointerType>(F.getArg(1)->getType());
      auto *ET = CompPtrTy->getPointerElementType();
      if (isSYCLHalfType(ET))
        ToExpandVIDWithSYCLHalfByValComp.push_back(&F);
      else
        llvm_unreachable("The component argument type of an "
                         "VectorInsertDynamic instruction can't be a "
                         "structure other than SYCL half.");
    }
  }

  for (auto *F : ToExpandVEDWithSYCLHalfSRetArg)
    expandVEDWithSYCLHalfSRetArg(F);
  for (auto *F : ToExpandVIDWithSYCLHalfByValComp)
    expandVIDWithSYCLHalfByValComp(F);
}

void SPIRVRegularizeLLVMBase::adaptStructTypes(StructType *ST) {
  if (!ST->hasName())
    return;
  StringRef STName = ST->getName();
  STName.consume_front("struct.");
  STName.consume_front("__spv::");
  StringRef MangledName = STName.substr(0, STName.find('.'));

  // Representation in LLVM IR before the translator is a pointer array wrapped
  // in a structure:
  // %struct.__spirv_JointMatrixINTEL = type { [R x [C x [L x [S x type]]]]* }
  // where R = Rows, C = Columnts, L = Layout + 1, S = Scope + 1
  // this '+1' for the Layout and Scope is required because both of them can
  // be '0', but array size can not be '0'.
  // The result should look like SPIR-V friendly LLVM IR:
  // %spirv.JointMatrixINTEL._char_2_2_0_3
  // Here we check the structure name yet again. Another option would be to
  // check SPIR-V friendly function calls (by their name) and obtain return
  // or their parameter types, assuming, that the appropriate types are Matrix
  // structure type. But in the near future, we will reuse Composite
  // instructions to do, for example, matrix initialization directly on AMX
  // register by OpCompositeConstruct. And we can't claim, that the Result type
  // of OpCompositeConstruct instruction is always the joint matrix type, it's
  // simply not true.
  if (MangledName == "__spirv_JointMatrixINTEL") {
    auto *PtrTy = dyn_cast<PointerType>(ST->getElementType(0));
    assert(PtrTy &&
           "Expected a pointer to an array to represent joint matrix type");
    size_t TypeLayout[4] = {0, 0, 0, 0};
    ArrayType *ArrayTy = dyn_cast<ArrayType>(PtrTy->getPointerElementType());
    assert(ArrayTy && "Expected a pointer element type of an array type to "
                      "represent joint matrix type");
    TypeLayout[0] = ArrayTy->getNumElements();
    for (size_t I = 1; I != 4; ++I) {
      ArrayTy = dyn_cast<ArrayType>(ArrayTy->getElementType());
      assert(ArrayTy &&
             "Expected a element type to represent joint matrix type");
      TypeLayout[I] = ArrayTy->getNumElements();
    }

    auto *ElemTy = ArrayTy->getElementType();
    std::string ElemTyStr;
    if (ElemTy->isIntegerTy()) {
      auto *IntElemTy = cast<IntegerType>(ElemTy);
      switch (IntElemTy->getBitWidth()) {
      case 8:
        ElemTyStr = "char";
        break;
      case 16:
        ElemTyStr = "short";
        break;
      case 32:
        ElemTyStr = "int";
        break;
      case 64:
        ElemTyStr = "long";
        break;
      default:
        ElemTyStr = "i" + std::to_string(IntElemTy->getBitWidth());
      }
    }
    // Check half type like this as well, but in DPC++ it most likelly will
    // be a class
    else if (ElemTy->isHalfTy())
      ElemTyStr = "half";
    else if (ElemTy->isFloatTy())
      ElemTyStr = "float";
    else if (ElemTy->isDoubleTy())
      ElemTyStr = "double";
    else {
      // Half type is special: in DPC++ we use `class half` instead of `half`
      // type natively supported by Clang.
      auto *STElemTy = dyn_cast<StructType>(ElemTy);
      if (!STElemTy && !STElemTy->hasName())
        llvm_unreachable("Unexpected type for matrix!");
      StringRef STElemTyName = STElemTy->getName();
      STElemTyName.consume_front("class.");
      if ((STElemTyName.startswith("cl::sycl::") ||
           STElemTyName.startswith("__sycl_internal::")) &&
          STElemTyName.endswith("::half"))
        ElemTyStr = "half";
      if (ElemTyStr.size() == 0)
        llvm_unreachable("Unexpected type for matrix!");
    }
    std::stringstream SPVName;
    SPVName << kSPIRVTypeName::PrefixAndDelim
            << kSPIRVTypeName::JointMatrixINTEL << kSPIRVTypeName::Delimiter
            << kSPIRVTypeName::PostfixDelim << ElemTyStr
            << kSPIRVTypeName::PostfixDelim << std::to_string(TypeLayout[0])
            << kSPIRVTypeName::PostfixDelim << std::to_string(TypeLayout[1])
            << kSPIRVTypeName::PostfixDelim << std::to_string(TypeLayout[2] - 1)
            << kSPIRVTypeName::PostfixDelim
            << std::to_string(TypeLayout[3] - 1);
    // Note, that this structure is not opaque and there is no way to make it
    // opaque but to recreate it entirely and replace it everywhere. Lets
    // keep the structure as is, dealing with it during SPIR-V generation.
    ST->setName(SPVName.str());
  }
}

bool SPIRVRegularizeLLVMBase::runRegularizeLLVM(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter SPIRVRegularizeLLVM:\n");
  regularize();
  LLVM_DEBUG(dbgs() << "After SPIRVRegularizeLLVM:\n" << *M);

  verifyRegularizationPass(*M, "SPIRVRegularizeLLVM");

  return true;
}

/// Remove entities not representable by SPIR-V
bool SPIRVRegularizeLLVMBase::regularize() {
  eraseUselessFunctions(M);
  lowerFuncPtr(M);
  expandSYCLHalfUsing(M);

  for (auto I = M->begin(), E = M->end(); I != E;) {
    Function *F = &(*I++);
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      continue;
    }

    std::vector<Instruction *> ToErase;
    for (BasicBlock &BB : *F) {
      for (Instruction &II : BB) {
        if (auto Call = dyn_cast<CallInst>(&II)) {
          Call->setTailCall(false);
          Function *CF = Call->getCalledFunction();
          if (CF && CF->isIntrinsic()) {
            removeFnAttr(Call, Attribute::NoUnwind);
            auto *II = cast<IntrinsicInst>(Call);
            if (II->getIntrinsicID() == Intrinsic::memset ||
                II->getIntrinsicID() == Intrinsic::bswap)
              lowerIntrinsicToFunction(II);
            else if (II->getIntrinsicID() == Intrinsic::fshl ||
                     II->getIntrinsicID() == Intrinsic::fshr)
              lowerFunnelShift(II);
            else if (II->getIntrinsicID() == Intrinsic::umul_with_overflow)
              lowerUMulWithOverflow(II);
          }
        }

        // Remove optimization info not supported by SPIRV
        if (auto BO = dyn_cast<BinaryOperator>(&II)) {
          if (isa<PossiblyExactOperator>(BO) && BO->isExact())
            BO->setIsExact(false);
        }

        // FIXME: This is not valid handling for freeze instruction
        if (auto FI = dyn_cast<FreezeInst>(&II)) {
          FI->replaceAllUsesWith(FI->getOperand(0));
          FI->dropAllReferences();
          ToErase.push_back(FI);
        }

        // Remove metadata not supported by SPIRV
        static const char *MDs[] = {
            "fpmath",
            "tbaa",
            "range",
        };
        for (auto &MDName : MDs) {
          if (II.getMetadata(MDName)) {
            II.setMetadata(MDName, nullptr);
          }
        }
        // Add an additional bitcast in case address space cast also changes
        // pointer element type.
        if (auto *ASCast = dyn_cast<AddrSpaceCastInst>(&II)) {
          Type *DestTy = ASCast->getDestTy();
          Type *SrcTy = ASCast->getSrcTy();
          if (DestTy->getPointerElementType() !=
              SrcTy->getPointerElementType()) {
            PointerType *InterTy =
                PointerType::get(DestTy->getPointerElementType(),
                                 SrcTy->getPointerAddressSpace());
            BitCastInst *NewBCast = new BitCastInst(
                ASCast->getPointerOperand(), InterTy, /*NameStr=*/"", ASCast);
            AddrSpaceCastInst *NewASCast =
                new AddrSpaceCastInst(NewBCast, DestTy, /*NameStr=*/"", ASCast);
            ToErase.push_back(ASCast);
            ASCast->dropAllReferences();
            ASCast->replaceAllUsesWith(NewASCast);
          }
        }
        if (auto Cmpxchg = dyn_cast<AtomicCmpXchgInst>(&II)) {
          // Transform:
          // %1 = cmpxchg i32* %ptr, i32 %comparator, i32 %0 seq_cst acquire
          // To:
          // %cmpxchg.res = call spir_func
          //   i32 @_Z29__spirv_AtomicCompareExchangePiiiiii(
          //   i32* %ptr, i32 1, i32 16, i32 2, i32 %0, i32 %comparator)
          // %cmpxchg.success = icmp eq i32 %cmpxchg.res, %comparator
          // %1 = insertvalue { i32, i1 } undef, i32 %cmpxchg.res, 0
          // %2 = insertvalue { i32, i1 } %1, i1 %cmpxchg.success, 1

          // To get memory scope argument we might use Cmpxchg->getSyncScopeID()
          // but LLVM's cmpxchg instruction is not aware of OpenCL(or SPIR-V)
          // memory scope enumeration. And assuming the produced SPIR-V module
          // will be consumed in an OpenCL environment, we can use the same
          // memory scope as OpenCL atomic functions that do not have
          // memory_scope argument, i.e. memory_scope_device. See the OpenCL C
          // specification p6.13.11. Atomic Functions

          // cmpxchg LLVM instruction returns a pair {i32, i1}: the original
          // value and a flag indicating success (true) or failure (false).
          // OpAtomicCompareExchange SPIR-V instruction returns only the
          // original value. To keep the return type({i32, i1}) we construct
          // a composite. The first element of the composite holds result of
          // OpAtomicCompareExchange, i.e. the original value. The second
          // element holds result of comparison of the returned value and the
          // comparator, which matches with semantics of the flag returned by
          // cmpxchg.
          Value *Ptr = Cmpxchg->getPointerOperand();
          Value *MemoryScope = getInt32(M, spv::ScopeDevice);
          auto SuccessOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getSuccessOrdering()));
          auto FailureOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getFailureOrdering()));
          Value *EqualSem = getInt32(M, OCLMemOrderMap::map(SuccessOrder));
          Value *UnequalSem = getInt32(M, OCLMemOrderMap::map(FailureOrder));
          Value *Val = Cmpxchg->getNewValOperand();
          Value *Comparator = Cmpxchg->getCompareOperand();

          llvm::Value *Args[] = {Ptr,        MemoryScope, EqualSem,
                                 UnequalSem, Val,         Comparator};
          auto *Res = addCallInstSPIRV(M, "__spirv_AtomicCompareExchange",
                                       Cmpxchg->getCompareOperand()->getType(),
                                       Args, nullptr, &II, "cmpxchg.res");
          IRBuilder<> Builder(Cmpxchg);
          auto *Cmp = Builder.CreateICmpEQ(Res, Comparator, "cmpxchg.success");
          auto *V1 = Builder.CreateInsertValue(
              UndefValue::get(Cmpxchg->getType()), Res, 0);
          auto *V2 = Builder.CreateInsertValue(V1, Cmp, 1, Cmpxchg->getName());
          Cmpxchg->replaceAllUsesWith(V2);
          ToErase.push_back(Cmpxchg);
        }
      }
    }
    for (Instruction *V : ToErase) {
      assert(V->user_empty() && "User non-empty\n");
      V->eraseFromParent();
    }
  }

  for (StructType *ST : M->getIdentifiedStructTypes())
    adaptStructTypes(ST);

  if (SPIRVDbgSaveRegularizedModule)
    saveLLVMModule(M, RegularizedModuleTmpFile);
  return true;
}

// Assume F is a SPIR-V builtin function with a function pointer argument which
// is a bitcast instruction casting a function to a void(void) function pointer.
void SPIRVRegularizeLLVMBase::lowerFuncPtr(Function *F, Op OC) {
  LLVM_DEBUG(dbgs() << "[lowerFuncPtr] " << *F << '\n');
  auto Name = decorateSPIRVFunction(getName(OC));
  std::set<Value *> InvokeFuncPtrs;
  auto Attrs = F->getAttributes();
  mutateFunction(
      F,
      [=, &InvokeFuncPtrs](CallInst *CI, std::vector<Value *> &Args) {
        for (auto &I : Args) {
          if (isFunctionPointerType(I->getType())) {
            InvokeFuncPtrs.insert(I);
            I = removeCast(I);
          }
        }
        return Name;
      },
      nullptr, &Attrs, false);
  for (auto &I : InvokeFuncPtrs)
    eraseIfNoUse(I);
}

void SPIRVRegularizeLLVMBase::lowerFuncPtr(Module *M) {
  std::vector<std::pair<Function *, Op>> Work;
  for (auto &F : *M) {
    auto AI = F.arg_begin();
    if (hasFunctionPointerArg(&F, AI)) {
      auto OC = getSPIRVFuncOC(F.getName());
      if (OC != OpNop) // builtin with a function pointer argument
        Work.push_back(std::make_pair(&F, OC));
    }
  }
  for (auto &I : Work)
    lowerFuncPtr(I.first, I.second);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVRegularizeLLVMLegacy, "spvregular",
                "Regularize LLVM for SPIR-V", false, false)

ModulePass *llvm::createSPIRVRegularizeLLVMLegacy() {
  return new SPIRVRegularizeLLVMLegacy();
}
