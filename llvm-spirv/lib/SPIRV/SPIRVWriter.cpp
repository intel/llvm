//===- SPIRVWriter.cpp - Converts LLVM to SPIR-V ----------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
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
/// \file
///
/// This file implements conversion of LLVM intermediate language to SPIR-V
/// binary.
///
//===----------------------------------------------------------------------===//

#include "SPIRVWriter.h"
#include "LLVMToSPIRVDbgTran.h"
#include "OCLToSPIRV.h"
#include "PreprocessMetadata.h"
#include "SPIRVAsm.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVLLVMUtil.h"
#include "SPIRVLowerBitCastToNonStandardType.h"
#include "SPIRVLowerBool.h"
#include "SPIRVLowerConstExpr.h"
#include "SPIRVLowerLLVMIntrinsic.h"
#include "SPIRVLowerMemmove.h"
#include "SPIRVLowerOCLBlocks.h"
#include "SPIRVMDWalker.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVModule.h"
#include "SPIRVRegularizeLLVM.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "VectorComputeUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <regex>
#include <set>
#include <vector>

#define DEBUG_TYPE "spirv"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace {

static SPIRVWord convertFloatToSPIRVWord(float F) {
  union {
    float F;
    SPIRVWord Spir;
  } FPMaxError;
  FPMaxError.F = F;
  return FPMaxError.Spir;
}

/// Return one of the SPIR-V 1.4 SignExtend or ZeroExtend image operands
/// for a function name, or 0 if the function does not return or
/// write an integer type.
int getImageSignZeroExt(Function *F) {
  bool IsSigned = false;
  bool IsUnsigned = false;

  ParamSignedness RetSignedness;
  SmallVector<ParamSignedness, 4> ArgSignedness;
  if (!getRetParamSignedness(F, RetSignedness, ArgSignedness))
    return 0;

  StringRef Name = F->getName();
  Name = Name.substr(Name.find(kSPIRVName::Prefix));
  Name.consume_front(kSPIRVName::Prefix);
  if (Name.consume_front("ImageRead") ||
      Name.consume_front("ImageSampleExplicitLod")) {
    if (RetSignedness == ParamSignedness::Signed)
      IsSigned = true;
    else if (RetSignedness == ParamSignedness::Unsigned)
      IsUnsigned = true;
    else if (F->getReturnType()->isIntOrIntVectorTy() &&
             Name.consume_front("_R")) {
      // Return type is mangled after _R, e.g. _Z23__spirv_ImageRead_Rint2li
      IsSigned = isMangledTypeSigned(Name[0]);
      IsUnsigned = Name.starts_with("u");
    }
  } else if (Name.starts_with("ImageWrite")) {
    IsSigned = (ArgSignedness[2] == ParamSignedness::Signed);
    IsUnsigned = (ArgSignedness[2] == ParamSignedness::Unsigned);
  }

  if (IsSigned)
    return ImageOperandsMask::ImageOperandsSignExtendMask;
  if (IsUnsigned)
    return ImageOperandsMask::ImageOperandsZeroExtendMask;
  return 0;
}

} // namespace

namespace SPIRV {

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(const std::string &Str, SPIRVFunctionParameter *BA)>
        Func) {
  assert(BF->getNumArguments() == MD->getNumOperands() &&
         "Invalid kernel metadata: Number of metadata operands and kernel "
         "arguments do not match");
  for (unsigned I = 0, E = MD->getNumOperands(); I != E; ++I) {
    SPIRVFunctionParameter *BA = BF->getArgument(I);
    Func(getMDOperandAsString(MD, I).str(), BA);
  }
}

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(Metadata *MDOp, SPIRVFunctionParameter *BA)> Func) {
  assert(BF->getNumArguments() == MD->getNumOperands() &&
         "Invalid kernel metadata: Number of metadata operands and kernel "
         "arguments do not match");
  for (unsigned I = 0, E = MD->getNumOperands(); I != E; ++I) {
    SPIRVFunctionParameter *BA = BF->getArgument(I);
    Func(MD->getOperand(I), BA);
  }
}

static SPIRVMemoryModelKind getMemoryModel(Module &M) {
  auto *MemoryModelMD = M.getNamedMetadata(kSPIRVMD::MemoryModel);
  if (MemoryModelMD && (MemoryModelMD->getNumOperands() > 0)) {
    auto *Ref0 = MemoryModelMD->getOperand(0);
    if (Ref0 && Ref0->getNumOperands() > 1) {
      auto &&ModelOp = Ref0->getOperand(1);
      auto *ModelCI = mdconst::dyn_extract<ConstantInt>(ModelOp);
      if (ModelCI && (ModelCI->getValue().getActiveBits() <= 64)) {
        auto Model = static_cast<SPIRVMemoryModelKind>(ModelCI->getZExtValue());
        return Model;
      }
    }
  }
  return SPIRVMemoryModelKind::MemoryModelMax;
}

static void translateSEVDecoration(Attribute Sev, SPIRVValue *Val) {
  assert(Sev.isStringAttribute() &&
         Sev.getKindAsString() == kVCMetadata::VCSingleElementVector);

  auto *Ty = Val->getType();
  assert((Ty->isTypeBool() || Ty->isTypeFloat() || Ty->isTypeInt() ||
          Ty->isTypePointer()) &&
         "This decoration is valid only for Scalar or Pointer types");

  if (Ty->isTypePointer()) {
    SPIRVWord IndirectLevelsOnElement = 0;
    Sev.getValueAsString().getAsInteger(0, IndirectLevelsOnElement);
    Val->addDecorate(DecorationSingleElementVectorINTEL,
                     IndirectLevelsOnElement);
  } else
    Val->addDecorate(DecorationSingleElementVectorINTEL);
}

LLVMToSPIRVBase::LLVMToSPIRVBase(SPIRVModule *SMod)
    : BuiltinCallHelper(ManglingRules::None), M(nullptr), Ctx(nullptr),
      BM(SMod), SrcLang(0), SrcLangVer(0) {
  DbgTran = std::make_unique<LLVMToSPIRVDbgTran>(nullptr, SMod, this);
}

LLVMToSPIRVBase::~LLVMToSPIRVBase() {
  for (auto *I : UnboundInst)
    I->deleteValue();
}

bool LLVMToSPIRVBase::runLLVMToSPIRV(Module &Mod) {
  M = &Mod;
  initialize(Mod);
  CG = std::make_unique<CallGraph>(Mod);
  Ctx = &M->getContext();
  DbgTran->setModule(M);
  assert(BM && "SPIR-V module not initialized");
  translate();
  return true;
}

SPIRVValue *LLVMToSPIRVBase::getTranslatedValue(const Value *V) const {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
}

bool LLVMToSPIRVBase::isKernel(Function *F) {
  if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
    return true;
  return false;
}

bool LLVMToSPIRVBase::isBuiltinTransToInst(Function *F) {
  StringRef DemangledName;
  if (!oclIsBuiltin(F->getName(), DemangledName) &&
      !isDecoratedSPIRVFunc(F, DemangledName))
    return false;
  SPIRVDBG(spvdbgs() << "CallInst: demangled name: " << DemangledName.str()
                     << '\n');
  return getSPIRVFuncOC(DemangledName) != OpNop;
}

bool LLVMToSPIRVBase::isBuiltinTransToExtInst(
    Function *F, SPIRVExtInstSetKind *ExtSet, SPIRVWord *ExtOp,
    SmallVectorImpl<std::string> *Dec) {
  StringRef DemangledName;
  if (!oclIsBuiltin(F->getName(), DemangledName))
    return false;
  LLVM_DEBUG(dbgs() << "[oclIsBuiltinTransToExtInst] CallInst: demangled name: "
                    << DemangledName << '\n');
  StringRef S = DemangledName;
  if (!S.starts_with(kSPIRVName::Prefix))
    return false;
  S = S.drop_front(strlen(kSPIRVName::Prefix));
  auto Loc = S.find(kSPIRVPostfix::Divider);
  auto ExtSetName = S.substr(0, Loc);
  SPIRVExtInstSetKind Set = SPIRVEIS_Count;
  if (!SPIRVExtSetShortNameMap::rfind(ExtSetName.str(), &Set))
    return false;
  assert((Set == SPIRVEIS_OpenCL || Set == BM->getDebugInfoEIS()) &&
         "Unsupported extended instruction set");

  auto ExtOpName = S.substr(Loc + 1);
  auto Splited = ExtOpName.split(kSPIRVPostfix::ExtDivider);
  OCLExtOpKind EOC;
  if (!OCLExtOpMap::rfind(Splited.first.str(), &EOC))
    return false;

  if (ExtSet)
    *ExtSet = Set;
  if (ExtOp)
    *ExtOp = EOC;
  if (Dec) {
    SmallVector<StringRef, 2> P;
    Splited.second.split(P, kSPIRVPostfix::Divider);
    for (auto &I : P)
      Dec->push_back(I.str());
  }
  return true;
}

bool isUniformGroupOperation(Function *F) {
  auto Name = F->getName();
  if (Name.contains("GroupIMulKHR") || Name.contains("GroupFMulKHR") ||
      Name.contains("GroupBitwiseAndKHR") ||
      Name.contains("GroupBitwiseOrKHR") ||
      Name.contains("GroupBitwiseXorKHR") ||
      Name.contains("GroupLogicalAndKHR") ||
      Name.contains("GroupLogicalOrKHR") || Name.contains("GroupLogicalXorKHR"))
    return true;
  return false;
}

static bool recursiveType(const StructType *ST, const Type *Ty) {
  SmallPtrSet<const StructType *, 4> Seen;

  std::function<bool(const Type *Ty)> Run = [&](const Type *Ty) {
    if (!(isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) &&
        !Ty->isPointerTy())
      return false;

    if (auto *StructTy = dyn_cast<StructType>(Ty)) {
      if (StructTy == ST)
        return true;

      if (Seen.count(StructTy))
        return false;

      Seen.insert(StructTy);

      return find_if(StructTy->element_begin(), StructTy->element_end(), Run) !=
             StructTy->element_end();
    }

    if (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
      return Run(ArrayTy->getArrayElementType());

    return false;
  };

  return Run(Ty);
}

// Add decoration if needed
void addFPBuiltinDecoration(SPIRVModule *BM, Instruction *Inst,
                            SPIRVInstruction *I) {
  bool AllowFPMaxError =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fp_max_error);

  auto *II = dyn_cast_or_null<IntrinsicInst>(Inst);
  if (II && II->getCalledFunction()->getName().starts_with("llvm.fpbuiltin")) {
    // Add a new decoration for llvm.builtin intrinsics, if needed
    if (II->getAttributes().hasFnAttr("fpbuiltin-max-error")) {
      BM->getErrorLog().checkError(AllowFPMaxError, SPIRVEC_RequiresExtension,
                                   "SPV_INTEL_fp_max_error\n");
      double F = 0.0;
      II->getAttributes()
          .getFnAttr("fpbuiltin-max-error")
          .getValueAsString()
          .getAsDouble(F);
      I->addDecorate(DecorationFPMaxErrorDecorationINTEL,
                     convertFloatToSPIRVWord(F));
    }
  } else if (auto *MD = Inst->getMetadata("fpmath")) {
    if (!AllowFPMaxError)
      return;
    auto *MDVal = mdconst::dyn_extract<ConstantFP>(MD->getOperand(0));
    double ValAsDouble = MDVal->getValue().convertToFloat();
    I->addDecorate(DecorationFPMaxErrorDecorationINTEL,
                   convertFloatToSPIRVWord(ValAsDouble));
  }
}

SPIRVType *LLVMToSPIRVBase::transType(Type *T) {
  LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(dbgs() << "[transType] " << *T << '\n');
  if (T->isVoidTy())
    return mapType(T, BM->addVoidType());

  if (T->isIntegerTy(1))
    return mapType(T, BM->addBoolType());

  if (T->isIntegerTy()) {
    unsigned BitWidth = T->getIntegerBitWidth();
    // SPIR-V 2.16.1. Universal Validation Rules: Scalar integer types can be
    // parameterized only as 32 bit, plus any additional sizes enabled by
    // capabilities.
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_arbitrary_precision_integers) ||
        BM->getErrorLog().checkError(
            BitWidth == 8 || BitWidth == 16 || BitWidth == 32 || BitWidth == 64,
            SPIRVEC_InvalidBitWidth, std::to_string(BitWidth))) {
      return mapType(T, BM->addIntegerType(T->getIntegerBitWidth()));
    }
  }

  if (T->isFloatingPointTy())
    return mapType(T, BM->addFloatType(T->getPrimitiveSizeInBits()));

  if (T->isTokenTy()) {
    BM->getErrorLog().checkError(
        BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_token_type),
        SPIRVEC_RequiresExtension,
        "SPV_INTEL_token_type\n"
        "NOTE: LLVM module contains token type, which doesn't have analogs in "
        "SPIR-V without extensions");
    return mapType(T, BM->addTokenTypeINTEL());
  }

  // A pointer to image or pipe type in LLVM is translated to a SPIRV
  // (non-pointer) image or pipe type.
  if (T->isPointerTy()) {
    auto *ET = Type::getInt8Ty(T->getContext());
    auto AddrSpc = T->getPointerAddressSpace();
    return transPointerType(ET, AddrSpc);
  }

  if (auto *TPT = dyn_cast<TypedPointerType>(T)) {
    return transPointerType(TPT->getElementType(), TPT->getAddressSpace());
  }

  if (auto *VecTy = dyn_cast<FixedVectorType>(T)) {
    if (VecTy->getElementType()->isPointerTy() ||
        isa<TypedPointerType>(VecTy->getElementType())) {
      // SPV_INTEL_masked_gather_scatter extension changes 2.16.1. Universal
      // Validation Rules:
      // Vector types must be parameterized only with numerical types,
      // [Physical Pointer Type] types or the [OpTypeBool] type.
      // Without it vector of pointers is not allowed in SPIR-V.
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_masked_gather_scatter)) {
        BM->getErrorLog().checkError(
            false, SPIRVEC_RequiresExtension,
            "SPV_INTEL_masked_gather_scatter\n"
            "NOTE: LLVM module contains vector of pointers, translation "
            "of which requires this extension");
        return nullptr;
      }
      BM->addExtension(ExtensionID::SPV_INTEL_masked_gather_scatter);
      BM->addCapability(internal::CapabilityMaskedGatherScatterINTEL);
    }
    return mapType(T, BM->addVectorType(transType(VecTy->getElementType()),
                                        VecTy->getNumElements()));
  }

  if (T->isArrayTy()) {
    // SPIR-V 1.3 s3.32.6: Length is the number of elements in the array.
    //                     It must be at least 1.
    if (T->getArrayNumElements() < 1) {
      std::string Str;
      llvm::raw_string_ostream OS(Str);
      OS << *T;
      SPIRVCK(T->getArrayNumElements() >= 1, InvalidArraySize, OS.str());
    }
    Type *ElTy = T->getArrayElementType();
    SPIRVType *TransType = BM->addArrayType(
        transType(ElTy),
        static_cast<SPIRVConstant *>(transValue(
            ConstantInt::get(getSizetType(), T->getArrayNumElements(), false),
            nullptr)));
    mapType(T, TransType);
    if (ElTy->isPointerTy()) {
      mapType(
          ArrayType::get(TypedPointerType::get(Type::getInt8Ty(*Ctx),
                                               ElTy->getPointerAddressSpace()),
                         T->getArrayNumElements()),
          TransType);
    }
    return TransType;
  }

  if (T->isStructTy() && !T->isSized()) {
    auto *ST = dyn_cast<StructType>(T);
    (void)ST; // Silence warning
    assert(!ST->getName().starts_with(kSPR2TypeName::PipeRO));
    assert(!ST->getName().starts_with(kSPR2TypeName::PipeWO));
    assert(!ST->getName().starts_with(kSPR2TypeName::ImagePrefix));
    return mapType(T, BM->addOpaqueType(T->getStructName().str()));
  }

  if (auto *ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    StringRef Name;
    if (ST->hasName())
      Name = ST->getName();

    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
      return transSPIRVOpaqueType("spirv.Sampler", SPIRAS_Constant);
    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
      return transSPIRVOpaqueType("spirv.PipeStorage", SPIRAS_Constant);

    constexpr size_t MaxNumElements = MaxWordCount - SPIRVTypeStruct::FixedWC;
    const size_t NumElements = ST->getNumElements();
    size_t SPIRVStructNumElements = NumElements;
    // In case number of elements is greater than maximum WordCount and
    // SPV_INTEL_long_constant_composite is not enabled, the error will be
    // emitted by validate functionality of SPIRVTypeStruct class.
    if (NumElements > MaxNumElements &&
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_long_constant_composite)) {
      SPIRVStructNumElements = MaxNumElements;
    }

    auto *Struct = BM->openStructType(SPIRVStructNumElements, Name.str());
    mapType(T, Struct);

    if (NumElements > MaxNumElements &&
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_long_constant_composite)) {
      uint64_t NumOfContinuedInstructions = NumElements / MaxNumElements - 1;
      for (uint64_t J = 0; J < NumOfContinuedInstructions; J++) {
        auto *Continued = BM->addTypeStructContinuedINTEL(MaxNumElements);
        Struct->addContinuedInstruction(
            static_cast<SPIRVTypeStruct::ContinuedInstType>(Continued));
      }
      uint64_t Remains = NumElements % MaxNumElements;
      if (Remains) {
        auto *Continued = BM->addTypeStructContinuedINTEL(Remains);
        Struct->addContinuedInstruction(
            static_cast<SPIRVTypeStruct::ContinuedInstType>(Continued));
      }
    }

    SmallVector<unsigned, 4> ForwardRefs;

    for (unsigned I = 0, E = T->getStructNumElements(); I != E; ++I) {
      auto *ElemTy = ST->getElementType(I);
      if ((isa<StructType>(ElemTy) || isa<ArrayType>(ElemTy) ||
           isa<VectorType>(ElemTy) || isa<PointerType>(ElemTy)) &&
          recursiveType(ST, ElemTy))
        ForwardRefs.push_back(I);
      else
        Struct->setMemberType(I, transType(ST->getElementType(I)));
    }

    BM->closeStructType(Struct, ST->isPacked());

    for (auto I : ForwardRefs)
      Struct->setMemberType(I, transType(ST->getElementType(I)));

    return Struct;
  }

  if (FunctionType *FT = dyn_cast<FunctionType>(T)) {
    SPIRVType *RT = transType(FT->getReturnType());
    std::vector<SPIRVType *> PT;
    for (FunctionType::param_iterator I = FT->param_begin(),
                                      E = FT->param_end();
         I != E; ++I)
      PT.push_back(transType(*I));
    return mapType(T, getSPIRVFunctionType(RT, PT));
  }

  if (auto *TargetTy = dyn_cast<TargetExtType>(T)) {
    StringRef Name = TargetTy->getName();
    if (Name.consume_front(kSPIRVTypeName::PrefixAndDelim)) {
      auto Opcode = SPIRVOpaqueTypeOpCodeMap::map(Name.str());
      auto CastAccess = [](unsigned Val) {
        return static_cast<SPIRVAccessQualifierKind>(Val);
      };
      switch (static_cast<size_t>(Opcode)) {
      case OpTypePipe: {
        auto *PipeT = BM->addPipeType();
        PipeT->setPipeAcessQualifier(CastAccess(TargetTy->getIntParameter(0)));
        return mapType(T, PipeT);
      }
      case OpTypeImage: {
        auto *SampledTy = transType(TargetTy->getTypeParameter(0));
        ArrayRef<unsigned> Ops = TargetTy->int_params();
        SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]),
                                      Ops[1], Ops[2], Ops[3], Ops[4], Ops[5]);
        return mapType(T,
                       BM->addImageType(SampledTy, Desc, CastAccess(Ops[6])));
      }
      case OpTypeSampledImage: {
        auto *ImageTy = static_cast<SPIRVTypeImage *>(transType(adjustImageType(
            T, kSPIRVTypeName::SampledImg, kSPIRVTypeName::Image)));
        return mapType(T, BM->addSampledImageType(ImageTy));
      }
      case OpTypeVmeImageINTEL: {
        auto *ImageTy = static_cast<SPIRVTypeImage *>(transType(adjustImageType(
            T, kSPIRVTypeName::VmeImageINTEL, kSPIRVTypeName::Image)));
        return mapType(T, BM->addVmeImageINTELType(ImageTy));
      }
      case OpTypeQueue:
        return mapType(T, BM->addQueueType());
      case OpTypeDeviceEvent:
        return mapType(T, BM->addDeviceEventType());
      case OpTypeBufferSurfaceINTEL: {
        ArrayRef<unsigned> Ops = TargetTy->int_params();
        return mapType(T, BM->addBufferSurfaceINTELType(CastAccess(Ops[0])));
      }
      case internal::OpTypeJointMatrixINTEL: {
        // The expected representation is:
        // target("spirv.JointMatrixINTEL", %element_type, %rows%, %cols%,
        //        %layout%, %scope%, %use%,
        //        (optional) %element_type_interpretation%)
        auto *ElemTy = transType(TargetTy->getTypeParameter(0));
        ArrayRef<unsigned> Ops = TargetTy->int_params();
        std::vector<SPIRVValue *> Args;
        for (const auto &Op : Ops)
          Args.emplace_back(transConstant(getUInt32(M, Op)));
        return mapType(T, BM->addJointMatrixINTELType(ElemTy, Args));
      }
      case OpTypeCooperativeMatrixKHR: {
        // The expected representation is:
        // target("spirv.CooperativeMatrixKHR", %element_type, %scope%, %rows%,
        // %cols%, %use%)
        auto *ElemTy = transType(TargetTy->getTypeParameter(0));
        ArrayRef<unsigned> Ops = TargetTy->int_params();
        std::vector<SPIRVValue *> Args;
        for (const auto &Op : Ops)
          Args.emplace_back(transConstant(getUInt32(M, Op)));
        return mapType(T, BM->addCooperativeMatrixKHRType(ElemTy, Args));
      }
      case internal::OpTypeTaskSequenceINTEL:
        return mapType(T, BM->addTaskSequenceINTELType());
      default:
        if (isSubgroupAvcINTELTypeOpCode(Opcode))
          return mapType(T, BM->addSubgroupAvcINTELType(Opcode));
        return mapType(T, BM->addOpaqueGenericType(Opcode));
      }
    }
  }

  llvm_unreachable("Not implemented!");
  return 0;
}

SPIRVType *LLVMToSPIRVBase::transPointerType(Type *ET, unsigned AddrSpc) {
  Type *T = PointerType::get(ET, AddrSpc);
  if (ET->isFunctionTy() &&
      !BM->checkExtension(ExtensionID::SPV_INTEL_function_pointers,
                          SPIRVEC_FunctionPointers, toString(T)))
    return nullptr;

  std::string TypeKey = (Twine((uintptr_t)ET) + Twine(AddrSpc)).str();
  auto Loc = PointeeTypeMap.find(TypeKey);
  if (Loc != PointeeTypeMap.end())
    return Loc->second;

  // A pointer to image or pipe type in LLVM is translated to a SPIRV
  // (non-pointer) image or pipe type.
  auto *ST = dyn_cast<StructType>(ET);
  // Lower global_device and global_host address spaces that were added in
  // SYCL as part of SYCL_INTEL_usm_address_spaces extension to just global
  // address space if device doesn't support SPV_INTEL_usm_storage_classes
  // extension
  if (!BM->isAllowedToUseExtension(
          ExtensionID::SPV_INTEL_usm_storage_classes) &&
      ((AddrSpc == SPIRAS_GlobalDevice) || (AddrSpc == SPIRAS_GlobalHost))) {
    return transPointerType(ET, SPIRAS_Global);
  }
  if (ST && !ST->isSized()) {
    Op OpCode;
    StringRef STName = ST->getName();
    // Workaround for non-conformant SPIR binary
    if (STName == "struct._event_t") {
      STName = kSPR2TypeName::Event;
      ST->setName(STName);
    }

    std::pair<StringRef, unsigned> Key = {STName, AddrSpc};
    if (auto *MappedTy = OpaqueStructMap.lookup(Key))
      return MappedTy;

    auto SaveType = [&](SPIRVType *MappedTy) {
      OpaqueStructMap[Key] = MappedTy;
      PointeeTypeMap[TypeKey] = MappedTy;
      return MappedTy;
    };

    if (STName.starts_with(kSPR2TypeName::PipeRO) ||
        STName.starts_with(kSPR2TypeName::PipeWO)) {
      auto *PipeT = BM->addPipeType();
      PipeT->setPipeAcessQualifier(STName.starts_with(kSPR2TypeName::PipeRO)
                                       ? AccessQualifierReadOnly
                                       : AccessQualifierWriteOnly);
      return SaveType(PipeT);
    }
    if (STName.starts_with(kSPR2TypeName::ImagePrefix)) {
      assert(AddrSpc == SPIRAS_Global);
      Type *ImageTy =
          adjustImageType(TypedPointerType::get(ST, AddrSpc),
                          kSPIRVTypeName::Image, kSPIRVTypeName::Image);
      return SaveType(transType(ImageTy));
    }
    if (STName == kSPR2TypeName::Sampler)
      return SaveType(transType(getSPIRVType(OpTypeSampler)));
    if (STName.starts_with(kSPIRVTypeName::PrefixAndDelim))
      return transSPIRVOpaqueType(STName, AddrSpc);

    if (STName.starts_with(kOCLSubgroupsAVCIntel::TypePrefix))
      return SaveType(BM->addSubgroupAvcINTELType(
          OCLSubgroupINTELTypeOpCodeMap::map(ST->getName().str())));

    if (OCLOpaqueTypeOpCodeMap::find(STName.str(), &OpCode)) {
      Type *RealType = getSPIRVType(OpCode);
      return SaveType(transType(RealType));
    }
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute)) {
      if (STName.starts_with(kVCType::VCBufferSurface)) {
        // VCBufferSurface always have Access Qualifier
        auto Access = getAccessQualifier(STName);
        return SaveType(BM->addBufferSurfaceINTELType(Access));
      }
    }

    if (ST->isOpaque()) {
      return SaveType(BM->addPointerType(
          SPIRSPIRVAddrSpaceMap::map(static_cast<SPIRAddressSpace>(AddrSpc)),
          transType(ET)));
    }
  } else {
    SPIRVType *ElementType = transType(ET);
    // ET, as a recursive type, may contain exactly the same pointer T, so it
    // may happen that after translation of ET we already have translated T,
    // added the translated pointer to the SPIR-V module and mapped T to this
    // pointer. Now we have to check PointeeTypeMap again.
    auto Loc = PointeeTypeMap.find(TypeKey);
    if (Loc != PointeeTypeMap.end()) {
      return Loc->second;
    }
    SPIRVType *TranslatedTy = transPointerType(ElementType, AddrSpc);
    PointeeTypeMap[TypeKey] = TranslatedTy;
    return TranslatedTy;
  }

  llvm_unreachable("Not implemented!");
  return nullptr;
}

SPIRVType *LLVMToSPIRVBase::transPointerType(SPIRVType *ET, unsigned AddrSpc) {
  std::string TypeKey = (Twine((uintptr_t)ET) + Twine(AddrSpc)).str();
  auto Loc = PointeeTypeMap.find(TypeKey);
  if (Loc != PointeeTypeMap.end())
    return Loc->second;

  SPIRVType *TranslatedTy = BM->addPointerType(
      SPIRSPIRVAddrSpaceMap::map(static_cast<SPIRAddressSpace>(AddrSpc)), ET);
  PointeeTypeMap[TypeKey] = TranslatedTy;
  return TranslatedTy;
}

SPIRVType *LLVMToSPIRVBase::transSPIRVOpaqueType(StringRef STName,
                                                 unsigned AddrSpace) {
  std::pair<StringRef, unsigned> Key = {STName, AddrSpace};
  if (auto *MappedTy = OpaqueStructMap.lookup(Key))
    return MappedTy;

  auto SaveType = [&](SPIRVType *MappedTy) {
    OpaqueStructMap[Key] = MappedTy;
    return MappedTy;
  };
  StructType *ST = StructType::getTypeByName(M->getContext(), STName);

  assert(STName.starts_with(kSPIRVTypeName::PrefixAndDelim) &&
         "Invalid SPIR-V opaque type name");
  SmallVector<std::string, 8> Postfixes;
  auto TN = decodeSPIRVTypeName(STName, Postfixes);
  if (TN == kSPIRVTypeName::Pipe) {
    assert(AddrSpace == SPIRAS_Global);
    assert(Postfixes.size() == 1 && "Invalid pipe type ops");
    auto *PipeT = BM->addPipeType();
    PipeT->setPipeAcessQualifier(
        static_cast<spv::AccessQualifier>(atoi(Postfixes[0].c_str())));
    return SaveType(PipeT);
  } else if (TN == kSPIRVTypeName::Image) {
    assert(AddrSpace == SPIRAS_Global);
    // The sampled type needs to be translated through LLVM type to guarantee
    // uniqueness.
    auto *SampledT = transType(
        getLLVMTypeForSPIRVImageSampledTypePostfix(Postfixes[0], *Ctx));
    SmallVector<int, 7> Ops;
    for (unsigned I = 1; I < 8; ++I)
      Ops.push_back(atoi(Postfixes[I].c_str()));
    SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]),
                                  Ops[1], Ops[2], Ops[3], Ops[4], Ops[5]);
    return SaveType(BM->addImageType(
        SampledT, Desc, static_cast<spv::AccessQualifier>(Ops[6])));
  } else if (TN == kSPIRVTypeName::SampledImg) {
    return SaveType(BM->addSampledImageType(static_cast<SPIRVTypeImage *>(
        transType(adjustImageType(TypedPointerType::get(ST, SPIRAS_Global),
                                  kSPIRVTypeName::SampledImg,
                                  kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::VmeImageINTEL) {
    // This type is the same as SampledImageType, but consumed by Subgroup AVC
    // Intel extension instructions.
    return SaveType(BM->addVmeImageINTELType(static_cast<SPIRVTypeImage *>(
        transType(adjustImageType(TypedPointerType::get(ST, SPIRAS_Global),
                                  kSPIRVTypeName::VmeImageINTEL,
                                  kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::Sampler)
    return SaveType(BM->addSamplerType());
  else if (TN == kSPIRVTypeName::DeviceEvent)
    return SaveType(BM->addDeviceEventType());
  else if (TN == kSPIRVTypeName::Queue)
    return SaveType(BM->addQueueType());
  else if (TN == kSPIRVTypeName::PipeStorage)
    return SaveType(BM->addPipeStorageType());
  else if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute) &&
           TN == kSPIRVTypeName::BufferSurfaceINTEL) {
    auto Access = getAccessQualifier(STName);
    return SaveType(BM->addBufferSurfaceINTELType(Access));
  } else
    return SaveType(
        BM->addOpaqueGenericType(SPIRVOpaqueTypeOpCodeMap::map(TN)));
}

SPIRVType *LLVMToSPIRVBase::transScavengedType(Value *V) {
  if (auto *F = dyn_cast<Function>(V)) {
    FunctionType *FnTy = Scavenger->getFunctionType(F);
    SPIRVType *RT = transType(FnTy->getReturnType());
    std::vector<SPIRVType *> PT;
    for (Argument &Arg : F->args()) {
      assert(OCLTypeToSPIRVPtr);
      Type *Ty = OCLTypeToSPIRVPtr->getAdaptedArgumentType(F, Arg.getArgNo());
      if (!Ty) {
        Ty = FnTy->getParamType(Arg.getArgNo());
      }
      PT.push_back(transType(Ty));
    }

    return getSPIRVFunctionType(RT, PT);
  }

  return transType(Scavenger->getScavengedType(V));
}

SPIRVType *
LLVMToSPIRVBase::getSPIRVFunctionType(SPIRVType *RT,
                                      const std::vector<SPIRVType *> &Args) {
  // Come up with a unique string identifier for the arguments. This is a hacky
  // way of doing so, but it works.
  std::string TypeKey;
  llvm::raw_string_ostream TKS(TypeKey);
  TKS << (uintptr_t)RT << ",";
  for (SPIRVType *ArgTy : Args) {
    TKS << (uintptr_t)ArgTy << ",";
  }

  // Create a SPIRVType for the function type. Since SPIRVModule doesn't do
  // any type uniquing for SPIRVType, we have to do it ourself.
  TKS.flush();
  auto It = PointeeTypeMap.find(TypeKey);
  if (It == PointeeTypeMap.end())
    It = PointeeTypeMap.insert({TypeKey, BM->addFunctionType(RT, Args)}).first;
  return It->second;
}

SPIRVFunction *LLVMToSPIRVBase::transFunctionDecl(Function *F) {
  if (auto *BF = getTranslatedValue(F))
    return static_cast<SPIRVFunction *>(BF);

  if (F->isIntrinsic() && (!BM->isSPIRVAllowUnknownIntrinsicsEnabled() ||
                           isKnownIntrinsic(F->getIntrinsicID()))) {
    // We should not translate LLVM intrinsics as a function
    assert(std::none_of(F->user_begin(), F->user_end(),
                        [this](User *U) { return getTranslatedValue(U); }) &&
           "LLVM intrinsics shouldn't be called in SPIRV");
    return nullptr;
  }

  SPIRVTypeFunction *BFT =
      static_cast<SPIRVTypeFunction *>(transScavengedType(F));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName()) {
    if (isUniformGroupOperation(F))
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_KHR_uniform_group_instructions),
          SPIRVEC_RequiresExtension, "SPV_KHR_uniform_group_instructions\n");

    BM->setName(BF, F->getName().str());
  }
  if (!isKernel(F) && F->getLinkage() != GlobalValue::InternalLinkage)
    BF->setLinkageType(transLinkageType(F));

  // Translate OpenCL/SYCL buffer_location metadata if it's attached to the
  // translated function declaration
  MDNode *BufferLocation = nullptr;
  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_buffer_location))
    BufferLocation = F->getMetadata("kernel_arg_buffer_location");

  // Translate runtime_aligned metadata if it's attached to the translated
  // function declaration
  MDNode *RuntimeAligned = nullptr;
  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_runtime_aligned))
    RuntimeAligned = F->getMetadata("kernel_arg_runtime_aligned");

  auto Attrs = F->getAttributes();

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    if (I->hasName())
      BM->setName(BA, I->getName().str());
    if (I->hasByValAttr())
      BA->addAttr(FunctionParameterAttributeByVal);
    if (I->hasNoAliasAttr())
      BA->addAttr(FunctionParameterAttributeNoAlias);
    if (I->hasNoCaptureAttr())
      BA->addAttr(FunctionParameterAttributeNoCapture);
    if (I->hasStructRetAttr())
      BA->addAttr(FunctionParameterAttributeSret);
    if (Attrs.hasParamAttr(ArgNo, Attribute::ReadOnly))
      BA->addAttr(FunctionParameterAttributeNoWrite);
    if (Attrs.hasParamAttr(ArgNo, Attribute::ReadNone))
      // TODO: intel/llvm customization
      // see https://github.com/intel/llvm/issues/7592
      // Need to return FunctionParameterAttributeNoReadWrite
      BA->addAttr(FunctionParameterAttributeNoWrite);
    if (Attrs.hasParamAttr(ArgNo, Attribute::ZExt))
      BA->addAttr(FunctionParameterAttributeZext);
    if (Attrs.hasParamAttr(ArgNo, Attribute::SExt))
      BA->addAttr(FunctionParameterAttributeSext);
    if (Attrs.hasParamAttr(ArgNo, Attribute::Alignment)) {
      SPIRVWord AlignmentBytes =
          Attrs.getParamAttr(ArgNo, Attribute::Alignment)
              .getAlignment()
              .valueOrOne()
              .value();
      BA->setAlignment(AlignmentBytes);
    }
    if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1) &&
        Attrs.hasParamAttr(ArgNo, Attribute::Dereferenceable))
      BA->addDecorate(DecorationMaxByteOffset,
                      Attrs.getParamAttr(ArgNo, Attribute::Dereferenceable)
                          .getDereferenceableBytes());
    if (BufferLocation && I->getType()->isPointerTy()) {
      // Order of integer numbers in MD node follows the order of function
      // parameters on which we shall attach the appropriate decoration. Add
      // decoration only if MD value is not negative.
      int LocID = -1;
      if (!isa<MDString>(BufferLocation->getOperand(ArgNo)) &&
          !isa<MDNode>(BufferLocation->getOperand(ArgNo)))
        LocID = getMDOperandAsInt(BufferLocation, ArgNo);
      if (LocID >= 0)
        BA->addDecorate(DecorationBufferLocationINTEL, LocID);
    }
    if (RuntimeAligned && I->getType()->isPointerTy()) {
      // Order of integer numbers in MD node follows the order of function
      // parameters on which we shall attach the appropriate decoration. Add
      // decoration only if MD value is 1.
      int LocID = 0;
      if (!isa<MDString>(RuntimeAligned->getOperand(ArgNo)) &&
          !isa<MDNode>(RuntimeAligned->getOperand(ArgNo)))
        LocID = getMDOperandAsInt(RuntimeAligned, ArgNo);
      if (LocID == 1)
        BA->addDecorate(internal::DecorationRuntimeAlignedINTEL, LocID);
    }
  }
  if (Attrs.hasRetAttr(Attribute::ZExt))
    BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
  if (Attrs.hasRetAttr(Attribute::SExt))
    BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeSext);
  if (Attrs.hasFnAttr("referenced-indirectly")) {
    assert(!isKernel(F) &&
           "kernel function was marked as referenced-indirectly");
    BF->addDecorate(DecorationReferencedIndirectlyINTEL);
  }

  if (Attrs.hasFnAttr(kVCMetadata::VCCallable) &&
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fast_composite)) {
    BF->addDecorate(internal::DecorationCallableFunctionINTEL);
  }

  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    transVectorComputeMetadata(F);

  transFPGAFunctionMetadata(BF, F);

  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_maximum_registers))
    transFunctionMetadataAsExecutionMode(BF, F);
  else
    transFunctionMetadataAsUserSemanticDecoration(BF, F);

  transAuxDataInst(BF, F);

  SPIRVDBG(dbgs() << "[transFunction] " << *F << " => ";
           spvdbgs() << *BF << '\n';)
  return BF;
}

void LLVMToSPIRVBase::transVectorComputeMetadata(Function *F) {
  using namespace VectorComputeUtil;
  if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    return;
  auto *BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
  assert(BF && "The SPIRVFunction pointer shouldn't be nullptr");
  auto Attrs = F->getAttributes();

  if (Attrs.hasFnAttr(kVCMetadata::VCStackCall))
    BF->addDecorate(DecorationStackCallINTEL);
  if (Attrs.hasFnAttr(kVCMetadata::VCFunction))
    BF->addDecorate(DecorationVectorComputeFunctionINTEL);

  if (Attrs.hasFnAttr(kVCMetadata::VCSIMTCall)) {
    SPIRVWord SIMTMode = 0;
    Attrs.getFnAttr(kVCMetadata::VCSIMTCall)
        .getValueAsString()
        .getAsInteger(0, SIMTMode);
    BF->addDecorate(DecorationSIMTCallINTEL, SIMTMode);
  }

  if (Attrs.hasRetAttr(kVCMetadata::VCSingleElementVector))
    translateSEVDecoration(
        Attrs.getAttributeAtIndex(AttributeList::ReturnIndex,
                                  kVCMetadata::VCSingleElementVector),
        BF);

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCArgumentIOKind)) {
      SPIRVWord Kind = {};
      Attrs.getParamAttr(ArgNo, kVCMetadata::VCArgumentIOKind)
          .getValueAsString()
          .getAsInteger(0, Kind);
      BA->addDecorate(DecorationFuncParamIOKindINTEL, Kind);
    }
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCSingleElementVector))
      translateSEVDecoration(
          Attrs.getParamAttr(ArgNo, kVCMetadata::VCSingleElementVector), BA);
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCArgumentKind)) {
      SPIRVWord Kind;
      Attrs.getParamAttr(ArgNo, kVCMetadata::VCArgumentKind)
          .getValueAsString()
          .getAsInteger(0, Kind);
      BA->addDecorate(internal::DecorationFuncParamKindINTEL, Kind);
    }
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCArgumentDesc)) {
      StringRef Desc =
          Attrs.getParamAttr(ArgNo, kVCMetadata::VCArgumentDesc)
              .getValueAsString();
      BA->addDecorate(new SPIRVDecorateFuncParamDescAttr(BA, Desc.str()));
    }
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCMediaBlockIO)) {
      assert(BA->getType()->isTypeImage() &&
             "VCMediaBlockIO attribute valid only on image parameters");
      BA->addDecorate(DecorationMediaBlockIOINTEL);
    }
  }
  if (!isKernel(F) &&
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_float_controls2) &&
      Attrs.hasFnAttr(kVCMetadata::VCFloatControl)) {

    SPIRVWord Mode = 0;
    Attrs
        .getFnAttr(kVCMetadata::VCFloatControl)
        .getValueAsString()
        .getAsInteger(0, Mode);
    VCFloatTypeSizeMap::foreach (
        [&](VCFloatType FloatType, unsigned TargetWidth) {
          BF->addDecorate(new SPIRVDecorateFunctionDenormModeINTEL(
              BF, TargetWidth, getFPDenormMode(Mode, FloatType)));

          BF->addDecorate(new SPIRVDecorateFunctionRoundingModeINTEL(
              BF, TargetWidth, getFPRoundingMode(Mode)));

          BF->addDecorate(new SPIRVDecorateFunctionFloatingPointModeINTEL(
              BF, TargetWidth, getFPOperationMode(Mode)));
        });
  }
}

static void transMetadataDecorations(Metadata *MD, SPIRVValue *Target);

void LLVMToSPIRVBase::transFPGAFunctionMetadata(SPIRVFunction *BF,
                                                Function *F) {
  if (MDNode *StallEnable = F->getMetadata(kSPIR2MD::StallEnable)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_cluster_attributes)) {
      if (getMDOperandAsInt(StallEnable, 0))
        BF->addDecorate(new SPIRVDecorateStallEnableINTEL(BF));
    }
  }
  if (MDNode *StallFree = F->getMetadata(kSPIR2MD::StallFree)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_cluster_attributes)) {
      if (getMDOperandAsInt(StallFree, 0)) {
        BF->addDecorate(new SPIRVDecorateStallFreeINTEL(BF));
      }
    }
  }
  if (MDNode *LoopFuse = F->getMetadata(kSPIR2MD::LoopFuse)) {
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_loop_fuse)) {
      size_t Depth = getMDOperandAsInt(LoopFuse, 0);
      size_t Independent = getMDOperandAsInt(LoopFuse, 1);
      BF->addDecorate(
          new SPIRVDecorateFuseLoopsInFunctionINTEL(BF, Depth, Independent));
    }
  }
  if (MDNode *PreferDSP = F->getMetadata(kSPIR2MD::PreferDSP)) {
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_dsp_control)) {
      size_t Mode = getMDOperandAsInt(PreferDSP, 0);
      MDNode *PropDSPPref = F->getMetadata(kSPIR2MD::PropDSPPref);
      size_t Propagate = PropDSPPref ? getMDOperandAsInt(PropDSPPref, 0) : 0;
      BF->addDecorate(new SPIRVDecorateMathOpDSPModeINTEL(BF, Mode, Propagate));
    }
  }
  if (MDNode *InitiationInterval =
          F->getMetadata(kSPIR2MD::InitiationInterval)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      if (size_t Cycles = getMDOperandAsInt(InitiationInterval, 0))
        BF->addDecorate(new SPIRVDecorateInitiationIntervalINTEL(BF, Cycles));
    }
  }
  if (MDNode *MaxConcurrency = F->getMetadata(kSPIR2MD::MaxConcurrency)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      size_t Invocations = getMDOperandAsInt(MaxConcurrency, 0);
      BF->addDecorate(new SPIRVDecorateMaxConcurrencyINTEL(BF, Invocations));
    }
  }
  if (MDNode *PipelineKernel = F->getMetadata(kSPIR2MD::PipelineKernel)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      size_t Pipeline = getMDOperandAsInt(PipelineKernel, 0);
      BF->addDecorate(new SPIRVDecoratePipelineEnableINTEL(BF, Pipeline));
    }
  }

  // In addition, process the decorations on the function
  if (auto *FDecoMD = F->getMetadata(SPIRV_MD_DECORATIONS))
    transMetadataDecorations(FDecoMD, BF);
}

void LLVMToSPIRVBase::transFunctionMetadataAsExecutionMode(SPIRVFunction *BF,
                                                           Function *F) {
  SmallVector<MDNode *, 1> RegisterAllocModeMDs;
  F->getMetadata("RegisterAllocMode", RegisterAllocModeMDs);

  for (unsigned I = 0; I < RegisterAllocModeMDs.size(); I++) {
    auto *RegisterAllocMode = RegisterAllocModeMDs[I]->getOperand(0).get();
    if (isa<MDString>(RegisterAllocMode)) {
      StringRef Str = getMDOperandAsString(RegisterAllocModeMDs[I], 0);
      NamedMaximumNumberOfRegisters NamedValue =
          SPIRVNamedMaximumNumberOfRegistersNameMap::rmap(Str.str());
      BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
          OpExecutionMode, BF, ExecutionModeNamedMaximumRegistersINTEL,
          NamedValue)));
    } else if (isa<MDNode>(RegisterAllocMode)) {
      auto *RegisterAllocNodeMDOp =
          getMDOperandAsMDNode(RegisterAllocModeMDs[I], 0);
      int Num = getMDOperandAsInt(RegisterAllocNodeMDOp, 0);
      auto *Const =
          BM->addConstant(transType(Type::getInt32Ty(F->getContext())), Num);
      BF->addExecutionMode(BM->add(new SPIRVExecutionModeId(
          BF, ExecutionModeMaximumRegistersIdINTEL, Const->getId())));
    } else {
      int64_t RegisterAllocVal =
          mdconst::dyn_extract<ConstantInt>(RegisterAllocMode)->getZExtValue();
      BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
          OpExecutionMode, BF, ExecutionModeMaximumRegistersINTEL,
          RegisterAllocVal)));
    }
  }
}

void LLVMToSPIRVBase::transFunctionMetadataAsUserSemanticDecoration(
    SPIRVFunction *BF, Function *F) {
  if (auto *RegisterAllocModeMD = F->getMetadata("RegisterAllocMode")) {
    // TODO: Once the design for per-kernel register size allocation is
    // finalized, we will need to move away from UserSemantic and introduce an
    // extension
    int RegisterAllocNodeMDOp = getMDOperandAsInt(RegisterAllocModeMD, 0);
    // The current RegisterAllocMode metadata format is as follows
    // AUTO - 0
    // SMALL - 1
    // LARGE - 2
    // DEFAULT - 3
    // Currently we only support AUTO, SMALL and LARGE
    if (RegisterAllocNodeMDOp == 0 || RegisterAllocNodeMDOp == 1 ||
        RegisterAllocNodeMDOp == 2) {
      // 4 threads per eu means large grf mode, and 8 threads per eu
      // means small grf mode, 0 means use internal heuristics to choose
      std::string NumThreads;
      switch (RegisterAllocNodeMDOp) {
      case 0:
        NumThreads = "0";
        break;
      case 1:
        NumThreads = "8";
        break;
      case 2:
        NumThreads = "4";
        break;
      default:
        llvm_unreachable("Not implemented");
      }
      BF->addDecorate(new SPIRVDecorateUserSemanticAttr(
          BF, "num-thread-per-eu " + NumThreads));
    }
  }
}

void LLVMToSPIRVBase::transAuxDataInst(SPIRVFunction *BF, Function *F) {
  auto *BM = BF->getModule();
  if (!BM->preserveAuxData())
    return;
  if (!BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_6))
    BM->addExtension(SPIRV::ExtensionID::SPV_KHR_non_semantic_info);
  else
    BM->setMinSPIRVVersion(VersionNumber::SPIRV_1_6);
  const auto &FnAttrs = F->getAttributes().getFnAttrs();
  for (const auto &Attr : FnAttrs) {
    std::vector<SPIRVWord> Ops;
    Ops.push_back(BF->getId());
    if (Attr.isStringAttribute()) {
      // Format for String attributes is:
      // NonSemanticAuxDataFunctionAttribute Fcn AttrName AttrValue
      // or, if no value:
      // NonSemanticAuxDataFunctionAttribute Fcn AttrName
      //
      // AttrName and AttrValue are always Strings
      StringRef AttrKind = Attr.getKindAsString();
      StringRef AttrValue = Attr.getValueAsString();
      auto *KindSpvString = BM->getString(AttrKind.str());
      Ops.push_back(KindSpvString->getId());
      if (!AttrValue.empty()) {
        auto *ValueSpvString = BM->getString(AttrValue.str());
        Ops.push_back(ValueSpvString->getId());
      }
    } else {
      // Format for other types is:
      // NonSemanticAuxDataFunctionAttribute Fcn AttrStr
      // AttrStr is always a String.
      std::string AttrStr = Attr.getAsString();
      auto *AttrSpvString = BM->getString(AttrStr);
      Ops.push_back(AttrSpvString->getId());
    }
    BM->addAuxData(NonSemanticAuxData::FunctionAttribute,
                   transType(Type::getVoidTy(F->getContext())), Ops);
  }
  SmallVector<std::pair<unsigned, MDNode *>> AllMD;
  SmallVector<StringRef> MDNames;
  F->getContext().getMDKindNames(MDNames);
  F->getAllMetadata(AllMD);
  for (const auto &MD : AllMD) {
    std::string MDName = MDNames[MD.first].str();

    // spirv.Decorations, spirv.ParameterDecorations and debug information are
    // handled elsewhere for both forward and reverse translation and are
    // complicated to support here, so just skip them.
    if (MDName == SPIRV_MD_DECORATIONS ||
        MDName == SPIRV_MD_PARAMETER_DECORATIONS ||
        MD.first == LLVMContext::MD_dbg)
      continue;

    // Format for metadata is:
    // NonSemanticAuxDataFunctionMetadata Fcn MDName MDVals...
    // MDName is always a String, MDVals have different types as explained
    // below. Also note this instruction has a variable number of operands
    std::vector<SPIRVWord> Ops;
    Ops.push_back(BF->getId());
    Ops.push_back(BM->getString(MDName)->getId());
    for (unsigned int OpIdx = 0; OpIdx < MD.second->getNumOperands(); OpIdx++) {
      const auto &CurOp = MD.second->getOperand(OpIdx);
      if (auto *MDStr = dyn_cast<MDString>(CurOp)) {
        // For MDString, MDVal is String
        auto *SPIRVStr = BM->getString(MDStr->getString().str());
        Ops.push_back(SPIRVStr->getId());
      } else if (auto *ValueAsMeta = dyn_cast<ValueAsMetadata>(CurOp)) {
        // For Value metadata, MDVal is a SPIRVValue
        auto *SPIRVVal = transValue(ValueAsMeta->getValue(), nullptr);
        Ops.push_back(SPIRVVal->getId());
      } else {
        assert(false && "Unsupported metadata type");
      }
    }
    BM->addAuxData(NonSemanticAuxData::FunctionMetadata,
                   transType(Type::getVoidTy(F->getContext())), Ops);
  }
}

SPIRVValue *LLVMToSPIRVBase::transConstantUse(Constant *C,
                                              SPIRVType *ExpectedType) {
  // Constant expressions expect their pointer types to be i8* in opaque pointer
  // mode, but the value may have a different "natural" type. If that is the
  // case, we need to adjust the type of the constant.
  SPIRVValue *Trans = transValue(C, nullptr, true, FuncTransMode::Pointer);
  if (Trans->getType() == ExpectedType || Trans->getType()->isTypePipeStorage())
    return Trans;

  assert(C->getType()->isPointerTy() &&
         "Only pointer type mismatches should be possible");
  // In the common case of strings ([N x i8] GVs), see if we can emit a GEP
  // instruction.
  if (auto *GV = dyn_cast<GlobalVariable>(C)) {
    if (GV->getValueType()->isArrayTy() &&
        GV->getValueType()->getArrayElementType()->isIntegerTy(8)) {
      SPIRVValue *Offset = transValue(getUInt32(M, 0), nullptr);
      return BM->addPtrAccessChainInst(ExpectedType, Trans, {Offset, Offset},
                                       nullptr, true);
    }
  }

  // Otherwise, just use a bitcast.
  return BM->addUnaryInst(OpBitcast, ExpectedType, Trans, nullptr);
}

SPIRVValue *LLVMToSPIRVBase::transConstant(Value *V) {
  SPIRVType *ExpectedType = transScavengedType(V);
  if (isa<ConstantPointerNull>(V))
    return BM->addNullConstant(bcast<SPIRVTypePointer>(ExpectedType));

  if (isa<ConstantTargetNone>(V))
    return BM->addNullConstant(ExpectedType);

  if (auto *CAZero = dyn_cast<ConstantAggregateZero>(V)) {
    Type *AggType = CAZero->getType();
    if (const StructType *ST = dyn_cast<StructType>(AggType))
      if (ST->hasName() &&
          ST->getName() == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
        return BM->addSamplerConstant(transType(AggType), 0, 0, 0);

    return BM->addNullConstant(transType(AggType));
  }

  if (auto *ConstI = dyn_cast<ConstantInt>(V)) {
    unsigned BitWidth = ConstI->getIntegerType()->getBitWidth();
    if (BitWidth > 64) {
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_integers),
          SPIRVEC_InvalidBitWidth, std::to_string(BitWidth));
      return BM->addConstant(ExpectedType, ConstI->getValue());
    }
    return BM->addConstant(ExpectedType, ConstI->getZExtValue());
  }

  if (auto *ConstFP = dyn_cast<ConstantFP>(V)) {
    auto *BT = static_cast<SPIRVType *>(ExpectedType);
    return BM->addConstant(
        BT, ConstFP->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  if (auto *ConstDA = dyn_cast<ConstantDataArray>(V)) {
    SPIRVType *InnerTy = ExpectedType->getArrayElementType();
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDA->getNumElements(); I != E; ++I)
      BV.push_back(transConstantUse(ConstDA->getElementAsConstant(I), InnerTy));
    return BM->addCompositeConstant(ExpectedType, BV);
  }

  if (auto *ConstA = dyn_cast<ConstantArray>(V)) {
    SPIRVType *InnerTy = ExpectedType->getArrayElementType();
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstA->op_begin(), E = ConstA->op_end(); I != E; ++I)
      BV.push_back(transConstantUse(cast<Constant>(*I), InnerTy));
    return BM->addCompositeConstant(ExpectedType, BV);
  }

  if (auto *ConstDV = dyn_cast<ConstantDataVector>(V)) {
    SPIRVType *InnerTy = ExpectedType->getScalarType();
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDV->getNumElements(); I != E; ++I)
      BV.push_back(transConstantUse(ConstDV->getElementAsConstant(I), InnerTy));
    return BM->addCompositeConstant(ExpectedType, BV);
  }

  if (auto *ConstV = dyn_cast<ConstantVector>(V)) {
    SPIRVType *InnerTy = ExpectedType->getScalarType();
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transConstantUse(cast<Constant>(*I), InnerTy));
    return BM->addCompositeConstant(ExpectedType, BV);
  }

  if (const auto *ConstV = dyn_cast<ConstantStruct>(V)) {
    StringRef StructName;
    if (ConstV->getType()->hasName())
      StructName = ConstV->getType()->getName();
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord AddrMode =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                Normalized =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                FilterMode =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(AddrMode < 5 && "Invalid addressing mode");
      assert(Normalized < 2 && "Invalid value of normalized coords");
      assert(FilterMode < 2 && "Invalid filter mode");
      SPIRVType *SamplerTy = transType(ConstV->getType());
      return BM->addSamplerConstant(SamplerTy, AddrMode, Normalized,
                                    FilterMode);
    }
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord PacketSize =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                PacketAlign =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                Capacity =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(PacketAlign >= 1 && "Invalid packet alignment");
      assert(PacketSize >= PacketAlign && PacketSize % PacketAlign == 0 &&
             "Invalid packet size and/or alignment.");
      SPIRVType *PipeStorageTy = transType(ConstV->getType());
      return BM->addPipeStorageConstant(PipeStorageTy, PacketSize, PacketAlign,
                                        Capacity);
    }
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I) {
      SPIRVType *InnerTy = ExpectedType->getStructMemberType(BV.size());
      BV.push_back(transConstantUse(cast<Constant>(*I), InnerTy));
    }
    return BM->addCompositeConstant(ExpectedType, BV);
  }

  if (auto *ConstUE = dyn_cast<ConstantExpr>(V)) {
    if (auto *GEP = dyn_cast<GEPOperator>(ConstUE)) {
      std::vector<SPIRVValue *> Indices;
      for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
        Indices.push_back(transValue(GEP->getOperand(I + 1), nullptr));
      auto *TransPointerOperand = transValue(GEP->getPointerOperand(), nullptr);
      SPIRVType *TranslatedTy = transScavengedType(GEP);
      return BM->addPtrAccessChainInst(TranslatedTy, TransPointerOperand,
                                       Indices, nullptr, GEP->isInBounds());
    }
    auto *Inst = ConstUE->getAsInstruction();
    SPIRVDBG(dbgs() << "ConstantExpr: " << *ConstUE << '\n';
             dbgs() << "Instruction: " << *Inst << '\n';)
    auto *BI = transValue(Inst, nullptr, false);
    Inst->dropAllReferences();
    UnboundInst.push_back(Inst);
    return BI;
  }

  // Translate aliases to their aliasee because they can't be represented
  // directly in SPIR-V.
  if (auto *const ConstAlias = dyn_cast<GlobalAlias>(V)) {
    return transValue(ConstAlias->getAliasee(), nullptr, false,
                      FuncTransMode::Pointer);
  }

  if (isa<UndefValue>(V)) {
    return BM->addUndef(ExpectedType);
  }

  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transValue(Value *V, SPIRVBasicBlock *BB,
                                        bool CreateForward,
                                        FuncTransMode FuncTrans) {
  LLVMToSPIRVValueMap::iterator Loc = ValueMap.find(V);
  if (Loc != ValueMap.end() && (!Loc->second->isForward() || CreateForward) &&
      // do not return forward-decl of a function if we
      // actually want to create a function pointer
      !(FuncTrans == FuncTransMode::Pointer && isa<Function>(V)))
    return Loc->second;

  SPIRVDBG(dbgs() << "[transValue] " << *V << '\n');
  assert((!isa<Instruction>(V) || isa<GetElementPtrInst>(V) ||
          isa<CastInst>(V) || isa<ExtractElementInst>(V) || isa<ICmpInst>(V) ||
          isa<BinaryOperator>(V) || BB) &&
         "Invalid SPIRV BB");

  auto *BV = transValueWithoutDecoration(V, BB, CreateForward, FuncTrans);
  if (!BV)
    return nullptr;
  // Only translate decorations for non-forward instructions.  Forward
  // instructions will have their decorations translated when the actual
  // instruction is seen and rewritten to a real SPIR-V instruction.
  if (!BV->isForward() && !transDecoration(V, BV))
    return nullptr;
  StringRef Name = V->getName();
  if (!Name.empty()) // Don't erase the name, which BM might already have
    BM->setName(BV, Name.str());
  return BV;
}

SPIRVInstruction *LLVMToSPIRVBase::transBinaryInst(BinaryOperator *B,
                                                   SPIRVBasicBlock *BB) {
  unsigned LLVMOC = B->getOpcode();
  auto *Op0 = transValue(B->getOperand(0), BB);
  SPIRVInstruction *BI = BM->addBinaryInst(
      transBoolOpCode(Op0, OpCodeMap::map(LLVMOC)), transType(B->getType()),
      Op0, transValue(B->getOperand(1), BB), BB);

  // BinaryOperator can have no parent if it is handled as an expression inside
  // another instruction.
  if (B->getParent() && isUnfusedMulAdd(B)) {
    Function *F = B->getFunction();
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": possible fma candidate " << *B << '\n');
    joinFPContract(F, FPContract::DISABLED);
  }

  return BI;
}

SPIRVInstruction *LLVMToSPIRVBase::transCmpInst(CmpInst *Cmp,
                                                SPIRVBasicBlock *BB) {
  auto *Op0 = Cmp->getOperand(0);
  SPIRVValue *TOp0 = transValue(Op0, BB);
  SPIRVValue *TOp1 = transValue(Cmp->getOperand(1), BB);
  // TODO: once the translator supports SPIR-V 1.4, update the condition below:
  // if (/* */->isPointerTy() && /* it is not allowed to use SPIR-V 1.4 */)
  if (Op0->getType()->isPointerTy()) {
    unsigned AS = cast<PointerType>(Op0->getType())->getAddressSpace();
    SPIRVType *Ty = transType(getSizetType(AS));
    TOp0 = BM->addUnaryInst(OpConvertPtrToU, Ty, TOp0, BB);
    TOp1 = BM->addUnaryInst(OpConvertPtrToU, Ty, TOp1, BB);
  }
  SPIRVInstruction *BI =
      BM->addCmpInst(transBoolOpCode(TOp0, CmpMap::map(Cmp->getPredicate())),
                     transType(Cmp->getType()), TOp0, TOp1, BB);
  return BI;
}

SPIRVValue *LLVMToSPIRVBase::transUnaryInst(UnaryInstruction *U,
                                            SPIRVBasicBlock *BB) {
  if (isa<BitCastInst>(U) && U->getType()->isPtrOrPtrVectorTy()) {
    if (isa<ConstantPointerNull>(U->getOperand(0))) {
      SPIRVType *ExpectedTy = transScavengedType(U);
      return BM->addNullConstant(bcast<SPIRVTypePointer>(ExpectedTy));
    }
    if (isa<UndefValue>(U->getOperand(0))) {
      SPIRVType *ExpectedTy = transScavengedType(U);
      return BM->addUndef(ExpectedTy);
    }
  }

  Op BOC = OpNop;
  if (auto *Cast = dyn_cast<AddrSpaceCastInst>(U)) {
    const auto SrcAddrSpace = Cast->getSrcTy()->getPointerAddressSpace();
    const auto DestAddrSpace = Cast->getDestTy()->getPointerAddressSpace();
    if (DestAddrSpace == SPIRAS_Generic) {
      getErrorLog().checkError(
          SrcAddrSpace != SPIRAS_Constant, SPIRVEC_InvalidModule, U,
          "Casts from constant address space to generic are illegal\n");
      BOC = OpPtrCastToGeneric;
      // In SPIR-V only casts to/from generic are allowed. But with
      // SPV_INTEL_usm_storage_classes we can also have casts from global_device
      // and global_host to global addr space and vice versa.
    } else if (SrcAddrSpace == SPIRAS_GlobalDevice ||
               SrcAddrSpace == SPIRAS_GlobalHost) {
      getErrorLog().checkError(DestAddrSpace == SPIRAS_Global ||
                                   DestAddrSpace == SPIRAS_Generic,
                               SPIRVEC_InvalidModule, U,
                               "Casts from global_device/global_host only "
                               "allowed to global/generic\n");
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes)) {
        if (DestAddrSpace == SPIRAS_Global)
          return nullptr;
        BOC = OpPtrCastToGeneric;
      } else {
        BOC = OpPtrCastToCrossWorkgroupINTEL;
      }
    } else if (DestAddrSpace == SPIRAS_GlobalDevice ||
               DestAddrSpace == SPIRAS_GlobalHost) {
      getErrorLog().checkError(SrcAddrSpace == SPIRAS_Global ||
                                   SrcAddrSpace == SPIRAS_Generic,
                               SPIRVEC_InvalidModule, U,
                               "Casts to global_device/global_host only "
                               "allowed from global/generic\n");
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes)) {
        if (SrcAddrSpace == SPIRAS_Global)
          return nullptr;
        BOC = OpGenericCastToPtr;
      } else {
        BOC = OpCrossWorkgroupCastToPtrINTEL;
      }
    } else {
      getErrorLog().checkError(
          SrcAddrSpace == SPIRAS_Generic, SPIRVEC_InvalidModule, U,
          "Casts from private/local/global address space are allowed only to "
          "generic\n");
      getErrorLog().checkError(
          DestAddrSpace != SPIRAS_Constant, SPIRVEC_InvalidModule, U,
          "Casts from generic address space to constant are illegal\n");
      BOC = OpGenericCastToPtr;
    }
  } else {
    auto OpCode = U->getOpcode();
    BOC = OpCodeMap::map(OpCode);
  }

  auto *Op = transValue(U->getOperand(0), BB, true, FuncTransMode::Pointer);
  SPIRVType *TransTy = transScavengedType(U);
  return BM->addUnaryInst(transBoolOpCode(Op, BOC), TransTy, Op, BB);
}

/// This helper class encapsulates information extraction from
/// "llvm.loop.parallel_access_indices" metadata hints. Initialize
/// with a pointer to an MDNode with the following structure:
/// !<Node> = !{!"llvm.loop.parallel_access_indices", !<Node>, !<Node>, ...}
/// OR:
/// !<Node> = !{!"llvm.loop.parallel_access_indices", !<Nodes...>, i32 <value>}
///
/// All of the MDNode-type operands mark the index groups for particular
/// array variables. An optional i32 value indicates the safelen (safe
/// number of iterations) for the optimization application to these
/// array variables. If the safelen value is absent, an infinite
/// number of iterations is implied.
class LLVMParallelAccessIndices {
public:
  LLVMParallelAccessIndices(
      MDNode *Node, LLVMToSPIRVBase::LLVMToSPIRVMetadataMap &IndexGroupArrayMap)
      : Node(Node), IndexGroupArrayMap(IndexGroupArrayMap) {

    assert(isValid() &&
           "LLVMParallelAccessIndices initialized from an invalid MDNode");

    unsigned NumOperands = Node->getNumOperands();
    auto *SafeLenExpression = mdconst::dyn_extract_or_null<ConstantInt>(
        Node->getOperand(NumOperands - 1));
    // If no safelen value is specified and the last operand
    // casts to an MDNode* rather than an int, 0 will be stored
    SafeLen = SafeLenExpression ? SafeLenExpression->getZExtValue() : 0;

    // Count MDNode operands that refer to index groups:
    // - operand [0] is a string literal and should be ignored;
    // - depending on whether a particular safelen is specified as the
    //   last operand, we may or may not want to extract the latter
    //   as an index group
    unsigned NumIdxGroups = SafeLen ? NumOperands - 2 : NumOperands - 1;
    for (unsigned I = 1; I <= NumIdxGroups; ++I) {
      MDNode *IdxGroupNode = getMDOperandAsMDNode(Node, I);
      assert(IdxGroupNode &&
             "Invalid operand in the MDNode for LLVMParallelAccessIndices");
      auto IdxGroupArrayPairIt = IndexGroupArrayMap.find(IdxGroupNode);
      // TODO: Some LLVM IR optimizations (e.g. loop inlining as part of
      // the function inlining) can result in invalid parallel_access_indices
      // metadata. Only valid cases will pass the subsequent check and
      // survive the translation. This check should be replaced with an
      // assertion once all known cases are handled.
      if (IdxGroupArrayPairIt != IndexGroupArrayMap.end())
        for (SPIRVId ArrayAccessId : IdxGroupArrayPairIt->second)
          ArrayVariablesVec.push_back(ArrayAccessId);
    }
  }

  bool isValid() {
    bool IsNamedCorrectly = getMDOperandAsString(Node, 0) == ExpectedName;
    return Node && IsNamedCorrectly;
  }

  unsigned getSafeLen() { return SafeLen; }
  const std::vector<SPIRVId> &getArrayVariables() { return ArrayVariablesVec; }

private:
  MDNode *Node;
  LLVMToSPIRVBase::LLVMToSPIRVMetadataMap &IndexGroupArrayMap;
  const std::string ExpectedName = "llvm.loop.parallel_access_indices";
  std::vector<SPIRVId> ArrayVariablesVec;
  unsigned SafeLen;
};

/// Go through the operands !llvm.loop metadata attached to the branch
/// instruction, fill the Loop Control mask and possible parameters for its
/// fields.
spv::LoopControlMask
LLVMToSPIRVBase::getLoopControl(const BranchInst *Branch,
                                std::vector<SPIRVWord> &Parameters) {
  if (!Branch)
    return spv::LoopControlMaskNone;
  MDNode *LoopMD = Branch->getMetadata("llvm.loop");
  if (!LoopMD)
    return spv::LoopControlMaskNone;

  size_t LoopControl = spv::LoopControlMaskNone;
  std::vector<std::pair<SPIRVWord, SPIRVWord>> ParametersToSort;
  // If only a subset of loop count parameters is defined in metadata
  // then undefined ones should have a default value -1 in SPIR-V.
  // Preset all loop count parameters with the default value.
  struct LoopCountInfo {
    int64_t Min = -1, Max = -1, Avg = -1;
  } LoopCount;

  // Unlike with most of the cases, some loop metadata specifications
  // can occur multiple times - for these, all correspondent tokens
  // need to be collected first, and only then added to SPIR-V loop
  // parameters in a separate routine
  std::vector<std::pair<SPIRVWord, SPIRVWord>> DependencyArrayParameters;

  for (const MDOperand &MDOp : LoopMD->operands()) {
    if (MDNode *Node = dyn_cast<MDNode>(MDOp)) {
      StringRef S = getMDOperandAsString(Node, 0);
      // Set the loop control bits. Parameters are set in the order described
      // in 3.23 SPIR-V Spec. rev. 1.4:
      // Bits that are set can indicate whether an additional operand follows,
      // as described by the table. If there are multiple following operands
      // indicated, they are ordered: Those indicated by smaller-numbered bits
      // appear first.
      if (S == "llvm.loop.unroll.disable")
        LoopControl |= spv::LoopControlDontUnrollMask;
      else if (S == "llvm.loop.unroll.enable")
        LoopControl |= spv::LoopControlUnrollMask;
      // Attempt to do full unroll of the loop and disable unrolling if the trip
      // count is not known at compile time by setting PartialCount to 1
      else if (S == "llvm.loop.unroll.full") {
        LoopControl |= spv::LoopControlUnrollMask;
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
          BM->setMinSPIRVVersion(VersionNumber::SPIRV_1_4);
          ParametersToSort.emplace_back(spv::LoopControlPartialCountMask, 1);
          LoopControl |= spv::LoopControlPartialCountMask;
        }
      }
      // PartialCount must not be used with the DontUnroll bit
      else if (S == "llvm.loop.unroll.count" &&
               !(LoopControl & LoopControlDontUnrollMask)) {
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
          BM->setMinSPIRVVersion(VersionNumber::SPIRV_1_4);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlPartialCountMask, I);
          LoopControl |= spv::LoopControlPartialCountMask;
        }
      } else if (S == "llvm.loop.ivdep.enable")
        LoopControl |= spv::LoopControlDependencyInfiniteMask;
      else if (S == "llvm.loop.ivdep.safelen") {
        size_t I = getMDOperandAsInt(Node, 1);
        ParametersToSort.emplace_back(spv::LoopControlDependencyLengthMask, I);
        LoopControl |= spv::LoopControlDependencyLengthMask;
      } else if (BM->isAllowedToUseExtension(
                     ExtensionID::SPV_INTEL_fpga_loop_controls)) {
        // Add Intel specific Loop Control masks
        if (S == "llvm.loop.ii.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlInitiationIntervalINTELMask, I);
          LoopControl |= spv::LoopControlInitiationIntervalINTELMask;
        } else if (S == "llvm.loop.max_concurrency.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlMaxConcurrencyINTELMask,
                                        I);
          LoopControl |= spv::LoopControlMaxConcurrencyINTELMask;
        } else if (S == "llvm.loop.parallel_access_indices") {
          // Intel FPGA IVDep loop attribute
          LLVMParallelAccessIndices IVDep(Node, IndexGroupArrayMap);
          // Store IVDep-specific parameters into an intermediate
          // container to address the case when there're multiple
          // IVDep metadata nodes and this condition gets entered multiple
          // times. The update of the main parameters vector & the loop control
          // mask will be done later, in the main scope of the function
          unsigned SafeLen = IVDep.getSafeLen();
          for (auto &ArrayId : IVDep.getArrayVariables())
            DependencyArrayParameters.emplace_back(ArrayId, SafeLen);
        } else if (S == "llvm.loop.intel.pipelining.enable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlPipelineEnableINTELMask,
                                        I);
          LoopControl |= spv::LoopControlPipelineEnableINTELMask;
        } else if (S == "llvm.loop.coalesce.enable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          ParametersToSort.emplace_back(spv::LoopControlLoopCoalesceINTELMask,
                                        0);
          LoopControl |= spv::LoopControlLoopCoalesceINTELMask;
        } else if (S == "llvm.loop.coalesce.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlLoopCoalesceINTELMask,
                                        I);
          LoopControl |= spv::LoopControlLoopCoalesceINTELMask;
        } else if (S == "llvm.loop.max_interleaving.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlMaxInterleavingINTELMask, I);
          LoopControl |= spv::LoopControlMaxInterleavingINTELMask;
        } else if (S == "llvm.loop.intel.speculated.iterations.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlSpeculatedIterationsINTELMask, I);
          LoopControl |= spv::LoopControlSpeculatedIterationsINTELMask;
        } else if (S == "llvm.loop.fusion.disable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopControl |= spv::LoopControlNoFusionINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_min") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Min = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_max") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Max = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_avg") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Avg = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.max_reinvocation_delay.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlMaxReinvocationDelayINTELMask, I);
          LoopControl |= spv::LoopControlMaxReinvocationDelayINTELMask;
        }
      }
    }
  }
  if (LoopControl & spv::LoopControlLoopCountINTELMask) {
    // LoopCountINTELMask have int64 literal parameters and we need to store
    // int64 into 2 SPIRVWords
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min));
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min >> 32));
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max));
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max >> 32));
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Avg));
    ParametersToSort.emplace_back(spv::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Avg >> 32));
  }
  // If any loop control parameters were held back until fully collected,
  // now is the time to move the information to the main parameters collection
  if (!DependencyArrayParameters.empty()) {
    // The first parameter states the number of <array, safelen> pairs to be
    // listed
    ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                  DependencyArrayParameters.size());
    for (auto &ArraySflnPair : DependencyArrayParameters) {
      ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                    ArraySflnPair.first);
      ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                    ArraySflnPair.second);
    }
    BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
    BM->addCapability(CapabilityFPGALoopControlsINTEL);
    LoopControl |= spv::LoopControlDependencyArrayINTELMask;
  }

  std::stable_sort(ParametersToSort.begin(), ParametersToSort.end(),
                   [](const std::pair<SPIRVWord, SPIRVWord> &CompareLeft,
                      const std::pair<SPIRVWord, SPIRVWord> &CompareRight) {
                     return CompareLeft.first < CompareRight.first;
                   });
  for (const auto &Param : ParametersToSort)
    Parameters.push_back(Param.second);

  return static_cast<spv::LoopControlMask>(LoopControl);
}

static int transAtomicOrdering(llvm::AtomicOrdering Ordering) {
  return OCLMemOrderMap::map(
      static_cast<OCLMemOrderKind>(llvm::toCABI(Ordering)));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicStore(StoreInst *ST,
                                              SPIRVBasicBlock *BB) {
  SmallVector<StringRef> SSIDs;
  ST->getContext().getSyncScopeNames(SSIDs);

  spv::Scope S;
  // Fill unknown syncscope value to default Device scope.
  if (!OCLStrMemScopeMap::find(SSIDs[ST->getSyncScopeID()].str(), &S)) {
    S = ScopeDevice;
  }

  std::vector<Value *> Ops{ST->getPointerOperand(), getUInt32(M, S),
                           getUInt32(M, transAtomicOrdering(ST->getOrdering())),
                           ST->getValueOperand()};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(ST, BM->addInstTemplate(OpAtomicStore, BM->getIds(SPIRVOps),
                                          BB, nullptr));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicLoad(LoadInst *LD,
                                             SPIRVBasicBlock *BB) {
  SmallVector<StringRef> SSIDs;
  LD->getContext().getSyncScopeNames(SSIDs);

  spv::Scope S;
  // Fill unknown syncscope value to default Device scope.
  if (!OCLStrMemScopeMap::find(SSIDs[LD->getSyncScopeID()].str(), &S)) {
    S = ScopeDevice;
  }

  std::vector<Value *> Ops{
      LD->getPointerOperand(), getUInt32(M, S),
      getUInt32(M, transAtomicOrdering(LD->getOrdering()))};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(LD, BM->addInstTemplate(OpAtomicLoad, BM->getIds(SPIRVOps),
                                          BB, transScavengedType(LD)));
}

// Aliasing list MD contains several scope MD nodes whithin it. Each scope MD
// has a selfreference and an extra MD node for aliasing domain and also it
// can contain an optional string operand. Domain MD contains a self-reference
// with an optional string operand. Here we unfold the list, creating SPIR-V
// aliasing instructions.
// TODO: add support for an optional string operand.
SPIRVEntry *addMemAliasingINTELInstructions(SPIRVModule *M,
                                            MDNode *AliasingListMD) {
  if (AliasingListMD->getNumOperands() == 0)
    return nullptr;
  std::vector<SPIRVId> ListId;
  for (const MDOperand &MDListOp : AliasingListMD->operands()) {
    if (MDNode *ScopeMD = dyn_cast<MDNode>(MDListOp)) {
      if (ScopeMD->getNumOperands() < 2)
        return nullptr;
      MDNode *DomainMD = dyn_cast<MDNode>(ScopeMD->getOperand(1));
      if (!DomainMD)
        return nullptr;
      auto *Domain =
          M->getOrAddAliasDomainDeclINTELInst(std::vector<SPIRVId>(), DomainMD);
      auto *Scope =
          M->getOrAddAliasScopeDeclINTELInst({Domain->getId()}, ScopeMD);
      ListId.push_back(Scope->getId());
    }
  }
  return M->getOrAddAliasScopeListDeclINTELInst(ListId, AliasingListMD);
}

// Translate alias.scope/noalias metadata attached to store and load
// instructions.
void transAliasingMemAccess(SPIRVModule *BM, MDNode *AliasingListMD,
                            std::vector<uint32_t> &MemoryAccess,
                            SPIRVWord MemAccessMask) {
  if (!BM->isAllowedToUseExtension(
        ExtensionID::SPV_INTEL_memory_access_aliasing))
    return;
  auto *MemAliasList = addMemAliasingINTELInstructions(BM, AliasingListMD);
  if (!MemAliasList)
    return;
  MemoryAccess[0] |= MemAccessMask;
  MemoryAccess.push_back(MemAliasList->getId());
}

/// An instruction may use an instruction from another BB which has not been
/// translated. SPIRVForward should be created as place holder for these
/// instructions and replaced later by the real instructions.
/// Use CreateForward = true to indicate such situation.
SPIRVValue *
LLVMToSPIRVBase::transValueWithoutDecoration(Value *V, SPIRVBasicBlock *BB,
                                             bool CreateForward,
                                             FuncTransMode FuncTrans) {
  if (auto *LBB = dyn_cast<BasicBlock>(V)) {
    auto *BF =
        static_cast<SPIRVFunction *>(getTranslatedValue(LBB->getParent()));
    assert(BF && "Function not translated");
    BB = static_cast<SPIRVBasicBlock *>(mapValue(V, BM->addBasicBlock(BF)));
    BM->setName(BB, LBB->getName().str());
    return BB;
  }

  if (auto *F = dyn_cast<Function>(V)) {
    if (FuncTrans == FuncTransMode::Decl)
      return transFunctionDecl(F);
    if (!BM->checkExtension(ExtensionID::SPV_INTEL_function_pointers,
                            SPIRVEC_FunctionPointers, toString(V)))
      return nullptr;
    return BM->addConstantFunctionPointerINTEL(
        transPointerType(transScavengedType(F), F->getAddressSpace()),
        static_cast<SPIRVFunction *>(transValue(F, nullptr)));
  }

  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    llvm::Type *Ty = Scavenger->getScavengedType(GV);
    // Though variables with common linkage type are initialized by 0,
    // they can be represented in SPIR-V as uninitialized variables with
    // 'Export' linkage type, just as tentative definitions look in C
    llvm::Constant *Init = GV->hasInitializer() && !GV->hasCommonLinkage()
                               ? GV->getInitializer()
                               : nullptr;
    SPIRVValue *BVarInit = nullptr;
    StructType *ST = Init ? dyn_cast<StructType>(Init->getType()) : nullptr;
    if (ST && ST->hasName() && isSPIRVConstantName(ST->getName())) {
      auto *BV = transConstant(Init);
      assert(BV);
      return mapValue(V, BV);
    }
    if (isa_and_nonnull<ConstantExpr>(Init)) {
      BVarInit = transValue(Init, nullptr);
    } else if (ST && isa<UndefValue>(Init)) {
      // Undef initializer for LLVM structure be can translated to
      // OpConstantComposite with OpUndef constituents.
      auto I = ValueMap.find(Init);
      if (I == ValueMap.end()) {
        std::vector<SPIRVValue *> Elements;
        for (Type *E : ST->elements())
          Elements.push_back(transValue(UndefValue::get(E), nullptr));
        BVarInit = BM->addCompositeConstant(transType(ST), Elements);
        ValueMap[Init] = BVarInit;
      } else
        BVarInit = I->second;
    } else if (Init && !isa<UndefValue>(Init)) {
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_long_constant_composite)) {
        if (auto *ArrTy = dyn_cast_or_null<ArrayType>(Init->getType())) {
          // First 3 words of OpConstantComposite encode: 1) word count &
          // opcode, 2) Result Type and 3) Result Id. Max length of SPIRV
          // instruction = 65535 words.
          constexpr int MaxNumElements =
              MaxWordCount - SPIRVSpecConstantComposite::FixedWC;
          if (ArrTy->getNumElements() > MaxNumElements &&
              !isa<ConstantAggregateZero>(Init)) {
            std::stringstream SS;
            SS << "Global variable has a constant array initializer with a "
               << "number of elements greater than OpConstantComposite can "
               << "have (" << MaxNumElements << "). Should the array be "
               << "split?\n Original LLVM value:\n"
               << toString(GV);
            getErrorLog().checkError(false, SPIRVEC_InvalidWordCount, SS.str());
          }
        }
      }
      SPIRVType *TransTy = transType(Ty);
      BVarInit = transConstantUse(Init, TransTy->getPointerElementType());
    }

    SPIRVStorageClassKind StorageClass;
    auto AddressSpace = static_cast<SPIRAddressSpace>(GV->getAddressSpace());
    bool IsVectorCompute =
        BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute) &&
        GV->hasAttribute(kVCMetadata::VCGlobalVariable);
    if (IsVectorCompute)
      StorageClass =
          VectorComputeUtil::getVCGlobalVarStorageClass(AddressSpace);
    else {
      // Lower global_device and global_host address spaces that were added in
      // SYCL as part of SYCL_INTEL_usm_address_spaces extension to just global
      // address space if device doesn't support SPV_INTEL_usm_storage_classes
      // extension
      if ((AddressSpace == SPIRAS_GlobalDevice ||
           AddressSpace == SPIRAS_GlobalHost) &&
          !BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes))
        AddressSpace = SPIRAS_Global;
      StorageClass = SPIRSPIRVAddrSpaceMap::map(AddressSpace);
      if (StorageClass == StorageClassFunction) {
        std::stringstream SS;
        SS << "Global variable cannot have Function storage class. "
           << "Consider setting a proper address space.\n "
           << "Original LLVM value:\n"
           << toString(GV);
        getErrorLog().checkError(false, SPIRVEC_InvalidInstruction, SS.str());
      }
    }

    SPIRVType *TranslatedTy = transType(Ty);
    auto *BVar = static_cast<SPIRVVariable *>(
        BM->addVariable(TranslatedTy, GV->isConstant(), transLinkageType(GV),
                        BVarInit, GV->getName().str(), StorageClass, nullptr));

    if (IsVectorCompute) {
      BVar->addDecorate(DecorationVectorComputeVariableINTEL);
      if (GV->hasAttribute(kVCMetadata::VCByteOffset)) {
        SPIRVWord Offset = {};
        GV->getAttribute(kVCMetadata::VCByteOffset)
            .getValueAsString()
            .getAsInteger(0, Offset);
        BVar->addDecorate(DecorationGlobalVariableOffsetINTEL, Offset);
      }
      if (GV->hasAttribute(kVCMetadata::VCVolatile))
        BVar->addDecorate(DecorationVolatile);

      if (GV->hasAttribute(kVCMetadata::VCSingleElementVector))
        translateSEVDecoration(
            GV->getAttribute(kVCMetadata::VCSingleElementVector), BVar);
    }

    mapValue(V, BVar);
    spv::BuiltIn Builtin = spv::BuiltInPosition;
    if (!GV->hasName() || !getSPIRVBuiltin(GV->getName().str(), Builtin))
      return BVar;
    if (static_cast<uint32_t>(Builtin) >= internal::BuiltInSubDeviceIDINTEL &&
        static_cast<uint32_t>(Builtin) <=
            internal::BuiltInGlobalHWThreadIDINTEL) {
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_hw_thread_queries)) {
        std::string ErrorStr = "Intel HW thread queries must be enabled by "
                               "SPV_INTEL_hw_thread_queries extension.\n"
                               "LLVM value that is being translated:\n";
        getErrorLog().checkError(false, SPIRVEC_InvalidModule, V, ErrorStr);
      }
      BM->addExtension(ExtensionID::SPV_INTEL_hw_thread_queries);
    }

    // TODO: it's W/A for intel/llvm to prevent not fixed SPIR-V consumers
    // see https://github.com/intel/llvm/issues/7592
    // from crashing. Need to remove, when we have the fixed drivers
    // to remove: begin
    {
      std::vector<Instruction *> GEPs;
      std::vector<Instruction *> Loads;
      auto *GVTy = GV->getType();
      auto *VecTy = GVTy->isOpaquePointerTy()
                        ? nullptr
                        : dyn_cast<FixedVectorType>(
                              GVTy->getNonOpaquePointerElementType());
      auto ReplaceIfLoad = [&](User *I, ConstantInt *Idx) -> void {
        auto *LD = dyn_cast<LoadInst>(I);
        if (!LD)
          return;
        Loads.push_back(LD);
        const DebugLoc &DLoc = LD->getDebugLoc();
        LoadInst *Load = new LoadInst(VecTy, GV, "", LD);
        ExtractElementInst *Extract = ExtractElementInst::Create(Load, Idx);
        if (DLoc)
          Extract->setDebugLoc(DLoc);
        Extract->insertAfter(cast<Instruction>(Load));
        LD->replaceAllUsesWith(Extract);
      };
      for (auto *UI : GV->users()) {
        if (!VecTy)
          break;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(UI)) {
          GEPs.push_back(GEP);
          for (auto *GEPUser : GEP->users()) {
            assert(GEP->getNumIndices() == 2 &&
                   "GEP to ID vector is expected to have exactly 2 indices");
            auto *Idx = cast<ConstantInt>(GEP->getOperand(2));
            ReplaceIfLoad(GEPUser, Idx);
          }
        }
      }
      auto Erase = [](std::vector<Instruction *> &ToErase) {
        for (Instruction *I : ToErase) {
          assert(I->user_empty());
          I->eraseFromParent();
        }
      };
      Erase(Loads);
      Erase(GEPs);
    }
    // to remove: end

    BVar->setBuiltin(Builtin);
    return BVar;
  }

  if (isa<Constant>(V)) {
    auto *BV = transConstant(V);
    assert(BV);
    // Don't store pointer constants in the map -- they are opaque and thus we
    // might reuse the wrong type (Example: a null value) if we do so.
    if (V->getType()->isPointerTy())
      return BV;
    return mapValue(V, BV);
  }

  if (auto *Arg = dyn_cast<Argument>(V)) {
    unsigned ArgNo = Arg->getArgNo();
    SPIRVFunction *BF = BB->getParent();
    // assert(BF->existArgument(ArgNo));
    return mapValue(V, BF->getArgument(ArgNo));
  }

  if (CreateForward)
    return mapValue(V, BM->addForward(transScavengedType(V)));

  if (StoreInst *ST = dyn_cast<StoreInst>(V)) {
    if (ST->isAtomic())
      return transAtomicStore(ST, BB);

    // Keep this vector to store MemoryAccess operands for both Alignment and
    // Aliasing information.
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    if (ST->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    MemoryAccess[0] |= MemoryAccessAlignedMask;
    MemoryAccess.push_back(ST->getAlign().value());
    if (ST->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    if (MDNode *AliasingListMD = ST->getMetadata(LLVMContext::MD_alias_scope))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             MemoryAccessAliasScopeINTELMaskMask);
    if (MDNode *AliasingListMD = ST->getMetadata(LLVMContext::MD_noalias))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             MemoryAccessNoAliasINTELMaskMask);
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();

    return mapValue(V,
                    BM->addStoreInst(transValue(ST->getPointerOperand(), BB),
                                     transValue(ST->getValueOperand(), BB, true,
                                                FuncTransMode::Pointer),
                                     MemoryAccess, BB));
  }

  if (LoadInst *LD = dyn_cast<LoadInst>(V)) {
    if (LD->isAtomic())
      return transAtomicLoad(LD, BB);

    // Keep this vector to store MemoryAccess operands for both Alignment and
    // Aliasing information.
    std::vector<uint32_t> MemoryAccess(1, 0);
    if (LD->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    MemoryAccess[0] |= MemoryAccessAlignedMask;
    MemoryAccess.push_back(LD->getAlign().value());
    if (LD->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    if (MDNode *AliasingListMD = LD->getMetadata(LLVMContext::MD_alias_scope))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             MemoryAccessAliasScopeINTELMaskMask);
    if (MDNode *AliasingListMD = LD->getMetadata(LLVMContext::MD_noalias))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             MemoryAccessNoAliasINTELMaskMask);
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();
    return mapValue(V, BM->addLoadInst(transValue(LD->getPointerOperand(), BB),
                                       MemoryAccess, BB));
  }

  if (BinaryOperator *B = dyn_cast<BinaryOperator>(V)) {
    SPIRVInstruction *BI = transBinaryInst(B, BB);
    return mapValue(V, BI);
  }

  if (dyn_cast<UnreachableInst>(V))
    return mapValue(V, BM->addUnreachableInst(BB));

  if (auto *RI = dyn_cast<ReturnInst>(V)) {
    if (auto *RV = RI->getReturnValue()) {
      if (auto *II = dyn_cast<IntrinsicInst>(RV)) {
        if (II->getIntrinsicID() == Intrinsic::frexp) {
          // create composite type from the return value and second operand
          auto *FrexpResult = transValue(RV, BB);
          SPIRVValue *IntFromFrexpResult =
              static_cast<SPIRVExtInst *>(FrexpResult)->getArgValues()[1];
          IntFromFrexpResult = BM->addLoadInst(IntFromFrexpResult, {}, BB);

          std::vector<SPIRVId> Operands = {FrexpResult->getId(),
                                           IntFromFrexpResult->getId()};
          auto *Compos = BM->addCompositeConstructInst(transType(RV->getType()),
                                                       Operands, BB);

          return mapValue(V, BM->addReturnValueInst(Compos, BB));
        }
      }
      return mapValue(V, BM->addReturnValueInst(transValue(RV, BB), BB));
    }
    return mapValue(V, BM->addReturnInst(BB));
  }

  if (CmpInst *Cmp = dyn_cast<CmpInst>(V)) {
    if (Cmp->getPredicate() == CmpInst::Predicate::FCMP_FALSE) {
      auto *CmpTy = Cmp->getType();
      SPIRVValue *FalseValue = CmpTy->isVectorTy()
                                   ? BM->addNullConstant(transType(CmpTy))
                                   : BM->addConstant(BM->addBoolType(), 0);
      return mapValue(V, FalseValue);
    }
    SPIRVInstruction *BI = transCmpInst(Cmp, BB);
    return mapValue(V, BI);
  }

  if (SelectInst *Sel = dyn_cast<SelectInst>(V))
    return mapValue(
        V,
        BM->addSelectInst(
            transValue(Sel->getCondition(), BB),
            transValue(Sel->getTrueValue(), BB, true, FuncTransMode::Pointer),
            transValue(Sel->getFalseValue(), BB, true, FuncTransMode::Pointer),
            BB));

  if (AllocaInst *Alc = dyn_cast<AllocaInst>(V)) {
    SPIRVType *TranslatedTy = transScavengedType(V);
    if (Alc->isArrayAllocation()) {
      SPIRVValue *Length = transValue(Alc->getArraySize(), BB);
      assert(Length && "Couldn't translate array size!");

      if (isSpecConstantOpCode(Length->getOpCode())) {
        // SPIR-V arrays length can be expressed using a specialization
        // constant.
        //
        // Spec Constant Length Arrays need special treatment, as the allocation
        // type will be 'OpTypePointer(Function, OpTypeArray(ElementType,
        // Length))', we need to bitcast the obtained pointer to the expected
        // type: 'OpTypePointer(Function, ElementType).
        SPIRVType *AllocationType = BM->addPointerType(
            StorageClassFunction,
            BM->addArrayType(transType(Alc->getAllocatedType()), Length));
        SPIRVValue *Arr = BM->addVariable(
            AllocationType, false, spv::internal::LinkageTypeInternal, nullptr,
            Alc->getName().str() + "_alloca", StorageClassFunction, BB);
        // Manually set alignment. OpBitcast created below will be decorated as
        // that's the SPIR-V value mapped to the original LLVM one.
        transAlign(Alc, Arr);
        return mapValue(V, BM->addUnaryInst(OpBitcast, TranslatedTy, Arr, BB));
      }

      if (!BM->checkExtension(ExtensionID::SPV_INTEL_variable_length_array,
                              SPIRVEC_InvalidInstruction,
                              toString(Alc) +
                                  "\nTranslation of dynamic alloca requires "
                                  "SPV_INTEL_variable_length_array extension."))
        return nullptr;

      return mapValue(V,
                      BM->addInstTemplate(OpVariableLengthArrayINTEL,
                                          {Length->getId()}, BB, TranslatedTy));
    }
    return mapValue(V, BM->addVariable(TranslatedTy, false,
                                       spv::internal::LinkageTypeInternal,
                                       nullptr, Alc->getName().str(),
                                       StorageClassFunction, BB));
  }

  if (auto *Switch = dyn_cast<SwitchInst>(V)) {
    std::vector<SPIRVSwitch::PairTy> Pairs;
    auto *Select = transValue(Switch->getCondition(), BB);

    for (auto I = Switch->case_begin(), E = Switch->case_end(); I != E; ++I) {
      SPIRVSwitch::LiteralTy Lit;
      uint64_t CaseValue = I->getCaseValue()->getZExtValue();

      Lit.push_back(CaseValue);
      assert(Select->getType()->getBitWidth() <= 64 &&
             "unexpected selector bitwidth");
      if (Select->getType()->getBitWidth() == 64)
        Lit.push_back(CaseValue >> 32);

      Pairs.push_back(
          std::make_pair(Lit, static_cast<SPIRVBasicBlock *>(
                                  transValue(I->getCaseSuccessor(), nullptr))));
    }

    return mapValue(
        V, BM->addSwitchInst(Select,
                             static_cast<SPIRVBasicBlock *>(
                                 transValue(Switch->getDefaultDest(), nullptr)),
                             Pairs, BB));
  }

  if (BranchInst *Branch = dyn_cast<BranchInst>(V)) {
    SPIRVLabel *SuccessorTrue =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(0), BB));

    /// Clang attaches !llvm.loop metadata to "latch" BB. This kind of blocks
    /// has an edge directed to the loop header. Thus latch BB matching to
    /// "Continue Target" per the SPIR-V spec. This statement is true only after
    /// applying the loop-simplify pass to the LLVM module.
    /// For "for" and "while" loops latch BB is terminated by an
    /// unconditional branch. Also for this kind of loops "Merge Block" can
    /// be found as block targeted by false edge of the "Header" BB.
    /// For "do while" loop the latch is terminated by a conditional branch
    /// with true edge going to the header and the false edge going out of
    /// the loop, which corresponds to a "Merge Block" per the SPIR-V spec.
    std::vector<SPIRVWord> Parameters;
    spv::LoopControlMask LoopControl = getLoopControl(Branch, Parameters);

    if (Branch->isUnconditional()) {
      // Usually, "for" and "while" loops llvm.loop metadata is attached to an
      // unconditional branch instruction.
      if (LoopControl != spv::LoopControlMaskNone) {
        // SuccessorTrue is the loop header BB.
        const SPIRVInstruction *Term = SuccessorTrue->getTerminateInstr();
        if (Term && Term->getOpCode() == OpBranchConditional) {
          const auto *Br = static_cast<const SPIRVBranchConditional *>(Term);
          BM->addLoopMergeInst(Br->getFalseLabel()->getId(), // Merge Block
                               BB->getId(),                  // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
        } else {
          if (BM->isAllowedToUseExtension(
                  ExtensionID::SPV_INTEL_unstructured_loop_controls)) {
            // For unstructured loop we add a special loop control instruction.
            // Simple example of unstructured loop is an infinite loop, that has
            // no terminate instruction.
            BM->addLoopControlINTELInst(LoopControl, Parameters, SuccessorTrue);
          }
        }
      }
      return mapValue(V, BM->addBranchInst(SuccessorTrue, BB));
    }
    // For "do-while" (and in some cases, for "for" and "while") loops,
    // llvm.loop metadata is attached to a conditional branch instructions
    SPIRVLabel *SuccessorFalse =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(1), BB));
    if (LoopControl != spv::LoopControlMaskNone) {
      Function *Fun = Branch->getFunction();
      DominatorTree DomTree(*Fun);
      LoopInfo LI(DomTree);
      for (const auto *LoopObj : LI.getLoopsInPreorder()) {
        // Check whether SuccessorFalse or SuccessorTrue is the loop header BB.
        // For example consider following LLVM IR:
        // br i1 %compare, label %for.body, label %for.end
        //   <- SuccessorTrue is 'for.body' aka successor(0)
        // br i1 %compare.not, label %for.end, label %for.body
        //   <- SuccessorTrue is 'for.end' aka successor(1)
        // meanwhile the true successor (by definition) should be a loop header
        // aka 'for.body'
        if (LoopObj->getHeader() == Branch->getSuccessor(1))
          // SuccessorFalse is the loop header BB.
          BM->addLoopMergeInst(SuccessorTrue->getId(), // Merge Block
                               BB->getId(),            // Continue Target
                               LoopControl, Parameters, SuccessorFalse);
        else
          // SuccessorTrue is the loop header BB.
          BM->addLoopMergeInst(SuccessorFalse->getId(), // Merge Block
                               BB->getId(),             // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
      }
    }
    return mapValue(
        V, BM->addBranchConditionalInst(transValue(Branch->getCondition(), BB),
                                        SuccessorTrue, SuccessorFalse, BB));
  }

  if (auto *Phi = dyn_cast<PHINode>(V)) {
    std::vector<SPIRVValue *> IncomingPairs;

    for (size_t I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
      IncomingPairs.push_back(transValue(Phi->getIncomingValue(I), BB, true,
                                         FuncTransMode::Pointer));
      IncomingPairs.push_back(transValue(Phi->getIncomingBlock(I), nullptr));
    }
    return mapValue(V,
                    BM->addPhiInst(transScavengedType(Phi), IncomingPairs, BB));
  }

  if (auto *Ext = dyn_cast<ExtractValueInst>(V)) {
    if (auto *II = dyn_cast<IntrinsicInst>(Ext->getAggregateOperand())) {
      if (II->getIntrinsicID() == Intrinsic::frexp) {
        unsigned Idx = Ext->getIndices()[0];
        auto *Val = transValue(II, BB);
        if (Idx == 0)
          return mapValue(V, Val);

        // Idx = 1
        SPIRVValue *IntFromFrexpResult =
            static_cast<SPIRVExtInst *>(Val)->getArgValues()[1];
        IntFromFrexpResult = BM->addLoadInst(IntFromFrexpResult, {}, BB);
        return mapValue(V, IntFromFrexpResult);
      }
    }
    return mapValue(V, BM->addCompositeExtractInst(
                           transScavengedType(Ext),
                           transValue(Ext->getAggregateOperand(), BB),
                           Ext->getIndices(), BB));
  }

  if (auto *Ins = dyn_cast<InsertValueInst>(V)) {
    return mapValue(V, BM->addCompositeInsertInst(
                           transValue(Ins->getInsertedValueOperand(), BB),
                           transValue(Ins->getAggregateOperand(), BB),
                           Ins->getIndices(), BB));
  }

  if (UnaryInstruction *U = dyn_cast<UnaryInstruction>(V)) {
    auto *UI = transUnaryInst(U, BB);
    return mapValue(V, UI ? UI : transValue(U->getOperand(0), BB));
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    std::vector<SPIRVValue *> Indices;
    for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
      Indices.push_back(transValue(GEP->getOperand(I + 1), BB));
    auto *PointerOperand = GEP->getPointerOperand();
    auto *TransPointerOperand = transValue(PointerOperand, BB);

    // Certain array-related optimization hints can be expressed via
    // LLVM metadata. For the purpose of linking this metadata with
    // the accessed array variables, our GEP may have been marked into
    // a so-called index group, an MDNode by itself.
    if (MDNode *IndexGroup = GEP->getMetadata("llvm.index.group")) {
      SPIRVValue *ActualMemoryPtr = TransPointerOperand;
      // If the source is a no-op bitcast (generated to fix up types), look
      // through it to the underlying gep if possible.
      if (auto *BC = dyn_cast<CastInst>(PointerOperand))
        if (BC->getSrcTy() == BC->getDestTy()) {
          PointerOperand = BC->getOperand(0);
        }
      if (auto *Load = dyn_cast<LoadInst>(PointerOperand)) {
        ActualMemoryPtr = transValue(Load->getPointerOperand(), BB);
      }
      SPIRVId AccessedArrayId = ActualMemoryPtr->getId();
      unsigned NumOperands = IndexGroup->getNumOperands();
      // When we're working with embedded loops, it's natural that
      // the outer loop's hints apply to all code contained within.
      // The inner loop's specific hints, however, should stay private
      // to the inner loop's scope.
      // Consequently, the following division of the index group metadata
      // nodes emerges:

      // 1) The metadata node has no operands. It will be directly referenced
      //    from within the optimization hint metadata.
      if (NumOperands == 0)
        IndexGroupArrayMap[IndexGroup].insert(AccessedArrayId);
      // 2) The metadata node has several operands. It serves to link an index
      //    group specific to some embedded loop with other index groups that
      //    mark the same array variable for the outer loop(s).
      for (unsigned I = 0; I < NumOperands; ++I) {
        auto *ContainedIndexGroup = getMDOperandAsMDNode(IndexGroup, I);
        IndexGroupArrayMap[ContainedIndexGroup].insert(AccessedArrayId);
      }
    }
    // GEP can return a vector of pointers, in this case GEP will calculate
    // addresses for each pointer in the vector
    SPIRVType *TranslatedTy = transScavengedType(GEP);
    return mapValue(V,
                    BM->addPtrAccessChainInst(TranslatedTy, TransPointerOperand,
                                              Indices, BB, GEP->isInBounds()));
  }

  if (auto *Ext = dyn_cast<ExtractElementInst>(V)) {
    auto *Index = Ext->getIndexOperand();
    if (auto *Const = dyn_cast<ConstantInt>(Index))
      return mapValue(V, BM->addCompositeExtractInst(
                             transScavengedType(Ext),
                             transValue(Ext->getVectorOperand(), BB),
                             std::vector<SPIRVWord>(1, Const->getZExtValue()),
                             BB));
    else
      return mapValue(V, BM->addVectorExtractDynamicInst(
                             transValue(Ext->getVectorOperand(), BB),
                             transValue(Index, BB), BB));
  }

  if (auto *Ins = dyn_cast<InsertElementInst>(V)) {
    auto *Index = Ins->getOperand(2);
    if (auto *Const = dyn_cast<ConstantInt>(Index)) {
      return mapValue(
          V,
          BM->addCompositeInsertInst(
              transValue(Ins->getOperand(1), BB, true, FuncTransMode::Pointer),
              transValue(Ins->getOperand(0), BB),
              std::vector<SPIRVWord>(1, Const->getZExtValue()), BB));
    } else
      return mapValue(
          V, BM->addVectorInsertDynamicInst(transValue(Ins->getOperand(0), BB),
                                            transValue(Ins->getOperand(1), BB),
                                            transValue(Index, BB), BB));
  }

  if (auto *SF = dyn_cast<ShuffleVectorInst>(V)) {
    std::vector<SPIRVWord> Comp;
    for (auto &I : SF->getShuffleMask())
      Comp.push_back(I);
    return mapValue(V, BM->addVectorShuffleInst(
                           transScavengedType(SF),
                           transValue(SF->getOperand(0), BB),
                           transValue(SF->getOperand(1), BB), Comp, BB));
  }

  if (AtomicRMWInst *ARMW = dyn_cast<AtomicRMWInst>(V)) {
    AtomicRMWInst::BinOp Op = ARMW->getOperation();
    bool SupportedAtomicInst =
        AtomicRMWInst::isFPOperation(Op)
            ? (Op == AtomicRMWInst::FAdd || Op == AtomicRMWInst::FSub ||
               Op == AtomicRMWInst::FMin || Op == AtomicRMWInst::FMax)
            : Op != AtomicRMWInst::Nand;
    if (!BM->getErrorLog().checkError(
            SupportedAtomicInst, SPIRVEC_InvalidInstruction, V,
            "Atomic " + AtomicRMWInst::getOperationName(Op).str() +
                " is not supported in SPIR-V!\n"))
      return nullptr;

    AtomicOrderingCABI Ordering = llvm::toCABI(ARMW->getOrdering());
    auto MemSem = OCLMemOrderMap::map(static_cast<OCLMemOrderKind>(Ordering));
    std::vector<Value *> Operands(4);
    Operands[0] = ARMW->getPointerOperand();
    // To get the memory scope argument we use ARMW->getSyncScopeID(), but
    // atomicrmw LLVM instruction is not aware of OpenCL(or SPIR-V) memory scope
    // enumeration. If the scope is not set and assuming the produced SPIR-V
    // module will be consumed in an OpenCL environment, we can use the same
    // memory scope as OpenCL atomic functions that don't have memory_scope
    // argument i.e. memory_scope_device. See the OpenCL C specification
    // p6.13.11. "Atomic Functions"
    SmallVector<StringRef> SSIDs;
    ARMW->getContext().getSyncScopeNames(SSIDs);

    spv::Scope S;
    // Fill unknown syncscope value to default Device scope.
    if (!OCLStrMemScopeMap::find(SSIDs[ARMW->getSyncScopeID()].str(), &S)) {
      S = ScopeDevice;
    }
    Operands[1] = getUInt32(M, S);
    Operands[2] = getUInt32(M, MemSem);
    Operands[3] = ARMW->getValOperand();
    std::vector<SPIRVValue *> OpVals = transValue(Operands, BB);
    std::vector<SPIRVId> Ops = BM->getIds(OpVals);
    SPIRVType *Ty = transScavengedType(ARMW);

    spv::Op OC;
    if (Op == AtomicRMWInst::FSub) {
      // Implement FSub through FNegate and AtomicFAddExt
      Ops[3] = BM->addUnaryInst(OpFNegate, Ty, OpVals[3], BB)->getId();
      OC = OpAtomicFAddEXT;
    } else
      OC = LLVMSPIRVAtomicRmwOpCodeMap::map(Op);

    return mapValue(V, BM->addInstTemplate(OC, Ops, BB, Ty));
  }

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(V)) {
    SPIRVValue *BV = transIntrinsicInst(II, BB);
    return BV ? mapValue(V, BV) : nullptr;
  }

  if (FenceInst *FI = dyn_cast<FenceInst>(V)) {
    SPIRVValue *BV = transFenceInst(FI, BB);
    return BV ? mapValue(V, BV) : nullptr;
  }

  if (InlineAsm *IA = dyn_cast<InlineAsm>(V))
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_inline_assembly))
      return mapValue(V, transAsmINTEL(IA));

  if (CallInst *CI = dyn_cast<CallInst>(V)) {
    if (auto *Alias =
            dyn_cast_or_null<llvm::GlobalAlias>(CI->getCalledOperand())) {
      CI->setCalledFunction(cast<Function>(Alias->getAliasee()));
    }
    return mapValue(V, transCallInst(CI, BB));
  }

  if (Instruction *Inst = dyn_cast<Instruction>(V)) {
    BM->SPIRVCK(false, InvalidInstruction, toString(Inst));
  }

  llvm_unreachable("Not implemented");
  return nullptr;
}

SPIRVType *LLVMToSPIRVBase::mapType(Type *T, SPIRVType *BT) {
  assert(!T->isPointerTy() && "Pointer types cannot be stored in the type map");
  auto EmplaceStatus = TypeMap.try_emplace(T, BT);
  // TODO: Uncomment the assertion, once the type mapping issue is resolved
  // assert(EmplaceStatus.second && "The type was already added to the map");
  SPIRVDBG(dbgs() << "[mapType] " << *T << " => "; spvdbgs() << *BT << '\n');
  if (!EmplaceStatus.second)
    return TypeMap[T];
  return BT;
}

SPIRVValue *LLVMToSPIRVBase::mapValue(Value *V, SPIRVValue *BV) {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end()) {
    if (Loc->second == BV)
      return BV;
    assert(Loc->second->isForward() &&
           "LLVM Value is mapped to different SPIRV Values");
    auto *Forward = static_cast<SPIRVForward *>(Loc->second);
    BM->replaceForward(Forward, BV);
  }
  ValueMap[V] = BV;
  SPIRVDBG(dbgs() << "[mapValue] " << *V << " => "; spvdbgs() << BV << "\n");
  return BV;
}

bool LLVMToSPIRVBase::shouldTryToAddMemAliasingDecoration(Instruction *Inst) {
  // Limit translation of aliasing metadata with only this set of instructions
  // gracefully considering others as compilation mistakes and ignoring them
  if (!Inst->mayReadOrWriteMemory())
    return false;
  // Loads and Stores are handled during memory access mask addition
  if (isa<StoreInst>(Inst) || isa<LoadInst>(Inst))
    return false;
  CallInst *CI = dyn_cast<CallInst>(Inst);
  if (!CI)
    return true;
  if (Function *Fun = CI->getCalledFunction()) {
    // Calls to intrinsics are skipped. At some point lifetime start/end will be
    // handled separately, but specification isn't ready.
    if (Fun->isIntrinsic())
      return false;
    // Also skip SPIR-V instructions that don't have result id to attach the
    // decorations
    if (isBuiltinTransToInst(Fun))
      if (Fun->getReturnType()->isVoidTy())
        return false;
  }
  return true;
}

void addFuncPointerCallArgumentAttributes(CallInst *CI,
                                          SPIRVValue *FuncPtrCall) {
  for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ++ArgNo) {
    for (const auto &I : CI->getAttributes().getParamAttrs(ArgNo)) {
      spv::FunctionParameterAttribute Attr = spv::FunctionParameterAttributeMax;
      SPIRSPIRVFuncParamAttrMap::find(I.getKindAsEnum(), &Attr);
      if (Attr != spv::FunctionParameterAttributeMax)
        FuncPtrCall->addDecorate(
            new SPIRVDecorate(spv::internal::DecorationArgumentAttributeINTEL,
                              FuncPtrCall, ArgNo, Attr));
    }
  }
}

#define ONE_STRING_DECORATION_CASE(NAME, NAMESPACE)                            \
  case NAMESPACE::Decoration##NAME: {                                          \
    ErrLog.checkError(NumOperands == 2, SPIRVEC_InvalidLlvmModule,             \
                      #NAME " requires exactly 1 extra operand");              \
    auto *StrDecoEO = dyn_cast<MDString>(DecoMD->getOperand(1));               \
    ErrLog.checkError(StrDecoEO, SPIRVEC_InvalidLlvmModule,                    \
                      #NAME " requires extra operand to be a string");         \
    Target->addDecorate(                                                       \
        new SPIRVDecorate##NAME##Attr(Target, StrDecoEO->getString().str()));  \
    break;                                                                     \
  }

#define ONE_INT_DECORATION_CASE(NAME, NAMESPACE, TYPE)                         \
  case NAMESPACE::Decoration##NAME: {                                          \
    ErrLog.checkError(NumOperands == 2, SPIRVEC_InvalidLlvmModule,             \
                      #NAME " requires exactly 1 extra operand");              \
    auto *IntDecoEO =                                                          \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));              \
    ErrLog.checkError(IntDecoEO, SPIRVEC_InvalidLlvmModule,                    \
                      #NAME " requires extra operand to be an integer");       \
    Target->addDecorate(new SPIRVDecorate##NAME(                               \
        Target, static_cast<TYPE>(IntDecoEO->getZExtValue())));                \
    break;                                                                     \
  }

#define TWO_INT_DECORATION_CASE(NAME, NAMESPACE, TYPE1, TYPE2)                 \
  case NAMESPACE::Decoration##NAME: {                                          \
    ErrLog.checkError(NumOperands == 3, SPIRVEC_InvalidLlvmModule,             \
                      #NAME " requires exactly 2 extra operands");             \
    auto *IntDecoEO1 =                                                         \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));              \
    ErrLog.checkError(IntDecoEO1, SPIRVEC_InvalidLlvmModule,                   \
                      #NAME " requires first extra operand to be an integer"); \
    auto *IntDecoEO2 =                                                         \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));              \
    ErrLog.checkError(IntDecoEO2, SPIRVEC_InvalidLlvmModule,                   \
                      #NAME                                                    \
                      " requires second extra operand to be an integer");      \
    Target->addDecorate(new SPIRVDecorate##NAME(                               \
        Target, static_cast<TYPE1>(IntDecoEO1->getZExtValue()),                \
        static_cast<TYPE2>(IntDecoEO2->getZExtValue())));                      \
    break;                                                                     \
  }

void checkIsGlobalVar(SPIRVEntry *E, Decoration Dec) {
  std::string ErrStr =
      SPIRVDecorationNameMap::map(Dec) + " can only be applied to a variable";

  E->getErrorLog().checkError(E->isVariable(), SPIRVEC_InvalidModule, ErrStr);

  auto AddrSpace = SPIRSPIRVAddrSpaceMap::rmap(
      static_cast<SPIRVVariable *>(E)->getStorageClass());
  ErrStr += " in a global (module) scope";
  E->getErrorLog().checkError(AddrSpace == SPIRAS_Global, SPIRVEC_InvalidModule,
                              ErrStr);
}

static void transMetadataDecorations(Metadata *MD, SPIRVValue *Target) {
  SPIRVErrorLog &ErrLog = Target->getErrorLog();

  auto *ArgDecoMD = dyn_cast<MDNode>(MD);
  assert(ArgDecoMD && "Decoration list must be a metadata node");
  for (unsigned I = 0, E = ArgDecoMD->getNumOperands(); I != E; ++I) {
    auto *DecoMD = dyn_cast<MDNode>(ArgDecoMD->getOperand(I));
    ErrLog.checkError(DecoMD, SPIRVEC_InvalidLlvmModule,
                      "Decoration does not name metadata");
    ErrLog.checkError(DecoMD->getNumOperands() > 0, SPIRVEC_InvalidLlvmModule,
                      "Decoration metadata must have at least one operand");
    auto *DecoKindConst =
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(0));
    ErrLog.checkError(DecoKindConst, SPIRVEC_InvalidLlvmModule,
                      "First operand of decoration must be the kind");
    auto DecoKind = static_cast<Decoration>(DecoKindConst->getZExtValue());

    const size_t NumOperands = DecoMD->getNumOperands();
    switch (static_cast<size_t>(DecoKind)) {
    case DecorationAlignment: {
      // Handle Alignment via SPIRVValue::setAlignment() to avoid duplicate
      // Alignment decorations.
      auto *Alignment =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(Alignment, SPIRVEC_InvalidLlvmModule,
                        "Alignment operand must be an integer.");
      Target->setAlignment(Alignment->getZExtValue());
      break;
    }

      ONE_STRING_DECORATION_CASE(MemoryINTEL, spv)
      ONE_STRING_DECORATION_CASE(UserSemantic, spv)
      ONE_INT_DECORATION_CASE(AliasScopeINTEL, spv, SPIRVId)
      ONE_INT_DECORATION_CASE(NoAliasINTEL, spv, SPIRVId)
      ONE_INT_DECORATION_CASE(InitiationIntervalINTEL, spv, SPIRVWord)
      ONE_INT_DECORATION_CASE(MaxConcurrencyINTEL, spv, SPIRVWord)
      ONE_INT_DECORATION_CASE(PipelineEnableINTEL, spv, SPIRVWord)
      TWO_INT_DECORATION_CASE(FunctionRoundingModeINTEL, spv, SPIRVWord,
                              FPRoundingMode);
      TWO_INT_DECORATION_CASE(FunctionDenormModeINTEL, spv, SPIRVWord,
                              FPDenormMode);
      TWO_INT_DECORATION_CASE(FunctionFloatingPointModeINTEL, spv, SPIRVWord,
                              FPOperationMode);
      TWO_INT_DECORATION_CASE(FuseLoopsInFunctionINTEL, spv, SPIRVWord,
                              SPIRVWord);
      TWO_INT_DECORATION_CASE(MathOpDSPModeINTEL, spv, SPIRVWord, SPIRVWord);

    case DecorationConduitKernelArgumentINTEL:
    case DecorationRegisterMapKernelArgumentINTEL:
    case DecorationStableKernelArgumentINTEL:
    case DecorationRestrict: {
      Target->addDecorate(new SPIRVDecorate(DecoKind, Target));
      break;
    }
    case DecorationBufferLocationINTEL:
    case DecorationMMHostInterfaceReadWriteModeINTEL:
    case DecorationMMHostInterfaceAddressWidthINTEL:
    case DecorationMMHostInterfaceDataWidthINTEL:
    case DecorationMMHostInterfaceLatencyINTEL:
    case DecorationMMHostInterfaceMaxBurstINTEL:
    case DecorationMMHostInterfaceWaitRequestINTEL: {
      ErrLog.checkError(NumOperands == 2, SPIRVEC_InvalidLlvmModule,
                        "MMHost Kernel Argument Annotation requires exactly 2 "
                        "extra operands");
      auto *DecoValEO1 =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      Target->addDecorate(
          new SPIRVDecorate(DecoKind, Target, DecoValEO1->getZExtValue()));
      break;
    }
    case DecorationStallEnableINTEL: {
      Target->addDecorate(new SPIRVDecorateStallEnableINTEL(Target));
      break;
    }
    case DecorationStallFreeINTEL: {
      Target->addDecorate(new SPIRVDecorateStallFreeINTEL(Target));
      break;
    }
    case DecorationMergeINTEL: {
      ErrLog.checkError(NumOperands == 3, SPIRVEC_InvalidLlvmModule,
                        "MergeINTEL requires exactly 3 extra operands");
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(1));
      ErrLog.checkError(
          Name, SPIRVEC_InvalidLlvmModule,
          "MergeINTEL requires first extra operand to be a string");
      auto *Direction = dyn_cast<MDString>(DecoMD->getOperand(2));
      ErrLog.checkError(
          Direction, SPIRVEC_InvalidLlvmModule,
          "MergeINTEL requires second extra operand to be a string");
      Target->addDecorate(new SPIRVDecorateMergeINTELAttr(
          Target, Name->getString().str(), Direction->getString().str()));
      break;
    }
    case DecorationLinkageAttributes: {
      ErrLog.checkError(NumOperands == 3, SPIRVEC_InvalidLlvmModule,
                        "LinkageAttributes requires exactly 3 extra operands");
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(1));
      ErrLog.checkError(
          Name, SPIRVEC_InvalidLlvmModule,
          "LinkageAttributes requires first extra operand to be a string");
      auto *Type = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      ErrLog.checkError(
          Type, SPIRVEC_InvalidLlvmModule,
          "LinkageAttributes requires second extra operand to be an int");
      auto TypeKind = static_cast<SPIRVLinkageTypeKind>(Type->getZExtValue());
      Target->addDecorate(new SPIRVDecorateLinkageAttr(
          Target, Name->getString().str(), TypeKind));
      break;
    }

    case spv::internal::DecorationHostAccessINTEL:
    case DecorationHostAccessINTEL: {
      checkIsGlobalVar(Target, DecoKind);

      ErrLog.checkError(NumOperands == 3, SPIRVEC_InvalidLlvmModule,
                        "HostAccessINTEL requires exactly 2 extra operands "
                        "after the decoration kind number");
      auto *AccessMode =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(
          AccessMode, SPIRVEC_InvalidLlvmModule,
          "HostAccessINTEL requires first extra operand to be an int");

      HostAccessQualifier Q =
          static_cast<HostAccessQualifier>(AccessMode->getZExtValue());
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(2));
      ErrLog.checkError(
          Name, SPIRVEC_InvalidLlvmModule,
          "HostAccessINTEL requires second extra operand to be a string");

      if (DecoKind == DecorationHostAccessINTEL)
        Target->addDecorate(new SPIRVDecorateHostAccessINTEL(
            Target, Q, Name->getString().str()));
      else
        Target->addDecorate(new SPIRVDecorateHostAccessINTELLegacy(
            Target, Q, Name->getString().str()));
      break;
    }

    case spv::internal::DecorationInitModeINTEL:
    case DecorationInitModeINTEL: {
      checkIsGlobalVar(Target, DecoKind);
      ErrLog.checkError(static_cast<SPIRVVariable *>(Target)->getInitializer(),
                        SPIRVEC_InvalidLlvmModule,
                        "InitModeINTEL only be applied to a global (module "
                        "scope) variable which has an Initializer operand");

      ErrLog.checkError(NumOperands == 2, SPIRVEC_InvalidLlvmModule,
                        "InitModeINTEL requires exactly 1 extra operand");
      auto *Trigger = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(Trigger, SPIRVEC_InvalidLlvmModule,
                        "InitModeINTEL requires extra operand to be an int");

      InitializationModeQualifier Q =
          static_cast<InitializationModeQualifier>(Trigger->getZExtValue());

      if (DecoKind == DecorationInitModeINTEL)
        Target->addDecorate(new SPIRVDecorateInitModeINTEL(Target, Q));
      else
        Target->addDecorate(new SPIRVDecorateInitModeINTELLegacy(Target, Q));

      break;
    }
    case spv::internal::DecorationImplementInCSRINTEL: {
      checkIsGlobalVar(Target, DecoKind);
      ErrLog.checkError(NumOperands == 2, SPIRVEC_InvalidLlvmModule,
                        "ImplementInCSRINTEL requires exactly 1 extra operand");
      auto *Value = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(
          Value, SPIRVEC_InvalidLlvmModule,
          "ImplementInCSRINTEL requires extra operand to be an integer");

      Target->addDecorate(
          new SPIRVDecorateImplementInCSRINTEL(Target, Value->getZExtValue()));
      break;
    }
    case DecorationImplementInRegisterMapINTEL: {
      checkIsGlobalVar(Target, DecoKind);
      ErrLog.checkError(
          NumOperands == 2, SPIRVEC_InvalidLlvmModule,
          "ImplementInRegisterMapINTEL requires exactly 1 extra operand");
      auto *Value = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(Value, SPIRVEC_InvalidLlvmModule,
                        "ImplementInRegisterMapINTEL requires extra operand to "
                        "be an integer");

      Target->addDecorate(new SPIRVDecorateImplementInRegisterMapINTEL(
          Target, Value->getZExtValue()));

      break;
    }

    case DecorationCacheControlLoadINTEL: {
      ErrLog.checkError(
          NumOperands == 3, SPIRVEC_InvalidLlvmModule,
          "CacheControlLoadINTEL requires exactly 2 extra operands");
      auto *CacheLevel =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      auto *CacheControl =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      ErrLog.checkError(CacheLevel, SPIRVEC_InvalidLlvmModule,
                        "CacheControlLoadINTEL cache level operand is required "
                        "to be an integer");
      ErrLog.checkError(CacheControl, SPIRVEC_InvalidLlvmModule,
                        "CacheControlLoadINTEL cache control operand is "
                        "required to be an integer");

      Target->addDecorate(new SPIRVDecorateCacheControlLoadINTEL(
          Target, CacheLevel->getZExtValue(),
          static_cast<LoadCacheControl>(CacheControl->getZExtValue())));
      break;
    }
    case DecorationCacheControlStoreINTEL: {
      ErrLog.checkError(
          NumOperands == 3, SPIRVEC_InvalidLlvmModule,
          "CacheControlStoreINTEL requires exactly 2 extra operands");
      auto *CacheLevel =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      auto *CacheControl =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      ErrLog.checkError(CacheLevel, SPIRVEC_InvalidLlvmModule,
                        "CacheControlStoreINTEL cache level operand is "
                        "required to be an integer");
      ErrLog.checkError(CacheControl, SPIRVEC_InvalidLlvmModule,
                        "CacheControlStoreINTEL cache control operand is "
                        "required to be an integer");

      Target->addDecorate(new SPIRVDecorateCacheControlStoreINTEL(
          Target, CacheLevel->getZExtValue(),
          static_cast<StoreCacheControl>(CacheControl->getZExtValue())));
      break;
    }
    default: {
      if (NumOperands == 1) {
        Target->addDecorate(new SPIRVDecorate(DecoKind, Target));
        break;
      }

      auto *DecoValEO1 =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      ErrLog.checkError(
          DecoValEO1, SPIRVEC_InvalidLlvmModule,
          "First extra operand in default decoration case must be integer.");
      if (NumOperands == 2) {
        Target->addDecorate(
            new SPIRVDecorate(DecoKind, Target, DecoValEO1->getZExtValue()));
        break;
      }

      auto *DecoValEO2 =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      ErrLog.checkError(
          DecoValEO2, SPIRVEC_InvalidLlvmModule,
          "Second extra operand in default decoration case must be integer.");

      ErrLog.checkError(NumOperands == 3, SPIRVEC_InvalidLlvmModule,
                        "At most 2 extra operands expected.");
      Target->addDecorate(new SPIRVDecorate(DecoKind, Target,
                                            DecoValEO1->getZExtValue(),
                                            DecoValEO2->getZExtValue()));
    }
    }
  }
}

#undef ONE_STRING_DECORATION_CASE
#undef ONE_INT_DECORATION_CASE
#undef TWO_INT_DECORATION_CASE

bool LLVMToSPIRVBase::transDecoration(Value *V, SPIRVValue *BV) {
  if (!transAlign(V, BV))
    return false;
  if ((isa<AtomicCmpXchgInst>(V) && cast<AtomicCmpXchgInst>(V)->isVolatile()) ||
      (isa<AtomicRMWInst>(V) && cast<AtomicRMWInst>(V)->isVolatile()))
    BV->setVolatile(true);

  if (auto *BVO = dyn_cast_or_null<OverflowingBinaryOperator>(V)) {
    if (BVO->hasNoSignedWrap()) {
      BV->setNoIntegerDecorationWrap<DecorationNoSignedWrap>(true);
    }
    if (BVO->hasNoUnsignedWrap()) {
      BV->setNoIntegerDecorationWrap<DecorationNoUnsignedWrap>(true);
    }
  }

  if (auto *BVF = dyn_cast_or_null<FPMathOperator>(V)) {
    auto Opcode = BVF->getOpcode();
    if (Opcode == Instruction::FAdd || Opcode == Instruction::FSub ||
        Opcode == Instruction::FMul || Opcode == Instruction::FDiv ||
        Opcode == Instruction::FRem ||
        ((Opcode == Instruction::FNeg || Opcode == Instruction::FCmp) &&
         BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_6))) {
      FastMathFlags FMF = BVF->getFastMathFlags();
      SPIRVWord M{0};
      if (FMF.isFast())
        M |= FPFastMathModeFastMask;
      else {
        if (FMF.noNaNs())
          M |= FPFastMathModeNotNaNMask;
        if (FMF.noInfs())
          M |= FPFastMathModeNotInfMask;
        if (FMF.noSignedZeros())
          M |= FPFastMathModeNSZMask;
        if (FMF.allowReciprocal())
          M |= FPFastMathModeAllowRecipMask;
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_fp_fast_math_mode)) {
          if (FMF.allowContract()) {
            M |= FPFastMathModeAllowContractFastINTELMask;
            BM->addCapability(CapabilityFPFastMathModeINTEL);
          }
          if (FMF.allowReassoc()) {
            M |= FPFastMathModeAllowReassocINTELMask;
            BM->addCapability(CapabilityFPFastMathModeINTEL);
          }
        }
      }
      if (M != 0)
        BV->setFPFastMathMode(M);
    }
  }
  if (Instruction *Inst = dyn_cast<Instruction>(V)) {
    if (shouldTryToAddMemAliasingDecoration(Inst))
      transMemAliasingINTELDecorations(Inst, BV);
    if (auto *IDecoMD = Inst->getMetadata(SPIRV_MD_DECORATIONS))
      transMetadataDecorations(IDecoMD, BV);
    if (BV->isInst())
      addFPBuiltinDecoration(BM, Inst, static_cast<SPIRVInstruction *>(BV));
  }

  if (auto *CI = dyn_cast<CallInst>(V)) {
    auto OC = BV->getOpCode();
    if (OC == OpSpecConstantTrue || OC == OpSpecConstantFalse ||
        OC == OpSpecConstant) {
      auto SpecId = cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
      BV->addDecorate(DecorationSpecId, SpecId);
    }
    if (OC == OpFunctionPointerCallINTEL)
      addFuncPointerCallArgumentAttributes(CI, BV);
  }

  if (auto *GV = dyn_cast<GlobalVariable>(V))
    if (auto *GVDecoMD = GV->getMetadata(SPIRV_MD_DECORATIONS))
      transMetadataDecorations(GVDecoMD, BV);

  return true;
}

bool LLVMToSPIRVBase::transAlign(Value *V, SPIRVValue *BV) {
  if (auto *AL = dyn_cast<AllocaInst>(V)) {
    BM->setAlignment(BV, AL->getAlign().value());
    return true;
  }
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    BM->setAlignment(BV, GV->getAlignment());
    return true;
  }
  return true;
}

// Apply aliasing decorations to instructions annotated with aliasing metadata.
// Do it for any instruction but loads and stores.
void LLVMToSPIRVBase::transMemAliasingINTELDecorations(Instruction *Inst,
                                                       SPIRVValue *BV) {
  if (!BM->isAllowedToUseExtension(
         ExtensionID::SPV_INTEL_memory_access_aliasing))
    return;
  if (MDNode *AliasingListMD =
          Inst->getMetadata(LLVMContext::MD_alias_scope)) {
    auto *MemAliasList =
        addMemAliasingINTELInstructions(BM, AliasingListMD);
    if (!MemAliasList)
      return;
    BV->addDecorate(new SPIRVDecorateId(DecorationAliasScopeINTEL, BV,
                                        MemAliasList->getId()));
  }
  if (MDNode *AliasingListMD = Inst->getMetadata(LLVMContext::MD_noalias)) {
    auto *MemAliasList =
        addMemAliasingINTELInstructions(BM, AliasingListMD);
    if (!MemAliasList)
      return;
    BV->addDecorate(
        new SPIRVDecorateId(DecorationNoAliasINTEL, BV, MemAliasList->getId()));
  }
}

/// Do this after source language is set.
bool LLVMToSPIRVBase::transBuiltinSet() {
  SPIRVId EISId;
  if (!BM->importBuiltinSet("OpenCL.std", &EISId))
    return false;
  if (SPIRVMDWalker(*M).getNamedMD("llvm.dbg.cu")) {
    if (!BM->importBuiltinSet(
            SPIRVBuiltinSetNameMap::map(BM->getDebugInfoEIS()), &EISId))
      return false;
  }
  if (BM->preserveAuxData()) {
    if (!BM->importBuiltinSet(
            SPIRVBuiltinSetNameMap::map(SPIRVEIS_NonSemantic_AuxData), &EISId))
      return false;
  }
  return true;
}

/// Translate sampler* spcv.cast(i32 arg) or
/// sampler* __translate_sampler_initializer(i32 arg)
/// Three cases are possible:
///   arg = ConstantInt x -> SPIRVConstantSampler
///   arg = i32 argument -> transValue(arg)
///   arg = load from sampler -> look through load
SPIRVValue *LLVMToSPIRVBase::oclTransSpvcCastSampler(CallInst *CI,
                                                     SPIRVBasicBlock *BB) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  auto *Arg = CI->getArgOperand(0);
  auto *TransRT = transType(getSPIRVType(OpTypeSampler));

  auto GetSamplerConstant = [&](uint64_t SamplerValue) {
    auto AddrMode = (SamplerValue & 0xE) >> 1;
    auto Param = SamplerValue & 0x1;
    auto Filter = SamplerValue ? ((SamplerValue & 0x30) >> 4) - 1 : 0;
    auto *BV = BM->addSamplerConstant(TransRT, AddrMode, Param, Filter);
    return BV;
  };

  if (auto *Const = dyn_cast<ConstantInt>(Arg)) {
    // Sampler is declared as a kernel scope constant
    return GetSamplerConstant(Const->getZExtValue());
  } else if (auto *Load = dyn_cast<LoadInst>(Arg)) {
    // If value of the sampler is loaded from a global constant, use its
    // initializer for initialization of the sampler.
    auto *Op = Load->getPointerOperand();
    assert(isa<GlobalVariable>(Op) && "Unknown sampler pattern!");
    auto *GV = cast<GlobalVariable>(Op);
    assert(GV->isConstant() ||
           GV->getType()->getPointerAddressSpace() == SPIRAS_Constant);
    auto *Initializer = GV->getInitializer();
    assert(isa<ConstantInt>(Initializer) && "sampler not constant int?");
    return GetSamplerConstant(cast<ConstantInt>(Initializer)->getZExtValue());
  }
  // Sampler is a function argument
  auto *BV = transValue(Arg, BB);
  assert(BV && BV->getType() == TransRT);
  return BV;
}

using DecorationsInfoVec =
    std::vector<std::pair<Decoration, std::vector<std::string>>>;

struct AnnotationDecorations {
  DecorationsInfoVec MemoryAttributesVec;
  DecorationsInfoVec MemoryAccessesVec;
  DecorationsInfoVec BufferLocationVec;
  DecorationsInfoVec LatencyControlVec;
  DecorationsInfoVec CacheControlVec;

  bool empty() {
    return (MemoryAttributesVec.empty() && MemoryAccessesVec.empty() &&
            BufferLocationVec.empty() && LatencyControlVec.empty() &&
            CacheControlVec.empty());
  }
};

struct IntelLSUControlsInfo {
  void setWithBitMask(unsigned ParamsBitMask) {
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::BurstCoalesce)
      BurstCoalesce = true;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::CacheSizeFlag)
      CacheSizeInfo = 0;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::DontStaticallyCoalesce)
      DontStaticallyCoalesce = true;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::PrefetchFlag)
      PrefetchInfo = 0;
  }

  DecorationsInfoVec getDecorationsFromCurrentState() {
    DecorationsInfoVec ResultVec;
    // Simple flags
    if (BurstCoalesce)
      ResultVec.emplace_back(DecorationBurstCoalesceINTEL,
                             std::vector<std::string>());
    if (DontStaticallyCoalesce)
      ResultVec.emplace_back(DecorationDontStaticallyCoalesceINTEL,
                             std::vector<std::string>());
    // Conditional values
    if (CacheSizeInfo.has_value()) {
      ResultVec.emplace_back(
          DecorationCacheSizeINTEL,
          std::vector<std::string>{std::to_string(CacheSizeInfo.value())});
    }
    if (PrefetchInfo.has_value()) {
      ResultVec.emplace_back(
          DecorationPrefetchINTEL,
          std::vector<std::string>{std::to_string(PrefetchInfo.value())});
    }
    return ResultVec;
  }

  bool BurstCoalesce = false;
  std::optional<unsigned> CacheSizeInfo;
  bool DontStaticallyCoalesce = false;
  std::optional<unsigned> PrefetchInfo;
};

// Handle optional var/ptr/global annotation parameter. It can be for example
// { %struct.S, i8*, void ()* } { %struct.S undef, i8* null,
//                                void ()* @_Z4blahv }
// Now we will just handle integer constants (wrapped in a constant
// struct, that is being bitcasted to i8*), converting them to string.
// TODO: remove this workaround when/if an extension spec that allows or adds
// variadic-arguments UserSemantic decoration
void processOptionalAnnotationInfo(Constant *Const,
                                   std::string &AnnotationString) {
  if (!Const->getNumOperands())
    return;
  if (auto *CStruct = dyn_cast<ConstantStruct>(Const->getOperand(0))) {
    uint32_t NumOperands = CStruct->getNumOperands();
    if (!NumOperands)
      return;
    if (auto *CInt = dyn_cast<ConstantInt>(CStruct->getOperand(0))) {
      AnnotationString += ": ";
      // For boolean, emit 0/1 for ease of readability.
      if (CInt->getType()->getIntegerBitWidth() == 1)
        AnnotationString += std::to_string(CInt->getZExtValue());
      else
        AnnotationString += std::to_string(CInt->getSExtValue());
    }
    for (uint32_t I = 1; I != NumOperands; ++I) {
      if (auto *CInt = dyn_cast<ConstantInt>(CStruct->getOperand(I))) {
        AnnotationString += ", ";
        AnnotationString += std::to_string(CInt->getSExtValue());
      }
    }
  } else if (auto *ZeroStruct =
                 dyn_cast<ConstantAggregateZero>(Const->getOperand(0))) {
    // It covers case when all elements of struct are 0 and they become
    // zeroinitializer. It represents like: { i32 i32 ... } zeroinitializer
    uint32_t NumOperands = ZeroStruct->getType()->getStructNumElements();
    AnnotationString += ": ";
    AnnotationString += "0";
    for (uint32_t I = 1; I != NumOperands; ++I) {
      AnnotationString += ", ";
      AnnotationString += "0";
    }
  }
}

// Process main var/ptr/global annotation string with the attached optional
// integer parameters
void processAnnotationString(IntrinsicInst *II, std::string &AnnotationString) {
  auto *StrVal = II->getArgOperand(1);
  auto *StrValTy = StrVal->getType();
  if (StrValTy->isPointerTy()) {
    StringRef StrRef;
    if (getConstantStringInfo(dyn_cast<Constant>(StrVal), StrRef))
      AnnotationString += StrRef.str();
    if (auto *C = dyn_cast_or_null<Constant>(II->getArgOperand(4)))
      processOptionalAnnotationInfo(C, AnnotationString);
    return;
  }
  if (auto *GEP = dyn_cast<GetElementPtrInst>(StrVal)) {
    if (auto *C = dyn_cast<Constant>(GEP->getOperand(0))) {
      StringRef StrRef;
      if (getConstantStringInfo(C, StrRef))
        AnnotationString += StrRef.str();
    }
  }
  if (auto *Cast = dyn_cast<BitCastInst>(II->getArgOperand(4)))
    if (auto *C = dyn_cast_or_null<Constant>(Cast->getOperand(0)))
      processOptionalAnnotationInfo(C, AnnotationString);
}

// Try to parse the annotation decoration values in a string. These values must
// be separated by a "," and must be either a word (including numbers) or a
// quotation mark enclosed string.
static bool tryParseAnnotationDecoValues(StringRef ValueStr,
                                         std::vector<std::string> &ParsedArgs) {
  unsigned ValueStart = 0;
  bool IsParsingStringLiteral = false;
  for (unsigned I = 0; I < ValueStr.size(); ++I) {
    const char CurrentC = ValueStr[I];
    if (IsParsingStringLiteral) {
      if (CurrentC == '"') {
        // We have reached the end of a string literal and have the arg string
        // between this character and the start of the string literal.
        IsParsingStringLiteral = false;
        ParsedArgs.push_back(ValueStr.substr(ValueStart, I - ValueStart).str());
        // End of a string literal must either be at the end of the values or
        // right before a comma.
        if (I + 1 != ValueStr.size() && ValueStr[I + 1] != ',')
          return false;
        // Skip the , delimiter and go directly to the start of next value.
        ValueStart = (++I) + 1;
        continue;
      }
    }
    if (CurrentC == ',') {
      // Since we are not currently in a string literal, comma denotes a
      // separation of decoration arguments and we can copy the substring we are
      // currently parsing.
      ParsedArgs.push_back(ValueStr.substr(ValueStart, I - ValueStart).str());
      ValueStart = I + 1;
      continue;
    }
    if (CurrentC == '"') {
      // We are entering a string literal. This must be either at the beginning
      // of the values or right after a comma.
      if (I != 0 && ValueStr[I - 1] != ',')
        return false;
      IsParsingStringLiteral = true;
      ValueStart = I + 1;
      continue;
    }
    // Any other character will be consumed as part of the argument.
  }
  // If we were still parsing a decoration argument when reaching the end of the
  // parsed string, we must be at the end of the argument.
  if (ValueStart < ValueStr.size())
    ParsedArgs.push_back(
        ValueStr.substr(ValueStart, ValueStr.size() - ValueStart).str());

  // At the end, the arguments parsed are valid if we were not parsing a string
  // literal with no end.
  return !IsParsingStringLiteral;
}

AnnotationDecorations tryParseAnnotationString(SPIRVModule *BM,
                                               StringRef AnnotatedCode) {
  AnnotationDecorations Decorates;
  // Annotation string decorations are separated into {word} OR
  // {word:value,value,...} blocks, where value is either a word (including
  // numbers) or a quotation mark enclosed string.
  std::regex DecorationRegex("\\{\\w([\\w:,-]|\"[^\"]*\")*\\}");
  using RegexIterT = std::regex_iterator<StringRef::const_iterator>;
  RegexIterT DecorationsIt(AnnotatedCode.begin(), AnnotatedCode.end(),
                           DecorationRegex);
  RegexIterT DecorationsEnd;

  // If we didn't find any annotations that are separated as described above,
  // then add a UserSemantic decoration
  if (DecorationsIt == DecorationsEnd) {
    Decorates.MemoryAttributesVec.emplace_back(
        DecorationUserSemantic, std::vector<std::string>{AnnotatedCode.str()});
    return Decorates;
  }

  const bool AllowFPGAMemAccesses =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_memory_accesses);
  const bool AllowFPGAMemAttr = BM->isAllowedToUseExtension(
      ExtensionID::SPV_INTEL_fpga_memory_attributes);
  const bool AllowFPGABufLoc =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_buffer_location);
  const bool AllowFPGALatencyControl =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_latency_control);
  const bool AllowCacheControls =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_cache_controls);

  bool ValidDecorationFound = false;
  DecorationsInfoVec DecorationsVec;
  IntelLSUControlsInfo LSUControls;
  for (; DecorationsIt != DecorationsEnd; ++DecorationsIt) {
    // Drop the braces surrounding the actual decoration
    const StringRef AnnotatedDecoration = AnnotatedCode.substr(
        DecorationsIt->position() + 1, DecorationsIt->length() - 2);

    std::pair<StringRef, StringRef> Split = AnnotatedDecoration.split(':');
    StringRef Name = Split.first, ValueStr = Split.second;
    SPIRVDBG(spvdbgs() << "[tryParseAnnotationString]: AnnotationString: "
                       << Name.str() << "\n");

    unsigned DecorationKind = 0;
    if (!Name.getAsInteger(10, DecorationKind)) {
      // If the name is a number it represents the decoration by its kind.
      std::vector<std::string> DecValues;
      if (tryParseAnnotationDecoValues(ValueStr, DecValues)) {
        ValidDecorationFound = true;

        if (AllowFPGABufLoc &&
            DecorationKind == DecorationBufferLocationINTEL) {
          Decorates.BufferLocationVec.emplace_back(
              static_cast<Decoration>(DecorationKind), std::move(DecValues));
        } else if (AllowFPGALatencyControl &&
                   (DecorationKind == DecorationLatencyControlLabelINTEL ||
                    DecorationKind ==
                        DecorationLatencyControlConstraintINTEL)) {
          Decorates.LatencyControlVec.emplace_back(
              static_cast<Decoration>(DecorationKind), std::move(DecValues));
        } else if (AllowCacheControls &&
                   DecorationKind == DecorationCacheControlLoadINTEL) {
          Decorates.CacheControlVec.emplace_back(
              static_cast<Decoration>(DecorationKind), std::move(DecValues));
        } else if (DecorationKind == DecorationMemoryINTEL) {
          // SPIRV doesn't allow the same Decoration to be applied multiple
          // times on a single SPIRVEntry, unless explicitly allowed by the
          // language spec. Filter out the less specific MemoryINTEL
          // decorations, if applied multiple times
          auto CanFilterOut = [](auto &Val) {
            if (!Val.second.empty())
              return (Val.second[0] == "DEFAULT");
            return false;
          };
          auto It = std::find_if(DecorationsVec.begin(), DecorationsVec.end(),
                                 CanFilterOut);

          if (It != DecorationsVec.end()) {
            // replace the less specific decoration
            *It = {static_cast<Decoration>(DecorationKind),
                   std::move(DecValues)};
          } else {
            // add new decoration
            DecorationsVec.emplace_back(static_cast<Decoration>(DecorationKind),
                                        std::move(DecValues));
          }
        } else {
          DecorationsVec.emplace_back(static_cast<Decoration>(DecorationKind),
                                      std::move(DecValues));
        }
      }
      continue;
    }

    if (AllowFPGAMemAccesses) {
      if (Name == "params") {
        ValidDecorationFound = true;
        unsigned ParamsBitMask = 0;
        bool Failure = ValueStr.getAsInteger(10, ParamsBitMask);
        assert(!Failure && "Non-integer LSU controls value");
        (void)Failure;
        LSUControls.setWithBitMask(ParamsBitMask);
      } else if (Name == "cache-size") {
        ValidDecorationFound = true;
        if (!LSUControls.CacheSizeInfo.has_value())
          continue;
        unsigned CacheSizeValue = 0;
        bool Failure = ValueStr.getAsInteger(10, CacheSizeValue);
        assert(!Failure && "Non-integer cache size value");
        (void)Failure;
        LSUControls.CacheSizeInfo = CacheSizeValue;
      } // TODO: Support LSU prefetch size, which currently defaults to 0
    }
    if (AllowFPGAMemAttr) {
      std::vector<std::string> DecValues;
      Decoration Dec;
      if (Name == "pump") {
        ValidDecorationFound = true;
        Dec = llvm::StringSwitch<Decoration>(ValueStr)
                  .Case("1", DecorationSinglepumpINTEL)
                  .Case("2", DecorationDoublepumpINTEL);
      } else if (Name == "register") {
        ValidDecorationFound = true;
        Dec = DecorationRegisterINTEL;
      } else if (Name == "simple_dual_port") {
        ValidDecorationFound = true;
        Dec = DecorationSimpleDualPortINTEL;
      } else {
        Dec = llvm::StringSwitch<Decoration>(Name)
                  .Case("memory", DecorationMemoryINTEL)
                  .Case("numbanks", DecorationNumbanksINTEL)
                  .Case("bankwidth", DecorationBankwidthINTEL)
                  .Case("private_copies", DecorationMaxPrivateCopiesINTEL)
                  .Case("max_replicates", DecorationMaxReplicatesINTEL)
                  .Case("bank_bits", DecorationBankBitsINTEL)
                  .Case("merge", DecorationMergeINTEL)
                  .Case("force_pow2_depth", DecorationForcePow2DepthINTEL)
                  .Case("stride_size", DecorationStridesizeINTEL)
                  .Case("word_size", DecorationWordsizeINTEL)
                  .Case("true_dual_port", DecorationTrueDualPortINTEL)
                  .Default(DecorationUserSemantic);
        if (Dec == DecorationUserSemantic)
          // Restore the braces to translate the whole input string
          DecValues =
              std::vector<std::string>({'{' + AnnotatedDecoration.str() + '}'});
        else if (Dec == DecorationMergeINTEL) {
          ValidDecorationFound = true;
          std::pair<StringRef, StringRef> MergeValues = ValueStr.split(':');
          DecValues = std::vector<std::string>(
              {MergeValues.first.str(), MergeValues.second.str()});
        } else if (Dec == DecorationBankBitsINTEL) {
          ValidDecorationFound = true;
          SmallVector<StringRef, 4> BitsStrs;
          ValueStr.split(BitsStrs, ',');
          DecValues.reserve(BitsStrs.size());
          for (const StringRef &BitsStr : BitsStrs)
            DecValues.push_back(BitsStr.str());
        } else {
          ValidDecorationFound = true;
          DecValues = std::vector<std::string>({ValueStr.str()});
        }
      }
      DecorationsVec.emplace_back(Dec, std::move(DecValues));
    }
  }

  // Even if there is an annotation string that is split in blocks like Intel
  // FPGA annotation, it's not necessarily an FPGA annotation. Translate the
  // whole string as UserSemantic decoration in this case.
  if (ValidDecorationFound)
    Decorates.MemoryAttributesVec = DecorationsVec;
  else
    Decorates.MemoryAttributesVec.emplace_back(
        DecorationUserSemantic,
        std::vector<std::string>({AnnotatedCode.str()}));
  Decorates.MemoryAccessesVec = LSUControls.getDecorationsFromCurrentState();

  return Decorates;
}

std::vector<SPIRVWord>
getLiteralsFromStrings(const std::vector<std::string> &Strings) {
  std::vector<SPIRVWord> Literals(Strings.size());
  for (size_t J = 0; J < Strings.size(); ++J)
    if (StringRef(Strings[J]).getAsInteger(10, Literals[J]))
      return {};
  return Literals;
}

void addAnnotationDecorations(SPIRVEntry *E, DecorationsInfoVec &Decorations) {
  SPIRVModule *M = E->getModule();
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, try to skip it
    if (E->hasDecorate(I.first, /*Index=*/0, /*Result=*/nullptr))
      // Allow multiple UserSemantic Decorations
      if (I.first != DecorationUserSemantic)
        continue;

    switch (static_cast<size_t>(I.first)) {
    case DecorationUserSemantic:
      M->getErrorLog().checkError(I.second.size() == 1,
                                  SPIRVEC_InvalidLlvmModule,
                                  "UserSemantic requires a single argument.");
      E->addDecorate(new SPIRVDecorateUserSemanticAttr(E, I.second[0]));
      break;
    case DecorationMemoryINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        M->getErrorLog().checkError(I.second.size() == 1,
                                    SPIRVEC_InvalidLlvmModule,
                                    "MemoryINTEL requires a single argument.");
        E->addDecorate(new SPIRVDecorateMemoryINTELAttr(E, I.second[0]));
      }
    } break;
    case DecorationMergeINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        M->getErrorLog().checkError(I.second.size() == 2,
                                    SPIRVEC_InvalidLlvmModule,
                                    "MergeINTEL requires two arguments.");
        // First argument is the name and the second argument is the direction.
        E->addDecorate(
            new SPIRVDecorateMergeINTELAttr(E, I.second[0], I.second[1]));
      }
    } break;
    case DecorationBankBitsINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        M->getErrorLog().checkError(
            I.second.size() > 0, SPIRVEC_InvalidLlvmModule,
            "BankBitsINTEL requires at least one argument.");
        E->addDecorate(new SPIRVDecorateBankBitsINTELAttr(
            E, getLiteralsFromStrings(I.second)));
      }
    } break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL:
    case DecorationTrueDualPortINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        M->getErrorLog().checkError(I.second.empty(), SPIRVEC_InvalidLlvmModule,
                                    "Decoration takes no arguments.");
        E->addDecorate(I.first);
      }
    } break;
    case DecorationBurstCoalesceINTEL:
    case DecorationDontStaticallyCoalesceINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        M->getErrorLog().checkError(I.second.empty(), SPIRVEC_InvalidLlvmModule,
                                    "Decoration takes no arguments.");
        E->addDecorate(I.first);
      }
    } break;
    case DecorationNumbanksINTEL:
    case DecorationBankwidthINTEL:
    case DecorationMaxPrivateCopiesINTEL:
    case DecorationMaxReplicatesINTEL:
    case DecorationForcePow2DepthINTEL:
    case DecorationStridesizeINTEL:
    case DecorationWordsizeINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        M->getErrorLog().checkError(I.second.size() == 1,
                                    SPIRVEC_InvalidLlvmModule,
                                    "Decoration requires a single argument.");
        SPIRVWord Result = 0;
        StringRef(I.second[0]).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    case DecorationCacheSizeINTEL:
    case DecorationPrefetchINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        M->getErrorLog().checkError(I.second.size() == 1,
                                    SPIRVEC_InvalidLlvmModule,
                                    "Decoration requires a single argument.");
        SPIRVWord Result = 0;
        StringRef(I.second[0]).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    case DecorationBufferLocationINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_buffer_location)) {
        M->getErrorLog().checkError(I.second.size() == 1,
                                    SPIRVEC_InvalidLlvmModule,
                                    "Decoration requires a single argument.");
        SPIRVWord Result = 0;
        StringRef(I.second[0]).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    case DecorationLatencyControlLabelINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_latency_control)) {
        M->getErrorLog().checkError(
            I.second.size() == 1, SPIRVEC_InvalidLlvmModule,
            "LatencyControlLabelINTEL requires exactly 1 extra operand");
        SPIRVWord Label = 0;
        StringRef(I.second[0]).getAsInteger(10, Label);
        E->addDecorate(
            new SPIRVDecorate(DecorationLatencyControlLabelINTEL, E, Label));
      }
      break;
    }
    case DecorationLatencyControlConstraintINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_latency_control)) {
        M->getErrorLog().checkError(
            I.second.size() == 3, SPIRVEC_InvalidLlvmModule,
            "LatencyControlConstraintINTEL requires exactly 3 extra operands");
        auto Literals = getLiteralsFromStrings(I.second);
        E->addDecorate(
            new SPIRVDecorate(DecorationLatencyControlConstraintINTEL, E,
                              Literals[0], Literals[1], Literals[2]));
      }
      break;
    }
    case DecorationCacheControlLoadINTEL: {
      if (M->isAllowedToUseExtension(ExtensionID::SPV_INTEL_cache_controls)) {
        M->getErrorLog().checkError(
            I.second.size() == 2, SPIRVEC_InvalidLlvmModule,
            "CacheControlLoadINTEL requires exactly 2 extra operands");
        SPIRVWord CacheLevel = 0;
        SPIRVWord CacheControl = 0;
        StringRef(I.second[0]).getAsInteger(10, CacheLevel);
        StringRef(I.second[1]).getAsInteger(10, CacheControl);
        E->addDecorate(new SPIRVDecorateCacheControlLoadINTEL(
            E, CacheLevel, static_cast<LoadCacheControl>(CacheControl)));
      }
    }

    default:
      // Other decorations are either not supported by the translator or
      // handled in other places.
      break;
    }
  }
}

void addAnnotationDecorationsForStructMember(SPIRVEntry *E,
                                             SPIRVWord MemberNumber,
                                             DecorationsInfoVec &Decorations) {
  SPIRVModule *M = E->getModule();
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, skip it
    if (E->hasMemberDecorate(I.first, /*Index=*/0, MemberNumber,
                             /*Result=*/nullptr))
      // Allow multiple UserSemantic Decorations
      if (I.first != DecorationUserSemantic)
        continue;

    switch (I.first) {
    case DecorationUserSemantic:
      M->getErrorLog().checkError(I.second.size() == 1,
                                  SPIRVEC_InvalidLlvmModule,
                                  "UserSemantic requires a single argument.");
      E->addMemberDecorate(new SPIRVMemberDecorateUserSemanticAttr(
          E, MemberNumber, I.second[0]));
      break;
    case DecorationMemoryINTEL:
      M->getErrorLog().checkError(I.second.size() == 1,
                                  SPIRVEC_InvalidLlvmModule,
                                  "MemoryINTEL requires a single argument.");
      E->addMemberDecorate(
          new SPIRVMemberDecorateMemoryINTELAttr(E, MemberNumber, I.second[0]));
      break;
    case DecorationMergeINTEL: {
      M->getErrorLog().checkError(I.second.size() == 2,
                                  SPIRVEC_InvalidLlvmModule,
                                  "MergeINTEL requires two arguments.");
      // First argument is the name, the other is the direction.
      E->addMemberDecorate(new SPIRVMemberDecorateMergeINTELAttr(
          E, MemberNumber, I.second[0], I.second[1]));
    } break;
    case DecorationBankBitsINTEL:
      M->getErrorLog().checkError(
          I.second.size() > 0, SPIRVEC_InvalidLlvmModule,
          "BankBitsINTEL requires at least one argument.");
      E->addMemberDecorate(new SPIRVMemberDecorateBankBitsINTELAttr(
          E, MemberNumber, getLiteralsFromStrings(I.second)));
      break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL:
    case DecorationTrueDualPortINTEL:
      M->getErrorLog().checkError(I.second.empty(), SPIRVEC_InvalidLlvmModule,
                                  "Member decoration takes no arguments.");
      E->addMemberDecorate(MemberNumber, I.first);
      break;
    // The rest of IntelFPGA decorations:
    // DecorationNumbanksINTEL
    // DecorationBankwidthINTEL
    // DecorationMaxPrivateCopiesINTEL
    // DecorationMaxReplicatesINTEL
    // DecorationForcePow2DepthINTEL
    // DecorarionStridesizeINTEL
    // DecorationWordsizeINTEL
    default:
      M->getErrorLog().checkError(
          I.second.size() == 1, SPIRVEC_InvalidLlvmModule,
          "Member decoration requires a single argument.");
      SPIRVWord Result = 0;
      StringRef(I.second[0]).getAsInteger(10, Result);
      E->addMemberDecorate(MemberNumber, I.first, Result);
      break;
    }
  }
}

bool LLVMToSPIRVBase::isKnownIntrinsic(Intrinsic::ID Id) {
  // Known intrinsics usually do not need translation of their declaration
  switch (Id) {
  case Intrinsic::abs:
  case Intrinsic::assume:
  case Intrinsic::bitreverse:
  case Intrinsic::ceil:
  case Intrinsic::copysign:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::fma:
  case Intrinsic::frexp:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::maximum:
  case Intrinsic::maxnum:
  case Intrinsic::smax:
  case Intrinsic::umax:
  case Intrinsic::minimum:
  case Intrinsic::minnum:
  case Intrinsic::smin:
  case Intrinsic::umin:
  case Intrinsic::nearbyint:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::sin:
  case Intrinsic::sqrt:
  case Intrinsic::trunc:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::expect:
  case Intrinsic::experimental_noalias_scope_decl:
  case Intrinsic::experimental_constrained_fadd:
  case Intrinsic::experimental_constrained_fsub:
  case Intrinsic::experimental_constrained_fmul:
  case Intrinsic::experimental_constrained_fdiv:
  case Intrinsic::experimental_constrained_frem:
  case Intrinsic::experimental_constrained_fma:
  case Intrinsic::experimental_constrained_fptoui:
  case Intrinsic::experimental_constrained_fptosi:
  case Intrinsic::experimental_constrained_uitofp:
  case Intrinsic::experimental_constrained_sitofp:
  case Intrinsic::experimental_constrained_fptrunc:
  case Intrinsic::experimental_constrained_fpext:
  case Intrinsic::experimental_constrained_fcmp:
  case Intrinsic::experimental_constrained_fcmps:
  case Intrinsic::experimental_constrained_fmuladd:
  case Intrinsic::fmuladd:
  case Intrinsic::memset:
  case Intrinsic::memcpy:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_value:
  case Intrinsic::annotation:
  case Intrinsic::var_annotation:
  case Intrinsic::ptr_annotation:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::dbg_label:
  case Intrinsic::trap:
  case Intrinsic::ubsantrap:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::arithmetic_fence:
  case Intrinsic::masked_gather:
  case Intrinsic::masked_scatter:
    return true;
  default:
    // Unknown intrinsics' declarations should always be translated
    return false;
  }
}

// Performs mapping of LLVM IR rounding mode to SPIR-V rounding mode
// Value *V is metadata <rounding mode> argument of
// llvm.experimental.constrained.* intrinsics
SPIRVInstruction *
LLVMToSPIRVBase::applyRoundingModeConstraint(Value *V, SPIRVInstruction *I) {
  StringRef RMode =
      cast<MDString>(cast<MetadataAsValue>(V)->getMetadata())->getString();
  if (RMode.ends_with("tonearest"))
    I->addFPRoundingMode(FPRoundingModeRTE);
  else if (RMode.ends_with("towardzero"))
    I->addFPRoundingMode(FPRoundingModeRTZ);
  else if (RMode.ends_with("upward"))
    I->addFPRoundingMode(FPRoundingModeRTP);
  else if (RMode.ends_with("downward"))
    I->addFPRoundingMode(FPRoundingModeRTN);
  return I;
}

static SPIRVWord getBuiltinIdForIntrinsic(Intrinsic::ID IID) {
  switch (IID) {
  // Note: In some cases the semantics of the OpenCL builtin are not identical
  //       to the semantics of the corresponding LLVM IR intrinsic. The LLVM
  //       intrinsics handled here assume the default floating point environment
  //       (no unmasked exceptions, round-to-nearest-ties-even rounding mode)
  //       and assume that the operations have no side effects (FP status flags
  //       aren't maintained), so the OpenCL builtin behavior should be
  //       acceptable.
  case Intrinsic::ceil:
    return OpenCLLIB::Ceil;
  case Intrinsic::copysign:
    return OpenCLLIB::Copysign;
  case Intrinsic::cos:
    return OpenCLLIB::Cos;
  case Intrinsic::exp:
    return OpenCLLIB::Exp;
  case Intrinsic::exp2:
    return OpenCLLIB::Exp2;
  case Intrinsic::fabs:
    return OpenCLLIB::Fabs;
  case Intrinsic::floor:
    return OpenCLLIB::Floor;
  case Intrinsic::fma:
    return OpenCLLIB::Fma;
  case Intrinsic::frexp:
    return OpenCLLIB::Frexp;
  case Intrinsic::log:
    return OpenCLLIB::Log;
  case Intrinsic::log10:
    return OpenCLLIB::Log10;
  case Intrinsic::log2:
    return OpenCLLIB::Log2;
  case Intrinsic::maximum:
    return OpenCLLIB::Fmax;
  case Intrinsic::maxnum:
    return OpenCLLIB::Fmax;
  case Intrinsic::minimum:
    return OpenCLLIB::Fmin;
  case Intrinsic::minnum:
    return OpenCLLIB::Fmin;
  case Intrinsic::nearbyint:
    return OpenCLLIB::Rint;
  case Intrinsic::pow:
    return OpenCLLIB::Pow;
  case Intrinsic::powi:
    return OpenCLLIB::Pown;
  case Intrinsic::rint:
    return OpenCLLIB::Rint;
  case Intrinsic::round:
    return OpenCLLIB::Round;
  case Intrinsic::roundeven:
    return OpenCLLIB::Rint;
  case Intrinsic::sin:
    return OpenCLLIB::Sin;
  case Intrinsic::sqrt:
    return OpenCLLIB::Sqrt;
  case Intrinsic::trunc:
    return OpenCLLIB::Trunc;
  default:
    assert(false && "Builtin ID requested for Unhandled intrinsic!");
    return 0;
  }
}

static SPIRVWord getNativeBuiltinIdForIntrinsic(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::cos:
    return OpenCLLIB::Native_cos;
  case Intrinsic::exp:
    return OpenCLLIB::Native_exp;
  case Intrinsic::exp2:
    return OpenCLLIB::Native_exp2;
  case Intrinsic::log:
    return OpenCLLIB::Native_log;
  case Intrinsic::log10:
    return OpenCLLIB::Native_log10;
  case Intrinsic::log2:
    return OpenCLLIB::Native_log2;
  case Intrinsic::sin:
    return OpenCLLIB::Native_sin;
  case Intrinsic::sqrt:
    return OpenCLLIB::Native_sqrt;
  default:
    return getBuiltinIdForIntrinsic(IID);
  }
}

static bool allowsApproxFunction(IntrinsicInst *II) {
  auto *Ty = II->getType();
  // OpenCL native_* built-ins only support single precision data type
  return II->hasApproxFunc() &&
         (Ty->isFloatTy() ||
          (Ty->isVectorTy() &&
           cast<VectorType>(Ty)->getElementType()->isFloatTy()));
}

namespace {
bool checkMemUser(User *User) {
  if (isa<LoadInst>(User) || isa<StoreInst>(User))
    return true;
  if (auto *III = dyn_cast<IntrinsicInst>(User)) {
    if (III->getIntrinsicID() == Intrinsic::memcpy)
      return true;
  }
  return false;
}
} // namespace

bool allowDecorateWithLatencyControlINTEL(IntrinsicInst *II) {
  for (auto *Inst : II->users()) {
    // if castInst, check Successors
    if (auto *Cast = dyn_cast<CastInst>(Inst)) {
      for (auto *Successor : Cast->users())
        if (checkMemUser(Successor))
          return true;
    } else {
      if (checkMemUser(Inst))
        return true;
    }
  }
  return false;
}

SPIRVValue *LLVMToSPIRVBase::transIntrinsicInst(IntrinsicInst *II,
                                                SPIRVBasicBlock *BB) {
  auto GetMemoryAccess =
      [](MemIntrinsic *MI,
         bool AllowTwoMemAccessMasks) -> std::vector<SPIRVWord> {
    std::vector<SPIRVWord> MemoryAccess(1, MemoryAccessMaskNone);
    MaybeAlign DestAlignVal = MI->getDestAlign();
    if (DestAlignVal) {
      Align AlignVal = *DestAlignVal;
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      if (auto *MTI = dyn_cast<MemCpyInst>(MI)) {
        MaybeAlign SourceAlignVal = MTI->getSourceAlign();
        assert(SourceAlignVal && "Missed Source alignment!");

        // In a case when alignment of source differs from dest one
        // we either preserve both (allowed since SPIR-V 1.4), or the least
        // value is guaranteed anyway.
        if (AllowTwoMemAccessMasks) {
          if (*DestAlignVal != *SourceAlignVal) {
            MemoryAccess.push_back(DestAlignVal.valueOrOne().value());
            MemoryAccess.push_back(MemoryAccessAlignedMask);
            AlignVal = *SourceAlignVal;
          }
        } else {
          AlignVal = std::min(*DestAlignVal, *SourceAlignVal);
        }
      }
      MemoryAccess.push_back(AlignVal.value());
    }
    if (MI->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    return MemoryAccess;
  };

  // LLVM intrinsics with known translation to SPIR-V are handled here. They
  // also must be registered at isKnownIntrinsic function in order to make
  // -spirv-allow-unknown-intrinsics work correctly.
  auto IID = II->getIntrinsicID();
  switch (IID) {
  case Intrinsic::assume: {
    // llvm.assume translation is currently supported only within
    // SPV_KHR_expect_assume extension, ignore it otherwise, since it's
    // an optimization hint
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_expect_assume)) {
      SPIRVValue *Condition = transValue(II->getArgOperand(0), BB);
      return BM->addAssumeTrueKHRInst(Condition, BB);
    }
    return nullptr;
  }
  case Intrinsic::bitreverse: {
    if (!BM->getErrorLog().checkError(
            BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_bit_instructions),
            SPIRVEC_InvalidFunctionCall, II,
            "Translation of llvm.bitreverse intrinsic requires "
            "SPV_KHR_bit_instructions extension.")) {
      return nullptr;
    }
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Op = transValue(II->getArgOperand(0), BB);
    return BM->addUnaryInst(OpBitReverse, Ty, Op, BB);
  }

  // Unary FP intrinsic
  case Intrinsic::ceil:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::nearbyint:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::sin:
  case Intrinsic::sqrt:
  case Intrinsic::trunc: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = allowsApproxFunction(II)
                          ? getNativeBuiltinIdForIntrinsic(IID)
                          : getBuiltinIdForIntrinsic(IID);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case Intrinsic::frexp: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(IID);

    SPIRVType *FTy = transType(II->getType()->getStructElementType(0));
    SPIRVTypePointer *ITy = static_cast<SPIRVTypePointer *>(transPointerType(
        II->getType()->getStructElementType(1), SPIRAS_Private));

    unsigned BitWidth = ITy->getElementType()->getBitWidth();
    BM->getErrorLog().checkError(BitWidth == 32, SPIRVEC_InvalidBitWidth,
                                 std::to_string(BitWidth));

    SPIRVValue *IntVal =
        BM->addVariable(ITy, false, spv::internal::LinkageTypeInternal, nullptr,
                        "", ITy->getStorageClass(), BB);

    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB), IntVal};

    return BM->addExtInst(FTy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  // Binary FP intrinsics
  case Intrinsic::copysign:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::maximum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::minnum: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = allowsApproxFunction(II)
                          ? getNativeBuiltinIdForIntrinsic(IID)
                          : getBuiltinIdForIntrinsic(IID);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB)};
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax: {
    Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
    SPIRVValue *FirstArgVal = transValue(II->getArgOperand(0), BB);
    SPIRVValue *SecondArgVal = transValue(II->getArgOperand(1), BB);

    Op OC =
        (IID == Intrinsic::smin)
            ? OpSLessThan
            : ((IID == Intrinsic::smax)
                   ? OpSGreaterThan
                   : ((IID == Intrinsic::umin) ? OpULessThan : OpUGreaterThan));
    if (auto *VecTy = dyn_cast<VectorType>(II->getArgOperand(0)->getType()))
      BoolTy = VectorType::get(BoolTy, VecTy->getElementCount());
    SPIRVValue *Cmp =
        BM->addCmpInst(OC, transType(BoolTy), FirstArgVal, SecondArgVal, BB);
    return BM->addSelectInst(Cmp, FirstArgVal, SecondArgVal, BB);
  }
  case Intrinsic::fma: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = OpenCLLIB::Fma;
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB),
                                  transValue(II->getArgOperand(2), BB)};
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case Intrinsic::abs: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    // LLVM has only one version of abs and it is only for signed integers. We
    // unconditionally choose SAbs here
    SPIRVWord ExtOp = OpenCLLIB::SAbs;
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case Intrinsic::ctpop: {
    return BM->addUnaryInst(OpBitCount, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::ctlz:
  case Intrinsic::cttz: {
    SPIRVWord ExtOp = IID == Intrinsic::ctlz ? OpenCLLIB::Clz : OpenCLLIB::Ctz;
    SPIRVType *Ty = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(Ty, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case Intrinsic::expect: {
    // llvm.expect translation is currently supported only within
    // SPV_KHR_expect_assume extension, replace it with a translated value of #0
    // operand otherwise, since it's an optimization hint
    SPIRVValue *Value = transValue(II->getArgOperand(0), BB);
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_expect_assume)) {
      SPIRVType *Ty = transType(II->getType());
      SPIRVValue *ExpectedValue = transValue(II->getArgOperand(1), BB);
      return BM->addExpectKHRInst(Ty, Value, ExpectedValue, BB);
    }
    return Value;
  }
  case Intrinsic::experimental_constrained_fadd: {
    auto *BI = BM->addBinaryInst(OpFAdd, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fsub: {
    auto *BI = BM->addBinaryInst(OpFSub, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fmul: {
    auto *BI = BM->addBinaryInst(OpFMul, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fdiv: {
    auto *BI = BM->addBinaryInst(OpFDiv, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_frem: {
    auto *BI = BM->addBinaryInst(OpFRem, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fma: {
    std::vector<SPIRVValue *> Args{transValue(II->getArgOperand(0), BB),
                                   transValue(II->getArgOperand(1), BB),
                                   transValue(II->getArgOperand(2), BB)};
    auto *BI = BM->addExtInst(transType(II->getType()),
                             BM->getExtInstSetId(SPIRVEIS_OpenCL),
                             OpenCLLIB::Fma, Args, BB);
    return applyRoundingModeConstraint(II->getOperand(3), BI);
  }
  case Intrinsic::experimental_constrained_fptoui: {
    return BM->addUnaryInst(OpConvertFToU, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_fptosi: {
    return BM->addUnaryInst(OpConvertFToS, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_uitofp: {
    auto *BI = BM->addUnaryInst(OpConvertUToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_sitofp: {
    auto *BI = BM->addUnaryInst(OpConvertSToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fpext: {
    return BM->addUnaryInst(OpFConvert, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_fptrunc: {
    auto *BI = BM->addUnaryInst(OpFConvert, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fcmp:
  case Intrinsic::experimental_constrained_fcmps: {
    auto *MetaMod = cast<MetadataAsValue>(II->getOperand(2))->getMetadata();
    Op CmpTypeOp = StringSwitch<Op>(cast<MDString>(MetaMod)->getString())
                       .Case("oeq", OpFOrdEqual)
                       .Case("ogt", OpFOrdGreaterThan)
                       .Case("oge", OpFOrdGreaterThanEqual)
                       .Case("olt", OpFOrdLessThan)
                       .Case("ole", OpFOrdLessThanEqual)
                       .Case("one", OpFOrdNotEqual)
                       .Case("ord", OpOrdered)
                       .Case("ueq", OpFUnordEqual)
                       .Case("ugt", OpFUnordGreaterThan)
                       .Case("uge", OpFUnordGreaterThanEqual)
                       .Case("ult", OpFUnordLessThan)
                       .Case("ule", OpFUnordLessThanEqual)
                       .Case("une", OpFUnordNotEqual)
                       .Case("uno", OpUnordered)
                       .Default(OpNop);
    assert(CmpTypeOp != OpNop && "Invalid condition code!");
    return BM->addCmpInst(CmpTypeOp, transType(II->getType()),
                          transValue(II->getOperand(0), BB),
                          transValue(II->getOperand(1), BB), BB);
  }
  case Intrinsic::experimental_constrained_fmuladd: {
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Mul =
        BM->addBinaryInst(OpFMul, Ty, transValue(II->getArgOperand(0), BB),
                          transValue(II->getArgOperand(1), BB), BB);
    auto *BI = BM->addBinaryInst(OpFAdd, Ty, Mul,
                                transValue(II->getArgOperand(2), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(3), BI);
  }
  case Intrinsic::fmuladd: {
    // For llvm.fmuladd.* fusion is not guaranteed. If a fused multiply-add
    // is required the corresponding llvm.fma.* intrinsic function should be
    // used instead.
    // If allowed, let's replace llvm.fmuladd.* with mad from OpenCL extended
    // instruction set, as it has the same semantic for FULL_PROFILE OpenCL
    // devices (implementation-defined for EMBEDDED_PROFILE).
    if (BM->shouldReplaceLLVMFmulAddWithOpenCLMad() ||
        BM->getExtInst() == SPIRV::ExtInst::OpenCL) {
      std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                    transValue(II->getArgOperand(1), BB),
                                    transValue(II->getArgOperand(2), BB)};
      return BM->addExtInst(transType(II->getType()),
                            BM->getExtInstSetId(SPIRVEIS_OpenCL),
                            OpenCLLIB::Mad, Ops, BB);
    }

    // Otherwise, just break llvm.fmuladd.* into a pair of fmul + fadd
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Mul =
        BM->addBinaryInst(OpFMul, Ty, transValue(II->getArgOperand(0), BB),
                          transValue(II->getArgOperand(1), BB), BB);
    return BM->addBinaryInst(OpFAdd, Ty, Mul,
                             transValue(II->getArgOperand(2), BB), BB);
  }
  case Intrinsic::fptoui_sat: {
    auto *UI = BM->addUnaryInst(OpConvertFToU, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB), BB);
    UI->setSaturatedConversion(true);
    return UI;
  }
  case Intrinsic::fptosi_sat: {
    auto *UI = BM->addUnaryInst(OpConvertFToS, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB), BB);
    UI->setSaturatedConversion(true);
    return UI;
  }
  case Intrinsic::uadd_sat:
  case Intrinsic::usub_sat:
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat: {
    SPIRVWord ExtOp;
    if (IID == Intrinsic::uadd_sat)
      ExtOp = OpenCLLIB::UAdd_sat;
    else if (IID == Intrinsic::usub_sat)
      ExtOp = OpenCLLIB::USub_sat;
    else if (IID == Intrinsic::sadd_sat)
      ExtOp = OpenCLLIB::SAdd_sat;
    else
      ExtOp = OpenCLLIB::SSub_sat;

    SPIRVType *Ty = transType(II->getType());
    std::vector<SPIRVValue *> Operands = {transValue(II->getArgOperand(0), BB),
                                          transValue(II->getArgOperand(1), BB)};
    return BM->addExtInst(Ty, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp,
                          std::move(Operands), BB);
  }
  case Intrinsic::uadd_with_overflow: {
    return BM->addBinaryInst(OpIAddCarry, transType(II->getType()),
                             transValue(II->getArgOperand(0), BB),
                             transValue(II->getArgOperand(1), BB), BB);
  }
  case Intrinsic::usub_with_overflow: {
    return BM->addBinaryInst(OpISubBorrow, transType(II->getType()),
                             transValue(II->getArgOperand(0), BB),
                             transValue(II->getArgOperand(1), BB), BB);
  }
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_xor: {
    Op Op;
    if (IID == Intrinsic::vector_reduce_add) {
      Op = OpIAdd;
    } else if (IID == Intrinsic::vector_reduce_mul) {
      Op = OpIMul;
    } else if (IID == Intrinsic::vector_reduce_and) {
      Op = OpBitwiseAnd;
    } else if (IID == Intrinsic::vector_reduce_or) {
      Op = OpBitwiseOr;
    } else {
      Op = OpBitwiseXor;
    }
    VectorType *VecTy = cast<VectorType>(II->getArgOperand(0)->getType());
    SPIRVValue *VecSVal = transValue(II->getArgOperand(0), BB);
    SPIRVTypeInt *ResultSType =
        BM->addIntegerType(VecTy->getElementType()->getIntegerBitWidth());
    SPIRVTypeInt *I32STy = BM->addIntegerType(32);
    unsigned VecSize = VecTy->getElementCount().getFixedValue();
    if (VecSize > 0) {
      SmallVector<SPIRVValue *, 16> Extracts(VecSize);
      for (unsigned Idx = 0; Idx < VecSize; ++Idx) {
        Extracts[Idx] = BM->addVectorExtractDynamicInst(
            VecSVal, BM->addIntegerConstant(I32STy, Idx), BB);
      }
      unsigned Counter = VecSize >> 1;
      while (Counter != 0) {
        for (unsigned Idx = 0; Idx < Counter; ++Idx) {
          Extracts[Idx] = BM->addBinaryInst(Op, ResultSType, Extracts[Idx << 1],
                                            Extracts[(Idx << 1) + 1], BB);
        }
        Counter >>= 1;
      }
      if ((VecSize & 1) != 0) {
        Extracts[0] = BM->addBinaryInst(Op, ResultSType, Extracts[0],
                                        Extracts[VecSize - 1], BB);
      }
      return Extracts[0];
    }
    assert(VecSize && "Zero Extracts size for vector reduce lowering");
    return nullptr;
  }
  case Intrinsic::vector_reduce_fadd:
  case Intrinsic::vector_reduce_fmul: {
    Op Op = IID == Intrinsic::vector_reduce_fadd ? OpFAdd : OpFMul;
    VectorType *VecTy = cast<VectorType>(II->getArgOperand(1)->getType());
    SPIRVValue *VecSVal = transValue(II->getArgOperand(1), BB);
    SPIRVValue *StartingSVal = transValue(II->getArgOperand(0), BB);
    SPIRVTypeInt *I32STy = BM->addIntegerType(32);
    unsigned VecSize = VecTy->getElementCount().getFixedValue();
    if (VecSize > 0) {
      SmallVector<SPIRVValue *, 16> Extracts(VecSize);
      for (unsigned Idx = 0; Idx < VecSize; ++Idx) {
        Extracts[Idx] = BM->addVectorExtractDynamicInst(
            VecSVal, BM->addIntegerConstant(I32STy, Idx), BB);
      }
      SPIRVValue *V = BM->addBinaryInst(Op, StartingSVal->getType(),
                                        StartingSVal, Extracts[0], BB);
      for (unsigned Idx = 1; Idx < VecSize; ++Idx) {
        V = BM->addBinaryInst(Op, StartingSVal->getType(), V, Extracts[Idx],
                              BB);
      }
      return V;
    }
    assert(VecSize && "Zero Extracts size for vector reduce lowering");
    return nullptr;
  }
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_umax:
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_fmax:
  case Intrinsic::vector_reduce_fmin:
  case Intrinsic::vector_reduce_fmaximum:
  case Intrinsic::vector_reduce_fminimum: {
    Op Op;
    if (IID == Intrinsic::vector_reduce_smax) {
      Op = OpSGreaterThan;
    } else if (IID == Intrinsic::vector_reduce_smin) {
      Op = OpSLessThan;
    } else if (IID == Intrinsic::vector_reduce_umax) {
      Op = OpUGreaterThan;
    } else if (IID == Intrinsic::vector_reduce_umin) {
      Op = OpULessThan;
    } else if (IID == Intrinsic::vector_reduce_fmax) {
      Op = OpFOrdGreaterThan;
    } else if (IID == Intrinsic::vector_reduce_fmin) {
      Op = OpFOrdLessThan;
    } else if (IID == Intrinsic::vector_reduce_fmaximum) {
      Op = OpFUnordGreaterThan;
    } else {
      Op = OpFUnordLessThan;
    }
    VectorType *VecTy = cast<VectorType>(II->getArgOperand(0)->getType());
    SPIRVValue *VecSVal = transValue(II->getArgOperand(0), BB);
    SPIRVType *BoolSTy = transType(Type::getInt1Ty(II->getContext()));
    SPIRVTypeInt *I32STy = BM->addIntegerType(32);
    unsigned VecSize = VecTy->getElementCount().getFixedValue();
    SmallVector<SPIRVValue *, 16> Extracts(VecSize);
    if (VecSize > 0) {
      for (unsigned Idx = 0; Idx < VecSize; ++Idx) {
        Extracts[Idx] = BM->addVectorExtractDynamicInst(
            VecSVal, BM->addIntegerConstant(I32STy, Idx), BB);
      }
      unsigned Counter = VecSize >> 1;
      while (Counter != 0) {
        for (unsigned Idx = 0; Idx < Counter; ++Idx) {
          SPIRVValue *Cond = BM->addBinaryInst(Op, BoolSTy, Extracts[Idx << 1],
                                               Extracts[(Idx << 1) + 1], BB);
          Extracts[Idx] = BM->addSelectInst(Cond, Extracts[Idx << 1],
                                            Extracts[(Idx << 1) + 1], BB);
        }
        Counter >>= 1;
      }
      if ((VecSize & 1) != 0) {
        SPIRVValue *Cond = BM->addBinaryInst(Op, BoolSTy, Extracts[0],
                                             Extracts[VecSize - 1], BB);
        Extracts[0] =
            BM->addSelectInst(Cond, Extracts[0], Extracts[VecSize - 1], BB);
      }
      return Extracts[0];
    }
    assert(VecSize && "Zero Extracts size for vector reduce lowering");
    return nullptr;
  }
  case Intrinsic::memset: {
    // Generally there is no direct mapping of memset to SPIR-V.  But it turns
    // out that memset is emitted by Clang for initialization in default
    // constructors so we need some basic support.  The code below only handles
    // cases with constant value and constant length.
    MemSetInst *MSI = cast<MemSetInst>(II);
    Value *Val = MSI->getValue();
    if (!isa<Constant>(Val)) {
      assert(false &&
             "Can't translate llvm.memset with non-const `value` argument");
      return nullptr;
    }
    Value *Len = MSI->getLength();
    if (!isa<ConstantInt>(Len)) {
      assert(false &&
             "Can't translate llvm.memset with non-const `length` argument");
      return nullptr;
    }
    uint64_t NumElements = static_cast<ConstantInt *>(Len)->getZExtValue();
    auto *AT = ArrayType::get(Val->getType(), NumElements);
    SPIRVTypeArray *CompositeTy = static_cast<SPIRVTypeArray *>(transType(AT));
    SPIRVValue *Init;
    if (cast<Constant>(Val)->isZeroValue()) {
      Init = BM->addNullConstant(CompositeTy);
    } else {
      // On 32-bit systems, size_type of std::vector is not a 64-bit type. Let's
      // assume that we won't encounter memset for more than 2^32 elements and
      // insert explicit cast to avoid possible warning/error about narrowing
      // conversion
      auto TNumElts =
          static_cast<std::vector<SPIRVValue *>::size_type>(NumElements);
      std::vector<SPIRVValue *> Elts(TNumElts, transValue(Val, BB));
      Init = BM->addCompositeConstant(CompositeTy, Elts);
    }
    SPIRVType *VarTy = transPointerType(AT, SPIRV::SPIRAS_Constant);
    SPIRVValue *Var = BM->addVariable(VarTy, /*isConstant*/ true,
                                      spv::internal::LinkageTypeInternal, Init,
                                      "", StorageClassUniformConstant, nullptr);
    SPIRVType *SourceTy =
        transPointerType(Val->getType(), SPIRV::SPIRAS_Constant);
    SPIRVValue *Source = BM->addUnaryInst(OpBitcast, SourceTy, Var, BB);
    SPIRVValue *Target = transValue(MSI->getRawDest(), BB);
    return BM->addCopyMemorySizedInst(
        Target, Source, CompositeTy->getLength(),
        GetMemoryAccess(MSI,
                        BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)),
        BB);
  } break;
  case Intrinsic::memcpy:
    return BM->addCopyMemorySizedInst(
        transValue(II->getOperand(0), BB), transValue(II->getOperand(1), BB),
        transValue(II->getOperand(2), BB),
        GetMemoryAccess(cast<MemIntrinsic>(II),
                        BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)),
        BB);
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end: {
    Op OC = (II->getIntrinsicID() == Intrinsic::lifetime_start)
                ? OpLifetimeStart
                : OpLifetimeStop;
    int64_t Size = dyn_cast<ConstantInt>(II->getOperand(0))->getSExtValue();
    if (Size == -1)
      Size = 0;
    return BM->addLifetimeInst(OC, transValue(II->getOperand(1), BB), Size, BB);
  }
  // We don't want to mix translation of regular code and debug info, because
  // it creates a mess, therefore translation of debug intrinsics is
  // postponed until LLVMToSPIRVDbgTran::finalizeDebug...() methods.
  case Intrinsic::dbg_declare:
    return DbgTran->createDebugDeclarePlaceholder(cast<DbgDeclareInst>(II), BB);
  case Intrinsic::dbg_value:
    return DbgTran->createDebugValuePlaceholder(cast<DbgValueInst>(II), BB);
  case Intrinsic::annotation: {
    SPIRVType *Ty = transScavengedType(II);
    Constant *C = cast<Constant>(II->getArgOperand(1)->stripPointerCasts());
    StringRef AnnotationString;
    if (!getConstantStringInfo(C, AnnotationString))
      return nullptr;

    if (AnnotationString == kOCLBuiltinName::FPGARegIntel) {
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_reg))
        return BM->addFPGARegINTELInst(Ty, transValue(II->getOperand(0), BB),
                                       BB);
      else
        return transValue(II->getOperand(0), BB);
    }

    return nullptr;
  }
  case Intrinsic::var_annotation: {
    SPIRVValue *SV;
    if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0))) {
      SV = transValue(BI->getOperand(0), BB);
    } else {
      SV = transValue(II->getOperand(0), BB);
    }

    std::string AnnotationString;
    processAnnotationString(II, AnnotationString);
    DecorationsInfoVec Decorations =
        tryParseAnnotationString(BM, AnnotationString).MemoryAttributesVec;

    // If we didn't find any IntelFPGA-specific decorations, let's add the whole
    // annotation string as UserSemantic Decoration
    if (Decorations.empty()) {
      SV->addDecorate(
          new SPIRVDecorateUserSemanticAttr(SV, AnnotationString.c_str()));
    } else {
      addAnnotationDecorations(SV, Decorations);
    }
    return SV;
  }
  // The layout of llvm.ptr.annotation is:
  // declare iN*   @llvm.ptr.annotation.p<address space>iN(
  // iN* <val>, i8* <str>, i8* <str>, i32  <int>, i8* <ptr>)
  // where N is a power of two number,
  // first i8* <str> stands for the annotation itself,
  // second i8* <str> is for the location (file name),
  // i8* <ptr> is a pointer on a GV, which can carry optinal variadic
  // clang::annotation attribute expression arguments.
  case Intrinsic::ptr_annotation: {
    Value *AnnotSubj = nullptr;
    if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0))) {
      AnnotSubj = BI->getOperand(0);
    } else {
      AnnotSubj = II->getOperand(0);
    }

    std::string AnnotationString;
    processAnnotationString(II, AnnotationString);
    AnnotationDecorations Decorations =
        tryParseAnnotationString(BM, AnnotationString);
    // Translate FPGARegIntel annotations to OpFPGARegINTEL.
    if (AnnotationString == kOCLBuiltinName::FPGARegIntel) {
      auto *Ty = transScavengedType(II);
      auto *BI = dyn_cast<BitCastInst>(II->getOperand(0));
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_reg))
        return BM->addFPGARegINTELInst(Ty, transValue(BI, BB), BB);
      return transValue(BI, BB);
    }

    SPIRVValue *DecSubj = transValue(AnnotSubj, BB);
    if (Decorations.empty()) {
      DecSubj->addDecorate(
          new SPIRVDecorateUserSemanticAttr(DecSubj, AnnotationString.c_str()));
    } else {
      addAnnotationDecorations(DecSubj, Decorations.MemoryAttributesVec);
      // Apply the LSU parameter decoration to the pointer result of a GEP
      // to the given struct member (InBoundsPtrAccessChain in SPIR-V).
      // Decorating the member itself with a MemberDecoration is not feasible,
      // because multiple accesses to the struct-held memory can require
      // different LSU parameters.
      addAnnotationDecorations(DecSubj, Decorations.MemoryAccessesVec);
      addAnnotationDecorations(DecSubj, Decorations.CacheControlVec);
      addAnnotationDecorations(DecSubj, Decorations.BufferLocationVec);
      if (allowDecorateWithLatencyControlINTEL(II)) {
        addAnnotationDecorations(DecSubj, Decorations.LatencyControlVec);
      }
    }
    II->replaceAllUsesWith(II->getOperand(0));
    return DecSubj;
  }
  case Intrinsic::stacksave: {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_variable_length_array)) {
      auto *Ty = transPointerType(Type::getInt8Ty(M->getContext()), 0);
      return BM->addInstTemplate(OpSaveMemoryINTEL, BB, Ty);
    }
    BM->getErrorLog().checkError(
        BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
        "Translation of llvm.stacksave intrinsic requires "
        "SPV_INTEL_variable_length_array extension or "
        "-spirv-allow-unknown-intrinsics option.");
    break;
  }
  case Intrinsic::stackrestore: {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_variable_length_array)) {
      auto *Ptr = transValue(II->getArgOperand(0), BB);
      return BM->addInstTemplate(OpRestoreMemoryINTEL, {Ptr->getId()}, BB,
                                 nullptr);
    }
    BM->getErrorLog().checkError(
        BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
        "Translation of llvm.restore intrinsic requires "
        "SPV_INTEL_variable_length_array extension or "
        "-spirv-allow-unknown-intrinsics option.");
    break;
  }
  // We can just ignore/drop some intrinsics, like optimizations hint.
  case Intrinsic::experimental_noalias_scope_decl:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::dbg_label:
  // llvm.trap intrinsic is not implemented. But for now don't crash. This
  // change is pending the trap/abort intrinsic implementation.
  case Intrinsic::trap:
  case Intrinsic::ubsantrap:
  // llvm.instrprof.* intrinsics are not supported
  case Intrinsic::instrprof_increment:
  case Intrinsic::instrprof_increment_step:
  case Intrinsic::instrprof_value_profile:
    return nullptr;
  case Intrinsic::is_constant: {
    auto *CO = dyn_cast<Constant>(II->getOperand(0));
    if (CO && isManifestConstant(CO))
      return transValue(ConstantInt::getTrue(II->getType()), BB, false);
    else
      return transValue(ConstantInt::getFalse(II->getType()), BB, false);
  }
  case Intrinsic::arithmetic_fence: {
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Op = transValue(II->getArgOperand(0), BB);
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_arithmetic_fence)) {
      BM->addCapability(internal::CapabilityFPArithmeticFenceINTEL);
      BM->addExtension(ExtensionID::SPV_INTEL_arithmetic_fence);
      return BM->addUnaryInst(internal::OpArithmeticFenceINTEL, Ty, Op, BB);
    }
    return Op;
  }
  case Intrinsic::masked_gather: {
    if (!BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_masked_gather_scatter)) {
      BM->getErrorLog().checkError(
          BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
          "Translation of llvm.masked.gather intrinsic requires "
          "SPV_INTEL_masked_gather_scatter extension or "
          "-spirv-allow-unknown-intrinsics option.");
      return nullptr;
    }
    SPIRVType *Ty = transScavengedType(II);
    auto *PtrVector = transValue(II->getArgOperand(0), BB);
    uint32_t Alignment =
        cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    auto *Mask = transValue(II->getArgOperand(2), BB);
    auto *FillEmpty = transValue(II->getArgOperand(3), BB);
    std::vector<SPIRVWord> Ops = {PtrVector->getId(), Alignment, Mask->getId(),
                                  FillEmpty->getId()};
    return BM->addInstTemplate(internal::OpMaskedGatherINTEL, Ops, BB, Ty);
  }
  case Intrinsic::masked_scatter: {
    if (!BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_masked_gather_scatter)) {
      BM->getErrorLog().checkError(
          BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
          "Translation of llvm.masked.scatter intrinsic requires "
          "SPV_INTEL_masked_gather_scatter extension or "
          "-spirv-allow-unknown-intrinsics option.");
      return nullptr;
    }
    auto *InputVector = transValue(II->getArgOperand(0), BB);
    auto *PtrVector = transValue(II->getArgOperand(1), BB);
    uint32_t Alignment =
        cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    auto *Mask = transValue(II->getArgOperand(3), BB);
    std::vector<SPIRVWord> Ops = {InputVector->getId(), PtrVector->getId(),
                                  Alignment, Mask->getId()};
    return BM->addInstTemplate(internal::OpMaskedScatterINTEL, Ops, BB,
                               nullptr);
  }
  case Intrinsic::is_fpclass: {
    // There is no direct counterpart for the intrinsic in SPIR-V, hence
    // we need to emulate its work by sequence of other instructions
    SPIRVType *ResTy = transType(II->getType());
    llvm::FPClassTest FPClass = static_cast<llvm::FPClassTest>(
        cast<ConstantInt>(II->getArgOperand(1))->getZExtValue());
    // if no tests are provided - return false
    if (FPClass == 0)
      return BM->addConstant(ResTy, false);
    // if all tests are provided - return true
    if (FPClass == fcAllFlags)
      return BM->addConstant(ResTy, true);

    Type *OpLLVMTy = II->getArgOperand(0)->getType();
    SPIRVValue *InputFloat = transValue(II->getArgOperand(0), BB);
    std::vector<SPIRVValue *> ResultVec;

    // Adds test for Negative/Positive values
    SPIRVValue *SignBitTest = nullptr;
    SPIRVValue *NoSignTest = nullptr;
    auto GetNegPosInstTest = [&](SPIRVValue *TestInst,
                                 bool IsNegative) -> SPIRVValue * {
      SignBitTest = (SignBitTest)
                        ? SignBitTest
                        : BM->addInstTemplate(OpSignBitSet,
                                              {InputFloat->getId()}, BB, ResTy);
      if (IsNegative) {
        return BM->addInstTemplate(
            OpLogicalAnd, {SignBitTest->getId(), TestInst->getId()}, BB, ResTy);
      }
      NoSignTest = (NoSignTest)
                       ? NoSignTest
                       : BM->addInstTemplate(OpLogicalNot,
                                             {SignBitTest->getId()}, BB, ResTy);
      return BM->addInstTemplate(
          OpLogicalAnd, {NoSignTest->getId(), TestInst->getId()}, BB, ResTy);
    };

    // Get LLVM Op type converted to integer. It can be either scalar or vector.
    const uint32_t BitSize = OpLLVMTy->getScalarSizeInBits();
    Type *IntOpLLVMTy = IntegerType::getIntNTy(M->getContext(), BitSize);
    if (OpLLVMTy->isVectorTy())
      IntOpLLVMTy = FixedVectorType::get(
          IntOpLLVMTy, cast<FixedVectorType>(OpLLVMTy)->getNumElements());
    SPIRVType *OpSPIRVTy = transType(IntOpLLVMTy);
    const llvm::fltSemantics &Semantics =
        OpLLVMTy->getScalarType()->getFltSemantics();
    const APInt Inf = APFloat::getInf(Semantics).bitcastToAPInt();
    const APInt AllOneMantissa =
        APFloat::getLargest(Semantics).bitcastToAPInt() & ~Inf;

    // Some checks can be inverted tests for simple cases, for example
    // simultaneous check for inf, normal, subnormal and zero is a check for
    // non nan.
    auto GetInvertedFPClassTest =
        [](const llvm::FPClassTest Test) -> llvm::FPClassTest {
      llvm::FPClassTest InvertedTest = ~Test & fcAllFlags;
      switch (InvertedTest) {
      default:
        break;
      case fcNan:
      case fcSNan:
      case fcQNan:
      case fcInf:
      case fcPosInf:
      case fcNegInf:
      case fcNormal:
      case fcPosNormal:
      case fcNegNormal:
      case fcSubnormal:
      case fcPosSubnormal:
      case fcNegSubnormal:
      case fcZero:
      case fcPosZero:
      case fcNegZero:
      case fcFinite:
      case fcPosFinite:
      case fcNegFinite:
        return InvertedTest;
      }
      return fcNone;
    };
    bool IsInverted = false;
    if (llvm::FPClassTest InvertedCheck = GetInvertedFPClassTest(FPClass)) {
      IsInverted = true;
      FPClass = InvertedCheck;
    }
    auto GetInvertedTestIfNeeded = [&](SPIRVValue *TestInst) -> SPIRVValue * {
      if (!IsInverted)
        return TestInst;
      return BM->addInstTemplate(OpLogicalNot, {TestInst->getId()}, BB, ResTy);
    };

    // TODO: we can add some optimization for fcFinite check by replacing it
    // with fabs + cmp to 0x7FF0000000000000

    // Integer parameter of the intrinsic is combined from several bit masks
    // referenced in FPClassTest enum from FloatingPointMode.h in LLVM.
    // Since a single intrinsic can provide multiple tests - here we might end
    // up adding several sequences of SPIR-V instructions
    if (FPClass & fcNan) {
      // Map on OpIsNan if we have both QNan and SNan test bits set
      if (FPClass & fcSNan && FPClass & fcQNan) {
        auto *TestIsNan =
            BM->addInstTemplate(OpIsNan, {InputFloat->getId()}, BB, ResTy);
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsNan));
      } else {
        // isquiet(V) ==> abs(V) >= (unsigned(Inf) | quiet_bit)
        APInt QNaNBitMask =
            APInt::getOneBitSet(BitSize, AllOneMantissa.getActiveBits() - 1);
        APInt InfWithQnanBit = Inf | QNaNBitMask;
        auto *QNanBitConst = transValue(
            Constant::getIntegerValue(IntOpLLVMTy, InfWithQnanBit), BB);
        auto *BitCastToInt =
            BM->addUnaryInst(OpBitcast, OpSPIRVTy, InputFloat, BB);
        auto *TestIsQNan = BM->addCmpInst(OpUGreaterThanEqual, ResTy,
                                          BitCastToInt, QNanBitConst, BB);
        if (FPClass & fcQNan) {
          ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsQNan));
        } else {
          // issignaling(V) ==> isnan(V) && !isquiet(V)
          auto *TestIsNan =
              BM->addInstTemplate(OpIsNan, {InputFloat->getId()}, BB, ResTy);
          auto *NotQNan = BM->addInstTemplate(OpLogicalNot,
                                              {TestIsQNan->getId()}, BB, ResTy);
          auto *TestIsSNan = BM->addInstTemplate(
              OpLogicalAnd, {TestIsNan->getId(), NotQNan->getId()}, BB, ResTy);
          ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsSNan));
        }
      }
    }
    if (FPClass & fcInf) {
      auto *TestIsInf =
          BM->addInstTemplate(OpIsInf, {InputFloat->getId()}, BB, ResTy);
      if (FPClass & fcNegInf && FPClass & fcPosInf)
        // Map on OpIsInf if we have both Inf test bits set
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsInf));
      else
        // Map on OpIsInf with following check for sign bit
        ResultVec.emplace_back(GetInvertedTestIfNeeded(
            GetNegPosInstTest(TestIsInf, FPClass & fcNegInf)));
    }
    if (FPClass & fcNormal) {
      auto *TestIsNormal =
          BM->addInstTemplate(OpIsNormal, {InputFloat->getId()}, BB, ResTy);
      if (FPClass & fcNegNormal && FPClass & fcPosNormal)
        // Map on OpIsNormal if we have both Normal test bits set
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsNormal));
      else
        // Map on OpIsNormal with following check for sign bit
        ResultVec.emplace_back(GetInvertedTestIfNeeded(
            GetNegPosInstTest(TestIsNormal, FPClass & fcNegNormal)));
    }
    if (FPClass & fcSubnormal) {
      // issubnormal(V) ==> unsigned(abs(V) - 1) < (all mantissa bits set)
      auto *BitCastToInt =
          BM->addUnaryInst(OpBitcast, OpSPIRVTy, InputFloat, BB);
      auto *MantissaConst = transValue(
          Constant::getIntegerValue(IntOpLLVMTy, AllOneMantissa), BB);
      auto *MinusOne =
          BM->addBinaryInst(OpISub, OpSPIRVTy, BitCastToInt, MantissaConst, BB);
      auto *TestIsSubnormal =
          BM->addCmpInst(OpULessThan, ResTy, MinusOne, MantissaConst, BB);
      if (FPClass & fcPosSubnormal && FPClass & fcNegSubnormal)
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsSubnormal));
      else
        ResultVec.emplace_back(GetInvertedTestIfNeeded(
            GetNegPosInstTest(TestIsSubnormal, FPClass & fcNegSubnormal)));
    }
    if (FPClass & fcZero) {
      // Create zero integer constant and check for equality with bitcasted to
      // int float value
      auto SetUpCMPToZero = [&](SPIRVValue *BitCastToInt,
                                bool IsPositive) -> SPIRVValue * {
        APInt ZeroInt = APInt::getZero(BitSize);
        if (IsPositive) {
          auto *ZeroConst =
              transValue(Constant::getIntegerValue(IntOpLLVMTy, ZeroInt), BB);
          return BM->addCmpInst(OpIEqual, ResTy, BitCastToInt, ZeroConst, BB);
        }
        // Created 'negated' zero
        ZeroInt.setSignBit();
        auto *NegZeroConst =
            transValue(Constant::getIntegerValue(IntOpLLVMTy, ZeroInt), BB);
        return BM->addCmpInst(OpIEqual, ResTy, BitCastToInt, NegZeroConst, BB);
      };
      auto *BitCastToInt =
          BM->addUnaryInst(OpBitcast, OpSPIRVTy, InputFloat, BB);
      if (FPClass & fcPosZero && FPClass & fcNegZero) {
        APInt ZeroInt = APInt::getZero(BitSize);
        auto *ZeroConst =
            transValue(Constant::getIntegerValue(IntOpLLVMTy, ZeroInt), BB);
        APInt MaskToClearSignBit = APInt::getSignedMaxValue(BitSize);
        auto *MaskToClearSignBitConst = transValue(
            Constant::getIntegerValue(IntOpLLVMTy, MaskToClearSignBit), BB);
        auto *BitwiseAndRes = BM->addBinaryInst(
            OpBitwiseAnd, OpSPIRVTy, BitCastToInt, MaskToClearSignBitConst, BB);
        auto *TestIsZero =
            BM->addCmpInst(OpIEqual, ResTy, BitwiseAndRes, ZeroConst, BB);
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsZero));
      } else if (FPClass & fcPosZero) {
        auto *TestIsPosZero =
            SetUpCMPToZero(BitCastToInt, true /*'positive' zero*/);
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsPosZero));
      } else {
        auto *TestIsNegZero =
            SetUpCMPToZero(BitCastToInt, false /*'negated' zero*/);
        ResultVec.emplace_back(GetInvertedTestIfNeeded(TestIsNegZero));
      }
    }
    if (ResultVec.size() == 1)
      return ResultVec.back();
    SPIRVValue *Result = ResultVec.front();
    for (size_t I = 1; I != ResultVec.size(); ++I) {
      // Create a sequence of LogicalOr instructions from ResultVec to get
      // the overall test result
      std::vector<SPIRVId> LogicOps = {Result->getId(), ResultVec[I]->getId()};
      Result = BM->addInstTemplate(OpLogicalOr, LogicOps, BB, ResTy);
    }
    return Result;
  }
  default:
    if (auto *BVar = transFPBuiltinIntrinsicInst(II, BB))
      return BVar;
    if (BM->isUnknownIntrinsicAllowed(II))
      return BM->addCallInst(
          transFunctionDecl(II->getCalledFunction()),
          transArguments(II, BB,
                         SPIRVEntry::createUnique(OpFunctionCall).get()),
          BB);
    else
      // Other LLVM intrinsics shouldn't get to SPIRV, because they
      // can't be represented in SPIRV or aren't implemented yet.
      BM->SPIRVCK(
          false, InvalidFunctionCall, II->getCalledOperand()->getName().str());
  }
  return nullptr;
}

LLVMToSPIRVBase::FPBuiltinType
LLVMToSPIRVBase::getFPBuiltinType(IntrinsicInst *II, StringRef &OpName) {
  StringRef Name = II->getCalledFunction()->getName();
  if (!Name.consume_front("llvm.fpbuiltin."))
    return FPBuiltinType::UNKNOWN;
  OpName = Name.split('.').first;
  FPBuiltinType Type =
      StringSwitch<FPBuiltinType>(OpName)
          .Cases("fadd", "fsub", "fmul", "fdiv", "frem",
                 FPBuiltinType::REGULAR_MATH)
          .Cases("sin", "cos", "tan", FPBuiltinType::EXT_1OPS)
          .Cases("sinh", "cosh", "tanh", FPBuiltinType::EXT_1OPS)
          .Cases("asin", "acos", "atan", FPBuiltinType::EXT_1OPS)
          .Cases("asinh", "acosh", "atanh", FPBuiltinType::EXT_1OPS)
          .Cases("exp", "exp2", "exp10", "expm1", FPBuiltinType::EXT_1OPS)
          .Cases("log", "log2", "log10", "log1p", FPBuiltinType::EXT_1OPS)
          .Cases("sqrt", "rsqrt", "erf", "erfc", FPBuiltinType::EXT_1OPS)
          .Cases("atan2", "pow", "hypot", "ldexp", FPBuiltinType::EXT_2OPS)
          .Case("sincos", FPBuiltinType::EXT_3OPS)
          .Default(FPBuiltinType::UNKNOWN);
  return Type;
}

SPIRVValue *LLVMToSPIRVBase::transFPBuiltinIntrinsicInst(IntrinsicInst *II,
                                                         SPIRVBasicBlock *BB) {
  StringRef OpName;
  auto FPBuiltinTypeVal = getFPBuiltinType(II, OpName);
  if (FPBuiltinTypeVal == FPBuiltinType::UNKNOWN)
    return nullptr;
  switch (FPBuiltinTypeVal) {
  case FPBuiltinType::REGULAR_MATH: {
    auto BinOp = StringSwitch<Op>(OpName)
                     .Case("fadd", OpFAdd)
                     .Case("fsub", OpFSub)
                     .Case("fmul", OpFMul)
                     .Case("fdiv", OpFDiv)
                     .Case("frem", OpFRem)
                     .Default(OpUndef);
    return BM->addBinaryInst(BinOp, transType(II->getType()),
                             transValue(II->getArgOperand(0), BB),
                             transValue(II->getArgOperand(1), BB), BB);
  }
  case FPBuiltinType::EXT_1OPS: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    auto ExtOp = StringSwitch<SPIRVWord>(OpName)
                     .Case("sin", OpenCLLIB::Sin)
                     .Case("cos", OpenCLLIB::Cos)
                     .Case("tan", OpenCLLIB::Tan)
                     .Case("sinh", OpenCLLIB::Sinh)
                     .Case("cosh", OpenCLLIB::Cosh)
                     .Case("tanh", OpenCLLIB::Tanh)
                     .Case("asin", OpenCLLIB::Asin)
                     .Case("acos", OpenCLLIB::Acos)
                     .Case("atan", OpenCLLIB::Atan)
                     .Case("asinh", OpenCLLIB::Asinh)
                     .Case("acosh", OpenCLLIB::Acosh)
                     .Case("atanh", OpenCLLIB::Atanh)
                     .Case("exp", OpenCLLIB::Exp)
                     .Case("exp2", OpenCLLIB::Exp2)
                     .Case("exp10", OpenCLLIB::Exp10)
                     .Case("expm1", OpenCLLIB::Expm1)
                     .Case("log", OpenCLLIB::Log)
                     .Case("log2", OpenCLLIB::Log2)
                     .Case("log10", OpenCLLIB::Log10)
                     .Case("log1p", OpenCLLIB::Log1p)
                     .Case("sqrt", OpenCLLIB::Sqrt)
                     .Case("rsqrt", OpenCLLIB::Rsqrt)
                     .Case("erf", OpenCLLIB::Erf)
                     .Case("erfc", OpenCLLIB::Erfc)
                     .Default(SPIRVWORD_MAX);
    assert(ExtOp != SPIRVWORD_MAX);
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case FPBuiltinType::EXT_2OPS: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB)};
    auto ExtOp = StringSwitch<SPIRVWord>(OpName)
                     .Case("atan2", OpenCLLIB::Atan2)
                     .Case("hypot", OpenCLLIB::Hypot)
                     .Case("pow", OpenCLLIB::Pow)
                     .Case("ldexp", OpenCLLIB::Ldexp)
                     .Default(SPIRVWORD_MAX);
    assert(ExtOp != SPIRVWORD_MAX);
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  case FPBuiltinType::EXT_3OPS: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB),
                                  transValue(II->getArgOperand(2), BB)};
    auto ExtOp = StringSwitch<SPIRVWord>(OpName)
                     .Case("sincos", OpenCLLIB::Sincos)
                     .Default(SPIRVWORD_MAX);
    assert(ExtOp != SPIRVWORD_MAX);
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                          BB);
  }
  default:
    return nullptr;
  }
  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transFenceInst(FenceInst *FI,
                                            SPIRVBasicBlock *BB) {
  SPIRVWord MemorySemantics;
  // Fence ordering may only be Acquire, Release, AcquireRelease, or
  // SequentiallyConsistent
  switch (FI->getOrdering()) {
  case llvm::AtomicOrdering::Acquire:
    MemorySemantics = MemorySemanticsAcquireMask;
    break;
  case llvm::AtomicOrdering::Release:
    MemorySemantics = MemorySemanticsReleaseMask;
    break;
  case llvm::AtomicOrdering::AcquireRelease:
    MemorySemantics = MemorySemanticsAcquireReleaseMask;
    break;
  case llvm::AtomicOrdering::SequentiallyConsistent:
    MemorySemantics = MemorySemanticsSequentiallyConsistentMask;
    break;
  default:
    assert(false && "Unexpected fence ordering");
    MemorySemantics = SPIRVWORD_MAX;
    break;
  }

  Module *M = FI->getParent()->getModule();
  SmallVector<StringRef> SSIDs;
  FI->getContext().getSyncScopeNames(SSIDs);
  spv::Scope S;
  // Treat all llvm.fence instructions as having CrossDevice scope by default
  if (!OCLStrMemScopeMap::find(SSIDs[FI->getSyncScopeID()].str(), &S)) {
    S = ScopeCrossDevice;
  }

  SPIRVValue *RetScope = transConstant(getUInt32(M, S));
  SPIRVValue *Val = transConstant(getUInt32(M, MemorySemantics));
  return BM->addMemoryBarrierInst(static_cast<Scope>(RetScope->getId()),
                                  Val->getId(), BB);
}

SPIRVValue *LLVMToSPIRVBase::transCallInst(CallInst *CI, SPIRVBasicBlock *BB) {
  assert(CI);
  Function *F = CI->getFunction();
  if (isa<InlineAsm>(CI->getCalledOperand()) &&
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_inline_assembly)) {
    // Inline asm is opaque, so we cannot reason about its FP contraction
    // requirements.
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": inline asm " << *CI << '\n');
    joinFPContract(F, FPContract::DISABLED);
    return transAsmCallINTEL(CI, BB);
  }

  if (CI->isIndirectCall()) {
    // The function is not known in advance
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": indirect call " << *CI << '\n');
    joinFPContract(F, FPContract::DISABLED);
    return transIndirectCallInst(CI, BB);
  }
  return transDirectCallInst(CI, BB);
}

SPIRVValue *LLVMToSPIRVBase::transDirectCallInst(CallInst *CI,
                                                 SPIRVBasicBlock *BB) {
  SPIRVExtInstSetKind ExtSetKind = SPIRVEIS_Count;
  SPIRVWord ExtOp = SPIRVWORD_MAX;
  llvm::Function *F = CI->getCalledFunction();
  auto MangledName = F->getName();
  StringRef DemangledName;

  if (MangledName.starts_with(SPCV_CAST) || MangledName == SAMPLER_INIT)
    return oclTransSpvcCastSampler(CI, BB);

  if (oclIsBuiltin(MangledName, DemangledName) ||
      isDecoratedSPIRVFunc(F, DemangledName)) {
    if (auto *BV = transBuiltinToConstant(DemangledName, CI))
      return BV;
    if (auto *BV = transBuiltinToInst(DemangledName, CI, BB))
      return BV;
  }

  SmallVector<std::string, 2> Dec;
  if (isBuiltinTransToExtInst(CI->getCalledFunction(), &ExtSetKind, &ExtOp,
                              &Dec)) {
    if (DemangledName.find("__spirv_ocl_printf") != StringRef::npos) {
      auto *FormatStrPtr = cast<PointerType>(CI->getArgOperand(0)->getType());
      if (FormatStrPtr->getAddressSpace() !=
          SPIR::TypeAttributeEnum::ATTR_CONST) {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_EXT_relaxed_printf_string_address_space)) {
          std::string ErrorStr =
              "Either SPV_EXT_relaxed_printf_string_address_space extension "
              "should be allowed to translate this module, because this LLVM "
              "module contains the printf function with format string, whose "
              "address space is not equal to 2 (constant).";
          getErrorLog().checkError(false, SPIRVEC_RequiresExtension, CI,
                                   ErrorStr);
        }
        BM->addExtension(
            ExtensionID::SPV_EXT_relaxed_printf_string_address_space);
      }
    }

    return addDecorations(
        BM->addExtInst(
            transScavengedType(CI), BM->getExtInstSetId(ExtSetKind), ExtOp,
            transArguments(CI, BB,
                           SPIRVEntry::createUnique(ExtSetKind, ExtOp).get()),
            BB),
        Dec);
  }

  Function *Callee = CI->getCalledFunction();
  if (Callee->isDeclaration()) {
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName().str()
                    << ": call to an undefined function " << *CI << '\n');
    joinFPContract(CI->getFunction(), FPContract::DISABLED);
  } else {
    FPContract CalleeFPC = getFPContract(Callee);
    joinFPContract(CI->getFunction(), CalleeFPC);
    if (CalleeFPC == FPContract::DISABLED) {
      SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName().str()
                      << ": call to a function with disabled contraction: "
                      << *CI << '\n');
    }
  }

  return BM->addCallInst(
      transFunctionDecl(Callee),
      transArguments(CI, BB, SPIRVEntry::createUnique(OpFunctionCall).get()),
      BB);
}

SPIRVValue *LLVMToSPIRVBase::transIndirectCallInst(CallInst *CI,
                                                   SPIRVBasicBlock *BB) {
  if (BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_function_pointers),
          SPIRVEC_FunctionPointers, CI)) {
    return BM->addIndirectCallInst(
        transValue(CI->getCalledOperand(), BB), transScavengedType(CI),
        transArguments(CI, BB, SPIRVEntry::createUnique(OpFunctionCall).get()),
        BB);
  }
  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transAsmINTEL(InlineAsm *IA) {
  assert(IA);

  // TODO: intention here is to provide information about actual target
  //       but in fact spir-64 is substituted as triple when translator works
  //       eventually we need to fix it (not urgent)
  StringRef TripleStr(M->getTargetTriple());
  auto *AsmTarget = static_cast<SPIRVAsmTargetINTEL *>(
      BM->getOrAddAsmTargetINTEL(TripleStr.str()));
  auto *SIA = BM->addAsmINTEL(
      static_cast<SPIRVTypeFunction *>(transType(IA->getFunctionType())),
      AsmTarget, IA->getAsmString(), IA->getConstraintString());
  if (IA->hasSideEffects())
    SIA->addDecorate(DecorationSideEffectsINTEL);
  return SIA;
}

SPIRVValue *LLVMToSPIRVBase::transAsmCallINTEL(CallInst *CI,
                                               SPIRVBasicBlock *BB) {
  assert(CI);
  auto *IA = cast<InlineAsm>(CI->getCalledOperand());
  return BM->addAsmCallINTELInst(
      static_cast<SPIRVAsmINTEL *>(transValue(IA, BB, false)),
      transArguments(CI, BB, SPIRVEntry::createUnique(OpAsmCallINTEL).get()),
      BB);
}

bool LLVMToSPIRVBase::transAddressingMode() {
  Triple TargetTriple(M->getTargetTriple());

  if (TargetTriple.isArch32Bit())
    BM->setAddressingModel(AddressingModelPhysical32);
  else
    BM->setAddressingModel(AddressingModelPhysical64);
  // Physical addressing model requires Addresses capability
  BM->addCapability(CapabilityAddresses);
  return true;
}
std::vector<SPIRVValue *>
LLVMToSPIRVBase::transValue(const std::vector<Value *> &Args,
                            SPIRVBasicBlock *BB) {
  std::vector<SPIRVValue *> BArgs;
  for (auto &I : Args)
    BArgs.push_back(transValue(I, BB));
  return BArgs;
}

std::vector<SPIRVWord>
LLVMToSPIRVBase::transValue(const std::vector<Value *> &Args,
                            SPIRVBasicBlock *BB, SPIRVEntry *Entry) {
  std::vector<SPIRVWord> Operands;
  for (size_t I = 0, E = Args.size(); I != E; ++I) {
    Operands.push_back(Entry->isOperandLiteral(I)
                           ? cast<ConstantInt>(Args[I])->getZExtValue()
                           : transValue(Args[I], BB)->getId());
  }
  return Operands;
}

std::vector<SPIRVWord> LLVMToSPIRVBase::transArguments(CallInst *CI,
                                                       SPIRVBasicBlock *BB,
                                                       SPIRVEntry *Entry) {
  return transValue(getArguments(CI), BB, Entry);
}

SPIRVWord LLVMToSPIRVBase::transFunctionControlMask(Function *F) {
  SPIRVWord FCM = 0;
  SPIRSPIRVFuncCtlMaskMap::foreach (
      [&](Attribute::AttrKind Attr, SPIRVFunctionControlMaskKind Mask) {
        if (F->hasFnAttribute(Attr)) {
          if (Attr == Attribute::OptimizeNone) {
            if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_optnone))
              return;
            BM->addExtension(ExtensionID::SPV_INTEL_optnone);
            BM->addCapability(internal::CapabilityOptNoneINTEL);
          }
          FCM |= Mask;
        }
      });
  return FCM;
}

void LLVMToSPIRVBase::transGlobalAnnotation(GlobalVariable *V) {
  SPIRVDBG(dbgs() << "[transGlobalAnnotation] " << *V << '\n');

  // @llvm.global.annotations is an array that contains structs with 4 fields.
  // Get the array of structs with metadata
  // TODO: actually, now it contains 5 fields, the fifth by default is nullptr
  // or undef, but it can be defined to include variadic arguments of
  // clang::annotation attribute. Need to refactor this function to turn on this
  // translation
  ConstantArray *CA = cast<ConstantArray>(V->getOperand(0));
  for (Value *Op : CA->operands()) {
    ConstantStruct *CS = cast<ConstantStruct>(Op);
    // The first field of the struct contains a pointer to annotated variable
    Value *AnnotatedVar = CS->getOperand(0)->stripPointerCasts();
    SPIRVValue *SV = transValue(AnnotatedVar, nullptr);

    // The second field contains a pointer to a global annotation string
    GlobalVariable *GV =
        cast<GlobalVariable>(CS->getOperand(1)->stripPointerCasts());

    StringRef AnnotationString;
    if (!getConstantStringInfo(GV, AnnotationString)) {
      assert(!"Annotation string missing");
      return;
    }
    DecorationsInfoVec Decorations =
        tryParseAnnotationString(BM, AnnotationString).MemoryAttributesVec;

    // If we didn't find any annotation decorations, let's add the whole
    // annotation string as UserSemantic Decoration
    if (Decorations.empty()) {
      SV->addDecorate(
          new SPIRVDecorateUserSemanticAttr(SV, AnnotationString.str()));
    } else {
      addAnnotationDecorations(SV, Decorations);
    }
  }
}

void LLVMToSPIRVBase::transGlobalIOPipeStorage(GlobalVariable *V, MDNode *IO) {
  SPIRVDBG(dbgs() << "[transGlobalIOPipeStorage] " << *V << '\n');
  SPIRVValue *SV = transValue(V, nullptr);
  assert(SV && "Failed to process OCL PipeStorage object");
  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_io_pipes)) {
    unsigned ID = getMDOperandAsInt(IO, 0);
    SV->addDecorate(DecorationIOPipeStorageINTEL, ID);
  }
}

bool LLVMToSPIRVBase::transGlobalVariables() {
  for (auto I = M->global_begin(), E = M->global_end(); I != E; ++I) {
    if ((*I).getName() == "llvm.global.annotations")
      transGlobalAnnotation(&(*I));
    else if ([I]() -> bool {
               // Check if the GV is used only in var/ptr instructions. If yes -
               // skip processing of this since it's only an annotation GV.
               if (I->user_empty())
                 return false;
               for (auto *U : I->users()) {
                 Value *V = U;
                 while (isa<BitCastInst>(V) || isa<AddrSpaceCastInst>(V))
                   V = cast<CastInst>(V)->getOperand(0);
                 auto *GEP = dyn_cast_or_null<GetElementPtrInst>(V);
                 if (!GEP)
                   return false;
                 for (auto *GEPU : GEP->users()) {
                   auto *II = dyn_cast<IntrinsicInst>(GEPU);
                   if (!II)
                     return false;
                   switch (II->getIntrinsicID()) {
                   case Intrinsic::var_annotation:
                   case Intrinsic::ptr_annotation:
                     continue;
                   default:
                     return false;
                   }
                 }
               }
               return true;
             }())
      continue;
    else if ((I->getName() == "llvm.global_ctors" ||
              I->getName() == "llvm.global_dtors") &&
             !BM->isAllowedToUseExtension(
                 ExtensionID::SPV_INTEL_function_pointers)) {
      // Function pointers are required to represent structor lists; do not
      // translate the variable if function pointers are not available.
      continue;
    } else if (MDNode *IO = ((*I).getMetadata("io_pipe_id")))
      transGlobalIOPipeStorage(&(*I), IO);
    else if (!transValue(&(*I), nullptr))
      return false;
  }
  return true;
}

bool LLVMToSPIRVBase::isAnyFunctionReachableFromFunction(
    const Function *FS,
    const std::unordered_set<const Function *> Funcs) const {
  std::unordered_set<const Function *> Done;
  std::unordered_set<const Function *> ToDo;
  ToDo.insert(FS);

  while (!ToDo.empty()) {
    auto It = ToDo.begin();
    const Function *F = *It;

    if (Funcs.find(F) != Funcs.end())
      return true;

    ToDo.erase(It);
    Done.insert(F);

    const CallGraphNode *FN = (*CG)[F];
    for (unsigned I = 0; I < FN->size(); ++I) {
      const CallGraphNode *NN = (*FN)[I];
      const Function *NNF = NN->getFunction();
      if (!NNF)
        continue;
      if (Done.find(NNF) == Done.end()) {
        ToDo.insert(NNF);
      }
    }
  }

  return false;
}

std::vector<SPIRVId>
LLVMToSPIRVBase::collectEntryPointInterfaces(SPIRVFunction *SF, Function *F) {
  std::vector<SPIRVId> Interface;
  for (auto &GV : M->globals()) {
    const auto AS = GV.getAddressSpace();
    SPIRVModule *BM = SF->getModule();
    if (!BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4))
      if (AS != SPIRAS_Input && AS != SPIRAS_Output)
        continue;

    std::unordered_set<const Function *> Funcs;

    for (const auto &U : GV.uses()) {
      const Instruction *Inst = dyn_cast<Instruction>(U.getUser());
      if (!Inst)
        continue;
      Funcs.insert(Inst->getFunction());
    }

    if (isAnyFunctionReachableFromFunction(F, Funcs)) {
      SPIRVWord ModuleVersion = static_cast<SPIRVWord>(BM->getSPIRVVersion());
      if (AS != SPIRAS_Input && AS != SPIRAS_Output &&
          ModuleVersion < static_cast<SPIRVWord>(VersionNumber::SPIRV_1_4))
        BM->setMinSPIRVVersion(VersionNumber::SPIRV_1_4);
      Interface.push_back(ValueMap[&GV]->getId());
    }
  }
  return Interface;
}

void LLVMToSPIRVBase::mutateFuncArgType(
    const std::unordered_map<unsigned, Type *> &ChangedType, Function *F) {
  for (auto &I : ChangedType) {
    for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE; ++UI) {
      auto *Call = dyn_cast<CallInst>(*UI);
      if (!Call)
        continue;
      auto *Arg = Call->getArgOperand(I.first);
      auto *OrigTy = Arg->getType();
      if (OrigTy == I.second)
        continue;
      SPIRVDBG(dbgs() << "[mutate arg type] " << *Call << ", " << *Arg << '\n');
      auto CastF = M->getOrInsertFunction(SPCV_CAST, I.second, OrigTy);
      std::vector<Value *> Args;
      Args.push_back(Arg);
      auto *Cast = CallInst::Create(CastF, Args, "", Call->getIterator());
      Call->replaceUsesOfWith(Arg, Cast);
      SPIRVDBG(dbgs() << "[mutate arg type] -> " << *Cast << '\n');
    }
  }
}

// Propagate contraction requirement of F up the call graph.
void LLVMToSPIRVBase::fpContractUpdateRecursive(Function *F, FPContract FPC) {
  std::queue<User *> Users;
  for (User *FU : F->users()) {
    Users.push(FU);
  }

  bool EnableLogger = FPC == FPContract::DISABLED && !Users.empty();
  if (EnableLogger) {
    SPIRVDBG(dbgs() << "[fp-contract] disabled for users of " << F->getName()
                    << '\n');
  }

  while (!Users.empty()) {
    User *U = Users.front();
    Users.pop();

    if (EnableLogger) {
      SPIRVDBG(dbgs() << "[fp-contract]   user: " << *U << '\n');
    }

    // Move from an Instruction to its Function
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Users.push(I->getFunction());
      continue;
    }

    if (Function *F = dyn_cast<Function>(U)) {
      if (!joinFPContract(F, FPC)) {
        // FP contract was not updated - no need to propagate
        // This also terminates a recursion (if any).
        if (EnableLogger) {
          SPIRVDBG(dbgs() << "[fp-contract] already disabled " << F->getName()
                          << '\n');
        }
        continue;
      }
      if (EnableLogger) {
        SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                        << '\n');
      }
      for (User *FU : F->users()) {
        Users.push(FU);
      }
      continue;
    }

    // Unwrap a constant until we reach an Instruction.
    // This is checked after the Function, because a Function is also a
    // Constant.
    if (Constant *C = dyn_cast<Constant>(U)) {
      for (User *CU : C->users()) {
        Users.push(CU);
      }
      continue;
    }

    llvm_unreachable("Unexpected use.");
  }
}

/// Returns a range that traverses \p F ensuring that dominator blocks are
/// visited before the blocks they dominate.
///
/// Compared to llvm::ReversePostOrderTraversal which also visits dominators
/// before dominated blocks, this traversal aims to be more stable and will keep
/// basic blocks in their original order as much as possible, only reordering
/// them to visit dominators ahead of their dominated blocks when needed. \p DT
/// is not copied by this function and needs to outlive any iterators created
/// from this range.
static auto stablePreDominatorTraversal(Function &F, const DominatorTree &DT) {

  // A local iterator type for traversing a function in the desired order.
  class StablePreDominatorIterator
      : public iterator_facade_base<StablePreDominatorIterator,
                                    std::forward_iterator_tag, BasicBlock> {

    // The passed DominatorTree; may be unset for end iterators.
    const DominatorTree *DT = nullptr;

    // The set of basic blocks already visited in this traversal.
    SmallPtrSet<const BasicBlock *, 4> VisitedBBs;

    // The next basic block in original function order, or nullptr if the
    // traversal is over.
    BasicBlock *NextBB = nullptr;

    // The current basic block in the traversal, or nullptr for end iterators.
    BasicBlock *CurBB = nullptr;

    // Returns the most immediate dominator of \p BB which does not have an
    // unvisited dominator and so can be visited in this traversal.
    BasicBlock *visitableDominator(BasicBlock *BB) const {

      // Find BB's dominator; if there is none, BB can be visited immediately.
      const auto *const BBNode = DT->getNode(BB);
      if (!BBNode)
        return BB;
      const auto *const DomNode = BBNode->getIDom();
      if (!DomNode)
        return BB;
      BasicBlock *const Dominator = DomNode->getBlock();

      // If the dominator's been visited, BB can now be visited.
      if (VisitedBBs.contains(Dominator))
        return BB;

      // Otherwise, find the dominator's visitable dominator instead.
      return visitableDominator(Dominator);
    }

    // Advances the iterator and returns the next basic block to be visited in
    // the traversal.
    BasicBlock *next() {

      // If NextBB is nullptr, the end of the traversal has been reached.
      if (!NextBB)
        return nullptr;

      // Check if NextBB has already been visited; if so, advance past it.
      if (VisitedBBs.contains(NextBB)) {
        NextBB = NextBB->getNextNode();
        return next();
      }

      // If NextBB is unvisited, visit its next visitable dominator.
      BasicBlock *const ToVisit = visitableDominator(NextBB);
      VisitedBBs.insert(ToVisit);
      return ToVisit;
    }

  public:
    // Constructs an end iterator.
    StablePreDominatorIterator() {}

    // Constructs a begin iterator at the start of \p F.
    StablePreDominatorIterator(Function &F, const DominatorTree &DT)
        : DT(&DT), NextBB(&F.getEntryBlock()) {
      ++*this;
    }

    // Methods required by iterator_facade_base.
    bool operator==(const StablePreDominatorIterator &Other) const {
      return CurBB == Other.CurBB;
    }
    BasicBlock &operator*() const { return *CurBB; }
    StablePreDominatorIterator &operator++() {
      CurBB = next();
      return *this;
    }
  };

  return make_range(StablePreDominatorIterator(F, DT),
                    StablePreDominatorIterator());
}

void LLVMToSPIRVBase::transFunction(Function *I) {
  SPIRVFunction *BF = transFunctionDecl(I);
  // Creating all basic blocks before creating any instruction. SPIR-V requires
  // that blocks appear after their dominators, so stablePreDominatorTraversal
  // is used to ensure blocks are written in the right order.
  const DominatorTree DT(*I);
  for (BasicBlock &FI : stablePreDominatorTraversal(*I, DT)) {
    FI.convertFromNewDbgValues();
    transValue(&FI, nullptr);
  }
  for (auto &FI : *I) {
    SPIRVBasicBlock *BB =
        static_cast<SPIRVBasicBlock *>(transValue(&FI, nullptr));
    for (auto &BI : FI) {
      transValue(&BI, BB, false);
    }
  }
  // Enable FP contraction unless proven otherwise
  joinFPContract(I, FPContract::ENABLED);
  fpContractUpdateRecursive(I, getFPContract(I));

  if (isKernel(I)) {
    auto Interface = collectEntryPointInterfaces(BF, I);
    BM->addEntryPoint(ExecutionModelKernel, BF->getId(), I->getName().str(),
                      Interface);
  }
}

bool isEmptyLLVMModule(Module *M) {
  return M->empty() &&      // No functions
         M->global_empty(); // No global variables
}

bool LLVMToSPIRVBase::translate() {
  BM->setGeneratorVer(KTranslatorVer);

  if (isEmptyLLVMModule(M))
    BM->addCapability(CapabilityLinkage);

  if (!lowerBuiltinCallsToVariables(M))
    return false;

  // Use the type scavenger to recover pointer element types.
  Scavenger = std::make_unique<SPIRVTypeScavenger>(*M);

  if (!transSourceLanguage())
    return false;
  if (!transExtension())
    return false;
  if (!transBuiltinSet())
    return false;
  if (!transAddressingMode())
    return false;
  if (!transGlobalVariables())
    return false;

  for (auto &F : *M) {
    auto *FT = F.getFunctionType();
    std::unordered_map<unsigned, Type *> ChangedType;
    oclGetMutatedArgumentTypesByBuiltin(FT, ChangedType, &F);
    mutateFuncArgType(ChangedType, &F);
  }

  // SPIR-V logical layout requires all function declarations go before
  // function definitions.
  std::vector<Function *> Decls, Defs;
  for (auto &F : *M) {
    if (isBuiltinTransToInst(&F) || isBuiltinTransToExtInst(&F) ||
        F.getName().starts_with(SPCV_CAST) ||
        F.getName().starts_with(LLVM_MEMCPY) ||
        F.getName().starts_with(SAMPLER_INIT))
      continue;
    if (F.isDeclaration())
      Decls.push_back(&F);
    else
      Defs.push_back(&F);
  }
  for (auto *I : Decls)
    transFunctionDecl(I);
  for (auto *I : Defs)
    transFunction(I);

  if (!transMetadata())
    return false;
  if (!transExecutionMode())
    return false;

  BM->resolveUnknownStructFields();
  DbgTran->transDebugMetadata();
  return true;
}

llvm::IntegerType *LLVMToSPIRVBase::getSizetType(unsigned AS) {
  return IntegerType::getIntNTy(M->getContext(),
                                M->getDataLayout().getPointerSizeInBits(AS));
}

void LLVMToSPIRVBase::oclGetMutatedArgumentTypesByBuiltin(
    llvm::FunctionType *FT, std::unordered_map<unsigned, Type *> &ChangedType,
    Function *F) {
  StringRef Demangled;
  if (!oclIsBuiltin(F->getName(), Demangled))
    return;
  // Note: kSPIRVName::ConvertHandleToSampledImageINTEL contains
  // kSPIRVName::SampledImage as a substring, but we still want to return in
  // this case.
  if (Demangled.find(kSPIRVName::SampledImage) == std::string::npos ||
      Demangled.find(kSPIRVName::ConvertHandleToSampledImageINTEL) !=
          std::string::npos)
    return;
  if (FT->getParamType(1)->isIntegerTy())
    ChangedType[1] = getSPIRVType(OpTypeSampler, true);
}

SPIRVValue *LLVMToSPIRVBase::transBuiltinToConstant(StringRef DemangledName,
                                                    CallInst *CI) {
  Op OC = getSPIRVFuncOC(DemangledName);
  if (!isSpecConstantOpCode(OC))
    return nullptr;
  if (OC == spv::OpSpecConstantComposite) {
    return BM->addSpecConstantComposite(transType(CI->getType()),
                                        transValue(getArguments(CI), nullptr));
  }
  Value *V = CI->getArgOperand(1);
  Type *Ty = CI->getType();
  assert(((Ty == V->getType()) ||
          // If bool is stored into memory, then clang will emit it as i8,
          // however for other usages of bool (like return type of a function),
          // it is emitted as i1.
          // Therefore, situation when we encounter
          // i1 _Z20__spirv_SpecConstant(i32, i8) is valid
          (Ty->isIntegerTy(1) && V->getType()->isIntegerTy(8))) &&
         "Type mismatch!");
  uint64_t Val = 0;
  if (Ty->isIntegerTy())
    Val = cast<ConstantInt>(V)->getZExtValue();
  else if (Ty->isFloatingPointTy())
    Val = cast<ConstantFP>(V)->getValueAPF().bitcastToAPInt().getZExtValue();
  else
    return nullptr;
  SPIRVValue *SC = BM->addSpecConstant(transType(Ty), Val);
  return SC;
}

SPIRVInstruction *LLVMToSPIRVBase::transBuiltinToInst(StringRef DemangledName,
                                                      CallInst *CI,
                                                      SPIRVBasicBlock *BB) {
  SmallVector<std::string, 2> Dec;
  auto OC = getSPIRVFuncOC(DemangledName, &Dec);

  if (OC == OpNop)
    return nullptr;

  if (OpReadPipeBlockingINTEL <= OC && OC <= OpWritePipeBlockingINTEL &&
      !BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_blocking_pipes))
    return nullptr;

  if (OpFixedSqrtINTEL <= OC && OC <= OpFixedExpINTEL)
    BM->getErrorLog().checkError(
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_arbitrary_precision_fixed_point),
        SPIRVEC_InvalidInstruction,
        CI->getCalledOperand()->getName().str() +
            "\nFixed point instructions can't be translated correctly without "
            "enabled SPV_INTEL_arbitrary_precision_fixed_point extension!\n");

  if ((OpArbitraryFloatSinCosPiINTEL <= OC &&
       OC <= OpArbitraryFloatCastToIntINTEL) ||
      (OpArbitraryFloatAddINTEL <= OC && OC <= OpArbitraryFloatPowNINTEL))
    BM->getErrorLog().checkError(
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_arbitrary_precision_floating_point),
        SPIRVEC_InvalidInstruction,
        CI->getCalledOperand()->getName().str() +
            "\nFloating point instructions can't be translated correctly "
            "without enabled SPV_INTEL_arbitrary_precision_floating_point "
            "extension!\n");

  auto *Inst = transBuiltinToInstWithoutDecoration(OC, CI, BB);
  addDecorations(Inst, Dec);
  return Inst;
}

bool LLVMToSPIRVBase::transExecutionMode() {
  if (auto NMD = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::ExecutionMode)) {
    while (!NMD.atEnd()) {
      unsigned EMode = ~0U;
      Function *F = nullptr;
      auto N = NMD.nextOp(); /* execution mode MDNode */
      N.get(F).get(EMode);

      SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
      assert(BF && "Invalid kernel function");
      if (!BF)
        return false;

      auto AddSingleArgExecutionMode = [&](ExecutionMode EMode) {
        uint32_t Arg = ~0u;
        N.get(Arg);
        BF->addExecutionMode(
            BM->add(new SPIRVExecutionMode(OpExecutionMode, BF, EMode, Arg)));
      };

      switch (EMode) {
      case spv::ExecutionModeContractionOff:
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            OpExecutionMode, BF, static_cast<ExecutionMode>(EMode))));
        break;
      case spv::ExecutionModeInitializer:
      case spv::ExecutionModeFinalizer:
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1)) {
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              OpExecutionMode, BF, static_cast<ExecutionMode>(EMode))));
        } else {
          getErrorLog().checkError(false, SPIRVEC_Requires1_1,
                                   "Initializer/Finalizer Execution Mode");
          return false;
        }
        break;
      case spv::ExecutionModeLocalSize:
      case spv::ExecutionModeLocalSizeHint: {
        unsigned X = 0, Y = 0, Z = 0;
        N.get(X).get(Y).get(Z);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            OpExecutionMode, BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
      } break;
      case spv::ExecutionModeMaxWorkgroupSizeINTEL: {
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes)) {
          unsigned X = 0, Y = 0, Z = 0;
          N.get(X).get(Y).get(Z);
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              OpExecutionMode, BF, static_cast<ExecutionMode>(EMode), X, Y,
              Z)));
          BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
          BM->addCapability(CapabilityKernelAttributesINTEL);
        }
      } break;
      case spv::ExecutionModeNoGlobalOffsetINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes))
          break;
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            OpExecutionMode, BF, static_cast<ExecutionMode>(EMode))));
        BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
        BM->addCapability(CapabilityKernelAttributesINTEL);
      } break;
      case spv::ExecutionModeVecTypeHint:
      case spv::ExecutionModeSubgroupSize:
      case spv::ExecutionModeSubgroupsPerWorkgroup:
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        break;
      case spv::ExecutionModeNumSIMDWorkitemsINTEL:
      case spv::ExecutionModeSchedulerTargetFmaxMhzINTEL:
      case spv::ExecutionModeMaxWorkDimINTEL:
      case spv::ExecutionModeRegisterMapInterfaceINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
        BM->addCapability(CapabilityFPGAKernelAttributesINTEL);
        // RegisterMapInterfaceINTEL mode is defined by the
        // CapabilityFPGAKernelAttributesv2INTEL capability and that
        // capability implicitly defines CapabilityFPGAKernelAttributesINTEL
        if (EMode == spv::ExecutionModeRegisterMapInterfaceINTEL)
          BM->addCapability(CapabilityFPGAKernelAttributesv2INTEL);
      } break;
      case spv::ExecutionModeStreamingInterfaceINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
        BM->addCapability(CapabilityFPGAKernelAttributesINTEL);
      } break;
      case spv::ExecutionModeSharedLocalMemorySizeINTEL: {
        if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
      } break;
      case spv::ExecutionModeNamedBarrierCountINTEL: {
        if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
          break;
        unsigned NBarrierCnt = 0;
        N.get(NBarrierCnt);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            OpExecutionMode, BF, static_cast<ExecutionMode>(EMode),
            NBarrierCnt)));
        BM->addExtension(ExtensionID::SPV_INTEL_vector_compute);
        BM->addCapability(CapabilityVectorComputeINTEL);
      } break;

      case spv::ExecutionModeDenormPreserve:
      case spv::ExecutionModeDenormFlushToZero:
      case spv::ExecutionModeSignedZeroInfNanPreserve:
      case spv::ExecutionModeRoundingModeRTE:
      case spv::ExecutionModeRoundingModeRTZ: {
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
          BM->setMinSPIRVVersion(VersionNumber::SPIRV_1_4);
          AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        } else if (BM->isAllowedToUseExtension(
                       ExtensionID::SPV_KHR_float_controls)) {
          BM->addExtension(ExtensionID::SPV_KHR_float_controls);
          AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        }
      } break;
      case spv::ExecutionModeRoundingModeRTPINTEL:
      case spv::ExecutionModeRoundingModeRTNINTEL:
      case spv::ExecutionModeFloatingPointModeALTINTEL:
      case spv::ExecutionModeFloatingPointModeIEEEINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_float_controls2))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
      } break;
      case spv::internal::ExecutionModeFastCompositeKernelINTEL: {
        if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fast_composite))
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              OpExecutionMode, BF, static_cast<ExecutionMode>(EMode))));
      } break;
      case spv::internal::ExecutionModeNamedSubgroupSizeINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_subgroup_requirements))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
      } break;
      default:
        llvm_unreachable("invalid execution mode");
      }
    }
  }

  transFPContract();

  return true;
}

void LLVMToSPIRVBase::transFPContract() {
  FPContractMode Mode = BM->getFPContractMode();

  for (Function &F : *M) {
    SPIRVValue *TranslatedF = getTranslatedValue(&F);
    if (!TranslatedF) {
      continue;
    }
    SPIRVFunction *BF = static_cast<SPIRVFunction *>(TranslatedF);

    bool IsKernelEntryPoint =
        BF->getModule()->isEntryPoint(spv::ExecutionModelKernel, BF->getId());
    if (!IsKernelEntryPoint)
      continue;

    FPContract FPC = getFPContract(&F);
    assert(FPC != FPContract::UNDEF);

    bool DisableContraction = false;
    switch (Mode) {
    case FPContractMode::Fast:
      DisableContraction = false;
      break;
    case FPContractMode::On:
      DisableContraction = FPC == FPContract::DISABLED;
      break;
    case FPContractMode::Off:
      DisableContraction = true;
      break;
    }

    if (DisableContraction) {
      BF->addExecutionMode(BF->getModule()->add(new SPIRVExecutionMode(
          OpExecutionMode, BF, spv::ExecutionModeContractionOff)));
    }
  }
}

bool LLVMToSPIRVBase::transMetadata() {
  if (!transOCLMetadata())
    return false;

  auto Model = getMemoryModel(*M);
  if (Model != SPIRVMemoryModelKind::MemoryModelMax)
    BM->setMemoryModel(static_cast<SPIRVMemoryModelKind>(Model));

  return true;
}

// Work around to translate kernel_arg_type and kernel_arg_type_qual metadata
static void transKernelArgTypeMD(SPIRVModule *BM, Function *F, MDNode *MD,
                                 std::string MDName) {
  std::string KernelArgTypesMDStr =
      std::string(MDName) + "." + F->getName().str() + ".";
  for (const auto &TyOp : MD->operands())
    KernelArgTypesMDStr += cast<MDString>(TyOp)->getString().str() + ",";
  BM->getString(KernelArgTypesMDStr);
}

bool LLVMToSPIRVBase::transOCLMetadata() {
  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(&F));
    assert(BF && "Kernel function should be translated first");

    // Create 'OpString' as a workaround to store information about
    // *orignal* (typedef'ed, unsigned integers) type names of kernel arguments.
    // OpString "kernel_arg_type.%kernel_name%.typename0,typename1,..."
    if (auto *KernelArgType = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE))
      if (BM->shouldPreserveOCLKernelArgTypeMetadataThroughString())
        transKernelArgTypeMD(BM, &F, KernelArgType, SPIR_MD_KERNEL_ARG_TYPE);

    if (auto *KernelArgTypeQual = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE_QUAL)) {
      foreachKernelArgMD(
          KernelArgTypeQual, BF,
          [](const std::string &Str, SPIRVFunctionParameter *BA) {
            if (Str.find("volatile") != std::string::npos)
              BA->addDecorate(new SPIRVDecorate(DecorationVolatile, BA));
            if (Str.find("restrict") != std::string::npos)
              BA->addDecorate(
                  new SPIRVDecorate(DecorationFuncParamAttr, BA,
                                    FunctionParameterAttributeNoAlias));
          });
      // Create 'OpString' as a workaround to store information about
      // constant qualifiers of pointer kernel arguments. Store empty string
      // for a non constant parameter.
      // OpString "kernel_arg_type_qual.%kernel_name%.qual0,qual1,..."
      if (BM->shouldPreserveOCLKernelArgTypeMetadataThroughString())
        transKernelArgTypeMD(BM, &F, KernelArgTypeQual,
                             SPIR_MD_KERNEL_ARG_TYPE_QUAL);
    }
    if (auto *KernelArgName = F.getMetadata(SPIR_MD_KERNEL_ARG_NAME)) {
      foreachKernelArgMD(
          KernelArgName, BF,
          [=](const std::string &Str, SPIRVFunctionParameter *BA) {
            BM->setName(BA, Str);
          });
    }
    if (auto *KernArgDecoMD = F.getMetadata(SPIRV_MD_PARAMETER_DECORATIONS))
      foreachKernelArgMD(KernArgDecoMD, BF, transMetadataDecorations);
  }
  return true;
}

bool LLVMToSPIRVBase::transSourceLanguage() {
  auto Src = getSPIRVSource(M);
  SrcLang = std::get<0>(Src);
  SrcLangVer = std::get<1>(Src);
  BM->setSourceLanguage(static_cast<SourceLanguage>(SrcLang), SrcLangVer);
  return true;
}

bool LLVMToSPIRVBase::transExtension() {
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::Extension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      ExtensionID ExtID = SPIRVMap<ExtensionID, std::string>::rmap(S);
      if (!BM->getErrorLog().checkError(BM->isAllowedToUseExtension(ExtID),
                                        SPIRVEC_RequiresExtension, S))
        return false;
      BM->getExtension().insert(S);
    }
  }
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::SourceExtension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      BM->getSourceExtension().insert(S);
    }
  }
  for (auto &I :
       map<SPIRVCapabilityKind>(rmap<OclExt::Kind>(BM->getExtension())))
    BM->addCapability(I);

  return true;
}

void LLVMToSPIRVBase::dumpUsers(Value *V) {
  SPIRVDBG(dbgs() << "Users of " << *V << " :\n");
  for (auto UI = V->user_begin(), UE = V->user_end(); UI != UE; ++UI)
    SPIRVDBG(dbgs() << "  " << **UI << '\n');
}

Op LLVMToSPIRVBase::transBoolOpCode(SPIRVValue *Opn, Op OC) {
  if (!Opn->getType()->isTypeVectorOrScalarBool())
    return OC;
  IntBoolOpMap::find(OC, &OC);
  return OC;
}

SPIRVInstruction *
LLVMToSPIRVBase::transBuiltinToInstWithoutDecoration(Op OC, CallInst *CI,
                                                     SPIRVBasicBlock *BB) {
  // OpAtomicCompareExchangeWeak is not "weak" at all,
  // but instead has the same semantics as OpAtomicCompareExchange.
  // Moreover, OpAtomicCompareExchangeWeak has been deprecated.
  if (OC == OpAtomicCompareExchangeWeak)
    OC = OpAtomicCompareExchange;

  // We should do this replacement only for SPIR-V 1.5, as OpLessOrGreater is
  // deprecated there. However we do such replacement for the usual pipeline
  // (not via SPIR-V friendly calls) without minding the version, so we can do
  // such thing here as well.
  if (OC == OpLessOrGreater &&
      BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_5))
    OC = OpFOrdNotEqual;

  if (isGroupOpCode(OC))
    BM->addCapability(CapabilityGroups);
  switch (static_cast<size_t>(OC)) {
  case OpControlBarrier: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addControlBarrierInst(BArgs[0], BArgs[1], BArgs[2], BB);
  } break;
  case OpGroupAsyncCopy: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addAsyncGroupCopy(BArgs[0], BArgs[1], BArgs[2], BArgs[3],
                                 BArgs[4], BArgs[5], BB);
  } break;
  case OpVectorExtractDynamic: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addVectorExtractDynamicInst(BArgs[0], BArgs[1], BB);
  } break;
  case OpVectorInsertDynamic: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addVectorInsertDynamicInst(BArgs[0], BArgs[1], BArgs[2], BB);
  } break;
  case OpSampledImage: {
    // Clang can generate SPIRV-friendly call for OpSampledImage instruction,
    // i.e. __spirv_SampledImage... But it can't generate correct return type
    // for this call, because there is no support for type corresponding to
    // OpTypeSampledImage. So, in this case, we create the required type here.
    Value *Image = CI->getArgOperand(0);
    Type *SampledImgTy =
        adjustImageType(getCallValueType(CI, 0), kSPIRVTypeName::Image,
                        kSPIRVTypeName::SampledImg);
    Value *Sampler = CI->getArgOperand(1);
    return BM->addSampledImageInst(transType(SampledImgTy),
                                   transValue(Image, BB),
                                   transValue(Sampler, BB), BB);
  }
  case OpImageRead:
  case OpImageSampleExplicitLod:
  case OpImageWrite: {
    // Image Op needs handling of SignExtend or ZeroExtend image operands.
    auto Args = getArguments(CI);
    SPIRVType *SPRetTy =
        CI->getType()->isVoidTy() ? nullptr : transScavengedType(CI);
    auto *SPI = SPIRVInstTemplateBase::create(OC);
    std::vector<SPIRVWord> SPArgs;
    for (size_t I = 0, E = Args.size(); I != E; ++I) {
      if (Args[I]->getType()->isPointerTy()) {
        [[maybe_unused]] Value *Pointee = Args[I]->stripPointerCasts();
        assert((Pointee == Args[I] || !isa<Function>(Pointee)) &&
               "Illegal use of a function pointer type");
      }
      SPArgs.push_back(SPI->isOperandLiteral(I)
                           ? cast<ConstantInt>(Args[I])->getZExtValue()
                           : transValue(Args[I], BB)->getId());
    }
    if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
      size_t ImOpIdx = getImageOperandsIndex(OC);
      if (Args.size() > ImOpIdx) {
        // Update existing ImageOperands with SignExtendMask/ZeroExtendMask.
        if (auto *ImOp = dyn_cast<ConstantInt>(Args[ImOpIdx])) {
          uint64_t ImOpVal = ImOp->getZExtValue();
          unsigned SignZeroExtMasks =
              ImageOperandsMask::ImageOperandsSignExtendMask |
              ImageOperandsMask::ImageOperandsZeroExtendMask;
          if (!(ImOpVal & SignZeroExtMasks))
            if (int SignZeroExt = getImageSignZeroExt(CI->getCalledFunction()))
              SPArgs[ImOpIdx] = ImOpVal | SignZeroExt;
        }
      } else {
        // Add new ImageOperands argument.
        if (int SignZeroExt = getImageSignZeroExt(CI->getCalledFunction()))
          SPArgs.push_back(SignZeroExt);
      }
    }
    BM->addInstTemplate(SPI, SPArgs, BB, SPRetTy);
    return SPI;
  }
  case OpFixedSqrtINTEL:
  case OpFixedRecipINTEL:
  case OpFixedRsqrtINTEL:
  case OpFixedSinINTEL:
  case OpFixedCosINTEL:
  case OpFixedSinCosINTEL:
  case OpFixedSinPiINTEL:
  case OpFixedCosPiINTEL:
  case OpFixedSinCosPiINTEL:
  case OpFixedLogINTEL:
  case OpFixedExpINTEL: {
    // LLVM fixed point functions return value:
    // iN (arbitrary precision integer of N bits length)
    // Arguments:
    // A(iN), S(i1), I(i32), rI(i32), Quantization(i32), Overflow(i32)
    // where A - integer input of any width.

    // SPIR-V fixed point instruction contains:
    // <id>ResTy Res<id> In<id> \
    // Literal S Literal I Literal rI Literal Q Literal O

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = CI->getParamStructRetType(0);
      OpItr++;
    }

    // Input - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *Input = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      Input = BM->addLoadInst(Input, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst =
        BM->addFixedPointIntelInst(OC, transType(ResTy), Input, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  case OpArbitraryFloatCastINTEL:
  case OpArbitraryFloatCastFromIntINTEL:
  case OpArbitraryFloatCastToIntINTEL:
  case OpArbitraryFloatRecipINTEL:
  case OpArbitraryFloatRSqrtINTEL:
  case OpArbitraryFloatCbrtINTEL:
  case OpArbitraryFloatSqrtINTEL:
  case OpArbitraryFloatLogINTEL:
  case OpArbitraryFloatLog2INTEL:
  case OpArbitraryFloatLog10INTEL:
  case OpArbitraryFloatLog1pINTEL:
  case OpArbitraryFloatExpINTEL:
  case OpArbitraryFloatExp2INTEL:
  case OpArbitraryFloatExp10INTEL:
  case OpArbitraryFloatExpm1INTEL:
  case OpArbitraryFloatSinINTEL:
  case OpArbitraryFloatCosINTEL:
  case OpArbitraryFloatSinCosINTEL:
  case OpArbitraryFloatSinPiINTEL:
  case OpArbitraryFloatCosPiINTEL:
  case OpArbitraryFloatSinCosPiINTEL:
  case OpArbitraryFloatASinINTEL:
  case OpArbitraryFloatASinPiINTEL:
  case OpArbitraryFloatACosINTEL:
  case OpArbitraryFloatACosPiINTEL:
  case OpArbitraryFloatATanINTEL:
  case OpArbitraryFloatATanPiINTEL: {
    // Format of instruction CastFromInt:
    //   LLVM arbitrary floating point functions return value type:
    //       iN (arbitrary precision integer of N bits length)
    //   Arguments: A(iN), Mout(i32), FromSign(bool), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal Mout Literal FromSign
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instruction CastToInt:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), ToSign(bool), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA Literal ToSign
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of other instructions:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), Mout(i32), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = CI->getParamStructRetType(0);
      OpItr++;
    }

    // Input - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *Input = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      Input = BM->addLoadInst(Input, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst = BM->addArbFloatPointIntelInst(OC, transType(ResTy), Input,
                                                    nullptr, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  case OpArbitraryFloatAddINTEL:
  case OpArbitraryFloatSubINTEL:
  case OpArbitraryFloatMulINTEL:
  case OpArbitraryFloatDivINTEL:
  case OpArbitraryFloatGTINTEL:
  case OpArbitraryFloatGEINTEL:
  case OpArbitraryFloatLTINTEL:
  case OpArbitraryFloatLEINTEL:
  case OpArbitraryFloatEQINTEL:
  case OpArbitraryFloatHypotINTEL:
  case OpArbitraryFloatATan2INTEL:
  case OpArbitraryFloatPowINTEL:
  case OpArbitraryFloatPowRINTEL:
  case OpArbitraryFloatPowNINTEL: {
    // Format of instructions Add, Sub, Mul, Div, Hypot, ATan2, Pow, PowR:
    //   LLVM arbitrary floating point functions return value:
    //       iN (arbitrary precision integer of N bits length)
    //   Arguments: A(iN), MA(i32), B(iN), MB(i32), Mout(i32),
    //              EnableSubnormals(i32), RoundingMode(i32),
    //              RoundingAccuracy(i32)
    //   where A, B and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal MB Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instruction PowN:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), B(iN), SignOfB(i1), Mout(i32),
    //              EnableSubnormals(i32), RoundingMode(i32),
    //              RoundingAccuracy(i32)
    //   where A, B and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal SignOfB Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instructions GT, GE, LT, LE, EQ:
    //   LLVM arbitrary floating point functions return value: Bool
    //   Arguments: A(iN), MA(i32), B(iN), MB(i32)
    //   where A and B are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal MB

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = CI->getParamStructRetType(0);
      OpItr++;
    }

    // InA - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *InA = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      InA = BM->addLoadInst(InA, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    Literals.push_back(cast<llvm::ConstantInt>(*OpItr++)->getZExtValue());

    // InB - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *InB = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy()) {
      std::vector<SPIRVWord> Mem;
      InB = BM->addLoadInst(InB, Mem, BB);
    }
    OpItr++;

    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst = BM->addArbFloatPointIntelInst(OC, transType(ResTy), InA,
                                                    InB, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  case internal::OpTaskSequenceGetINTEL: {
    Type *ResTy = nullptr;
    auto OpItr = CI->value_op_begin();

    if (CI->hasStructRetAttr()) {
      assert(CI->getType()->isVoidTy() && "Return type is not void");
      ResTy = CI->getParamStructRetType(0);
      OpItr++;
    }

    SPIRVType *RetTy = ResTy ? transType(ResTy) : transScavengedType(CI);
    auto *TaskSeqGet =
        BM->addTaskSequenceGetINTELInst(RetTy, transValue(*OpItr++, BB), BB);

    if (!CI->hasStructRetAttr())
      return TaskSeqGet;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), TaskSeqGet,
                            {}, BB);
  }
  case OpLoad: {
    std::vector<SPIRVWord> MemoryAccess;
    assert(CI->arg_size() > 0 && "Expected at least 1 operand for OpLoad call");
    for (size_t I = 1; I < CI->arg_size(); ++I)
      MemoryAccess.push_back(
          cast<ConstantInt>(CI->getArgOperand(I))->getZExtValue());
    return BM->addLoadInst(transValue(CI->getArgOperand(0), BB), MemoryAccess,
                           BB);
  }
  case OpStore: {
    std::vector<SPIRVWord> MemoryAccess;
    assert(CI->arg_size() > 1 &&
           "Expected at least 2 operands for OpStore call");
    for (size_t I = 2; I < CI->arg_size(); ++I)
      MemoryAccess.push_back(
          cast<ConstantInt>(CI->getArgOperand(I))->getZExtValue());
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB),
                            transValue(CI->getArgOperand(1), BB), MemoryAccess,
                            BB);
  }
  case OpCompositeConstruct: {
    std::vector<SPIRVId> Operands = {
        transValue(CI->getArgOperand(0), BB)->getId()};
    return BM->addCompositeConstructInst(transType(CI->getType()), Operands,
                                         BB);
  }
  case OpIAddCarry: {
    Function *F = CI->getCalledFunction();
    StructType *St = cast<StructType>(F->getParamStructRetType(0));
    SPIRVValue *V = BM->addBinaryInst(OpIAddCarry, transType(St),
                                      transValue(CI->getArgOperand(1), BB),
                                      transValue(CI->getArgOperand(2), BB), BB);
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), V, {}, BB);
  }
  case OpISubBorrow: {
    Function *F = CI->getCalledFunction();
    StructType *St = cast<StructType>(F->getParamStructRetType(0));
    SPIRVValue *V = BM->addBinaryInst(OpISubBorrow, transType(St),
                                      transValue(CI->getArgOperand(1), BB),
                                      transValue(CI->getArgOperand(2), BB), BB);
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), V, {}, BB);
  }
  case OpGroupNonUniformShuffleDown: {
    Function *F = CI->getCalledFunction();
    if (F->arg_size() && F->getArg(0)->hasStructRetAttr()) {
      StructType *St = cast<StructType>(F->getParamStructRetType(0));
      assert(isSYCLHalfType(St) || isSYCLBfloat16Type(St));
      SPIRVValue *InValue =
          transValue(CI->getArgOperand(0)->stripPointerCasts(), BB);
      SPIRVId ScopeId = transValue(CI->getArgOperand(1), BB)->getId();
      SPIRVValue *Delta = transValue(CI->getArgOperand(3), BB);
      SPIRVValue *Composite0 = BM->addLoadInst(InValue, {}, BB);
      Type *MemberTy = St->getElementType(0);
      SPIRVType *ElementTy = transType(MemberTy);
      SPIRVValue *Element0 =
          BM->addCompositeExtractInst(ElementTy, Composite0, {0}, BB);
      SPIRVValue *Src =
          BM->addGroupInst(OpGroupNonUniformShuffleDown, ElementTy,
                           static_cast<Scope>(ScopeId), {Element0, Delta}, BB);
      SPIRVValue *Composite1 =
          BM->addCompositeInsertInst(Src, Composite0, {0}, BB);
      return BM->addStoreInst(InValue, Composite1, {}, BB);
    }
    [[fallthrough]];
  }
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit) {
      return BM->addUnaryInst(OC, transScavengedType(CI),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else if (isCmpOpCode(OC) || isUnaryPredicateOpCode(OC)) {
      auto *ResultTy = CI->getType();
      Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
      auto IsVector = ResultTy->isVectorTy();
      if (IsVector)
        BoolTy = FixedVectorType::get(
            BoolTy, cast<FixedVectorType>(ResultTy)->getNumElements());
      auto *BBT = transType(BoolTy);
      SPIRVInstruction *Res;
      if (isCmpOpCode(OC)) {
        assert(CI && CI->arg_size() == 2 && "Invalid call inst");
        Res = BM->addCmpInst(OC, BBT, transValue(CI->getArgOperand(0), BB),
                             transValue(CI->getArgOperand(1), BB), BB);
      } else {
        assert(CI && CI->arg_size() == 1 && "Invalid call inst");
        Res =
            BM->addUnaryInst(OC, BBT, transValue(CI->getArgOperand(0), BB), BB);
      }
      // OpenCL C and OpenCL C++ built-ins may have different return type
      if (ResultTy == BoolTy)
        return Res;
      assert(IsVector || (!IsVector && ResultTy->isIntegerTy(32)));
      auto *Zero = transValue(Constant::getNullValue(ResultTy), BB);
      auto *One = transValue(
          IsVector ? Constant::getAllOnesValue(ResultTy) : getInt32(M, 1), BB);
      return BM->addSelectInst(Res, One, Zero, BB);
    } else if (isBinaryOpCode(OC)) {
      assert(CI && CI->arg_size() == 2 && "Invalid call inst");
      return BM->addBinaryInst(OC, transScavengedType(CI),
                               transValue(CI->getArgOperand(0), BB),
                               transValue(CI->getArgOperand(1), BB), BB);
    } else if (CI->arg_size() == 1 && !CI->getType()->isVoidTy() &&
               !hasExecScope(OC) && !isAtomicOpCode(OC)) {
      return BM->addUnaryInst(OC, transScavengedType(CI),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else {
      auto Args = getArguments(CI);
      SPIRVType *SPRetTy = nullptr;
      Type *RetTy = CI->getType();
      auto *F = CI->getCalledFunction();
      if (!RetTy->isVoidTy()) {
        SPRetTy = transScavengedType(CI);
      } else if (Args.size() > 0 && F->arg_begin()->hasStructRetAttr()) {
        SPRetTy = transType(F->getParamStructRetType(0));
        Args.erase(Args.begin());
      }
      auto *SPI = SPIRVInstTemplateBase::create(OC);
      std::vector<SPIRVWord> SPArgs;
      for (size_t I = 0, E = Args.size(); I != E; ++I) {
        if (Args[I]->getType()->isPointerTy()) {
          Value *Pointee = Args[I]->stripPointerCasts();
          (void)Pointee;
          assert((Pointee == Args[I] || !isa<Function>(Pointee)) &&
                 "Illegal use of a function pointer type");
        }
        SPArgs.push_back(SPI->isOperandLiteral(I)
                             ? cast<ConstantInt>(Args[I])->getZExtValue()
                             : transValue(Args[I], BB)->getId());
      }
      BM->addInstTemplate(SPI, SPArgs, BB, SPRetTy);
      if (!SPRetTy || !SPRetTy->isTypeStruct())
        return SPI;
      std::vector<SPIRVWord> Mem;
      SPIRVDBG(spvdbgs() << *SPI << '\n');
      return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), SPI, Mem,
                              BB);
    }
  }
  }
  return nullptr;
}

SPIRV::SPIRVLinkageTypeKind
LLVMToSPIRVBase::transLinkageType(const GlobalValue *GV) {
  if (GV->isDeclarationForLinker())
    return SPIRVLinkageTypeKind::LinkageTypeImport;
  if (GV->hasInternalLinkage() || GV->hasPrivateLinkage())
    return spv::internal::LinkageTypeInternal;
  if (GV->hasLinkOnceODRLinkage())
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_linkonce_odr))
      return SPIRVLinkageTypeKind::LinkageTypeLinkOnceODR;
  return SPIRVLinkageTypeKind::LinkageTypeExport;
}

LLVMToSPIRVBase::FPContract LLVMToSPIRVBase::getFPContract(Function *F) {
  auto It = FPContractMap.find(F);
  if (It == FPContractMap.end()) {
    return FPContract::UNDEF;
  }
  return It->second;
}

bool LLVMToSPIRVBase::joinFPContract(Function *F, FPContract C) {
  FPContract &Existing = FPContractMap[F];
  switch (Existing) {
  case FPContract::UNDEF:
    if (C != FPContract::UNDEF) {
      Existing = C;
      return true;
    }
    return false;
  case FPContract::ENABLED:
    if (C == FPContract::DISABLED) {
      Existing = C;
      return true;
    }
    return false;
  case FPContract::DISABLED:
    return false;
  }
  llvm_unreachable("Unhandled FPContract value.");
}

} // namespace SPIRV

char LLVMToSPIRVLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(LLVMToSPIRVLegacy, "llvmtospv",
                      "Translate LLVM to SPIR-V", false, false)
INITIALIZE_PASS_DEPENDENCY(OCLTypeToSPIRVLegacy)
INITIALIZE_PASS_END(LLVMToSPIRVLegacy, "llvmtospv", "Translate LLVM to SPIR-V",
                    false, false)

ModulePass *llvm::createLLVMToSPIRVLegacy(SPIRVModule *SMod) {
  return new LLVMToSPIRVLegacy(SMod);
}

void addPassesForSPIRV(ModulePassManager &PassMgr,
                       const SPIRV::TranslatorOpts &Opts) {
  if (Opts.isSPIRVMemToRegEnabled())
    PassMgr.addPass(createModuleToFunctionPassAdaptor(PromotePass()));
  PassMgr.addPass(PreprocessMetadataPass());
  PassMgr.addPass(SPIRVLowerOCLBlocksPass());
  PassMgr.addPass(OCLToSPIRVPass());
  PassMgr.addPass(SPIRVRegularizeLLVMPass());
  PassMgr.addPass(SPIRVLowerConstExprPass());
  PassMgr.addPass(SPIRVLowerBoolPass());
  PassMgr.addPass(SPIRVLowerMemmovePass());
  PassMgr.addPass(SPIRVLowerLLVMIntrinsicPass(Opts));
  PassMgr.addPass(createModuleToFunctionPassAdaptor(
      SPIRVLowerBitCastToNonStandardTypePass(Opts)));
}

bool isValidLLVMModule(Module *M, SPIRVErrorLog &ErrorLog) {
  if (!M)
    return false;

  if (isEmptyLLVMModule(M))
    return true;

  Triple TT(M->getTargetTriple());
  if (!ErrorLog.checkError(isSupportedTriple(TT), SPIRVEC_InvalidTargetTriple,
                           "Actual target triple is " + M->getTargetTriple()))
    return false;

  return true;
}

namespace {

VersionNumber getVersionFromTriple(const Triple &TT, SPIRVErrorLog &ErrorLog) {
  switch (TT.getSubArch()) {
  case Triple::SPIRVSubArch_v10:
    return VersionNumber::SPIRV_1_0;
  case Triple::SPIRVSubArch_v11:
    return VersionNumber::SPIRV_1_1;
  case Triple::SPIRVSubArch_v12:
    return VersionNumber::SPIRV_1_2;
  case Triple::SPIRVSubArch_v13:
    return VersionNumber::SPIRV_1_3;
  case Triple::SPIRVSubArch_v14:
    return VersionNumber::SPIRV_1_4;
  case Triple::SPIRVSubArch_v15:
    return VersionNumber::SPIRV_1_5;
  default:
    ErrorLog.checkError(false, SPIRVEC_InvalidSubArch, TT.getArchName().str());
    return VersionNumber::MaximumVersion;
  }
}

bool runSpirvWriterPasses(Module *M, std::ostream *OS, std::string &ErrMsg,
                          const SPIRV::TranslatorOpts &Opts) {
  // Perform the conversion and write the resulting SPIR-V if an ostream has
  // been given; otherwise only perform regularization.
  bool WriteSpirv = OS != nullptr;

  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule(Opts));
  if (!isValidLLVMModule(M, BM->getErrorLog()))
    return false;

  // If the module carries a SPIR-V triple with a version subarch, target
  // that SPIR-V version.
  Triple TargetTriple(M->getTargetTriple());
  if ((TargetTriple.getArch() == Triple::spirv32 ||
       TargetTriple.getArch() == Triple::spirv64) &&
      TargetTriple.getSubArch() != Triple::NoSubArch) {
    VersionNumber ModuleVer =
        getVersionFromTriple(TargetTriple, BM->getErrorLog());
    if (!BM->getErrorLog().checkError(ModuleVer <= Opts.getMaxVersion(),
                                      SPIRVEC_TripleMaxVersionIncompatible))
      return false;
    BM->setMinSPIRVVersion(ModuleVer);
    BM->setMaxSPIRVVersion(ModuleVer);
  }

  ModulePassManager PassMgr;
  addPassesForSPIRV(PassMgr, Opts);
  if (WriteSpirv) {
    // Run loop simplify pass in order to avoid duplicate OpLoopMerge
    // instruction. It can happen in case of continue operand in the loop.
    if (hasLoopMetadata(M))
      PassMgr.addPass(createModuleToFunctionPassAdaptor(LoopSimplifyPass()));
    PassMgr.addPass(LLVMToSPIRVPass(BM.get()));
  }

  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  MAM.registerPass([&] { return OCLTypeToSPIRVPass(); });
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  PassMgr.run(*M, MAM);

  if (BM->getError(ErrMsg) != SPIRVEC_Success)
    return false;

  if (WriteSpirv)
    *OS << *BM;

  return true;
}

} // namespace

bool llvm::writeSpirv(Module *M, std::ostream &OS, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  // To preserve old behavior of the translator, let's enable all extensions
  // by default in this API
  DefaultOpts.enableAllExtensions();
  return llvm::writeSpirv(M, DefaultOpts, OS, ErrMsg);
}

bool llvm::writeSpirv(Module *M, const SPIRV::TranslatorOpts &Opts,
                      std::ostream &OS, std::string &ErrMsg) {
  return runSpirvWriterPasses(M, &OS, ErrMsg, Opts);
}

bool llvm::regularizeLlvmForSpirv(Module *M, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  // To preserve old behavior of the translator, let's enable all extensions
  // by default in this API
  DefaultOpts.enableAllExtensions();
  return llvm::regularizeLlvmForSpirv(M, ErrMsg, DefaultOpts);
}

bool llvm::regularizeLlvmForSpirv(Module *M, std::string &ErrMsg,
                                  const SPIRV::TranslatorOpts &Opts) {
  return runSpirvWriterPasses(M, nullptr, ErrMsg, Opts);
}
