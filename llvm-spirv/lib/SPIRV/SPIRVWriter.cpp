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
#include "SPIRVAsm.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVLLVMUtil.h"
#include "SPIRVMDWalker.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "VectorComputeUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h" // loop-simplify pass

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

namespace SPIRV {

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(const std::string &Str, SPIRVFunctionParameter *BA)>
        Func) {
  for (unsigned I = 0, E = MD->getNumOperands(); I != E; ++I) {
    SPIRVFunctionParameter *BA = BF->getArgument(I);
    Func(getMDOperandAsString(MD, I), BA);
  }
}

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(Metadata *MDOp, SPIRVFunctionParameter *BA)> Func) {
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
    : M(nullptr), Ctx(nullptr), BM(SMod), SrcLang(0), SrcLangVer(0) {
  DbgTran = std::make_unique<LLVMToSPIRVDbgTran>(nullptr, SMod, this);
}

LLVMToSPIRVBase::~LLVMToSPIRVBase() {
  for (auto *I : UnboundInst)
    I->deleteValue();
}

bool LLVMToSPIRVBase::runLLVMToSPIRV(Module &Mod) {
  M = &Mod;
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
  if (!S.startswith(kSPIRVName::Prefix))
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

    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      Type *ElTy = PtrTy->getPointerElementType();
      if (auto *FTy = dyn_cast<FunctionType>(ElTy)) {
        // If we have a function pointer, then argument types and return type of
        // the referenced function also need to be checked
        return Run(FTy->getReturnType()) ||
               any_of(FTy->param_begin(), FTy->param_end(), Run);
      }

      return Run(ElTy);
    }

    if (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
      return Run(ArrayTy->getArrayElementType());

    return false;
  };

  return Run(Ty);
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
    auto ET = T->getPointerElementType();
    if (ET->isFunctionTy() &&
        !BM->checkExtension(ExtensionID::SPV_INTEL_function_pointers,
                            SPIRVEC_FunctionPointers, toString(T)))
      return nullptr;
    auto ST = dyn_cast<StructType>(ET);
    auto AddrSpc = T->getPointerAddressSpace();
    // Lower global_device and global_host address spaces that were added in
    // SYCL as part of SYCL_INTEL_usm_address_spaces extension to just global
    // address space if device doesn't support SPV_INTEL_usm_storage_classes
    // extension
    if (!BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_usm_storage_classes) &&
        ((AddrSpc == SPIRAS_GlobalDevice) || (AddrSpc == SPIRAS_GlobalHost))) {
      auto NewType =
          PointerType::get(T->getPointerElementType(), SPIRAS_Global);
      return mapType(T, transType(NewType));
    }
    if (ST && !ST->isSized()) {
      Op OpCode;
      StringRef STName = ST->getName();
      // Workaround for non-conformant SPIR binary
      if (STName == "struct._event_t") {
        STName = kSPR2TypeName::Event;
        ST->setName(STName);
      }
      if (STName.startswith(kSPR2TypeName::PipeRO) ||
          STName.startswith(kSPR2TypeName::PipeWO)) {
        auto PipeT = BM->addPipeType();
        PipeT->setPipeAcessQualifier(STName.startswith(kSPR2TypeName::PipeRO)
                                         ? AccessQualifierReadOnly
                                         : AccessQualifierWriteOnly);
        return mapType(T, PipeT);
      }
      if (STName.startswith(kSPR2TypeName::ImagePrefix)) {
        assert(AddrSpc == SPIRAS_Global);
        auto SPIRVImageTy = getSPIRVImageTypeFromOCL(M, T);
        return mapType(T, transType(SPIRVImageTy));
      }
      if (STName == kSPR2TypeName::Sampler)
        return mapType(T, transType(getSamplerType(M)));
      if (STName.startswith(kSPIRVTypeName::PrefixAndDelim))
        return transSPIRVOpaqueType(T);

      if (STName.startswith(kOCLSubgroupsAVCIntel::TypePrefix))
        return mapType(
            T, BM->addSubgroupAvcINTELType(
                   OCLSubgroupINTELTypeOpCodeMap::map(ST->getName().str())));

      if (OCLOpaqueTypeOpCodeMap::find(STName.str(), &OpCode)) {
        switch (OpCode) {
        default:
          return mapType(T, BM->addOpaqueGenericType(OpCode));
        case OpTypeDeviceEvent:
          return mapType(T, BM->addDeviceEventType());
        case OpTypeQueue:
          return mapType(T, BM->addQueueType());
        }
      }
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute)) {
        if (STName.startswith(kVCType::VCBufferSurface)) {
          // VCBufferSurface always have Access Qualifier
          auto Access = getAccessQualifier(STName);
          return mapType(T, BM->addBufferSurfaceINTELType(Access));
        }
      }

      if (isPointerToOpaqueStructType(T)) {
        return mapType(
            T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                      static_cast<SPIRAddressSpace>(AddrSpc)),
                                  transType(ET)));
      }
    } else {
      // JointMatrixINTEL type is not necessarily an opaque type, it can be
      // represented as a structure with pointer to a multidimensional array
      // member.
      if (ST && ST->hasName()) {
        StringRef STName = ST->getName();
        if (STName.startswith(kSPIRVTypeName::PrefixAndDelim)) {
          SmallVector<std::string, 8> Postfixes;
          auto TN = decodeSPIRVTypeName(STName, Postfixes);
          if (TN == kSPIRVTypeName::JointMatrixINTEL) {
            return transSPIRVJointMatrixINTELType(T, Postfixes);
          }
        }
      }
      SPIRVType *ElementType = transType(ET);
      // ET, as a recursive type, may contain exactly the same pointer T, so it
      // may happen that after translation of ET we already have translated T,
      // added the translated pointer to the SPIR-V module and mapped T to this
      // pointer. Now we have to check TypeMap again.
      LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
      if (Loc != TypeMap.end()) {
        return Loc->second;
      }
      return mapType(
          T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                    static_cast<SPIRAddressSpace>(AddrSpc)),
                                ElementType));
    }
  }

  if (auto *VecTy = dyn_cast<FixedVectorType>(T))
    return mapType(T, BM->addVectorType(transType(VecTy->getElementType()),
                                        VecTy->getNumElements()));

  if (T->isArrayTy()) {
    // SPIR-V 1.3 s3.32.6: Length is the number of elements in the array.
    //                     It must be at least 1.
    if (T->getArrayNumElements() < 1) {
      std::string Str;
      llvm::raw_string_ostream OS(Str);
      OS << *T;
      SPIRVCK(T->getArrayNumElements() >= 1, InvalidArraySize, OS.str());
    }
    return mapType(T, BM->addArrayType(
                          transType(T->getArrayElementType()),
                          static_cast<SPIRVConstant *>(transValue(
                              ConstantInt::get(getSizetType(),
                                               T->getArrayNumElements(), false),
                              nullptr))));
  }

  if (T->isStructTy() && !T->isSized()) {
    auto ST = dyn_cast<StructType>(T);
    (void)ST; // Silence warning
    assert(!ST->getName().startswith(kSPR2TypeName::PipeRO));
    assert(!ST->getName().startswith(kSPR2TypeName::PipeWO));
    assert(!ST->getName().startswith(kSPR2TypeName::ImagePrefix));
    return mapType(T, BM->addOpaqueType(T->getStructName().str()));
  }

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    StringRef Name;
    if (ST->hasName())
      Name = ST->getName();

    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
      return transType(getSamplerType(M));
    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
      return transType(getPipeStorageType(M));

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
    return mapType(T, BM->addFunctionType(RT, PT));
  }

  llvm_unreachable("Not implemented!");
  return 0;
}

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
SPIRVType *LLVMToSPIRVBase::transSPIRVJointMatrixINTELType(
    Type *T, SmallVector<std::string, 8> Postfixes) {
  Type *ElemTy = nullptr;
  StringRef Ty{Postfixes[0]};
  auto NumBits = llvm::StringSwitch<unsigned>(Ty)
                     .Case("char", 8)
                     .Case("short", 16)
                     .Case("int", 32)
                     .Case("long", 64)
                     .Default(0);
  if (NumBits)
    ElemTy = IntegerType::get(M->getContext(), NumBits);
  else if (Ty == "half")
    ElemTy = Type::getHalfTy(M->getContext());
  else if (Ty == "float")
    ElemTy = Type::getFloatTy(M->getContext());
  else if (Ty == "double")
    ElemTy = Type::getDoubleTy(M->getContext());
  else
    llvm_unreachable("Unexpected type for matrix!");

  auto ParseInteger = [this](StringRef Postfix) -> ConstantInt * {
    unsigned long long N = 0;
    consumeUnsignedInteger(Postfix, 10, N);
    return getUInt32(M, N);
  };
  SPIRVValue *Rows = transConstant(ParseInteger(Postfixes[1]));
  SPIRVValue *Columns = transConstant(ParseInteger(Postfixes[2]));
  SPIRVValue *Layout = transConstant(ParseInteger(Postfixes[3]));
  SPIRVValue *Scope = transConstant(ParseInteger(Postfixes[4]));
  return mapType(T, BM->addJointMatrixINTELType(transType(ElemTy), Rows,
                                                Columns, Layout, Scope));
}

SPIRVType *LLVMToSPIRVBase::transSPIRVOpaqueType(Type *T) {
  auto ET = T->getPointerElementType();
  auto ST = cast<StructType>(ET);
  auto STName = ST->getStructName();
  assert(STName.startswith(kSPIRVTypeName::PrefixAndDelim) &&
         "Invalid SPIR-V opaque type name");
  SmallVector<std::string, 8> Postfixes;
  auto TN = decodeSPIRVTypeName(STName, Postfixes);
  if (TN == kSPIRVTypeName::Pipe) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);
    assert(Postfixes.size() == 1 && "Invalid pipe type ops");
    auto PipeT = BM->addPipeType();
    PipeT->setPipeAcessQualifier(
        static_cast<spv::AccessQualifier>(atoi(Postfixes[0].c_str())));
    return mapType(T, PipeT);
  } else if (TN == kSPIRVTypeName::Image) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);
    // The sampled type needs to be translated through LLVM type to guarantee
    // uniqueness.
    auto SampledT = transType(
        getLLVMTypeForSPIRVImageSampledTypePostfix(Postfixes[0], *Ctx));
    SmallVector<int, 7> Ops;
    for (unsigned I = 1; I < 8; ++I)
      Ops.push_back(atoi(Postfixes[I].c_str()));
    SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]),
                                  Ops[1], Ops[2], Ops[3], Ops[4], Ops[5]);
    return mapType(T,
                   BM->addImageType(SampledT, Desc,
                                    static_cast<spv::AccessQualifier>(Ops[6])));
  } else if (TN == kSPIRVTypeName::SampledImg) {
    return mapType(
        T, BM->addSampledImageType(static_cast<SPIRVTypeImage *>(
               transType(getSPIRVTypeByChangeBaseTypeName(
                   M, T, kSPIRVTypeName::SampledImg, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::VmeImageINTEL) {
    // This type is the same as SampledImageType, but consumed by Subgroup AVC
    // Intel extension instructions.
    return mapType(
        T,
        BM->addVmeImageINTELType(static_cast<SPIRVTypeImage *>(
            transType(getSPIRVTypeByChangeBaseTypeName(
                M, T, kSPIRVTypeName::VmeImageINTEL, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::Sampler)
    return mapType(T, BM->addSamplerType());
  else if (TN == kSPIRVTypeName::DeviceEvent)
    return mapType(T, BM->addDeviceEventType());
  else if (TN == kSPIRVTypeName::Queue)
    return mapType(T, BM->addQueueType());
  else if (TN == kSPIRVTypeName::PipeStorage)
    return mapType(T, BM->addPipeStorageType());
  else if (TN == kSPIRVTypeName::JointMatrixINTEL) {
    return transSPIRVJointMatrixINTELType(T, Postfixes);
  } else
    return mapType(T,
                   BM->addOpaqueGenericType(SPIRVOpaqueTypeOpCodeMap::map(TN)));
}

SPIRVFunction *LLVMToSPIRVBase::transFunctionDecl(Function *F) {
  if (auto BF = getTranslatedValue(F))
    return static_cast<SPIRVFunction *>(BF);

  if (F->isIntrinsic() && (!BM->isSPIRVAllowUnknownIntrinsicsEnabled() ||
                           isKnownIntrinsic(F->getIntrinsicID()))) {
    // We should not translate LLVM intrinsics as a function
    assert(std::none_of(F->user_begin(), F->user_end(),
                        [this](User *U) { return getTranslatedValue(U); }) &&
           "LLVM intrinsics shouldn't be called in SPIRV");
    return nullptr;
  }

  SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
      transType(OCLTypeToSPIRVPtr->getAdaptedType(F)));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName())
    BM->setName(BF, F->getName().str());
  if (isKernel(F))
    BM->addEntryPoint(ExecutionModelKernel, BF->getId());
  else if (F->getLinkage() != GlobalValue::InternalLinkage)
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
    if (I->onlyReadsMemory())
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

  SPIRVDBG(dbgs() << "[transFunction] " << *F << " => ";
           spvdbgs() << *BF << '\n';)
  return BF;
}

void LLVMToSPIRVBase::transVectorComputeMetadata(Function *F) {
  using namespace VectorComputeUtil;
  if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    return;
  auto BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
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

void LLVMToSPIRVBase::transFPGAFunctionMetadata(SPIRVFunction *BF,
                                                Function *F) {
  if (MDNode *StallEnable = F->getMetadata(kSPIR2MD::StallEnable)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_cluster_attributes)) {
      if (getMDOperandAsInt(StallEnable, 0))
        BF->addDecorate(new SPIRVDecorateStallEnableINTEL(BF));
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
  if (MDNode *DisableLoopPipelining =
          F->getMetadata(kSPIR2MD::DisableLoopPipelining)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      size_t Disable = getMDOperandAsInt(DisableLoopPipelining, 0);
      BF->addDecorate(new SPIRVDecoratePipelineEnableINTEL(BF, !Disable));
    }
  }
}

SPIRVValue *LLVMToSPIRVBase::transConstant(Value *V) {
  if (auto CPNull = dyn_cast<ConstantPointerNull>(V))
    return BM->addNullConstant(
        bcast<SPIRVTypePointer>(transType(CPNull->getType())));

  if (auto CAZero = dyn_cast<ConstantAggregateZero>(V)) {
    Type *AggType = CAZero->getType();
    if (const StructType *ST = dyn_cast<StructType>(AggType))
      if (ST->hasName() &&
          ST->getName() == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
        return BM->addSamplerConstant(transType(AggType), 0, 0, 0);

    return BM->addNullConstant(transType(AggType));
  }

  if (auto ConstI = dyn_cast<ConstantInt>(V)) {
    unsigned BitWidth = ConstI->getType()->getBitWidth();
    if (BitWidth > 64) {
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_integers),
          SPIRVEC_InvalidBitWidth, std::to_string(BitWidth));
      return BM->addConstant(transType(V->getType()), ConstI->getValue());
    }
    return BM->addConstant(transType(V->getType()), ConstI->getZExtValue());
  }

  if (auto ConstFP = dyn_cast<ConstantFP>(V)) {
    auto BT = static_cast<SPIRVType *>(transType(V->getType()));
    return BM->addConstant(
        BT, ConstFP->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  if (auto ConstDA = dyn_cast<ConstantDataArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDA->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDA->getElementAsConstant(I), nullptr, true,
                              FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstA = dyn_cast<ConstantArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstA->op_begin(), E = ConstA->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstDV = dyn_cast<ConstantDataVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDV->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDV->getElementAsConstant(I), nullptr, true,
                              FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstV = dyn_cast<ConstantVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
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
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstUE = dyn_cast<ConstantExpr>(V)) {
    auto Inst = ConstUE->getAsInstruction();
    SPIRVDBG(dbgs() << "ConstantExpr: " << *ConstUE << '\n';
             dbgs() << "Instruction: " << *Inst << '\n';)
    auto BI = transValue(Inst, nullptr, false);
    Inst->dropAllReferences();
    UnboundInst.push_back(Inst);
    return BI;
  }

  if (isa<UndefValue>(V)) {
    return BM->addUndef(transType(V->getType()));
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
          isa<CastInst>(V) || isa<ExtractElementInst>(V) ||
          isa<BinaryOperator>(V) || BB) &&
         "Invalid SPIRV BB");

  auto BV = transValueWithoutDecoration(V, BB, CreateForward, FuncTrans);
  if (!BV || !transDecoration(V, BV))
    return nullptr;
  StringRef Name = V->getName();
  if (!Name.empty()) // Don't erase the name, which BM might already have
    BM->setName(BV, Name.str());
  return BV;
}

SPIRVInstruction *LLVMToSPIRVBase::transBinaryInst(BinaryOperator *B,
                                                   SPIRVBasicBlock *BB) {
  unsigned LLVMOC = B->getOpcode();
  auto Op0 = transValue(B->getOperand(0), BB);
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

SPIRV::SPIRVInstruction *LLVMToSPIRVBase::transUnaryInst(UnaryInstruction *U,
                                                         SPIRVBasicBlock *BB) {
  Op BOC = OpNop;
  if (auto Cast = dyn_cast<AddrSpaceCastInst>(U)) {
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

  auto Op = transValue(U->getOperand(0), BB, true, FuncTransMode::Pointer);
  return BM->addUnaryInst(transBoolOpCode(Op, BOC), transType(U->getType()), Op,
                          BB);
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
      : Node(Node), IndexGroupArrayMap(IndexGroupArrayMap) {}

  void initialize() {
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
      std::string S = getMDOperandAsString(Node, 0);
      // Set the loop control bits. Parameters are set in the order described
      // in 3.23 SPIR-V Spec. rev. 1.4:
      // Bits that are set can indicate whether an additional operand follows,
      // as described by the table. If there are multiple following operands
      // indicated, they are ordered: Those indicated by smaller-numbered bits
      // appear first.
      if (S == "llvm.loop.unroll.disable")
        LoopControl |= spv::LoopControlDontUnrollMask;
      else if (S == "llvm.loop.unroll.full" || S == "llvm.loop.unroll.enable")
        LoopControl |= spv::LoopControlUnrollMask;
      // PartialCount must not be used with the DontUnroll bit
      else if (S == "llvm.loop.unroll.count" &&
               !(LoopControl & LoopControlDontUnrollMask)) {
        size_t I = getMDOperandAsInt(Node, 1);
        ParametersToSort.emplace_back(spv::LoopControlPartialCountMask, I);
        LoopControl |= spv::LoopControlPartialCountMask;
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
          IVDep.initialize();
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
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_max") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Max = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_avg") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Avg = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        }
      }
    }
  }
  if (LoopControl & spv::internal::LoopControlLoopCountINTELMask) {
    // LoopCountINTELMask have int64 literal parameters and we need to store
    // int64 into 2 SPIRVWords
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min >> 32));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max >> 32));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Avg));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
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
  for (auto Param : ParametersToSort)
    Parameters.push_back(Param.second);

  return static_cast<spv::LoopControlMask>(LoopControl);
}

static int transAtomicOrdering(llvm::AtomicOrdering Ordering) {
  return OCLMemOrderMap::map(
      static_cast<OCLMemOrderKind>(llvm::toCABI(Ordering)));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicStore(StoreInst *ST,
                                              SPIRVBasicBlock *BB) {
  std::vector<Value *> Ops{ST->getPointerOperand(),
                           getUInt32(M, spv::ScopeDevice),
                           getUInt32(M, transAtomicOrdering(ST->getOrdering())),
                           ST->getValueOperand()};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(ST, BM->addInstTemplate(OpAtomicStore, BM->getIds(SPIRVOps),
                                          BB, nullptr));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicLoad(LoadInst *LD,
                                             SPIRVBasicBlock *BB) {
  std::vector<Value *> Ops{
      LD->getPointerOperand(), getUInt32(M, spv::ScopeDevice),
      getUInt32(M, transAtomicOrdering(LD->getOrdering()))};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(LD, BM->addInstTemplate(OpAtomicLoad, BM->getIds(SPIRVOps),
                                          BB, transType(LD->getType())));
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
  if (auto LBB = dyn_cast<BasicBlock>(V)) {
    auto BF =
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
        transType(F->getType()),
        static_cast<SPIRVFunction *>(transValue(F, nullptr)));
  }

  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    llvm::PointerType *Ty = GV->getType();
    // Though variables with common linkage type are initialized by 0,
    // they can be represented in SPIR-V as uninitialized variables with
    // 'Export' linkage type, just as tentative definitions look in C
    llvm::Value *Init = GV->hasInitializer() && !GV->hasCommonLinkage()
                            ? GV->getInitializer()
                            : nullptr;
    SPIRVValue *BVarInit = nullptr;
    StructType *ST = Init ? dyn_cast<StructType>(Init->getType()) : nullptr;
    if (ST && ST->hasName() && isSPIRVConstantName(ST->getName())) {
      auto BV = transConstant(Init);
      assert(BV);
      return mapValue(V, BV);
    } else if (ConstantExpr *ConstUE = dyn_cast_or_null<ConstantExpr>(Init)) {
      Instruction *Inst = ConstUE->getAsInstruction();
      if (isSpecialTypeInitializer(Inst)) {
        Init = Inst->getOperand(0);
        Ty = static_cast<PointerType *>(Init->getType());
      }
      Inst->dropAllReferences();
      UnboundInst.push_back(Inst);
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
        if (auto ArrTy = dyn_cast_or_null<ArrayType>(Init->getType())) {
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
      BVarInit = transValue(Init, nullptr);
    }

    SPIRVStorageClassKind StorageClass;
    auto AddressSpace = static_cast<SPIRAddressSpace>(Ty->getAddressSpace());
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
    }

    auto BVar = static_cast<SPIRVVariable *>(
        BM->addVariable(transType(Ty), GV->isConstant(), transLinkageType(GV),
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

    BVar->setBuiltin(Builtin);
    return BVar;
  }

  if (isa<Constant>(V)) {
    auto BV = transConstant(V);
    assert(BV);
    return mapValue(V, BV);
  }

  if (auto Arg = dyn_cast<Argument>(V)) {
    unsigned ArgNo = Arg->getArgNo();
    SPIRVFunction *BF = BB->getParent();
    // assert(BF->existArgument(ArgNo));
    return mapValue(V, BF->getArgument(ArgNo));
  }

  if (CreateForward)
    return mapValue(V, BM->addForward(transType(V->getType())));

  if (StoreInst *ST = dyn_cast<StoreInst>(V)) {
    if (ST->isAtomic())
      return transAtomicStore(ST, BB);

    // Keep this vector to store MemoryAccess operands for both Alignment and
    // Aliasing information.
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    if (ST->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    if (ST->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(ST->getAlignment());
    }
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
    if (LD->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(LD->getAlignment());
    }
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

  if (auto RI = dyn_cast<ReturnInst>(V)) {
    if (auto RV = RI->getReturnValue())
      return mapValue(V, BM->addReturnValueInst(transValue(RV, BB), BB));
    return mapValue(V, BM->addReturnInst(BB));
  }

  if (CmpInst *Cmp = dyn_cast<CmpInst>(V)) {
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
    if (Alc->isArrayAllocation()) {
      if (!BM->checkExtension(ExtensionID::SPV_INTEL_variable_length_array,
                              SPIRVEC_InvalidInstruction,
                              toString(Alc) +
                                  "\nTranslation of dynamic alloca requires "
                                  "SPV_INTEL_variable_length_array extension."))
        return nullptr;

      SPIRVValue *Length = transValue(Alc->getArraySize(), BB);
      assert(Length && "Couldn't translate array size!");
      return mapValue(V, BM->addInstTemplate(OpVariableLengthArrayINTEL,
                                             {Length->getId()}, BB,
                                             transType(Alc->getType())));
    }
    return mapValue(V, BM->addVariable(transType(Alc->getType()), false,
                                       spv::internal::LinkageTypeInternal,
                                       nullptr, Alc->getName().str(),
                                       StorageClassFunction, BB));
  }

  if (auto *Switch = dyn_cast<SwitchInst>(V)) {
    std::vector<SPIRVSwitch::PairTy> Pairs;
    auto Select = transValue(Switch->getCondition(), BB);

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

  if (auto Phi = dyn_cast<PHINode>(V)) {
    std::vector<SPIRVValue *> IncomingPairs;

    for (size_t I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
      IncomingPairs.push_back(transValue(Phi->getIncomingValue(I), BB, true,
                                         FuncTransMode::Pointer));
      IncomingPairs.push_back(transValue(Phi->getIncomingBlock(I), nullptr));
    }
    return mapValue(
        V, BM->addPhiInst(transType(Phi->getType()), IncomingPairs, BB));
  }

  if (auto Ext = dyn_cast<ExtractValueInst>(V)) {
    return mapValue(V, BM->addCompositeExtractInst(
                           transType(Ext->getType()),
                           transValue(Ext->getAggregateOperand(), BB),
                           Ext->getIndices(), BB));
  }

  if (auto Ins = dyn_cast<InsertValueInst>(V)) {
    return mapValue(V, BM->addCompositeInsertInst(
                           transValue(Ins->getInsertedValueOperand(), BB),
                           transValue(Ins->getAggregateOperand(), BB),
                           Ins->getIndices(), BB));
  }

  if (UnaryInstruction *U = dyn_cast<UnaryInstruction>(V)) {
    if (isSpecialTypeInitializer(U))
      return mapValue(V, transValue(U->getOperand(0), BB));
    auto UI = transUnaryInst(U, BB);
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

    return mapValue(V, BM->addPtrAccessChainInst(transType(GEP->getType()),
                                                 TransPointerOperand, Indices,
                                                 BB, GEP->isInBounds()));
  }

  if (auto Ext = dyn_cast<ExtractElementInst>(V)) {
    auto Index = Ext->getIndexOperand();
    if (auto Const = dyn_cast<ConstantInt>(Index))
      return mapValue(V, BM->addCompositeExtractInst(
                             transType(Ext->getType()),
                             transValue(Ext->getVectorOperand(), BB),
                             std::vector<SPIRVWord>(1, Const->getZExtValue()),
                             BB));
    else
      return mapValue(V, BM->addVectorExtractDynamicInst(
                             transValue(Ext->getVectorOperand(), BB),
                             transValue(Index, BB), BB));
  }

  if (auto Ins = dyn_cast<InsertElementInst>(V)) {
    auto Index = Ins->getOperand(2);
    if (auto Const = dyn_cast<ConstantInt>(Index)) {
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

  if (auto SF = dyn_cast<ShuffleVectorInst>(V)) {
    std::vector<SPIRVWord> Comp;
    for (auto &I : SF->getShuffleMask())
      Comp.push_back(I);
    return mapValue(V, BM->addVectorShuffleInst(
                           transType(SF->getType()),
                           transValue(SF->getOperand(0), BB),
                           transValue(SF->getOperand(1), BB), Comp, BB));
  }

  if (AtomicRMWInst *ARMW = dyn_cast<AtomicRMWInst>(V)) {
    AtomicRMWInst::BinOp Op = ARMW->getOperation();
    if (!BM->getErrorLog().checkError(
            !AtomicRMWInst::isFPOperation(Op) && Op != AtomicRMWInst::Nand,
            SPIRVEC_InvalidInstruction, V,
            "Atomic " + AtomicRMWInst::getOperationName(Op).str() +
                " is not supported in SPIR-V!\n"))
      return nullptr;

    spv::Op OC = LLVMSPIRVAtomicRmwOpCodeMap::map(Op);
    AtomicOrderingCABI Ordering = llvm::toCABI(ARMW->getOrdering());
    auto MemSem = OCLMemOrderMap::map(static_cast<OCLMemOrderKind>(Ordering));
    std::vector<Value *> Operands(4);
    Operands[0] = ARMW->getPointerOperand();
    // To get the memory scope argument we might use ARMW->getSyncScopeID(), but
    // atomicrmw LLVM instruction is not aware of OpenCL(or SPIR-V) memory scope
    // enumeration. And assuming the produced SPIR-V module will be consumed in
    // an OpenCL environment, we can use the same memory scope as OpenCL atomic
    // functions that don't have memory_scope argument i.e. memory_scope_device.
    // See the OpenCL C specification p6.13.11. "Atomic Functions"
    Operands[1] = getUInt32(M, spv::ScopeDevice);
    Operands[2] = getUInt32(M, MemSem);
    Operands[3] = ARMW->getValOperand();
    std::vector<SPIRVId> Ops = BM->getIds(transValue(Operands, BB));
    SPIRVType *Ty = transType(ARMW->getType());

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
    if (auto Alias =
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
    auto Forward = static_cast<SPIRVForward *>(Loc->second);
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
    assert(NumOperands == 2 && #NAME " requires exactly 1 extra operand");     \
    auto *StrDecoEO = dyn_cast<MDString>(DecoMD->getOperand(1));               \
    assert(StrDecoEO &&#NAME " requires extra operand to be a string");        \
    Target->addDecorate(                                                       \
        new SPIRVDecorate##NAME##Attr(Target, StrDecoEO->getString().str()));  \
    break;                                                                     \
  }

#define ONE_INT_DECORATION_CASE(NAME, NAMESPACE, TYPE)                         \
  case NAMESPACE::Decoration##NAME: {                                          \
    assert(NumOperands == 2 && #NAME " requires exactly 1 extra operand");     \
    auto *IntDecoEO =                                                          \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));              \
    assert(IntDecoEO &&#NAME " requires extra operand to be an integer");      \
    Target->addDecorate(new SPIRVDecorate##NAME(                               \
        Target, static_cast<TYPE>(IntDecoEO->getZExtValue())));                \
    break;                                                                     \
  }

#define TWO_INT_DECORATION_CASE(NAME, NAMESPACE, TYPE1, TYPE2)                 \
  case NAMESPACE::Decoration##NAME: {                                          \
    assert(NumOperands == 3 && #NAME " requires exactly 2 extra operand");     \
    auto *IntDecoEO1 =                                                         \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));              \
    assert(IntDecoEO1 &&#NAME                                                  \
           " requires first extra operand to be an integer");                  \
    auto *IntDecoEO2 =                                                         \
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));              \
    assert(IntDecoEO2 &&#NAME                                                  \
           " requires second extra operand to be an integer");                 \
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

static void transMetadataDecorations(Metadata *MD, SPIRVEntry *Target) {
  auto *ArgDecoMD = dyn_cast<MDNode>(MD);
  assert(ArgDecoMD && "Decoration list must be a metadata node");
  for (unsigned I = 0, E = ArgDecoMD->getNumOperands(); I != E; ++I) {
    auto *DecoMD = dyn_cast<MDNode>(ArgDecoMD->getOperand(I));
    assert(DecoMD && "Decoration does not name metadata");
    assert(DecoMD->getNumOperands() > 0 &&
           "Decoration metadata must have at least one operand");
    auto *DecoKindConst =
        mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(0));
    assert(DecoKindConst && "First operand of decoration must be the kind");
    auto DecoKind = static_cast<Decoration>(DecoKindConst->getZExtValue());

    const size_t NumOperands = DecoMD->getNumOperands();
    switch (static_cast<size_t>(DecoKind)) {
      ONE_STRING_DECORATION_CASE(MemoryINTEL, spv)
      ONE_STRING_DECORATION_CASE(UserSemantic, spv)
      ONE_INT_DECORATION_CASE(AliasScopeINTEL, spv, SPIRVId)
      ONE_INT_DECORATION_CASE(NoAliasINTEL, spv, SPIRVId)
      ONE_INT_DECORATION_CASE(InitiationIntervalINTEL, spv::internal, SPIRVWord)
      ONE_INT_DECORATION_CASE(MaxConcurrencyINTEL, spv::internal, SPIRVWord)
      ONE_INT_DECORATION_CASE(PipelineEnableINTEL, spv::internal, SPIRVWord)
      TWO_INT_DECORATION_CASE(FunctionRoundingModeINTEL, spv, SPIRVWord,
                              FPRoundingMode);
      TWO_INT_DECORATION_CASE(FunctionDenormModeINTEL, spv, SPIRVWord,
                              FPDenormMode);
      TWO_INT_DECORATION_CASE(FunctionFloatingPointModeINTEL, spv, SPIRVWord,
                              FPOperationMode);
      TWO_INT_DECORATION_CASE(FuseLoopsInFunctionINTEL, spv, SPIRVWord,
                              SPIRVWord);
      TWO_INT_DECORATION_CASE(MathOpDSPModeINTEL, spv::internal, SPIRVWord,
                              SPIRVWord);
    case DecorationStallEnableINTEL: {
      Target->addDecorate(new SPIRVDecorateStallEnableINTEL(Target));
      break;
    }
    case DecorationMergeINTEL: {
      assert(NumOperands == 3 && "MergeINTEL requires exactly 3 extra operand");
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(1));
      assert(Name && "MergeINTEL requires first extra operand to be a string");
      auto *Direction = dyn_cast<MDString>(DecoMD->getOperand(2));
      assert(Direction &&
             "MergeINTEL requires second extra operand to be a string");
      Target->addDecorate(new SPIRVDecorateMergeINTELAttr(
          Target, Name->getString().str(), Direction->getString().str()));
      break;
    }
    case DecorationLinkageAttributes: {
      assert(NumOperands == 3 &&
             "LinkageAttributes requires exactly 3 extra operand");
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(1));
      assert(Name &&
             "LinkageAttributes requires first extra operand to be a string");
      auto *Type = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      assert(Type &&
             "LinkageAttributes requires second extra operand to be an int");
      auto TypeKind = static_cast<SPIRVLinkageTypeKind>(Type->getZExtValue());
      Target->addDecorate(new SPIRVDecorateLinkageAttr(
          Target, Name->getString().str(), TypeKind));
      break;
    }
    case spv::internal::DecorationHostAccessINTEL: {
      checkIsGlobalVar(Target, DecoKind);

      assert(NumOperands == 3 && "HostAccessINTEL requires 2 extra operands "
                                 "after the decoration kind number");
      auto *AccessMode =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      assert(AccessMode &&
             "HostAccessINTEL requires first extra operand to be an int");
      auto *Name = dyn_cast<MDString>(DecoMD->getOperand(2));
      assert(Name &&
             "HostAccessINTEL requires second extra operand to be a string");

      Target->addDecorate(new SPIRVDecorateHostAccessINTEL(
          Target, AccessMode->getZExtValue(), Name->getString().str()));
      break;
    }
    case spv::internal::DecorationInitModeINTEL: {
      checkIsGlobalVar(Target, DecoKind);
      assert(static_cast<SPIRVVariable *>(Target)->getInitializer() &&
             "InitModeINTEL only be applied to a global (module scope) "
             "variable which has an Initializer operand");

      assert(NumOperands == 2 &&
             "InitModeINTEL requires exactly 1 extra operand");
      auto *Trigger = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      assert(Trigger &&
             "InitModeINTEL requires extra operand to be an integer");

      Target->addDecorate(
          new SPIRVDecorateInitModeINTEL(Target, Trigger->getZExtValue()));
      break;
    }
    case spv::internal::DecorationImplementInCSRINTEL: {
      checkIsGlobalVar(Target, DecoKind);

      assert(NumOperands == 2 &&
             "ImplementInCSRINTEL requires exactly 1 extra operand");
      auto *Value = mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      assert(Value &&
             "ImplementInCSRINTEL requires extra operand to be an integer");

      Target->addDecorate(
          new SPIRVDecorateImplementInCSRINTEL(Target, Value->getZExtValue()));
      break;
    }
    default: {
      if (NumOperands == 1) {
        Target->addDecorate(new SPIRVDecorate(DecoKind, Target));
        break;
      }

      auto *DecoValEO1 =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(1));
      assert(DecoValEO1 &&
             "First extra operand in default decoration case must be integer.");
      if (NumOperands == 2) {
        Target->addDecorate(
            new SPIRVDecorate(DecoKind, Target, DecoValEO1->getZExtValue()));
        break;
      }

      auto *DecoValEO2 =
          mdconst::dyn_extract<ConstantInt>(DecoMD->getOperand(2));
      assert(DecoValEO2 &&
             "First extra operand in default decoration case must be integer.");
      assert(NumOperands == 3 && "At most 2 extra operands expected.");
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

  if (auto BVO = dyn_cast_or_null<OverflowingBinaryOperator>(V)) {
    if (BVO->hasNoSignedWrap()) {
      BV->setNoSignedWrap(true);
    }
    if (BVO->hasNoUnsignedWrap()) {
      BV->setNoUnsignedWrap(true);
    }
  }

  if (auto BVF = dyn_cast_or_null<FPMathOperator>(V)) {
    auto Opcode = BVF->getOpcode();
    if (Opcode == Instruction::FAdd || Opcode == Instruction::FSub ||
        Opcode == Instruction::FMul || Opcode == Instruction::FDiv ||
        Opcode == Instruction::FRem) {
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
  if (Instruction *Inst = dyn_cast<Instruction>(V))
    if (shouldTryToAddMemAliasingDecoration(Inst))
      transMemAliasingINTELDecorations(Inst, BV);

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
  if (auto AL = dyn_cast<AllocaInst>(V)) {
    BM->setAlignment(BV, AL->getAlignment());
    return true;
  }
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
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
  return true;
}

/// Transforms SPV-IR work-item builtin calls to SPIRV builtin variables.
/// e.g.
///  SPV-IR: @_Z33__spirv_BuiltInGlobalInvocationIdi(i)
///    is transformed as:
///  x = load GlobalInvocationId; extract x, i
/// e.g.
///  SPV-IR: @_Z22__spirv_BuiltInWorkDim()
///    is transformed as:
///  load WorkDim
bool LLVMToSPIRVBase::transWorkItemBuiltinCallsToVariables() {
  LLVM_DEBUG(dbgs() << "Enter transWorkItemBuiltinCallsToVariables\n");
  // Store instructions and functions that need to be removed.
  SmallVector<Value *, 16> ToRemove;
  for (auto &F : *M) {
    // Builtins should be declaration only.
    if (!F.isDeclaration())
      continue;
    StringRef DemangledName;
    if (!oclIsBuiltin(F.getName(), DemangledName))
      continue;
    LLVM_DEBUG(dbgs() << "Function demangled name: " << DemangledName << '\n');
    SmallVector<StringRef, 2> Postfix;
    // Deprefix "__spirv_"
    StringRef Name = dePrefixSPIRVName(DemangledName, Postfix);
    // Lookup SPIRV Builtin map.
    if (!SPIRVBuiltInNameMap::rfind(Name.str(), nullptr))
      continue;
    std::string BuiltinVarName = DemangledName.str();
    LLVM_DEBUG(dbgs() << "builtin variable name: " << BuiltinVarName << '\n');
    bool IsVec = F.getFunctionType()->getNumParams() > 0;
    Type *GVType =
        IsVec ? FixedVectorType::get(F.getReturnType(), 3) : F.getReturnType();
    auto *BV = new GlobalVariable(
        *M, GVType, /*isConstant=*/true, GlobalValue::ExternalLinkage, nullptr,
        BuiltinVarName, 0, GlobalVariable::NotThreadLocal, SPIRAS_Input);
    for (auto *U : F.users()) {
      auto *CI = dyn_cast<CallInst>(U);
      assert(CI && "invalid instruction");
      const DebugLoc &DLoc = CI->getDebugLoc();
      Instruction *NewValue = new LoadInst(GVType, BV, "", CI);
      if (DLoc)
        NewValue->setDebugLoc(DLoc);
      LLVM_DEBUG(dbgs() << "Transform: " << *CI << " => " << *NewValue << '\n');
      if (IsVec) {
        NewValue =
            ExtractElementInst::Create(NewValue, CI->getArgOperand(0), "", CI);
        if (DLoc)
          NewValue->setDebugLoc(DLoc);
        LLVM_DEBUG(dbgs() << *NewValue << '\n');
      }
      NewValue->takeName(CI);
      CI->replaceAllUsesWith(NewValue);
      ToRemove.push_back(CI);
    }
    ToRemove.push_back(&F);
  }
  for (auto *V : ToRemove) {
    if (auto *I = dyn_cast<Instruction>(V))
      I->eraseFromParent();
    else if (auto *F = dyn_cast<Function>(V))
      F->eraseFromParent();
    else
      llvm_unreachable("Unexpected value to remove!");
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
  llvm::Function *F = CI->getCalledFunction();
  auto FT = F->getFunctionType();
  auto RT = FT->getReturnType();
  assert(FT->getNumParams() == 1);
  assert((isSPIRVType(RT, kSPIRVTypeName::Sampler) ||
          isPointerToOpaqueStructType(RT, kSPR2TypeName::Sampler)) &&
         FT->getParamType(0)->isIntegerTy() && "Invalid sampler type");
  auto Arg = CI->getArgOperand(0);

  auto GetSamplerConstant = [&](uint64_t SamplerValue) {
    auto AddrMode = (SamplerValue & 0xE) >> 1;
    auto Param = SamplerValue & 0x1;
    auto Filter = SamplerValue ? ((SamplerValue & 0x30) >> 4) - 1 : 0;
    auto BV = BM->addSamplerConstant(transType(RT), AddrMode, Param, Filter);
    return BV;
  };

  if (auto Const = dyn_cast<ConstantInt>(Arg)) {
    // Sampler is declared as a kernel scope constant
    return GetSamplerConstant(Const->getZExtValue());
  } else if (auto Load = dyn_cast<LoadInst>(Arg)) {
    // If value of the sampler is loaded from a global constant, use its
    // initializer for initialization of the sampler.
    auto Op = Load->getPointerOperand();
    assert(isa<GlobalVariable>(Op) && "Unknown sampler pattern!");
    auto GV = cast<GlobalVariable>(Op);
    assert(GV->isConstant() ||
           GV->getType()->getPointerAddressSpace() == SPIRAS_Constant);
    auto Initializer = GV->getInitializer();
    assert(isa<ConstantInt>(Initializer) && "sampler not constant int?");
    return GetSamplerConstant(cast<ConstantInt>(Initializer)->getZExtValue());
  }
  // Sampler is a function argument
  auto BV = transValue(Arg, BB);
  assert(BV && BV->getType() == transType(RT));
  return BV;
}

using DecorationsInfoVec = std::vector<std::pair<Decoration, std::string>>;

struct AnnotationDecorations {
  DecorationsInfoVec MemoryAttributesVec;
  DecorationsInfoVec MemoryAccessesVec;
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
      ResultVec.emplace_back(DecorationBurstCoalesceINTEL, "");
    if (DontStaticallyCoalesce)
      ResultVec.emplace_back(DecorationDontStaticallyCoalesceINTEL, "");
    // Conditional values
    if (CacheSizeInfo.hasValue()) {
      ResultVec.emplace_back(DecorationCacheSizeINTEL,
                             std::to_string(CacheSizeInfo.getValue()));
    }
    if (PrefetchInfo.hasValue()) {
      ResultVec.emplace_back(DecorationPrefetchINTEL,
                             std::to_string(PrefetchInfo.getValue()));
    }
    return ResultVec;
  }

  bool BurstCoalesce = false;
  llvm::Optional<unsigned> CacheSizeInfo;
  bool DontStaticallyCoalesce = false;
  llvm::Optional<unsigned> PrefetchInfo;
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
      AnnotationString += std::to_string(CInt->getSExtValue());
    }
    for (uint32_t I = 1; I != NumOperands; ++I) {
      if (auto *CInt = dyn_cast<ConstantInt>(CStruct->getOperand(I))) {
        AnnotationString += ", ";
        AnnotationString += std::to_string(CInt->getSExtValue());
      }
    }
  }
}

// Process main var/ptr/global annotation string with the attached optional
// integer parameters
void processAnnotationString(IntrinsicInst *II, std::string &AnnotationString) {
  if (auto *GEP = dyn_cast<GetElementPtrInst>(II->getArgOperand(1))) {
    if (auto *C = dyn_cast<Constant>(GEP->getOperand(0))) {
      StringRef StrRef;
      getConstantStringInfo(C, StrRef);
      AnnotationString += StrRef.str();
    }
  }
  if (auto *Cast = dyn_cast<BitCastInst>(II->getArgOperand(4)))
    if (auto *C = dyn_cast_or_null<Constant>(Cast->getOperand(0)))
      processOptionalAnnotationInfo(C, AnnotationString);
}

AnnotationDecorations tryParseAnnotationString(SPIRVModule *BM,
                                               StringRef AnnotatedCode) {
  AnnotationDecorations Decorates;
  const bool AllowFPGAMemAccesses =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_memory_accesses);
  const bool AllowFPGAMemAttr = BM->isAllowedToUseExtension(
      ExtensionID::SPV_INTEL_fpga_memory_attributes);
  if (!AllowFPGAMemAccesses && !AllowFPGAMemAttr) {
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
    return Decorates;
  }

  IntelLSUControlsInfo LSUControls;

  // Intel FPGA decorations are separated into
  // {word} OR {word:value,value,...} blocks
  std::regex DecorationRegex("\\{[\\w:,-]+\\}");
  using RegexIterT = std::regex_iterator<StringRef::const_iterator>;
  RegexIterT DecorationsIt(AnnotatedCode.begin(), AnnotatedCode.end(),
                           DecorationRegex);
  RegexIterT DecorationsEnd;
  // If we didn't find any FPGA specific annotations that are seprated as
  // described above, then add a UserSemantic decoration
  if (DecorationsIt == DecorationsEnd)
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
  bool IntelFPGADecorationFound = false;
  DecorationsInfoVec IntelFPGADecorationsVec;
  for (; DecorationsIt != DecorationsEnd; ++DecorationsIt) {
    // Drop the braces surrounding the actual decoration
    const StringRef AnnotatedDecoration = AnnotatedCode.substr(
        DecorationsIt->position() + 1, DecorationsIt->length() - 2);

    std::pair<StringRef, StringRef> Split = AnnotatedDecoration.split(':');
    StringRef Name = Split.first, ValueStr = Split.second;
    if (AllowFPGAMemAccesses) {
      if (Name == "params") {
        IntelFPGADecorationFound = true;
        unsigned ParamsBitMask = 0;
        bool Failure = ValueStr.getAsInteger(10, ParamsBitMask);
        assert(!Failure && "Non-integer LSU controls value");
        (void)Failure;
        LSUControls.setWithBitMask(ParamsBitMask);
      } else if (Name == "cache-size") {
        IntelFPGADecorationFound = true;
        if (!LSUControls.CacheSizeInfo.hasValue())
          continue;
        unsigned CacheSizeValue = 0;
        bool Failure = ValueStr.getAsInteger(10, CacheSizeValue);
        assert(!Failure && "Non-integer cache size value");
        (void)Failure;
        LSUControls.CacheSizeInfo = CacheSizeValue;
      } // TODO: Support LSU prefetch size, which currently defaults to 0
    }
    if (AllowFPGAMemAttr) {
      StringRef Annotation;
      Decoration Dec;
      if (Name == "pump") {
        IntelFPGADecorationFound = true;
        Dec = llvm::StringSwitch<Decoration>(ValueStr)
                  .Case("1", DecorationSinglepumpINTEL)
                  .Case("2", DecorationDoublepumpINTEL);
      } else if (Name == "register") {
        IntelFPGADecorationFound = true;
        Dec = DecorationRegisterINTEL;
      } else if (Name == "simple_dual_port") {
        IntelFPGADecorationFound = true;
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
                  .Default(DecorationUserSemantic);
        if (Dec == DecorationUserSemantic)
          Annotation = AnnotatedDecoration;
        else {
          IntelFPGADecorationFound = true;
          Annotation = ValueStr;
        }
      }
      IntelFPGADecorationsVec.emplace_back(Dec, Annotation.str());
    }
  }
  // Even if there is an annotation string that is split in blocks like Intel
  // FPGA annotation, it's not necessarily an FPGA annotation. Translate the
  // whole string as UserSemantic decoration in this case.
  if (IntelFPGADecorationFound)
    Decorates.MemoryAttributesVec = IntelFPGADecorationsVec;
  else
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
  Decorates.MemoryAccessesVec = LSUControls.getDecorationsFromCurrentState();

  return Decorates;
}

std::vector<SPIRVWord> getBankBitsFromString(StringRef S) {
  SmallVector<StringRef, 4> BitsString;
  S.split(BitsString, ',');

  std::vector<SPIRVWord> Bits(BitsString.size());
  for (size_t J = 0; J < BitsString.size(); ++J)
    if (BitsString[J].getAsInteger(10, Bits[J]))
      return {};

  return Bits;
}

void addAnnotationDecorations(SPIRVEntry *E, DecorationsInfoVec &Decorations) {
  SPIRVModule *M = E->getModule();
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, skip it
    if (E->hasDecorate(I.first, /*Index=*/0, /*Result=*/nullptr)) {
      continue;
    }

    switch (I.first) {
    case DecorationUserSemantic:
      E->addDecorate(new SPIRVDecorateUserSemanticAttr(E, I.second));
      break;
    case DecorationMemoryINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes))
        E->addDecorate(new SPIRVDecorateMemoryINTELAttr(E, I.second));
    } break;
    case DecorationMergeINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        StringRef Name = StringRef(I.second).split(':').first;
        StringRef Direction = StringRef(I.second).split(':').second;
        E->addDecorate(
            new SPIRVDecorateMergeINTELAttr(E, Name.str(), Direction.str()));
      }
    } break;
    case DecorationBankBitsINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes))
        E->addDecorate(new SPIRVDecorateBankBitsINTELAttr(
            E, getBankBitsFromString(I.second)));
    } break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        assert(I.second.empty());
        E->addDecorate(I.first);
      }
    } break;
    case DecorationBurstCoalesceINTEL:
    case DecorationDontStaticallyCoalesceINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        assert(I.second.empty());
        E->addDecorate(I.first);
      }
    } break;
    case DecorationNumbanksINTEL:
    case DecorationBankwidthINTEL:
    case DecorationMaxPrivateCopiesINTEL:
    case DecorationMaxReplicatesINTEL:
    case DecorationForcePow2DepthINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        SPIRVWord Result = 0;
        StringRef(I.second).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    case DecorationCacheSizeINTEL:
    case DecorationPrefetchINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        SPIRVWord Result = 0;
        StringRef(I.second).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
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
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, skip it
    if (E->hasMemberDecorate(I.first, /*Index=*/0, MemberNumber,
                             /*Result=*/nullptr)) {
      continue;
    }

    switch (I.first) {
    case DecorationUserSemantic:
      E->addMemberDecorate(
          new SPIRVMemberDecorateUserSemanticAttr(E, MemberNumber, I.second));
      break;
    case DecorationMemoryINTEL:
      E->addMemberDecorate(
          new SPIRVMemberDecorateMemoryINTELAttr(E, MemberNumber, I.second));
      break;
    case DecorationMergeINTEL: {
      StringRef Name = StringRef(I.second).split(':').first;
      StringRef Direction = StringRef(I.second).split(':').second;
      E->addMemberDecorate(new SPIRVMemberDecorateMergeINTELAttr(
          E, MemberNumber, Name.str(), Direction.str()));
    } break;
    case DecorationBankBitsINTEL:
      E->addMemberDecorate(new SPIRVMemberDecorateBankBitsINTELAttr(
          E, MemberNumber, getBankBitsFromString(I.second)));
      break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL:
      assert(I.second.empty());
      E->addMemberDecorate(MemberNumber, I.first);
      break;
    // The rest of IntelFPGA decorations:
    // DecorationNumbanksINTEL
    // DecorationBankwidthINTEL
    // DecorationMaxPrivateCopiesINTEL
    // DecorationMaxReplicatesINTEL
    // DecorationForcePow2DepthINTEL
    default:
      SPIRVWord Result = 0;
      StringRef(I.second).getAsInteger(10, Result);
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
  case Intrinsic::arithmetic_fence:
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
  if (RMode.endswith("tonearest"))
    I->addFPRoundingMode(FPRoundingModeRTE);
  else if (RMode.endswith("towardzero"))
    I->addFPRoundingMode(FPRoundingModeRTZ);
  else if (RMode.endswith("upward"))
    I->addFPRoundingMode(FPRoundingModeRTP);
  else if (RMode.endswith("downward"))
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

SPIRVValue *LLVMToSPIRVBase::transIntrinsicInst(IntrinsicInst *II,
                                                SPIRVBasicBlock *BB) {
  auto GetMemoryAccess = [](MemIntrinsic *MI) -> std::vector<SPIRVWord> {
    std::vector<SPIRVWord> MemoryAccess(1, MemoryAccessMaskNone);
    if (SPIRVWord AlignVal = MI->getDestAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
        SPIRVWord SourceAlignVal = MTI->getSourceAlignment();
        assert(SourceAlignVal && "Missed Source alignment!");

        // In a case when alignment of source differs from dest one
        // least value is guaranteed anyway.
        AlignVal = std::min(AlignVal, SourceAlignVal);
      }
      MemoryAccess.push_back(AlignVal);
    }
    if (MI->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    return MemoryAccess;
  };

  // LLVM intrinsics with known translation to SPIR-V are handled here. They
  // also must be registered at isKnownIntrinsic function in order to make
  // -spirv-allow-unknown-intrinsics work correctly.
  switch (II->getIntrinsicID()) {
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
    BM->addCapability(CapabilityShader);
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
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID());
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(STy, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
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
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID());
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

    Op OC = (II->getIntrinsicID() == Intrinsic::smin)
                ? OpSLessThan
                : ((II->getIntrinsicID() == Intrinsic::smax)
                       ? OpSGreaterThan
                       : ((II->getIntrinsicID() == Intrinsic::umin)
                              ? OpULessThan
                              : OpUGreaterThan));
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
    SPIRVWord ExtOp = II->getIntrinsicID() == Intrinsic::ctlz ? OpenCLLIB::Clz
                                                              : OpenCLLIB::Ctz;
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
    auto BI = BM->addBinaryInst(OpFAdd, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fsub: {
    auto BI = BM->addBinaryInst(OpFSub, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fmul: {
    auto BI = BM->addBinaryInst(OpFMul, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fdiv: {
    auto BI = BM->addBinaryInst(OpFDiv, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_frem: {
    auto BI = BM->addBinaryInst(OpFRem, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fma: {
    std::vector<SPIRVValue *> Args{transValue(II->getArgOperand(0), BB),
                                   transValue(II->getArgOperand(1), BB),
                                   transValue(II->getArgOperand(2), BB)};
    auto BI = BM->addExtInst(transType(II->getType()),
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
    auto BI = BM->addUnaryInst(OpConvertUToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_sitofp: {
    auto BI = BM->addUnaryInst(OpConvertSToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fpext: {
    return BM->addUnaryInst(OpFConvert, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_fptrunc: {
    auto BI = BM->addUnaryInst(OpFConvert, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fcmp:
  case Intrinsic::experimental_constrained_fcmps: {
    auto MetaMod = cast<MetadataAsValue>(II->getOperand(2))->getMetadata();
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
    auto BI = BM->addBinaryInst(OpFAdd, Ty, Mul,
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
    if (BM->shouldReplaceLLVMFmulAddWithOpenCLMad()) {
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
  case Intrinsic::usub_sat: {
    // usub.sat(a, b) -> (a > b) ? a - b : 0
    SPIRVType *Ty = transType(II->getType());
    Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
    SPIRVValue *FirstArgVal = transValue(II->getArgOperand(0), BB);
    SPIRVValue *SecondArgVal = transValue(II->getArgOperand(1), BB);

    SPIRVValue *Sub =
        BM->addBinaryInst(OpISub, Ty, FirstArgVal, SecondArgVal, BB);
    SPIRVValue *Cmp = BM->addCmpInst(OpUGreaterThan, transType(BoolTy),
                                     FirstArgVal, SecondArgVal, BB);
    SPIRVValue *Zero = transValue(Constant::getNullValue(II->getType()), BB);
    return BM->addSelectInst(Cmp, Sub, Zero, BB);
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
    SPIRVType *VarTy = transType(PointerType::get(AT, SPIRV::SPIRAS_Constant));
    SPIRVValue *Var = BM->addVariable(VarTy, /*isConstant*/ true,
                                      spv::internal::LinkageTypeInternal, Init,
                                      "", StorageClassUniformConstant, nullptr);
    SPIRVType *SourceTy =
        transType(PointerType::get(Val->getType(), SPIRV::SPIRAS_Constant));
    SPIRVValue *Source = BM->addUnaryInst(OpBitcast, SourceTy, Var, BB);
    SPIRVValue *Target = transValue(MSI->getRawDest(), BB);
    return BM->addCopyMemorySizedInst(Target, Source, CompositeTy->getLength(),
                                      GetMemoryAccess(MSI), BB);
  } break;
  case Intrinsic::memcpy:
    return BM->addCopyMemorySizedInst(
        transValue(II->getOperand(0), BB), transValue(II->getOperand(1), BB),
        transValue(II->getOperand(2), BB),
        GetMemoryAccess(cast<MemIntrinsic>(II)), BB);
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
    SPIRVType *Ty = transType(II->getType());

    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(II->getArgOperand(1));
    if (!GEP)
      return nullptr;
    Constant *C = cast<Constant>(GEP->getOperand(0));
    StringRef AnnotationString;
    getConstantStringInfo(C, AnnotationString);

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
    // Strip all bitcast and addrspace casts from the pointer argument:
    //   llvm annotation intrinsic only takes i8*, so the original pointer
    //   probably had to loose its addrspace and its original type.
    Value *AnnotSubj = II->getArgOperand(0);
    while (isa<BitCastInst>(AnnotSubj) || isa<AddrSpaceCastInst>(AnnotSubj)) {
      AnnotSubj = cast<CastInst>(AnnotSubj)->getOperand(0);
    }

    std::string AnnotationString;
    processAnnotationString(II, AnnotationString);
    AnnotationDecorations Decorations =
        tryParseAnnotationString(BM, AnnotationString);

    // If the pointer is a GEP on a struct, then we have to emit a member
    // decoration for the GEP-accessed struct, or a memory access decoration
    // for the GEP itself.
    auto *GI = dyn_cast<GetElementPtrInst>(AnnotSubj);
    if (GI && isa<StructType>(GI->getSourceElementType())) {
      auto *Ty = transType(GI->getSourceElementType());
      auto *ResPtr = transValue(GI, BB);
      SPIRVWord MemberNumber =
          dyn_cast<ConstantInt>(GI->getOperand(2))->getZExtValue();

      // If we didn't find any IntelFPGA-specific decorations, let's add the
      // whole annotation string as UserSemantic Decoration
      if (Decorations.MemoryAttributesVec.empty() &&
          Decorations.MemoryAccessesVec.empty()) {
        // TODO: Is there a way to detect that the annotation belongs solely
        // to struct member memory atributes or struct member memory access
        // controls? This would allow emitting just the necessary decoration.
        Ty->addMemberDecorate(new SPIRVMemberDecorateUserSemanticAttr(
            Ty, MemberNumber, AnnotationString.c_str()));
        ResPtr->addDecorate(new SPIRVDecorateUserSemanticAttr(
            ResPtr, AnnotationString.c_str()));
      } else {
        addAnnotationDecorationsForStructMember(
            Ty, MemberNumber, Decorations.MemoryAttributesVec);
        // Apply the LSU parameter decoration to the pointer result of a GEP
        // to the given struct member (InBoundsPtrAccessChain in SPIR-V).
        // Decorating the member itself with a MemberDecoration is not feasible,
        // because multiple accesses to the struct-held memory can require
        // different LSU parameters.
        addAnnotationDecorations(ResPtr, Decorations.MemoryAccessesVec);
      }
      II->replaceAllUsesWith(II->getOperand(0));
    } else {
      auto *Ty = transType(II->getType());
      auto *BI = dyn_cast<BitCastInst>(II->getOperand(0));
      if (AnnotationString == kOCLBuiltinName::FPGARegIntel) {
        if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_reg))
          return BM->addFPGARegINTELInst(Ty, transValue(BI, BB), BB);
        return transValue(BI, BB);
      } else {
        // Memory accesses to a standalone pointer variable
        auto *DecSubj = transValue(II->getArgOperand(0), BB);
        if (Decorations.MemoryAccessesVec.empty())
          DecSubj->addDecorate(new SPIRVDecorateUserSemanticAttr(
              DecSubj, AnnotationString.c_str()));
        else
          // Apply the LSU parameter decoration to the pointer result of an
          // instruction. Note it's the address to the accessed memory that's
          // loaded from the original pointer variable, and not the value
          // accessed by the latter.
          addAnnotationDecorations(DecSubj, Decorations.MemoryAccessesVec);
        II->replaceAllUsesWith(II->getOperand(0));
      }
    }
    return nullptr;
  }
  case Intrinsic::stacksave: {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_variable_length_array)) {
      auto *Ty = transType(II->getType());
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
  default:
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
  // Treat all llvm.fence instructions as having CrossDevice scope:
  SPIRVValue *RetScope = transConstant(getUInt32(M, ScopeCrossDevice));
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

  if (MangledName.startswith(SPCV_CAST) || MangledName == SAMPLER_INIT)
    return oclTransSpvcCastSampler(CI, BB);

  if (oclIsBuiltin(MangledName, DemangledName) ||
      isDecoratedSPIRVFunc(F, DemangledName)) {
    if (auto BV = transBuiltinToConstant(DemangledName, CI))
      return BV;
    if (auto BV = transBuiltinToInst(DemangledName, CI, BB))
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
                ExtensionID::SPV_INTEL_non_constant_addrspace_printf)) {
          std::string ErrorStr =
              "The SPV_INTEL_non_constant_addrspace_printf extension should be "
              "allowed to translate this module, because this LLVM module "
              "contains the printf function with format string, whose address "
              "space is not equal to 2 (constant).";
          getErrorLog().checkError(false, SPIRVEC_RequiresExtension, CI,
                                   ErrorStr);
        }
        BM->addExtension(ExtensionID::SPV_INTEL_non_constant_addrspace_printf);
        BM->addCapability(internal::CapabilityNonConstantAddrspacePrintfINTEL);
      }
    }

    return addDecorations(
        BM->addExtInst(
            transType(CI->getType()), BM->getExtInstSetId(ExtSetKind), ExtOp,
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
        transValue(CI->getCalledOperand(), BB), transType(CI->getType()),
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
  auto AsmTarget = static_cast<SPIRVAsmTargetINTEL *>(
      BM->getOrAddAsmTargetINTEL(TripleStr.str()));
  auto SIA = BM->addAsmINTEL(
      static_cast<SPIRVTypeFunction *>(transType(IA->getFunctionType())),
      AsmTarget, IA->getAsmString(), IA->getConstraintString());
  if (IA->hasSideEffects())
    SIA->addDecorate(DecorationSideEffectsINTEL);
  return SIA;
}

SPIRVValue *LLVMToSPIRVBase::transAsmCallINTEL(CallInst *CI,
                                               SPIRVBasicBlock *BB) {
  assert(CI);
  auto IA = cast<InlineAsm>(CI->getCalledOperand());
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
    getConstantStringInfo(GV, AnnotationString);
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

void LLVMToSPIRVBase::collectInputOutputVariables(SPIRVFunction *SF,
                                                  Function *F) {
  for (auto &GV : M->globals()) {
    const auto AS = GV.getAddressSpace();
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
      SF->addVariable(ValueMap[&GV]);
    }
  }
}

void LLVMToSPIRVBase::mutateFuncArgType(
    const std::map<unsigned, Type *> &ChangedType, Function *F) {
  for (auto &I : ChangedType) {
    for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE; ++UI) {
      auto Call = dyn_cast<CallInst>(*UI);
      if (!Call)
        continue;
      auto Arg = Call->getArgOperand(I.first);
      auto OrigTy = Arg->getType();
      if (OrigTy == I.second)
        continue;
      SPIRVDBG(dbgs() << "[mutate arg type] " << *Call << ", " << *Arg << '\n');
      auto CastF = M->getOrInsertFunction(SPCV_CAST, I.second, OrigTy);
      std::vector<Value *> Args;
      Args.push_back(Arg);
      auto Cast = CallInst::Create(CastF, Args, "", Call);
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

void LLVMToSPIRVBase::transFunction(Function *I) {
  SPIRVFunction *BF = transFunctionDecl(I);
  // Creating all basic blocks before creating any instruction.
  for (auto &FI : *I) {
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

  bool IsKernelEntryPoint = isKernel(I);

  if (IsKernelEntryPoint) {
    collectInputOutputVariables(BF, I);
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

  // Transform SPV-IR builtin calls to builtin variables.
  if (!transWorkItemBuiltinCallsToVariables())
    return false;

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
    auto FT = F.getFunctionType();
    std::map<unsigned, Type *> ChangedType;
    oclGetMutatedArgumentTypesByBuiltin(FT, ChangedType, &F);
    mutateFuncArgType(ChangedType, &F);
  }

  // SPIR-V logical layout requires all function declarations go before
  // function definitions.
  std::vector<Function *> Decls, Defs;
  for (auto &F : *M) {
    if (isBuiltinTransToInst(&F) || isBuiltinTransToExtInst(&F) ||
        F.getName().startswith(SPCV_CAST) ||
        F.getName().startswith(LLVM_MEMCPY) ||
        F.getName().startswith(SAMPLER_INIT))
      continue;
    if (F.isDeclaration())
      Decls.push_back(&F);
    else
      Defs.push_back(&F);
  }
  for (auto I : Decls)
    transFunctionDecl(I);
  for (auto I : Defs)
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
    llvm::FunctionType *FT, std::map<unsigned, Type *> &ChangedType,
    Function *F) {
  StringRef Demangled;
  if (!oclIsBuiltin(F->getName(), Demangled))
    return;
  if (Demangled.find(kSPIRVName::SampledImage) == std::string::npos)
    return;
  if (FT->getParamType(1)->isIntegerTy())
    ChangedType[1] = getSamplerType(F->getParent());
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

  auto Inst = transBuiltinToInstWithoutDecoration(OC, CI, BB);
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

      switch (EMode) {
      case spv::ExecutionModeContractionOff:
        BF->addExecutionMode(BM->add(
            new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        break;
      case spv::ExecutionModeInitializer:
      case spv::ExecutionModeFinalizer:
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1)) {
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        } else {
          getErrorLog().checkError(false, SPIRVEC_Requires1_1,
                                   "Initializer/Finalizer Execution Mode");
          return false;
        }
        break;
      case spv::ExecutionModeLocalSize:
      case spv::ExecutionModeLocalSizeHint: {
        unsigned X, Y, Z;
        N.get(X).get(Y).get(Z);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
      } break;
      case spv::ExecutionModeMaxWorkgroupSizeINTEL: {
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes)) {
          unsigned X, Y, Z;
          N.get(X).get(Y).get(Z);
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
          BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
          BM->addCapability(CapabilityKernelAttributesINTEL);
        }
      } break;
      case spv::ExecutionModeNoGlobalOffsetINTEL: {
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes)) {
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
          BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
          BM->addCapability(CapabilityKernelAttributesINTEL);
        }
      } break;
      case spv::ExecutionModeVecTypeHint:
      case spv::ExecutionModeSubgroupSize:
      case spv::ExecutionModeSubgroupsPerWorkgroup: {
        unsigned X;
        N.get(X);
        BF->addExecutionMode(BM->add(
            new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode), X)));
      } break;
      case spv::ExecutionModeNumSIMDWorkitemsINTEL:
      case spv::ExecutionModeSchedulerTargetFmaxMhzINTEL:
      case spv::ExecutionModeMaxWorkDimINTEL:
      case spv::internal::ExecutionModeStreamingInterfaceINTEL: {
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes)) {
          unsigned X;
          N.get(X);
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              BF, static_cast<ExecutionMode>(EMode), X)));
          BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
          BM->addCapability(CapabilityFPGAKernelAttributesINTEL);
        }
      } break;
      case spv::ExecutionModeSharedLocalMemorySizeINTEL: {
        if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
          break;
        unsigned SLMSize;
        N.get(SLMSize);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), SLMSize)));
      } break;

      case spv::ExecutionModeDenormPreserve:
      case spv::ExecutionModeDenormFlushToZero:
      case spv::ExecutionModeSignedZeroInfNanPreserve:
      case spv::ExecutionModeRoundingModeRTE:
      case spv::ExecutionModeRoundingModeRTZ: {
        if (!BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_float_controls))
          break;
        unsigned TargetWidth;
        N.get(TargetWidth);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), TargetWidth)));
      } break;
      case spv::ExecutionModeRoundingModeRTPINTEL:
      case spv::ExecutionModeRoundingModeRTNINTEL:
      case spv::ExecutionModeFloatingPointModeALTINTEL:
      case spv::ExecutionModeFloatingPointModeIEEEINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_float_controls2))
          break;
        unsigned TargetWidth;
        N.get(TargetWidth);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), TargetWidth)));
      } break;
      case spv::internal::ExecutionModeFastCompositeKernelINTEL: {
        if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fast_composite))
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
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
      BF->addExecutionMode(BF->getModule()->add(
          new SPIRVExecutionMode(BF, spv::ExecutionModeContractionOff)));
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
  if (isGroupOpCode(OC))
    BM->addCapability(CapabilityGroups);
  switch (OC) {
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
    Type *ImageTy = Image->getType();
    if (isOCLImageType(ImageTy))
      ImageTy = getSPIRVImageTypeFromOCL(M, ImageTy);
    Type *SampledImgTy = getSPIRVTypeByChangeBaseTypeName(
        M, ImageTy, kSPIRVTypeName::Image, kSPIRVTypeName::SampledImg);
    Value *Sampler = CI->getArgOperand(1);
    return BM->addSampledImageInst(transType(SampledImgTy),
                                   transValue(Image, BB),
                                   transValue(Sampler, BB), BB);
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
      ResTy = cast<PointerType>(OpItr->getType())->getPointerElementType();
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
      ResTy = cast<PointerType>(OpItr->getType())->getPointerElementType();
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
      ResTy = cast<PointerType>(OpItr->getType())->getPointerElementType();
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
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else if (isCmpOpCode(OC) || isUnaryPredicateOpCode(OC)) {
      auto ResultTy = CI->getType();
      Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
      auto IsVector = ResultTy->isVectorTy();
      if (IsVector)
        BoolTy = FixedVectorType::get(
            BoolTy, cast<FixedVectorType>(ResultTy)->getNumElements());
      auto BBT = transType(BoolTy);
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
      auto Zero = transValue(Constant::getNullValue(ResultTy), BB);
      auto One = transValue(
          IsVector ? Constant::getAllOnesValue(ResultTy) : getInt32(M, 1), BB);
      return BM->addSelectInst(Res, One, Zero, BB);
    } else if (isBinaryOpCode(OC)) {
      assert(CI && CI->arg_size() == 2 && "Invalid call inst");
      return BM->addBinaryInst(OC, transType(CI->getType()),
                               transValue(CI->getArgOperand(0), BB),
                               transValue(CI->getArgOperand(1), BB), BB);
    } else if (CI->arg_size() == 1 && !CI->getType()->isVoidTy() &&
               !hasExecScope(OC) && !isAtomicOpCode(OC)) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else {
      auto Args = getArguments(CI);
      SPIRVType *SPRetTy = nullptr;
      Type *RetTy = CI->getType();
      auto F = CI->getCalledFunction();
      if (!RetTy->isVoidTy()) {
        SPRetTy = transType(RetTy);
      } else if (Args.size() > 0 && F->arg_begin()->hasStructRetAttr()) {
        SPRetTy = transType(F->arg_begin()->getType()->getPointerElementType());
        Args.erase(Args.begin());
      }
      auto *SPI = SPIRVInstTemplateBase::create(OC);
      std::vector<SPIRVWord> SPArgs;
      for (size_t I = 0, E = Args.size(); I != E; ++I) {
        assert((!isFunctionPointerType(Args[I]->getType()) ||
                isa<Function>(Args[I])) &&
               "Invalid function pointer argument");
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

void addPassesForSPIRV(legacy::PassManager &PassMgr,
                       const SPIRV::TranslatorOpts &Opts) {
  if (Opts.isSPIRVMemToRegEnabled())
    PassMgr.add(createPromoteMemoryToRegisterPass());
  PassMgr.add(createPreprocessMetadataLegacy());
  PassMgr.add(createSPIRVLowerSPIRBlocksLegacy());
  PassMgr.add(createOCLTypeToSPIRVLegacy());
  PassMgr.add(createSPIRVLowerOCLBlocksLegacy());
  PassMgr.add(createOCLToSPIRVLegacy());
  PassMgr.add(createSPIRVRegularizeLLVMLegacy());
  PassMgr.add(createSPIRVLowerConstExprLegacy());
  PassMgr.add(createSPIRVLowerBoolLegacy());
  PassMgr.add(createSPIRVLowerMemmoveLegacy());
  PassMgr.add(createSPIRVLowerSaddWithOverflowLegacy());
  PassMgr.add(createSPIRVLowerBitCastToNonStandardTypeLegacy(Opts));
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

bool llvm::writeSpirv(Module *M, std::ostream &OS, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  // To preserve old behavior of the translator, let's enable all extensions
  // by default in this API
  DefaultOpts.enableAllExtensions();
  return llvm::writeSpirv(M, DefaultOpts, OS, ErrMsg);
}

bool llvm::writeSpirv(Module *M, const SPIRV::TranslatorOpts &Opts,
                      std::ostream &OS, std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule(Opts));
  if (!isValidLLVMModule(M, BM->getErrorLog()))
    return false;

  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr, Opts);
  // Run loop simplify pass in order to avoid duplicate OpLoopMerge
  // instruction. It can happen in case of continue operand in the loop.
  if (hasLoopMetadata(M))
    PassMgr.add(createLoopSimplifyPass());
  PassMgr.add(createLLVMToSPIRVLegacy(BM.get()));
  PassMgr.run(*M);

  if (BM->getError(ErrMsg) != SPIRVEC_Success)
    return false;
  OS << *BM;
  return true;
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
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  if (!isValidLLVMModule(M, BM->getErrorLog()))
    return false;

  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr, Opts);
  PassMgr.run(*M);
  return true;
}
