//===- SPIRVReader.cpp - Converts SPIR-V to LLVM ----------------*- C++ -*-===//
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
/// This file implements conversion of SPIR-V binary to LLVM IR.
///
//===----------------------------------------------------------------------===//
#include "SPIRVReader.h"
#include "OCLUtil.h"
#include "SPIRVAsm.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVMDBuilder.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVModule.h"
#include "SPIRVToLLVMDbgTran.h"
#include "SPIRVToOCL.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "VectorComputeUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerImpl.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>

#define DEBUG_TYPE "spirv"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

cl::opt<bool> SPIRVEnableStepExpansion(
    "spirv-expand-step", cl::init(true),
    cl::desc("Enable expansion of OpenCL step and smoothstep function"));

// Prefix for placeholder global variable name.
const char *KPlaceholderPrefix = "placeholder.";

// Save the translated LLVM before validation for debugging purpose.
static bool DbgSaveTmpLLVM = false;
static const char *DbgTmpLLVMFileName = "_tmp_llvmbil.ll";

namespace kOCLTypeQualifierName {
const static char *Volatile = "volatile";
const static char *Restrict = "restrict";
const static char *Pipe = "pipe";
} // namespace kOCLTypeQualifierName

static bool isKernel(SPIRVFunction *BF) {
  return BF->getModule()->isEntryPoint(ExecutionModelKernel, BF->getId());
}

static void dumpLLVM(Module *M, const std::string &FName) {
  std::error_code EC;
  raw_fd_ostream FS(FName, EC, sys::fs::OF_None);
  if (!EC) {
    FS << *M;
    FS.close();
  }
}

static MDNode *getMDNodeStringIntVec(LLVMContext *Context,
                                     const std::vector<SPIRVWord> &IntVals) {
  std::vector<Metadata *> ValueVec;
  for (auto &I : IntVals)
    ValueVec.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(*Context), I)));
  return MDNode::get(*Context, ValueVec);
}

static MDNode *getMDTwoInt(LLVMContext *Context, unsigned Int1, unsigned Int2) {
  std::vector<Metadata *> ValueVec;
  ValueVec.push_back(ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(*Context), Int1)));
  ValueVec.push_back(ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(*Context), Int2)));
  return MDNode::get(*Context, ValueVec);
}

static void addOCLVersionMetadata(LLVMContext *Context, Module *M,
                                  const std::string &MDName, unsigned Major,
                                  unsigned Minor) {
  NamedMDNode *NamedMD = M->getOrInsertNamedMetadata(MDName);
  NamedMD->addOperand(getMDTwoInt(Context, Major, Minor));
}

static void addNamedMetadataStringSet(LLVMContext *Context, Module *M,
                                      const std::string &MDName,
                                      const std::set<std::string> &StrSet) {
  NamedMDNode *NamedMD = M->getOrInsertNamedMetadata(MDName);
  std::vector<Metadata *> ValueVec;
  for (auto &&Str : StrSet) {
    ValueVec.push_back(MDString::get(*Context, Str));
  }
  NamedMD->addOperand(MDNode::get(*Context, ValueVec));
}

static void addKernelArgumentMetadata(
    LLVMContext *Context, const std::string &MDName, SPIRVFunction *BF,
    llvm::Function *Fn,
    std::function<Metadata *(SPIRVFunctionParameter *)> ForeachFnArg) {
  std::vector<Metadata *> ValueVec;
  BF->foreachArgument([&](SPIRVFunctionParameter *Arg) {
    ValueVec.push_back(ForeachFnArg(Arg));
  });
  Fn->setMetadata(MDName, MDNode::get(*Context, ValueVec));
}

static void addBufferLocationMetadata(
    LLVMContext *Context, SPIRVFunction *BF, llvm::Function *Fn,
    std::function<Metadata *(SPIRVFunctionParameter *)> ForeachFnArg) {
  std::vector<Metadata *> ValueVec;
  bool DecorationFound = false;
  BF->foreachArgument([&](SPIRVFunctionParameter *Arg) {
    if (Arg->getType()->isTypePointer() &&
        Arg->hasDecorate(DecorationBufferLocationINTEL)) {
      DecorationFound = true;
      ValueVec.push_back(ForeachFnArg(Arg));
    } else {
      llvm::Metadata *DefaultNode = ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(*Context), -1));
      ValueVec.push_back(DefaultNode);
    }
  });
  if (DecorationFound)
    Fn->setMetadata("kernel_arg_buffer_location",
                    MDNode::get(*Context, ValueVec));
}

static void addRuntimeAlignedMetadata(
    LLVMContext *Context, SPIRVFunction *BF, llvm::Function *Fn,
    std::function<Metadata *(SPIRVFunctionParameter *)> ForeachFnArg) {
  std::vector<Metadata *> ValueVec;
  bool RuntimeAlignedFound = false;
  [[maybe_unused]] llvm::Metadata *DefaultNode =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(*Context), 0));
  BF->foreachArgument([&](SPIRVFunctionParameter *Arg) {
    if (Arg->hasAttr(FunctionParameterAttributeRuntimeAlignedINTEL) ||
        Arg->hasDecorate(internal::DecorationRuntimeAlignedINTEL)) {
      RuntimeAlignedFound = true;
      ValueVec.push_back(ForeachFnArg(Arg));
    } else {
      ValueVec.push_back(DefaultNode);
    }
  });
  if (RuntimeAlignedFound)
    Fn->setMetadata("kernel_arg_runtime_aligned",
                    MDNode::get(*Context, ValueVec));
}

Value *SPIRVToLLVM::getTranslatedValue(SPIRVValue *BV) {
  auto Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
}

static std::optional<llvm::Attribute>
translateSEVMetadata(SPIRVValue *BV, llvm::LLVMContext &Context) {
  std::optional<llvm::Attribute> RetAttr;

  if (!BV->hasDecorate(DecorationSingleElementVectorINTEL))
    return RetAttr;

  auto VecDecorateSEV = BV->getDecorations(DecorationSingleElementVectorINTEL);
  assert(VecDecorateSEV.size() == 1 &&
         "Entry must have no more than one SingleElementVectorINTEL "
         "decoration");
  auto *DecorateSEV = VecDecorateSEV.back();
  auto LiteralCount = DecorateSEV->getLiteralCount();
  assert(LiteralCount <= 1 && "SingleElementVectorINTEL decoration must "
                              "have no more than one literal");

  SPIRVWord IndirectLevelsOnElement =
      (LiteralCount == 1) ? DecorateSEV->getLiteral(0) : 0;

  RetAttr = Attribute::get(Context, kVCMetadata::VCSingleElementVector,
                           std::to_string(IndirectLevelsOnElement));
  return RetAttr;
}

IntrinsicInst *SPIRVToLLVM::getLifetimeStartIntrinsic(Instruction *I) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  if (II && II->getIntrinsicID() == Intrinsic::lifetime_start)
    return II;
  // Bitcast might be inserted during translation of OpLifetimeStart
  auto *BC = dyn_cast<BitCastInst>(I);
  if (BC) {
    for (const auto &U : BC->users()) {
      II = dyn_cast<IntrinsicInst>(U);
      if (II && II->getIntrinsicID() == Intrinsic::lifetime_start)
        return II;
      ;
    }
  }
  return nullptr;
}

SPIRVErrorLog &SPIRVToLLVM::getErrorLog() { return BM->getErrorLog(); }

void SPIRVToLLVM::setCallingConv(CallInst *Call) {
  Function *F = Call->getCalledFunction();
  assert(F && "Function pointers are not allowed in SPIRV");
  Call->setCallingConv(F->getCallingConv());
}

// For integer types shorter than 32 bit, unsigned/signedness can be inferred
// from zext/sext attribute.
MDString *SPIRVToLLVM::transOCLKernelArgTypeName(SPIRVFunctionParameter *Arg) {
  auto *Ty =
      Arg->isByVal() ? Arg->getType()->getPointerElementType() : Arg->getType();
  return MDString::get(*Context, transTypeToOCLTypeName(Ty, !Arg->isZext()));
}

Value *SPIRVToLLVM::mapFunction(SPIRVFunction *BF, Function *F) {
  SPIRVDBG(spvdbgs() << "[mapFunction] " << *BF << " -> ";
           dbgs() << *F << '\n';)
  FuncMap[BF] = F;
  return F;
}

std::optional<uint64_t> SPIRVToLLVM::transIdAsConstant(SPIRVId Id) {
  auto *V = BM->get<SPIRVValue>(Id);
  const auto *ConstValue =
      dyn_cast<ConstantInt>(transValue(V, nullptr, nullptr));
  if (!ConstValue)
    return {};
  return ConstValue->getZExtValue();
}

std::optional<uint64_t> SPIRVToLLVM::getAlignment(SPIRVValue *V) {
  SPIRVWord AlignmentBytes = 0;
  if (V->hasAlignment(&AlignmentBytes)) {
    return AlignmentBytes;
  }

  // If there was no Alignment decoration, look for AlignmentId instead.
  SPIRVId AlignId;
  if (V->hasDecorateId(DecorationAlignmentId, 0, &AlignId)) {
    return transIdAsConstant(AlignId);
  }
  return {};
}

Type *SPIRVToLLVM::transFPType(SPIRVType *T) {
  switch (T->getFloatBitWidth()) {
  case 16:
    if (T->isTypeFloat(16, FPEncodingBFloat16KHR))
      return Type::getBFloatTy(*Context);
    return Type::getHalfTy(*Context);
  case 32:
    return Type::getFloatTy(*Context);
  case 64:
    return Type::getDoubleTy(*Context);
  default:
    llvm_unreachable("Invalid type");
    return nullptr;
  }
}

std::string SPIRVToLLVM::transVCTypeName(SPIRVTypeBufferSurfaceINTEL *PST) {
  if (PST->hasAccessQualifier())
    return VectorComputeUtil::getVCBufferSurfaceName(PST->getAccessQualifier());
  return VectorComputeUtil::getVCBufferSurfaceName();
}

template <typename ImageType>
std::optional<SPIRVAccessQualifierKind> getAccessQualifier(ImageType *T) {
  if (!T->hasAccessQualifier())
    return {};
  return T->getAccessQualifier();
}

Type *SPIRVToLLVM::transType(SPIRVType *T, bool UseTPT) {
  // Try to reuse a known type if it's already matched. However, if we want to
  // produce a TypedPointerType in lieu of a PointerType, we *do not* want to
  // pull a PointerType out of the type map, nor do we want to store a
  // TypedPointerType in there. This is generally safe to do, as types are
  // usually uniqued by LLVM, but we need to be cautious around struct types.
  auto Loc = TypeMap.find(T);
  if (Loc != TypeMap.end() && !UseTPT)
    return Loc->second;

  SPIRVDBG(spvdbgs() << "[transType] " << *T << " -> ";)
  T->validate();
  switch (static_cast<SPIRVWord>(T->getOpCode())) {
  case OpTypeVoid:
    return mapType(T, Type::getVoidTy(*Context));
  case OpTypeBool:
    return mapType(T, Type::getInt1Ty(*Context));
  case OpTypeInt:
    return mapType(T, Type::getIntNTy(*Context, T->getIntegerBitWidth()));
  case OpTypeFloat:
    return mapType(T, transFPType(T));
  case OpTypeArray: {
    // The length might be an OpSpecConstantOp, that needs to be specialized
    // and evaluated before the LLVM ArrayType can be constructed.
    auto *LenExpr = static_cast<const SPIRVTypeArray *>(T)->getLength();
    auto *LenValue = cast<ConstantInt>(transValue(LenExpr, nullptr, nullptr));
    return mapType(T, ArrayType::get(transType(T->getArrayElementType()),
                                     LenValue->getZExtValue()));
  }
  case internal::OpTypeTokenINTEL:
    return mapType(T, Type::getTokenTy(*Context));
  case OpTypePointer: {
    unsigned AS = SPIRSPIRVAddrSpaceMap::rmap(T->getPointerStorageClass());
    if (AS == SPIRAS_CodeSectionINTEL && !BM->shouldEmitFunctionPtrAddrSpace())
      AS = SPIRAS_Private;
    if (BM->shouldEmitFunctionPtrAddrSpace() &&
        T->getPointerElementType()->getOpCode() == OpTypeFunction)
      AS = SPIRAS_CodeSectionINTEL;
    Type *ElementTy = transType(T->getPointerElementType(), UseTPT);
    if (UseTPT)
      return TypedPointerType::get(ElementTy, AS);
    return mapType(T, PointerType::get(*Context, AS));
  }
  case OpTypeUntypedPointerKHR: {
    unsigned AS = SPIRSPIRVAddrSpaceMap::rmap(T->getPointerStorageClass());
    if (AS == SPIRAS_CodeSectionINTEL && !BM->shouldEmitFunctionPtrAddrSpace())
      AS = SPIRAS_Private;
    return mapType(T, PointerType::get(*Context, AS));
  }
  case OpTypeVector:
    return mapType(T,
                   FixedVectorType::get(transType(T->getVectorComponentType()),
                                        T->getVectorComponentCount()));
  case OpTypeMatrix:
    return mapType(T, ArrayType::get(transType(T->getMatrixColumnType()),
                                     T->getMatrixColumnCount()));
  case OpTypeOpaque:
    return mapType(T, StructType::create(*Context, T->getName()));
  case OpTypeFunction: {
    auto *FT = static_cast<SPIRVTypeFunction *>(T);
    auto *RT = transType(FT->getReturnType());
    std::vector<Type *> PT;
    for (size_t I = 0, E = FT->getNumParameters(); I != E; ++I)
      PT.push_back(transType(FT->getParameterType(I)));
    return mapType(T, FunctionType::get(RT, PT, false));
  }
  case OpTypeImage: {
    auto *ST = static_cast<SPIRVTypeImage *>(T);
    if (ST->isOCLImage())
      return mapType(T,
                     getSPIRVType(OpTypeImage, transType(ST->getSampledType()),
                                  ST->getDescriptor(), getAccessQualifier(ST),
                                  !UseTPT));
    else
      llvm_unreachable("Unsupported image type");
    return nullptr;
  }
  case OpTypeSampledImage: {
    const auto *ST = static_cast<SPIRVTypeSampledImage *>(T)->getImageType();
    return mapType(
        T, getSPIRVType(OpTypeSampledImage, transType(ST->getSampledType()),
                        ST->getDescriptor(), getAccessQualifier(ST), !UseTPT));
  }
  case OpTypeStruct: {
    // We do not generate structs with any TypedPointerType members. To ensure
    // that uniqueness of struct types is maintained, reuse an existing struct
    // type in the type map, even if UseTPT is true.
    if (Loc != TypeMap.end())
      return Loc->second;
    auto *ST = static_cast<SPIRVTypeStruct *>(T);
    auto Name = ST->getName();
    if (!Name.empty()) {
      if (auto *OldST = StructType::getTypeByName(*Context, Name))
        OldST->setName("");
    } else {
      Name = "structtype";
    }
    auto *StructTy = StructType::create(*Context, Name);
    mapType(ST, StructTy);
    SmallVector<Type *, 4> MT;
    for (size_t I = 0, E = ST->getMemberCount(); I != E; ++I)
      MT.push_back(transType(ST->getMemberType(I)));
    for (auto &CI : ST->getContinuedInstructions())
      for (size_t I = 0, E = CI->getNumElements(); I != E; ++I)
        MT.push_back(transType(CI->getMemberType(I)));
    StructTy->setBody(MT, ST->isPacked());
    return StructTy;
  }
  case OpTypePipe: {
    auto *PT = static_cast<SPIRVTypePipe *>(T);
    return mapType(T,
                   getSPIRVType(OpTypePipe, PT->getAccessQualifier(), !UseTPT));
  }
  case OpTypePipeStorage: {
    StringRef FullName = "spirv.PipeStorage";
    auto *STy = StructType::getTypeByName(*Context, FullName);
    if (!STy)
      STy = StructType::create(*Context, FullName);
    if (UseTPT) {
      return mapType(T, TypedPointerType::get(STy, 1));
    }
    return mapType(T, PointerType::get(*Context, 1));
  }
  case OpTypeVmeImageINTEL: {
    auto *VT = static_cast<SPIRVTypeVmeImageINTEL *>(T)->getImageType();
    return mapType(
        T, getSPIRVType(OpTypeVmeImageINTEL, transType(VT->getSampledType()),
                        VT->getDescriptor(), getAccessQualifier(VT), !UseTPT));
  }
  case OpTypeBufferSurfaceINTEL: {
    auto *PST = static_cast<SPIRVTypeBufferSurfaceINTEL *>(T);
    Type *Ty = nullptr;
    if (UseTPT) {
      Type *StructTy = getOrCreateOpaqueStructType(M, transVCTypeName(PST));
      Ty = TypedPointerType::get(StructTy, SPIRAS_Global);
    } else {
      std::vector<unsigned> Params;
      if (PST->hasAccessQualifier()) {
        unsigned Access = static_cast<unsigned>(PST->getAccessQualifier());
        Params.push_back(Access);
      }
      Ty = TargetExtType::get(*Context, "spirv.BufferSurfaceINTEL", {}, Params);
    }
    return mapType(T, Ty);
  }
  case internal::OpTypeJointMatrixINTEL: {
    auto *MT = static_cast<SPIRVTypeJointMatrixINTEL *>(T);
    auto R = static_cast<SPIRVConstant *>(MT->getRows())->getZExtIntValue();
    auto C = static_cast<SPIRVConstant *>(MT->getColumns())->getZExtIntValue();
    std::vector<unsigned> Params = {(unsigned)R, (unsigned)C};
    if (auto *Layout = MT->getLayout())
      Params.push_back(static_cast<SPIRVConstant *>(Layout)->getZExtIntValue());
    Params.push_back(
        static_cast<SPIRVConstant *>(MT->getScope())->getZExtIntValue());
    if (auto *Use = MT->getUse())
      Params.push_back(static_cast<SPIRVConstant *>(Use)->getZExtIntValue());
    auto *CTI = MT->getComponentTypeInterpretation();
    if (!CTI)
      return mapType(
          T, llvm::TargetExtType::get(*Context, "spirv.JointMatrixINTEL",
                                      transType(MT->getCompType()), Params));
    const unsigned CTIValue =
        static_cast<SPIRVConstant *>(CTI)->getZExtIntValue();
    assert(CTIValue <= internal::InternalJointMatrixCTI::PackedInt4 &&
           "Unknown matrix component type interpretation");
    Params.push_back(CTIValue);
    return mapType(
        T, llvm::TargetExtType::get(*Context, "spirv.JointMatrixINTEL",
                                    transType(MT->getCompType()), Params));
  }
  case OpTypeCooperativeMatrixKHR: {
    auto *MT = static_cast<SPIRVTypeCooperativeMatrixKHR *>(T);
    unsigned Scope =
        static_cast<SPIRVConstant *>(MT->getScope())->getZExtIntValue();
    unsigned Rows =
        static_cast<SPIRVConstant *>(MT->getRows())->getZExtIntValue();
    unsigned Cols =
        static_cast<SPIRVConstant *>(MT->getColumns())->getZExtIntValue();
    unsigned Use =
        static_cast<SPIRVConstant *>(MT->getUse())->getZExtIntValue();

    std::vector<unsigned> Params = {Scope, Rows, Cols, Use};
    return mapType(
        T, llvm::TargetExtType::get(*Context, "spirv.CooperativeMatrixKHR",
                                    transType(MT->getCompType()), Params));
  }
  case OpTypeForwardPointer: {
    SPIRVTypeForwardPointer *FP =
        static_cast<SPIRVTypeForwardPointer *>(static_cast<SPIRVEntry *>(T));
    return mapType(T, transType(static_cast<SPIRVType *>(
                          BM->getEntry(FP->getPointerId()))));
  }
  case internal::OpTypeTaskSequenceINTEL:
    return mapType(
        T, llvm::TargetExtType::get(*Context, "spirv.TaskSequenceINTEL"));

  default: {
    auto OC = T->getOpCode();
    if (isOpaqueGenericTypeOpCode(OC) || isSubgroupAvcINTELTypeOpCode(OC)) {
      return mapType(T, getSPIRVType(OC, !UseTPT));
    }
    llvm_unreachable("Not implemented!");
  }
  }
  return 0;
}

std::string SPIRVToLLVM::transTypeToOCLTypeName(SPIRVType *T, bool IsSigned) {
  switch (T->getOpCode()) {
  case OpTypeVoid:
    return "void";
  case OpTypeBool:
    return "bool";
  case OpTypeInt: {
    std::string Prefix = IsSigned ? "" : "u";
    switch (T->getIntegerBitWidth()) {
    case 8:
      return Prefix + "char";
    case 16:
      return Prefix + "short";
    case 32:
      return Prefix + "int";
    case 64:
      return Prefix + "long";
    default:
      // Arbitrary precision integer
      return Prefix + std::string("int") + T->getIntegerBitWidth() + "_t";
    }
  } break;
  case OpTypeFloat:
    switch (T->getFloatBitWidth()) {
    case 16:
      return "half";
    case 32:
      return "float";
    case 64:
      return "double";
    default:
      llvm_unreachable("invalid floating pointer bitwidth");
      return std::string("float") + T->getFloatBitWidth() + "_t";
    }
    break;
  case OpTypeArray:
    return "array";
  case OpTypePointer: {
    SPIRVType *ET = T->getPointerElementType();
    if (isa<OpTypeFunction>(ET)) {
      SPIRVTypeFunction *TF = static_cast<SPIRVTypeFunction *>(ET);
      std::string name = transTypeToOCLTypeName(TF->getReturnType());
      name += " (*)(";
      for (unsigned I = 0, E = TF->getNumParameters(); I < E; ++I)
        name += transTypeToOCLTypeName(TF->getParameterType(I)) + ',';
      name.back() = ')'; // replace the last comma with a closing brace.
      return name;
    }
    return transTypeToOCLTypeName(ET) + "*";
  }
  case OpTypeUntypedPointerKHR:
    return "int*";
  case OpTypeVector:
    return transTypeToOCLTypeName(T->getVectorComponentType()) +
           T->getVectorComponentCount();
  case OpTypeMatrix:
    return transTypeToOCLTypeName(T->getMatrixColumnType()) +
           T->getMatrixColumnCount();
  case OpTypeOpaque:
    return T->getName();
  case OpTypeFunction:
    llvm_unreachable("Unsupported");
    return "function";
  case OpTypeStruct: {
    auto Name = T->getName();
    if (Name.find("struct.") == 0)
      Name[6] = ' ';
    else if (Name.find("union.") == 0)
      Name[5] = ' ';
    return Name;
  }
  case OpTypePipe:
    return "pipe";
  case OpTypeSampler:
    return "sampler_t";
  case OpTypeImage: {
    std::string Name;
    Name = rmap<std::string>(static_cast<SPIRVTypeImage *>(T)->getDescriptor());
    return Name;
  }
  default:
    if (isOpaqueGenericTypeOpCode(T->getOpCode())) {
      return OCLOpaqueTypeOpCodeMap::rmap(T->getOpCode());
    }
    llvm_unreachable("Not implemented");
    return "unknown";
  }
}

std::vector<Type *>
SPIRVToLLVM::transTypeVector(const std::vector<SPIRVType *> &BT, bool UseTPT) {
  std::vector<Type *> T;
  for (auto *I : BT)
    T.push_back(transType(I, UseTPT));
  return T;
}

static Type *opaquifyType(Type *Ty) {
  if (auto *TPT = dyn_cast<TypedPointerType>(Ty)) {
    Ty = PointerType::get(Ty->getContext(), TPT->getAddressSpace());
  }
  return Ty;
}

static void opaquifyTypedPointers(MutableArrayRef<Type *> Types) {
  for (Type *&Ty : Types) {
    Ty = opaquifyType(Ty);
  }
}

std::vector<Value *>
SPIRVToLLVM::transValue(const std::vector<SPIRVValue *> &BV, Function *F,
                        BasicBlock *BB) {
  std::vector<Value *> V;
  for (auto *I : BV)
    V.push_back(transValue(I, F, BB));
  return V;
}

void SPIRVToLLVM::setName(llvm::Value *V, SPIRVValue *BV) {
  auto Name = BV->getName();
  if (!Name.empty() && (!V->hasName() || Name != V->getName()))
    V->setName(Name);
}

inline llvm::Metadata *SPIRVToLLVM::getMetadataFromName(std::string Name) {
  return llvm::MDNode::get(*Context, llvm::MDString::get(*Context, Name));
}

inline std::vector<llvm::Metadata *>
SPIRVToLLVM::getMetadataFromNameAndParameter(std::string Name,
                                             SPIRVWord Parameter) {
  return {MDString::get(*Context, Name),
          ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(*Context), Parameter))};
}

inline llvm::MDNode *
SPIRVToLLVM::getMetadataFromNameAndParameter(std::string Name,
                                             int64_t Parameter) {
  std::vector<llvm::Metadata *> Metadata = {
      MDString::get(*Context, Name),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(*Context), Parameter))};
  return llvm::MDNode::get(*Context, Metadata);
}

template <typename LoopInstType>
void SPIRVToLLVM::setLLVMLoopMetadata(const LoopInstType *LM,
                                      const Loop *LoopObj) {
  if (!LM)
    return;

  auto Temp = MDNode::getTemporary(*Context, std::nullopt);
  auto *Self = MDNode::get(*Context, Temp.get());
  Self->replaceOperandWith(0, Self);
  SPIRVWord LC = LM->getLoopControl();
  if (LC == LoopControlMaskNone) {
    LoopObj->setLoopID(Self);
    return;
  }

  unsigned NumParam = 0;
  std::vector<llvm::Metadata *> Metadata;
  std::vector<SPIRVWord> LoopControlParameters = LM->getLoopControlParameters();
  Metadata.push_back(llvm::MDNode::get(*Context, Self));

  // To correctly decode loop control parameters, order of checks for loop
  // control masks must match with the order given in the spec (see 3.23),
  // i.e. check smaller-numbered bits first.
  // Unroll and UnrollCount loop controls can't be applied simultaneously with
  // DontUnroll loop control.
  if (LC & LoopControlUnrollMask && !(LC & LoopControlPartialCountMask))
    Metadata.push_back(getMetadataFromName("llvm.loop.unroll.enable"));
  else if (LC & LoopControlDontUnrollMask)
    Metadata.push_back(getMetadataFromName("llvm.loop.unroll.disable"));
  if (LC & LoopControlDependencyInfiniteMask)
    Metadata.push_back(getMetadataFromName("llvm.loop.ivdep.enable"));
  if (LC & LoopControlDependencyLengthMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.ivdep.safelen",
                                        LoopControlParameters[NumParam])));
    ++NumParam;
    // TODO: Fix the increment/assertion logic in all of the conditions
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  // Placeholder for LoopControls added in SPIR-V 1.4 spec (see 3.23)
  if (LC & LoopControlMinIterationsMask) {
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlMaxIterationsMask) {
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlIterationMultipleMask) {
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlPeelCountMask) {
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlPartialCountMask && !(LC & LoopControlDontUnrollMask)) {
    // If unroll factor is set as '1' and Unroll mask is applied attempt to do
    // full unrolling and disable it if the trip count is not known at compile
    // time.
    if (1 == LoopControlParameters[NumParam] && (LC & LoopControlUnrollMask))
      Metadata.push_back(getMetadataFromName("llvm.loop.unroll.full"));
    else
      Metadata.push_back(llvm::MDNode::get(
          *Context,
          getMetadataFromNameAndParameter("llvm.loop.unroll.count",
                                          LoopControlParameters[NumParam])));
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlInitiationIntervalINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context, getMetadataFromNameAndParameter(
                      "llvm.loop.ii.count", LoopControlParameters[NumParam])));
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlMaxConcurrencyINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.max_concurrency.count",
                                        LoopControlParameters[NumParam])));
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlDependencyArrayINTELMask) {
    // Collect pointer variable <-> safelen information
    std::unordered_map<Value *, unsigned> PointerSflnMap;
    unsigned NumOperandPairs = LoopControlParameters[NumParam];
    unsigned OperandsEndIndex = NumParam + NumOperandPairs * 2;
    assert(OperandsEndIndex <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
    SPIRVModule *M = LM->getModule();
    while (NumParam < OperandsEndIndex) {
      SPIRVId ArraySPIRVId = LoopControlParameters[++NumParam];
      Value *PointerVar = ValueMap[M->getValue(ArraySPIRVId)];
      unsigned Safelen = LoopControlParameters[++NumParam];
      PointerSflnMap.emplace(PointerVar, Safelen);
    }

    // A single run over the loop to retrieve all GetElementPtr instructions
    // that access relevant array variables
    std::unordered_map<Value *, std::vector<GetElementPtrInst *>> ArrayGEPMap;
    for (const auto &BB : LoopObj->blocks()) {
      for (Instruction &I : *BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP)
          continue;

        Value *AccessedPointer = GEP->getPointerOperand();
        if (auto *BC = dyn_cast<CastInst>(AccessedPointer))
          if (BC->getSrcTy() == BC->getDestTy())
            AccessedPointer = BC->getOperand(0);
        if (auto *LI = dyn_cast<LoadInst>(AccessedPointer))
          AccessedPointer = LI->getPointerOperand();
        auto PointerSflnIt = PointerSflnMap.find(AccessedPointer);
        if (PointerSflnIt != PointerSflnMap.end()) {
          ArrayGEPMap[AccessedPointer].push_back(GEP);
        }
      }
    }

    // Create index group metadata nodes - one per each of the array
    // variables. Mark each GEP accessing a particular array variable
    // into a corresponding index group
    std::map<unsigned, SmallSet<MDNode *, 4>> SafelenIdxGroupMap;
    // Whenever a kernel closure field access is pointed to instead of
    // an array/pointer variable, ensure that all GEPs to that memory
    // share the same index group by hashing the newly added index groups.
    // "Memory offset info" represents a handle to the whole closure block
    // + an integer offset to a particular captured parameter.
    using MemoryOffsetInfo = std::pair<Value *, unsigned>;
    std::map<MemoryOffsetInfo, MDNode *> OffsetIdxGroupMap;

    for (auto &ArrayGEPIt : ArrayGEPMap) {
      MDNode *CurrentDepthIdxGroup = nullptr;
      if (auto *PrecedingGEP = dyn_cast<GetElementPtrInst>(ArrayGEPIt.first)) {
        Value *ClosureFieldPointer = PrecedingGEP->getPointerOperand();
        unsigned Offset =
            cast<ConstantInt>(PrecedingGEP->getOperand(2))->getZExtValue();
        MemoryOffsetInfo Info{ClosureFieldPointer, Offset};
        auto OffsetIdxGroupIt = OffsetIdxGroupMap.find(Info);
        if (OffsetIdxGroupIt == OffsetIdxGroupMap.end()) {
          // This is the first GEP encountered for this closure field.
          // Emit a distinct index group that will be referenced from
          // llvm.loop.parallel_access_indices metadata; hash the new
          // MDNode for future accesses to the same memory.
          CurrentDepthIdxGroup =
              llvm::MDNode::getDistinct(*Context, std::nullopt);
          OffsetIdxGroupMap.emplace(Info, CurrentDepthIdxGroup);
        } else {
          // Previous accesses to that field have already been indexed,
          // just use the already-existing metadata.
          CurrentDepthIdxGroup = OffsetIdxGroupIt->second;
        }
      } else /* Regular kernel-scope array/pointer variable */ {
        // Emit a distinct index group that will be referenced from
        // llvm.loop.parallel_access_indices metadata
        CurrentDepthIdxGroup =
            llvm::MDNode::getDistinct(*Context, std::nullopt);
      }

      unsigned Safelen = PointerSflnMap.find(ArrayGEPIt.first)->second;
      SafelenIdxGroupMap[Safelen].insert(CurrentDepthIdxGroup);
      for (auto *GEP : ArrayGEPIt.second) {
        StringRef IdxGroupMDName("llvm.index.group");
        llvm::MDNode *PreviousIdxGroup = GEP->getMetadata(IdxGroupMDName);
        if (!PreviousIdxGroup) {
          GEP->setMetadata(IdxGroupMDName, CurrentDepthIdxGroup);
          continue;
        }

        // If we're dealing with an embedded loop, it may be the case
        // that GEP instructions for some of the arrays were already
        // marked by the algorithm when it went over the outer level loops.
        // In order to retain the IVDep information for each "loop
        // dimension", we will mark such GEP's into a separate joined node
        // that will refer to the previous levels' index groups AND to the
        // index group specific to the current loop.
        std::vector<llvm::Metadata *> CurrentDepthOperands(
            PreviousIdxGroup->op_begin(), PreviousIdxGroup->op_end());
        if (CurrentDepthOperands.empty())
          CurrentDepthOperands.push_back(PreviousIdxGroup);
        CurrentDepthOperands.push_back(CurrentDepthIdxGroup);
        auto *JointIdxGroup = llvm::MDNode::get(*Context, CurrentDepthOperands);
        GEP->setMetadata(IdxGroupMDName, JointIdxGroup);
      }
    }

    for (auto &SflnIdxGroupIt : SafelenIdxGroupMap) {
      auto *Name = MDString::get(*Context, "llvm.loop.parallel_access_indices");
      unsigned SflnValue = SflnIdxGroupIt.first;
      llvm::Metadata *SafelenMDOp =
          SflnValue ? ConstantAsMetadata::get(ConstantInt::get(
                          Type::getInt32Ty(*Context), SflnValue))
                    : nullptr;
      std::vector<llvm::Metadata *> Parameters{Name};
      for (auto *Node : SflnIdxGroupIt.second)
        Parameters.push_back(Node);
      if (SafelenMDOp)
        Parameters.push_back(SafelenMDOp);
      Metadata.push_back(llvm::MDNode::get(*Context, Parameters));
    }
    ++NumParam;
  }
  if (LC & LoopControlPipelineEnableINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.intel.pipelining.enable",
                                        LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlLoopCoalesceINTELMask) {
    // If LoopCoalesce has a parameter of '0'
    if (!LoopControlParameters[NumParam]) {
      Metadata.push_back(llvm::MDNode::get(
          *Context, getMetadataFromName("llvm.loop.coalesce.enable")));
    } else {
      Metadata.push_back(llvm::MDNode::get(
          *Context,
          getMetadataFromNameAndParameter("llvm.loop.coalesce.count",
                                          LoopControlParameters[NumParam])));
    }
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlMaxInterleavingINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.max_interleaving.count",
                                        LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlSpeculatedIterationsINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context, getMetadataFromNameAndParameter(
                      "llvm.loop.intel.speculated.iterations.count",
                      LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlNoFusionINTELMask)
    Metadata.push_back(getMetadataFromName("llvm.loop.fusion.disable"));
  if (LC & spv::LoopControlLoopCountINTELMask) {
    // LoopCountINTELMask parameters are int64 and each parameter is stored
    // as 2 SPIRVWords (int32)
    assert(NumParam + 6 <= LoopControlParameters.size() &&
           "Missing loop control parameter!");

    uint64_t LoopCountMin =
        static_cast<uint64_t>(LoopControlParameters[NumParam++]);
    LoopCountMin |= static_cast<uint64_t>(LoopControlParameters[NumParam++])
                    << 32;
    if (static_cast<int64_t>(LoopCountMin) >= 0) {
      Metadata.push_back(getMetadataFromNameAndParameter(
          "llvm.loop.intel.loopcount_min", static_cast<int64_t>(LoopCountMin)));
    }

    uint64_t LoopCountMax =
        static_cast<uint64_t>(LoopControlParameters[NumParam++]);
    LoopCountMax |= static_cast<uint64_t>(LoopControlParameters[NumParam++])
                    << 32;
    if (static_cast<int64_t>(LoopCountMax) >= 0) {
      Metadata.push_back(getMetadataFromNameAndParameter(
          "llvm.loop.intel.loopcount_max", static_cast<int64_t>(LoopCountMax)));
    }

    uint64_t LoopCountAvg =
        static_cast<uint64_t>(LoopControlParameters[NumParam++]);
    LoopCountAvg |= static_cast<uint64_t>(LoopControlParameters[NumParam++])
                    << 32;
    if (static_cast<int64_t>(LoopCountAvg) >= 0) {
      Metadata.push_back(getMetadataFromNameAndParameter(
          "llvm.loop.intel.loopcount_avg", static_cast<int64_t>(LoopCountAvg)));
    }
  }
  if (LC & spv::LoopControlMaxReinvocationDelayINTELMask) {
    Metadata.push_back(llvm::MDNode::get(
        *Context, getMetadataFromNameAndParameter(
                      "llvm.loop.intel.max_reinvocation_delay.count",
                      LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  llvm::MDNode *Node = llvm::MDNode::get(*Context, Metadata);

  // Set the first operand to refer itself
  Node->replaceOperandWith(0, Node);
  LoopObj->setLoopID(Node);
}

void SPIRVToLLVM::transLLVMLoopMetadata(const Function *F) {
  assert(F);

  if (FuncLoopMetadataMap.empty())
    return;

  // Function declaration doesn't contain loop metadata.
  if (F->isDeclaration())
    return;

  DominatorTree DomTree(*(const_cast<Function *>(F)));
  LoopInfo LI(DomTree);

  // In SPIRV loop metadata is linked to a header basic block of a loop
  // whilst in LLVM IR it is linked to a latch basic block (the one
  // whose back edge goes to a header basic block) of the loop.
  // To ensure consistent behaviour, we can rely on the `llvm::Loop`
  // class to handle the metadata placement
  for (const auto *LoopObj : LI.getLoopsInPreorder()) {
    // Check that loop header BB contains loop metadata.
    const auto LMDItr = FuncLoopMetadataMap.find(LoopObj->getHeader());
    if (LMDItr == FuncLoopMetadataMap.end())
      continue;

    const auto *LMD = LMDItr->second;
    if (LMD->getOpCode() == OpLoopMerge) {
      const auto *LM = static_cast<const SPIRVLoopMerge *>(LMD);
      setLLVMLoopMetadata<SPIRVLoopMerge>(LM, LoopObj);
    } else if (LMD->getOpCode() == OpLoopControlINTEL) {
      const auto *LCI = static_cast<const SPIRVLoopControlINTEL *>(LMD);
      setLLVMLoopMetadata<SPIRVLoopControlINTEL>(LCI, LoopObj);
    }

    FuncLoopMetadataMap.erase(LMDItr);
  }
}

Value *SPIRVToLLVM::transValue(SPIRVValue *BV, Function *F, BasicBlock *BB,
                               bool CreatePlaceHolder) {
  SPIRVToLLVMValueMap::iterator Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end() && (!PlaceholderMap.count(BV) || CreatePlaceHolder))
    return Loc->second;

  SPIRVDBG(spvdbgs() << "[transValue] " << *BV << " -> ";)
  BV->validate();

  auto *V = transValueWithoutDecoration(BV, F, BB, CreatePlaceHolder);
  if (!V) {
    SPIRVDBG(dbgs() << " Warning ! nullptr\n";)
    return nullptr;
  }
  setName(V, BV);
  if (!transDecoration(BV, V)) {
    assert(0 && "trans decoration fail");
    return nullptr;
  }

  SPIRVDBG(dbgs() << *V << '\n';)

  return V;
}

Value *SPIRVToLLVM::transConvertInst(SPIRVValue *BV, Function *F,
                                     BasicBlock *BB) {
  SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
  auto *Src = transValue(BC->getOperand(0), F, BB, BB ? true : false);
  auto *Dst = transType(BC->getType());
  CastInst::CastOps CO = Instruction::BitCast;
  bool IsExt =
      Dst->getScalarSizeInBits() > Src->getType()->getScalarSizeInBits();
  switch (BC->getOpCode()) {
  case OpPtrCastToGeneric:
  case OpGenericCastToPtr:
  case OpPtrCastToCrossWorkgroupINTEL:
  case OpCrossWorkgroupCastToPtrINTEL: {
    // If module has pointers with DeviceOnlyINTEL and HostOnlyINTEL storage
    // classes there will be a situation, when global_device/global_host
    // address space will be lowered to just global address space. If there also
    // is an addrspacecast - we need to replace it with source pointer.
    if (Src->getType()->getPointerAddressSpace() ==
        Dst->getPointerAddressSpace())
      return Src;
    CO = Instruction::AddrSpaceCast;
    break;
  }
  case OpSConvert:
    CO = IsExt ? Instruction::SExt : Instruction::Trunc;
    break;
  case OpUConvert:
    CO = IsExt ? Instruction::ZExt : Instruction::Trunc;
    break;
  case OpFConvert:
    CO = IsExt ? Instruction::FPExt : Instruction::FPTrunc;
    break;
  case OpBitcast:
    // OpBitcast need to be handled as a special-case when the source is a
    // pointer and the destination is not a pointer, and where the source is not
    // a pointer and the destination is a pointer. This is supported by the
    // SPIR-V bitcast, but not by the LLVM bitcast.
    CO = Instruction::BitCast;
    if (Src->getType()->isPointerTy() && !Dst->isPointerTy()) {
      if (auto *DstVecTy = dyn_cast<FixedVectorType>(Dst)) {
        unsigned TotalBitWidth =
            DstVecTy->getElementType()->getIntegerBitWidth() *
            DstVecTy->getNumElements();
        auto *IntTy = Type::getIntNTy(Src->getContext(), TotalBitWidth);
        if (BB) {
          Src = CastInst::CreatePointerCast(Src, IntTy, "", BB);
        } else {
          Src = ConstantExpr::getPointerCast(dyn_cast<Constant>(Src), IntTy);
        }
      } else {
        CO = Instruction::PtrToInt;
      }
    } else if (!Src->getType()->isPointerTy() && Dst->isPointerTy()) {
      if (auto *SrcVecTy = dyn_cast<FixedVectorType>(Src->getType())) {
        unsigned TotalBitWidth =
            SrcVecTy->getElementType()->getIntegerBitWidth() *
            SrcVecTy->getNumElements();
        auto *IntTy = Type::getIntNTy(Src->getContext(), TotalBitWidth);
        if (BB) {
          Src = CastInst::Create(Instruction::BitCast, Src, IntTy, "", BB);
        } else {
          Src = ConstantExpr::getBitCast(dyn_cast<Constant>(Src), IntTy);
        }
      }
      CO = Instruction::IntToPtr;
    }
    break;
  default:
    CO = static_cast<CastInst::CastOps>(OpCodeMap::rmap(BC->getOpCode()));
  }
  assert(CastInst::isCast(CO) && "Invalid cast op code");
  SPIRVDBG(if (!CastInst::castIsValid(CO, Src, Dst)) {
    spvdbgs() << "Invalid cast: " << *BV << " -> ";
    dbgs() << "Op = " << CO << ", Src = " << *Src << " Dst = " << *Dst << '\n';
  })
  if (BB)
    return CastInst::Create(CO, Src, Dst, BV->getName(), BB);
  return ConstantExpr::getCast(CO, dyn_cast<Constant>(Src), Dst);
}

static void applyNoIntegerWrapDecorations(const SPIRVValue *BV,
                                          Instruction *Inst) {
  if (BV->hasDecorate(DecorationNoSignedWrap)) {
    Inst->setHasNoSignedWrap(true);
  }

  if (BV->hasDecorate(DecorationNoUnsignedWrap)) {
    Inst->setHasNoUnsignedWrap(true);
  }
}

static void applyFPFastMathModeDecorations(const SPIRVValue *BV,
                                           Instruction *Inst) {
  SPIRVWord V;
  FastMathFlags FMF;
  if (BV->hasDecorate(DecorationFPFastMathMode, 0, &V)) {
    if (V & FPFastMathModeNotNaNMask)
      FMF.setNoNaNs();
    if (V & FPFastMathModeNotInfMask)
      FMF.setNoInfs();
    if (V & FPFastMathModeNSZMask)
      FMF.setNoSignedZeros();
    if (V & FPFastMathModeAllowRecipMask)
      FMF.setAllowReciprocal();
    if (V & FPFastMathModeAllowContractFastINTELMask)
      FMF.setAllowContract();
    if (V & FPFastMathModeAllowReassocINTELMask)
      FMF.setAllowReassoc();
    if (V & FPFastMathModeFastMask)
      FMF.setFast();
    Inst->setFastMathFlags(FMF);
  }
}

Value *SPIRVToLLVM::transShiftLogicalBitwiseInst(SPIRVValue *BV, BasicBlock *BB,
                                                 Function *F) {
  SPIRVBinary *BBN = static_cast<SPIRVBinary *>(BV);
  if (BV->getType()->isTypeCooperativeMatrixKHR()) {
    return mapValue(BV, transSPIRVBuiltinFromInst(BBN, BB));
  }
  Instruction::BinaryOps BO;
  auto OP = BBN->getOpCode();
  if (isLogicalOpCode(OP))
    OP = IntBoolOpMap::rmap(OP);
  BO = static_cast<Instruction::BinaryOps>(OpCodeMap::rmap(OP));

  Value *Op0 = transValue(BBN->getOperand(0), F, BB);
  Value *Op1 = transValue(BBN->getOperand(1), F, BB);

  IRBuilder<> Builder(*Context);
  if (BB) {
    Builder.SetInsertPoint(BB);
  }

  Value *NewOp = Builder.CreateBinOp(BO, Op0, Op1, BV->getName());
  if (auto *Inst = dyn_cast<Instruction>(NewOp)) {
    applyNoIntegerWrapDecorations(BV, Inst);
    applyFPFastMathModeDecorations(BV, Inst);
  }
  return NewOp;
}

Value *SPIRVToLLVM::transCmpInst(SPIRVValue *BV, BasicBlock *BB, Function *F) {
  SPIRVCompare *BC = static_cast<SPIRVCompare *>(BV);
  SPIRVType *BT = BC->getOperand(0)->getType();
  Value *Inst = nullptr;
  auto OP = BC->getOpCode();
  if (isLogicalOpCode(OP))
    OP = IntBoolOpMap::rmap(OP);

  Value *Op0 = transValue(BC->getOperand(0), F, BB);
  Value *Op1 = transValue(BC->getOperand(1), F, BB);

  IRBuilder<> Builder(*Context);
  if (BB) {
    Builder.SetInsertPoint(BB);
  }

  if (OP == OpLessOrGreater)
    OP = OpFOrdNotEqual;

  if (BT->isTypeVectorOrScalarInt() || BT->isTypeVectorOrScalarBool() ||
      BT->isTypePointer())
    Inst = Builder.CreateICmp(CmpMap::rmap(OP), Op0, Op1);
  else if (BT->isTypeVectorOrScalarFloat())
    Inst = Builder.CreateFCmp(CmpMap::rmap(OP), Op0, Op1);
  assert(Inst && "not implemented");
  applyFPFastMathModeDecorations(BV, static_cast<Instruction *>(Inst));
  return Inst;
}

Type *SPIRVToLLVM::mapType(SPIRVType *BT, Type *T) {
  SPIRVDBG(dbgs() << *T << '\n';)
  // We don't want to store a TypedPointerType in the type map, since we can't
  // actually use it in LLVM IR directly. Note that in the cases where we do
  // want to construct TypedPointerType, we don't check the type map here.
  if (!isa<TypedPointerType>(T))
    TypeMap[BT] = T;
  return T;
}

Value *SPIRVToLLVM::mapValue(SPIRVValue *BV, Value *V) {
  auto Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end()) {
    if (Loc->second == V)
      return V;
    auto *LD = dyn_cast<LoadInst>(Loc->second);
    auto *Placeholder = dyn_cast<GlobalVariable>(LD->getPointerOperand());
    assert(LD && Placeholder &&
           Placeholder->getName().starts_with(KPlaceholderPrefix) &&
           "A value is translated twice");
    // Replaces placeholders for PHI nodes
    LD->replaceAllUsesWith(V);
    LD->eraseFromParent();
    Placeholder->eraseFromParent();
  }
  ValueMap[BV] = V;
  return V;
}

CallInst *
SPIRVToLLVM::expandOCLBuiltinWithScalarArg(CallInst *CI,
                                           const std::string &FuncName) {
  if (!CI->getOperand(0)->getType()->isVectorTy() &&
      CI->getOperand(1)->getType()->isVectorTy()) {
    auto VecElemCount =
        cast<VectorType>(CI->getOperand(1)->getType())->getElementCount();
    auto Mutator = mutateCallInst(CI, FuncName);
    Mutator.mapArg(0, [=](Value *Arg) {
      Value *NewVec = nullptr;
      if (auto *CA = dyn_cast<Constant>(Arg))
        NewVec = ConstantVector::getSplat(VecElemCount, CA);
      else {
        NewVec = ConstantVector::getSplat(
            VecElemCount, Constant::getNullValue(Arg->getType()));
        NewVec = InsertElementInst::Create(NewVec, Arg, getInt32(M, 0), "",
                                           CI->getIterator());
        NewVec = new ShuffleVectorInst(
            NewVec, NewVec,
            ConstantVector::getSplat(VecElemCount, getInt32(M, 0)), "",
            CI->getIterator());
      }
      NewVec->takeName(Arg);
      return NewVec;
    });
    return cast<CallInst>(Mutator.getMutated());
  }
  return CI;
}

std::string
SPIRVToLLVM::transOCLPipeTypeAccessQualifier(SPIRV::SPIRVTypePipe *ST) {
  return SPIRSPIRVAccessQualifierMap::rmap(ST->getAccessQualifier());
}

void SPIRVToLLVM::transGeneratorMD() {
  SPIRVMDBuilder B(*M);
  B.addNamedMD(kSPIRVMD::Generator)
      .addOp()
      .addU16(BM->getGeneratorId())
      .addU16(BM->getGeneratorVer())
      .done();
}

Value *SPIRVToLLVM::oclTransConstantSampler(SPIRV::SPIRVConstantSampler *BCS,
                                            BasicBlock *BB) {
  auto *SamplerT = getSPIRVType(OpTypeSampler, true);
  auto *I32Ty = IntegerType::getInt32Ty(*Context);
  auto *FTy = FunctionType::get(SamplerT, {I32Ty}, false);

  FunctionCallee Func = M->getOrInsertFunction(SAMPLER_INIT, FTy);

  auto Lit = (BCS->getAddrMode() << 1) | BCS->getNormalized() |
             ((BCS->getFilterMode() + 1) << 4);

  return CallInst::Create(Func, {ConstantInt::get(I32Ty, Lit)}, "", BB);
}

Value *SPIRVToLLVM::oclTransConstantPipeStorage(
    SPIRV::SPIRVConstantPipeStorage *BCPS) {

  std::string CPSName = std::string(kSPIRVTypeName::PrefixAndDelim) +
                        kSPIRVTypeName::ConstantPipeStorage;

  auto *Int32Ty = IntegerType::getInt32Ty(*Context);
  auto *CPSTy = StructType::getTypeByName(*Context, CPSName);
  if (!CPSTy) {
    Type *CPSElemsTy[] = {Int32Ty, Int32Ty, Int32Ty};
    CPSTy = StructType::create(*Context, CPSElemsTy, CPSName);
  }

  assert(CPSTy != nullptr && "Could not create spirv.ConstantPipeStorage");

  Constant *CPSElems[] = {ConstantInt::get(Int32Ty, BCPS->getPacketSize()),
                          ConstantInt::get(Int32Ty, BCPS->getPacketAlign()),
                          ConstantInt::get(Int32Ty, BCPS->getCapacity())};

  return new GlobalVariable(*M, CPSTy, false, GlobalValue::LinkOnceODRLinkage,
                            ConstantStruct::get(CPSTy, CPSElems),
                            BCPS->getName(), nullptr,
                            GlobalValue::NotThreadLocal, SPIRAS_Global);
}

// Translate aliasing memory access masks for SPIRVLoad and SPIRVStore
// instructions. These masks are mapped on alias.scope and noalias
// metadata in LLVM. Translation of optional string operand isn't yet supported
// in the translator.
template <typename SPIRVInstType>
void SPIRVToLLVM::transAliasingMemAccess(SPIRVInstType *BI, Instruction *I) {
  static_assert(std::is_same<SPIRVInstType, SPIRVStore>::value ||
                    std::is_same<SPIRVInstType, SPIRVLoad>::value,
                "Only stores and loads can be aliased by memory access mask");
  if (BI->SPIRVMemoryAccess::isNoAlias())
    addMemAliasMetadata(I, BI->SPIRVMemoryAccess::getNoAliasInstID(),
                        LLVMContext::MD_noalias);
  if (BI->SPIRVMemoryAccess::isAliasScope())
    addMemAliasMetadata(I, BI->SPIRVMemoryAccess::getAliasScopeInstID(),
                        LLVMContext::MD_alias_scope);
}

// Create and apply alias.scope/noalias metadata
void SPIRVToLLVM::addMemAliasMetadata(Instruction *I, SPIRVId AliasListId,
                                      uint32_t AliasMDKind) {
  SPIRVAliasScopeListDeclINTEL *AliasList =
      BM->get<SPIRVAliasScopeListDeclINTEL>(AliasListId);
  std::vector<SPIRVId> AliasScopeIds = AliasList->getArguments();
  MDBuilder MDB(*Context);
  SmallVector<Metadata *, 4> MDScopes;
  for (const auto ScopeId : AliasScopeIds) {
    SPIRVAliasScopeDeclINTEL *AliasScope =
        BM->get<SPIRVAliasScopeDeclINTEL>(ScopeId);
    std::vector<SPIRVId> AliasDomainIds = AliasScope->getArguments();
    // Currently we expect exactly one argument for aliasing scope
    // instruction.
    // TODO: add translation of string scope and domain operand.
    assert(AliasDomainIds.size() == 1 &&
           "AliasScopeDeclINTEL must have exactly one argument");
    SPIRVId AliasDomainId = AliasDomainIds[0];
    // Create and store unique domain and scope metadata
    MDAliasDomainMap.emplace(AliasDomainId,
                             MDB.createAnonymousAliasScopeDomain());
    MDAliasScopeMap.emplace(ScopeId, MDB.createAnonymousAliasScope(
                                         MDAliasDomainMap[AliasDomainId]));
    MDScopes.emplace_back(MDAliasScopeMap[ScopeId]);
  }
  // Create and store unique alias.scope/noalias metadata
  MDAliasListMap.emplace(AliasListId,
                         MDNode::concatenate(I->getMetadata(AliasMDKind),
                                             MDNode::get(*Context, MDScopes)));
  I->setMetadata(AliasMDKind, MDAliasListMap[AliasListId]);
}

void SPIRVToLLVM::transFunctionPointerCallArgumentAttributes(
    SPIRVValue *BV, CallInst *CI, SPIRVTypeFunction *CalledFnTy) {
  std::vector<SPIRVDecorate const *> ArgumentAttributes =
      BV->getDecorations(internal::DecorationArgumentAttributeINTEL);

  for (const auto *Dec : ArgumentAttributes) {
    std::vector<SPIRVWord> Literals = Dec->getVecLiteral();
    SPIRVWord ArgNo = Literals[0];
    SPIRVWord SpirvAttr = Literals[1];
    // There is no value to rmap SPIR-V FunctionParameterAttributeNoCapture, as
    // LLVM does not have Attribute::NoCapture anymore. Adding special handling
    // for this case.
    if (SpirvAttr == FunctionParameterAttributeNoCapture) {
      CI->addParamAttr(ArgNo, Attribute::getWithCaptureInfo(
                                  CI->getContext(), CaptureInfo::none()));
      continue;
    }
    Attribute::AttrKind LlvmAttrKind = SPIRSPIRVFuncParamAttrMap::rmap(
        static_cast<SPIRVFuncParamAttrKind>(SpirvAttr));
    auto LlvmAttr =
        Attribute::isTypeAttrKind(LlvmAttrKind)
            ? Attribute::get(CI->getContext(), LlvmAttrKind,
                             transType(CalledFnTy->getParameterType(ArgNo)
                                           ->getPointerElementType()))
            : Attribute::get(CI->getContext(), LlvmAttrKind);
    CI->addParamAttr(ArgNo, LlvmAttr);
  }
}

/// For instructions, this function assumes they are created in order
/// and appended to the given basic block. An instruction may use a
/// instruction from another BB which has not been translated. Such
/// instructions should be translated to place holders at the point
/// of first use, then replaced by real instructions when they are
/// created.
///
/// When CreatePlaceHolder is true, create a load instruction of a
/// global variable as placeholder for SPIRV instruction. Otherwise,
/// create instruction and replace placeholder if there is one.
Value *SPIRVToLLVM::transValueWithoutDecoration(SPIRVValue *BV, Function *F,
                                                BasicBlock *BB,
                                                bool CreatePlaceHolder) {

  auto OC = BV->getOpCode();
  IntBoolOpMap::rfind(OC, &OC);

  // Translation of non-instruction values
  switch (OC) {
  case OpConstant:
  case OpSpecConstant: {
    SPIRVConstant *BConst = static_cast<SPIRVConstant *>(BV);
    SPIRVType *BT = BV->getType();
    Type *LT = transType(BT);
    uint64_t ConstValue = BConst->getZExtIntValue();
    SPIRVWord SpecId = 0;
    if (OC == OpSpecConstant && BV->hasDecorate(DecorationSpecId, 0, &SpecId)) {
      // Update the value with possibly provided external specialization.
      if (BM->getSpecializationConstant(SpecId, ConstValue)) {
        assert(
            (BT->getBitWidth() == 64 ||
             (ConstValue >> BT->getBitWidth()) == 0) &&
            "Size of externally provided specialization constant value doesn't"
            "fit into the specialization constant type");
      }
    }
    switch (BT->getOpCode()) {
    case OpTypeBool:
    case OpTypeInt: {
      const unsigned NumBits = BT->getBitWidth();
      if (NumBits > 64) {
        // Translate huge arbitrary precision integer constants
        const unsigned RawDataNumWords = BConst->getNumWords();
        const unsigned BigValNumWords = (RawDataNumWords + 1) / 2;
        std::vector<uint64_t> BigValVec(BigValNumWords);
        const std::vector<SPIRVWord> &RawData = BConst->getSPIRVWords();
        // SPIRV words are integers of 32-bit width, meanwhile llvm::APInt
        // is storing data using an array of 64-bit words. Here we pack SPIRV
        // words into 64-bit integer array.
        for (size_t I = 0; I != RawDataNumWords / 2; ++I)
          BigValVec[I] =
              (static_cast<uint64_t>(RawData[2 * I + 1]) << SpirvWordBitWidth) |
              RawData[2 * I];
        if (RawDataNumWords % 2)
          BigValVec.back() = RawData.back();
        return mapValue(BV, ConstantInt::get(LT, APInt(NumBits, BigValVec)));
      }
      return mapValue(
          BV, ConstantInt::get(LT, ConstValue,
                               static_cast<SPIRVTypeInt *>(BT)->isSigned()));
    }
    case OpTypeFloat: {
      const llvm::fltSemantics *FS = nullptr;
      switch (BT->getFloatBitWidth()) {
      case 16:
        FS =
            (BT->isTypeFloat(16, FPEncodingBFloat16KHR) ? &APFloat::BFloat()
                                                        : &APFloat::IEEEhalf());
        break;
      case 32:
        FS = &APFloat::IEEEsingle();
        break;
      case 64:
        FS = &APFloat::IEEEdouble();
        break;
      default:
        llvm_unreachable("invalid floating-point type");
      }
      APFloat FPConstValue(*FS, APInt(BT->getFloatBitWidth(), ConstValue));
      return mapValue(BV, ConstantFP::get(*Context, FPConstValue));
    }
    default:
      llvm_unreachable("Not implemented");
      return nullptr;
    }
  }

  case OpConstantTrue:
    return mapValue(BV, ConstantInt::getTrue(*Context));

  case OpConstantFalse:
    return mapValue(BV, ConstantInt::getFalse(*Context));

  case OpSpecConstantTrue:
  case OpSpecConstantFalse: {
    bool IsTrue = OC == OpSpecConstantTrue;
    SPIRVWord SpecId = 0;
    if (BV->hasDecorate(DecorationSpecId, 0, &SpecId)) {
      uint64_t ConstValue = 0;
      if (BM->getSpecializationConstant(SpecId, ConstValue)) {
        IsTrue = ConstValue;
      }
    }
    return mapValue(BV, IsTrue ? ConstantInt::getTrue(*Context)
                               : ConstantInt::getFalse(*Context));
  }

  case OpConstantNull: {
    auto *LT = transType(BV->getType());
    return mapValue(BV, Constant::getNullValue(LT));
  }

  case OpConstantComposite:
  case OpSpecConstantComposite: {
    auto *BCC = static_cast<SPIRVConstantComposite *>(BV);
    std::vector<Constant *> CV;
    for (auto &I : BCC->getElements())
      CV.push_back(dyn_cast<Constant>(transValue(I, F, BB)));
    for (auto &CI : BCC->getContinuedInstructions()) {
      for (auto &I : CI->getElements())
        CV.push_back(dyn_cast<Constant>(transValue(I, F, BB)));
    }
    switch (BV->getType()->getOpCode()) {
    case OpTypeVector:
      return mapValue(BV, ConstantVector::get(CV));
    case OpTypeMatrix:
    case OpTypeArray: {
      auto *AT = cast<ArrayType>(transType(BCC->getType()));
      for (size_t I = 0; I != AT->getNumElements(); ++I) {
        auto *ElemTy = AT->getElementType();
        if (auto *ElemPtrTy = dyn_cast<PointerType>(ElemTy)) {
          assert(isa<PointerType>(CV[I]->getType()) &&
                 "Constant type doesn't match constexpr array element type");
          if (ElemPtrTy->getAddressSpace() !=
              cast<PointerType>(CV[I]->getType())->getAddressSpace())
            CV[I] = ConstantExpr::getAddrSpaceCast(CV[I], AT->getElementType());
        }
      }

      return mapValue(BV, ConstantArray::get(AT, CV));
    }
    case OpTypeStruct: {
      auto *BCCTy = cast<StructType>(transType(BCC->getType()));
      auto Members = BCCTy->getNumElements();
      auto Constants = CV.size();
      // if we try to initialize constant TypeStruct, add bitcasts
      // if src and dst types are both pointers but to different types
      if (Members == Constants) {
        for (unsigned I = 0; I < Members; ++I) {
          if (CV[I]->getType() == BCCTy->getElementType(I))
            continue;
          if (!CV[I]->getType()->isPointerTy() ||
              !BCCTy->getElementType(I)->isPointerTy())
            continue;

          if (cast<PointerType>(CV[I]->getType())->getAddressSpace() !=
              cast<PointerType>(BCCTy->getElementType(I))->getAddressSpace())
            CV[I] =
                ConstantExpr::getAddrSpaceCast(CV[I], BCCTy->getElementType(I));
          else
            CV[I] = ConstantExpr::getBitCast(CV[I], BCCTy->getElementType(I));
        }
      }

      return mapValue(BV, ConstantStruct::get(BCCTy, CV));
    }
    case OpTypeCooperativeMatrixKHR: {
      assert(CV.size() == 1 &&
             "expecting exactly one operand for cooperative matrix types");
      llvm::Type *RetTy = transType(BCC->getType());
      llvm::Type *EltTy = transType(
          static_cast<const SPIRVTypeCooperativeMatrixKHR *>(BV->getType())
              ->getCompType());
      auto *FTy = FunctionType::get(RetTy, {EltTy}, false);
      FunctionCallee Func =
          M->getOrInsertFunction(getSPIRVFuncName(OC, RetTy), FTy);
      IRBuilder<> Builder(BB);
      CallInst *Call = Builder.CreateCall(Func, CV.front());
      Call->setCallingConv(CallingConv::SPIR_FUNC);
      return Call;
    }
    default:
      llvm_unreachable("not implemented");
      return nullptr;
    }
  }

  case OpConstantSampler: {
    auto *BCS = static_cast<SPIRVConstantSampler *>(BV);
    // Intentially do not map this value. We want to generate constant
    // sampler initializer every time constant sampler is used, otherwise
    // initializer may not dominate all its uses.
    return oclTransConstantSampler(BCS, BB);
  }

  case OpConstantPipeStorage: {
    auto *BCPS = static_cast<SPIRVConstantPipeStorage *>(BV);
    return mapValue(BV, oclTransConstantPipeStorage(BCPS));
  }

  case OpSpecConstantOp: {
    auto *BI =
        createInstFromSpecConstantOp(static_cast<SPIRVSpecConstantOp *>(BV));
    return mapValue(BV, transValue(BI, nullptr, nullptr, false));
  }

  case OpConstantFunctionPointerINTEL: {
    SPIRVConstantFunctionPointerINTEL *BC =
        static_cast<SPIRVConstantFunctionPointerINTEL *>(BV);
    SPIRVFunction *F = BC->getFunction();
    BV->setName(F->getName());
    const unsigned AS = BM->shouldEmitFunctionPtrAddrSpace()
                            ? SPIRAS_CodeSectionINTEL
                            : SPIRAS_Private;
    return mapValue(BV, transFunction(F, AS));
  }

  case OpUndef:
    return mapValue(BV, UndefValue::get(transType(BV->getType())));

  case OpSizeOf: {
    Type *ResTy = transType(BV->getType());
    auto *BI = static_cast<SPIRVSizeOf *>(BV);
    SPIRVType *TypeArg = reinterpret_cast<SPIRVType *>(BI->getOpValue(0));
    Type *EltTy = transType(TypeArg->getPointerElementType());
    uint64_t Size = M->getDataLayout().getTypeStoreSize(EltTy).getFixedValue();
    return mapValue(BV, ConstantInt::get(ResTy, Size));
  }

  case OpVariable:
  case OpUntypedVariableKHR: {
    auto *BVar = static_cast<SPIRVVariableBase *>(BV);
    SPIRVType *PreTransTy = BVar->getType()->getPointerElementType();
    if (BVar->getType()->isTypeUntypedPointerKHR()) {
      auto *UntypedVar = static_cast<SPIRVUntypedVariableKHR *>(BVar);
      if (SPIRVType *DT = UntypedVar->getDataType())
        PreTransTy = DT;
    }
    auto *Ty = transType(PreTransTy);
    bool IsConst = BVar->isConstant();
    llvm::GlobalValue::LinkageTypes LinkageTy = transLinkageType(BVar);
    SPIRVStorageClassKind BS = BVar->getStorageClass();
    SPIRVValue *Init = BVar->getInitializer();

    if (PreTransTy->isTypeSampler() && BS == StorageClassUniformConstant) {
      // Skip generating llvm code during translation of a variable definition,
      // generate code only for its uses
      if (!BB)
        return nullptr;

      assert(Init && "UniformConstant OpVariable with sampler type must have "
                     "an initializer!");
      return transValue(Init, F, BB);
    }

    if (BS == StorageClassFunction) {
      // A Function storage class variable needs storage for each dynamic
      // execution instance, so emit an alloca instead of a global.
      assert(BB && "OpVariable with Function storage class requires BB");
      IRBuilder<> Builder(BB);
      AllocaInst *AI = Builder.CreateAlloca(Ty, nullptr, BV->getName());
      if (Init) {
        auto *Src = transValue(Init, F, BB);
        const bool IsVolatile = BVar->hasDecorate(DecorationVolatile);
        Builder.CreateStore(Src, AI, IsVolatile);
      }
      return mapValue(BV, AI);
    }

    SPIRAddressSpace AddrSpace;
    bool IsVectorCompute =
        BVar->hasDecorate(DecorationVectorComputeVariableINTEL);
    Constant *Initializer = nullptr;
    if (IsVectorCompute) {
      AddrSpace = VectorComputeUtil::getVCGlobalVarAddressSpace(BS);
      Initializer = PoisonValue::get(Ty);
    } else
      AddrSpace = SPIRSPIRVAddrSpaceMap::rmap(BS);
    // Force SPIRV BuiltIn variable's name to be __spirv_BuiltInXXXX.
    // No matter what BV's linkage name is.
    SPIRVBuiltinVariableKind BVKind;
    if (BVar->isBuiltin(&BVKind))
      BV->setName(prefixSPIRVName(SPIRVBuiltInNameMap::map(BVKind)));
    auto *LVar = new GlobalVariable(*M, Ty, IsConst, LinkageTy,
                                    /*Initializer=*/nullptr, BV->getName(), 0,
                                    GlobalVariable::NotThreadLocal, AddrSpace);
    auto *Res = mapValue(BV, LVar);
    if (Init)
      Initializer = dyn_cast<Constant>(transValue(Init, F, BB, false));
    else if (LinkageTy == GlobalValue::CommonLinkage)
      // In LLVM, variables with common linkage type must be initialized to 0.
      Initializer = Constant::getNullValue(Ty);
    else if (BS == SPIRVStorageClassKind::StorageClassWorkgroup &&
             LinkageTy != GlobalValue::ExternalLinkage)
      Initializer = dyn_cast<Constant>(PoisonValue::get(Ty));
    else if ((LinkageTy != GlobalValue::ExternalLinkage) &&
             (BS == SPIRVStorageClassKind::StorageClassCrossWorkgroup))
      Initializer = Constant::getNullValue(Ty);

    LVar->setUnnamedAddr((IsConst && Ty->isArrayTy() &&
                          Ty->getArrayElementType()->isIntegerTy(8))
                             ? GlobalValue::UnnamedAddr::Global
                             : GlobalValue::UnnamedAddr::None);
    LVar->setInitializer(Initializer);

    if (IsVectorCompute) {
      LVar->addAttribute(kVCMetadata::VCGlobalVariable);
      SPIRVWord Offset;
      if (BVar->hasDecorate(DecorationGlobalVariableOffsetINTEL, 0, &Offset))
        LVar->addAttribute(kVCMetadata::VCByteOffset, utostr(Offset));
      if (BVar->hasDecorate(DecorationVolatile))
        LVar->addAttribute(kVCMetadata::VCVolatile);
      auto SEVAttr = translateSEVMetadata(BVar, LVar->getContext());
      if (SEVAttr)
        LVar->addAttribute(SEVAttr.value().getKindAsString(),
                           SEVAttr.value().getValueAsString());
    }

    return Res;
  }

  case OpFunctionParameter: {
    auto *BA = static_cast<SPIRVFunctionParameter *>(BV);
    assert(F && "Invalid function");
    unsigned ArgNo = 0;
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
         ++I, ++ArgNo) {
      if (ArgNo == BA->getArgNo())
        return mapValue(BV, &(*I));
    }
    llvm_unreachable("Invalid argument");
    return nullptr;
  }

  case OpFunction:
    return mapValue(BV, transFunction(static_cast<SPIRVFunction *>(BV)));

  case OpAsmINTEL:
    return mapValue(BV, transAsmINTEL(static_cast<SPIRVAsmINTEL *>(BV)));

  case OpLabel:
    return mapValue(BV, BasicBlock::Create(*Context, BV->getName(), F));

  default:
    // do nothing
    break;
  }

  // During translation of OpSpecConstantOp we create an instruction
  // corresponding to the Opcode operand and then translate this instruction.
  // For such instruction BB and F should be nullptr, because it is a constant
  // expression declared out of scope of any basic block or function.
  // All other values require valid BB pointer.
  assert(((isSpecConstantOpAllowedOp(OC) && !F && !BB) || BB) && "Invalid BB");

  // Creation of place holder
  if (CreatePlaceHolder) {
    auto *Ty = transType(BV->getType());
    auto *GV =
        new GlobalVariable(*M, Ty, false, GlobalValue::PrivateLinkage, nullptr,
                           std::string(KPlaceholderPrefix) + BV->getName(), 0,
                           GlobalVariable::NotThreadLocal, 0);
    auto *LD = new LoadInst(Ty, GV, BV->getName(), BB);
    PlaceholderMap[BV] = LD;
    return mapValue(BV, LD);
  }

  // Translation of instructions
  int OpCode = BV->getOpCode();
  switch (OpCode) {
  case OpVariableLengthArrayINTEL: {
    auto *VLA = static_cast<SPIRVVariableLengthArrayINTEL *>(BV);
    llvm::Type *Ty = transType(BV->getType()->getPointerElementType());
    llvm::Value *ArrSize = transValue(VLA->getOperand(0), F, BB);
    return mapValue(BV,
                    new AllocaInst(Ty, M->getDataLayout().getAllocaAddrSpace(),
                                   ArrSize, BV->getName(), BB));
  }

  case OpRestoreMemoryINTEL: {
    IRBuilder<> Builder(BB);
    auto *Restore = static_cast<SPIRVRestoreMemoryINTEL *>(BV);
    llvm::Value *Ptr = transValue(Restore->getOperand(0), F, BB);
    auto *StackRestore = Builder.CreateStackRestore(Ptr);
    return mapValue(BV, StackRestore);
  }

  case OpSaveMemoryINTEL: {
    IRBuilder<> Builder(BB);
    auto *StackSave = Builder.CreateStackSave();
    return mapValue(BV, StackSave);
  }

  case OpBranch: {
    auto *BR = static_cast<SPIRVBranch *>(BV);
    auto *BI = BranchInst::Create(
        cast<BasicBlock>(transValue(BR->getTargetLabel(), F, BB)), BB);
    // Loop metadata will be translated in the end of function translation.
    return mapValue(BV, BI);
  }

  case OpBranchConditional: {
    auto *BR = static_cast<SPIRVBranchConditional *>(BV);
    auto *BC = BranchInst::Create(
        cast<BasicBlock>(transValue(BR->getTrueLabel(), F, BB)),
        cast<BasicBlock>(transValue(BR->getFalseLabel(), F, BB)),
        transValue(BR->getCondition(), F, BB), BB);
    // Loop metadata will be translated in the end of function translation.
    return mapValue(BV, BC);
  }

  case OpPhi: {
    auto *Phi = static_cast<SPIRVPhi *>(BV);
    auto *LPhi = dyn_cast<PHINode>(mapValue(
        BV, PHINode::Create(transType(Phi->getType()),
                            Phi->getPairs().size() / 2, Phi->getName(), BB)));
    Phi->foreachPair([&](SPIRVValue *IncomingV, SPIRVBasicBlock *IncomingBB,
                         size_t Index) {
      auto *Translated = transValue(IncomingV, F, BB);
      LPhi->addIncoming(Translated,
                        dyn_cast<BasicBlock>(transValue(IncomingBB, F, BB)));
    });
    return LPhi;
  }

  case OpUnreachable:
    return mapValue(BV, new UnreachableInst(*Context, BB));

  case OpReturn:
    return mapValue(BV, ReturnInst::Create(*Context, BB));

  case OpReturnValue: {
    auto *RV = static_cast<SPIRVReturnValue *>(BV);
    return mapValue(
        BV, ReturnInst::Create(*Context,
                               transValue(RV->getReturnValue(), F, BB), BB));
  }

  case OpLifetimeStart: {
    SPIRVLifetimeStart *LTStart = static_cast<SPIRVLifetimeStart *>(BV);
    IRBuilder<> Builder(BB);
    SPIRVWord Size = LTStart->getSize();
    ConstantInt *S = nullptr;
    if (Size)
      S = Builder.getInt64(Size);
    Value *Var = transValue(LTStart->getObject(), F, BB);
    CallInst *Start = Builder.CreateLifetimeStart(Var, S);
    return mapValue(BV, Start);
  }

  case OpLifetimeStop: {
    SPIRVLifetimeStop *LTStop = static_cast<SPIRVLifetimeStop *>(BV);
    IRBuilder<> Builder(BB);
    SPIRVWord Size = LTStop->getSize();
    ConstantInt *S = nullptr;
    if (Size)
      S = Builder.getInt64(Size);
    auto *Var = transValue(LTStop->getObject(), F, BB);
    for (const auto &I : Var->users())
      if (auto *II = getLifetimeStartIntrinsic(dyn_cast<Instruction>(I)))
        return mapValue(BV, Builder.CreateLifetimeEnd(II->getOperand(1), S));
    return mapValue(BV, Builder.CreateLifetimeEnd(Var, S));
  }

  case OpStore: {
    SPIRVStore *BS = static_cast<SPIRVStore *>(BV);
    StoreInst *SI = nullptr;
    auto *Src = transValue(BS->getSrc(), F, BB);
    auto *Dst = transValue(BS->getDst(), F, BB);

    bool isVolatile = BS->SPIRVMemoryAccess::isVolatile();
    uint64_t AlignValue = BS->SPIRVMemoryAccess::getAlignment();
    if (0 == AlignValue)
      SI = new StoreInst(Src, Dst, isVolatile, BB);
    else
      SI = new StoreInst(Src, Dst, isVolatile, Align(AlignValue), BB);
    if (BS->SPIRVMemoryAccess::isNonTemporal())
      transNonTemporalMetadata(SI);
    transAliasingMemAccess<SPIRVStore>(BS, SI);
    return mapValue(BV, SI);
  }

  case OpLoad: {
    SPIRVLoad *BL = static_cast<SPIRVLoad *>(BV);
    auto *V = transValue(BL->getSrc(), F, BB);

    Type *Ty = transType(BL->getType());
    LoadInst *LI = nullptr;
    uint64_t AlignValue = BL->SPIRVMemoryAccess::getAlignment();
    if (0 == AlignValue) {
      LI = new LoadInst(Ty, V, BV->getName(),
                        BL->SPIRVMemoryAccess::isVolatile(), BB);
    } else {
      LI = new LoadInst(Ty, V, BV->getName(),
                        BL->SPIRVMemoryAccess::isVolatile(), Align(AlignValue),
                        BB);
    }
    if (BL->SPIRVMemoryAccess::isNonTemporal())
      transNonTemporalMetadata(LI);
    transAliasingMemAccess<SPIRVLoad>(BL, LI);
    return mapValue(BV, LI);
  }

  case OpCopyMemory: {
    auto *BC = static_cast<SPIRVCopyMemory *>(BV);
    llvm::Value *Dst = transValue(BC->getTarget(), F, BB);
    MaybeAlign Align(BC->getAlignment());
    MaybeAlign SrcAlign =
        BC->getSrcAlignment() ? MaybeAlign(BC->getSrcAlignment()) : Align;
    Type *EltTy =
        transType(BC->getSource()->getType()->getPointerElementType());
    uint64_t Size = M->getDataLayout().getTypeStoreSize(EltTy).getFixedValue();
    bool IsVolatile = BC->SPIRVMemoryAccess::isVolatile();
    IRBuilder<> Builder(BB);

    llvm::Value *Src = transValue(BC->getSource(), F, BB);
    CallInst *CI =
        Builder.CreateMemCpy(Dst, Align, Src, SrcAlign, Size, IsVolatile);
    if (isFuncNoUnwind())
      CI->getFunction()->addFnAttr(Attribute::NoUnwind);
    return mapValue(BV, CI);
  }

  case OpCopyMemorySized: {
    SPIRVCopyMemorySized *BC = static_cast<SPIRVCopyMemorySized *>(BV);
    llvm::Value *Dst = transValue(BC->getTarget(), F, BB);
    MaybeAlign Align(BC->getAlignment());
    MaybeAlign SrcAlign =
        BC->getSrcAlignment() ? MaybeAlign(BC->getSrcAlignment()) : Align;
    llvm::Value *Size = transValue(BC->getSize(), F, BB);
    bool IsVolatile = BC->SPIRVMemoryAccess::isVolatile();
    IRBuilder<> Builder(BB);

    llvm::Value *Src = transValue(BC->getSource(), F, BB);
    CallInst *CI =
        Builder.CreateMemCpy(Dst, Align, Src, SrcAlign, Size, IsVolatile);
    if (isFuncNoUnwind())
      CI->getFunction()->addFnAttr(Attribute::NoUnwind);
    return mapValue(BV, CI);
  }

  case OpSelect: {
    SPIRVSelect *BS = static_cast<SPIRVSelect *>(BV);
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    return mapValue(BV,
                    Builder.CreateSelect(transValue(BS->getCondition(), F, BB),
                                         transValue(BS->getTrueValue(), F, BB),
                                         transValue(BS->getFalseValue(), F, BB),
                                         BV->getName()));
  }

  case OpLine:
  case OpSelectionMerge: // OpenCL Compiler does not use this instruction
    return nullptr;

  case OpLoopMerge:        // Will be translated after all other function's
  case OpLoopControlINTEL: // instructions are translated.
    FuncLoopMetadataMap[BB] = BV;
    return nullptr;

  case OpSwitch: {
    auto *BS = static_cast<SPIRVSwitch *>(BV);
    auto *Select = transValue(BS->getSelect(), F, BB);
    auto *LS = SwitchInst::Create(
        Select, dyn_cast<BasicBlock>(transValue(BS->getDefault(), F, BB)),
        BS->getNumPairs(), BB);
    BS->foreachPair(
        [&](SPIRVSwitch::LiteralTy Literals, SPIRVBasicBlock *Label) {
          assert(!Literals.empty() && "Literals should not be empty");
          assert(Literals.size() <= 2 &&
                 "Number of literals should not be more then two");
          uint64_t Literal = uint64_t(Literals.at(0));
          if (Literals.size() == 2) {
            Literal += uint64_t(Literals.at(1)) << 32;
          }
          LS->addCase(
              ConstantInt::get(cast<IntegerType>(Select->getType()), Literal),
              cast<BasicBlock>(transValue(Label, F, BB)));
        });
    return mapValue(BV, LS);
  }

  case OpVectorTimesScalar: {
    auto *VTS = static_cast<SPIRVVectorTimesScalar *>(BV);
    IRBuilder<> Builder(BB);
    auto *Scalar = transValue(VTS->getScalar(), F, BB);
    auto *Vector = transValue(VTS->getVector(), F, BB);
    auto *VecTy = cast<FixedVectorType>(Vector->getType());
    unsigned VecSize = VecTy->getNumElements();
    auto *NewVec =
        Builder.CreateVectorSplat(VecSize, Scalar, Scalar->getName());
    NewVec->takeName(Scalar);
    auto *Scale = Builder.CreateFMul(Vector, NewVec, "scale");
    return mapValue(BV, Scale);
  }

  case OpVectorTimesMatrix: {
    auto *VTM = static_cast<SPIRVVectorTimesMatrix *>(BV);
    IRBuilder<> Builder(BB);
    Value *Mat = transValue(VTM->getMatrix(), F, BB);
    Value *Vec = transValue(VTM->getVector(), F, BB);

    // Vec is of N elements.
    // Mat is of M columns and N rows.
    // Mat consists of vectors: V_1, V_2, ..., V_M
    //
    // The product is:
    //
    //                |------- M ----------|
    // Result = sum ( {Vec_1, Vec_1, ..., Vec_1} * {V_1_1, V_2_1, ..., V_M_1},
    //                {Vec_2, Vec_2, ..., Vec_2} * {V_1_2, V_2_2, ..., V_M_2},
    //                ...
    //                {Vec_N, Vec_N, ..., Vec_N} * {V_1_N, V_2_N, ..., V_M_N});

    unsigned M = Mat->getType()->getArrayNumElements();

    auto *VecTy = cast<FixedVectorType>(Vec->getType());
    FixedVectorType *VTy = FixedVectorType::get(VecTy->getElementType(), M);
    auto *ETy = VTy->getElementType();
    unsigned N = VecTy->getNumElements();
    Value *V = Builder.CreateVectorSplat(M, ConstantFP::get(ETy, 0.0));

    for (unsigned Idx = 0; Idx != N; ++Idx) {
      Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(Idx));
      Value *Lhs = Builder.CreateVectorSplat(M, S);
      Value *Rhs = PoisonValue::get(VTy);
      for (unsigned Idx2 = 0; Idx2 != M; ++Idx2) {
        Value *Vx = Builder.CreateExtractValue(Mat, Idx2);
        Value *Vxi = Builder.CreateExtractElement(Vx, Builder.getInt32(Idx));
        Rhs = Builder.CreateInsertElement(Rhs, Vxi, Builder.getInt32(Idx2));
      }
      Value *Mul = Builder.CreateFMul(Lhs, Rhs);
      V = Builder.CreateFAdd(V, Mul);
    }

    return mapValue(BV, V);
  }

  case OpMatrixTimesScalar: {
    auto *MTS = static_cast<SPIRVMatrixTimesScalar *>(BV);
    IRBuilder<> Builder(BB);
    auto *Scalar = transValue(MTS->getScalar(), F, BB);
    auto *Matrix = transValue(MTS->getMatrix(), F, BB);
    uint64_t ColNum = Matrix->getType()->getArrayNumElements();
    auto *ColType = cast<ArrayType>(Matrix->getType())->getElementType();
    auto VecSize = cast<FixedVectorType>(ColType)->getNumElements();
    auto *NewVec =
        Builder.CreateVectorSplat(VecSize, Scalar, Scalar->getName());
    NewVec->takeName(Scalar);

    Value *V = PoisonValue::get(Matrix->getType());
    for (uint64_t Idx = 0; Idx != ColNum; Idx++) {
      auto *Col = Builder.CreateExtractValue(Matrix, Idx);
      auto *I = Builder.CreateFMul(Col, NewVec);
      V = Builder.CreateInsertValue(V, I, Idx);
    }

    return mapValue(BV, V);
  }

  case OpMatrixTimesVector: {
    auto *MTV = static_cast<SPIRVMatrixTimesVector *>(BV);
    IRBuilder<> Builder(BB);
    Value *Mat = transValue(MTV->getMatrix(), F, BB);
    Value *Vec = transValue(MTV->getVector(), F, BB);

    // Result is similar to Matrix * Matrix
    // Mat is of M columns and N rows.
    // Mat consists of vectors: V_1, V_2, ..., V_M
    // where each vector is of size N.
    //
    // Vec is of size M.
    // The product is a vector of size N.
    //
    //                |------- N ----------|
    // Result = sum ( {Vec_1, Vec_1, ..., Vec_1} * V_1,
    //                {Vec_2, Vec_2, ..., Vec_2} * V_2,
    //                ...
    //                {Vec_M, Vec_M, ..., Vec_M} * V_N );
    //
    // where sum is defined as vector sum.

    unsigned M = Mat->getType()->getArrayNumElements();
    FixedVectorType *VTy = cast<FixedVectorType>(
        cast<ArrayType>(Mat->getType())->getElementType());
    unsigned N = VTy->getNumElements();
    auto *ETy = VTy->getElementType();
    Value *V = Builder.CreateVectorSplat(N, ConstantFP::get(ETy, 0.0));

    for (unsigned Idx = 0; Idx != M; ++Idx) {
      Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(Idx));
      Value *Lhs = Builder.CreateVectorSplat(N, S);
      Value *Vx = Builder.CreateExtractValue(Mat, Idx);
      Value *Mul = Builder.CreateFMul(Lhs, Vx);
      V = Builder.CreateFAdd(V, Mul);
    }

    return mapValue(BV, V);
  }

  case OpMatrixTimesMatrix: {
    auto *MTM = static_cast<SPIRVMatrixTimesMatrix *>(BV);
    IRBuilder<> Builder(BB);
    Value *M1 = transValue(MTM->getLeftMatrix(), F, BB);
    Value *M2 = transValue(MTM->getRightMatrix(), F, BB);

    // Each matrix consists of a list of columns.
    // M1 (the left matrix) is of C1 columns and R1 rows.
    // M1 consists of a list of vectors: V_1, V_2, ..., V_C1
    // where V_x are vectors of size R1.
    //
    // M2 (the right matrix) is of C2 columns and R2 rows.
    // M2 consists of a list of vectors: U_1, U_2, ..., U_C2
    // where U_x are vectors of size R2.
    //
    // Now M1 * M2 requires C1 == R2.
    // The result is a matrix of C2 columns and R1 rows.
    // That is, consists of C2 vectors of size R1.
    //
    // M1 * M2 algorithm is as below:
    //
    // Result = { dot_product(U_1, M1),
    //            dot_product(U_2, M1),
    //            ...
    //            dot_product(U_C2, M1) };
    // where
    // dot_product (U, M) is defined as:
    //
    //                 |-------- C1 ------|
    // Result = sum ( {U[1], U[1], ..., U[1]} * V_1,
    //                {U[2], U[2], ..., U[2]} * V_2,
    //                ...
    //                {U[R2], U[R2], ..., U[R2]} * V_C1 );
    // Note that C1 == R2
    // sum is defined as vector sum.

    unsigned C1 = M1->getType()->getArrayNumElements();
    unsigned C2 = M2->getType()->getArrayNumElements();
    FixedVectorType *V1Ty =
        cast<FixedVectorType>(cast<ArrayType>(M1->getType())->getElementType());
    FixedVectorType *V2Ty =
        cast<FixedVectorType>(cast<ArrayType>(M2->getType())->getElementType());
    unsigned R1 = V1Ty->getNumElements();
    unsigned R2 = V2Ty->getNumElements();
    auto *ETy = V1Ty->getElementType();

    (void)C1;
    assert(C1 == R2 && "Unmatched matrix");

    auto *VTy = FixedVectorType::get(ETy, R1);
    auto *ResultTy = ArrayType::get(VTy, C2);

    Value *Res = PoisonValue::get(ResultTy);

    for (unsigned Idx = 0; Idx != C2; ++Idx) {
      Value *U = Builder.CreateExtractValue(M2, Idx);

      // Calculate dot_product(U, M1)
      Value *Dot = Builder.CreateVectorSplat(R1, ConstantFP::get(ETy, 0.0));

      for (unsigned Idx2 = 0; Idx2 != R2; ++Idx2) {
        Value *Ux = Builder.CreateExtractElement(U, Builder.getInt32(Idx2));
        Value *Lhs = Builder.CreateVectorSplat(R1, Ux);
        Value *Rhs = Builder.CreateExtractValue(M1, Idx2);
        Value *Mul = Builder.CreateFMul(Lhs, Rhs);
        Dot = Builder.CreateFAdd(Dot, Mul);
      }

      Res = Builder.CreateInsertValue(Res, Dot, Idx);
    }

    return mapValue(BV, Res);
  }

  case OpTranspose: {
    auto *TR = static_cast<SPIRVTranspose *>(BV);
    IRBuilder<> Builder(BB);
    auto *Matrix = transValue(TR->getMatrix(), F, BB);
    unsigned ColNum = Matrix->getType()->getArrayNumElements();
    FixedVectorType *ColTy = cast<FixedVectorType>(
        cast<ArrayType>(Matrix->getType())->getElementType());
    unsigned RowNum = ColTy->getNumElements();

    auto *VTy = FixedVectorType::get(ColTy->getElementType(), ColNum);
    auto *ResultTy = ArrayType::get(VTy, RowNum);
    Value *V = PoisonValue::get(ResultTy);

    SmallVector<Value *, 16> MCache;
    MCache.reserve(ColNum);
    for (unsigned Idx = 0; Idx != ColNum; ++Idx)
      MCache.push_back(Builder.CreateExtractValue(Matrix, Idx));

    if (ColNum == RowNum) {
      // Fastpath
      switch (ColNum) {
      case 2: {
        Value *V1 = Builder.CreateShuffleVector(MCache[0], MCache[1],
                                                ArrayRef<int>({0, 2}));
        V = Builder.CreateInsertValue(V, V1, 0);
        Value *V2 = Builder.CreateShuffleVector(MCache[0], MCache[1],
                                                ArrayRef<int>({1, 3}));
        V = Builder.CreateInsertValue(V, V2, 1);
        return mapValue(BV, V);
      }

      case 4: {
        for (int Idx = 0; Idx < 4; ++Idx) {
          Value *V1 = Builder.CreateShuffleVector(MCache[0], MCache[1],
                                                  ArrayRef<int>{Idx, Idx + 4});
          Value *V2 = Builder.CreateShuffleVector(MCache[2], MCache[3],
                                                  ArrayRef<int>{Idx, Idx + 4});
          Value *V3 =
              Builder.CreateShuffleVector(V1, V2, ArrayRef<int>({0, 1, 2, 3}));
          V = Builder.CreateInsertValue(V, V3, Idx);
        }
        return mapValue(BV, V);
      }

      default:
        break;
      }
    }

    // Slowpath
    for (unsigned Idx = 0; Idx != RowNum; ++Idx) {
      Value *Vec = PoisonValue::get(VTy);

      for (unsigned Idx2 = 0; Idx2 != ColNum; ++Idx2) {
        Value *S =
            Builder.CreateExtractElement(MCache[Idx2], Builder.getInt32(Idx));
        Vec = Builder.CreateInsertElement(Vec, S, Idx2);
      }

      V = Builder.CreateInsertValue(V, Vec, Idx);
    }

    return mapValue(BV, V);
  }

  case OpCopyObject: {
    SPIRVCopyObject *CO = static_cast<SPIRVCopyObject *>(BV);
    auto *Ty = transType(CO->getOperand()->getType());
    AllocaInst *AI =
        new AllocaInst(Ty, M->getDataLayout().getAllocaAddrSpace(), "", BB);
    new StoreInst(transValue(CO->getOperand(), F, BB), AI, BB);
    LoadInst *LI = new LoadInst(Ty, AI, "", BB);
    return mapValue(BV, LI);
  }
  case OpCopyLogical: {
    SPIRVCopyLogical *CL = static_cast<SPIRVCopyLogical *>(BV);

    auto *SrcTy = transType(CL->getOperand()->getType());
    auto *DstTy = transType(CL->getType());

    assert(M->getDataLayout().getTypeStoreSize(SrcTy).getFixedValue() ==
               M->getDataLayout().getTypeStoreSize(DstTy).getFixedValue() &&
           "Size mismatch in OpCopyLogical");

    IRBuilder<> Builder(BB);

    auto *SrcAI = Builder.CreateAlloca(SrcTy);
    Builder.CreateAlignedStore(transValue(CL->getOperand(), F, BB), SrcAI,
                               SrcAI->getAlign());

    auto *LI = Builder.CreateAlignedLoad(DstTy, SrcAI, SrcAI->getAlign());
    return mapValue(BV, LI);
  }

  case OpAccessChain:
  case OpInBoundsAccessChain:
  case OpPtrAccessChain:
  case OpInBoundsPtrAccessChain:
  case OpUntypedAccessChainKHR:
  case OpUntypedInBoundsAccessChainKHR:
  case OpUntypedPtrAccessChainKHR:
  case OpUntypedInBoundsPtrAccessChainKHR: {
    auto *AC = static_cast<SPIRVAccessChainBase *>(BV);
    auto *Base = transValue(AC->getBase(), F, BB);
    SPIRVType *BaseSPVTy = AC->getBaseType();
    if ((BaseSPVTy->isTypePointer() &&
         BaseSPVTy->getPointerElementType()->isTypeCooperativeMatrixKHR()) ||
        (isUntypedAccessChainOpCode(OC) &&
         BaseSPVTy->isTypeCooperativeMatrixKHR())) {
      return mapValue(BV, transSPIRVBuiltinFromInst(AC, BB));
    }
    Type *BaseTy =
        BaseSPVTy->isTypeVector()
            ? transType(
                  BaseSPVTy->getVectorComponentType()->getPointerElementType())
        : BaseSPVTy->isTypePointer()
            ? transType(BaseSPVTy->getPointerElementType())
            : transType(BaseSPVTy);
    auto Index = transValue(AC->getIndices(), F, BB);
    if (!AC->hasPtrIndex())
      Index.insert(Index.begin(), getInt32(M, 0));
    auto IsInbound = AC->isInBounds();
    Value *V = nullptr;

    if (GEPOrUseMap.count(Base)) {
      auto IdxToInstMap = GEPOrUseMap[Base];
      auto Idx = AC->getIndices();

      // In transIntelFPGADecorations we generated GEPs only for the fields of
      // structure, meaning that GEP to `0` accesses the Structure itself, and
      // the second `Id` is a Key in the map.
      if (Idx.size() == 2) {
        unsigned Idx1 = static_cast<ConstantInt *>(getTranslatedValue(Idx[0]))
                            ->getZExtValue();
        if (Idx1 == 0) {
          unsigned Idx2 = static_cast<ConstantInt *>(getTranslatedValue(Idx[1]))
                              ->getZExtValue();

          // If we already have the instruction in a map, use it.
          if (IdxToInstMap.count(Idx2))
            return mapValue(BV, IdxToInstMap[Idx2]);
        }
      }
    }

    if (BB) {
      auto *GEP =
          GetElementPtrInst::Create(BaseTy, Base, Index, BV->getName(), BB);
      GEP->setIsInBounds(IsInbound);
      V = GEP;
    } else {
      auto *CT = cast<Constant>(Base);
      V = ConstantExpr::getGetElementPtr(BaseTy, CT, Index, IsInbound);
    }
    return mapValue(BV, V);
  }

  case OpPtrEqual:
  case OpPtrNotEqual: {
    auto *BC = static_cast<SPIRVBinary *>(BV);
    auto Ops = transValue(BC->getOperands(), F, BB);

    IRBuilder<> Builder(BB);
    Value *Op1 = Builder.CreatePtrToInt(Ops[0], Type::getInt64Ty(*Context));
    Value *Op2 = Builder.CreatePtrToInt(Ops[1], Type::getInt64Ty(*Context));
    CmpInst::Predicate P =
        OC == OpPtrEqual ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE;
    Value *V = Builder.CreateICmp(P, Op1, Op2);
    return mapValue(BV, V);
  }

  case OpPtrDiff: {
    auto *BC = static_cast<SPIRVBinary *>(BV);
    auto SPVOps = BC->getOperands();
    auto Ops = transValue(SPVOps, F, BB);
    IRBuilder<> Builder(BB);

    Type *ElemTy = nullptr;
    if (SPVOps[0]->isUntypedVariable())
      ElemTy = transType(
          static_cast<SPIRVUntypedVariableKHR *>(SPVOps[0])->getDataType());
    else
      ElemTy = transType(SPVOps[0]->getType()->getPointerElementType());

    Value *V = Builder.CreatePtrDiff(ElemTy, Ops[0], Ops[1]);
    return mapValue(BV, V);
  }

  case OpCompositeConstruct: {
    auto *CC = static_cast<SPIRVCompositeConstruct *>(BV);
    auto Constituents = transValue(CC->getOperands(), F, BB);
    std::vector<Constant *> CV;
    bool HasRtValues = false;
    for (const auto &I : Constituents) {
      auto *C = dyn_cast<Constant>(I);
      CV.push_back(C);
      if (!HasRtValues && C == nullptr)
        HasRtValues = true;
    }

    switch (static_cast<size_t>(BV->getType()->getOpCode())) {
    case OpTypeVector: {
      if (!HasRtValues)
        return mapValue(BV, ConstantVector::get(CV));

      auto *VT = cast<FixedVectorType>(transType(CC->getType()));
      Value *NewVec = ConstantVector::getSplat(
          VT->getElementCount(), PoisonValue::get(VT->getElementType()));

      for (size_t I = 0; I < Constituents.size(); I++) {
        NewVec = InsertElementInst::Create(NewVec, Constituents[I],
                                           getInt32(M, I), "", BB);
      }
      return mapValue(BV, NewVec);
    }
    case OpTypeArray: {
      auto *AT = cast<ArrayType>(transType(CC->getType()));
      if (!HasRtValues)
        return mapValue(BV, ConstantArray::get(AT, CV));

      AllocaInst *Alloca =
          new AllocaInst(AT, M->getDataLayout().getAllocaAddrSpace(), "", BB);

      // get pointer to the element of the array
      // store the result of argument
      for (size_t I = 0; I < Constituents.size(); I++) {
        auto *GEP = GetElementPtrInst::Create(
            AT, Alloca, {getInt32(M, 0), getInt32(M, I)}, "gep", BB);
        GEP->setIsInBounds(true);
        new StoreInst(Constituents[I], GEP, false, BB);
      }

      auto *Load = new LoadInst(AT, Alloca, "load", false, BB);
      return mapValue(BV, Load);
    }
    case OpTypeStruct: {
      auto *ST = cast<StructType>(transType(CC->getType()));
      if (!HasRtValues)
        return mapValue(BV, ConstantStruct::get(ST, CV));

      AllocaInst *Alloca =
          new AllocaInst(ST, M->getDataLayout().getAllocaAddrSpace(), "", BB);

      // get pointer to the element of structure
      // store the result of argument
      for (size_t I = 0; I < Constituents.size(); I++) {
        auto *GEP = GetElementPtrInst::Create(
            ST, Alloca, {getInt32(M, 0), getInt32(M, I)}, "gep", BB);
        GEP->setIsInBounds(true);
        new StoreInst(Constituents[I], GEP, false, BB);
      }

      auto *Load = new LoadInst(ST, Alloca, "load", false, BB);
      return mapValue(BV, Load);
    }
    case internal::OpTypeJointMatrixINTEL:
    case OpTypeCooperativeMatrixKHR:
    case internal::OpTypeTaskSequenceINTEL:
      return mapValue(BV, transSPIRVBuiltinFromInst(CC, BB));
    default:
      llvm_unreachable("Unhandled type!");
    }
  }

  case OpCompositeExtract: {
    SPIRVCompositeExtract *CE = static_cast<SPIRVCompositeExtract *>(BV);
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    if (CE->getComposite()->getType()->isTypeVector()) {
      assert(CE->getIndices().size() == 1 && "Invalid index");
      return mapValue(
          BV, Builder.CreateExtractElement(
                  transValue(CE->getComposite(), F, BB),
                  ConstantInt::get(*Context, APInt(32, CE->getIndices()[0])),
                  BV->getName()));
    }
    return mapValue(
        BV, Builder.CreateExtractValue(transValue(CE->getComposite(), F, BB),
                                       CE->getIndices(), BV->getName()));
  }

  case OpVectorExtractDynamic: {
    auto *VED = static_cast<SPIRVVectorExtractDynamic *>(BV);
    SPIRVValue *Vec = VED->getVector();
    if (Vec->getType()->getOpCode() == internal::OpTypeJointMatrixINTEL) {
      return mapValue(BV, transSPIRVBuiltinFromInst(VED, BB));
    }
    return mapValue(
        BV, ExtractElementInst::Create(transValue(Vec, F, BB),
                                       transValue(VED->getIndex(), F, BB),
                                       BV->getName(), BB));
  }

  case OpCompositeInsert: {
    auto *CI = static_cast<SPIRVCompositeInsert *>(BV);
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    if (CI->getComposite()->getType()->isTypeVector()) {
      assert(CI->getIndices().size() == 1 && "Invalid index");
      return mapValue(
          BV, Builder.CreateInsertElement(
                  transValue(CI->getComposite(), F, BB),
                  transValue(CI->getObject(), F, BB),
                  ConstantInt::get(*Context, APInt(32, CI->getIndices()[0])),
                  BV->getName()));
    }
    return mapValue(
        BV, Builder.CreateInsertValue(transValue(CI->getComposite(), F, BB),
                                      transValue(CI->getObject(), F, BB),
                                      CI->getIndices(), BV->getName()));
  }

  case OpVectorInsertDynamic: {
    auto *VID = static_cast<SPIRVVectorInsertDynamic *>(BV);
    SPIRVValue *Vec = VID->getVector();
    if (Vec->getType()->getOpCode() == internal::OpTypeJointMatrixINTEL) {
      return mapValue(BV, transSPIRVBuiltinFromInst(VID, BB));
    }
    return mapValue(
        BV, InsertElementInst::Create(
                transValue(Vec, F, BB), transValue(VID->getComponent(), F, BB),
                transValue(VID->getIndex(), F, BB), BV->getName(), BB));
  }

  case OpVectorShuffle: {
    auto *VS = static_cast<SPIRVVectorShuffle *>(BV);
    std::vector<Constant *> Components;
    IntegerType *Int32Ty = IntegerType::get(*Context, 32);
    for (auto I : VS->getComponents()) {
      if (I == static_cast<SPIRVWord>(-1))
        Components.push_back(PoisonValue::get(Int32Ty));
      else
        Components.push_back(ConstantInt::get(Int32Ty, I));
    }
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    Value *Vec1 = transValue(VS->getVector1(), F, BB);
    Value *Vec2 = transValue(VS->getVector2(), F, BB);
    auto *Vec1Ty = cast<FixedVectorType>(Vec1->getType());
    auto *Vec2Ty = cast<FixedVectorType>(Vec2->getType());
    if (Vec1Ty->getNumElements() != Vec2Ty->getNumElements()) {
      // LLVM's shufflevector requires that the two vector operands have the
      // same type; SPIR-V's OpVectorShuffle allows the vector operands to
      // differ in the number of components.  Adjust for that by extending
      // the smaller vector.
      if (Vec1Ty->getNumElements() < Vec2Ty->getNumElements()) {
        Vec1 = extendVector(Vec1, Vec2Ty, Builder);
        // Extending Vec1 requires offsetting any Vec2 indices in Components by
        // the number of new elements.
        unsigned Offset = Vec2Ty->getNumElements() - Vec1Ty->getNumElements();
        unsigned Vec2Start = Vec1Ty->getNumElements();
        for (auto &C : Components) {
          if (auto *CI = dyn_cast<ConstantInt>(C)) {
            uint64_t V = CI->getZExtValue();
            if (V >= Vec2Start) {
              // This is a Vec2 index; add the offset to it.
              C = ConstantInt::get(Int32Ty, V + Offset);
            }
          }
        }
      } else {
        Vec2 = extendVector(Vec2, Vec1Ty, Builder);
      }
    }
    return mapValue(
        BV, Builder.CreateShuffleVector(
                Vec1, Vec2, ConstantVector::get(Components), BV->getName()));
  }

  case OpBitReverse: {
    auto *BR = static_cast<SPIRVUnary *>(BV);
    auto *Ty = transType(BV->getType());
    Function *intr =
        Intrinsic::getOrInsertDeclaration(M, llvm::Intrinsic::bitreverse, Ty);
    auto *Call = CallInst::Create(intr, transValue(BR->getOperand(0), F, BB),
                                  BR->getName(), BB);
    return mapValue(BV, Call);
  }

  case OpFunctionCall: {
    SPIRVFunctionCall *BC = static_cast<SPIRVFunctionCall *>(BV);
    std::vector<Value *> Args = transValue(BC->getArgumentValues(), F, BB);
    auto *Call = CallInst::Create(transFunction(BC->getFunction()), Args,
                                  BC->getName(), BB);
    setCallingConv(Call);
    setAttrByCalledFunc(Call);
    return mapValue(BV, Call);
  }

  case OpAsmCallINTEL:
    return mapValue(
        BV, transAsmCallINTEL(static_cast<SPIRVAsmCallINTEL *>(BV), F, BB));

  case OpFunctionPointerCallINTEL: {
    SPIRVFunctionPointerCallINTEL *BC =
        static_cast<SPIRVFunctionPointerCallINTEL *>(BV);
    auto *V = transValue(BC->getCalledValue(), F, BB);
    auto *SpirvFnTy = BC->getCalledValue()->getType()->getPointerElementType();
    auto *FnTy = cast<FunctionType>(transType(SpirvFnTy));
    auto *Call = CallInst::Create(
        FnTy, V, transValue(BC->getArgumentValues(), F, BB), BC->getName(), BB);
    transFunctionPointerCallArgumentAttributes(
        BV, Call, static_cast<SPIRVTypeFunction *>(SpirvFnTy));
    // Assuming we are calling a regular device function
    Call->setCallingConv(CallingConv::SPIR_FUNC);
    // Don't set attributes, because at translation time we don't know which
    // function exactly we are calling.
    return mapValue(BV, Call);
  }

  case OpAssumeTrueKHR: {
    IRBuilder<> Builder(BB);
    SPIRVAssumeTrueKHR *BC = static_cast<SPIRVAssumeTrueKHR *>(BV);
    Value *Condition = transValue(BC->getCondition(), F, BB);
    return mapValue(BV, Builder.CreateAssumption(Condition));
  }

  case OpExpectKHR: {
    IRBuilder<> Builder(BB);
    SPIRVExpectKHRInstBase *BC = static_cast<SPIRVExpectKHRInstBase *>(BV);
    Type *RetTy = transType(BC->getType());
    Value *Val = transValue(BC->getOperand(0), F, BB);
    Value *ExpVal = transValue(BC->getOperand(1), F, BB);
    return mapValue(
        BV, Builder.CreateIntrinsic(Intrinsic::expect, RetTy, {Val, ExpVal}));
  }

  case OpUntypedPrefetchKHR: {
    // Do the same as transOCLBuiltinFromExtInst() but for OpUntypedPrefetchKHR.
    auto *BC = static_cast<SPIRVUntypedPrefetchKHR *>(BV);

    std::vector<Type *> ArgTypes =
        transTypeVector(BC->getValueTypes(BC->getArguments()), true);
    Type *RetTy = Type::getVoidTy(*Context);

    std::string MangledName =
        getSPIRVFriendlyIRFunctionName(OpenCLLIB::Prefetch, ArgTypes, RetTy);
    opaquifyTypedPointers(ArgTypes);

    FunctionType *FT = FunctionType::get(RetTy, ArgTypes, false);
    Function *F = M->getFunction(MangledName);
    if (!F) {
      F = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
      F->setCallingConv(CallingConv::SPIR_FUNC);
      if (isFuncNoUnwind())
        F->addFnAttr(Attribute::NoUnwind);
      if (isFuncReadNone(OCLExtOpMap::map(OpenCLLIB::Prefetch)))
        F->setDoesNotAccessMemory();
    }

    auto Args = transValue(BC->getValues(BC->getArguments()), F, BB);
    CallInst *CI = CallInst::Create(F, Args, BC->getName(), BB);
    setCallingConv(CI);
    addFnAttr(CI, Attribute::NoUnwind);

    return mapValue(BV, CI);
  }
  case OpExtInst: {
    auto *ExtInst = static_cast<SPIRVExtInst *>(BV);
    switch (ExtInst->getExtSetKind()) {
    case SPIRVEIS_OpenCL: {
      auto *V = mapValue(BV, transOCLBuiltinFromExtInst(ExtInst, BB));
      applyFPFastMathModeDecorations(BV, static_cast<Instruction *>(V));
      return V;
    }
    case SPIRVEIS_Debug:
    case SPIRVEIS_OpenCL_DebugInfo_100:
    case SPIRVEIS_NonSemantic_Shader_DebugInfo_100:
    case SPIRVEIS_NonSemantic_Shader_DebugInfo_200:
      DbgTran->transDebugIntrinsic(ExtInst, BB);
      return mapValue(BV, nullptr);
    default:
      llvm_unreachable("Unknown extended instruction set!");
    }
  }

  case OpSNegate: {
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    if (BV->getType()->isTypeCooperativeMatrixKHR()) {
      return mapValue(BV, transSPIRVBuiltinFromInst(BC, BB));
    }
    auto *Neg =
        Builder.CreateNeg(transValue(BC->getOperand(0), F, BB), BV->getName());
    if (auto *NegInst = dyn_cast<Instruction>(Neg)) {
      applyNoIntegerWrapDecorations(BV, NegInst);
    }
    return mapValue(BV, Neg);
  }

  case OpFMod: {
    // translate OpFMod(a, b) to:
    //   r = frem(a, b)
    //   c = copysign(r, b)
    //   needs_fixing = islessgreater(r, c)
    //   result = needs_fixing ? r + b : c
    IRBuilder<> Builder(BB);
    SPIRVFMod *FMod = static_cast<SPIRVFMod *>(BV);
    auto *Dividend = transValue(FMod->getOperand(0), F, BB);
    auto *Divisor = transValue(FMod->getOperand(1), F, BB);
    auto *FRem = Builder.CreateFRem(Dividend, Divisor, "frem.res");
    auto *CopySign = Builder.CreateBinaryIntrinsic(
        llvm::Intrinsic::copysign, FRem, Divisor, nullptr, "copysign.res");
    auto *FAdd = Builder.CreateFAdd(FRem, Divisor, "fadd.res");
    auto *Cmp = Builder.CreateFCmpONE(FRem, CopySign, "cmp.res");
    auto *Select = Builder.CreateSelect(Cmp, FAdd, CopySign);
    return mapValue(BV, Select);
  }

  case OpSMod: {
    // translate OpSMod(a, b) to:
    //   r = srem(a, b)
    //   needs_fixing = ((a < 0) != (b < 0) && r != 0)
    //   result = needs_fixing ? r + b : r
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    SPIRVSMod *SMod = static_cast<SPIRVSMod *>(BV);
    auto *Dividend = transValue(SMod->getOperand(0), F, BB);
    auto *Divisor = transValue(SMod->getOperand(1), F, BB);
    auto *SRem = Builder.CreateSRem(Dividend, Divisor, "srem.res");
    auto *Xor = Builder.CreateXor(Dividend, Divisor, "xor.res");
    auto *Zero = ConstantInt::getNullValue(Dividend->getType());
    auto *CmpSign = Builder.CreateICmpSLT(Xor, Zero, "cmpsign.res");
    auto *CmpSRem = Builder.CreateICmpNE(SRem, Zero, "cmpsrem.res");
    auto *Add = Builder.CreateNSWAdd(SRem, Divisor, "add.res");
    auto *Cmp = Builder.CreateAnd(CmpSign, CmpSRem, "cmp.res");
    auto *Select = Builder.CreateSelect(Cmp, Add, SRem);
    return mapValue(BV, Select);
  }

  case OpFNegate: {
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    if (BV->getType()->isTypeCooperativeMatrixKHR()) {
      return mapValue(BV, transSPIRVBuiltinFromInst(BC, BB));
    }
    auto *Neg = UnaryOperator::CreateFNeg(transValue(BC->getOperand(0), F, BB),
                                          BV->getName(), BB);
    applyFPFastMathModeDecorations(BV, Neg);
    return mapValue(BV, Neg);
  }

  case OpNot:
  case OpLogicalNot: {
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    return mapValue(BV, Builder.CreateNot(transValue(BC->getOperand(0), F, BB),
                                          BV->getName()));
  }

  case OpAll:
  case OpAny:
    return mapValue(BV, transAllAny(static_cast<SPIRVInstruction *>(BV), BB));

  case OpIsFinite:
  case OpIsInf:
  case OpIsNan:
  case OpIsNormal:
  case OpSignBitSet:
    return mapValue(BV,
                    transRelational(static_cast<SPIRVInstruction *>(BV), BB));
  case OpIAddCarry:
  case OpISubBorrow: {
    IRBuilder Builder(BB);
    auto *BC = static_cast<SPIRVBinary *>(BV);
    Intrinsic::ID ID = OC == OpIAddCarry ? Intrinsic::uadd_with_overflow
                                         : Intrinsic::usub_with_overflow;
    auto *Inst =
        Builder.CreateBinaryIntrinsic(ID, transValue(BC->getOperand(0), F, BB),
                                      transValue(BC->getOperand(1), F, BB));

    // Extract components of the result.
    auto *Result = Builder.CreateExtractValue(Inst, 0); // iN result
    auto *Carry = Builder.CreateExtractValue(Inst, 1);  // i1 overflow

    // Convert {iN, i1} into {iN, iN} for SPIR-V compatibility.
    Value *CarryInt;
    if (Carry->getType()->isVectorTy()) {
      CarryInt = Builder.CreateZExt(
          Carry, VectorType::get(
                     cast<VectorType>(Result->getType())->getElementType(),
                     cast<VectorType>(Carry->getType())->getElementCount()));
    } else {
      CarryInt = Builder.CreateZExt(Carry, Result->getType());
    }
    auto *ResultStruct =
        Builder.CreateInsertValue(PoisonValue::get(StructType::get(
                                      Result->getType(), CarryInt->getType())),
                                  Result, 0);
    ResultStruct = Builder.CreateInsertValue(ResultStruct, CarryInt, 1);

    return mapValue(BV, ResultStruct);
  }
  case OpSMulExtended: {
    auto *BC = static_cast<SPIRVBinary *>(BV);
    return mapValue(BV, transBuiltinFromInst("__spirv_SMulExtended", BC, BB));
  }
  case OpUMulExtended: {
    auto *BC = static_cast<SPIRVBinary *>(BV);
    return mapValue(BV, transBuiltinFromInst("__spirv_UMulExtended", BC, BB));
  }
  case OpGetKernelWorkGroupSize:
  case OpGetKernelPreferredWorkGroupSizeMultiple:
    return mapValue(
        BV, transWGSizeQueryBI(static_cast<SPIRVInstruction *>(BV), BB));
  case OpGetKernelNDrangeMaxSubGroupSize:
  case OpGetKernelNDrangeSubGroupCount:
    return mapValue(
        BV, transSGSizeQueryBI(static_cast<SPIRVInstruction *>(BV), BB));
  case OpFPGARegINTEL: {
    IRBuilder<> Builder(BB);

    SPIRVFPGARegINTELInstBase *BC =
        static_cast<SPIRVFPGARegINTELInstBase *>(BV);

    PointerType *Int8PtrTyPrivate = PointerType::get(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = Type::getInt32Ty(*Context);

    Value *UndefInt8Ptr = PoisonValue::get(Int8PtrTyPrivate);
    Value *UndefInt32 = PoisonValue::get(Int32Ty);

    Constant *GS = Builder.CreateGlobalString(kOCLBuiltinName::FPGARegIntel);

    Type *Ty = transType(BC->getType());
    Value *Val = transValue(BC->getOperand(0), F, BB);

    Value *ValAsArg = Val;
    Type *RetTy = Ty;
    auto IID = Intrinsic::annotation;
    if (!isa<IntegerType>(Ty)) {
      // All scalar types can be bitcasted to a same-sized integer
      if (!isa<PointerType>(Ty) && !isa<StructType>(Ty)) {
        RetTy = IntegerType::get(*Context, Ty->getPrimitiveSizeInBits());
        ValAsArg = Builder.CreateBitCast(Val, RetTy);
      }
      // If pointer type or struct type
      else {
        IID = Intrinsic::ptr_annotation;
        auto *PtrTy = dyn_cast<PointerType>(Ty);
        if (PtrTy) {
          RetTy = PtrTy;
        } else {
          // If a struct - bitcast to i8*
          RetTy = Int8PtrTyPrivate;
          ValAsArg = Builder.CreateBitCast(Val, RetTy);
        }
        Value *Args[] = {ValAsArg, GS, UndefInt8Ptr, UndefInt32, UndefInt8Ptr};
        auto *IntrinsicCall =
            Builder.CreateIntrinsic(IID, {RetTy, GS->getType()}, Args);
        return mapValue(BV, IntrinsicCall);
      }
    }

    Value *Args[] = {ValAsArg, GS, UndefInt8Ptr, UndefInt32};
    auto *IntrinsicCall =
        Builder.CreateIntrinsic(IID, {RetTy, GS->getType()}, Args);
    return mapValue(BV, IntrinsicCall);
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
  case OpFixedExpINTEL:
    return mapValue(
        BV, transFixedPointInst(static_cast<SPIRVInstruction *>(BV), BB));

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
  case OpArbitraryFloatATanPiINTEL:
    return mapValue(BV,
                    transArbFloatInst(static_cast<SPIRVInstruction *>(BV), BB));

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
  case OpArbitraryFloatPowNINTEL:
    return mapValue(
        BV, transArbFloatInst(static_cast<SPIRVInstruction *>(BV), BB, true));

  case OpArithmeticFenceEXT: {
    IRBuilder<> Builder(BB);
    auto *BC = static_cast<SPIRVUnary *>(BV);
    Type *RetTy = transType(BC->getType());
    Value *Val = transValue(BC->getOperand(0), F, BB);
    return mapValue(
        BV, Builder.CreateIntrinsic(Intrinsic::arithmetic_fence, RetTy, Val));
  }
  case internal::OpMaskedGatherINTEL: {
    IRBuilder<> Builder(BB);
    auto *Inst = static_cast<SPIRVMaskedGatherINTELInst *>(BV);
    Type *RetTy = transType(Inst->getType());
    Value *PtrVector = transValue(Inst->getOperand(0), F, BB);
    uint32_t Alignment = Inst->getOpWord(1);
    Value *Mask = transValue(Inst->getOperand(2), F, BB);
    Value *FillEmpty = transValue(Inst->getOperand(3), F, BB);
    return mapValue(BV, Builder.CreateMaskedGather(RetTy, PtrVector,
                                                   Align(Alignment), Mask,
                                                   FillEmpty));
  }

  case internal::OpMaskedScatterINTEL: {
    IRBuilder<> Builder(BB);
    auto *Inst = static_cast<SPIRVMaskedScatterINTELInst *>(BV);
    Value *InputVector = transValue(Inst->getOperand(0), F, BB);
    Value *PtrVector = transValue(Inst->getOperand(1), F, BB);
    uint32_t Alignment = Inst->getOpWord(2);
    Value *Mask = transValue(Inst->getOperand(3), F, BB);
    return mapValue(BV, Builder.CreateMaskedScatter(InputVector, PtrVector,
                                                    Align(Alignment), Mask));
  }

  default: {
    auto OC = BV->getOpCode();
    if (isCmpOpCode(OC))
      return mapValue(BV, transCmpInst(BV, BB, F));

    if (OCLSPIRVBuiltinMap::rfind(OC, nullptr))
      return mapValue(BV, transSPIRVBuiltinFromInst(
                              static_cast<SPIRVInstruction *>(BV), BB));

    if (isBinaryShiftLogicalBitwiseOpCode(OC) || isLogicalOpCode(OC))
      return mapValue(BV, transShiftLogicalBitwiseInst(BV, BB, F));

    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit) {
      auto *BI = static_cast<SPIRVInstruction *>(BV);
      Value *Inst = nullptr;
      if (BI->hasFPRoundingMode() || BI->isSaturatedConversion() ||
          BI->getType()->isTypeCooperativeMatrixKHR())
        Inst = transSPIRVBuiltinFromInst(BI, BB);
      else
        Inst = transConvertInst(BV, F, BB);
      return mapValue(BV, Inst);
    }
    return mapValue(
        BV, transSPIRVBuiltinFromInst(static_cast<SPIRVInstruction *>(BV), BB));
  }
  }
}

// Get meaningful suffix for adding at the end of the function name to avoid
// ascending numerical suffixes. It is useful in situations, where the same
// function is called twice or more in one basic block. So, the function name is
// formed in the following way: [FuncName].[ReturnTy].[InputTy]
static std::string getFuncAPIntSuffix(const Type *RetTy, const Type *In1Ty,
                                      const Type *In2Ty = nullptr) {
  std::stringstream Suffix;
  Suffix << ".i" << RetTy->getIntegerBitWidth() << ".i"
         << In1Ty->getIntegerBitWidth();
  if (In2Ty)
    Suffix << ".i" << In2Ty->getIntegerBitWidth();
  return Suffix.str();
}

Value *SPIRVToLLVM::transFixedPointInst(SPIRVInstruction *BI, BasicBlock *BB) {
  // LLVM fixed point functions return value:
  // iN (arbitrary precision integer of N bits length)
  // Arguments:
  // A(iN), S(i1), I(i32), rI(i32), Quantization(i32), Overflow(i32)
  // If return value wider than 64 bit:
  // iN addrspace(4)* sret(iN), A(iN), S(i1), I(i32), rI(i32),
  // Quantization(i32), Overflow(i32)

  // SPIR-V fixed point instruction contains:
  // <id>ResTy Res<id> In<id> Literal S Literal I Literal rI Literal Q Literal O

  Type *RetTy = transType(BI->getType());

  auto *Inst = static_cast<SPIRVFixedPointIntelInst *>(BI);
  Type *InTy = transType(Inst->getOperand(0)->getType());

  IntegerType *Int32Ty = IntegerType::get(*Context, 32);
  IntegerType *Int1Ty = IntegerType::get(*Context, 1);

  SmallVector<Type *, 8> ArgTys;
  std::vector<Value *> Args;
  Args.reserve(8);
  if (RetTy->getIntegerBitWidth() > 64) {
    llvm::PointerType *RetPtrTy =
        llvm::PointerType::get(*Context, SPIRAS_Generic);
    Value *Alloca =
        new AllocaInst(RetTy, M->getDataLayout().getAllocaAddrSpace(), "", BB);
    Value *RetValPtr = new AddrSpaceCastInst(Alloca, RetPtrTy, "", BB);
    ArgTys.emplace_back(RetPtrTy);
    Args.emplace_back(RetValPtr);
  }

  ArgTys.insert(ArgTys.end(),
                {InTy, Int1Ty, Int32Ty, Int32Ty, Int32Ty, Int32Ty});

  auto Words = Inst->getOpWords();
  Args.emplace_back(transValue(Inst->getOperand(0), BB->getParent(), BB));
  Args.emplace_back(ConstantInt::get(Int1Ty, Words[1]));
  for (int I = 2; I <= 5; I++)
    Args.emplace_back(ConstantInt::get(Int32Ty, Words[I]));

  Type *FuncRetTy =
      (RetTy->getIntegerBitWidth() <= 64) ? RetTy : Type::getVoidTy(*Context);
  FunctionType *FT = FunctionType::get(FuncRetTy, ArgTys, false);

  Op OpCode = Inst->getOpCode();
  std::string FuncName =
      SPIRVFixedPointIntelMap::rmap(OpCode) + getFuncAPIntSuffix(RetTy, InTy);

  FunctionCallee FCallee = M->getOrInsertFunction(FuncName, FT);

  auto *Func = cast<Function>(FCallee.getCallee());
  Func->setCallingConv(CallingConv::SPIR_FUNC);
  if (isFuncNoUnwind())
    Func->addFnAttr(Attribute::NoUnwind);

  if (RetTy->getIntegerBitWidth() <= 64)
    return CallInst::Create(FCallee, Args, "", BB);

  Func->addParamAttr(
      0, Attribute::get(*Context, Attribute::AttrKind::StructRet, RetTy));
  CallInst *APIntInst = CallInst::Create(FCallee, Args, "", BB);
  APIntInst->addParamAttr(
      0, Attribute::get(*Context, Attribute::AttrKind::StructRet, RetTy));

  return static_cast<Value *>(new LoadInst(RetTy, Args[0], "", false, BB));
}

Value *SPIRVToLLVM::transArbFloatInst(SPIRVInstruction *BI, BasicBlock *BB,
                                      bool IsBinaryInst) {
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

  // Format of instructions GT, GE, LT, LE, EQ:
  //   LLVM arbitrary floating point functions return value: Bool
  //   Arguments: A(iN), MA(i32), B(iN), MB(i32)
  //   where A and B are of arbitrary precision integer type.
  //   SPIR-V arbitrary floating point instruction layout:
  //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal MB

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

  // Format of instruction CastFromInt:
  //   LLVM arbitrary floating point functions return value: iN
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
  //   <id>ResTy Res<id> A<id> Literal MA Literal Mout Literal EnableSubnormals
  //       Literal RoundingMode Literal RoundingAccuracy

  Type *RetTy = transType(BI->getType());
  IntegerType *Int1Ty = Type::getInt1Ty(*Context);
  IntegerType *Int32Ty = Type::getInt32Ty(*Context);

  auto *Inst = static_cast<SPIRVArbFloatIntelInst *>(BI);

  Type *ATy = transType(Inst->getOperand(0)->getType());
  Type *BTy = nullptr;

  // Words contain:
  // A<id> [Literal MA] [B<id>] [Literal MB] [Literal Mout] [Literal Sign]
  //   [Literal EnableSubnormals Literal RoundingMode Literal RoundingAccuracy]
  const std::vector<SPIRVWord> Words = Inst->getOpWords();
  auto WordsItr = Words.begin() + 1; /* Skip word for A input id */

  SmallVector<Type *, 8> ArgTys;
  std::vector<Value *> Args;

  if (RetTy->getIntegerBitWidth() > 64) {
    llvm::PointerType *RetPtrTy =
        llvm::PointerType::get(*Context, SPIRAS_Generic);
    ArgTys.push_back(RetPtrTy);
    Value *Alloca =
        new AllocaInst(RetTy, M->getDataLayout().getAllocaAddrSpace(), "", BB);
    Value *RetValPtr = new AddrSpaceCastInst(Alloca, RetPtrTy, "", BB);
    Args.push_back(RetValPtr);
  }

  ArgTys.insert(ArgTys.end(), {ATy, Int32Ty});
  // A - input
  Args.emplace_back(transValue(Inst->getOperand(0), BB->getParent(), BB));
  // MA/Mout - width of mantissa
  Args.emplace_back(ConstantInt::get(Int32Ty, *WordsItr++));

  Op OC = Inst->getOpCode();
  if (OC == OpArbitraryFloatCastFromIntINTEL ||
      OC == OpArbitraryFloatCastToIntINTEL) {
    ArgTys.push_back(Int1Ty);
    Args.push_back(ConstantInt::get(Int1Ty, *WordsItr++)); /* ToSign/FromSign */
  }

  if (IsBinaryInst) {
    /* B - input */
    BTy = transType(Inst->getOperand(2)->getType());
    ArgTys.push_back(BTy);
    Args.push_back(transValue(Inst->getOperand(2), BB->getParent(), BB));
    ++WordsItr; /* Skip word for B input id */
    if (OC == OpArbitraryFloatPowNINTEL) {
      ArgTys.push_back(Int1Ty);
      Args.push_back(ConstantInt::get(Int1Ty, *WordsItr++)); /* SignOfB */
    }
  }

  std::fill_n(std::back_inserter(ArgTys), Words.end() - WordsItr, Int32Ty);
  std::transform(WordsItr, Words.end(), std::back_inserter(Args),
                 [Int32Ty](const SPIRVWord &Word) {
                   return ConstantInt::get(Int32Ty, Word);
                 });

  std::string FuncName =
      SPIRVArbFloatIntelMap::rmap(OC) + getFuncAPIntSuffix(RetTy, ATy, BTy);

  Type *FuncRetTy =
      (RetTy->getIntegerBitWidth() <= 64) ? RetTy : Type::getVoidTy(*Context);
  FunctionType *FT = FunctionType::get(FuncRetTy, ArgTys, false);
  FunctionCallee FCallee = M->getOrInsertFunction(FuncName, FT);

  auto *Func = cast<Function>(FCallee.getCallee());
  Func->setCallingConv(CallingConv::SPIR_FUNC);
  if (isFuncNoUnwind())
    Func->addFnAttr(Attribute::NoUnwind);

  if (RetTy->getIntegerBitWidth() <= 64)
    return CallInst::Create(Func, Args, "", BB);

  Func->addParamAttr(
      0, Attribute::get(*Context, Attribute::AttrKind::StructRet, RetTy));
  CallInst *APFloatInst = CallInst::Create(FCallee, Args, "", BB);
  APFloatInst->addParamAttr(
      0, Attribute::get(*Context, Attribute::AttrKind::StructRet, RetTy));

  return static_cast<Value *>(new LoadInst(RetTy, Args[0], "", false, BB));
}

template <class SourceTy, class FuncTy>
bool SPIRVToLLVM::foreachFuncCtlMask(SourceTy Source, FuncTy Func) {
  SPIRVWord FCM = Source->getFuncCtlMask();
  SPIRSPIRVFuncCtlMaskMap::foreach (
      [&](Attribute::AttrKind Attr, SPIRVFunctionControlMaskKind Mask) {
        if (FCM & Mask)
          Func(Attr);
      });
  return true;
}

void SPIRVToLLVM::transFunctionAttrs(SPIRVFunction *BF, Function *F) {
  if (BF->hasDecorate(DecorationReferencedIndirectlyINTEL))
    F->addFnAttr("referenced-indirectly");
  if (isFuncNoUnwind())
    F->addFnAttr(Attribute::NoUnwind);
  foreachFuncCtlMask(BF, [&](Attribute::AttrKind Attr) { F->addFnAttr(Attr); });

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto *BA = BF->getArgument(I->getArgNo());
    mapValue(BA, &(*I));
    setName(&(*I), BA);
    AttributeMask IllegalAttrs =
        AttributeFuncs::typeIncompatible(I->getType(), I->getAttributes());
    BA->foreachAttr([&](SPIRVFuncParamAttrKind Kind) {
      // Skip this function parameter attribute as it will translated among
      // OpenCL metadata
      if (Kind == FunctionParameterAttributeRuntimeAlignedINTEL)
        return;
      if (Kind == FunctionParameterAttributeNoCapture) {
        I->addAttr(Attribute::getWithCaptureInfo(F->getContext(),
                                                 CaptureInfo::none()));
        return;
      }
      Attribute::AttrKind LLVMKind = SPIRSPIRVFuncParamAttrMap::rmap(Kind);
      if (IllegalAttrs.contains(LLVMKind))
        return;
      Type *AttrTy = nullptr;
      switch (LLVMKind) {
      case Attribute::AttrKind::ByVal:
      case Attribute::AttrKind::StructRet:
        AttrTy = transType(BA->getType()->getPointerElementType());
        break;
      default:
        break; // do nothing
      }
      // Make sure to use a correct constructor for a typed/typeless attribute
      auto A = AttrTy ? Attribute::get(*Context, LLVMKind, AttrTy)
                      : Attribute::get(*Context, LLVMKind);
      I->addAttr(A);
    });

    AttrBuilder Builder(*Context);
    SPIRVWord MaxOffset = 0;
    if (BA->hasDecorate(DecorationMaxByteOffset, 0, &MaxOffset))
      Builder.addDereferenceableAttr(MaxOffset);
    else {
      SPIRVId MaxOffsetId;
      if (BA->hasDecorateId(DecorationMaxByteOffsetId, 0, &MaxOffsetId)) {
        if (auto MaxOffsetVal = transIdAsConstant(MaxOffsetId)) {
          Builder.addDereferenceableAttr(*MaxOffsetVal);
        }
      }
    }
    if (auto Alignment = getAlignment(BA)) {
      Builder.addAlignmentAttr(*Alignment);
    }
    I->addAttrs(Builder);
  }
  BF->foreachReturnValueAttr([&](SPIRVFuncParamAttrKind Kind) {
    if (Kind == FunctionParameterAttributeNoWrite)
      return;
    F->addRetAttr(SPIRSPIRVFuncParamAttrMap::rmap(Kind));
  });
}

namespace {
// One basic block can be a predecessor to another basic block more than
// once (https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2702).
// This function fixes any PHIs that break this rule.
static void validatePhiPredecessors(Function *F) {
  for (BasicBlock &BB : *F) {
    bool UniquePreds = true;
    DenseMap<BasicBlock *, unsigned> PredsCnt;
    for (BasicBlock *PredBB : predecessors(&BB)) {
      auto It = PredsCnt.try_emplace(PredBB, 1);
      if (!It.second) {
        UniquePreds = false;
        ++It.first->second;
      }
    }
    if (UniquePreds)
      continue;
    // `phi` requires an incoming value per each predecessor instance, even
    // it's the same basic block that has been already inserted as an incoming
    // value of the `phi`.
    for (PHINode &Phi : BB.phis()) {
      SmallVector<Value *> Vs;
      SmallVector<BasicBlock *> Bs;
      SmallPtrSet<BasicBlock *, 8> UsedB;
      for (auto [V, B] : zip(Phi.incoming_values(), Phi.blocks())) {
        if (!UsedB.insert(B).second)
          continue;
        unsigned N = PredsCnt[B];
        Vs.insert(Vs.end(), N, V);
        Bs.insert(Bs.end(), N, B);
      }
      unsigned I = 0;
      for (unsigned N = Phi.getNumIncomingValues(); I < N; ++I) {
        Phi.setIncomingValue(I, Vs[I]);
        Phi.setIncomingBlock(I, Bs[I]);
      }
      for (unsigned N = Vs.size(); I < N; ++I)
        Phi.addIncoming(Vs[I], Bs[I]);
    }
  }
}
} // namespace

Function *SPIRVToLLVM::transFunction(SPIRVFunction *BF, unsigned AS) {
  auto Loc = FuncMap.find(BF);
  if (Loc != FuncMap.end())
    return Loc->second;

  auto IsKernel = isKernel(BF);

  if (IsKernel) {
    // search for a previous function with the same name
    // upgrade it to a kernel and drop this if it's found
    for (auto &I : FuncMap) {
      auto BFName = I.getFirst()->getName();
      if (BF->getName() == BFName) {
        auto *F = I.getSecond();
        F->setCallingConv(CallingConv::SPIR_KERNEL);
        F->setLinkage(GlobalValue::ExternalLinkage);
        F->setDSOLocal(false);
        F = cast<Function>(mapValue(BF, F));
        mapFunction(BF, F);
        transFunctionAttrs(BF, F);
        return F;
      }
    }
  }

  auto Linkage = IsKernel ? GlobalValue::ExternalLinkage : transLinkageType(BF);
  FunctionType *FT = cast<FunctionType>(transType(BF->getFunctionType()));
  std::string FuncName = BF->getName();
  StringRef FuncNameRef(FuncName);
  // Transform "@spirv.llvm_memset_p0i8_i32.volatile" to @llvm.memset.p0i8.i32
  // assuming llvm.memset is supported by the device compiler. If this
  // assumption is not safe, we should have a command line option to control
  // this behavior.
  if (FuncNameRef.starts_with("spirv.llvm_memset_p")) {
    // We can't guarantee that the name is correctly mangled due to opaque
    // pointers. Derive the correct name from the function type.
    FuncName =
        Intrinsic::getOrInsertDeclaration(
            M, Intrinsic::memset, {FT->getParamType(0), FT->getParamType(2)})
            ->getName();
  }
  if (FuncNameRef.consume_front("spirv.")) {
    FuncNameRef.consume_back(".volatile");
    FuncName = FuncNameRef.str();
    std::replace(FuncName.begin(), FuncName.end(), '_', '.');
  }
  Function *F = M->getFunction(FuncName);
  if (!F)
    F = Function::Create(FT, Linkage, AS, FuncName, M);
  F = cast<Function>(mapValue(BF, F));
  mapFunction(BF, F);

  if (F->isIntrinsic()) {
    if (F->getIntrinsicID() != Intrinsic::umul_with_overflow)
      return F;
    std::string Name = F->getName().str();
    auto *ST = cast<StructType>(F->getReturnType());
    auto *FT = F->getFunctionType();
    auto *NewST = StructType::get(ST->getContext(), ST->elements());
    auto *NewFT = FunctionType::get(NewST, FT->params(), FT->isVarArg());
    F->setName("old_" + Name);
    auto *NewFn = Function::Create(NewFT, F->getLinkage(), F->getAddressSpace(),
                                   Name, F->getParent());
    return NewFn;
  }

  F->setCallingConv(IsKernel ? CallingConv::SPIR_KERNEL
                             : CallingConv::SPIR_FUNC);
  transFunctionAttrs(BF, F);

  // Creating all basic blocks before creating instructions.
  for (size_t I = 0, E = BF->getNumBasicBlock(); I != E; ++I) {
    transValue(BF->getBasicBlock(I), F, nullptr);
  }

  for (size_t I = 0, E = BF->getNumBasicBlock(); I != E; ++I) {
    SPIRVBasicBlock *BBB = BF->getBasicBlock(I);
    BasicBlock *BB = cast<BasicBlock>(transValue(BBB, F, nullptr));
    for (size_t BI = 0, BE = BBB->getNumInst(); BI != BE; ++BI) {
      SPIRVInstruction *BInst = BBB->getInst(BI);
      transValue(BInst, F, BB, false);
    }
  }

  validatePhiPredecessors(F);
  transLLVMLoopMetadata(F);

  return F;
}

Value *SPIRVToLLVM::transAsmINTEL(SPIRVAsmINTEL *BA) {
  assert(BA);
  bool HasSideEffect = BA->hasDecorate(DecorationSideEffectsINTEL);
  return InlineAsm::get(
      cast<FunctionType>(transType(BA->getFunctionType())),
      BA->getInstructions(), BA->getConstraints(), HasSideEffect,
      /* IsAlignStack */ false, InlineAsm::AsmDialect::AD_ATT);
}

CallInst *SPIRVToLLVM::transAsmCallINTEL(SPIRVAsmCallINTEL *BI, Function *F,
                                         BasicBlock *BB) {
  assert(BI);
  auto *IA = cast<InlineAsm>(transValue(BI->getAsm(), F, BB));
  auto Args = transValue(BM->getValues(BI->getArguments()), F, BB);
  return CallInst::Create(cast<FunctionType>(IA->getFunctionType()), IA, Args,
                          BI->getName(), BB);
}

/// LLVM convert builtin functions is translated to two instructions:
/// y = i32 islessgreater(float x, float z) ->
///     y = i32 ZExt(bool LessOrGreater(float x, float z))
/// When translating back, for simplicity, a trunc instruction is inserted
/// w = bool LessOrGreater(float x, float z) ->
///     w = bool Trunc(i32 islessgreater(float x, float z))
/// Optimizer should be able to remove the redundant trunc/zext
void SPIRVToLLVM::transOCLBuiltinFromInstPreproc(
    SPIRVInstruction *BI, Type *&RetTy, std::vector<SPIRVValue *> &Args) {
  if (!BI->hasType())
    return;
  auto *BT = BI->getType();
  if (isCmpOpCode(BI->getOpCode())) {
    if (BT->isTypeBool())
      RetTy = IntegerType::getInt32Ty(*Context);
    else if (BT->isTypeVectorBool())
      RetTy = FixedVectorType::get(
          IntegerType::get(
              *Context,
              Args[0]->getType()->getVectorComponentType()->getBitWidth()),
          BT->getVectorComponentCount());
    else
      llvm_unreachable("invalid compare instruction");
  }
}

Instruction *
SPIRVToLLVM::transOCLBuiltinPostproc(SPIRVInstruction *BI, CallInst *CI,
                                     BasicBlock *BB,
                                     const std::string &DemangledName) {
  auto OC = BI->getOpCode();
  if (isCmpOpCode(OC) && BI->getType()->isTypeVectorOrScalarBool()) {
    return CastInst::Create(Instruction::Trunc, CI, transType(BI->getType()),
                            "cvt", BB);
  }
  if (SPIRVEnableStepExpansion &&
      (DemangledName == "smoothstep" || DemangledName == "step"))
    return expandOCLBuiltinWithScalarArg(CI, DemangledName);
  return CI;
}

Value *SPIRVToLLVM::transBlockInvoke(SPIRVValue *Invoke, BasicBlock *BB) {
  auto *TranslatedInvoke = transFunction(static_cast<SPIRVFunction *>(Invoke));
  auto *Int8PtrTyGen = PointerType::get(*Context, SPIRAS_Generic);
  return CastInst::CreatePointerBitCastOrAddrSpaceCast(TranslatedInvoke,
                                                       Int8PtrTyGen, "", BB);
}

Instruction *SPIRVToLLVM::transWGSizeQueryBI(SPIRVInstruction *BI,
                                             BasicBlock *BB) {
  std::string FName =
      (BI->getOpCode() == OpGetKernelWorkGroupSize)
          ? "__get_kernel_work_group_size_impl"
          : "__get_kernel_preferred_work_group_size_multiple_impl";

  Function *F = M->getFunction(FName);
  if (!F) {
    auto *Int8PtrTyGen = PointerType::get(*Context, SPIRAS_Generic);
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(*Context),
                                         {Int8PtrTyGen, Int8PtrTyGen}, false);
    F = Function::Create(FT, GlobalValue::ExternalLinkage, FName, M);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
  }
  auto Ops = BI->getOperands();
  SmallVector<Value *, 2> Args = {transBlockInvoke(Ops[0], BB),
                                  transValue(Ops[1], F, BB, false)};
  auto *Call = CallInst::Create(F, Args, "", BB);
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  return Call;
}

Instruction *SPIRVToLLVM::transSGSizeQueryBI(SPIRVInstruction *BI,
                                             BasicBlock *BB) {
  std::string FName = (BI->getOpCode() == OpGetKernelNDrangeMaxSubGroupSize)
                          ? "__get_kernel_max_sub_group_size_for_ndrange_impl"
                          : "__get_kernel_sub_group_count_for_ndrange_impl";

  auto Ops = BI->getOperands();
  Function *F = M->getFunction(FName);
  if (!F) {
    auto *Int8PtrTyGen = PointerType::get(*Context, SPIRAS_Generic);
    SmallVector<Type *, 3> Tys = {
        transType(Ops[0]->getType()), // ndrange
        Int8PtrTyGen,                 // block_invoke
        Int8PtrTyGen                  // block_literal
    };
    auto *FT = FunctionType::get(Type::getInt32Ty(*Context), Tys, false);
    F = Function::Create(FT, GlobalValue::ExternalLinkage, FName, M);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
  }
  SmallVector<Value *, 2> Args = {
      transValue(Ops[0], F, BB, false), // ndrange
      transBlockInvoke(Ops[1], BB),     // block_invoke
      transValue(Ops[2], F, BB, false)  // block_literal
  };
  auto *Call = CallInst::Create(F, Args, "", BB);
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  return Call;
}

Instruction *SPIRVToLLVM::transBuiltinFromInst(const std::string &FuncName,
                                               SPIRVInstruction *BI,
                                               BasicBlock *BB) {
  std::string MangledName;
  auto Ops = BI->getOperands();
  Op OC = BI->getOpCode();
  if (isUntypedAccessChainOpCode(OC)) {
    auto *AC = static_cast<SPIRVAccessChainBase *>(BI);
    if (AC->getBaseType()->isTypeCooperativeMatrixKHR())
      Ops.erase(Ops.begin());
  }
  Type *RetTy =
      BI->hasType() ? transType(BI->getType()) : Type::getVoidTy(*Context);
  transOCLBuiltinFromInstPreproc(BI, RetTy, Ops);
  std::vector<Type *> ArgTys =
      transTypeVector(SPIRVInstruction::getOperandTypes(Ops), true);

  auto Ptr = findFirstPtrType(ArgTys);
  if (Ptr < ArgTys.size() &&
      BI->getValueType(Ops[Ptr]->getId())->isTypeUntypedPointerKHR()) {
    // Special handling for "truly" untyped pointers to preserve correct
    // builtin mangling of atomic and matrix operations.
    if (isAtomicOpCodeUntypedPtrSupported(OC)) {
      auto *AI = static_cast<SPIRVAtomicInstBase *>(BI);
      ArgTys[Ptr] = TypedPointerType::get(
          transType(AI->getSemanticType()),
          SPIRSPIRVAddrSpaceMap::rmap(
              BI->getValueType(Ops[Ptr]->getId())->getPointerStorageClass()));
    }
  }

  for (unsigned I = 0; I < ArgTys.size(); I++) {
    if (isa<PointerType>(ArgTys[I])) {
      SPIRVType *OpTy = BI->getValueType(Ops[I]->getId());
      // `Param` must be a pointer to an 8-bit integer type scalar.
      // Avoid demangling for this argument if it's a pointer to get `Pc`
      // mangling.
      if (OC == OpEnqueueKernel && I == 7) {
        if (ArgTys[I]->isPointerTy())
          continue;
      }
      if (OpTy->isTypeUntypedPointerKHR()) {
        auto *Val = transValue(Ops[I], BB->getParent(), BB);
        Val = Val->stripPointerCasts();
        if (isUntypedAccessChainOpCode(Ops[I]->getOpCode())) {
          SPIRVType *BaseTy =
              reinterpret_cast<SPIRVAccessChainBase *>(Ops[I])->getBaseType();

          Type *Ty = nullptr;
          if (BaseTy->isTypeArray())
            Ty = transType(BaseTy->getArrayElementType());
          else if (BaseTy->isTypeVector())
            Ty = transType(BaseTy->getVectorComponentType());
          else
            Ty = transType(BaseTy);
          ArgTys[I] = TypedPointerType::get(
              Ty, SPIRSPIRVAddrSpaceMap::rmap(OpTy->getPointerStorageClass()));
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Val)) {
          ArgTys[I] = TypedPointerType::get(
              GEP->getSourceElementType(),
              SPIRSPIRVAddrSpaceMap::rmap(OpTy->getPointerStorageClass()));
        } else if (Ops[I]->getOpCode() == OpUntypedVariableKHR) {
          SPIRVUntypedVariableKHR *UV =
              static_cast<SPIRVUntypedVariableKHR *>(Ops[I]);
          Type *Ty = transType(UV->getDataType());
          ArgTys[I] = TypedPointerType::get(
              Ty, SPIRSPIRVAddrSpaceMap::rmap(OpTy->getPointerStorageClass()));
        } else if (auto *AI = dyn_cast<AllocaInst>(Val)) {
          ArgTys[I] = TypedPointerType::get(
              AI->getAllocatedType(),
              SPIRSPIRVAddrSpaceMap::rmap(OpTy->getPointerStorageClass()));
        } else if (Ops[I]->getOpCode() == OpFunctionParameter &&
                   !RetTy->isVoidTy()) {
          // Pointer could be a function parameter. Assume that the type of
          // the pointer is the same as the return type.
          Type *Ty = nullptr;
          // it return type is array type, assign its element type to Ty
          if (RetTy->isArrayTy())
            Ty = RetTy->getArrayElementType();
          else if (RetTy->isVectorTy())
            Ty = cast<VectorType>(RetTy)->getElementType();
          else
            Ty = RetTy;

          ArgTys[I] = TypedPointerType::get(
              Ty, SPIRSPIRVAddrSpaceMap::rmap(OpTy->getPointerStorageClass()));
        }
      }
    }
  }

  for (auto &I : ArgTys) {
    if (isa<FunctionType>(I)) {
      I = TypedPointerType::get(I, SPIRAS_Private);
    }
  }

  if (BM->getDesiredBIsRepresentation() != BIsRepresentation::SPIRVFriendlyIR)
    mangleOpenClBuiltin(FuncName, ArgTys, MangledName);
  else
    MangledName = getSPIRVFriendlyIRFunctionName(FuncName, OC, ArgTys, Ops);

  opaquifyTypedPointers(ArgTys);

  Function *Func = M->getFunction(MangledName);
  FunctionType *FT = FunctionType::get(RetTy, ArgTys, false);
  // ToDo: Some intermediate functions have duplicate names with
  // different function types. This is OK if the function name
  // is used internally and finally translated to unique function
  // names. However it is better to have a way to differentiate
  // between intermidiate functions and final functions and make
  // sure final functions have unique names.
  SPIRVDBG(if (Func && Func->getFunctionType() != FT) {
    dbgs() << "Warning: Function name conflict:\n"
           << *Func << '\n'
           << " => " << *FT << '\n';
  })
  if (!Func || Func->getFunctionType() != FT) {
    LLVM_DEBUG(for (auto &I : ArgTys) { dbgs() << *I << '\n'; });
    Func = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    Func->setCallingConv(CallingConv::SPIR_FUNC);
    if (isFuncNoUnwind())
      Func->addFnAttr(Attribute::NoUnwind);
    if (isGroupOpCode(OC) || isGroupNonUniformOpcode(OC) ||
        isIntelSubgroupOpCode(OC) || isSplitBarrierINTELOpCode(OC) ||
        OC == OpControlBarrier)
      Func->addFnAttr(Attribute::Convergent);
  }
  CallInst *Call;
  // TODO: Remove the check for matrix type once drivers are updated.
  if (OC == OpCooperativeMatrixLengthKHR &&
      Ops[0]->getOpCode() == OpTypeCooperativeMatrixKHR) {
    // OpCooperativeMatrixLengthKHR needs special handling as its operand is
    // a Type instead of a Value.
    llvm::Type *MatTy = transType(reinterpret_cast<SPIRVType *>(Ops[0]));
    Call = CallInst::Create(Func, Constant::getNullValue(MatTy), "", BB);
  } else {
    Call = CallInst::Create(Func, transValue(Ops, BB->getParent(), BB), "", BB);
  }
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  SPIRVDBG(spvdbgs() << "[transInstToBuiltinCall] " << *BI << " -> ";
           dbgs() << *Call << '\n';)
  Instruction *Inst = transOCLBuiltinPostproc(BI, Call, BB, FuncName);
  return Inst;
}

SPIRVToLLVM::SPIRVToLLVM(Module *LLVMModule, SPIRVModule *TheSPIRVModule)
    : BuiltinCallHelper(ManglingRules::OpenCL), M(LLVMModule),
      BM(TheSPIRVModule) {
  assert(M && "Initialization without an LLVM module is not allowed");
  initialize(*M);
  Context = &M->getContext();
  if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR)
    UseTargetTypes = true;
  DbgTran.reset(new SPIRVToLLVMDbgTran(TheSPIRVModule, LLVMModule, this));
}

std::string getSPIRVFuncSuffix(SPIRVInstruction *BI) {
  std::string Suffix = "";
  if (BI->getOpCode() == OpCreatePipeFromPipeStorage) {
    auto *CPFPS = static_cast<SPIRVCreatePipeFromPipeStorage *>(BI);
    assert(CPFPS->getType()->isTypePipe() &&
           "Invalid type of CreatePipeFromStorage");
    auto *PipeType = static_cast<SPIRVTypePipe *>(CPFPS->getType());
    switch (PipeType->getAccessQualifier()) {
    default:
    case AccessQualifierReadOnly:
      Suffix = "_read";
      break;
    case AccessQualifierWriteOnly:
      Suffix = "_write";
      break;
    case AccessQualifierReadWrite:
      Suffix = "_read_write";
      break;
    }
  }
  if (BI->hasDecorate(DecorationSaturatedConversion)) {
    Suffix += kSPIRVPostfix::Divider;
    Suffix += kSPIRVPostfix::Sat;
  }
  SPIRVFPRoundingModeKind Kind;
  if (BI->hasFPRoundingMode(&Kind)) {
    Suffix += kSPIRVPostfix::Divider;
    Suffix += SPIRSPIRVFPRoundingModeMap::rmap(Kind);
  }
  if (BI->getOpCode() == OpGenericCastToPtrExplicit) {
    Suffix += kSPIRVPostfix::Divider;
    auto *Ty = BI->getType();
    auto GenericCastToPtrInst =
        Ty->isTypeVectorPointer()
            ? Ty->getVectorComponentType()->getPointerStorageClass()
            : Ty->getPointerStorageClass();
    switch (GenericCastToPtrInst) {
    case StorageClassCrossWorkgroup:
      Suffix += std::string(kSPIRVPostfix::ToGlobal);
      break;
    case StorageClassWorkgroup:
      Suffix += std::string(kSPIRVPostfix::ToLocal);
      break;
    case StorageClassFunction:
      Suffix += std::string(kSPIRVPostfix::ToPrivate);
      break;
    default:
      llvm_unreachable("Invalid address space");
    }
  }
  if (BI->getOpCode() == OpBuildNDRange) {
    Suffix += kSPIRVPostfix::Divider;
    auto *NDRangeInst = static_cast<SPIRVBuildNDRange *>(BI);
    auto *EleTy = ((NDRangeInst->getOperands())[0])->getType();
    int Dim = EleTy->isTypeArray() ? EleTy->getArrayLength() : 1;
    assert((EleTy->isTypeInt() && Dim == 1) ||
           (EleTy->isTypeArray() && Dim >= 2 && Dim <= 3));
    std::ostringstream OS;
    OS << Dim;
    Suffix += OS.str() + "D";
  }
  return Suffix;
}

Instruction *SPIRVToLLVM::transSPIRVBuiltinFromInst(SPIRVInstruction *BI,
                                                    BasicBlock *BB) {
  assert(BB && "Invalid BB");
  const auto OC = BI->getOpCode();

  bool AddRetTypePostfix = false;
  switch (static_cast<size_t>(OC)) {
  case OpImageQuerySizeLod:
  case OpImageQuerySize:
  case OpImageRead:
  case OpSubgroupImageBlockReadINTEL:
  case OpSubgroupImageMediaBlockReadINTEL:
  case OpSubgroupBlockReadINTEL:
  case OpImageSampleExplicitLod:
  case OpSDotKHR:
  case OpUDotKHR:
  case OpSUDotKHR:
  case OpSDotAccSatKHR:
  case OpUDotAccSatKHR:
  case OpSUDotAccSatKHR:
  case OpReadClockKHR:
  case internal::OpJointMatrixLoadINTEL:
  case OpCooperativeMatrixLoadKHR:
  case internal::OpCooperativeMatrixLoadCheckedINTEL:
  case internal::OpCooperativeMatrixLoadOffsetINTEL:
  case internal::OpTaskSequenceCreateINTEL:
  case internal::OpConvertHandleToImageINTEL:
  case internal::OpConvertHandleToSampledImageINTEL:
    AddRetTypePostfix = true;
    break;
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit)
      AddRetTypePostfix = true;
    break;
  }
  }

  bool IsRetSigned = true;
  switch (OC) {
  case OpConvertFToU:
  case OpSatConvertSToU:
  case OpUConvert:
  case OpUDotKHR:
  case OpUDotAccSatKHR:
  case OpReadClockKHR:
    IsRetSigned = false;
    break;
  case OpImageRead:
  case OpImageSampleExplicitLod: {
    size_t Idx = getImageOperandsIndex(OC);
    if (auto Ops = BI->getOperands(); Ops.size() > Idx) {
      auto ImOp = static_cast<SPIRVConstant *>(Ops[Idx])->getZExtIntValue();
      IsRetSigned = !(ImOp & ImageOperandsMask::ImageOperandsZeroExtendMask);
    }
    break;
  }
  default:
    break;
  }

  if (AddRetTypePostfix) {
    const Type *RetTy = BI->hasType() ? transType(BI->getType(), true)
                                      : Type::getVoidTy(*Context);
    Type *PET = nullptr;
    if (auto *TPT = dyn_cast<TypedPointerType>(RetTy))
      PET = TPT->getElementType();
    return transBuiltinFromInst(getSPIRVFuncName(OC, RetTy, IsRetSigned, PET) +
                                    getSPIRVFuncSuffix(BI),
                                BI, BB);
  }
  return transBuiltinFromInst(getSPIRVFuncName(OC, getSPIRVFuncSuffix(BI)), BI,
                              BB);
}

bool SPIRVToLLVM::translate() {
  if (!transAddressingModel())
    return false;

  // Entry Points should be translated before all debug intrinsics.
  for (SPIRVExtInst *EI : BM->getDebugInstVec()) {
    if (EI->getExtOp() == SPIRVDebug::EntryPoint)
      DbgTran->transDebugInst(EI);
  }

  // Compile unit might be needed during translation of debug intrinsics.
  for (SPIRVExtInst *EI : BM->getDebugInstVec()) {
    // Translate Compile Units first.
    if (EI->getExtOp() == SPIRVDebug::CompilationUnit)
      DbgTran->transDebugInst(EI);
  }

  for (unsigned I = 0, E = BM->getNumVariables(); I != E; ++I) {
    auto *BV = BM->getVariable(I);
    if (BV->getStorageClass() != StorageClassFunction)
      transValue(BV, nullptr, nullptr);
    transGlobalCtorDtors(BV);
  }

  // Then translate all debug instructions.
  for (SPIRVExtInst *EI : BM->getDebugInstVec()) {
    DbgTran->transDebugInst(EI);
  }

  for (auto *FP : BM->getFunctionPointers()) {
    SPIRVConstantFunctionPointerINTEL *BC =
        static_cast<SPIRVConstantFunctionPointerINTEL *>(FP);
    SPIRVFunction *F = BC->getFunction();
    FP->setName(F->getName());
    const unsigned AS = BM->shouldEmitFunctionPtrAddrSpace()
                            ? SPIRAS_CodeSectionINTEL
                            : SPIRAS_Private;
    mapValue(FP, transFunction(F, AS));
  }

  for (unsigned I = 0, E = BM->getNumFunctions(); I != E; ++I) {
    transFunction(BM->getFunction(I));
    transUserSemantic(BM->getFunction(I));
  }

  transGlobalAnnotations();

  if (!transMetadata())
    return false;
  if (!transFPContractMetadata())
    return false;
  transSourceLanguage();
  if (!transSourceExtension())
    return false;
  transGeneratorMD();
  if (!lowerBuiltins(BM, M))
    return false;
  if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR) {
    SPIRVWord SrcLangVer = 0;
    BM->getSourceLanguage(&SrcLangVer);
    bool IsCpp =
        SrcLangVer == kOCLVer::CLCXX10 || SrcLangVer == kOCLVer::CLCXX2021;
    if (!postProcessBuiltinsReturningStruct(M, IsCpp))
      return false;
  }

  for (SPIRVExtInst *EI : BM->getAuxDataInstVec()) {
    transAuxDataInst(EI);
  }

  eraseUselessFunctions(M);

  DbgTran->addDbgInfoVersion();
  DbgTran->finalize();

  return true;
}

bool SPIRVToLLVM::transAddressingModel() {
  switch (BM->getAddressingModel()) {
  case AddressingModelPhysical64:
    M->setTargetTriple(Triple(SPIR_TARGETTRIPLE64));
    M->setDataLayout(SPIR_DATALAYOUT64);
    break;
  case AddressingModelPhysical32:
    M->setTargetTriple(Triple(SPIR_TARGETTRIPLE32));
    M->setDataLayout(SPIR_DATALAYOUT32);
    break;
  case AddressingModelLogical:
    // Do not set target triple and data layout
    break;
  default:
    SPIRVCKRT(0, InvalidAddressingModel,
              "Actual addressing mode is " +
                  std::to_string(BM->getAddressingModel()));
  }
  return true;
}

void generateIntelFPGAAnnotation(
    const SPIRVEntry *E, std::vector<llvm::SmallString<256>> &AnnotStrVec) {
  llvm::SmallString<256> AnnotStr;
  llvm::raw_svector_ostream Out(AnnotStr);
  if (E->hasDecorate(DecorationRegisterINTEL))
    Out << "{register:1}";

  SPIRVWord Result = 0;
  if (E->hasDecorate(DecorationMemoryINTEL))
    Out << "{memory:"
        << E->getDecorationStringLiteral(DecorationMemoryINTEL).front() << '}';
  if (E->hasDecorate(DecorationBankwidthINTEL, 0, &Result))
    Out << "{bankwidth:" << Result << '}';
  if (E->hasDecorate(DecorationNumbanksINTEL, 0, &Result))
    Out << "{numbanks:" << Result << '}';
  if (E->hasDecorate(DecorationMaxPrivateCopiesINTEL, 0, &Result))
    Out << "{private_copies:" << Result << '}';
  if (E->hasDecorate(DecorationSinglepumpINTEL))
    Out << "{pump:1}";
  if (E->hasDecorate(DecorationDoublepumpINTEL))
    Out << "{pump:2}";
  if (E->hasDecorate(DecorationMaxReplicatesINTEL, 0, &Result))
    Out << "{max_replicates:" << Result << '}';
  if (E->hasDecorate(DecorationSimpleDualPortINTEL))
    Out << "{simple_dual_port:1}";
  if (E->hasDecorate(DecorationMergeINTEL)) {
    Out << "{merge";
    for (const auto &Str : E->getDecorationStringLiteral(DecorationMergeINTEL))
      Out << ":" << Str;
    Out << '}';
  }
  if (E->hasDecorate(DecorationBankBitsINTEL)) {
    Out << "{bank_bits:";
    auto Literals = E->getDecorationLiterals(DecorationBankBitsINTEL);
    for (size_t I = 0; I < Literals.size() - 1; ++I)
      Out << Literals[I] << ",";
    Out << Literals.back() << '}';
  }
  if (E->hasDecorate(DecorationForcePow2DepthINTEL, 0, &Result))
    Out << "{force_pow2_depth:" << Result << '}';
  if (E->hasDecorate(DecorationStridesizeINTEL, 0, &Result))
    Out << "{stride_size:" << Result << "}";
  if (E->hasDecorate(DecorationWordsizeINTEL, 0, &Result))
    Out << "{word_size:" << Result << "}";
  if (E->hasDecorate(DecorationTrueDualPortINTEL))
    Out << "{true_dual_port}";
  if (E->hasDecorate(DecorationBufferLocationINTEL, 0, &Result))
    Out << "{sycl-buffer-location:" << Result << '}';
  if (E->hasDecorate(DecorationLatencyControlLabelINTEL, 0, &Result))
    Out << "{sycl-latency-anchor-id:" << Result << '}';
  if (E->hasDecorate(DecorationLatencyControlConstraintINTEL)) {
    auto Literals =
        E->getDecorationLiterals(DecorationLatencyControlConstraintINTEL);
    assert(Literals.size() == 3 &&
           "Latency Control Constraint decoration shall have 3 extra operands");
    Out << "{sycl-latency-constraint:" << Literals[0] << "," << Literals[1]
        << "," << Literals[2] << '}';
  }

  unsigned LSUParamsBitmask = 0;
  llvm::SmallString<32> AdditionalParamsStr;
  llvm::raw_svector_ostream ParamsOut(AdditionalParamsStr);
  if (E->hasDecorate(DecorationBurstCoalesceINTEL, 0))
    LSUParamsBitmask |= IntelFPGAMemoryAccessesVal::BurstCoalesce;
  if (E->hasDecorate(DecorationCacheSizeINTEL, 0, &Result)) {
    LSUParamsBitmask |= IntelFPGAMemoryAccessesVal::CacheSizeFlag;
    ParamsOut << "{cache-size:" << Result << "}";
  }
  if (E->hasDecorate(DecorationDontStaticallyCoalesceINTEL, 0))
    LSUParamsBitmask |= IntelFPGAMemoryAccessesVal::DontStaticallyCoalesce;
  if (E->hasDecorate(DecorationPrefetchINTEL, 0, &Result)) {
    LSUParamsBitmask |= IntelFPGAMemoryAccessesVal::PrefetchFlag;
    // TODO: Enable prefetch size backwards translation
    // once it is supported
  }
  if (LSUParamsBitmask)
    Out << "{params:" << LSUParamsBitmask << "}" << AdditionalParamsStr;
  if (!AnnotStr.empty())
    AnnotStrVec.emplace_back(AnnotStr);

  if (E->hasDecorate(DecorationUserSemantic)) {
    auto Annotations =
        E->getAllDecorationStringLiterals(DecorationUserSemantic);
    for (size_t I = 0; I != Annotations.size(); ++I) {
      // UserSemantic has a single literal string
      llvm::SmallString<256> UserSemanticStr;
      llvm::raw_svector_ostream UserSemanticOut(UserSemanticStr);
      for (const auto &Str : Annotations[I])
        UserSemanticOut << Str;
      AnnotStrVec.emplace_back(UserSemanticStr);
    }
  }
}

void generateIntelFPGAAnnotationForStructMember(
    const SPIRVEntry *E, SPIRVWord MemberNumber,
    std::vector<llvm::SmallString<256>> &AnnotStrVec) {
  llvm::SmallString<256> AnnotStr;
  llvm::raw_svector_ostream Out(AnnotStr);
  if (E->hasMemberDecorate(DecorationRegisterINTEL, 0, MemberNumber))
    Out << "{register:1}";

  SPIRVWord Result = 0;
  if (E->hasMemberDecorate(DecorationMemoryINTEL, 0, MemberNumber, &Result))
    Out << "{memory:"
        << E->getMemberDecorationStringLiteral(DecorationMemoryINTEL,
                                               MemberNumber)
               .front()
        << '}';
  if (E->hasMemberDecorate(DecorationBankwidthINTEL, 0, MemberNumber, &Result))
    Out << "{bankwidth:" << Result << '}';
  if (E->hasMemberDecorate(DecorationNumbanksINTEL, 0, MemberNumber, &Result))
    Out << "{numbanks:" << Result << '}';
  if (E->hasMemberDecorate(DecorationMaxPrivateCopiesINTEL, 0, MemberNumber,
                           &Result))
    Out << "{private_copies:" << Result << '}';
  if (E->hasMemberDecorate(DecorationSinglepumpINTEL, 0, MemberNumber))
    Out << "{pump:1}";
  if (E->hasMemberDecorate(DecorationDoublepumpINTEL, 0, MemberNumber))
    Out << "{pump:2}";
  if (E->hasMemberDecorate(DecorationMaxReplicatesINTEL, 0, MemberNumber,
                           &Result))
    Out << "{max_replicates:" << Result << '}';
  if (E->hasMemberDecorate(DecorationSimpleDualPortINTEL, 0, MemberNumber))
    Out << "{simple_dual_port:1}";
  if (E->hasMemberDecorate(DecorationMergeINTEL, 0, MemberNumber)) {
    Out << "{merge";
    for (const auto &Str : E->getMemberDecorationStringLiteral(
             DecorationMergeINTEL, MemberNumber))
      Out << ":" << Str;
    Out << '}';
  }
  if (E->hasMemberDecorate(DecorationBankBitsINTEL, 0, MemberNumber)) {
    Out << "{bank_bits:";
    auto Literals =
        E->getMemberDecorationLiterals(DecorationBankBitsINTEL, MemberNumber);
    for (size_t I = 0; I < Literals.size() - 1; ++I)
      Out << Literals[I] << ",";
    Out << Literals.back() << '}';
  }
  if (E->hasMemberDecorate(DecorationForcePow2DepthINTEL, 0, MemberNumber,
                           &Result))
    Out << "{force_pow2_depth:" << Result << '}';
  if (E->hasMemberDecorate(DecorationStridesizeINTEL, 0, MemberNumber, &Result))
    Out << "{stride_size:" << Result << "}";
  if (E->hasMemberDecorate(DecorationWordsizeINTEL, 0, MemberNumber, &Result))
    Out << "{word_size:" << Result << "}";
  if (E->hasMemberDecorate(DecorationTrueDualPortINTEL, 0, MemberNumber))
    Out << "{true_dual_port}";
  if (!AnnotStr.empty())
    AnnotStrVec.emplace_back(AnnotStr);

  if (E->hasMemberDecorate(DecorationUserSemantic, 0, MemberNumber)) {
    auto Annotations = E->getAllMemberDecorationStringLiterals(
        DecorationUserSemantic, MemberNumber);
    for (size_t I = 0; I != Annotations.size(); ++I) {
      // UserSemantic has a single literal string
      llvm::SmallString<256> UserSemanticStr;
      llvm::raw_svector_ostream UserSemanticOut(UserSemanticStr);
      for (const auto &Str : Annotations[I])
        UserSemanticOut << Str;
      AnnotStrVec.emplace_back(UserSemanticStr);
    }
  }
}

void SPIRVToLLVM::transIntelFPGADecorations(SPIRVValue *BV, Value *V) {
  if (!BV->isVariable() && !BV->isInst())
    return;

  if (auto *Inst = dyn_cast<Instruction>(V)) {
    auto *AL = dyn_cast<AllocaInst>(Inst);
    Type *AllocatedTy = AL ? AL->getAllocatedType() : Inst->getType();

    IRBuilder<> Builder(Inst->getParent());

    Type *Int8PtrTyPrivate = PointerType::get(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = IntegerType::get(*Context, 32);

    Value *UndefInt8Ptr = PoisonValue::get(Int8PtrTyPrivate);
    Value *UndefInt32 = PoisonValue::get(Int32Ty);

    if (AL && BV->getType()->getPointerElementType()->isTypeStruct()) {
      auto *ST = BV->getType()->getPointerElementType();
      SPIRVTypeStruct *STS = static_cast<SPIRVTypeStruct *>(ST);

      for (SPIRVWord I = 0; I < STS->getMemberCount(); ++I) {
        std::vector<SmallString<256>> AnnotStrVec;
        generateIntelFPGAAnnotationForStructMember(ST, I, AnnotStrVec);
        CallInst *AnnotationCall = nullptr;
        for (const auto &AnnotStr : AnnotStrVec) {
          auto *GS = Builder.CreateGlobalString(AnnotStr);

          Instruction *PtrAnnFirstArg = nullptr;

          if (GEPOrUseMap.count(AL)) {
            auto IdxToInstMap = GEPOrUseMap[AL];
            if (IdxToInstMap.count(I)) {
              PtrAnnFirstArg = IdxToInstMap[I];
            }
          }

          Type *IntTy = nullptr;

          if (!PtrAnnFirstArg) {
            GetElementPtrInst *GEP = cast<GetElementPtrInst>(
                Builder.CreateConstInBoundsGEP2_32(AllocatedTy, AL, 0, I));

            IntTy = GEP->getResultElementType()->isIntegerTy()
                        ? GEP->getType()
                        : Int8PtrTyPrivate;
            PtrAnnFirstArg = GEP;
          } else {
            IntTy = PtrAnnFirstArg->getType();
          }

          auto *AnnotationFn = llvm::Intrinsic::getOrInsertDeclaration(
              M, Intrinsic::ptr_annotation, {IntTy, Int8PtrTyPrivate});

          llvm::Value *Args[] = {
              Builder.CreateBitCast(PtrAnnFirstArg, IntTy,
                                    PtrAnnFirstArg->getName()),
              Builder.CreateBitCast(GS, Int8PtrTyPrivate), UndefInt8Ptr,
              UndefInt32, UndefInt8Ptr};
          AnnotationCall = Builder.CreateCall(AnnotationFn, Args);
          GEPOrUseMap[AL][I] = AnnotationCall;
        }
        if (AnnotationCall)
          ValueMap[BV] = AnnotationCall;
      }
    }

    std::vector<SmallString<256>> AnnotStrVec;
    generateIntelFPGAAnnotation(BV, AnnotStrVec);
    CallInst *AnnotationCall = nullptr;
    for (const auto &AnnotStr : AnnotStrVec) {
      Constant *GS = nullptr;
      const auto StringAnnotStr = static_cast<std::string>(AnnotStr);
      auto AnnotItr = AnnotationsMap.find(StringAnnotStr);
      if (AnnotItr != AnnotationsMap.end()) {
        GS = AnnotItr->second;
      } else {
        GS = Builder.CreateGlobalString(AnnotStr);
        AnnotationsMap.emplace(std::move(StringAnnotStr), GS);
      }

      Value *BaseInst = nullptr;
      if (AnnotationCall && !AnnotationCall->getType()->isVoidTy())
        BaseInst = AnnotationCall;
      else
        BaseInst = AL ? Builder.CreateBitCast(V, Int8PtrTyPrivate, V->getName())
                      : Inst;

      // Try to find alloca instruction for statically allocated variables.
      // Alloca might be hidden by a couple of casts.
      bool isStaticMemoryAttribute = AL ? true : false;
      while (!isStaticMemoryAttribute && Inst &&
             (isa<BitCastInst>(Inst) || isa<AddrSpaceCastInst>(Inst))) {
        Inst = dyn_cast<Instruction>(Inst->getOperand(0));
        isStaticMemoryAttribute = (Inst && isa<AllocaInst>(Inst));
      }
      auto *AnnotationFn = llvm::Intrinsic::getOrInsertDeclaration(
          M,
          isStaticMemoryAttribute ? Intrinsic::var_annotation
                                  : Intrinsic::ptr_annotation,
          {BaseInst->getType(), Int8PtrTyPrivate});

      llvm::Value *Args[] = {BaseInst,
                             Builder.CreateBitCast(GS, Int8PtrTyPrivate),
                             UndefInt8Ptr, UndefInt32, UndefInt8Ptr};
      AnnotationCall = Builder.CreateCall(AnnotationFn, Args);
    }
    if (AnnotationCall && !AnnotationCall->getType()->isVoidTy())
      ValueMap[BV] = AnnotationCall;
  } else if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    // Do not add annotations for builtin variables if they will be translated
    // to function calls.
    SPIRVBuiltinVariableKind Kind;
    if (BM->getBuiltinFormat() == BuiltinFormat::Function &&
        isSPIRVBuiltinVariable(GV, &Kind))
      return;

    std::vector<SmallString<256>> AnnotStrVec;
    generateIntelFPGAAnnotation(BV, AnnotStrVec);

    if (AnnotStrVec.empty()) {
      // Check if IO pipe decoration is applied to the global
      SPIRVWord ID;
      if (BV->hasDecorate(DecorationIOPipeStorageINTEL, 0, &ID)) {
        auto Literals = BV->getDecorationLiterals(DecorationIOPipeStorageINTEL);
        assert(Literals.size() == 1 &&
               "IO PipeStorage decoration shall have 1 extra operand");
        GV->setMetadata("io_pipe_id", getMDNodeStringIntVec(Context, Literals));
      }
      return;
    }

    for (const auto &AnnotStr : AnnotStrVec) {
      Constant *StrConstant =
          ConstantDataArray::getString(*Context, StringRef(AnnotStr));

      auto *GS = new GlobalVariable(
          *GV->getParent(), StrConstant->getType(),
          /*IsConstant*/ true, GlobalValue::PrivateLinkage, StrConstant, "");

      GS->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      GS->setSection("llvm.metadata");

      Type *ResType = PointerType::get(
          GV->getContext(), M->getDataLayout().getDefaultGlobalsAddressSpace());
      Constant *C = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, ResType);

      Type *Int8PtrTyPrivate = PointerType::get(*Context, SPIRAS_Private);
      IntegerType *Int32Ty = Type::getInt32Ty(*Context);

      llvm::Constant *Fields[5] = {
          C, ConstantExpr::getBitCast(GS, Int8PtrTyPrivate),
          PoisonValue::get(Int8PtrTyPrivate), PoisonValue::get(Int32Ty),
          PoisonValue::get(Int8PtrTyPrivate)};

      GlobalAnnotations.push_back(ConstantStruct::getAnon(Fields));
    }
  }
}

// Translate aliasing decorations applied to instructions. These decorations
// are mapped on alias.scope and noalias metadata in LLVM. Translation of
// optional string operand isn't yet supported in the translator.
void SPIRVToLLVM::transMemAliasingINTELDecorations(SPIRVValue *BV, Value *V) {
  if (!BV->isInst())
    return;
  Instruction *Inst = dyn_cast<Instruction>(V);
  if (!Inst)
    return;
  if (BV->hasDecorateId(DecorationAliasScopeINTEL)) {
    std::vector<SPIRVId> AliasListIds;
    AliasListIds = BV->getDecorationIdLiterals(DecorationAliasScopeINTEL);
    assert(AliasListIds.size() == 1 &&
           "Memory aliasing decorations must have one argument");
    addMemAliasMetadata(Inst, AliasListIds[0], LLVMContext::MD_alias_scope);
  }
  if (BV->hasDecorateId(DecorationNoAliasINTEL)) {
    std::vector<SPIRVId> AliasListIds;
    AliasListIds = BV->getDecorationIdLiterals(DecorationNoAliasINTEL);
    assert(AliasListIds.size() == 1 &&
           "Memory aliasing decorations must have one argument");
    addMemAliasMetadata(Inst, AliasListIds[0], LLVMContext::MD_noalias);
  }
}

// Having UserSemantic decoration on Function is against the spec, but we allow
// this for various purposes (like prototyping new features when we need to
// attach some information on function and propagate that through SPIR-V and
// ect.)
void SPIRVToLLVM::transUserSemantic(SPIRV::SPIRVFunction *Fun) {
  auto *TransFun = transFunction(Fun);
  for (const auto &UsSem :
       Fun->getDecorationStringLiteral(DecorationUserSemantic)) {
    auto *V = cast<Value>(TransFun);
    Constant *StrConstant =
        ConstantDataArray::getString(*Context, StringRef(UsSem));
    auto *GS = new GlobalVariable(
        *TransFun->getParent(), StrConstant->getType(),
        /*IsConstant*/ true, GlobalValue::PrivateLinkage, StrConstant, "");

    GS->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GS->setSection("llvm.metadata");

    Type *ResType = PointerType::get(
        V->getContext(), M->getDataLayout().getDefaultGlobalsAddressSpace());
    Constant *C =
        ConstantExpr::getPointerBitCastOrAddrSpaceCast(TransFun, ResType);

    Type *Int8PtrTyPrivate = PointerType::get(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = Type::getInt32Ty(*Context);

    llvm::Constant *Fields[5] = {
        C, ConstantExpr::getBitCast(GS, Int8PtrTyPrivate),
        PoisonValue::get(Int8PtrTyPrivate), PoisonValue::get(Int32Ty),
        PoisonValue::get(Int8PtrTyPrivate)};
    GlobalAnnotations.push_back(ConstantStruct::getAnon(Fields));
  }
}

void SPIRVToLLVM::transGlobalAnnotations() {
  if (!GlobalAnnotations.empty()) {
    Constant *Array =
        ConstantArray::get(ArrayType::get(GlobalAnnotations[0]->getType(),
                                          GlobalAnnotations.size()),
                           GlobalAnnotations);
    auto *GV = new GlobalVariable(*M, Array->getType(), /*IsConstant*/ false,
                                  GlobalValue::AppendingLinkage, Array,
                                  "llvm.global.annotations");
    GV->setSection("llvm.metadata");
  }
}

static llvm::MDNode *
transDecorationsToMetadataList(llvm::LLVMContext *Context,
                               std::vector<SPIRVDecorate const *> Decorates) {
  SmallVector<Metadata *, 4> MDs;
  MDs.reserve(Decorates.size());
  for (const auto *Deco : Decorates) {
    std::vector<Metadata *> OPs;
    auto *KindMD = ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(*Context), Deco->getDecorateKind()));
    OPs.push_back(KindMD);
    switch (static_cast<size_t>(Deco->getDecorateKind())) {
    case DecorationLinkageAttributes: {
      const auto *const LinkAttrDeco =
          static_cast<const SPIRVDecorateLinkageAttr *>(Deco);
      auto *const LinkNameMD =
          MDString::get(*Context, LinkAttrDeco->getLinkageName());
      auto *const LinkTypeMD = ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt32Ty(*Context), LinkAttrDeco->getLinkageType()));
      OPs.push_back(LinkNameMD);
      OPs.push_back(LinkTypeMD);
      break;
    }
    case spv::internal::DecorationHostAccessINTEL:
    case DecorationHostAccessINTEL: {
      const auto *const HostAccDeco =
          static_cast<const SPIRVDecorateHostAccessINTEL *>(Deco);
      auto *const AccModeMD = ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt32Ty(*Context), HostAccDeco->getAccessMode()));
      auto *const NameMD = MDString::get(*Context, HostAccDeco->getVarName());
      OPs.push_back(AccModeMD);
      OPs.push_back(NameMD);
      break;
    }
    case DecorationMergeINTEL: {
      const auto MergeAttrLits = Deco->getVecLiteral();
      std::string FirstString = getString(MergeAttrLits);
      std::string SecondString =
          getString(MergeAttrLits.cbegin() + getVec(FirstString).size(),
                    MergeAttrLits.cend());
      OPs.push_back(MDString::get(*Context, FirstString));
      OPs.push_back(MDString::get(*Context, SecondString));
      break;
    }
    case DecorationMemoryINTEL:
    case DecorationUserSemantic: {
      auto *const StrMD =
          MDString::get(*Context, getString(Deco->getVecLiteral()));
      OPs.push_back(StrMD);
      break;
    }
    default: {
      for (const SPIRVWord Lit : Deco->getVecLiteral()) {
        auto *const LitMD = ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(*Context), Lit));
        OPs.push_back(LitMD);
      }
      break;
    }
    }
    MDs.push_back(MDNode::get(*Context, OPs));
  }
  return MDNode::get(*Context, MDs);
}

void SPIRVToLLVM::transDecorationsToMetadata(SPIRVValue *BV, Value *V) {
  if (!BV->isVariable() && !BV->isInst())
    return;

  auto SetDecorationsMetadata = [&](auto V) {
    std::vector<SPIRVDecorate const *> Decorates = BV->getDecorations();
    if (!Decorates.empty()) {
      MDNode *MDList = transDecorationsToMetadataList(Context, Decorates);
      V->setMetadata(SPIRV_MD_DECORATIONS, MDList);
    }
  };

  if (auto *GV = dyn_cast<GlobalVariable>(V))
    SetDecorationsMetadata(GV);
  else if (auto *I = dyn_cast<Instruction>(V))
    SetDecorationsMetadata(I);
}

namespace {

static float convertSPIRVWordToFloat(SPIRVWord Spir) {
  union {
    float F;
    SPIRVWord Spir;
  } FPMaxError;
  FPMaxError.Spir = Spir;
  return FPMaxError.F;
}

static bool transFPMaxErrorDecoration(SPIRVValue *BV, Value *V,
                                      LLVMContext *Context) {
  SPIRVWord ID;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (BV->hasDecorate(DecorationFPMaxErrorDecorationINTEL, 0, &ID)) {
      auto Literals =
          BV->getDecorationLiterals(DecorationFPMaxErrorDecorationINTEL);
      assert(Literals.size() == 1 &&
             "FP Max Error decoration shall have 1 operand");
      auto F = convertSPIRVWordToFloat(Literals[0]);
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        // Add attribute
        auto A = llvm::Attribute::get(*Context, "fpbuiltin-max-error",
                                      std::to_string(F));
        CI->addFnAttr(A);
      } else {
        // Add metadata
        MDNode *N =
            MDNode::get(*Context, MDString::get(*Context, std::to_string(F)));
        I->setMetadata("fpbuiltin-max-error", N);
      }
      return true;
    }
  return false;
}
} // namespace

bool SPIRVToLLVM::transDecoration(SPIRVValue *BV, Value *V) {
  if (transFPMaxErrorDecoration(BV, V, Context))
    return true;

  if (!transAlign(BV, V))
    return false;

  transIntelFPGADecorations(BV, V);
  transMemAliasingINTELDecorations(BV, V);

  // Decoration metadata is only enabled in SPIR-V friendly mode
  if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR)
    transDecorationsToMetadata(BV, V);

  DbgTran->transDbgInfo(BV, V);
  return true;
}

void SPIRVToLLVM::transGlobalCtorDtors(SPIRVVariableBase *BV) {
  if (BV->getName() != "llvm.global_ctors" &&
      BV->getName() != "llvm.global_dtors")
    return;

  Value *V = transValue(BV, nullptr, nullptr);
  cast<GlobalValue>(V)->setLinkage(GlobalValue::AppendingLinkage);
}

void SPIRVToLLVM::createCXXStructor(const char *ListName,
                                    SmallVectorImpl<Function *> &Funcs) {
  if (Funcs.empty())
    return;

  // If the SPIR-V input contained a variable for the structor list and it
  // has already been translated, then don't interfere.
  if (M->getGlobalVariable(ListName))
    return;

  // Type of a structor entry: { i32, void ()*, i8* }
  Type *PriorityTy = Type::getInt32Ty(*Context);
  PointerType *CtorTy = PointerType::getUnqual(*Context);
  PointerType *ComdatTy = PointerType::getUnqual(*Context);
  StructType *StructorTy = StructType::get(PriorityTy, CtorTy, ComdatTy);

  ArrayType *ArrTy = ArrayType::get(StructorTy, Funcs.size());

  GlobalVariable *GV =
      cast<GlobalVariable>(M->getOrInsertGlobal(ListName, ArrTy));
  GV->setLinkage(GlobalValue::AppendingLinkage);

  // Build the initializer.
  SmallVector<Constant *, 2> ArrayElts;
  for (auto *F : Funcs) {
    SmallVector<Constant *, 3> Elts;
    // SPIR-V does not specify an order between Initializers, so set default
    // priority.
    Elts.push_back(ConstantInt::get(PriorityTy, 65535));
    Elts.push_back(ConstantExpr::getBitCast(F, CtorTy));
    Elts.push_back(ConstantPointerNull::get(ComdatTy));
    ArrayElts.push_back(ConstantStruct::get(StructorTy, Elts));
  }

  Constant *NewArray = ConstantArray::get(ArrTy, ArrayElts);
  GV->setInitializer(NewArray);
}

bool SPIRVToLLVM::transFPContractMetadata() {
  bool ContractOff = false;
  for (unsigned I = 0, E = BM->getNumFunctions(); I != E; ++I) {
    SPIRVFunction *BF = BM->getFunction(I);
    if (!isKernel(BF))
      continue;
    if (BF->getExecutionMode(ExecutionModeContractionOff)) {
      ContractOff = true;
      break;
    }
  }
  if (!ContractOff)
    M->getOrInsertNamedMetadata(kSPIR2MD::FPContract);
  return true;
}

std::string
SPIRVToLLVM::transOCLImageTypeAccessQualifier(SPIRV::SPIRVTypeImage *ST) {
  return SPIRSPIRVAccessQualifierMap::rmap(ST->hasAccessQualifier()
                                               ? ST->getAccessQualifier()
                                               : AccessQualifierReadOnly);
}

bool SPIRVToLLVM::transNonTemporalMetadata(Instruction *I) {
  Constant *One = ConstantInt::get(Type::getInt32Ty(*Context), 1);
  MDNode *Node = MDNode::get(*Context, ConstantAsMetadata::get(One));
  I->setMetadata(M->getMDKindID("nontemporal"), Node);
  return true;
}

// Information of types of kernel arguments may be additionally stored in
// 'OpString "kernel_arg_type.%kernel_name%.type1,type2,type3,..' instruction.
// Try to find such instruction and generate metadata based on it.
// Return 'true' if 'OpString' was found and 'kernel_arg_type' metadata
// generated and 'false' otherwise.
static bool transKernelArgTypeMedataFromString(LLVMContext *Ctx,
                                               SPIRVModule *BM,
                                               Function *Kernel,
                                               std::string MDName) {
  // Run W/A translation only if the appropriate option is passed
  if (!BM->shouldPreserveOCLKernelArgTypeMetadataThroughString())
    return false;
  std::string ArgTypePrefix =
      std::string(MDName) + "." + Kernel->getName().str() + ".";
  auto ArgTypeStrIt = std::find_if(
      BM->getStringVec().begin(), BM->getStringVec().end(),
      [=](SPIRVString *S) { return S->getStr().find(ArgTypePrefix) == 0; });

  if (ArgTypeStrIt == BM->getStringVec().end())
    return false;

  std::string ArgTypeStr =
      (*ArgTypeStrIt)->getStr().substr(ArgTypePrefix.size());
  std::vector<Metadata *> TypeMDs;

  int CountBraces = 0;
  std::string::size_type Start = 0;

  for (std::string::size_type I = 0; I < ArgTypeStr.length(); I++) {
    switch (ArgTypeStr[I]) {
    case '<':
      CountBraces++;
      break;
    case '>':
      CountBraces--;
      break;
    case ',':
      if (CountBraces == 0) {
        TypeMDs.push_back(
            MDString::get(*Ctx, ArgTypeStr.substr(Start, I - Start)));
        Start = I + 1;
      }
    }
  }

  Kernel->setMetadata(MDName, MDNode::get(*Ctx, TypeMDs));
  return true;
}

void SPIRVToLLVM::transFunctionDecorationsToMetadata(SPIRVFunction *BF,
                                                     Function *F) {
  size_t TotalParameterDecorations = 0;
  BF->foreachArgument([&](SPIRVFunctionParameter *Arg) {
    TotalParameterDecorations += Arg->getNumDecorations();
  });
  if (TotalParameterDecorations == 0)
    return;

  // Generate metadata for spirv.ParameterDecorations
  addKernelArgumentMetadata(Context, SPIRV_MD_PARAMETER_DECORATIONS, BF, F,
                            [=](SPIRVFunctionParameter *Arg) {
                              return transDecorationsToMetadataList(
                                  Context, Arg->getDecorations());
                            });
}

bool SPIRVToLLVM::transMetadata() {
  SmallVector<Function *, 2> CtorKernels;
  for (unsigned I = 0, E = BM->getNumFunctions(); I != E; ++I) {
    SPIRVFunction *BF = BM->getFunction(I);
    Function *F = static_cast<Function *>(getTranslatedValue(BF));
    assert(F && "Invalid translated function");

    transOCLMetadata(BF);
    transVectorComputeMetadata(BF);
    transFPGAFunctionMetadata(BF, F);

    // Decoration metadata is only enabled in SPIR-V friendly mode
    if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR)
      transFunctionDecorationsToMetadata(BF, F);

    if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Generate metadata for reqd_work_group_size
    if (auto *EM = BF->getExecutionMode(ExecutionModeLocalSize)) {
      F->setMetadata(kSPIR2MD::WGSize,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    } else if (auto *EM = BF->getExecutionModeId(ExecutionModeLocalSizeId)) {
      std::vector<SPIRVWord> Values;
      for (const auto Id : EM->getLiterals()) {
        if (auto Val = transIdAsConstant(Id)) {
          Values.emplace_back(static_cast<SPIRVWord>(*Val));
        }
      }
      F->setMetadata(kSPIR2MD::WGSize, getMDNodeStringIntVec(Context, Values));
    }
    // Generate metadata for work_group_size_hint
    if (auto *EM = BF->getExecutionMode(ExecutionModeLocalSizeHint)) {
      F->setMetadata(kSPIR2MD::WGSizeHint,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    } else if (auto *EM =
                   BF->getExecutionModeId(ExecutionModeLocalSizeHintId)) {
      std::vector<SPIRVWord> Values;
      for (const auto Id : EM->getLiterals()) {
        if (auto Val = transIdAsConstant(Id)) {
          Values.emplace_back(static_cast<SPIRVWord>(*Val));
        }
      }
      F->setMetadata(kSPIR2MD::WGSizeHint,
                     getMDNodeStringIntVec(Context, Values));
    }
    // Generate metadata for vec_type_hint
    if (auto *EM = BF->getExecutionMode(ExecutionModeVecTypeHint)) {
      std::vector<Metadata *> MetadataVec;
      Type *VecHintTy = decodeVecTypeHint(*Context, EM->getLiterals()[0]);
      assert(VecHintTy);
      MetadataVec.push_back(ValueAsMetadata::get(PoisonValue::get(VecHintTy)));
      MetadataVec.push_back(ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(*Context), 1)));
      F->setMetadata(kSPIR2MD::VecTyHint, MDNode::get(*Context, MetadataVec));
    }
    // Generate metadata for Initializer.
    if (BF->getExecutionMode(ExecutionModeInitializer)) {
      CtorKernels.push_back(F);
    }
    // Generate metadata for intel_reqd_sub_group_size
    if (auto *EM = BF->getExecutionMode(ExecutionModeSubgroupSize)) {
      auto *SizeMD =
          ConstantAsMetadata::get(getUInt32(M, EM->getLiterals()[0]));
      F->setMetadata(kSPIR2MD::SubgroupSize, MDNode::get(*Context, SizeMD));
    }
    // Generate metadata for intel_reqd_sub_group_size
    if (BF->getExecutionMode(internal::ExecutionModeNamedSubgroupSizeINTEL)) {
      // For now, there is only one named sub group size: primary, which is
      // represented as a value of 0 as the argument of the OpExecutionMode.
      assert(BF->getExecutionMode(internal::ExecutionModeNamedSubgroupSizeINTEL)
                     ->getLiterals()[0] == 0 &&
             "Invalid named sub group size");
      // On the LLVM IR side, this is represented as the metadata
      // intel_reqd_sub_group_size with value -1.
      auto *SizeMD = ConstantAsMetadata::get(getInt32(M, -1));
      F->setMetadata(kSPIR2MD::SubgroupSize, MDNode::get(*Context, SizeMD));
    }
    // Generate metadata for SubgroupsPerWorkgroup/SubgroupsPerWorkgroupId.
    auto EmitSubgroupsPerWorkgroupMD = [this, F](SPIRVExecutionModeKind EMK,
                                                 uint64_t Value) {
      NamedMDNode *ExecModeMD =
          M->getOrInsertNamedMetadata(kSPIRVMD::ExecutionMode);
      SmallVector<Metadata *, 2> OperandVec;
      OperandVec.push_back(ConstantAsMetadata::get(F));
      OperandVec.push_back(ConstantAsMetadata::get(getUInt32(M, EMK)));
      OperandVec.push_back(ConstantAsMetadata::get(getUInt32(M, Value)));
      ExecModeMD->addOperand(MDNode::get(*Context, OperandVec));
    };
    if (auto *EM = BF->getExecutionMode(ExecutionModeSubgroupsPerWorkgroup)) {
      EmitSubgroupsPerWorkgroupMD(EM->getExecutionMode(), EM->getLiterals()[0]);
    } else if (auto *EM = BF->getExecutionModeId(
                   ExecutionModeSubgroupsPerWorkgroupId)) {
      if (auto Val = transIdAsConstant(EM->getLiterals()[0])) {
        EmitSubgroupsPerWorkgroupMD(EM->getExecutionMode(), *Val);
      }
    }
    // Generate metadata for max_work_group_size
    if (auto *EM = BF->getExecutionMode(ExecutionModeMaxWorkgroupSizeINTEL)) {
      F->setMetadata(kSPIR2MD::MaxWGSize,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for no_global_work_offset
    if (BF->getExecutionMode(ExecutionModeNoGlobalOffsetINTEL)) {
      F->setMetadata(kSPIR2MD::NoGlobalOffset, MDNode::get(*Context, {}));
    }
    // Generate metadata for max_global_work_dim
    if (auto *EM = BF->getExecutionMode(ExecutionModeMaxWorkDimINTEL)) {
      F->setMetadata(kSPIR2MD::MaxWGDim,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for num_simd_work_items
    if (auto *EM = BF->getExecutionMode(ExecutionModeNumSIMDWorkitemsINTEL)) {
      F->setMetadata(kSPIR2MD::NumSIMD,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for scheduler_target_fmax_mhz
    if (auto *EM =
            BF->getExecutionMode(ExecutionModeSchedulerTargetFmaxMhzINTEL)) {
      F->setMetadata(kSPIR2MD::FmaxMhz,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for Intel FPGA register map interface
    if (auto *EM =
            BF->getExecutionMode(ExecutionModeRegisterMapInterfaceINTEL)) {
      std::vector<uint32_t> InterfaceVec = EM->getLiterals();
      assert(InterfaceVec.size() == 1 &&
             "Expected RegisterMapInterfaceINTEL to have exactly 1 literal");
      std::vector<Metadata *> InterfaceMDVec =
          [&]() -> std::vector<Metadata *> {
        switch (InterfaceVec[0]) {
        case 0:
          return {MDString::get(*Context, "csr")};
        case 1:
          return {MDString::get(*Context, "csr"),
                  MDString::get(*Context, "wait_for_done_write")};
        default:
          llvm_unreachable("Invalid register map interface mode");
        }
      }();
      F->setMetadata(kSPIR2MD::IntelFPGAIPInterface,
                     MDNode::get(*Context, InterfaceMDVec));
    }
    // Generate metadata for Intel FPGA streaming interface
    if (auto *EM = BF->getExecutionMode(ExecutionModeStreamingInterfaceINTEL)) {
      std::vector<uint32_t> InterfaceVec = EM->getLiterals();
      assert(InterfaceVec.size() == 1 &&
             "Expected StreamingInterfaceINTEL to have exactly 1 literal");
      std::vector<Metadata *> InterfaceMDVec =
          [&]() -> std::vector<Metadata *> {
        switch (InterfaceVec[0]) {
        case 0:
          return {MDString::get(*Context, "streaming")};
        case 1:
          return {MDString::get(*Context, "streaming"),
                  MDString::get(*Context, "stall_free_return")};
        default:
          llvm_unreachable("Invalid streaming interface mode");
        }
      }();
      F->setMetadata(kSPIR2MD::IntelFPGAIPInterface,
                     MDNode::get(*Context, InterfaceMDVec));
    }
    if (auto *EM = BF->getExecutionMode(ExecutionModeMaximumRegistersINTEL)) {
      NamedMDNode *ExecModeMD =
          M->getOrInsertNamedMetadata(kSPIRVMD::ExecutionMode);

      SmallVector<Metadata *, 4> ValueVec;
      ValueVec.push_back(ConstantAsMetadata::get(F));
      ValueVec.push_back(
          ConstantAsMetadata::get(getUInt32(M, EM->getExecutionMode())));
      ValueVec.push_back(
          ConstantAsMetadata::get(getUInt32(M, EM->getLiterals()[0])));
      ExecModeMD->addOperand(MDNode::get(*Context, ValueVec));
    }
    if (auto *EM = BF->getExecutionMode(ExecutionModeMaximumRegistersIdINTEL)) {
      NamedMDNode *ExecModeMD =
          M->getOrInsertNamedMetadata(kSPIRVMD::ExecutionMode);

      SmallVector<Metadata *, 4> ValueVec;
      ValueVec.push_back(ConstantAsMetadata::get(F));
      ValueVec.push_back(
          ConstantAsMetadata::get(getUInt32(M, EM->getExecutionMode())));

      auto *ExecOp = BF->getModule()->getValue(EM->getLiterals()[0]);
      ValueVec.push_back(
          MDNode::get(*Context, ConstantAsMetadata::get(cast<ConstantInt>(
                                    transValue(ExecOp, nullptr, nullptr)))));
      ExecModeMD->addOperand(MDNode::get(*Context, ValueVec));
    }
    if (auto *EM =
            BF->getExecutionMode(ExecutionModeNamedMaximumRegistersINTEL)) {
      NamedMDNode *ExecModeMD =
          M->getOrInsertNamedMetadata(kSPIRVMD::ExecutionMode);

      SmallVector<Metadata *, 4> ValueVec;
      ValueVec.push_back(ConstantAsMetadata::get(F));
      ValueVec.push_back(
          ConstantAsMetadata::get(getUInt32(M, EM->getExecutionMode())));

      assert(EM->getLiterals()[0] == 0 &&
             "Invalid named maximum number of registers");
      ValueVec.push_back(MDString::get(*Context, "AutoINTEL"));
      ExecModeMD->addOperand(MDNode::get(*Context, ValueVec));
    }
  }
  NamedMDNode *MemoryModelMD =
      M->getOrInsertNamedMetadata(kSPIRVMD::MemoryModel);
  MemoryModelMD->addOperand(
      getMDTwoInt(Context, static_cast<unsigned>(BM->getAddressingModel()),
                  static_cast<unsigned>(BM->getMemoryModel())));
  createCXXStructor("llvm.global_ctors", CtorKernels);
  return true;
}

bool SPIRVToLLVM::transOCLMetadata(SPIRVFunction *BF) {
  Function *F = static_cast<Function *>(getTranslatedValue(BF));
  assert(F && "Invalid translated function");
  if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
    return true;

  if (BF->hasDecorate(DecorationVectorComputeFunctionINTEL))
    return true;

  // Generate metadata for kernel_arg_addr_space
  addKernelArgumentMetadata(
      Context, SPIR_MD_KERNEL_ARG_ADDR_SPACE, BF, F,
      [=](SPIRVFunctionParameter *Arg) {
        SPIRVType *ArgTy = Arg->getType();
        SPIRAddressSpace AS = SPIRAS_Private;
        if (ArgTy->isTypePointer())
          AS = SPIRSPIRVAddrSpaceMap::rmap(ArgTy->getPointerStorageClass());
        else if (ArgTy->isTypeOCLImage() || ArgTy->isTypePipe())
          AS = SPIRAS_Global;
        return ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(*Context), AS));
      });
  // Generate metadata for kernel_arg_access_qual
  addKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_ACCESS_QUAL, BF, F,
                            [=](SPIRVFunctionParameter *Arg) {
                              std::string Qual;
                              auto *T = Arg->getType();
                              if (T->isTypeOCLImage()) {
                                auto *ST = static_cast<SPIRVTypeImage *>(T);
                                Qual = transOCLImageTypeAccessQualifier(ST);
                              } else if (T->isTypePipe()) {
                                auto *PT = static_cast<SPIRVTypePipe *>(T);
                                Qual = transOCLPipeTypeAccessQualifier(PT);
                              } else
                                Qual = "none";
                              return MDString::get(*Context, Qual);
                            });
  // Generate metadata for kernel_arg_type
  if (!transKernelArgTypeMedataFromString(Context, BM, F,
                                          SPIR_MD_KERNEL_ARG_TYPE))
    addKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_TYPE, BF, F,
                              [=](SPIRVFunctionParameter *Arg) {
                                return transOCLKernelArgTypeName(Arg);
                              });
  // Generate metadata for kernel_arg_type_qual
  if (!transKernelArgTypeMedataFromString(Context, BM, F,
                                          SPIR_MD_KERNEL_ARG_TYPE_QUAL))
    addKernelArgumentMetadata(
        Context, SPIR_MD_KERNEL_ARG_TYPE_QUAL, BF, F,
        [=](SPIRVFunctionParameter *Arg) {
          std::string Qual;
          if (Arg->hasDecorate(DecorationVolatile))
            Qual = kOCLTypeQualifierName::Volatile;
          Arg->foreachAttr([&](SPIRVFuncParamAttrKind Kind) {
            Qual += Qual.empty() ? "" : " ";
            if (Kind == FunctionParameterAttributeNoAlias)
              Qual += kOCLTypeQualifierName::Restrict;
          });
          if (Arg->getType()->isTypePipe()) {
            Qual += Qual.empty() ? "" : " ";
            Qual += kOCLTypeQualifierName::Pipe;
          }
          return MDString::get(*Context, Qual);
        });
  // Generate metadata for kernel_arg_base_type
  addKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_BASE_TYPE, BF, F,
                            [=](SPIRVFunctionParameter *Arg) {
                              return transOCLKernelArgTypeName(Arg);
                            });
  // Generate metadata for kernel_arg_name
  if (BM->isGenArgNameMDEnabled()) {
    addKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_NAME, BF, F,
                              [=](SPIRVFunctionParameter *Arg) {
                                return MDString::get(*Context, Arg->getName());
                              });
  }
  // Generate metadata for kernel_arg_buffer_location
  addBufferLocationMetadata(Context, BF, F, [=](SPIRVFunctionParameter *Arg) {
    auto Literals = Arg->getDecorationLiterals(DecorationBufferLocationINTEL);
    assert(Literals.size() == 1 &&
           "BufferLocationINTEL decoration shall have 1 ID literal");

    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(*Context), Literals[0]));
  });
  // Generate metadata for kernel_arg_runtime_aligned
  addRuntimeAlignedMetadata(Context, BF, F, [=](SPIRVFunctionParameter *Arg) {
    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt1Ty(*Context), 1));
  });
  // Generate metadata for spirv.ParameterDecorations
  addKernelArgumentMetadata(Context, SPIRV_MD_PARAMETER_DECORATIONS, BF, F,
                            [=](SPIRVFunctionParameter *Arg) {
                              return transDecorationsToMetadataList(
                                  Context, Arg->getDecorations());
                            });
  return true;
}

bool SPIRVToLLVM::transVectorComputeMetadata(SPIRVFunction *BF) {
  using namespace VectorComputeUtil;
  Function *F = static_cast<Function *>(getTranslatedValue(BF));
  assert(F && "Invalid translated function");

  if (BF->hasDecorate(DecorationStackCallINTEL))
    F->addFnAttr(kVCMetadata::VCStackCall);

  if (BF->hasDecorate(DecorationVectorComputeFunctionINTEL))
    F->addFnAttr(kVCMetadata::VCFunction);

  SPIRVWord SIMTMode = 0;
  if (BF->hasDecorate(DecorationSIMTCallINTEL, 0, &SIMTMode))
    F->addFnAttr(kVCMetadata::VCSIMTCall, std::to_string(SIMTMode));

  auto SEVAttr = translateSEVMetadata(BF, F->getContext());

  if (SEVAttr)
    F->addAttributeAtIndex(AttributeList::ReturnIndex, SEVAttr.value());

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    SPIRVWord Kind;
    if (BA->hasDecorate(DecorationFuncParamIOKindINTEL, 0, &Kind)) {
      Attribute Attr = Attribute::get(*Context, kVCMetadata::VCArgumentIOKind,
                                      std::to_string(Kind));
      F->addParamAttr(ArgNo, Attr);
    }
    SEVAttr = translateSEVMetadata(BA, F->getContext());
    if (SEVAttr)
      F->addParamAttr(ArgNo, SEVAttr.value());
    if (BA->hasDecorate(DecorationMediaBlockIOINTEL)) {
      assert(BA->getType()->isTypeImage() &&
             "MediaBlockIOINTEL decoration is valid only on image parameters");
      F->addParamAttr(ArgNo,
                      Attribute::get(*Context, kVCMetadata::VCMediaBlockIO));
    }
  }

  // Do not add float control if there is no any
  bool IsVCFloatControl = false;
  unsigned FloatControl = 0;
  // RoundMode and FloatMode are always same for all types in Cm
  // While Denorm could be different for double, float and half
  if (isKernel(BF)) {
    FPRoundingModeExecModeMap::foreach (
        [&](FPRoundingMode VCRM, ExecutionMode EM) {
          if (BF->getExecutionMode(EM)) {
            IsVCFloatControl = true;
            FloatControl |= getVCFloatControl(VCRM);
          }
        });
    FPOperationModeExecModeMap::foreach (
        [&](FPOperationMode VCFM, ExecutionMode EM) {
          if (BF->getExecutionMode(EM)) {
            IsVCFloatControl = true;
            FloatControl |= getVCFloatControl(VCFM);
          }
        });
    FPDenormModeExecModeMap::foreach ([&](FPDenormMode VCDM, ExecutionMode EM) {
      auto ExecModes = BF->getExecutionModeRange(EM);
      for (auto It = ExecModes.first; It != ExecModes.second; It++) {
        IsVCFloatControl = true;
        unsigned TargetWidth = (*It).second->getLiterals()[0];
        VCFloatType FloatType = VCFloatTypeSizeMap::rmap(TargetWidth);
        FloatControl |= getVCFloatControl(VCDM, FloatType);
      }
    });
  } else {
    if (BF->hasDecorate(DecorationFunctionRoundingModeINTEL)) {
      std::vector<SPIRVDecorate const *> RoundModes =
          BF->getDecorations(DecorationFunctionRoundingModeINTEL);

      assert(RoundModes.size() == 3 && "Function must have precisely 3 "
                                       "FunctionRoundingModeINTEL decoration");

      auto *DecRound =
          static_cast<SPIRVDecorateFunctionRoundingModeINTEL const *>(
              RoundModes.at(0));
      auto RoundingMode = DecRound->getRoundingMode();
#ifndef NDEBUG
      for (auto *DecPreCast : RoundModes) {
        auto *Dec = static_cast<SPIRVDecorateFunctionRoundingModeINTEL const *>(
            DecPreCast);
        assert(Dec->getRoundingMode() == RoundingMode &&
               "Rounding Mode must be equal within all targets");
      }
#endif
      IsVCFloatControl = true;
      FloatControl |= getVCFloatControl(RoundingMode);
    }

    if (BF->hasDecorate(DecorationFunctionDenormModeINTEL)) {
      std::vector<SPIRVDecorate const *> DenormModes =
          BF->getDecorations(DecorationFunctionDenormModeINTEL);
      IsVCFloatControl = true;

      for (const auto *DecPtr : DenormModes) {
        const auto *DecDenorm =
            static_cast<SPIRVDecorateFunctionDenormModeINTEL const *>(DecPtr);
        VCFloatType FType =
            VCFloatTypeSizeMap::rmap(DecDenorm->getTargetWidth());
        FloatControl |= getVCFloatControl(DecDenorm->getDenormMode(), FType);
      }
    }

    if (BF->hasDecorate(DecorationFunctionFloatingPointModeINTEL)) {
      std::vector<SPIRVDecorate const *> FloatModes =
          BF->getDecorations(DecorationFunctionFloatingPointModeINTEL);

      assert(FloatModes.size() == 3 &&
             "Function must have precisely 3 FunctionFloatingPointModeINTEL "
             "decoration");

      auto *DecFlt =
          static_cast<SPIRVDecorateFunctionFloatingPointModeINTEL const *>(
              FloatModes.at(0));
      auto FloatingMode = DecFlt->getOperationMode();
#ifndef NDEBUG
      for (auto *DecPreCast : FloatModes) {
        auto *Dec =
            static_cast<SPIRVDecorateFunctionFloatingPointModeINTEL const *>(
                DecPreCast);
        assert(Dec->getOperationMode() == FloatingMode &&
               "Rounding Mode must be equal within all targets");
      }
#endif

      IsVCFloatControl = true;
      FloatControl |= getVCFloatControl(FloatingMode);
    }
  }

  if (IsVCFloatControl) {
    Attribute Attr = Attribute::get(*Context, kVCMetadata::VCFloatControl,
                                    std::to_string(FloatControl));
    F->addFnAttr(Attr);
  }

  if (auto *EM =
          BF->getExecutionMode(ExecutionModeSharedLocalMemorySizeINTEL)) {
    unsigned int SLMSize = EM->getLiterals()[0];
    Attribute Attr = Attribute::get(*Context, kVCMetadata::VCSLMSize,
                                    std::to_string(SLMSize));
    F->addFnAttr(Attr);
  }

  if (auto *EM = BF->getExecutionMode(ExecutionModeNamedBarrierCountINTEL)) {
    unsigned int NBarrierCnt = EM->getLiterals()[0];
    Attribute Attr = Attribute::get(*Context, kVCMetadata::VCNamedBarrierCount,
                                    std::to_string(NBarrierCnt));
    F->addFnAttr(Attr);
  }

  return true;
}

bool SPIRVToLLVM::transFPGAFunctionMetadata(SPIRVFunction *BF, Function *F) {
  if (BF->hasDecorate(DecorationStallEnableINTEL)) {
    std::vector<Metadata *> MetadataVec;
    MetadataVec.push_back(ConstantAsMetadata::get(getInt32(M, 1)));
    F->setMetadata(kSPIR2MD::StallEnable, MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(DecorationStallFreeINTEL)) {
    std::vector<Metadata *> MetadataVec;
    MetadataVec.push_back(ConstantAsMetadata::get(getInt32(M, 1)));
    F->setMetadata(kSPIR2MD::StallFree, MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(DecorationFuseLoopsInFunctionINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals =
        BF->getDecorationLiterals(DecorationFuseLoopsInFunctionINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[1])));
    F->setMetadata(kSPIR2MD::LoopFuse, MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(DecorationMathOpDSPModeINTEL)) {
    std::vector<SPIRVWord> Literals =
        BF->getDecorationLiterals(DecorationMathOpDSPModeINTEL);
    assert(Literals.size() == 2 &&
           "MathOpDSPModeINTEL decoration shall have 2 literals");
    F->setMetadata(kSPIR2MD::PreferDSP,
                   MDNode::get(*Context, ConstantAsMetadata::get(
                                             getUInt32(M, Literals[0]))));
    if (Literals[1] != 0) {
      F->setMetadata(kSPIR2MD::PropDSPPref,
                     MDNode::get(*Context, ConstantAsMetadata::get(
                                               getUInt32(M, Literals[1]))));
    }
  }
  if (BF->hasDecorate(DecorationInitiationIntervalINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals =
        BF->getDecorationLiterals(DecorationInitiationIntervalINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    F->setMetadata(kSPIR2MD::InitiationInterval,
                   MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(DecorationMaxConcurrencyINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals = BF->getDecorationLiterals(DecorationMaxConcurrencyINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    F->setMetadata(kSPIR2MD::MaxConcurrency,
                   MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(DecorationPipelineEnableINTEL)) {
    auto Literals = BF->getDecorationLiterals(DecorationPipelineEnableINTEL);
    std::vector<Metadata *> MetadataVec;
    MetadataVec.push_back(ConstantAsMetadata::get(getInt32(M, Literals[0])));
    F->setMetadata(kSPIR2MD::PipelineKernel,
                   MDNode::get(*Context, MetadataVec));
  }
  return true;
}

bool SPIRVToLLVM::transAlign(SPIRVValue *BV, Value *V) {
  if (auto *AL = dyn_cast<AllocaInst>(V)) {
    if (auto Align = getAlignment(BV))
      AL->setAlignment(llvm::Align(*Align));
    return true;
  }
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    if (auto Align = getAlignment(BV))
      GV->setAlignment(MaybeAlign(*Align));
    return true;
  }
  return true;
}

Instruction *SPIRVToLLVM::transOCLBuiltinFromExtInst(SPIRVExtInst *BC,
                                                     BasicBlock *BB) {
  assert(BB && "Invalid BB");
  auto ExtOp = static_cast<OCLExtOpKind>(BC->getExtOp());
  std::string UnmangledName = OCLExtOpMap::map(ExtOp);

  assert(BM->getBuiltinSet(BC->getExtSetId()) == SPIRVEIS_OpenCL &&
         "Not OpenCL extended instruction");

  std::vector<Type *> ArgTypes = transTypeVector(BC->getArgTypes(), true);
  for (unsigned I = 0; I < ArgTypes.size(); I++) {
    // Special handling for "truly" untyped pointers to preserve correct OCL
    // bultin mangling.
    if (isa<PointerType>(ArgTypes[I]) &&
        BC->getArgValue(I)->isUntypedVariable()) {
      auto *BVar = static_cast<SPIRVUntypedVariableKHR *>(BC->getArgValue(I));
      ArgTypes[I] = TypedPointerType::get(
          transType(BVar->getDataType()),
          SPIRSPIRVAddrSpaceMap::rmap(BVar->getStorageClass()));
    }
  }

  Type *RetTy = transType(BC->getType());
  std::string MangledName =
      getSPIRVFriendlyIRFunctionName(ExtOp, ArgTypes, RetTy);
  opaquifyTypedPointers(ArgTypes);

  SPIRVDBG(spvdbgs() << "[transOCLBuiltinFromExtInst] UnmangledName: "
                     << UnmangledName << " MangledName: " << MangledName
                     << '\n');

  FunctionType *FT = FunctionType::get(RetTy, ArgTypes, false);
  Function *F = M->getFunction(MangledName);
  if (!F) {
    F = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    F->setCallingConv(CallingConv::SPIR_FUNC);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
    if (isFuncReadNone(UnmangledName))
      F->setDoesNotAccessMemory();
  }
  auto Args = transValue(BC->getArgValues(), F, BB);
  SPIRVDBG(dbgs() << "[transOCLBuiltinFromExtInst] Function: " << *F
                  << ", Args: ";
           for (auto &I : Args) dbgs() << *I << ", "; dbgs() << '\n');
  CallInst *CI = CallInst::Create(F, Args, BC->getName(), BB);
  setCallingConv(CI);
  addFnAttr(CI, Attribute::NoUnwind);
  return CI;
}

void SPIRVToLLVM::transAuxDataInst(SPIRVExtInst *BC) {
  assert(BC->getExtSetKind() == SPIRV::SPIRVEIS_NonSemantic_AuxData);
  if (!BC->getModule()->preserveAuxData())
    return;
  auto Args = BC->getArguments();
  // Args 0 and 1 are common between attributes and metadata.
  // 0 is the global object, 1 is the name of the attribute/metadata as a string
  auto *Arg0 = BC->getModule()->getValue(Args[0]);
  auto *GO = cast<GlobalObject>(getTranslatedValue(Arg0));
  auto *F = dyn_cast<Function>(GO);
  auto *GV = dyn_cast<GlobalVariable>(GO);
  assert((F || GV) && "Value should already have been translated!");
  auto AttrOrMDName = BC->getModule()->get<SPIRVString>(Args[1])->getStr();
  switch (BC->getExtOp()) {
  case NonSemanticAuxData::FunctionAttribute:
  case NonSemanticAuxData::GlobalVariableAttribute: {
    assert(Args.size() < 4 && "Unexpected FunctionAttribute Args");
    // If this attr was specially handled and added elsewhere, skip it.
    Attribute::AttrKind AsKind = Attribute::getAttrKindFromName(AttrOrMDName);
    if (AsKind != Attribute::None)
      if ((F && F->hasFnAttribute(AsKind)) || (GV && GV->hasAttribute(AsKind)))
        return;
    if (AsKind == Attribute::None)
      if ((F && F->hasFnAttribute(AttrOrMDName)) ||
          (GV && GV->hasAttribute(AttrOrMDName)))
        return;
    // For attributes, arg 2 is the attribute value as a string, which may not
    // exist.
    if (Args.size() == 3) {
      auto AttrValue = BC->getModule()->get<SPIRVString>(Args[2])->getStr();
      if (F)
        F->addFnAttr(AttrOrMDName, AttrValue);
      else
        GV->addAttribute(AttrOrMDName, AttrValue);
    } else {
      if (AsKind != Attribute::None) {
        if (F)
          F->addFnAttr(AsKind);
        else
          GV->addAttribute(AsKind);
      } else {
        if (F)
          F->addFnAttr(AttrOrMDName);
        else
          GV->addAttribute(AttrOrMDName);
      }
    }
    break;
  }
  case NonSemanticAuxData::FunctionMetadata:
  case NonSemanticAuxData::GlobalVariableMetadata: {
    // If this metadata was specially handled and added elsewhere, skip it.
    if (GO->hasMetadata(AttrOrMDName))
      return;
    SmallVector<Metadata *> MetadataArgs;
    // Process the metadata values.
    for (size_t CurArg = 2; CurArg < Args.size(); CurArg++) {
      auto *Arg = BC->getModule()->get<SPIRVEntry>(Args[CurArg]);
      // For metadata, the metadata values can be either values or strings.
      if (Arg->getOpCode() == OpString) {
        auto *ArgAsStr = static_cast<SPIRVString *>(Arg);
        MetadataArgs.push_back(
            MDString::get(GO->getContext(), ArgAsStr->getStr()));
      } else {
        auto *ArgAsVal = static_cast<SPIRVValue *>(Arg);
        auto *TranslatedMD = transValue(ArgAsVal, nullptr, nullptr);
        MetadataArgs.push_back(ValueAsMetadata::get(TranslatedMD));
      }
    }
    GO->setMetadata(AttrOrMDName, MDNode::get(*Context, MetadataArgs));
    break;
  }
  default:
    llvm_unreachable("Invalid op");
  }
}

// SPIR-V only contains language version. Use OpenCL language version as
// SPIR version.
void SPIRVToLLVM::transSourceLanguage() {
  SPIRVWord Ver = 0;
  SourceLanguage Lang = BM->getSourceLanguage(&Ver);
  if (Lang != SourceLanguageUnknown && // Allow unknown for debug info test
      Lang != SourceLanguageOpenCL_C && Lang != SourceLanguageCPP_for_OpenCL &&
      Lang != SourceLanguageOpenCL_CPP)
    return;
  unsigned short Major = 0;
  unsigned char Minor = 0;
  unsigned char Rev = 0;
  std::tie(Major, Minor, Rev) = decodeOCLVer(Ver);
  SPIRVMDBuilder Builder(*M);
  Builder.addNamedMD(kSPIRVMD::Source).addOp().add(Lang).add(Ver).done();
  // ToDo: Phasing out usage of old SPIR metadata
  if (Ver <= kOCLVer::CL12)
    addOCLVersionMetadata(Context, M, kSPIR2MD::SPIRVer, 1, 2);
  else
    addOCLVersionMetadata(Context, M, kSPIR2MD::SPIRVer, 2, 0);

  if (Lang == SourceLanguageOpenCL_C) {
    addOCLVersionMetadata(Context, M, kSPIR2MD::OCLVer, Major, Minor);
    return;
  }
  if (Lang == SourceLanguageCPP_for_OpenCL) {
    addOCLVersionMetadata(Context, M, kSPIR2MD::OCLCXXVer, Major, Minor);
    addOCLVersionMetadata(Context, M, kSPIR2MD::OCLVer,
                          Ver == kOCLVer::CLCXX10 ? 2 : 3, 0);
  }
}

bool SPIRVToLLVM::transSourceExtension() {
  auto ExtSet = rmap<OclExt::Kind>(BM->getExtension());
  auto CapSet = rmap<OclExt::Kind>(BM->getCapability());
  ExtSet.insert(CapSet.begin(), CapSet.end());
  auto OCLExtensions = map<std::string>(ExtSet);
  std::set<std::string> OCLOptionalCoreFeatures;
  static const char *OCLOptCoreFeatureNames[] = {
      "cl_images",
      "cl_doubles",
  };
  for (auto &I : OCLOptCoreFeatureNames) {
    auto Loc = OCLExtensions.find(I);
    if (Loc != OCLExtensions.end()) {
      OCLExtensions.erase(Loc);
      OCLOptionalCoreFeatures.insert(I);
    }
  }
  addNamedMetadataStringSet(Context, M, kSPIR2MD::Extensions, OCLExtensions);
  addNamedMetadataStringSet(Context, M, kSPIR2MD::OptFeatures,
                            OCLOptionalCoreFeatures);
  return true;
}

llvm::GlobalValue::LinkageTypes
SPIRVToLLVM::transLinkageType(const SPIRVValue *V) {
  std::string ValueName = V->getName();
  if (ValueName == "llvm.used" || ValueName == "llvm.compiler.used")
    return GlobalValue::AppendingLinkage;
  int LT = V->getLinkageType();
  switch (LT) {
  case internal::LinkageTypeInternal:
    return GlobalValue::InternalLinkage;
  case LinkageTypeImport:
    // Function declaration
    if (V->getOpCode() == OpFunction) {
      if (static_cast<const SPIRVFunction *>(V)->getNumBasicBlock() == 0)
        return GlobalValue::ExternalLinkage;
    }
    // Variable declaration
    if (V->getOpCode() == OpVariable ||
        V->getOpCode() == OpUntypedVariableKHR) {
      if (static_cast<const SPIRVVariableBase *>(V)->getInitializer() == 0)
        return GlobalValue::ExternalLinkage;
    }
    // Definition
    return GlobalValue::AvailableExternallyLinkage;
  case LinkageTypeExport:
    if (V->getOpCode() == OpVariable ||
        V->getOpCode() == OpUntypedVariableKHR) {
      if (static_cast<const SPIRVVariableBase *>(V)->getInitializer() == 0)
        // Tentative definition
        return GlobalValue::CommonLinkage;
    }
    return GlobalValue::ExternalLinkage;
  case LinkageTypeLinkOnceODR:
    return GlobalValue::LinkOnceODRLinkage;
  default:
    llvm_unreachable("Invalid linkage type");
  }
}

Instruction *SPIRVToLLVM::transAllAny(SPIRVInstruction *I, BasicBlock *BB) {
  CallInst *CI = cast<CallInst>(transSPIRVBuiltinFromInst(I, BB));
  auto Mutator = mutateCallInst(
      CI, getSPIRVFuncName(I->getOpCode(), getSPIRVFuncSuffix(I)));
  Mutator.mapArg(0, [](IRBuilder<> &Builder, Value *OldArg) {
    auto *NewArgTy = OldArg->getType()->getWithNewBitWidth(8);
    return Builder.CreateSExtOrBitCast(OldArg, NewArgTy);
  });
  return cast<Instruction>(Mutator.getMutated());
}

Instruction *SPIRVToLLVM::transRelational(SPIRVInstruction *I, BasicBlock *BB) {
  CallInst *CI = cast<CallInst>(transSPIRVBuiltinFromInst(I, BB));
  auto Mutator = mutateCallInst(
      CI, getSPIRVFuncName(I->getOpCode(), getSPIRVFuncSuffix(I)));
  if (CI->getType()->isVectorTy()) {
    Type *RetTy = CI->getType()->getWithNewBitWidth(8);
    Mutator.changeReturnType(RetTy, [=](IRBuilder<> &Builder, CallInst *NewCI) {
      return Builder.CreateTruncOrBitCast(NewCI, CI->getType());
    });
  }
  return cast<Instruction>(Mutator.getMutated());
}

std::optional<SPIRVModuleReport> getSpirvReport(std::istream &IS) {
  int IgnoreErrCode;
  return getSpirvReport(IS, IgnoreErrCode);
}

std::optional<SPIRVModuleReport> getSpirvReport(std::istream &IS,
                                                int &ErrCode) {
  SPIRVWord Word;
  std::string Name;
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  SPIRVDecoder D(IS, *BM);
  D >> Word;
  if (Word != MagicNumber) {
    ErrCode = SPIRVEC_InvalidMagicNumber;
    return {};
  }
  D >> Word;
  if (!isSPIRVVersionKnown(static_cast<VersionNumber>(Word))) {
    ErrCode = SPIRVEC_InvalidVersionNumber;
    return {};
  }
  SPIRVModuleReport Report;
  Report.Version = static_cast<SPIRV::VersionNumber>(Word);
  // Skip: Generator’s magic number, Bound and Reserved word
  D.ignore(3);

  bool IsReportGenCompleted = false, IsMemoryModelDefined = false;
  while (!IS.bad() && !IsReportGenCompleted && D.getWordCountAndOpCode()) {
    switch (D.OpCode) {
    case OpCapability:
      D >> Word;
      Report.Capabilities.push_back(Word);
      break;
    case OpExtension:
      Name.clear();
      D >> Name;
      Report.Extensions.push_back(Name);
      break;
    case OpExtInstImport:
      Name.clear();
      D >> Word >> Name;
      Report.ExtendedInstructionSets.push_back(Name);
      break;
    case OpMemoryModel:
      if (IsMemoryModelDefined) {
        ErrCode = SPIRVEC_RepeatedMemoryModel;
        return {};
      }
      SPIRVAddressingModelKind AddrModel;
      SPIRVMemoryModelKind MemoryModel;
      D >> AddrModel >> MemoryModel;
      if (!isValid(AddrModel)) {
        ErrCode = SPIRVEC_InvalidAddressingModel;
        return {};
      }
      if (!isValid(MemoryModel)) {
        ErrCode = SPIRVEC_InvalidMemoryModel;
        return {};
      }
      Report.MemoryModel = MemoryModel;
      Report.AddrModel = AddrModel;
      IsMemoryModelDefined = true;
      // In this report we don't analyze instructions after OpMemoryModel
      IsReportGenCompleted = true;
      break;
    default:
      // No more instructions to gather information about
      IsReportGenCompleted = true;
    }
  }
  if (IS.bad()) {
    ErrCode = SPIRVEC_InvalidModule;
    return {};
  }
  if (!IsMemoryModelDefined) {
    ErrCode = SPIRVEC_UnspecifiedMemoryModel;
    return {};
  }
  ErrCode = SPIRVEC_Success;
  return std::make_optional(std::move(Report));
}

constexpr std::string_view formatAddressingModel(uint32_t AddrModel) {
  switch (AddrModel) {
  case AddressingModelLogical:
    return "Logical";
  case AddressingModelPhysical32:
    return "Physical32";
  case AddressingModelPhysical64:
    return "Physical64";
  case AddressingModelPhysicalStorageBuffer64:
    return "PhysicalStorageBuffer64";
  default:
    return "Unknown";
  }
}

constexpr std::string_view formatMemoryModel(uint32_t MemoryModel) {
  switch (MemoryModel) {
  case MemoryModelSimple:
    return "Simple";
  case MemoryModelGLSL450:
    return "GLSL450";
  case MemoryModelOpenCL:
    return "OpenCL";
  case MemoryModelVulkan:
    return "Vulkan";
  default:
    return "Unknown";
  }
}

SPIRVModuleTextReport formatSpirvReport(const SPIRVModuleReport &Report) {
  SPIRVModuleTextReport TextReport;
  TextReport.Version =
      formatVersionNumber(static_cast<uint32_t>(Report.Version));
  TextReport.AddrModel = formatAddressingModel(Report.AddrModel);
  TextReport.MemoryModel = formatMemoryModel(Report.MemoryModel);
  // format capability codes as strings
  std::string Name;
  for (auto Capability : Report.Capabilities) {
    bool Found = SPIRVCapabilityNameMap::find(
        static_cast<SPIRVCapabilityKind>(Capability), &Name);
    TextReport.Capabilities.push_back(Found ? Name : "Unknown");
  }
  // other fields with string content can be copied as is
  TextReport.Extensions = Report.Extensions;
  TextReport.ExtendedInstructionSets = Report.ExtendedInstructionSets;
  return TextReport;
}

std::unique_ptr<SPIRVModule> readSpirvModule(std::istream &IS,
                                             const SPIRV::TranslatorOpts &Opts,
                                             std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule(Opts));

  IS >> *BM;
  if (!BM->isModuleValid()) {
    BM->getError(ErrMsg);
    return nullptr;
  }
  return BM;
}

std::unique_ptr<SPIRVModule> readSpirvModule(std::istream &IS,
                                             std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  return readSpirvModule(IS, DefaultOpts, ErrMsg);
}

} // namespace SPIRV

std::unique_ptr<Module>
llvm::convertSpirvToLLVM(LLVMContext &C, SPIRVModule &BM,
                         const SPIRV::TranslatorOpts &Opts,
                         std::string &ErrMsg) {
  std::unique_ptr<Module> M(new Module("", C));

  SPIRVToLLVM BTL(M.get(), &BM);

  if (!BTL.translate()) {
    BM.getError(ErrMsg);
    return nullptr;
  }

  llvm::ModulePassManager PassMgr;
  addSPIRVBIsLoweringPass(PassMgr, Opts.getDesiredBIsRepresentation());
  llvm::ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  PassMgr.run(*M, MAM);

  return M;
}

std::unique_ptr<Module>
llvm::convertSpirvToLLVM(LLVMContext &C, SPIRVModule &BM, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  return llvm::convertSpirvToLLVM(C, BM, DefaultOpts, ErrMsg);
}

bool llvm::readSpirv(LLVMContext &C, std::istream &IS, Module *&M,
                     std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
  // As it is stated in the documentation, the translator accepts all SPIR-V
  // extensions by default
  DefaultOpts.enableAllExtensions();
  return llvm::readSpirv(C, DefaultOpts, IS, M, ErrMsg);
}

bool llvm::readSpirv(LLVMContext &C, const SPIRV::TranslatorOpts &Opts,
                     std::istream &IS, Module *&M, std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(readSpirvModule(IS, Opts, ErrMsg));

  if (!BM)
    return false;

  M = convertSpirvToLLVM(C, *BM, Opts, ErrMsg).release();

  if (!M)
    return false;

  if (DbgSaveTmpLLVM)
    dumpLLVM(M, DbgTmpLLVMFileName);

  return true;
}

bool llvm::getSpecConstInfo(std::istream &IS,
                            std::vector<SpecConstInfoTy> &SpecConstInfo) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  BM->setAutoAddExtensions(false);
  SPIRVDecoder D(IS, *BM);
  SPIRVWord Magic;
  D >> Magic;
  if (!BM->getErrorLog().checkError(Magic == MagicNumber, SPIRVEC_InvalidModule,
                                    "invalid magic number")) {
    return false;
  }
  // Skip the rest of the header
  D.ignore(4);

  // According to the logical layout of SPIRV module (p2.4 of the spec),
  // all constant instructions must appear before function declarations.
  while (D.OpCode != OpFunction && D.getWordCountAndOpCode()) {
    switch (D.OpCode) {
    case OpDecorate:
      // The decoration is added to the module in scope of SPIRVDecorate::decode
      D.getEntry();
      break;
    case OpTypeBool:
    case OpTypeInt:
    case OpTypeFloat:
      BM->addEntry(D.getEntry());
      break;
    case OpSpecConstant:
    case OpSpecConstantTrue:
    case OpSpecConstantFalse: {
      auto *C = BM->addConstant(static_cast<SPIRVValue *>(D.getEntry()));
      SPIRVWord SpecConstIdLiteral = 0;
      if (C->hasDecorate(DecorationSpecId, 0, &SpecConstIdLiteral)) {
        SPIRVType *Ty = C->getType();
        uint32_t SpecConstSize = Ty->isTypeBool() ? 1 : Ty->getBitWidth() / 8;
        std::string TypeString = "";
        if (Ty->isTypeBool()) {
          TypeString = "i1";
        } else if (Ty->isTypeInt()) {
          switch (SpecConstSize) {
          case 1:
            TypeString = "i8";
            break;
          case 2:
            TypeString = "i16";
            break;
          case 4:
            TypeString = "i32";
            break;
          case 8:
            TypeString = "i64";
            break;
          }
        } else if (Ty->isTypeFloat()) {
          switch (SpecConstSize) {
          case 2:
            TypeString = "f16";
            break;
          case 4:
            TypeString = "f32";
            break;
          case 8:
            TypeString = "f64";
            break;
          }
        }
        if (TypeString == "")
          return false;

        SpecConstInfo.emplace_back(
            SpecConstInfoTy({SpecConstIdLiteral, SpecConstSize, TypeString}));
      }
      break;
    }
    default:
      D.ignoreInstruction();
    }
  }
  return !IS.bad();
}

// clang-format off
const StringSet<> SPIRVToLLVM::BuiltInConstFunc {
  "convert", "get_work_dim", "get_global_size", "sub_group_ballot_bit_count",
  "get_global_id", "get_local_size", "get_local_id", "get_num_groups",
  "get_group_id", "get_global_offset", "acos", "acosh", "acospi",
  "asin", "asinh", "asinpi", "atan", "atan2", "atanh", "atanpi",
  "atan2pi", "cbrt", "ceil", "copysign", "cos", "cosh", "cospi",
  "erfc", "erf", "exp", "exp2", "exp10", "expm1", "fabs", "fdim",
  "floor", "fma", "fmax", "fmin", "fmod", "ilogb", "ldexp", "lgamma",
  "log", "log2", "log10", "log1p", "logb", "mad", "maxmag", "minmag",
  "nan", "nextafter", "pow", "pown", "powr", "remainder", "rint",
  "rootn", "round", "rsqrt", "sin", "sinh", "sinpi", "sqrt", "tan",
  "tanh", "tanpi", "tgamma", "trunc", "half_cos", "half_divide", "half_exp",
  "half_exp2", "half_exp10", "half_log", "half_log2", "half_log10", "half_powr",
  "half_recip", "half_rsqrt", "half_sin", "half_sqrt", "half_tan", "native_cos",
  "native_divide", "native_exp", "native_exp2", "native_exp10", "native_log",
  "native_log2", "native_log10", "native_powr", "native_recip", "native_rsqrt",
  "native_sin", "native_sqrt", "native_tan", "abs", "abs_diff", "add_sat", "hadd",
  "rhadd", "clamp", "clz", "mad_hi", "mad_sat", "max", "min", "mul_hi", "rotate",
  "sub_sat", "upsample", "popcount", "mad24", "mul24", "degrees", "mix", "radians",
  "step", "smoothstep", "sign", "cross", "dot", "distance", "length", "normalize",
  "fast_distance", "fast_length", "fast_normalize", "isequal", "isnotequal",
  "isgreater", "isgreaterequal", "isless", "islessequal", "islessgreater",
  "isfinite", "isinf", "isnan", "isnormal", "isordered", "isunordered", "signbit",
  "any", "all", "bitselect", "select", "shuffle", "shuffle2", "get_image_width",
  "get_image_height", "get_image_depth", "get_image_channel_data_type",
  "get_image_channel_order", "get_image_dim", "get_image_array_size",
  "get_image_array_size", "sub_group_inverse_ballot", "sub_group_ballot_bit_extract",
};
// clang-format on
