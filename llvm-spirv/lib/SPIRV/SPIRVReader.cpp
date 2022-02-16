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
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "VectorComputeUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
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

using namespace std;
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
  bool DecorationFound = false;
  BF->foreachArgument([&](SPIRVFunctionParameter *Arg) {
    if (Arg->getType()->isTypePointer() &&
        Arg->hasDecorate(internal::DecorationRuntimeAlignedINTEL)) {
      DecorationFound = true;
      ValueVec.push_back(ForeachFnArg(Arg));
    } else {
      llvm::Metadata *DefaultNode = ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt1Ty(*Context), 0));
      ValueVec.push_back(DefaultNode);
    }
  });
  if (DecorationFound)
    Fn->setMetadata("kernel_arg_runtime_aligned",
                    MDNode::get(*Context, ValueVec));
}

Value *SPIRVToLLVM::getTranslatedValue(SPIRVValue *BV) {
  auto Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
}

static llvm::Optional<llvm::Attribute>
translateSEVMetadata(SPIRVValue *BV, llvm::LLVMContext &Context) {
  llvm::Optional<llvm::Attribute> RetAttr;

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
  auto II = dyn_cast<IntrinsicInst>(I);
  if (II && II->getIntrinsicID() == Intrinsic::lifetime_start)
    return II;
  // Bitcast might be inserted during translation of OpLifetimeStart
  auto BC = dyn_cast<BitCastInst>(I);
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
  auto Ty =
      Arg->isByVal() ? Arg->getType()->getPointerElementType() : Arg->getType();
  return MDString::get(*Context, transTypeToOCLTypeName(Ty, !Arg->isZext()));
}

Value *SPIRVToLLVM::mapFunction(SPIRVFunction *BF, Function *F) {
  SPIRVDBG(spvdbgs() << "[mapFunction] " << *BF << " -> ";
           dbgs() << *F << '\n';)
  FuncMap[BF] = F;
  return F;
}

Type *SPIRVToLLVM::transFPType(SPIRVType *T) {
  switch (T->getFloatBitWidth()) {
  case 16:
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

std::string SPIRVToLLVM::transOCLImageTypeName(SPIRV::SPIRVTypeImage *ST) {
  return getSPIRVTypeName(
      kSPIRVTypeName::Image,
      getSPIRVImageTypePostfixes(
          getSPIRVImageSampledTypeName(ST->getSampledType()),
          ST->getDescriptor(),
          ST->hasAccessQualifier() ? ST->getAccessQualifier()
                                   : AccessQualifierReadOnly));
}

std::string
SPIRVToLLVM::transOCLSampledImageTypeName(SPIRV::SPIRVTypeSampledImage *ST) {
  return getSPIRVTypeName(
      kSPIRVTypeName::SampledImg,
      getSPIRVImageTypePostfixes(
          getSPIRVImageSampledTypeName(ST->getImageType()->getSampledType()),
          ST->getImageType()->getDescriptor(),
          ST->getImageType()->hasAccessQualifier()
              ? ST->getImageType()->getAccessQualifier()
              : AccessQualifierReadOnly));
}

std::string
SPIRVToLLVM::transVMEImageTypeName(SPIRV::SPIRVTypeVmeImageINTEL *VT) {
  return getSPIRVTypeName(
      kSPIRVTypeName::VmeImageINTEL,
      getSPIRVImageTypePostfixes(
          getSPIRVImageSampledTypeName(VT->getImageType()->getSampledType()),
          VT->getImageType()->getDescriptor(),
          VT->getImageType()->hasAccessQualifier()
              ? VT->getImageType()->getAccessQualifier()
              : AccessQualifierReadOnly));
}

std::string SPIRVToLLVM::transPipeTypeName(SPIRV::SPIRVTypePipe *PT) {
  SPIRVAccessQualifierKind PipeAccess = PT->getAccessQualifier();

  assert((PipeAccess == AccessQualifierReadOnly ||
          PipeAccess == AccessQualifierWriteOnly) &&
         "Invalid access qualifier");

  return std::string(kSPIRVTypeName::PrefixAndDelim) + kSPIRVTypeName::Pipe +
         kSPIRVTypeName::Delimiter + kSPIRVTypeName::PostfixDelim + PipeAccess;
}

std::string
SPIRVToLLVM::transOCLPipeStorageTypeName(SPIRV::SPIRVTypePipeStorage *PST) {
  return std::string(kSPIRVTypeName::PrefixAndDelim) +
         kSPIRVTypeName::PipeStorage;
}

std::string SPIRVToLLVM::transVCTypeName(SPIRVTypeBufferSurfaceINTEL *PST) {
  if (PST->hasAccessQualifier())
    return VectorComputeUtil::getVCBufferSurfaceName(PST->getAccessQualifier());
  return VectorComputeUtil::getVCBufferSurfaceName();
}

Type *SPIRVToLLVM::transType(SPIRVType *T, bool IsClassMember) {
  auto Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
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
  case OpTypePointer:
    return mapType(
        T, PointerType::get(
               transType(T->getPointerElementType(), IsClassMember),
               SPIRSPIRVAddrSpaceMap::rmap(T->getPointerStorageClass())));
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
    auto FT = static_cast<SPIRVTypeFunction *>(T);
    auto RT = transType(FT->getReturnType());
    std::vector<Type *> PT;
    for (size_t I = 0, E = FT->getNumParameters(); I != E; ++I)
      PT.push_back(transType(FT->getParameterType(I)));
    return mapType(T, FunctionType::get(RT, PT, false));
  }
  case OpTypeImage: {
    auto ST = static_cast<SPIRVTypeImage *>(T);
    if (ST->isOCLImage())
      return mapType(T, getOrCreateOpaquePtrType(M, transOCLImageTypeName(ST)));
    else
      llvm_unreachable("Unsupported image type");
    return nullptr;
  }
  case OpTypeSampledImage: {
    auto ST = static_cast<SPIRVTypeSampledImage *>(T);
    return mapType(
        T, getOrCreateOpaquePtrType(M, transOCLSampledImageTypeName(ST)));
  }
  case OpTypeStruct: {
    auto ST = static_cast<SPIRVTypeStruct *>(T);
    auto Name = ST->getName();
    if (!Name.empty()) {
      if (auto OldST = StructType::getTypeByName(*Context, Name))
        OldST->setName("");
    } else {
      Name = "structtype";
    }
    auto *StructTy = StructType::create(*Context, Name);
    mapType(ST, StructTy);
    SmallVector<Type *, 4> MT;
    for (size_t I = 0, E = ST->getMemberCount(); I != E; ++I)
      MT.push_back(transType(ST->getMemberType(I), true));
    for (auto &CI : ST->getContinuedInstructions())
      for (size_t I = 0, E = CI->getNumElements(); I != E; ++I)
        MT.push_back(transType(CI->getMemberType(I), true));
    StructTy->setBody(MT, ST->isPacked());
    return StructTy;
  }
  case OpTypePipe: {
    auto PT = static_cast<SPIRVTypePipe *>(T);
    return mapType(
        T, getOrCreateOpaquePtrType(M, transPipeTypeName(PT),
                                    getOCLOpaqueTypeAddrSpace(T->getOpCode())));
  }
  case OpTypePipeStorage: {
    auto PST = static_cast<SPIRVTypePipeStorage *>(T);
    return mapType(
        T, getOrCreateOpaquePtrType(M, transOCLPipeStorageTypeName(PST),
                                    getOCLOpaqueTypeAddrSpace(T->getOpCode())));
  }
  case OpTypeVmeImageINTEL: {
    auto *VT = static_cast<SPIRVTypeVmeImageINTEL *>(T);
    return mapType(T, getOrCreateOpaquePtrType(M, transVMEImageTypeName(VT)));
  }
  case OpTypeBufferSurfaceINTEL: {
    auto PST = static_cast<SPIRVTypeBufferSurfaceINTEL *>(T);
    return mapType(T,
                   getOrCreateOpaquePtrType(M, transVCTypeName(PST),
                                            SPIRAddressSpace::SPIRAS_Global));
  }

  case internal::OpTypeJointMatrixINTEL: {
    auto *MT = static_cast<SPIRVTypeJointMatrixINTEL *>(T);
    auto R = static_cast<SPIRVConstant *>(MT->getRows())->getZExtIntValue();
    auto C = static_cast<SPIRVConstant *>(MT->getColumns())->getZExtIntValue();
    std::stringstream SS;
    SS << kSPIRVTypeName::PostfixDelim;
    SS << transTypeToOCLTypeName(MT->getCompType());
    auto L = static_cast<SPIRVConstant *>(MT->getLayout())->getZExtIntValue();
    auto S = static_cast<SPIRVConstant *>(MT->getScope())->getZExtIntValue();
    SS << kSPIRVTypeName::PostfixDelim << R << kSPIRVTypeName::PostfixDelim << C
       << kSPIRVTypeName::PostfixDelim << L << kSPIRVTypeName::PostfixDelim
       << S;
    std::string Name =
        getSPIRVTypeName(kSPIRVTypeName::JointMatrixINTEL, SS.str());
    return mapType(T, getOrCreateOpaquePtrType(M, Name));
  }

  default: {
    auto OC = T->getOpCode();
    if (isOpaqueGenericTypeOpCode(OC) || isSubgroupAvcINTELTypeOpCode(OC))
      return mapType(T, getSPIRVOpaquePtrType(M, OC));
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
SPIRVToLLVM::transTypeVector(const std::vector<SPIRVType *> &BT) {
  std::vector<Type *> T;
  for (auto I : BT)
    T.push_back(transType(I));
  return T;
}

std::vector<Value *>
SPIRVToLLVM::transValue(const std::vector<SPIRVValue *> &BV, Function *F,
                        BasicBlock *BB) {
  std::vector<Value *> V;
  for (auto I : BV)
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

  auto Temp = MDNode::getTemporary(*Context, None);
  auto Self = MDNode::get(*Context, Temp.get());
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
  if (LC & LoopControlUnrollMask)
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
    // If unroll factor is set as '1' - disable loop unrolling
    if (1 == LoopControlParameters[NumParam])
      Metadata.push_back(getMetadataFromName("llvm.loop.unroll.disable"));
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
    std::map<Value *, unsigned> PointerSflnMap;
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
    std::map<Value *, std::vector<GetElementPtrInst *>> ArrayGEPMap;
    for (const auto &BB : LoopObj->blocks()) {
      for (Instruction &I : *BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP)
          continue;

        Value *AccessedPointer = GEP->getPointerOperand();
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
          CurrentDepthIdxGroup = llvm::MDNode::getDistinct(*Context, None);
          OffsetIdxGroupMap.emplace(Info, CurrentDepthIdxGroup);
        } else {
          // Previous accesses to that field have already been indexed,
          // just use the already-existing metadata.
          CurrentDepthIdxGroup = OffsetIdxGroupIt->second;
        }
      } else /* Regular kernel-scope array/pointer variable */ {
        // Emit a distinct index group that will be referenced from
        // llvm.loop.parallel_access_indices metadata
        CurrentDepthIdxGroup = llvm::MDNode::getDistinct(*Context, None);
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
  if (LC & spv::internal::LoopControlLoopCountINTELMask) {
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

  auto V = transValueWithoutDecoration(BV, F, BB, CreatePlaceHolder);
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
  auto Src = transValue(BC->getOperand(0), F, BB, BB ? true : false);
  auto Dst = transType(BC->getType());
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
  return Inst;
}

Type *SPIRVToLLVM::mapType(SPIRVType *BT, Type *T) {
  SPIRVDBG(dbgs() << *T << '\n';)
  TypeMap[BT] = T;
  return T;
}

Value *SPIRVToLLVM::mapValue(SPIRVValue *BV, Value *V) {
  auto Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end()) {
    if (Loc->second == V)
      return V;
    auto LD = dyn_cast<LoadInst>(Loc->second);
    auto Placeholder = dyn_cast<GlobalVariable>(LD->getPointerOperand());
    assert(LD && Placeholder &&
           Placeholder->getName().startswith(KPlaceholderPrefix) &&
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
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  if (!CI->getOperand(0)->getType()->isVectorTy() &&
      CI->getOperand(1)->getType()->isVectorTy()) {
    return mutateCallInstOCL(
        M, CI,
        [=](CallInst *, std::vector<Value *> &Args) {
          auto VecElemCount =
              cast<VectorType>(CI->getOperand(1)->getType())->getElementCount();
          Value *NewVec = nullptr;
          if (auto CA = dyn_cast<Constant>(Args[0]))
            NewVec = ConstantVector::getSplat(VecElemCount, CA);
          else {
            NewVec = ConstantVector::getSplat(
                VecElemCount, Constant::getNullValue(Args[0]->getType()));
            NewVec = InsertElementInst::Create(NewVec, Args[0], getInt32(M, 0),
                                               "", CI);
            NewVec = new ShuffleVectorInst(
                NewVec, NewVec,
                ConstantVector::getSplat(VecElemCount, getInt32(M, 0)), "", CI);
          }
          NewVec->takeName(Args[0]);
          Args[0] = NewVec;
          return FuncName;
        },
        &Attrs);
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
  auto *SamplerT = getSPIRVOpaquePtrType(M, OpTypeSampler);
  auto *I32Ty = IntegerType::getInt32Ty(*Context);
  auto *FTy = FunctionType::get(SamplerT, {I32Ty}, false);

  FunctionCallee Func = M->getOrInsertFunction(SAMPLER_INIT, FTy);

  auto Lit = (BCS->getAddrMode() << 1) | BCS->getNormalized() |
             ((BCS->getFilterMode() + 1) << 4);

  return CallInst::Create(Func, {ConstantInt::get(I32Ty, Lit)}, "", BB);
}

Value *SPIRVToLLVM::oclTransConstantPipeStorage(
    SPIRV::SPIRVConstantPipeStorage *BCPS) {

  string CPSName = string(kSPIRVTypeName::PrefixAndDelim) +
                   kSPIRVTypeName::ConstantPipeStorage;

  auto Int32Ty = IntegerType::getInt32Ty(*Context);
  auto CPSTy = StructType::getTypeByName(*Context, CPSName);
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

// A pointer annotation may have been generated for the operand. If the operand
// is used further in IR, it should be replaced with the intrinsic call result.
// Otherwise, the generated pointer annotation call is left unused.
static void replaceOperandWithAnnotationIntrinsicCallResult(Value *&V) {
  if (Use *SingleUse = V->getSingleUndroppableUse()) {
    if (auto *II = dyn_cast<IntrinsicInst>(SingleUse->getUser())) {
      if (II->getIntrinsicID() == Intrinsic::ptr_annotation &&
          II->getType() == V->getType())
        // Overwrite the future operand with the intrinsic call result.
        V = II;
    }
  }
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

void transFunctionPointerCallArgumentAttributes(SPIRVValue *BV, CallInst *CI) {
  std::vector<SPIRVDecorate const *> ArgumentAttributes =
      BV->getDecorations(internal::DecorationArgumentAttributeINTEL);

  for (const auto *Dec : ArgumentAttributes) {
    std::vector<SPIRVWord> Literals = Dec->getVecLiteral();
    SPIRVWord ArgNo = Literals[0];
    SPIRVWord SpirvAttr = Literals[1];
    Attribute::AttrKind LlvmAttrKind = SPIRSPIRVFuncParamAttrMap::rmap(
        static_cast<SPIRVFuncParamAttrKind>(SpirvAttr));
    auto LlvmAttr =
        Attribute::isTypeAttrKind(LlvmAttrKind)
            ? Attribute::get(CI->getContext(), LlvmAttrKind,
                             cast<PointerType>(CI->getOperand(ArgNo)->getType())
                                 ->getPointerElementType())
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
        FS = &APFloat::IEEEhalf();
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
    auto LT = transType(BV->getType());
    return mapValue(BV, Constant::getNullValue(LT));
  }

  case OpConstantComposite:
  case OpSpecConstantComposite: {
    auto BCC = static_cast<SPIRVConstantComposite *>(BV);
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
    case OpTypeArray:
      return mapValue(
          BV, ConstantArray::get(dyn_cast<ArrayType>(transType(BCC->getType())),
                                 CV));
    case OpTypeStruct: {
      auto BCCTy = dyn_cast<StructType>(transType(BCC->getType()));
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

          CV[I] = ConstantExpr::getBitCast(CV[I], BCCTy->getElementType(I));
        }
      }

      return mapValue(BV,
                      ConstantStruct::get(
                          dyn_cast<StructType>(transType(BCC->getType())), CV));
    }
    default:
      llvm_unreachable("not implemented");
      return nullptr;
    }
  }

  case OpConstantSampler: {
    auto BCS = static_cast<SPIRVConstantSampler *>(BV);
    // Intentially do not map this value. We want to generate constant
    // sampler initializer every time constant sampler is used, otherwise
    // initializer may not dominate all its uses.
    return oclTransConstantSampler(BCS, BB);
  }

  case OpConstantPipeStorage: {
    auto BCPS = static_cast<SPIRVConstantPipeStorage *>(BV);
    return mapValue(BV, oclTransConstantPipeStorage(BCPS));
  }

  case OpSpecConstantOp: {
    auto BI =
        createInstFromSpecConstantOp(static_cast<SPIRVSpecConstantOp *>(BV));
    return mapValue(BV, transValue(BI, nullptr, nullptr, false));
  }

  case OpConstantFunctionPointerINTEL: {
    SPIRVConstantFunctionPointerINTEL *BC =
        static_cast<SPIRVConstantFunctionPointerINTEL *>(BV);
    SPIRVFunction *F = BC->getFunction();
    BV->setName(F->getName());
    return mapValue(BV, transFunction(F));
  }

  case OpUndef:
    return mapValue(BV, UndefValue::get(transType(BV->getType())));

  case OpVariable: {
    auto BVar = static_cast<SPIRVVariable *>(BV);
    auto Ty = transType(BVar->getType()->getPointerElementType());
    bool IsConst = BVar->isConstant();
    llvm::GlobalValue::LinkageTypes LinkageTy = transLinkageType(BVar);
    SPIRVStorageClassKind BS = BVar->getStorageClass();
    SPIRVValue *Init = BVar->getInitializer();

    if (isSPIRVSamplerType(Ty) && BS == StorageClassUniformConstant) {
      // Skip generating llvm code during translation of a variable definition,
      // generate code only for its uses
      if (!BB)
        return nullptr;

      assert(Init && "UniformConstant OpVariable with sampler type must have "
                     "an initializer!");
      return transValue(Init, F, BB);
    }

    if (BS == StorageClassFunction && !Init) {
      assert(BB && "Invalid BB");
      return mapValue(BV, new AllocaInst(Ty, 0, BV->getName(), BB));
    }

    SPIRAddressSpace AddrSpace;
    bool IsVectorCompute =
        BVar->hasDecorate(DecorationVectorComputeVariableINTEL);
    Constant *Initializer = nullptr;
    if (IsVectorCompute) {
      AddrSpace = VectorComputeUtil::getVCGlobalVarAddressSpace(BS);
      Initializer = UndefValue::get(Ty);
    } else
      AddrSpace = SPIRSPIRVAddrSpaceMap::rmap(BS);
    // Force SPIRV BuiltIn variable's name to be __spirv_BuiltInXXXX.
    // No matter what BV's linkage name is.
    SPIRVBuiltinVariableKind BVKind;
    if (BVar->isBuiltin(&BVKind))
      BV->setName(prefixSPIRVName(SPIRVBuiltInNameMap::map(BVKind)));
    auto LVar = new GlobalVariable(*M, Ty, IsConst, LinkageTy,
                                   /*Initializer=*/nullptr, BV->getName(), 0,
                                   GlobalVariable::NotThreadLocal, AddrSpace);
    auto Res = mapValue(BV, LVar);
    if (Init)
      Initializer = dyn_cast<Constant>(transValue(Init, F, BB, false));
    else if (LinkageTy == GlobalValue::CommonLinkage)
      // In LLVM, variables with common linkage type must be initialized to 0.
      Initializer = Constant::getNullValue(Ty);
    else if (BS == SPIRVStorageClassKind::StorageClassWorkgroup)
      Initializer = dyn_cast<Constant>(UndefValue::get(Ty));
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
        LVar->addAttribute(SEVAttr.getValue().getKindAsString(),
                           SEVAttr.getValue().getValueAsString());
    }

    return Res;
  }

  case OpFunctionParameter: {
    auto BA = static_cast<SPIRVFunctionParameter *>(BV);
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
    auto Ty = transType(BV->getType());
    auto GV = new GlobalVariable(
        *M, Ty, false, GlobalValue::PrivateLinkage,
        nullptr, std::string(KPlaceholderPrefix) + BV->getName(), 0,
        GlobalVariable::NotThreadLocal, 0);
    auto LD = new LoadInst(Ty, GV, BV->getName(), BB);
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
    return mapValue(
        BV, new AllocaInst(Ty, SPIRAS_Private, ArrSize, BV->getName(), BB));
  }

  case OpRestoreMemoryINTEL: {
    auto *Restore = static_cast<SPIRVRestoreMemoryINTEL *>(BV);
    llvm::Value *Ptr = transValue(Restore->getOperand(0), F, BB);
    Function *StackRestore =
        Intrinsic::getDeclaration(M, Intrinsic::stackrestore);
    return mapValue(BV, CallInst::Create(StackRestore, {Ptr}, "", BB));
  }

  case OpSaveMemoryINTEL: {
    Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
    return mapValue(BV, CallInst::Create(StackSave, "", BB));
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
    auto Phi = static_cast<SPIRVPhi *>(BV);
    auto LPhi = dyn_cast<PHINode>(mapValue(
        BV, PHINode::Create(transType(Phi->getType()),
                            Phi->getPairs().size() / 2, Phi->getName(), BB)));
    Phi->foreachPair([&](SPIRVValue *IncomingV, SPIRVBasicBlock *IncomingBB,
                         size_t Index) {
      auto Translated = transValue(IncomingV, F, BB);
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
    auto RV = static_cast<SPIRVReturnValue *>(BV);
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
    auto Var = transValue(LTStop->getObject(), F, BB);
    for (const auto &I : Var->users())
      if (auto II = getLifetimeStartIntrinsic(dyn_cast<Instruction>(I)))
        return mapValue(BV, Builder.CreateLifetimeEnd(II->getOperand(1), S));
    return mapValue(BV, Builder.CreateLifetimeEnd(Var, S));
  }

  case OpStore: {
    SPIRVStore *BS = static_cast<SPIRVStore *>(BV);
    StoreInst *SI = nullptr;
    auto *Src = transValue(BS->getSrc(), F, BB);
    auto *Dst = transValue(BS->getDst(), F, BB);
    // A ptr.annotation may have been generated for the source variable.
    replaceOperandWithAnnotationIntrinsicCallResult(Src);
    // A ptr.annotation may have been generated for the destination variable.
    replaceOperandWithAnnotationIntrinsicCallResult(Dst);

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
    // A ptr.annotation may have been generated for the source variable.
    replaceOperandWithAnnotationIntrinsicCallResult(V);

    Type *Ty = V->getType()->getPointerElementType();
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

  case OpCopyMemorySized: {
    SPIRVCopyMemorySized *BC = static_cast<SPIRVCopyMemorySized *>(BV);
    CallInst *CI = nullptr;
    llvm::Value *Dst = transValue(BC->getTarget(), F, BB);
    MaybeAlign Align(BC->getAlignment());
    llvm::Value *Size = transValue(BC->getSize(), F, BB);
    bool IsVolatile = BC->SPIRVMemoryAccess::isVolatile();
    IRBuilder<> Builder(BB);

    // If we copy from zero-initialized array, we can optimize it to llvm.memset
    if (BC->getSource()->getOpCode() == OpBitcast) {
      SPIRVValue *Source =
          static_cast<SPIRVBitcast *>(BC->getSource())->getOperand(0);
      if (Source->isVariable()) {
        auto *Init = static_cast<SPIRVVariable *>(Source)->getInitializer();
        if (isa<OpConstantNull>(Init)) {
          SPIRVType *Ty = static_cast<SPIRVConstantNull *>(Init)->getType();
          if (isa<OpTypeArray>(Ty)) {
            Type *Int8Ty = Type::getInt8Ty(Dst->getContext());
            llvm::Value *Src = ConstantInt::get(Int8Ty, 0);
            llvm::Value *NewDst = Dst;
            if (!Dst->getType()->getPointerElementType()->isIntegerTy(8)) {
              Type *Int8PointerTy = Type::getInt8PtrTy(
                  Dst->getContext(), Dst->getType()->getPointerAddressSpace());
              NewDst = llvm::BitCastInst::CreatePointerCast(Dst, Int8PointerTy,
                                                            "", BB);
            }
            CI = Builder.CreateMemSet(NewDst, Src, Size, Align, IsVolatile);
          }
        }
      }
    }
    if (!CI) {
      llvm::Value *Src = transValue(BC->getSource(), F, BB);
      CI = Builder.CreateMemCpy(Dst, Align, Src, Align, Size, IsVolatile);
    }
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
    auto BS = static_cast<SPIRVSwitch *>(BV);
    auto Select = transValue(BS->getSelect(), F, BB);
    auto LS = SwitchInst::Create(
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
    auto VTS = static_cast<SPIRVVectorTimesScalar *>(BV);
    IRBuilder<> Builder(BB);
    auto Scalar = transValue(VTS->getScalar(), F, BB);
    auto Vector = transValue(VTS->getVector(), F, BB);
    auto *VecTy = cast<FixedVectorType>(Vector->getType());
    unsigned VecSize = VecTy->getNumElements();
    auto NewVec = Builder.CreateVectorSplat(VecSize, Scalar, Scalar->getName());
    NewVec->takeName(Scalar);
    auto Scale = Builder.CreateFMul(Vector, NewVec, "scale");
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
    auto ETy = VTy->getElementType();
    unsigned N = VecTy->getNumElements();
    Value *V = Builder.CreateVectorSplat(M, ConstantFP::get(ETy, 0.0));

    for (unsigned Idx = 0; Idx != N; ++Idx) {
      Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(Idx));
      Value *Lhs = Builder.CreateVectorSplat(M, S);
      Value *Rhs = UndefValue::get(VTy);
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
    auto MTS = static_cast<SPIRVMatrixTimesScalar *>(BV);
    IRBuilder<> Builder(BB);
    auto Scalar = transValue(MTS->getScalar(), F, BB);
    auto Matrix = transValue(MTS->getMatrix(), F, BB);
    uint64_t ColNum = Matrix->getType()->getArrayNumElements();
    auto ColType = cast<ArrayType>(Matrix->getType())->getElementType();
    auto VecSize = cast<FixedVectorType>(ColType)->getNumElements();
    auto NewVec = Builder.CreateVectorSplat(VecSize, Scalar, Scalar->getName());
    NewVec->takeName(Scalar);

    Value *V = UndefValue::get(Matrix->getType());
    for (uint64_t Idx = 0; Idx != ColNum; Idx++) {
      auto Col = Builder.CreateExtractValue(Matrix, Idx);
      auto I = Builder.CreateFMul(Col, NewVec);
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
    auto ETy = VTy->getElementType();
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
    auto ETy = V1Ty->getElementType();

    (void)C1;
    assert(C1 == R2 && "Unmatched matrix");

    auto VTy = FixedVectorType::get(ETy, R1);
    auto ResultTy = ArrayType::get(VTy, C2);

    Value *Res = UndefValue::get(ResultTy);

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
    auto TR = static_cast<SPIRVTranspose *>(BV);
    IRBuilder<> Builder(BB);
    auto Matrix = transValue(TR->getMatrix(), F, BB);
    unsigned ColNum = Matrix->getType()->getArrayNumElements();
    FixedVectorType *ColTy = cast<FixedVectorType>(
        cast<ArrayType>(Matrix->getType())->getElementType());
    unsigned RowNum = ColTy->getNumElements();

    auto VTy = FixedVectorType::get(ColTy->getElementType(), ColNum);
    auto ResultTy = ArrayType::get(VTy, RowNum);
    Value *V = UndefValue::get(ResultTy);

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
          Value *V1 = Builder.CreateShuffleVector(
              MCache[0], MCache[1], ArrayRef<int>({Idx, Idx + 4}));
          Value *V2 = Builder.CreateShuffleVector(
              MCache[2], MCache[3], ArrayRef<int>({Idx, Idx + 4}));
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
      Value *Vec = UndefValue::get(VTy);

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
    auto Ty = transType(CO->getOperand()->getType());
    AllocaInst *AI =
        new AllocaInst(Ty, 0, "", BB);
    new StoreInst(transValue(CO->getOperand(), F, BB), AI, BB);
    LoadInst *LI = new LoadInst(Ty, AI, "", BB);
    return mapValue(BV, LI);
  }

  case OpAccessChain:
  case OpInBoundsAccessChain:
  case OpPtrAccessChain:
  case OpInBoundsPtrAccessChain: {
    auto AC = static_cast<SPIRVAccessChainBase *>(BV);
    auto Base = transValue(AC->getBase(), F, BB);
    Type *BaseTy = cast<PointerType>(Base->getType())->getPointerElementType();
    auto Index = transValue(AC->getIndices(), F, BB);
    if (!AC->hasPtrIndex())
      Index.insert(Index.begin(), getInt32(M, 0));
    auto IsInbound = AC->isInBounds();
    Value *V = nullptr;
    if (BB) {
      auto GEP =
          GetElementPtrInst::Create(BaseTy, Base, Index, BV->getName(), BB);
      GEP->setIsInBounds(IsInbound);
      V = GEP;
    } else {
      V = ConstantExpr::getGetElementPtr(BaseTy, dyn_cast<Constant>(Base),
                                         Index, IsInbound);
    }
    return mapValue(BV, V);
  }

  case OpCompositeConstruct: {
    auto CC = static_cast<SPIRVCompositeConstruct *>(BV);
    auto Constituents = transValue(CC->getOperands(), F, BB);
    std::vector<Constant *> CV;
    for (const auto &I : Constituents) {
      CV.push_back(dyn_cast<Constant>(I));
    }
    switch (static_cast<size_t>(BV->getType()->getOpCode())) {
    case OpTypeVector:
      return mapValue(BV, ConstantVector::get(CV));
    case OpTypeArray:
      return mapValue(
          BV, ConstantArray::get(dyn_cast<ArrayType>(transType(CC->getType())),
                                 CV));
    case OpTypeStruct:
      return mapValue(BV,
                      ConstantStruct::get(
                          dyn_cast<StructType>(transType(CC->getType())), CV));
    case internal::OpTypeJointMatrixINTEL:
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
    auto CI = static_cast<SPIRVCompositeInsert *>(BV);
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
    auto VS = static_cast<SPIRVVectorShuffle *>(BV);
    std::vector<Constant *> Components;
    IntegerType *Int32Ty = IntegerType::get(*Context, 32);
    for (auto I : VS->getComponents()) {
      if (I == static_cast<SPIRVWord>(-1))
        Components.push_back(UndefValue::get(Int32Ty));
      else
        Components.push_back(ConstantInt::get(Int32Ty, I));
    }
    IRBuilder<> Builder(*Context);
    if (BB) {
      Builder.SetInsertPoint(BB);
    }
    return mapValue(BV, Builder.CreateShuffleVector(
                            transValue(VS->getVector1(), F, BB),
                            transValue(VS->getVector2(), F, BB),
                            ConstantVector::get(Components), BV->getName()));
  }

  case OpBitReverse: {
    auto *BR = static_cast<SPIRVUnary *>(BV);
    auto Ty = transType(BV->getType());
    Function *intr =
        Intrinsic::getDeclaration(M, llvm::Intrinsic::bitreverse, Ty);
    auto *Call = CallInst::Create(intr, transValue(BR->getOperand(0), F, BB),
                                  BR->getName(), BB);
    return mapValue(BV, Call);
  }

  case OpFunctionCall: {
    SPIRVFunctionCall *BC = static_cast<SPIRVFunctionCall *>(BV);
    auto Call = CallInst::Create(transFunction(BC->getFunction()),
                                 transValue(BC->getArgumentValues(), F, BB),
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
    auto V = transValue(BC->getCalledValue(), F, BB);
    auto Call = CallInst::Create(
        cast<FunctionType>(V->getType()->getPointerElementType()), V,
        transValue(BC->getArgumentValues(), F, BB), BC->getName(), BB);
    transFunctionPointerCallArgumentAttributes(BV, Call);
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

  case OpExtInst: {
    auto *ExtInst = static_cast<SPIRVExtInst *>(BV);
    switch (ExtInst->getExtSetKind()) {
    case SPIRVEIS_OpenCL:
      return mapValue(BV, transOCLBuiltinFromExtInst(ExtInst, BB));
    case SPIRVEIS_Debug:
    case SPIRVEIS_OpenCL_DebugInfo_100:
      return mapValue(BV, DbgTran->transDebugIntrinsic(ExtInst, BB));
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
    auto Neg =
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
    auto Dividend = transValue(FMod->getOperand(0), F, BB);
    auto Divisor = transValue(FMod->getOperand(1), F, BB);
    auto FRem = Builder.CreateFRem(Dividend, Divisor, "frem.res");
    auto CopySign = Builder.CreateBinaryIntrinsic(
        llvm::Intrinsic::copysign, FRem, Divisor, nullptr, "copysign.res");
    auto FAdd = Builder.CreateFAdd(FRem, Divisor, "fadd.res");
    auto Cmp = Builder.CreateFCmpONE(FRem, CopySign, "cmp.res");
    auto Select = Builder.CreateSelect(Cmp, FAdd, CopySign);
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
    auto Dividend = transValue(SMod->getOperand(0), F, BB);
    auto Divisor = transValue(SMod->getOperand(1), F, BB);
    auto SRem = Builder.CreateSRem(Dividend, Divisor, "srem.res");
    auto Xor = Builder.CreateXor(Dividend, Divisor, "xor.res");
    auto Zero = ConstantInt::getNullValue(Dividend->getType());
    auto CmpSign = Builder.CreateICmpSLT(Xor, Zero, "cmpsign.res");
    auto CmpSRem = Builder.CreateICmpNE(SRem, Zero, "cmpsrem.res");
    auto Add = Builder.CreateNSWAdd(SRem, Divisor, "add.res");
    auto Cmp = Builder.CreateAnd(CmpSign, CmpSRem, "cmp.res");
    auto Select = Builder.CreateSelect(Cmp, Add, SRem);
    return mapValue(BV, Select);
  }

  case OpFNegate: {
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    auto Neg = UnaryOperator::CreateFNeg(transValue(BC->getOperand(0), F, BB),
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

    PointerType *Int8PtrTyPrivate =
        Type::getInt8PtrTy(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = Type::getInt32Ty(*Context);

    Value *UndefInt8Ptr = UndefValue::get(Int8PtrTyPrivate);
    Value *UndefInt32 = UndefValue::get(Int32Ty);

    Constant *GS = Builder.CreateGlobalStringPtr(kOCLBuiltinName::FPGARegIntel);

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
        if (PtrTy && isa<IntegerType>(PtrTy->getPointerElementType()))
          RetTy = PtrTy;
        // Whether a struct or a pointer to some other type,
        // bitcast to i8*
        else {
          RetTy = Int8PtrTyPrivate;
          ValAsArg = Builder.CreateBitCast(Val, Int8PtrTyPrivate);
        }
        Value *Args[] = {ValAsArg, GS, UndefInt8Ptr, UndefInt32, UndefInt8Ptr};
        auto *IntrinsicCall = Builder.CreateIntrinsic(IID, RetTy, Args);
        return mapValue(BV, IntrinsicCall);
      }
    }

    Value *Args[] = {ValAsArg, GS, UndefInt8Ptr, UndefInt32};
    auto *IntrinsicCall = Builder.CreateIntrinsic(IID, RetTy, Args);
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

  case internal::OpArithmeticFenceINTEL: {
    IRBuilder<> Builder(BB);
    auto *BC = static_cast<SPIRVUnary *>(BV);
    Type *RetTy = transType(BC->getType());
    Value *Val = transValue(BC->getOperand(0), F, BB);
    return mapValue(
        BV, Builder.CreateIntrinsic(Intrinsic::arithmetic_fence, RetTy, Val));
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
      auto BI = static_cast<SPIRVInstruction *>(BV);
      Value *Inst = nullptr;
      if (BI->hasFPRoundingMode() || BI->isSaturatedConversion())
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

  auto Inst = static_cast<SPIRVFixedPointIntelInst *>(BI);
  Type *InTy = transType(Inst->getOperand(0)->getType());

  IntegerType *Int32Ty = IntegerType::get(*Context, 32);
  IntegerType *Int1Ty = IntegerType::get(*Context, 1);

  SmallVector<Type *, 8> ArgTys;
  std::vector<Value *> Args;
  Args.reserve(8);
  if (RetTy->getIntegerBitWidth() > 64) {
    llvm::PointerType *RetPtrTy = llvm::PointerType::get(RetTy, SPIRAS_Generic);
    Value *Alloca = new AllocaInst(RetTy, SPIRAS_Private, "", BB);
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

  auto Inst = static_cast<SPIRVArbFloatIntelInst *>(BI);

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
    llvm::PointerType *RetPtrTy = llvm::PointerType::get(RetTy, SPIRAS_Generic);
    ArgTys.push_back(RetPtrTy);
    Value *Alloca = new AllocaInst(RetTy, SPIRAS_Private, "", BB);
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

Function *SPIRVToLLVM::transFunction(SPIRVFunction *BF) {
  auto Loc = FuncMap.find(BF);
  if (Loc != FuncMap.end())
    return Loc->second;

  auto IsKernel = isKernel(BF);
  auto Linkage = IsKernel ? GlobalValue::ExternalLinkage : transLinkageType(BF);
  FunctionType *FT = dyn_cast<FunctionType>(transType(BF->getFunctionType()));
  std::string FuncName = BF->getName();
  StringRef FuncNameRef(FuncName);
  // Transform "@spirv.llvm_memset_p0i8_i32.volatile" to @llvm.memset.p0i8.i32
  // assuming llvm.memset is supported by the device compiler. If this
  // assumption is not safe, we should have a command line option to control
  // this behavior.
  if (FuncNameRef.consume_front("spirv.")) {
    FuncNameRef.consume_back(".volatile");
    FuncName = FuncNameRef.str();
    std::replace(FuncName.begin(), FuncName.end(), '_', '.');
  }
  Function *F = M->getFunction(FuncName);
  if (!F)
    F = Function::Create(FT, Linkage, FuncName, M);
  F = cast<Function>(mapValue(BF, F));
  mapFunction(BF, F);

  if (F->isIntrinsic())
    return F;

  F->setCallingConv(IsKernel ? CallingConv::SPIR_KERNEL
                             : CallingConv::SPIR_FUNC);
  if (BF->hasDecorate(DecorationReferencedIndirectlyINTEL))
    F->addFnAttr("referenced-indirectly");
  if (isFuncNoUnwind())
    F->addFnAttr(Attribute::NoUnwind);
  foreachFuncCtlMask(BF, [&](Attribute::AttrKind Attr) { F->addFnAttr(Attr); });

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto BA = BF->getArgument(I->getArgNo());
    mapValue(BA, &(*I));
    setName(&(*I), BA);
    BA->foreachAttr([&](SPIRVFuncParamAttrKind Kind) {
      Attribute::AttrKind LLVMKind = SPIRSPIRVFuncParamAttrMap::rmap(Kind);
      Type *AttrTy = nullptr;
      switch (LLVMKind) {
      case Attribute::AttrKind::ByVal:
        AttrTy = cast<PointerType>(I->getType())->getPointerElementType();
        break;
      case Attribute::AttrKind::StructRet:
        AttrTy = I->getType();
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
    SPIRVWord AlignmentBytes = 0;
    if (BA->hasDecorate(DecorationAlignment, 0, &AlignmentBytes))
      Builder.addAlignmentAttr(AlignmentBytes);
    I->addAttrs(Builder);
  }
  BF->foreachReturnValueAttr([&](SPIRVFuncParamAttrKind Kind) {
    if (Kind == FunctionParameterAttributeNoWrite)
      return;
    F->addRetAttr(SPIRSPIRVFuncParamAttrMap::rmap(Kind));
  });

  // Creating all basic blocks before creating instructions.
  for (size_t I = 0, E = BF->getNumBasicBlock(); I != E; ++I) {
    transValue(BF->getBasicBlock(I), F, nullptr);
  }

  for (size_t I = 0, E = BF->getNumBasicBlock(); I != E; ++I) {
    SPIRVBasicBlock *BBB = BF->getBasicBlock(I);
    BasicBlock *BB = dyn_cast<BasicBlock>(transValue(BBB, F, nullptr));
    for (size_t BI = 0, BE = BBB->getNumInst(); BI != BE; ++BI) {
      SPIRVInstruction *BInst = BBB->getInst(BI);
      transValue(BInst, F, BB, false);
    }
  }

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
  auto BT = BI->getType();
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
  auto *Int8PtrTyGen = Type::getInt8PtrTy(*Context, SPIRAS_Generic);
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
    auto Int8PtrTyGen = Type::getInt8PtrTy(*Context, SPIRAS_Generic);
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(*Context),
                                         {Int8PtrTyGen, Int8PtrTyGen}, false);
    F = Function::Create(FT, GlobalValue::ExternalLinkage, FName, M);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
  }
  auto Ops = BI->getOperands();
  SmallVector<Value *, 2> Args = {transBlockInvoke(Ops[0], BB),
                                  transValue(Ops[1], F, BB, false)};
  auto Call = CallInst::Create(F, Args, "", BB);
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
    auto Int8PtrTyGen = Type::getInt8PtrTy(*Context, SPIRAS_Generic);
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
  auto Call = CallInst::Create(F, Args, "", BB);
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  return Call;
}

Instruction *SPIRVToLLVM::transBuiltinFromInst(const std::string &FuncName,
                                               SPIRVInstruction *BI,
                                               BasicBlock *BB) {
  std::string MangledName;
  auto Ops = BI->getOperands();
  Type *RetTy =
      BI->hasType() ? transType(BI->getType()) : Type::getVoidTy(*Context);
  transOCLBuiltinFromInstPreproc(BI, RetTy, Ops);
  std::vector<Type *> ArgTys =
      transTypeVector(SPIRVInstruction::getOperandTypes(Ops));
  for (auto &I : ArgTys) {
    if (isa<FunctionType>(I)) {
      I = PointerType::get(I, SPIRAS_Private);
    }
  }

  if (BM->getDesiredBIsRepresentation() != BIsRepresentation::SPIRVFriendlyIR)
    mangleOpenClBuiltin(FuncName, ArgTys, MangledName);
  else
    MangledName =
        getSPIRVFriendlyIRFunctionName(FuncName, BI->getOpCode(), ArgTys);

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
    auto OC = BI->getOpCode();
    if (isGroupOpCode(OC) || isIntelSubgroupOpCode(OC) ||
        OC == OpControlBarrier)
      Func->addFnAttr(Attribute::Convergent);
  }
  auto Call =
      CallInst::Create(Func, transValue(Ops, BB->getParent(), BB), "", BB);
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  SPIRVDBG(spvdbgs() << "[transInstToBuiltinCall] " << *BI << " -> ";
           dbgs() << *Call << '\n';)
  Instruction *Inst = transOCLBuiltinPostproc(BI, Call, BB, FuncName);
  return Inst;
}

SPIRVToLLVM::SPIRVToLLVM(Module *LLVMModule, SPIRVModule *TheSPIRVModule)
    : M(LLVMModule), BM(TheSPIRVModule) {
  assert(M && "Initialization without an LLVM module is not allowed");
  Context = &M->getContext();
  DbgTran.reset(new SPIRVToLLVMDbgTran(TheSPIRVModule, LLVMModule, this));
}

std::string getSPIRVFuncSuffix(SPIRVInstruction *BI) {
  string Suffix = "";
  if (BI->getOpCode() == OpCreatePipeFromPipeStorage) {
    auto CPFPS = static_cast<SPIRVCreatePipeFromPipeStorage *>(BI);
    assert(CPFPS->getType()->isTypePipe() &&
           "Invalid type of CreatePipeFromStorage");
    auto PipeType = static_cast<SPIRVTypePipe *>(CPFPS->getType());
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
    auto GenericCastToPtrInst = BI->getType()->getPointerStorageClass();
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
    ostringstream OS;
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
  case internal::OpJointMatrixLoadINTEL:
    AddRetTypePostfix = true;
    break;
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit)
      AddRetTypePostfix = true;
    break;
  }
  }

  bool IsRetSigned;
  switch (OC) {
  case OpConvertFToU:
  case OpSatConvertSToU:
  case OpUConvert:
  case OpUDotKHR:
  case OpUDotAccSatKHR:
    IsRetSigned = false;
    break;
  default:
    IsRetSigned = true;
  }

  if (AddRetTypePostfix) {
    const Type *RetTy =
        BI->hasType() ? transType(BI->getType()) : Type::getVoidTy(*Context);
    return transBuiltinFromInst(getSPIRVFuncName(OC, RetTy, IsRetSigned) +
                                    getSPIRVFuncSuffix(BI),
                                BI, BB);
  }
  return transBuiltinFromInst(getSPIRVFuncName(OC, getSPIRVFuncSuffix(BI)), BI,
                              BB);
}

bool SPIRVToLLVM::translate() {
  if (!transAddressingModel())
    return false;

  for (unsigned I = 0, E = BM->getNumVariables(); I != E; ++I) {
    auto BV = BM->getVariable(I);
    if (BV->getStorageClass() != StorageClassFunction)
      transValue(BV, nullptr, nullptr);
    else
      transGlobalCtorDtors(BV);
  }

  // Compile unit might be needed during translation of debug intrinsics.
  for (SPIRVExtInst *EI : BM->getDebugInstVec()) {
    // Translate Compile Unit first.
    // It shuldn't be far from the beginig of the vector
    if (EI->getExtOp() == SPIRVDebug::CompilationUnit) {
      DbgTran->transDebugInst(EI);
      // Fixme: there might be more then one Compile Unit.
      break;
    }
  }
  // Then translate all debug instructions.
  for (SPIRVExtInst *EI : BM->getDebugInstVec()) {
    DbgTran->transDebugInst(EI);
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
  // TODO: add an option to control the builtin format in SPV-IR.
  // The primary format should be function calls:
  //   e.g. call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
  // The secondary format should be global variables:
  //   e.g. load i32, i32* @__spirv_BuiltInGlobalLinearId, align 4
  // If the desired format is global variables, we don't have to lower them
  // as calls.
  if (!lowerBuiltinVariablesToCalls(M))
    return false;
  if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR) {
    SPIRVWord SrcLangVer = 0;
    BM->getSourceLanguage(&SrcLangVer);
    bool IsCpp = SrcLangVer == kOCLVer::CL21;
    if (!postProcessBuiltinsReturningStruct(M, IsCpp))
      return false;
  }
  eraseUselessFunctions(M);

  DbgTran->addDbgInfoVersion();
  DbgTran->finalize();
  return true;
}

bool SPIRVToLLVM::transAddressingModel() {
  switch (BM->getAddressingModel()) {
  case AddressingModelPhysical64:
    M->setTargetTriple(SPIR_TARGETTRIPLE64);
    M->setDataLayout(SPIR_DATALAYOUT64);
    break;
  case AddressingModelPhysical32:
    M->setTargetTriple(SPIR_TARGETTRIPLE32);
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

void generateIntelFPGAAnnotation(const SPIRVEntry *E,
                                 llvm::SmallString<256> &AnnotStr) {
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
    for (auto Str : E->getDecorationStringLiteral(DecorationMergeINTEL))
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
  if (E->hasDecorate(DecorationUserSemantic))
    Out << E->getDecorationStringLiteral(DecorationUserSemantic).front();

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
  if (LSUParamsBitmask == 0)
    return;
  Out << "{params:" << LSUParamsBitmask << "}" << AdditionalParamsStr;
}

void generateIntelFPGAAnnotationForStructMember(
    const SPIRVEntry *E, SPIRVWord MemberNumber,
    llvm::SmallString<256> &AnnotStr) {
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
    for (auto Str : E->getMemberDecorationStringLiteral(DecorationMergeINTEL,
                                                        MemberNumber))
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

  if (E->hasMemberDecorate(DecorationUserSemantic, 0, MemberNumber))
    Out << E->getMemberDecorationStringLiteral(DecorationUserSemantic,
                                               MemberNumber)
               .front();
}

void SPIRVToLLVM::transIntelFPGADecorations(SPIRVValue *BV, Value *V) {
  if (!BV->isVariable() && !BV->isInst())
    return;

  if (auto *Inst = dyn_cast<Instruction>(V)) {
    auto *AL = dyn_cast<AllocaInst>(Inst);
    Type *AllocatedTy = AL ? AL->getAllocatedType() : Inst->getType();

    IRBuilder<> Builder(Inst->getParent());

    Type *Int8PtrTyPrivate = Type::getInt8PtrTy(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = IntegerType::get(*Context, 32);

    Value *UndefInt8Ptr = UndefValue::get(Int8PtrTyPrivate);
    Value *UndefInt32 = UndefValue::get(Int32Ty);

    if (AL && BV->getType()->getPointerElementType()->isTypeStruct()) {
      auto *ST = BV->getType()->getPointerElementType();
      SPIRVTypeStruct *STS = static_cast<SPIRVTypeStruct *>(ST);

      for (SPIRVWord I = 0; I < STS->getMemberCount(); ++I) {
        SmallString<256> AnnotStr;
        generateIntelFPGAAnnotationForStructMember(ST, I, AnnotStr);
        if (!AnnotStr.empty()) {
          auto *GS = Builder.CreateGlobalStringPtr(AnnotStr);

          auto GEP = Builder.CreateConstInBoundsGEP2_32(AllocatedTy, AL, 0, I);

          Type *IntTy = GEP->getType()->getPointerElementType()->isIntegerTy()
                            ? GEP->getType()
                            : Int8PtrTyPrivate;

          auto AnnotationFn = llvm::Intrinsic::getDeclaration(
              M, Intrinsic::ptr_annotation, IntTy);

          llvm::Value *Args[] = {
              Builder.CreateBitCast(GEP, IntTy, GEP->getName()),
              Builder.CreateBitCast(GS, Int8PtrTyPrivate), UndefInt8Ptr,
              UndefInt32, UndefInt8Ptr};
          Builder.CreateCall(AnnotationFn, Args);
        }
      }
    }

    SmallString<256> AnnotStr;
    generateIntelFPGAAnnotation(BV, AnnotStr);
    if (!AnnotStr.empty()) {
      Constant *GS = nullptr;
      std::string StringAnnotStr = AnnotStr.c_str();
      auto AnnotItr = AnnotationsMap.find(StringAnnotStr);
      if (AnnotItr != AnnotationsMap.end()) {
        GS = AnnotItr->second;
      } else {
        GS = Builder.CreateGlobalStringPtr(AnnotStr);
        AnnotationsMap.emplace(std::move(StringAnnotStr), GS);
      }

      Value *BaseInst =
          AL ? Builder.CreateBitCast(V, Int8PtrTyPrivate, V->getName()) : Inst;

      // Try to find alloca instruction for statically allocated variables.
      // Alloca might be hidden by a couple of casts.
      bool isStaticMemoryAttribute = AL ? true : false;
      while (!isStaticMemoryAttribute && Inst &&
             (isa<BitCastInst>(Inst) || isa<AddrSpaceCastInst>(Inst))) {
        Inst = dyn_cast<Instruction>(Inst->getOperand(0));
        isStaticMemoryAttribute = (Inst && isa<AllocaInst>(Inst));
      }
      auto AnnotationFn =
          isStaticMemoryAttribute
              ? llvm::Intrinsic::getDeclaration(M, Intrinsic::var_annotation)
              : llvm::Intrinsic::getDeclaration(M, Intrinsic::ptr_annotation,
                                                BaseInst->getType());

      llvm::Value *Args[] = {BaseInst,
                             Builder.CreateBitCast(GS, Int8PtrTyPrivate),
                             UndefInt8Ptr, UndefInt32, UndefInt8Ptr};
      Builder.CreateCall(AnnotationFn, Args);
    }
  } else if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    SmallString<256> AnnotStr;
    generateIntelFPGAAnnotation(BV, AnnotStr);

    if (AnnotStr.empty()) {
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

    Constant *StrConstant =
        ConstantDataArray::getString(*Context, StringRef(AnnotStr));

    auto *GS = new GlobalVariable(*GV->getParent(), StrConstant->getType(),
                                  /*IsConstant*/ true,
                                  GlobalValue::PrivateLinkage, StrConstant, "");

    GS->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GS->setSection("llvm.metadata");

    Type *ResType = PointerType::getInt8PtrTy(
        GV->getContext(), GV->getType()->getPointerAddressSpace());
    Constant *C = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, ResType);

    Type *Int8PtrTyPrivate = Type::getInt8PtrTy(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = Type::getInt32Ty(*Context);

    llvm::Constant *Fields[5] = {
        C, ConstantExpr::getBitCast(GS, Int8PtrTyPrivate),
        UndefValue::get(Int8PtrTyPrivate), UndefValue::get(Int32Ty),
        UndefValue::get(Int8PtrTyPrivate)};

    GlobalAnnotations.push_back(ConstantStruct::getAnon(Fields));
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
  if (BV->hasDecorateId(internal::DecorationAliasScopeINTEL)) {
    std::vector<SPIRVId> AliasListIds;
    AliasListIds =
        BV->getDecorationIdLiterals(internal::DecorationAliasScopeINTEL);
    assert(AliasListIds.size() == 1 &&
           "Memory aliasing decorations must have one argument");
    addMemAliasMetadata(Inst, AliasListIds[0], LLVMContext::MD_alias_scope);
  }
  if (BV->hasDecorateId(internal::DecorationNoAliasINTEL)) {
    std::vector<SPIRVId> AliasListIds;
    AliasListIds =
        BV->getDecorationIdLiterals(internal::DecorationNoAliasINTEL);
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
  auto TransFun = transFunction(Fun);
  for (auto UsSem : Fun->getDecorationStringLiteral(DecorationUserSemantic)) {
    auto V = cast<Value>(TransFun);
    Constant *StrConstant =
        ConstantDataArray::getString(*Context, StringRef(UsSem));
    auto *GS = new GlobalVariable(
        *TransFun->getParent(), StrConstant->getType(),
        /*IsConstant*/ true, GlobalValue::PrivateLinkage, StrConstant, "");

    GS->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GS->setSection("llvm.metadata");

    Type *ResType = PointerType::getInt8PtrTy(
        V->getContext(), V->getType()->getPointerAddressSpace());
    Constant *C =
        ConstantExpr::getPointerBitCastOrAddrSpaceCast(TransFun, ResType);

    Type *Int8PtrTyPrivate = Type::getInt8PtrTy(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = Type::getInt32Ty(*Context);

    llvm::Constant *Fields[5] = {
        C, ConstantExpr::getBitCast(GS, Int8PtrTyPrivate),
        UndefValue::get(Int8PtrTyPrivate), UndefValue::get(Int32Ty),
        UndefValue::get(Int8PtrTyPrivate)};
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
    case spv::internal::DecorationHostAccessINTEL: {
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

void SPIRVToLLVM::transVarDecorationsToMetadata(SPIRVValue *BV, Value *V) {
  if (!BV->isVariable())
    return;

  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    std::vector<SPIRVDecorate const *> Decorates = BV->getDecorations();
    if (!Decorates.empty()) {
      MDNode *MDList = transDecorationsToMetadataList(Context, Decorates);
      GV->setMetadata(SPIRV_MD_DECORATIONS, MDList);
    }
  }
}

bool SPIRVToLLVM::transDecoration(SPIRVValue *BV, Value *V) {
  if (!transAlign(BV, V))
    return false;

  transIntelFPGADecorations(BV, V);
  transMemAliasingINTELDecorations(BV, V);

  // Decoration metadata is only enabled in SPIR-V friendly mode
  if (BM->getDesiredBIsRepresentation() == BIsRepresentation::SPIRVFriendlyIR)
    transVarDecorationsToMetadata(BV, V);

  DbgTran->transDbgInfo(BV, V);
  return true;
}

void SPIRVToLLVM::transGlobalCtorDtors(SPIRVVariable *BV) {
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
  PointerType *CtorTy = PointerType::getUnqual(
      FunctionType::get(Type::getVoidTy(*Context), false));
  PointerType *ComdatTy = Type::getInt8PtrTy(*Context);
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

    if (BF->hasDecorate(internal::DecorationCallableFunctionINTEL))
      F->addFnAttr(kVCMetadata::VCCallable);
    if (isKernel(BF) &&
        BF->getExecutionMode(internal::ExecutionModeFastCompositeKernelINTEL))
      F->addFnAttr(kVCMetadata::VCFCEntry);

    if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Generate metadata for reqd_work_group_size
    if (auto EM = BF->getExecutionMode(ExecutionModeLocalSize)) {
      F->setMetadata(kSPIR2MD::WGSize,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for work_group_size_hint
    if (auto EM = BF->getExecutionMode(ExecutionModeLocalSizeHint)) {
      F->setMetadata(kSPIR2MD::WGSizeHint,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for vec_type_hint
    if (auto EM = BF->getExecutionMode(ExecutionModeVecTypeHint)) {
      std::vector<Metadata *> MetadataVec;
      Type *VecHintTy = decodeVecTypeHint(*Context, EM->getLiterals()[0]);
      assert(VecHintTy);
      MetadataVec.push_back(ValueAsMetadata::get(UndefValue::get(VecHintTy)));
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
      auto SizeMD = ConstantAsMetadata::get(getUInt32(M, EM->getLiterals()[0]));
      F->setMetadata(kSPIR2MD::SubgroupSize, MDNode::get(*Context, SizeMD));
    }
    // Generate metadata for max_work_group_size
    if (auto EM = BF->getExecutionMode(ExecutionModeMaxWorkgroupSizeINTEL)) {
      F->setMetadata(kSPIR2MD::MaxWGSize,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for no_global_work_offset
    if (BF->getExecutionMode(ExecutionModeNoGlobalOffsetINTEL)) {
      F->setMetadata(kSPIR2MD::NoGlobalOffset, MDNode::get(*Context, {}));
    }
    // Generate metadata for max_global_work_dim
    if (auto EM = BF->getExecutionMode(ExecutionModeMaxWorkDimINTEL)) {
      F->setMetadata(kSPIR2MD::MaxWGDim,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for num_simd_work_items
    if (auto EM = BF->getExecutionMode(ExecutionModeNumSIMDWorkitemsINTEL)) {
      F->setMetadata(kSPIR2MD::NumSIMD,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for scheduler_target_fmax_mhz
    if (auto EM =
            BF->getExecutionMode(ExecutionModeSchedulerTargetFmaxMhzINTEL)) {
      F->setMetadata(kSPIR2MD::FmaxMhz,
                     getMDNodeStringIntVec(Context, EM->getLiterals()));
    }
    // Generate metadata for Intel FPGA streaming interface
    if (auto *EM = BF->getExecutionMode(
            internal::ExecutionModeStreamingInterfaceINTEL)) {
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
    auto Literals =
        Arg->getDecorationLiterals(internal::DecorationRuntimeAlignedINTEL);
    assert(Literals.size() == 1 &&
           "RuntimeAlignedINTEL decoration shall have 1 ID literal");

    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt1Ty(*Context), Literals[0]));
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
    F->addAttributeAtIndex(AttributeList::ReturnIndex, SEVAttr.getValue());

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
    if (BA->hasDecorate(internal::DecorationFuncParamKindINTEL, 0, &Kind)) {
      Attribute Attr = Attribute::get(*Context, kVCMetadata::VCArgumentKind,
                                      std::to_string(Kind));
      F->addParamAttr(ArgNo, Attr);
    }
    SEVAttr = translateSEVMetadata(BA, F->getContext());
    if (SEVAttr)
      F->addParamAttr(ArgNo, SEVAttr.getValue());
    if (BA->hasDecorate(internal::DecorationFuncParamDescINTEL)) {
      auto Desc =
          BA->getDecorationStringLiteral(internal::DecorationFuncParamDescINTEL)
              .front();
      Attribute Attr =
          Attribute::get(*Context, kVCMetadata::VCArgumentDesc, Desc);
      F->addParamAttr(ArgNo, Attr);
    }
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

      for (auto DecPtr : DenormModes) {
        auto DecDenorm =
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

  if (auto EM = BF->getExecutionMode(ExecutionModeSharedLocalMemorySizeINTEL)) {
    unsigned int SLMSize = EM->getLiterals()[0];
    Attribute Attr = Attribute::get(*Context, kVCMetadata::VCSLMSize,
                                    std::to_string(SLMSize));
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
  if (BF->hasDecorate(DecorationFuseLoopsInFunctionINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals =
        BF->getDecorationLiterals(DecorationFuseLoopsInFunctionINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[1])));
    F->setMetadata(kSPIR2MD::LoopFuse, MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(internal::DecorationMathOpDSPModeINTEL)) {
    std::vector<SPIRVWord> Literals =
        BF->getDecorationLiterals(internal::DecorationMathOpDSPModeINTEL);
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
  if (BF->hasDecorate(internal::DecorationInitiationIntervalINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals =
        BF->getDecorationLiterals(internal::DecorationInitiationIntervalINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    F->setMetadata(kSPIR2MD::InitiationInterval,
                   MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(internal::DecorationMaxConcurrencyINTEL)) {
    std::vector<Metadata *> MetadataVec;
    auto Literals =
        BF->getDecorationLiterals(internal::DecorationMaxConcurrencyINTEL);
    MetadataVec.push_back(ConstantAsMetadata::get(getUInt32(M, Literals[0])));
    F->setMetadata(kSPIR2MD::MaxConcurrency,
                   MDNode::get(*Context, MetadataVec));
  }
  if (BF->hasDecorate(internal::DecorationPipelineEnableINTEL)) {
    auto Literals =
        BF->getDecorationLiterals(internal::DecorationPipelineEnableINTEL);
    std::vector<Metadata *> MetadataVec;
    MetadataVec.push_back(ConstantAsMetadata::get(getInt32(M, !Literals[0])));
    F->setMetadata(kSPIR2MD::DisableLoopPipelining,
                   MDNode::get(*Context, MetadataVec));
  }
  return true;
}

bool SPIRVToLLVM::transAlign(SPIRVValue *BV, Value *V) {
  if (auto AL = dyn_cast<AllocaInst>(V)) {
    SPIRVWord Align = 0;
    if (BV->hasAlignment(&Align))
      AL->setAlignment(llvm::Align(Align));
    return true;
  }
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    SPIRVWord Align = 0;
    if (BV->hasAlignment(&Align))
      GV->setAlignment(MaybeAlign(Align));
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

  std::vector<Type *> ArgTypes = transTypeVector(BC->getArgTypes());
  Type *RetTy = transType(BC->getType());
  std::string MangledName =
      getSPIRVFriendlyIRFunctionName(ExtOp, ArgTypes, RetTy);

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
      F->addFnAttr(Attribute::ReadNone);
  }
  auto Args = transValue(BC->getArgValues(), F, BB);
  SPIRVDBG(dbgs() << "[transOCLBuiltinFromExtInst] Function: " << *F
                  << ", Args: ";
           for (auto &I
                : Args) dbgs()
           << *I << ", ";
           dbgs() << '\n');
  CallInst *CI = CallInst::Create(F, Args, BC->getName(), BB);
  setCallingConv(CI);
  addFnAttr(CI, Attribute::NoUnwind);
  return CI;
}

// SPIR-V only contains language version. Use OpenCL language version as
// SPIR version.
void SPIRVToLLVM::transSourceLanguage() {
  SPIRVWord Ver = 0;
  SourceLanguage Lang = BM->getSourceLanguage(&Ver);
  if (Lang != SourceLanguageUnknown && // Allow unknown for debug info test
      Lang != SourceLanguageOpenCL_C && Lang != SourceLanguageOpenCL_CPP)
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

  addOCLVersionMetadata(Context, M, kSPIR2MD::OCLVer, Major, Minor);
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
    if (V->getOpCode() == OpVariable) {
      if (static_cast<const SPIRVVariable *>(V)->getInitializer() == 0)
        return GlobalValue::ExternalLinkage;
    }
    // Definition
    return GlobalValue::AvailableExternallyLinkage;
  case LinkageTypeExport:
    if (V->getOpCode() == OpVariable) {
      if (static_cast<const SPIRVVariable *>(V)->getInitializer() == 0)
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
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  BuiltinFuncMangleInfo BtnInfo;
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return cast<Instruction>(mapValue(
      I, mutateCallInst(
             M, CI,
             [=](CallInst *, std::vector<Value *> &Args) {
               auto *OldArg = CI->getOperand(0);
               auto *NewArgTy = FixedVectorType::get(
                   Type::getInt8Ty(*Context),
                   cast<FixedVectorType>(OldArg->getType())->getNumElements());
               auto *NewArg =
                   CastInst::CreateSExtOrBitCast(OldArg, NewArgTy, "", CI);
               Args[0] = NewArg;
               return getSPIRVFuncName(I->getOpCode(), getSPIRVFuncSuffix(I));
             },
             &BtnInfo, &Attrs, /*TakeFuncName=*/true)));
}

Instruction *SPIRVToLLVM::transRelational(SPIRVInstruction *I, BasicBlock *BB) {
  CallInst *CI = cast<CallInst>(transSPIRVBuiltinFromInst(I, BB));
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  BuiltinFuncMangleInfo BtnInfo;
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return cast<Instruction>(mapValue(
      I, mutateCallInst(
             M, CI,
             [=](CallInst *, std::vector<Value *> &Args, llvm::Type *&RetTy) {
               if (CI->getType()->isVectorTy()) {
                 RetTy = FixedVectorType::get(
                     Type::getInt8Ty(*Context),
                     cast<FixedVectorType>(CI->getType())->getNumElements());
               }
               return getSPIRVFuncName(I->getOpCode(), getSPIRVFuncSuffix(I));
             },
             [=](CallInst *NewCI) -> Instruction * {
               Type *RetTy = CI->getType();
               if (RetTy == NewCI->getType())
                 return NewCI;
               return CastInst::CreateTruncOrBitCast(NewCI, RetTy, "",
                                                     NewCI->getNextNode());
             },
             &BtnInfo, &Attrs, /*TakeFuncName=*/true)));
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

  llvm::ModulePass *LoweringPass =
      createSPIRVBIsLoweringPass(*M, Opts.getDesiredBIsRepresentation());
  if (LoweringPass) {
    // nullptr means no additional lowering is required
    llvm::legacy::PassManager PassMgr;
    PassMgr.add(LoweringPass);
    PassMgr.run(*M);
  }

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
        SpecConstInfo.emplace_back(SpecConstIdLiteral, SpecConstSize);
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
