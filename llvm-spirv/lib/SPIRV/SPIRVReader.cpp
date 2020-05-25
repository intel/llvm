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
#include "SPIRVModule.h"
#include "SPIRVToLLVMDbgTran.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"

#include "llvm/ADT/DenseMap.h"
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
const static char *Const = "const";
const static char *Volatile = "volatile";
const static char *Restrict = "restrict";
const static char *Pipe = "pipe";
} // namespace kOCLTypeQualifierName

static bool isOpenCLKernel(SPIRVFunction *BF) {
  return BF->getModule()->isEntryPoint(ExecutionModelKernel, BF->getId());
}

static void dumpLLVM(Module *M, const std::string &FName) {
  std::error_code EC;
  raw_fd_ostream FS(FName, EC, sys::fs::F_None);
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

static void addOCLKernelArgumentMetadata(
    LLVMContext *Context, const std::string &MDName, SPIRVFunction *BF,
    llvm::Function *Fn,
    std::function<Metadata *(SPIRVFunctionParameter *)> Func) {
  std::vector<Metadata *> ValueVec;
  BF->foreachArgument(
      [&](SPIRVFunctionParameter *Arg) { ValueVec.push_back(Func(Arg)); });
  Fn->setMetadata(MDName, MDNode::get(*Context, ValueVec));
}

Value *SPIRVToLLVM::getTranslatedValue(SPIRVValue *BV) {
  auto Loc = ValueMap.find(BV);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
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

void SPIRVToLLVM::setAttrByCalledFunc(CallInst *Call) {
  Function *F = Call->getCalledFunction();
  assert(F);
  if (F->isIntrinsic()) {
    return;
  }
  Call->setCallingConv(F->getCallingConv());
  Call->setAttributes(F->getAttributes());
}

bool SPIRVToLLVM::transOCLBuiltinsFromVariables() {
  std::vector<GlobalVariable *> WorkList;
  for (auto I = M->global_begin(), E = M->global_end(); I != E; ++I) {
    SPIRVBuiltinVariableKind Kind;
    if (!isSPIRVBuiltinVariable(&(*I), &Kind))
      continue;
    if (!transOCLBuiltinFromVariable(&(*I), Kind))
      return false;
    WorkList.push_back(&(*I));
  }
  for (auto &I : WorkList) {
    I->eraseFromParent();
  }
  return true;
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

// Variable like GlobalInvolcationId[x] -> get_global_id(x).
// Variable like WorkDim -> get_work_dim().
bool SPIRVToLLVM::transOCLBuiltinFromVariable(GlobalVariable *GV,
                                              SPIRVBuiltinVariableKind Kind) {
  std::string FuncName = SPIRSPIRVBuiltinVariableMap::rmap(Kind);
  std::string MangledName;
  Type *ReturnTy = GV->getType()->getPointerElementType();
  bool IsVec = ReturnTy->isVectorTy();
  if (IsVec)
    ReturnTy = cast<VectorType>(ReturnTy)->getElementType();
  std::vector<Type *> ArgTy;
  if (IsVec)
    ArgTy.push_back(Type::getInt32Ty(*Context));
  mangleOpenClBuiltin(FuncName, ArgTy, MangledName);
  Function *Func = M->getFunction(MangledName);
  if (!Func) {
    FunctionType *FT = FunctionType::get(ReturnTy, ArgTy, false);
    Func = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    Func->setCallingConv(CallingConv::SPIR_FUNC);
    Func->addFnAttr(Attribute::NoUnwind);
    Func->addFnAttr(Attribute::ReadNone);
  }
  std::vector<Instruction *> Deletes;
  std::vector<Instruction *> Uses;
  for (auto UI = GV->user_begin(), UE = GV->user_end(); UI != UE; ++UI) {
    LoadInst *LD = nullptr;
    AddrSpaceCastInst *ASCast = dyn_cast<AddrSpaceCastInst>(*UI);
    if (ASCast) {
      LD = cast<LoadInst>(*ASCast->user_begin());
    } else {
      LD = cast<LoadInst>(*UI);
    }
    if (!IsVec) {
      Uses.push_back(LD);
      Deletes.push_back(LD);
      if (ASCast)
        Deletes.push_back(ASCast);
      continue;
    }
    for (auto LDUI = LD->user_begin(), LDUE = LD->user_end(); LDUI != LDUE;
         ++LDUI) {
      assert(isa<ExtractElementInst>(*LDUI) && "Unsupported use");
      auto EEI = dyn_cast<ExtractElementInst>(*LDUI);
      Uses.push_back(EEI);
      Deletes.push_back(EEI);
    }
    Deletes.push_back(LD);
    if (ASCast)
      Deletes.push_back(ASCast);
  }
  for (auto &I : Uses) {
    std::vector<Value *> Arg;
    if (auto EEI = dyn_cast<ExtractElementInst>(I))
      Arg.push_back(EEI->getIndexOperand());
    auto Call = CallInst::Create(Func, Arg, "", I);
    Call->takeName(I);
    setAttrByCalledFunc(Call);
    SPIRVDBG(dbgs() << "[transOCLBuiltinFromVariable] " << *I << " -> " << *Call
                    << '\n';)
    I->replaceAllUsesWith(Call);
  }
  for (auto &I : Deletes) {
    I->eraseFromParent();
  }
  return true;
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
  std::string Name = std::string(kSPR2TypeName::OCLPrefix) +
                     rmap<std::string>(ST->getDescriptor());
  SPIRVToLLVM::insertImageNameAccessQualifier(ST, Name);
  return Name;
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
SPIRVToLLVM::transOCLPipeTypeName(SPIRV::SPIRVTypePipe *PT,
                                  bool UseSPIRVFriendlyFormat,
                                  SPIRVAccessQualifierKind PipeAccess) {
  assert((PipeAccess == AccessQualifierReadOnly ||
          PipeAccess == AccessQualifierWriteOnly) &&
         "Invalid access qualifier");

  if (!UseSPIRVFriendlyFormat)
    return PipeAccess == AccessQualifierWriteOnly ? kSPR2TypeName::PipeWO
                                                  : kSPR2TypeName::PipeRO;
  else
    return std::string(kSPIRVTypeName::PrefixAndDelim) + kSPIRVTypeName::Pipe +
           kSPIRVTypeName::Delimiter + kSPIRVTypeName::PostfixDelim +
           PipeAccess;
}

std::string
SPIRVToLLVM::transOCLPipeStorageTypeName(SPIRV::SPIRVTypePipeStorage *PST) {
  return std::string(kSPIRVTypeName::PrefixAndDelim) +
         kSPIRVTypeName::PipeStorage;
}

Type *SPIRVToLLVM::transType(SPIRVType *T, bool IsClassMember) {
  auto Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(spvdbgs() << "[transType] " << *T << " -> ";)
  T->validate();
  switch (T->getOpCode()) {
  case OpTypeVoid:
    return mapType(T, Type::getVoidTy(*Context));
  case OpTypeBool:
    return mapType(T, Type::getInt1Ty(*Context));
  case OpTypeInt:
    return mapType(T, Type::getIntNTy(*Context, T->getIntegerBitWidth()));
  case OpTypeFloat:
    return mapType(T, transFPType(T));
  case OpTypeArray:
    return mapType(T, ArrayType::get(transType(T->getArrayElementType()),
                                     T->getArrayLength()));
  case OpTypePointer:
    return mapType(
        T, PointerType::get(
               transType(T->getPointerElementType(), IsClassMember),
               SPIRSPIRVAddrSpaceMap::rmap(T->getPointerStorageClass())));
  case OpTypeVector:
    return mapType(T, VectorType::get(transType(T->getVectorComponentType()),
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
      if (auto OldST = M->getTypeByName(Name))
        OldST->setName("");
    } else {
      Name = "structtype";
    }
    auto *StructTy = StructType::create(*Context, Name);
    mapType(ST, StructTy);
    SmallVector<Type *, 4> MT;
    for (size_t I = 0, E = ST->getMemberCount(); I != E; ++I)
      MT.push_back(transType(ST->getMemberType(I), true));
    StructTy->setBody(MT, ST->isPacked());
    return StructTy;
  }
  case OpTypePipe: {
    auto PT = static_cast<SPIRVTypePipe *>(T);
    return mapType(T, getOrCreateOpaquePtrType(
                          M,
                          transOCLPipeTypeName(PT, IsClassMember,
                                               PT->getAccessQualifier()),
                          getOCLOpaqueTypeAddrSpace(T->getOpCode())));
  }
  case OpTypePipeStorage: {
    auto PST = static_cast<SPIRVTypePipeStorage *>(T);
    return mapType(
        T, getOrCreateOpaquePtrType(M, transOCLPipeStorageTypeName(PST),
                                    getOCLOpaqueTypeAddrSpace(T->getOpCode())));
  }
  // OpenCL Compiler does not use this instruction
  case OpTypeVmeImageINTEL:
    return nullptr;
  default: {
    auto OC = T->getOpCode();
    if (isOpaqueGenericTypeOpCode(OC) || isSubgroupAvcINTELTypeOpCode(OC)) {
      auto Name = isSubgroupAvcINTELTypeOpCode(OC)
                      ? OCLSubgroupINTELTypeOpCodeMap::rmap(OC)
                      : OCLOpaqueTypeOpCodeMap::rmap(OC);
      return mapType(
          T, getOrCreateOpaquePtrType(M, Name, getOCLOpaqueTypeAddrSpace(OC)));
    }
    llvm_unreachable("Not implemented");
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
      llvm_unreachable("invalid integer size");
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

bool SPIRVToLLVM::isSPIRVCmpInstTransToLLVMInst(SPIRVInstruction *BI) const {
  auto OC = BI->getOpCode();
  return isCmpOpCode(OC) && !(OC >= OpLessOrGreater && OC <= OpUnordered);
}

bool SPIRVToLLVM::isDirectlyTranslatedToOCL(Op OpCode) const {
  // Not every spirv opcode which is placed in OCLSPIRVBuiltinMap is
  // translated directly to OCL builtin. Some of them are translated
  // to LLVM representation without any modifications (SPIRV format of
  // instruction is represented in LLVM) and then its translated to
  // clang-consistent format in SPIRVToOCL pass.
  if (isSubgroupAvcINTELInstructionOpCode(OpCode) ||
      isIntelSubgroupOpCode(OpCode))
    return true;
  if (OCLSPIRVBuiltinMap::rfind(OpCode, nullptr)) {
    // everything except atomics, groups, pipes and media_block_io_intel is
    // directly translated
    return !(isAtomicOpCode(OpCode) || isGroupOpCode(OpCode) ||
             isPipeOpCode(OpCode) || isMediaBlockINTELOpcode(OpCode));
  }
  return false;
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
    if (!LoopControlParameters.empty()) {
      Metadata.push_back(llvm::MDNode::get(
          *Context,
          getMetadataFromNameAndParameter("llvm.loop.ivdep.safelen",
                                          LoopControlParameters[NumParam])));
      ++NumParam;
      // TODO: Fix the increment/assertion logic in all of the conditions
      assert(NumParam <= LoopControlParameters.size() &&
             "Missing loop control parameter!");
    }
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
  if (LC & LoopControlInitiationIntervalINTEL) {
    Metadata.push_back(llvm::MDNode::get(
        *Context, getMetadataFromNameAndParameter(
                      "llvm.loop.ii.count", LoopControlParameters[NumParam])));
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlMaxConcurrencyINTEL) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.max_concurrency.count",
                                        LoopControlParameters[NumParam])));
    ++NumParam;
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlDependencyArrayINTEL) {
    // Collect array variable <-> safelen information
    std::map<Value *, unsigned> ArraySflnMap;
    unsigned NumOperandPairs = LoopControlParameters[NumParam];
    unsigned OperandsEndIndex = NumParam + NumOperandPairs * 2;
    assert(OperandsEndIndex <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
    SPIRVModule *M = LM->getModule();
    while (NumParam < OperandsEndIndex) {
      SPIRVId ArraySPIRVId = LoopControlParameters[++NumParam];
      Value *ArrayVar = ValueMap[M->getValue(ArraySPIRVId)];
      unsigned Safelen = LoopControlParameters[++NumParam];
      ArraySflnMap.emplace(ArrayVar, Safelen);
    }

    // A single run over the loop to retrieve all GetElementPtr instructions
    // that access relevant array variables
    std::map<Value *, std::vector<GetElementPtrInst *>> ArrayGEPMap;
    for (const auto &BB : LoopObj->blocks()) {
      for (Instruction &I : *BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP)
          continue;

        Value *AccessedArray = GEP->getPointerOperand();
        auto ArraySflnIt = ArraySflnMap.find(AccessedArray);
        if (ArraySflnIt != ArraySflnMap.end())
          ArrayGEPMap[AccessedArray].push_back(GEP);
      }
    }

    // Create index group metadata nodes - one per each array
    // variables. Mark each GEP accessing a particular array variable
    // into a corresponding index group
    std::map<unsigned, std::vector<MDNode *>> SafelenIdxGroupMap;
    for (auto &ArrayGEPIt : ArrayGEPMap) {
      // Emit a distinct index group that will be referenced from
      // llvm.loop.parallel_access_indices metadata
      auto *CurrentDepthIdxGroup = llvm::MDNode::getDistinct(*Context, None);
      unsigned Safelen = ArraySflnMap.find(ArrayGEPIt.first)->second;
      SafelenIdxGroupMap[Safelen].push_back(CurrentDepthIdxGroup);

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
  }
  if (LC & LoopControlPipelineEnableINTEL) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.intel.pipelining.enable",
                                        LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlLoopCoalesceINTEL) {
    // If LoopCoalesce has no parameters
    if (LoopControlParameters.empty()) {
      Metadata.push_back(llvm::MDNode::get(
          *Context, getMetadataFromName("llvm.loop.coalesce.enable")));
    } else {
      Metadata.push_back(llvm::MDNode::get(
          *Context,
          getMetadataFromNameAndParameter("llvm.loop.coalesce.count",
                                          LoopControlParameters[NumParam++])));
    }
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlMaxInterleavingINTEL) {
    Metadata.push_back(llvm::MDNode::get(
        *Context,
        getMetadataFromNameAndParameter("llvm.loop.max_interleaving.count",
                                        LoopControlParameters[NumParam++])));
    assert(NumParam <= LoopControlParameters.size() &&
           "Missing loop control parameter!");
  }
  if (LC & LoopControlSpeculatedIterationsINTEL) {
    Metadata.push_back(llvm::MDNode::get(
        *Context, getMetadataFromNameAndParameter(
                      "llvm.loop.intel.speculated.iterations.count",
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

void SPIRVToLLVM::insertImageNameAccessQualifier(SPIRV::SPIRVTypeImage *ST,
                                                 std::string &Name) {
  SPIRVAccessQualifierKind Acc = ST->hasAccessQualifier()
                                     ? ST->getAccessQualifier()
                                     : AccessQualifierReadOnly;
  std::string QName = rmap<std::string>(Acc);
  // transform: read_only -> ro, write_only -> wo, read_write -> rw
  QName = QName.substr(0, 1) + QName.substr(QName.find("_") + 1, 1) + "_";
  assert(!Name.empty() && "image name should not be empty");
  Name.insert(Name.size() - 1, QName);
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

Value *SPIRVToLLVM::transDeviceEvent(SPIRVValue *BV, Function *F,
                                     BasicBlock *BB) {
  auto Val = transValue(BV, F, BB, false);
  auto Ty = dyn_cast<PointerType>(Val->getType());
  assert(Ty && "Invalid Device Event");
  if (Ty->getAddressSpace() == SPIRAS_Generic)
    return Val;

  IRBuilder<> Builder(BB);
  auto EventTy = PointerType::get(Ty->getElementType(), SPIRAS_Generic);
  return Builder.CreateAddrSpaceCast(Val, EventTy);
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
    CO = Instruction::AddrSpaceCast;
    break;
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
    if (V & FPFastMathModeFastMask)
      FMF.setFast();
    Inst->setFastMathFlags(FMF);
  }
}

BinaryOperator *SPIRVToLLVM::transShiftLogicalBitwiseInst(SPIRVValue *BV,
                                                          BasicBlock *BB,
                                                          Function *F) {
  SPIRVBinary *BBN = static_cast<SPIRVBinary *>(BV);
  assert(BB && "Invalid BB");
  Instruction::BinaryOps BO;
  auto OP = BBN->getOpCode();
  if (isLogicalOpCode(OP))
    OP = IntBoolOpMap::rmap(OP);
  BO = static_cast<Instruction::BinaryOps>(OpCodeMap::rmap(OP));
  auto Inst = BinaryOperator::Create(BO, transValue(BBN->getOperand(0), F, BB),
                                     transValue(BBN->getOperand(1), F, BB),
                                     BV->getName(), BB);
  applyNoIntegerWrapDecorations(BV, Inst);
  applyFPFastMathModeDecorations(BV, Inst);
  return Inst;
}

Instruction *SPIRVToLLVM::transCmpInst(SPIRVValue *BV, BasicBlock *BB,
                                       Function *F) {
  SPIRVCompare *BC = static_cast<SPIRVCompare *>(BV);
  assert(BB && "Invalid BB");
  SPIRVType *BT = BC->getOperand(0)->getType();
  Instruction *Inst = nullptr;
  auto OP = BC->getOpCode();
  if (isLogicalOpCode(OP))
    OP = IntBoolOpMap::rmap(OP);
  if (BT->isTypeVectorOrScalarInt() || BT->isTypeVectorOrScalarBool() ||
      BT->isTypePointer())
    Inst = new ICmpInst(*BB, CmpMap::rmap(OP),
                        transValue(BC->getOperand(0), F, BB),
                        transValue(BC->getOperand(1), F, BB));
  else if (BT->isTypeVectorOrScalarFloat())
    Inst = new FCmpInst(*BB, CmpMap::rmap(OP),
                        transValue(BC->getOperand(0), F, BB),
                        transValue(BC->getOperand(1), F, BB));
  assert(Inst && "not implemented");
  return Inst;
}

bool SPIRVToLLVM::postProcessOCL() {
  StringRef DemangledName;
  SPIRVWord SrcLangVer = 0;
  BM->getSourceLanguage(&SrcLangVer);
  bool IsCpp = SrcLangVer == kOCLVer::CL21;
  for (auto I = M->begin(), E = M->end(); I != E;) {
    auto F = I++;
    if (F->hasName() && F->isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[postProcessOCL sret] " << *F << '\n');
      if (F->getReturnType()->isStructTy() &&
          oclIsBuiltin(F->getName(), DemangledName, IsCpp)) {
        if (!postProcessOCLBuiltinReturnStruct(&(*F)))
          return false;
      }
    }
  }
  for (auto I = M->begin(), E = M->end(); I != E;) {
    auto F = I++;
    if (F->hasName() && F->isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[postProcessOCL array arg] " << *F << '\n');
      if (hasArrayArg(&(*F)) &&
          oclIsBuiltin(F->getName(), DemangledName, IsCpp))
        if (!postProcessOCLBuiltinWithArrayArguments(&(*F), DemangledName))
          return false;
    }
  }
  return true;
}

bool SPIRVToLLVM::postProcessOCLBuiltinReturnStruct(Function *F) {
  std::string Name = F->getName().str();
  F->setName(Name + ".old");
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    if (auto CI = dyn_cast<CallInst>(*I++)) {
      auto ST = dyn_cast<StoreInst>(*(CI->user_begin()));
      assert(ST);
      std::vector<Type *> ArgTys;
      getFunctionTypeParameterTypes(F->getFunctionType(), ArgTys);
      ArgTys.insert(ArgTys.begin(),
                    PointerType::get(F->getReturnType(), SPIRAS_Private));
      auto NewF =
          getOrCreateFunction(M, Type::getVoidTy(*Context), ArgTys, Name);
      NewF->setCallingConv(F->getCallingConv());
      auto Args = getArguments(CI);
      Args.insert(Args.begin(), ST->getPointerOperand());
      auto NewCI = CallInst::Create(NewF, Args, CI->getName(), CI);
      NewCI->setCallingConv(CI->getCallingConv());
      ST->eraseFromParent();
      CI->eraseFromParent();
    }
  }
  F->eraseFromParent();
  return true;
}

bool SPIRVToLLVM::postProcessOCLBuiltinWithArrayArguments(
    Function *F, StringRef DemangledName) {
  LLVM_DEBUG(dbgs() << "[postProcessOCLBuiltinWithArrayArguments] " << *F
                    << '\n');
  auto Attrs = F->getAttributes();
  auto Name = F->getName();
  mutateFunction(
      F,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        auto FBegin =
            CI->getParent()->getParent()->begin()->getFirstInsertionPt();
        for (auto &I : Args) {
          auto T = I->getType();
          if (!T->isArrayTy())
            continue;
          auto Alloca = new AllocaInst(T, 0, "", &(*FBegin));
          new StoreInst(I, Alloca, false, CI);
          auto Zero =
              ConstantInt::getNullValue(Type::getInt32Ty(T->getContext()));
          Value *Index[] = {Zero, Zero};
          I = GetElementPtrInst::CreateInBounds(Alloca, Index, "", CI);
        }
        return Name.str();
      },
      nullptr, &Attrs);
  return true;
}

static char getTypeSuffix(Type *T) {
  char Suffix;

  Type *ST = T->getScalarType();
  if (ST->isHalfTy())
    Suffix = 'h';
  else if (ST->isFloatTy())
    Suffix = 'f';
  else
    Suffix = 'i';

  return Suffix;
}

// ToDo: Handle unsigned integer return type. May need spec change.
Instruction *SPIRVToLLVM::postProcessOCLReadImage(SPIRVInstruction *BI,
                                                  CallInst *CI,
                                                  const std::string &FuncName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  StringRef ImageTypeName;
  bool IsDepthImage = false;
  if (isOCLImageType(
          (cast<CallInst>(CI->getOperand(0)))->getArgOperand(0)->getType(),
          &ImageTypeName))
    IsDepthImage = ImageTypeName.contains("_depth_");
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, llvm::Type *&RetTy) {
        CallInst *CallSampledImg = cast<CallInst>(Args[0]);
        auto Img = CallSampledImg->getArgOperand(0);
        assert(isOCLImageType(Img->getType()));
        auto Sampler = CallSampledImg->getArgOperand(1);
        Args[0] = Img;
        Args.insert(Args.begin() + 1, Sampler);
        if (Args.size() > 4) {
          ConstantInt *ImOp = dyn_cast<ConstantInt>(Args[3]);
          ConstantFP *LodVal = dyn_cast<ConstantFP>(Args[4]);
          // Drop "Image Operands" argument.
          Args.erase(Args.begin() + 3, Args.begin() + 4);
          // If the image operand is LOD and its value is zero, drop it too.
          if (ImOp && LodVal && LodVal->isNullValue() &&
              ImOp->getZExtValue() == ImageOperandsMask::ImageOperandsLodMask)
            Args.erase(Args.begin() + 3, Args.end());
        }
        if (CallSampledImg->hasOneUse()) {
          CallSampledImg->replaceAllUsesWith(
              UndefValue::get(CallSampledImg->getType()));
          CallSampledImg->dropAllReferences();
          CallSampledImg->eraseFromParent();
        }
        Type *T = CI->getType();
        if (auto VT = dyn_cast<VectorType>(T))
          T = VT->getElementType();
        RetTy = IsDepthImage ? T : CI->getType();
        return std::string(kOCLBuiltinName::SampledReadImage) +
               getTypeSuffix(T);
      },
      [=](CallInst *NewCI) -> Instruction * {
        if (IsDepthImage)
          return InsertElementInst::Create(
              UndefValue::get(VectorType::get(NewCI->getType(), 4)), NewCI,
              getSizet(M, 0), "", NewCI->getParent());
        return NewCI;
      },
      &Attrs);
}

CallInst *
SPIRVToLLVM::postProcessOCLWriteImage(SPIRVInstruction *BI, CallInst *CI,
                                      const std::string &DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        llvm::Type *T = Args[2]->getType();
        if (Args.size() > 4) {
          ConstantInt *ImOp = dyn_cast<ConstantInt>(Args[3]);
          ConstantFP *LodVal = dyn_cast<ConstantFP>(Args[4]);
          // Drop "Image Operands" argument.
          Args.erase(Args.begin() + 3, Args.begin() + 4);
          // If the image operand is LOD and its value is zero, drop it too.
          if (ImOp && LodVal && LodVal->isNullValue() &&
              ImOp->getZExtValue() == ImageOperandsMask::ImageOperandsLodMask)
            Args.erase(Args.begin() + 3, Args.end());
          else
            std::swap(Args[2], Args[3]);
        }
        return std::string(kOCLBuiltinName::WriteImage) + getTypeSuffix(T);
      },
      &Attrs);
}

CallInst *SPIRVToLLVM::postProcessOCLBuildNDRange(SPIRVInstruction *BI,
                                                  CallInst *CI,
                                                  const std::string &FuncName) {
  assert(CI->getNumArgOperands() == 3);
  auto GWS = CI->getArgOperand(0);
  auto LWS = CI->getArgOperand(1);
  auto GWO = CI->getArgOperand(2);
  CI->setArgOperand(0, GWO);
  CI->setArgOperand(1, GWS);
  CI->setArgOperand(2, LWS);
  return CI;
}

Instruction *
SPIRVToLLVM::postProcessGroupAllAny(CallInst *CI,
                                    const std::string &DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, llvm::Type *&RetTy) {
        Type *Int32Ty = Type::getInt32Ty(*Context);
        RetTy = Int32Ty;
        Args[1] = CastInst::CreateZExtOrBitCast(Args[1], Int32Ty, "", CI);
        return DemangledName;
      },
      [=](CallInst *NewCI) -> Instruction * {
        Type *RetTy = Type::getInt1Ty(*Context);
        return CastInst::CreateTruncOrBitCast(NewCI, RetTy, "",
                                              NewCI->getNextNode());
      },
      &Attrs);
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

bool SPIRVToLLVM::isSPIRVBuiltinVariable(GlobalVariable *GV,
                                         SPIRVBuiltinVariableKind *Kind) {
  auto Loc = BuiltinGVMap.find(GV);
  if (Loc == BuiltinGVMap.end())
    return false;
  if (Kind)
    *Kind = Loc->second;
  return true;
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
  auto *SamplerT =
      getOrCreateOpaquePtrType(M, OCLOpaqueTypeOpCodeMap::rmap(OpTypeSampler),
                               getOCLOpaqueTypeAddrSpace(BCS->getOpCode()));
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
  auto CPSTy = M->getTypeByName(CPSName);
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
    case OpTypeInt:
      return mapValue(
          BV, ConstantInt::get(LT, ConstValue,
                               static_cast<SPIRVTypeInt *>(BT)->isSigned()));
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
    return mapValue(BV, oclTransConstantSampler(BCS, BB));
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

  case OpUndef:
    return mapValue(BV, UndefValue::get(transType(BV->getType())));

  case OpVariable: {
    auto BVar = static_cast<SPIRVVariable *>(BV);
    auto Ty = transType(BVar->getType()->getPointerElementType());
    bool IsConst = BVar->isConstant();
    llvm::GlobalValue::LinkageTypes LinkageTy = transLinkageType(BVar);
    Constant *Initializer = nullptr;
    SPIRVValue *Init = BVar->getInitializer();
    if (Init)
      Initializer = dyn_cast<Constant>(transValue(Init, F, BB, false));
    else if (LinkageTy == GlobalValue::CommonLinkage)
      // In LLVM variables with common linkage type must be initilized by 0
      Initializer = Constant::getNullValue(Ty);
    else if (BVar->getStorageClass() ==
             SPIRVStorageClassKind::StorageClassWorkgroup)
      Initializer = dyn_cast<Constant>(UndefValue::get(Ty));

    SPIRVStorageClassKind BS = BVar->getStorageClass();
    if (BS == StorageClassFunction && !Init) {
      assert(BB && "Invalid BB");
      return mapValue(BV, new AllocaInst(Ty, 0, BV->getName(), BB));
    }
    auto AddrSpace = SPIRSPIRVAddrSpaceMap::rmap(BS);
    auto LVar = new GlobalVariable(*M, Ty, IsConst, LinkageTy, Initializer,
                                   BV->getName(), 0,
                                   GlobalVariable::NotThreadLocal, AddrSpace);
    LVar->setUnnamedAddr((IsConst && Ty->isArrayTy() &&
                          Ty->getArrayElementType()->isIntegerTy(8))
                             ? GlobalValue::UnnamedAddr::Global
                             : GlobalValue::UnnamedAddr::None);
    SPIRVBuiltinVariableKind BVKind;
    if (BVar->isBuiltin(&BVKind))
      BuiltinGVMap[LVar] = BVKind;
    return mapValue(BV, LVar);
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
  switch (BV->getOpCode()) {
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
    bool isVolatile = BS->SPIRVMemoryAccess::isVolatile();
    uint64_t AlignValue = BS->SPIRVMemoryAccess::getAlignment();
    if (0 == AlignValue)
      SI = new StoreInst(Src, Dst, isVolatile, BB);
    else
      SI = new StoreInst(Src, Dst, isVolatile, Align(AlignValue), BB);
    if (BS->SPIRVMemoryAccess::isNonTemporal())
      transNonTemporalMetadata(SI);
    return mapValue(BV, SI);
  }

  case OpLoad: {
    SPIRVLoad *BL = static_cast<SPIRVLoad *>(BV);
    auto V = transValue(BL->getSrc(), F, BB);
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
    return mapValue(BV,
                    SelectInst::Create(transValue(BS->getCondition(), F, BB),
                                       transValue(BS->getTrueValue(), F, BB),
                                       transValue(BS->getFalseValue(), F, BB),
                                       BV->getName(), BB));
  }

  case OpVmeImageINTEL:
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
    auto *VecTy = cast<VectorType>(Vector->getType());
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

    auto *VecTy = cast<VectorType>(Vec->getType());
    VectorType *VTy = VectorType::get(VecTy->getElementType(), M);
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
    auto VecSize = cast<VectorType>(ColType)->getNumElements();
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
    VectorType *VTy =
        cast<VectorType>(cast<ArrayType>(Mat->getType())->getElementType());
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
    VectorType *V1Ty =
        cast<VectorType>(cast<ArrayType>(M1->getType())->getElementType());
    VectorType *V2Ty =
        cast<VectorType>(cast<ArrayType>(M2->getType())->getElementType());
    unsigned R1 = V1Ty->getNumElements();
    unsigned R2 = V2Ty->getNumElements();
    auto ETy = V1Ty->getElementType();

    (void)C1;
    assert(C1 == R2 && "Unmatched matrix");

    auto VTy = VectorType::get(ETy, R1);
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
    VectorType *ColTy =
        cast<VectorType>(cast<ArrayType>(Matrix->getType())->getElementType());
    unsigned RowNum = ColTy->getNumElements();

    auto VTy = VectorType::get(ColTy->getElementType(), ColNum);
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
    auto Index = transValue(AC->getIndices(), F, BB);
    if (!AC->hasPtrIndex())
      Index.insert(Index.begin(), getInt32(M, 0));
    auto IsInbound = AC->isInBounds();
    Value *V = nullptr;
    if (BB) {
      auto GEP =
          GetElementPtrInst::Create(nullptr, Base, Index, BV->getName(), BB);
      GEP->setIsInBounds(IsInbound);
      V = GEP;
    } else {
      V = ConstantExpr::getGetElementPtr(nullptr, dyn_cast<Constant>(Base),
                                         Index, IsInbound);
    }
    return mapValue(BV, V);
  }

  case OpCompositeConstruct: {
    auto CC = static_cast<SPIRVCompositeConstruct *>(BV);
    auto Constituents = transValue(CC->getConstituents(), F, BB);
    std::vector<Constant *> CV;
    for (const auto &I : Constituents) {
      CV.push_back(dyn_cast<Constant>(I));
    }
    switch (BV->getType()->getOpCode()) {
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
    default:
      llvm_unreachable("Unhandled type!");
    }
  }

  case OpCompositeExtract: {
    SPIRVCompositeExtract *CE = static_cast<SPIRVCompositeExtract *>(BV);
    if (CE->getComposite()->getType()->isTypeVector()) {
      assert(CE->getIndices().size() == 1 && "Invalid index");
      return mapValue(
          BV, ExtractElementInst::Create(
                  transValue(CE->getComposite(), F, BB),
                  ConstantInt::get(*Context, APInt(32, CE->getIndices()[0])),
                  BV->getName(), BB));
    }
    return mapValue(
        BV, ExtractValueInst::Create(transValue(CE->getComposite(), F, BB),
                                     CE->getIndices(), BV->getName(), BB));
  }

  case OpVectorExtractDynamic: {
    auto CE = static_cast<SPIRVVectorExtractDynamic *>(BV);
    return mapValue(
        BV, ExtractElementInst::Create(transValue(CE->getVector(), F, BB),
                                       transValue(CE->getIndex(), F, BB),
                                       BV->getName(), BB));
  }

  case OpCompositeInsert: {
    auto CI = static_cast<SPIRVCompositeInsert *>(BV);
    if (CI->getComposite()->getType()->isTypeVector()) {
      assert(CI->getIndices().size() == 1 && "Invalid index");
      return mapValue(
          BV, InsertElementInst::Create(
                  transValue(CI->getComposite(), F, BB),
                  transValue(CI->getObject(), F, BB),
                  ConstantInt::get(*Context, APInt(32, CI->getIndices()[0])),
                  BV->getName(), BB));
    }
    return mapValue(
        BV, InsertValueInst::Create(transValue(CI->getComposite(), F, BB),
                                    transValue(CI->getObject(), F, BB),
                                    CI->getIndices(), BV->getName(), BB));
  }

  case OpVectorInsertDynamic: {
    auto CI = static_cast<SPIRVVectorInsertDynamic *>(BV);
    return mapValue(
        BV, InsertElementInst::Create(transValue(CI->getVector(), F, BB),
                                      transValue(CI->getComponent(), F, BB),
                                      transValue(CI->getIndex(), F, BB),
                                      BV->getName(), BB));
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
    return mapValue(BV,
                    new ShuffleVectorInst(transValue(VS->getVector1(), F, BB),
                                          transValue(VS->getVector2(), F, BB),
                                          ConstantVector::get(Components),
                                          BV->getName(), BB));
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
    // Assuming we are calling a regular device function
    Call->setCallingConv(CallingConv::SPIR_FUNC);
    // Don't set attributes, because at translation time we don't know which
    // function exactly we are calling.
    return mapValue(BV, Call);
  }

  case OpFunctionPointerINTEL: {
    SPIRVFunctionPointerINTEL *BC =
        static_cast<SPIRVFunctionPointerINTEL *>(BV);
    SPIRVFunction *F = BC->getFunction();
    BV->setName(F->getName());
    return mapValue(BV, transFunction(F));
  }

  case OpAssumeTrueINTEL: {
    IRBuilder<> Builder(BB);
    SPIRVAssumeTrueINTEL *BC = static_cast<SPIRVAssumeTrueINTEL *>(BV);
    Value *Condition = transValue(BC->getCondition(), F, BB);
    return mapValue(BV, Builder.CreateAssumption(Condition));
  }

  case OpExpectINTEL: {
    IRBuilder<> Builder(BB);
    SPIRVExpectINTELInstBase *BC = static_cast<SPIRVExpectINTELInstBase *>(BV);
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
      return mapValue(BV, DbgTran->transDebugIntrinsic(ExtInst, BB));
    default:
      llvm_unreachable("Unknown extended instruction set!");
    }
  }

  case OpSNegate: {
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    auto Neg = BinaryOperator::CreateNeg(transValue(BC->getOperand(0), F, BB),
                                         BV->getName(), BB);
    applyNoIntegerWrapDecorations(BV, Neg);
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
    auto Dividend = transValue(FMod->getDividend(), F, BB);
    auto Divisor = transValue(FMod->getDivisor(), F, BB);
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
    IRBuilder<> Builder(BB);
    SPIRVSMod *SMod = static_cast<SPIRVSMod *>(BV);
    auto Dividend = transValue(SMod->getDividend(), F, BB);
    auto Divisor = transValue(SMod->getDivisor(), F, BB);
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
    SPIRVUnary *BC = static_cast<SPIRVUnary *>(BV);
    return mapValue(
        BV, BinaryOperator::CreateNot(transValue(BC->getOperand(0), F, BB),
                                      BV->getName(), BB));
  }

  case OpAll:
  case OpAny:
    return mapValue(BV,
                    transOCLAllAny(static_cast<SPIRVInstruction *>(BV), BB));

  case OpIsFinite:
  case OpIsInf:
  case OpIsNan:
  case OpIsNormal:
  case OpSignBitSet:
    return mapValue(
        BV, transOCLRelational(static_cast<SPIRVInstruction *>(BV), BB));
  case OpEnqueueKernel:
    return mapValue(
        BV, transEnqueueKernelBI(static_cast<SPIRVInstruction *>(BV), BB));
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
        if (PtrTy && isa<IntegerType>(PtrTy->getElementType()))
          RetTy = PtrTy;
        // Whether a struct or a pointer to some other type,
        // bitcast to i8*
        else {
          RetTy = Int8PtrTyPrivate;
          ValAsArg = Builder.CreateBitCast(Val, Int8PtrTyPrivate);
        }
      }
    }

    Value *Args[] = {ValAsArg, GS, UndefInt8Ptr, UndefInt32};
    auto *IntrinsicCall = Builder.CreateIntrinsic(IID, RetTy, Args);
    return mapValue(BV, IntrinsicCall);
  }
  default: {
    auto OC = BV->getOpCode();
    if (isSPIRVCmpInstTransToLLVMInst(static_cast<SPIRVInstruction *>(BV))) {
      return mapValue(BV, transCmpInst(BV, BB, F));
    } else if (isDirectlyTranslatedToOCL(OC)) {
      return mapValue(
          BV, transOCLBuiltinFromInst(static_cast<SPIRVInstruction *>(BV), BB));
    } else if (isBinaryShiftLogicalBitwiseOpCode(OC) || isLogicalOpCode(OC)) {
      return mapValue(BV, transShiftLogicalBitwiseInst(BV, BB, F));
    } else if (isCvtOpCode(OC)) {
      auto BI = static_cast<SPIRVInstruction *>(BV);
      Value *Inst = nullptr;
      if (BI->hasFPRoundingMode() || BI->isSaturatedConversion())
        Inst = transOCLBuiltinFromInst(BI, BB);
      else
        Inst = transConvertInst(BV, F, BB);
      return mapValue(BV, Inst);
    }
    return mapValue(
        BV, transSPIRVBuiltinFromInst(static_cast<SPIRVInstruction *>(BV), BB));
  }
  }
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

  auto IsKernel = BM->isEntryPoint(ExecutionModelKernel, BF->getId());
  auto Linkage = IsKernel ? GlobalValue::ExternalLinkage : transLinkageType(BF);
  FunctionType *FT = dyn_cast<FunctionType>(transType(BF->getFunctionType()));
  Function *F = cast<Function>(
      mapValue(BF, Function::Create(FT, Linkage, BF->getName(), M)));
  mapFunction(BF, F);

  if (BF->hasDecorate(DecorationReferencedIndirectlyINTEL))
    F->addFnAttr("referenced-indirectly");

  if (!F->isIntrinsic()) {
    F->setCallingConv(IsKernel ? CallingConv::SPIR_KERNEL
                               : CallingConv::SPIR_FUNC);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
    foreachFuncCtlMask(BF,
                       [&](Attribute::AttrKind Attr) { F->addFnAttr(Attr); });
  }

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto BA = BF->getArgument(I->getArgNo());
    mapValue(BA, &(*I));
    setName(&(*I), BA);
    BA->foreachAttr([&](SPIRVFuncParamAttrKind Kind) {
      if (Kind == FunctionParameterAttributeNoWrite)
        return;
      F->addAttribute(I->getArgNo() + 1, SPIRSPIRVFuncParamAttrMap::rmap(Kind));
    });

    SPIRVWord MaxOffset = 0;
    if (BA->hasDecorate(DecorationMaxByteOffset, 0, &MaxOffset)) {
      AttrBuilder Builder;
      Builder.addDereferenceableAttr(MaxOffset);
      I->addAttrs(Builder);
    }
  }
  BF->foreachReturnValueAttr([&](SPIRVFuncParamAttrKind Kind) {
    if (Kind == FunctionParameterAttributeNoWrite)
      return;
    F->addAttribute(AttributeList::ReturnIndex,
                    SPIRSPIRVFuncParamAttrMap::rmap(Kind));
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
  auto OC = BI->getOpCode();
  if (isCmpOpCode(BI->getOpCode())) {
    if (BT->isTypeBool())
      RetTy = IntegerType::getInt32Ty(*Context);
    else if (BT->isTypeVectorBool())
      RetTy = VectorType::get(
          IntegerType::get(
              *Context,
              Args[0]->getType()->getVectorComponentType()->getBitWidth()),
          BT->getVectorComponentCount());
    else
      llvm_unreachable("invalid compare instruction");
  } else if (OC == OpGenericCastToPtrExplicit)
    Args.pop_back();
  else if (OC == OpImageRead && Args.size() > 2) {
    // Drop "Image operands" argument
    Args.erase(Args.begin() + 2);
  } else if (isSubgroupAvcINTELEvaluateOpcode(OC)) {
    // There are three types of AVC Intel Evaluate opcodes:
    // 1. With multi reference images - does not use OpVmeImageINTEL opcode for
    // reference images
    // 2. With dual reference images - uses two OpVmeImageINTEL opcodes for
    // reference image
    // 3. With single reference image - uses one OpVmeImageINTEL opcode for
    // reference image
    int NumImages =
        std::count_if(Args.begin(), Args.end(), [](SPIRVValue *Arg) {
          return static_cast<SPIRVInstruction *>(Arg)->getOpCode() ==
                 OpVmeImageINTEL;
        });
    if (NumImages) {
      SPIRVInstruction *SrcImage = static_cast<SPIRVInstruction *>(Args[0]);
      assert(SrcImage &&
             "Src image operand not found in avc evaluate instruction");
      if (NumImages == 1) {
        // Multi reference opcode - remove src image OpVmeImageINTEL opcode
        // and replace it with corresponding OpImage and OpSampler arguments
        size_t SamplerPos = Args.size() - 1;
        Args.erase(Args.begin(), Args.begin() + 1);
        Args.insert(Args.begin(), SrcImage->getOperands()[0]);
        Args.insert(Args.begin() + SamplerPos, SrcImage->getOperands()[1]);
      } else {
        SPIRVInstruction *FwdRefImage =
            static_cast<SPIRVInstruction *>(Args[1]);
        SPIRVInstruction *BwdRefImage =
            static_cast<SPIRVInstruction *>(Args[2]);
        assert(FwdRefImage && "invalid avc evaluate instruction");
        // Single reference opcode - remove src and ref image OpVmeImageINTEL
        // opcodes and replace them with src and ref OpImage opcodes and
        // OpSampler
        Args.erase(Args.begin(), Args.begin() + NumImages);
        // insert source OpImage and OpSampler
        auto SrcOps = SrcImage->getOperands();
        Args.insert(Args.begin(), SrcOps.begin(), SrcOps.end());
        // insert reference OpImage
        Args.insert(Args.begin() + 1, FwdRefImage->getOperands()[0]);
        if (NumImages == 3) {
          // Dual reference opcode - insert second reference OpImage argument
          assert(BwdRefImage && "invalid avc evaluate instruction");
          Args.insert(Args.begin() + 2, BwdRefImage->getOperands()[0]);
        }
      }
    } else
      llvm_unreachable("invalid avc instruction");
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
  if (OC == OpImageSampleExplicitLod)
    return postProcessOCLReadImage(BI, CI, DemangledName);
  if (OC == OpImageWrite) {
    return postProcessOCLWriteImage(BI, CI, DemangledName);
  }
  if (OC == OpGenericPtrMemSemantics)
    return BinaryOperator::CreateShl(CI, getInt32(M, 8), "", BB);
  if (OC == OpImageQueryFormat)
    return BinaryOperator::CreateSub(
        CI, getInt32(M, OCLImageChannelDataTypeOffset), "", BB);
  if (OC == OpImageQueryOrder)
    return BinaryOperator::CreateSub(
        CI, getInt32(M, OCLImageChannelOrderOffset), "", BB);
  if (OC == OpBuildNDRange)
    return postProcessOCLBuildNDRange(BI, CI, DemangledName);
  if (OC == OpGroupAll || OC == OpGroupAny)
    return postProcessGroupAllAny(CI, DemangledName);
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

Instruction *SPIRVToLLVM::transEnqueueKernelBI(SPIRVInstruction *BI,
                                               BasicBlock *BB) {
  Type *Int32Ty = Type::getInt32Ty(*Context);
  Type *Int64Ty = Type::getInt64Ty(*Context);
  Type *IntTy =
      M->getDataLayout().getPointerSizeInBits(0) == 32 ? Int32Ty : Int64Ty;

  // Find or create enqueue kernel BI declaration
  auto Ops = BI->getOperands();
  bool HasVaargs = Ops.size() > 10;
  bool HasEvents = true;
  SPIRVValue *EventRet = Ops[5];
  if (EventRet->getOpCode() == OpConstantNull) {
    SPIRVValue *NumEvents = Ops[3];
    if (NumEvents->getOpCode() == OpConstant) {
      SPIRVConstant *NE = static_cast<SPIRVConstant *>(NumEvents);
      HasEvents = NE->getZExtIntValue() != 0;
    } else if (NumEvents->getOpCode() == OpConstantNull)
      HasEvents = false;
  }

  std::string FName = "";
  if (!HasVaargs && !HasEvents)
    FName = "__enqueue_kernel_basic";
  else if (!HasVaargs && HasEvents)
    FName = "__enqueue_kernel_basic_events";
  else if (HasVaargs && !HasEvents)
    FName = "__enqueue_kernel_varargs";
  else
    FName = "__enqueue_kernel_events_varargs";

  Function *F = M->getFunction(FName);
  if (!F) {
    SmallVector<Type *, 8> Tys = {
        transType(Ops[0]->getType()), // queue
        Int32Ty,                      // flags
        transType(Ops[2]->getType()), // ndrange
    };
    if (HasEvents) {
      Type *EventTy =
          PointerType::get(getOrCreateOpaquePtrType(
                               M, SPIR_TYPE_NAME_CLK_EVENT_T,
                               getOCLOpaqueTypeAddrSpace(OpTypeDeviceEvent)),
                           SPIRAS_Generic);

      Tys.push_back(Int32Ty);
      Tys.push_back(EventTy);
      Tys.push_back(EventTy);
    }

    Tys.push_back(Type::getInt8PtrTy(*Context, SPIRAS_Generic));
    Tys.push_back(Type::getInt8PtrTy(*Context, SPIRAS_Generic));

    if (HasVaargs) {
      // Number of block invoke arguments (local arguments)
      Tys.push_back(Int32Ty);
      // Array of sizes of block invoke arguments
      Tys.push_back(PointerType::get(IntTy, SPIRAS_Private));
    }

    FunctionType *FT = FunctionType::get(Int32Ty, Tys, false);
    F = Function::Create(FT, GlobalValue::ExternalLinkage, FName, M);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
  }

  // Create call to enqueue kernel BI
  SmallVector<Value *, 8> Args = {
      transValue(Ops[0], F, BB, false), // queue
      transValue(Ops[1], F, BB, false), // flags
      transValue(Ops[2], F, BB, false), // ndrange
  };

  if (HasEvents) {
    Args.push_back(transValue(Ops[3], F, BB, false)); // events number
    Args.push_back(transDeviceEvent(Ops[4], F, BB));  // event_wait_list
    Args.push_back(transDeviceEvent(Ops[5], F, BB));  // event_ret
  }

  Args.push_back(transBlockInvoke(Ops[6], BB));     // block_invoke
  Args.push_back(transValue(Ops[7], F, BB, false)); // block_literal

  if (HasVaargs) {
    // Number of local arguments
    Args.push_back(ConstantInt::get(Int32Ty, Ops.size() - 10));
    // GEP to array of sizes of local arguments
    if (Ops[10]->getOpCode() == OpPtrAccessChain)
      Args.push_back(transValue(Ops[10], F, BB, false));
    else
      llvm_unreachable("Not implemented");
  }
  auto Call = CallInst::Create(F, Args, "", BB);
  setName(Call, BI);
  setAttrByCalledFunc(Call);
  return Call;
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
  bool HasFuncPtrArg = false;
  for (auto &I : ArgTys) {
    if (isa<FunctionType>(I)) {
      I = PointerType::get(I, SPIRAS_Private);
      HasFuncPtrArg = true;
    }
  }
  if (!HasFuncPtrArg)
    mangleOpenClBuiltin(FuncName, ArgTys, MangledName);
  else
    MangledName = decorateSPIRVFunction(FuncName);
  Function *Func = M->getFunction(MangledName);
  FunctionType *FT = FunctionType::get(RetTy, ArgTys, false);
  // ToDo: Some intermediate functions have duplicate names with
  // different function types. This is OK if the function name
  // is used internally and finally translated to unique function
  // names. However it is better to have a way to differentiate
  // between intermidiate functions and final functions and make
  // sure final functions have unique names.
  SPIRVDBG(if (!HasFuncPtrArg && Func && Func->getFunctionType() != FT) {
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

std::string SPIRVToLLVM::getOCLBuiltinName(SPIRVInstruction *BI) {
  auto OC = BI->getOpCode();
  if (OC == OpGenericCastToPtrExplicit)
    return getOCLGenericCastToPtrName(BI);
  if (isCvtOpCode(OC))
    return getOCLConvertBuiltinName(BI);
  if (OC == OpBuildNDRange) {
    auto NDRangeInst = static_cast<SPIRVBuildNDRange *>(BI);
    auto EleTy = ((NDRangeInst->getOperands())[0])->getType();
    int Dim = EleTy->isTypeArray() ? EleTy->getArrayLength() : 1;
    // cygwin does not have std::to_string
    ostringstream OS;
    OS << Dim;
    assert((EleTy->isTypeInt() && Dim == 1) ||
           (EleTy->isTypeArray() && Dim >= 2 && Dim <= 3));
    return std::string(kOCLBuiltinName::NDRangePrefix) + OS.str() + "D";
  }
  if (isIntelSubgroupOpCode(OC)) {
    std::stringstream Name;
    SPIRVType *DataTy = nullptr;
    switch (OC) {
    case OpSubgroupBlockReadINTEL:
    case OpSubgroupImageBlockReadINTEL:
      Name << "intel_sub_group_block_read";
      DataTy = BI->getType();
      break;
    case OpSubgroupBlockWriteINTEL:
      Name << "intel_sub_group_block_write";
      DataTy = BI->getOperands()[1]->getType();
      break;
    case OpSubgroupImageBlockWriteINTEL:
      Name << "intel_sub_group_block_write";
      DataTy = BI->getOperands()[2]->getType();
      break;
    default:
      return OCLSPIRVBuiltinMap::rmap(OC);
    }
    assert(DataTy && "Intel subgroup block builtins should have data type");
    unsigned VectorNumElements = 1;
    if (DataTy->isTypeVector())
      VectorNumElements = DataTy->getVectorComponentCount();
    unsigned ElementBitSize = DataTy->getBitWidth();
    Name << getIntelSubgroupBlockDataPostfix(ElementBitSize, VectorNumElements);
    return Name.str();
  }
  if (isSubgroupAvcINTELInstructionOpCode(OC))
    return OCLSPIRVSubgroupAVCIntelBuiltinMap::rmap(OC);

  auto Name = OCLSPIRVBuiltinMap::rmap(OC);

  SPIRVType *T = nullptr;
  switch (OC) {
  case OpImageRead:
    T = BI->getType();
    break;
  case OpImageWrite:
    T = BI->getOperands()[2]->getType();
    break;
  default:
    // do nothing
    break;
  }
  if (T && T->isTypeVector())
    T = T->getVectorComponentType();
  if (T) {
    if (T->isTypeFloat(16))
      Name += 'h';
    else if (T->isTypeFloat(32))
      Name += 'f';
    else
      Name += 'i';
  }

  return Name;
}

Instruction *SPIRVToLLVM::transOCLBuiltinFromInst(SPIRVInstruction *BI,
                                                  BasicBlock *BB) {
  assert(BB && "Invalid BB");
  auto FuncName = getOCLBuiltinName(BI);
  return transBuiltinFromInst(FuncName, BI, BB);
}

Instruction *SPIRVToLLVM::transSPIRVBuiltinFromInst(SPIRVInstruction *BI,
                                                    BasicBlock *BB) {
  assert(BB && "Invalid BB");
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

  return transBuiltinFromInst(getSPIRVFuncName(BI->getOpCode(), Suffix), BI,
                              BB);
}

bool SPIRVToLLVM::translate() {
  if (!transAddressingModel())
    return false;

  for (unsigned I = 0, E = BM->getNumVariables(); I != E; ++I) {
    auto BV = BM->getVariable(I);
    if (BV->getStorageClass() != StorageClassFunction)
      transValue(BV, nullptr, nullptr);
  }
  transGlobalAnnotations();

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
  }

  if (!transKernelMetadata())
    return false;
  if (!transFPContractMetadata())
    return false;
  if (!transSourceLanguage())
    return false;
  if (!transSourceExtension())
    return false;
  transGeneratorMD();
  if (!transOCLBuiltinsFromVariables())
    return false;
  if (!postProcessOCL())
    return false;
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
  if (!BV->isVariable())
    return;

  if (auto AL = dyn_cast<AllocaInst>(V)) {
    IRBuilder<> Builder(AL->getParent());

    SPIRVType *ST = BV->getType()->getPointerElementType();

    Type *Int8PtrTyPrivate = Type::getInt8PtrTy(*Context, SPIRAS_Private);
    IntegerType *Int32Ty = IntegerType::get(*Context, 32);

    Value *UndefInt8Ptr = UndefValue::get(Int8PtrTyPrivate);
    Value *UndefInt32 = UndefValue::get(Int32Ty);

    if (ST->isTypeStruct()) {
      SPIRVTypeStruct *STS = static_cast<SPIRVTypeStruct *>(ST);

      for (SPIRVWord I = 0; I < STS->getMemberCount(); ++I) {
        SmallString<256> AnnotStr;
        generateIntelFPGAAnnotationForStructMember(ST, I, AnnotStr);
        if (!AnnotStr.empty()) {
          auto *GS = Builder.CreateGlobalStringPtr(AnnotStr);

          auto GEP = Builder.CreateConstInBoundsGEP2_32(AL->getAllocatedType(),
                                                        AL, 0, I);

          Type *IntTy = GEP->getType()->getPointerElementType()->isIntegerTy()
                            ? GEP->getType()
                            : Int8PtrTyPrivate;

          auto AnnotationFn = llvm::Intrinsic::getDeclaration(
              M, Intrinsic::ptr_annotation, IntTy);

          llvm::Value *Args[] = {
              Builder.CreateBitCast(GEP, IntTy, GEP->getName()),
              Builder.CreateBitCast(GS, Int8PtrTyPrivate), UndefInt8Ptr,
              UndefInt32};
          Builder.CreateCall(AnnotationFn, Args);
        }
      }
    }

    SmallString<256> AnnotStr;
    generateIntelFPGAAnnotation(BV, AnnotStr);
    if (!AnnotStr.empty()) {
      auto *GS = Builder.CreateGlobalStringPtr(AnnotStr);

      auto AnnotationFn =
          llvm::Intrinsic::getDeclaration(M, Intrinsic::var_annotation);

      llvm::Value *Args[] = {
          Builder.CreateBitCast(V, Int8PtrTyPrivate, V->getName()),
          Builder.CreateBitCast(GS, Int8PtrTyPrivate), UndefInt8Ptr,
          UndefInt32};
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

    llvm::Constant *Fields[4] = {
        C, ConstantExpr::getBitCast(GS, Int8PtrTyPrivate),
        UndefValue::get(Int8PtrTyPrivate), UndefValue::get(Int32Ty)};

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

bool SPIRVToLLVM::transDecoration(SPIRVValue *BV, Value *V) {
  if (!transAlign(BV, V))
    return false;

  transIntelFPGADecorations(BV, V);

  DbgTran->transDbgInfo(BV, V);
  return true;
}

bool SPIRVToLLVM::transFPContractMetadata() {
  bool ContractOff = false;
  for (unsigned I = 0, E = BM->getNumFunctions(); I != E; ++I) {
    SPIRVFunction *BF = BM->getFunction(I);
    if (!isOpenCLKernel(BF))
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
                                               Function *Kernel) {
  std::string ArgTypePrefix = std::string(SPIR_MD_KERNEL_ARG_TYPE) + "." +
                              Kernel->getName().str() + ".";
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

  Kernel->setMetadata(SPIR_MD_KERNEL_ARG_TYPE, MDNode::get(*Ctx, TypeMDs));
  return true;
}

bool SPIRVToLLVM::transKernelMetadata() {
  for (unsigned I = 0, E = BM->getNumFunctions(); I != E; ++I) {
    SPIRVFunction *BF = BM->getFunction(I);
    Function *F = static_cast<Function *>(getTranslatedValue(BF));
    assert(F && "Invalid translated function");
    if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Generate metadata for kernel_arg_address_spaces
    addOCLKernelArgumentMetadata(
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
    addOCLKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_ACCESS_QUAL, BF, F,
                                 [=](SPIRVFunctionParameter *Arg) {
                                   std::string Qual;
                                   auto T = Arg->getType();
                                   if (T->isTypeOCLImage()) {
                                     auto ST = static_cast<SPIRVTypeImage *>(T);
                                     Qual =
                                         transOCLImageTypeAccessQualifier(ST);
                                   } else if (T->isTypePipe()) {
                                     auto PT = static_cast<SPIRVTypePipe *>(T);
                                     Qual = transOCLPipeTypeAccessQualifier(PT);
                                   } else
                                     Qual = "none";
                                   return MDString::get(*Context, Qual);
                                 });
    // Generate metadata for kernel_arg_type
    if (!transKernelArgTypeMedataFromString(Context, BM, F))
      addOCLKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_TYPE, BF, F,
                                   [=](SPIRVFunctionParameter *Arg) {
                                     return transOCLKernelArgTypeName(Arg);
                                   });
    // Generate metadata for kernel_arg_type_qual
    addOCLKernelArgumentMetadata(
        Context, SPIR_MD_KERNEL_ARG_TYPE_QUAL, BF, F,
        [=](SPIRVFunctionParameter *Arg) {
          std::string Qual;
          if (Arg->hasDecorate(DecorationVolatile))
            Qual = kOCLTypeQualifierName::Volatile;
          Arg->foreachAttr([&](SPIRVFuncParamAttrKind Kind) {
            Qual += Qual.empty() ? "" : " ";
            switch (Kind) {
            case FunctionParameterAttributeNoAlias:
              Qual += kOCLTypeQualifierName::Restrict;
              break;
            case FunctionParameterAttributeNoWrite:
              Qual += kOCLTypeQualifierName::Const;
              break;
            default:
              // do nothing.
              break;
            }
          });
          if (Arg->getType()->isTypePipe()) {
            Qual += Qual.empty() ? "" : " ";
            Qual += kOCLTypeQualifierName::Pipe;
          }
          return MDString::get(*Context, Qual);
        });
    // Generate metadata for kernel_arg_base_type
    addOCLKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_BASE_TYPE, BF, F,
                                 [=](SPIRVFunctionParameter *Arg) {
                                   return transOCLKernelArgTypeName(Arg);
                                 });
    // Generate metadata for kernel_arg_name
    if (BM->isGenArgNameMDEnabled()) {
      addOCLKernelArgumentMetadata(Context, SPIR_MD_KERNEL_ARG_NAME, BF, F,
                                   [=](SPIRVFunctionParameter *Arg) {
                                     return MDString::get(*Context,
                                                          Arg->getName());
                                   });
    }
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

void SPIRVToLLVM::transOCLVectorLoadStore(std::string &UnmangledName,
                                          std::vector<SPIRVWord> &BArgs) {
  if (UnmangledName.find("vload") == 0 &&
      UnmangledName.find("n") != std::string::npos) {
    if (BArgs.back() != 1) {
      std::stringstream SS;
      SS << BArgs.back();
      UnmangledName.replace(UnmangledName.find("n"), 1, SS.str());
    } else {
      UnmangledName.erase(UnmangledName.find("n"), 1);
    }
    BArgs.pop_back();
  } else if (UnmangledName.find("vstore") == 0) {
    if (UnmangledName.find("n") != std::string::npos) {
      auto T = BM->getValueType(BArgs[0]);
      if (T->isTypeVector()) {
        auto W = T->getVectorComponentCount();
        std::stringstream SS;
        SS << W;
        UnmangledName.replace(UnmangledName.find("n"), 1, SS.str());
      } else {
        UnmangledName.erase(UnmangledName.find("n"), 1);
      }
    }
    if (UnmangledName.find("_r") != std::string::npos) {
      UnmangledName.replace(
          UnmangledName.find("_r"), 2,
          std::string("_") +
              SPIRSPIRVFPRoundingModeMap::rmap(
                  static_cast<SPIRVFPRoundingModeKind>(BArgs.back())));
      BArgs.pop_back();
    }
  }
}

// printf is not mangled. The function type should have just one argument.
// read_image*: the second argument should be mangled as sampler.
Instruction *SPIRVToLLVM::transOCLBuiltinFromExtInst(SPIRVExtInst *BC,
                                                     BasicBlock *BB) {
  assert(BB && "Invalid BB");
  std::string MangledName;
  SPIRVWord EntryPoint = BC->getExtOp();
  bool IsVarArg = false;
  bool IsPrintf = false;
  std::string UnmangledName;
  auto BArgs = BC->getArguments();

  assert(BM->getBuiltinSet(BC->getExtSetId()) == SPIRVEIS_OpenCL &&
         "Not OpenCL extended instruction");
  if (EntryPoint == OpenCLLIB::Printf)
    IsPrintf = true;
  else {
    UnmangledName = OCLExtOpMap::map(static_cast<OCLExtOpKind>(EntryPoint));
  }

  SPIRVDBG(spvdbgs() << "[transOCLBuiltinFromExtInst] OrigUnmangledName: "
                     << UnmangledName << '\n');
  transOCLVectorLoadStore(UnmangledName, BArgs);

  std::vector<Type *> ArgTypes = transTypeVector(BC->getValueTypes(BArgs));

  if (IsPrintf) {
    MangledName = "printf";
    IsVarArg = true;
    ArgTypes.resize(1);
  } else if (UnmangledName.find("read_image") == 0) {
    auto ModifiedArgTypes = ArgTypes;
    ModifiedArgTypes[1] = getOrCreateOpaquePtrType(M, "opencl.sampler_t");
    mangleOpenClBuiltin(UnmangledName, ModifiedArgTypes, MangledName);
  } else {
    mangleOpenClBuiltin(UnmangledName, ArgTypes, MangledName);
  }
  SPIRVDBG(spvdbgs() << "[transOCLBuiltinFromExtInst] ModifiedUnmangledName: "
                     << UnmangledName << " MangledName: " << MangledName
                     << '\n');

  FunctionType *FT =
      FunctionType::get(transType(BC->getType()), ArgTypes, IsVarArg);
  Function *F = M->getFunction(MangledName);
  if (!F) {
    F = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    F->setCallingConv(CallingConv::SPIR_FUNC);
    if (isFuncNoUnwind())
      F->addFnAttr(Attribute::NoUnwind);
  }
  auto Args = transValue(BC->getValues(BArgs), F, BB);
  SPIRVDBG(dbgs() << "[transOCLBuiltinFromExtInst] Function: " << *F
                  << ", Args: ";
           for (auto &I
                : Args) dbgs()
           << *I << ", ";
           dbgs() << '\n');
  CallInst *Call = CallInst::Create(F, Args, BC->getName(), BB);
  setCallingConv(Call);
  addFnAttr(Call, Attribute::NoUnwind);
  return transOCLBuiltinPostproc(BC, Call, BB, UnmangledName);
}

// SPIR-V only contains language version. Use OpenCL language version as
// SPIR version.
bool SPIRVToLLVM::transSourceLanguage() {
  SPIRVWord Ver = 0;
  SourceLanguage Lang = BM->getSourceLanguage(&Ver);
  assert((Lang == SourceLanguageUnknown || // Allow unknown for debug info test
          Lang == SourceLanguageOpenCL_C || Lang == SourceLanguageOpenCL_CPP) &&
         "Unsupported source language");
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
  return true;
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

// If the argument is unsigned return uconvert*, otherwise return convert*.
std::string SPIRVToLLVM::getOCLConvertBuiltinName(SPIRVInstruction *BI) {
  auto OC = BI->getOpCode();
  assert(isCvtOpCode(OC) && "Not convert instruction");
  auto U = static_cast<SPIRVUnary *>(BI);
  std::string Name;
  if (isCvtFromUnsignedOpCode(OC))
    Name = "u";
  Name += "convert_";
  Name += mapSPIRVTypeToOCLType(U->getType(), !isCvtToUnsignedOpCode(OC));
  SPIRVFPRoundingModeKind Rounding;
  if (U->isSaturatedConversion())
    Name += "_sat";
  if (U->hasFPRoundingMode(&Rounding)) {
    Name += "_";
    Name += SPIRSPIRVFPRoundingModeMap::rmap(Rounding);
  }
  return Name;
}

// Check Address Space of the Pointer Type
std::string SPIRVToLLVM::getOCLGenericCastToPtrName(SPIRVInstruction *BI) {
  auto GenericCastToPtrInst = BI->getType()->getPointerStorageClass();
  switch (GenericCastToPtrInst) {
  case StorageClassCrossWorkgroup:
    return std::string(kOCLBuiltinName::ToGlobal);
  case StorageClassWorkgroup:
    return std::string(kOCLBuiltinName::ToLocal);
  case StorageClassFunction:
    return std::string(kOCLBuiltinName::ToPrivate);
  default:
    llvm_unreachable("Invalid address space");
    return "";
  }
}

llvm::GlobalValue::LinkageTypes
SPIRVToLLVM::transLinkageType(const SPIRVValue *V) {
  if (V->getLinkageType() == LinkageTypeInternal) {
    return GlobalValue::InternalLinkage;
  } else if (V->getLinkageType() == LinkageTypeImport) {
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
  } else { // LinkageTypeExport
    if (V->getOpCode() == OpVariable) {
      if (static_cast<const SPIRVVariable *>(V)->getInitializer() == 0)
        // Tentative definition
        return GlobalValue::CommonLinkage;
    }
    return GlobalValue::ExternalLinkage;
  }
}

Instruction *SPIRVToLLVM::transOCLAllAny(SPIRVInstruction *I, BasicBlock *BB) {
  CallInst *CI = cast<CallInst>(transSPIRVBuiltinFromInst(I, BB));
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return cast<Instruction>(mapValue(
      I, mutateCallInstOCL(
             M, CI,
             [=](CallInst *, std::vector<Value *> &Args, llvm::Type *&RetTy) {
               Type *Int32Ty = Type::getInt32Ty(*Context);
               auto OldArg = CI->getOperand(0);
               auto NewArgTy = VectorType::get(
                   Int32Ty,
                   cast<VectorType>(OldArg->getType())->getNumElements());
               auto NewArg =
                   CastInst::CreateSExtOrBitCast(OldArg, NewArgTy, "", CI);
               Args[0] = NewArg;
               RetTy = Int32Ty;
               return CI->getCalledFunction()->getName().str();
             },
             [=](CallInst *NewCI) -> Instruction * {
               return CastInst::CreateTruncOrBitCast(
                   NewCI, Type::getInt1Ty(*Context), "", NewCI->getNextNode());
             },
             &Attrs)));
}

Instruction *SPIRVToLLVM::transOCLRelational(SPIRVInstruction *I,
                                             BasicBlock *BB) {
  CallInst *CI = cast<CallInst>(transSPIRVBuiltinFromInst(I, BB));
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return cast<Instruction>(mapValue(
      I, mutateCallInstOCL(
             M, CI,
             [=](CallInst *, std::vector<Value *> &Args, llvm::Type *&RetTy) {
               Type *IntTy = Type::getInt32Ty(*Context);
               RetTy = IntTy;
               if (CI->getType()->isVectorTy()) {
                 if (cast<VectorType>(CI->getOperand(0)->getType())
                         ->getElementType()
                         ->isDoubleTy())
                   IntTy = Type::getInt64Ty(*Context);
                 if (cast<VectorType>(CI->getOperand(0)->getType())
                         ->getElementType()
                         ->isHalfTy())
                   IntTy = Type::getInt16Ty(*Context);
                 RetTy = VectorType::get(
                     IntTy, cast<VectorType>(CI->getType())->getNumElements());
               }
               return CI->getCalledFunction()->getName().str();
             },
             [=](CallInst *NewCI) -> Instruction * {
               Type *RetTy = Type::getInt1Ty(*Context);
               if (NewCI->getType()->isVectorTy())
                 RetTy = VectorType::get(
                     Type::getInt1Ty(*Context),
                     cast<VectorType>(NewCI->getType())->getNumElements());
               return CastInst::CreateTruncOrBitCast(NewCI, RetTy, "",
                                                     NewCI->getNextNode());
             },
             &Attrs)));
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
  return !IS.fail();
}
