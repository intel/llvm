//===- SPIRVUtil.cpp - SPIR-V Utilities -------------------------*- C++ -*-===//
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
/// \file
///
/// This file defines utility classes and functions shared by SPIR-V
/// reader/writer.
///
//===----------------------------------------------------------------------===//

#include "FunctionDescriptor.h"
#include "ManglingUtils.h"
#include "NameMangleAPI.h"
#include "OCLUtil.h"
#include "ParameterType.h"
#include "SPIRVInternal.h"
#include "SPIRVMDWalker.h"
#include "libSPIRV/SPIRVDecorate.h"
#include "libSPIRV/SPIRVValue.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include <functional>
#include <sstream>

#define DEBUG_TYPE "spirv"

namespace SPIRV {

#ifdef _SPIRV_SUPPORT_TEXT_FMT
cl::opt<bool, true>
    UseTextFormat("spirv-text",
                  cl::desc("Use text format for SPIR-V for debugging purpose"),
                  cl::location(SPIRVUseTextFormat));
#endif

#ifdef _SPIRVDBG
cl::opt<bool, true> EnableDbgOutput("spirv-debug",
                                    cl::desc("Enable SPIR-V debug output"),
                                    cl::location(SPIRVDbgEnable));
#endif

bool isSupportedTriple(Triple T) { return T.isSPIR(); }

void addFnAttr(CallInst *Call, Attribute::AttrKind Attr) {
  Call->addAttribute(AttributeList::FunctionIndex, Attr);
}

void removeFnAttr(CallInst *Call, Attribute::AttrKind Attr) {
  Call->removeAttribute(AttributeList::FunctionIndex, Attr);
}

Value *removeCast(Value *V) {
  auto Cast = dyn_cast<ConstantExpr>(V);
  if (Cast && Cast->isCast()) {
    return removeCast(Cast->getOperand(0));
  }
  if (auto Cast = dyn_cast<CastInst>(V))
    return removeCast(Cast->getOperand(0));
  return V;
}

void saveLLVMModule(Module *M, const std::string &OutputFile) {
  std::error_code EC;
  ToolOutputFile Out(OutputFile.c_str(), EC, sys::fs::OF_None);
  if (EC) {
    SPIRVDBG(errs() << "Fails to open output file: " << EC.message();)
    return;
  }

  WriteBitcodeToFile(*M, Out.os());
  Out.keep();
}

std::string mapLLVMTypeToOCLType(const Type *Ty, bool Signed) {
  if (Ty->isHalfTy())
    return "half";
  if (Ty->isFloatTy())
    return "float";
  if (Ty->isDoubleTy())
    return "double";
  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    std::string SignPrefix;
    std::string Stem;
    if (!Signed)
      SignPrefix = "u";
    switch (IntTy->getIntegerBitWidth()) {
    case 8:
      Stem = "char";
      break;
    case 16:
      Stem = "short";
      break;
    case 32:
      Stem = "int";
      break;
    case 64:
      Stem = "long";
      break;
    default:
      Stem = "invalid_type";
      break;
    }
    return SignPrefix + Stem;
  }
  if (auto VecTy = dyn_cast<FixedVectorType>(Ty)) {
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getNumElements();
    std::stringstream Ss;
    Ss << mapLLVMTypeToOCLType(EleTy, Signed) << Size;
    return Ss.str();
  }
  return "invalid_type";
}

std::string mapSPIRVTypeToOCLType(SPIRVType *Ty, bool Signed) {
  if (Ty->isTypeFloat()) {
    auto W = Ty->getBitWidth();
    switch (W) {
    case 16:
      return "half";
    case 32:
      return "float";
    case 64:
      return "double";
    default:
      assert(0 && "Invalid floating pointer type");
      return std::string("float") + W + "_t";
    }
  }
  if (Ty->isTypeInt()) {
    std::string SignPrefix;
    std::string Stem;
    if (!Signed)
      SignPrefix = "u";
    auto W = Ty->getBitWidth();
    switch (W) {
    case 8:
      Stem = "char";
      break;
    case 16:
      Stem = "short";
      break;
    case 32:
      Stem = "int";
      break;
    case 64:
      Stem = "long";
      break;
    default:
      llvm_unreachable("Invalid integer type");
      Stem = std::string("int") + W + "_t";
      break;
    }
    return SignPrefix + Stem;
  }
  if (Ty->isTypeVector()) {
    auto EleTy = Ty->getVectorComponentType();
    auto Size = Ty->getVectorComponentCount();
    std::stringstream Ss;
    Ss << mapSPIRVTypeToOCLType(EleTy, Signed) << Size;
    return Ss.str();
  }
  llvm_unreachable("Invalid type");
  return "unknown_type";
}

PointerType *getOrCreateOpaquePtrType(Module *M, const std::string &Name,
                                      unsigned AddrSpace) {
  auto OpaqueType = StructType::getTypeByName(M->getContext(), Name);
  if (!OpaqueType)
    OpaqueType = StructType::create(M->getContext(), Name);
  return PointerType::get(OpaqueType, AddrSpace);
}

PointerType *getSamplerType(Module *M) {
  return getOrCreateOpaquePtrType(M, getSPIRVTypeName(kSPIRVTypeName::Sampler),
                                  SPIRAS_Constant);
}

PointerType *getPipeStorageType(Module *M) {
  return getOrCreateOpaquePtrType(
      M, getSPIRVTypeName(kSPIRVTypeName::PipeStorage), SPIRAS_Constant);
}

void getFunctionTypeParameterTypes(llvm::FunctionType *FT,
                                   std::vector<Type *> &ArgTys) {
  for (auto I = FT->param_begin(), E = FT->param_end(); I != E; ++I) {
    ArgTys.push_back(*I);
  }
}

bool isVoidFuncTy(FunctionType *FT) {
  return FT->getReturnType()->isVoidTy() && FT->getNumParams() == 0;
}

bool isPointerToOpaqueStructType(llvm::Type *Ty) {
  if (auto PT = dyn_cast<PointerType>(Ty))
    if (auto ST = dyn_cast<StructType>(PT->getElementType()))
      if (ST->isOpaque())
        return true;
  return false;
}

bool isPointerToOpaqueStructType(llvm::Type *Ty, const std::string &Name) {
  if (auto PT = dyn_cast<PointerType>(Ty))
    if (auto ST = dyn_cast<StructType>(PT->getElementType()))
      if (ST->isOpaque() && ST->getName() == Name)
        return true;
  return false;
}

bool isOCLImageType(llvm::Type *Ty, StringRef *Name) {
  if (auto PT = dyn_cast<PointerType>(Ty))
    if (auto ST = dyn_cast<StructType>(PT->getElementType()))
      if (ST->isOpaque()) {
        auto FullName = ST->getName();
        if (FullName.find(kSPR2TypeName::ImagePrefix) == 0) {
          if (Name)
            *Name = FullName.drop_front(strlen(kSPR2TypeName::OCLPrefix));
          return true;
        }
      }
  return false;
}

/// \param BaseTyName is the type Name as in spirv.BaseTyName.Postfixes
/// \param Postfix contains postfixes extracted from the SPIR-V image
///   type Name as spirv.BaseTyName.Postfixes.
bool isSPIRVType(llvm::Type *Ty, StringRef BaseTyName, StringRef *Postfix) {
  if (auto PT = dyn_cast<PointerType>(Ty))
    if (auto ST = dyn_cast<StructType>(PT->getElementType()))
      if (ST->isOpaque()) {
        auto FullName = ST->getName();
        std::string N =
            std::string(kSPIRVTypeName::PrefixAndDelim) + BaseTyName.str();
        if (FullName != N)
          N = N + kSPIRVTypeName::Delimiter;
        if (FullName.startswith(N)) {
          if (Postfix)
            *Postfix = FullName.drop_front(N.size());
          return true;
        }
      }
  return false;
}

Function *getOrCreateFunction(Module *M, Type *RetTy, ArrayRef<Type *> ArgTypes,
                              StringRef Name, BuiltinFuncMangleInfo *Mangle,
                              AttributeList *Attrs, bool TakeName) {
  std::string MangledName{Name};
  bool IsVarArg = false;
  if (Mangle) {
    MangledName = mangleBuiltin(Name, ArgTypes, Mangle);
    IsVarArg = 0 <= Mangle->getVarArg();
    if (IsVarArg)
      ArgTypes = ArgTypes.slice(0, Mangle->getVarArg());
  }
  FunctionType *FT = FunctionType::get(RetTy, ArgTypes, IsVarArg);
  Function *F = M->getFunction(MangledName);
  if (!TakeName && F && F->getFunctionType() != FT && Mangle != nullptr) {
    std::string S;
    raw_string_ostream SS(S);
    SS << "Error: Attempt to redefine function: " << *F << " => " << *FT
       << '\n';
    report_fatal_error(SS.str(), false);
  }
  if (!F || F->getFunctionType() != FT) {
    auto NewF =
        Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    if (F && TakeName) {
      NewF->takeName(F);
      LLVM_DEBUG(
          dbgs() << "[getOrCreateFunction] Warning: taking function Name\n");
    }
    if (NewF->getName() != MangledName) {
      LLVM_DEBUG(
          dbgs() << "[getOrCreateFunction] Warning: function Name changed\n");
    }
    LLVM_DEBUG(dbgs() << "[getOrCreateFunction] ";
               if (F) dbgs() << *F << " => "; dbgs() << *NewF << '\n';);
    F = NewF;
    F->setCallingConv(CallingConv::SPIR_FUNC);
    if (Attrs)
      F->setAttributes(*Attrs);
  }
  return F;
}

std::vector<Value *> getArguments(CallInst *CI, unsigned Start, unsigned End) {
  std::vector<Value *> Args;
  if (End == 0)
    End = CI->getNumArgOperands();
  for (; Start != End; ++Start) {
    Args.push_back(CI->getArgOperand(Start));
  }
  return Args;
}

uint64_t getArgAsInt(CallInst *CI, unsigned I) {
  return cast<ConstantInt>(CI->getArgOperand(I))->getZExtValue();
}

Scope getArgAsScope(CallInst *CI, unsigned I) {
  return static_cast<Scope>(getArgAsInt(CI, I));
}

Decoration getArgAsDecoration(CallInst *CI, unsigned I) {
  return static_cast<Decoration>(getArgAsInt(CI, I));
}

std::string decorateSPIRVFunction(const std::string &S) {
  return std::string(kSPIRVName::Prefix) + S + kSPIRVName::Postfix;
}

StringRef undecorateSPIRVFunction(StringRef S) {
  assert(S.find(kSPIRVName::Prefix) == 0);
  const size_t Start = strlen(kSPIRVName::Prefix);
  auto End = S.rfind(kSPIRVName::Postfix);
  return S.substr(Start, End - Start);
}

std::string prefixSPIRVName(const std::string &S) {
  return std::string(kSPIRVName::Prefix) + S;
}

StringRef dePrefixSPIRVName(StringRef R, SmallVectorImpl<StringRef> &Postfix) {
  const size_t Start = strlen(kSPIRVName::Prefix);
  if (!R.startswith(kSPIRVName::Prefix))
    return R;
  R = R.drop_front(Start);
  R.split(Postfix, "_", -1, false);
  auto Name = Postfix.front();
  Postfix.erase(Postfix.begin());
  return Name;
}

std::string getSPIRVFuncName(Op OC, StringRef PostFix) {
  return prefixSPIRVName(getName(OC) + PostFix.str());
}

std::string getSPIRVFuncName(Op OC, const Type *PRetTy, bool IsSigned) {
  return prefixSPIRVName(getName(OC) + kSPIRVPostfix::Divider +
                         getPostfixForReturnType(PRetTy, IsSigned));
}

std::string getSPIRVExtFuncName(SPIRVExtInstSetKind Set, unsigned ExtOp,
                                StringRef PostFix) {
  std::string ExtOpName;
  switch (Set) {
  default:
    llvm_unreachable("invalid extended instruction set");
    ExtOpName = "unknown";
    break;
  case SPIRVEIS_OpenCL:
    ExtOpName = getName(static_cast<OCLExtOpKind>(ExtOp));
    break;
  }
  return prefixSPIRVName(SPIRVExtSetShortNameMap::map(Set) + '_' + ExtOpName +
                         PostFix.str());
}

SPIRVDecorate *mapPostfixToDecorate(StringRef Postfix, SPIRVEntry *Target) {
  if (Postfix == kSPIRVPostfix::Sat)
    return new SPIRVDecorate(spv::DecorationSaturatedConversion, Target);

  if (Postfix.startswith(kSPIRVPostfix::Rt))
    return new SPIRVDecorate(spv::DecorationFPRoundingMode, Target,
                             map<SPIRVFPRoundingModeKind>(Postfix.str()));

  return nullptr;
}

SPIRVValue *addDecorations(SPIRVValue *Target,
                           const SmallVectorImpl<std::string> &Decs) {
  for (auto &I : Decs)
    if (auto Dec = mapPostfixToDecorate(I, Target))
      Target->addDecorate(Dec);
  return Target;
}

std::string getPostfix(Decoration Dec, unsigned Value) {
  switch (Dec) {
  default:
    llvm_unreachable("not implemented");
    return "unknown";
  case spv::DecorationSaturatedConversion:
    return kSPIRVPostfix::Sat;
  case spv::DecorationFPRoundingMode:
    return rmap<std::string>(static_cast<SPIRVFPRoundingModeKind>(Value));
  }
}

std::string getPostfixForReturnType(CallInst *CI, bool IsSigned) {
  return getPostfixForReturnType(CI->getType(), IsSigned);
}

std::string getPostfixForReturnType(const Type *PRetTy, bool IsSigned) {
  return std::string(kSPIRVPostfix::Return) +
         mapLLVMTypeToOCLType(PRetTy, IsSigned);
}

// Enqueue kernel, kernel query, pipe and address space cast built-ins
// are not mangled.
bool isNonMangledOCLBuiltin(StringRef Name) {
  if (!Name.startswith("__"))
    return false;

  return isEnqueueKernelBI(Name) || isKernelQueryBI(Name) ||
         isPipeOrAddressSpaceCastBI(Name.drop_front(2));
}

Op getSPIRVFuncOC(StringRef S, SmallVectorImpl<std::string> *Dec) {
  Op OC;
  SmallVector<StringRef, 2> Postfix;
  StringRef Name;
  if (!oclIsBuiltin(S, Name))
    Name = S;
  StringRef R(Name);
  if ((!Name.startswith(kSPIRVName::Prefix) && !isNonMangledOCLBuiltin(S)) ||
      !getByName(dePrefixSPIRVName(R, Postfix).str(), OC)) {
    return OpNop;
  }
  if (Dec)
    for (auto &I : Postfix)
      Dec->push_back(I.str());
  return OC;
}

bool getSPIRVBuiltin(const std::string &OrigName, spv::BuiltIn &B) {
  SmallVector<StringRef, 2> Postfix;
  StringRef R(OrigName);
  R = dePrefixSPIRVName(R, Postfix);
  assert(Postfix.empty() && "Invalid SPIR-V builtin Name");
  return getByName(R.str(), B);
}

// Demangled name is a substring of the name. The DemangledName is updated only
// if true is returned
bool oclIsBuiltin(StringRef Name, StringRef &DemangledName, bool IsCpp) {
  if (Name == "printf") {
    DemangledName = Name;
    return true;
  }
  if (isNonMangledOCLBuiltin(Name)) {
    DemangledName = Name.drop_front(2);
    return true;
  }
  if (!Name.startswith("_Z"))
    return false;
  // OpenCL C++ built-ins are declared in cl namespace.
  // TODO: consider using 'St' abbriviation for cl namespace mangling.
  // Similar to ::std:: in C++.
  if (IsCpp) {
    if (!Name.startswith("_ZN"))
      // Attempt to demangle as C. This is useful for "extern C" functions
      // that have manually mangled names.
      return false;
    // Skip CV and ref qualifiers.
    size_t NameSpaceStart = Name.find_first_not_of("rVKRO", 3);
    // All built-ins are in the ::cl:: namespace.
    if (Name.substr(NameSpaceStart, 11) != "2cl7__spirv")
      return false;
    size_t DemangledNameLenStart = NameSpaceStart + 11;
    size_t Start = Name.find_first_not_of("0123456789", DemangledNameLenStart);
    size_t Len = 0;
    Name.substr(DemangledNameLenStart, Start - DemangledNameLenStart)
        .getAsInteger(10, Len);
    DemangledName = Name.substr(Start, Len);
  } else {
    size_t Start = Name.find_first_not_of("0123456789", 2);
    size_t Len = 0;
    Name.substr(2, Start - 2).getAsInteger(10, Len);
    DemangledName = Name.substr(Start, Len);
  }
  return DemangledName.size() != 0;
}

// Check if a mangled type Name is unsigned
bool isMangledTypeUnsigned(char Mangled) {
  return Mangled == 'h'    /* uchar */
         || Mangled == 't' /* ushort */
         || Mangled == 'j' /* uint */
         || Mangled == 'm' /* ulong */;
}

// Check if a mangled type Name is signed
bool isMangledTypeSigned(char Mangled) {
  return Mangled == 'c'    /* char */
         || Mangled == 'a' /* signed char */
         || Mangled == 's' /* short */
         || Mangled == 'i' /* int */
         || Mangled == 'l' /* long */;
}

// Check if a mangled type Name is floating point (excludes half)
bool isMangledTypeFP(char Mangled) {
  return Mangled == 'f'     /* float */
         || Mangled == 'd'; /* double */
}

// Check if a mangled type Name is half
bool isMangledTypeHalf(std::string Mangled) {
  return Mangled == "Dh"; /* half */
}

void eraseSubstitutionFromMangledName(std::string &MangledName) {
  auto Len = MangledName.length();
  while (Len >= 2 && MangledName.substr(Len - 2, 2) == "S_") {
    Len -= 2;
    MangledName.erase(Len, 2);
  }
}

ParamType lastFuncParamType(StringRef MangledName) {
  std::string Copy(MangledName);
  eraseSubstitutionFromMangledName(Copy);
  char Mangled = Copy.back();
  std::string Mangled2 = Copy.substr(Copy.size() - 2);

  if (isMangledTypeFP(Mangled) || isMangledTypeHalf(Mangled2)) {
    return ParamType::FLOAT;
  } else if (isMangledTypeUnsigned(Mangled)) {
    return ParamType::UNSIGNED;
  } else if (isMangledTypeSigned(Mangled)) {
    return ParamType::SIGNED;
  }

  return ParamType::UNKNOWN;
}

// Check if the last argument is signed
bool isLastFuncParamSigned(StringRef MangledName) {
  return lastFuncParamType(MangledName) == ParamType::SIGNED;
}

// Check if a mangled function Name contains unsigned atomic type
bool containsUnsignedAtomicType(StringRef Name) {
  auto Loc = Name.find(kMangledName::AtomicPrefixIncoming);
  if (Loc == StringRef::npos)
    return false;
  return isMangledTypeUnsigned(
      Name[Loc + strlen(kMangledName::AtomicPrefixIncoming)]);
}

bool isFunctionPointerType(Type *T) {
  if (isa<PointerType>(T) && isa<FunctionType>(T->getPointerElementType())) {
    return true;
  }
  return false;
}

bool hasFunctionPointerArg(Function *F, Function::arg_iterator &AI) {
  AI = F->arg_begin();
  for (auto AE = F->arg_end(); AI != AE; ++AI) {
    LLVM_DEBUG(dbgs() << "[hasFuncPointerArg] " << *AI << '\n');
    if (isFunctionPointerType(AI->getType())) {
      return true;
    }
  }
  return false;
}

Constant *castToVoidFuncPtr(Function *F) {
  auto T = getVoidFuncPtrType(F->getParent());
  return ConstantExpr::getBitCast(F, T);
}

bool hasArrayArg(Function *F) {
  for (auto I = F->arg_begin(), E = F->arg_end(); I != E; ++I) {
    LLVM_DEBUG(dbgs() << "[hasArrayArg] " << *I << '\n');
    if (I->getType()->isArrayTy()) {
      return true;
    }
  }
  return false;
}

CallInst *mutateCallInst(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    BuiltinFuncMangleInfo *Mangle, AttributeList *Attrs, bool TakeFuncName) {
  LLVM_DEBUG(dbgs() << "[mutateCallInst] " << *CI);

  auto Args = getArguments(CI);
  auto NewName = ArgMutate(CI, Args);
  std::string InstName;
  if (!CI->getType()->isVoidTy() && CI->hasName()) {
    InstName = CI->getName().str();
    CI->setName(InstName + ".old");
  }
  auto NewCI = addCallInst(M, NewName, CI->getType(), Args, Attrs, CI, Mangle,
                           InstName, TakeFuncName);
  NewCI->setDebugLoc(CI->getDebugLoc());
  LLVM_DEBUG(dbgs() << " => " << *NewCI << '\n');
  CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
  return NewCI;
}

Instruction *mutateCallInst(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate,
    BuiltinFuncMangleInfo *Mangle, AttributeList *Attrs, bool TakeFuncName) {
  LLVM_DEBUG(dbgs() << "[mutateCallInst] " << *CI);

  auto Args = getArguments(CI);
  Type *RetTy = CI->getType();
  auto NewName = ArgMutate(CI, Args, RetTy);
  StringRef InstName = CI->getName();
  auto NewCI = addCallInst(M, NewName, RetTy, Args, Attrs, CI, Mangle, InstName,
                           TakeFuncName);
  auto NewI = RetMutate(NewCI);
  NewI->takeName(CI);
  NewI->setDebugLoc(CI->getDebugLoc());
  LLVM_DEBUG(dbgs() << " => " << *NewI << '\n');
  if (!CI->getType()->isVoidTy())
    CI->replaceAllUsesWith(NewI);
  CI->eraseFromParent();
  return NewI;
}

void mutateFunction(
    Function *F,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    BuiltinFuncMangleInfo *Mangle, AttributeList *Attrs, bool TakeFuncName) {
  auto M = F->getParent();
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    if (auto CI = dyn_cast<CallInst>(*I++))
      mutateCallInst(M, CI, ArgMutate, Mangle, Attrs, TakeFuncName);
  }
  if (F->use_empty())
    F->eraseFromParent();
}

CallInst *mutateCallInstSPIRV(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    AttributeList *Attrs) {
  BuiltinFuncMangleInfo BtnInfo;
  return mutateCallInst(M, CI, ArgMutate, &BtnInfo, Attrs);
}

Instruction *mutateCallInstSPIRV(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate, AttributeList *Attrs) {
  BuiltinFuncMangleInfo BtnInfo;
  return mutateCallInst(M, CI, ArgMutate, RetMutate, &BtnInfo, Attrs);
}

CallInst *addCallInst(Module *M, StringRef FuncName, Type *RetTy,
                      ArrayRef<Value *> Args, AttributeList *Attrs,
                      Instruction *Pos, BuiltinFuncMangleInfo *Mangle,
                      StringRef InstName, bool TakeFuncName) {

  auto F = getOrCreateFunction(M, RetTy, getTypes(Args), FuncName, Mangle,
                               Attrs, TakeFuncName);
  // Cannot assign a Name to void typed values
  auto CI = CallInst::Create(F, Args, RetTy->isVoidTy() ? "" : InstName, Pos);
  CI->setCallingConv(F->getCallingConv());
  CI->setAttributes(F->getAttributes());
  return CI;
}

CallInst *addCallInstSPIRV(Module *M, StringRef FuncName, Type *RetTy,
                           ArrayRef<Value *> Args, AttributeList *Attrs,
                           Instruction *Pos, StringRef InstName) {
  BuiltinFuncMangleInfo BtnInfo;
  return addCallInst(M, FuncName, RetTy, Args, Attrs, Pos, &BtnInfo, InstName);
}

bool isValidVectorSize(unsigned I) {
  return I == 2 || I == 3 || I == 4 || I == 8 || I == 16;
}

Value *addVector(Instruction *InsPos, ValueVecRange Range) {
  size_t VecSize = Range.second - Range.first;
  if (VecSize == 1)
    return *Range.first;
  assert(isValidVectorSize(VecSize) && "Invalid vector size");
  IRBuilder<> Builder(InsPos);
  auto Vec = Builder.CreateVectorSplat(VecSize, *Range.first);
  unsigned Index = 1;
  for (++Range.first; Range.first != Range.second; ++Range.first, ++Index)
    Vec = Builder.CreateInsertElement(
        Vec, *Range.first,
        ConstantInt::get(Type::getInt32Ty(InsPos->getContext()), Index, false));
  return Vec;
}

void makeVector(Instruction *InsPos, std::vector<Value *> &Ops,
                ValueVecRange Range) {
  auto Vec = addVector(InsPos, Range);
  Ops.erase(Range.first, Range.second);
  Ops.push_back(Vec);
}

void expandVector(Instruction *InsPos, std::vector<Value *> &Ops,
                  size_t VecPos) {
  auto Vec = Ops[VecPos];
  auto *VT = dyn_cast<FixedVectorType>(Vec->getType());
  if (!VT)
    return;
  size_t N = VT->getNumElements();
  IRBuilder<> Builder(InsPos);
  for (size_t I = 0; I != N; ++I)
    Ops.insert(Ops.begin() + VecPos + I,
               Builder.CreateExtractElement(
                   Vec, ConstantInt::get(Type::getInt32Ty(InsPos->getContext()),
                                         I, false)));
  Ops.erase(Ops.begin() + VecPos + N);
}

Constant *castToInt8Ptr(Constant *V, unsigned Addr = 0) {
  return ConstantExpr::getBitCast(V, Type::getInt8PtrTy(V->getContext(), Addr));
}

PointerType *getInt8PtrTy(PointerType *T) {
  return Type::getInt8PtrTy(T->getContext(), T->getAddressSpace());
}

Value *castToInt8Ptr(Value *V, Instruction *Pos) {
  return CastInst::CreatePointerCast(
      V, getInt8PtrTy(cast<PointerType>(V->getType())), "", Pos);
}

CallInst *addBlockBind(Module *M, Function *InvokeFunc, Value *BlkCtx,
                       Value *CtxLen, Value *CtxAlign, Instruction *InsPos,
                       StringRef InstName) {
  auto BlkTy =
      getOrCreateOpaquePtrType(M, SPIR_TYPE_NAME_BLOCK_T, SPIRAS_Private);
  auto &Ctx = M->getContext();
  Value *BlkArgs[] = {
      castToInt8Ptr(InvokeFunc),
      CtxLen ? CtxLen : UndefValue::get(Type::getInt32Ty(Ctx)),
      CtxAlign ? CtxAlign : UndefValue::get(Type::getInt32Ty(Ctx)),
      BlkCtx ? BlkCtx : UndefValue::get(Type::getInt8PtrTy(Ctx))};
  return addCallInst(M, SPIR_INTRINSIC_BLOCK_BIND, BlkTy, BlkArgs, nullptr,
                     InsPos, nullptr, InstName);
}

IntegerType *getSizetType(Module *M) {
  return IntegerType::getIntNTy(M->getContext(),
                                M->getDataLayout().getPointerSizeInBits(0));
}

Type *getVoidFuncType(Module *M) {
  return FunctionType::get(Type::getVoidTy(M->getContext()), false);
}

Type *getVoidFuncPtrType(Module *M, unsigned AddrSpace) {
  return PointerType::get(getVoidFuncType(M), AddrSpace);
}

ConstantInt *getInt64(Module *M, int64_t Value) {
  return ConstantInt::getSigned(Type::getInt64Ty(M->getContext()), Value);
}

ConstantInt *getUInt64(Module *M, uint64_t Value) {
  return ConstantInt::get(Type::getInt64Ty(M->getContext()), Value, false);
}

Constant *getFloat32(Module *M, float Value) {
  return ConstantFP::get(Type::getFloatTy(M->getContext()), Value);
}

ConstantInt *getInt32(Module *M, int Value) {
  return ConstantInt::get(Type::getInt32Ty(M->getContext()), Value, true);
}

ConstantInt *getUInt32(Module *M, unsigned Value) {
  return ConstantInt::get(Type::getInt32Ty(M->getContext()), Value, false);
}

ConstantInt *getInt(Module *M, int64_t Value) {
  return Value >> 32 ? getInt64(M, Value)
                     : getInt32(M, static_cast<int32_t>(Value));
}

ConstantInt *getUInt(Module *M, uint64_t Value) {
  return Value >> 32 ? getUInt64(M, Value)
                     : getUInt32(M, static_cast<uint32_t>(Value));
}

ConstantInt *getUInt16(Module *M, unsigned short Value) {
  return ConstantInt::get(Type::getInt16Ty(M->getContext()), Value, false);
}

std::vector<Value *> getInt32(Module *M, const std::vector<int> &Values) {
  std::vector<Value *> V;
  for (auto &I : Values)
    V.push_back(getInt32(M, I));
  return V;
}

ConstantInt *getSizet(Module *M, uint64_t Value) {
  return ConstantInt::get(getSizetType(M), Value, false);
}

///////////////////////////////////////////////////////////////////////////////
//
// Functions for getting metadata
//
///////////////////////////////////////////////////////////////////////////////
int getMDOperandAsInt(MDNode *N, unsigned I) {
  return mdconst::dyn_extract<ConstantInt>(N->getOperand(I))->getZExtValue();
}

// Additional helper function to be reused by getMDOperandAs* helpers
Metadata *getMDOperandOrNull(MDNode *N, unsigned I) {
  if (!N)
    return nullptr;
  return N->getOperand(I);
}

std::string getMDOperandAsString(MDNode *N, unsigned I) {
  if (auto *Str = dyn_cast_or_null<MDString>(getMDOperandOrNull(N, I)))
    return Str->getString().str();
  return "";
}

MDNode *getMDOperandAsMDNode(MDNode *N, unsigned I) {
  return dyn_cast_or_null<MDNode>(getMDOperandOrNull(N, I));
}

Type *getMDOperandAsType(MDNode *N, unsigned I) {
  return cast<ValueAsMetadata>(N->getOperand(I))->getType();
}

std::set<std::string> getNamedMDAsStringSet(Module *M,
                                            const std::string &MDName) {
  NamedMDNode *NamedMD = M->getNamedMetadata(MDName);
  std::set<std::string> StrSet;
  if (!NamedMD)
    return StrSet;

  assert(NamedMD->getNumOperands() > 0 && "Invalid SPIR");

  for (unsigned I = 0, E = NamedMD->getNumOperands(); I != E; ++I) {
    MDNode *MD = NamedMD->getOperand(I);
    if (!MD || MD->getNumOperands() == 0)
      continue;
    for (unsigned J = 0, N = MD->getNumOperands(); J != N; ++J)
      StrSet.insert(getMDOperandAsString(MD, J));
  }

  return StrSet;
}

std::tuple<unsigned, unsigned, std::string> getSPIRVSource(Module *M) {
  std::tuple<unsigned, unsigned, std::string> Tup;
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::Source).nextOp())
    N.get(std::get<0>(Tup))
        .get(std::get<1>(Tup))
        .setQuiet(true)
        .get(std::get<2>(Tup));
  return Tup;
}

ConstantInt *mapUInt(Module *M, ConstantInt *I,
                     std::function<unsigned(unsigned)> F) {
  return ConstantInt::get(I->getType(), F(I->getZExtValue()), false);
}

ConstantInt *mapSInt(Module *M, ConstantInt *I, std::function<int(int)> F) {
  return ConstantInt::get(I->getType(), F(I->getSExtValue()), true);
}

bool isDecoratedSPIRVFunc(const Function *F, StringRef &UndecoratedName) {
  if (!F->hasName() || !F->getName().startswith(kSPIRVName::Prefix))
    return false;
  UndecoratedName = F->getName();
  return true;
}

/// Get TypePrimitiveEnum for special OpenCL type except opencl.block.
SPIR::TypePrimitiveEnum getOCLTypePrimitiveEnum(StringRef TyName) {
  return StringSwitch<SPIR::TypePrimitiveEnum>(TyName)
      .Case("opencl.image1d_ro_t", SPIR::PRIMITIVE_IMAGE1D_RO_T)
      .Case("opencl.image1d_array_ro_t", SPIR::PRIMITIVE_IMAGE1D_ARRAY_RO_T)
      .Case("opencl.image1d_buffer_ro_t", SPIR::PRIMITIVE_IMAGE1D_BUFFER_RO_T)
      .Case("opencl.image2d_ro_t", SPIR::PRIMITIVE_IMAGE2D_RO_T)
      .Case("opencl.image2d_array_ro_t", SPIR::PRIMITIVE_IMAGE2D_ARRAY_RO_T)
      .Case("opencl.image2d_depth_ro_t", SPIR::PRIMITIVE_IMAGE2D_DEPTH_RO_T)
      .Case("opencl.image2d_array_depth_ro_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T)
      .Case("opencl.image2d_msaa_ro_t", SPIR::PRIMITIVE_IMAGE2D_MSAA_RO_T)
      .Case("opencl.image2d_array_msaa_ro_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T)
      .Case("opencl.image2d_msaa_depth_ro_t",
            SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T)
      .Case("opencl.image2d_array_msaa_depth_ro_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T)
      .Case("opencl.image3d_ro_t", SPIR::PRIMITIVE_IMAGE3D_RO_T)
      .Case("opencl.image1d_wo_t", SPIR::PRIMITIVE_IMAGE1D_WO_T)
      .Case("opencl.image1d_array_wo_t", SPIR::PRIMITIVE_IMAGE1D_ARRAY_WO_T)
      .Case("opencl.image1d_buffer_wo_t", SPIR::PRIMITIVE_IMAGE1D_BUFFER_WO_T)
      .Case("opencl.image2d_wo_t", SPIR::PRIMITIVE_IMAGE2D_WO_T)
      .Case("opencl.image2d_array_wo_t", SPIR::PRIMITIVE_IMAGE2D_ARRAY_WO_T)
      .Case("opencl.image2d_depth_wo_t", SPIR::PRIMITIVE_IMAGE2D_DEPTH_WO_T)
      .Case("opencl.image2d_array_depth_wo_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T)
      .Case("opencl.image2d_msaa_wo_t", SPIR::PRIMITIVE_IMAGE2D_MSAA_WO_T)
      .Case("opencl.image2d_array_msaa_wo_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T)
      .Case("opencl.image2d_msaa_depth_wo_t",
            SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T)
      .Case("opencl.image2d_array_msaa_depth_wo_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T)
      .Case("opencl.image3d_wo_t", SPIR::PRIMITIVE_IMAGE3D_WO_T)
      .Case("opencl.image1d_rw_t", SPIR::PRIMITIVE_IMAGE1D_RW_T)
      .Case("opencl.image1d_array_rw_t", SPIR::PRIMITIVE_IMAGE1D_ARRAY_RW_T)
      .Case("opencl.image1d_buffer_rw_t", SPIR::PRIMITIVE_IMAGE1D_BUFFER_RW_T)
      .Case("opencl.image2d_rw_t", SPIR::PRIMITIVE_IMAGE2D_RW_T)
      .Case("opencl.image2d_array_rw_t", SPIR::PRIMITIVE_IMAGE2D_ARRAY_RW_T)
      .Case("opencl.image2d_depth_rw_t", SPIR::PRIMITIVE_IMAGE2D_DEPTH_RW_T)
      .Case("opencl.image2d_array_depth_rw_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T)
      .Case("opencl.image2d_msaa_rw_t", SPIR::PRIMITIVE_IMAGE2D_MSAA_RW_T)
      .Case("opencl.image2d_array_msaa_rw_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T)
      .Case("opencl.image2d_msaa_depth_rw_t",
            SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T)
      .Case("opencl.image2d_array_msaa_depth_rw_t",
            SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T)
      .Case("opencl.image3d_rw_t", SPIR::PRIMITIVE_IMAGE3D_RW_T)
      .Case("opencl.event_t", SPIR::PRIMITIVE_EVENT_T)
      .Case("opencl.pipe_ro_t", SPIR::PRIMITIVE_PIPE_RO_T)
      .Case("opencl.pipe_wo_t", SPIR::PRIMITIVE_PIPE_WO_T)
      .Case("opencl.reserve_id_t", SPIR::PRIMITIVE_RESERVE_ID_T)
      .Case("opencl.queue_t", SPIR::PRIMITIVE_QUEUE_T)
      .Case("opencl.clk_event_t", SPIR::PRIMITIVE_CLK_EVENT_T)
      .Case("opencl.sampler_t", SPIR::PRIMITIVE_SAMPLER_T)
      .Case("struct.ndrange_t", SPIR::PRIMITIVE_NDRANGE_T)
      .Case("opencl.intel_sub_group_avc_mce_payload_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_MCE_PAYLOAD_T)
      .Case("opencl.intel_sub_group_avc_ime_payload_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_PAYLOAD_T)
      .Case("opencl.intel_sub_group_avc_ref_payload_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_REF_PAYLOAD_T)
      .Case("opencl.intel_sub_group_avc_sic_payload_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_SIC_PAYLOAD_T)
      .Case("opencl.intel_sub_group_avc_mce_result_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_MCE_RESULT_T)
      .Case("opencl.intel_sub_group_avc_ime_result_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_RESULT_T)
      .Case("opencl.intel_sub_group_avc_ref_result_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_REF_RESULT_T)
      .Case("opencl.intel_sub_group_avc_sic_result_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_SIC_RESULT_T)
      .Case(
          "opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t",
          SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_SINGLE_REF_STREAMOUT_T)
      .Case("opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_DUAL_REF_STREAMOUT_T)
      .Case("opencl.intel_sub_group_avc_ime_single_reference_streamin_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_SINGLE_REF_STREAMIN_T)
      .Case("opencl.intel_sub_group_avc_ime_dual_reference_streamin_t",
            SPIR::PRIMITIVE_SUB_GROUP_AVC_IME_DUAL_REF_STREAMIN_T)
      .Default(SPIR::PRIMITIVE_NONE);
}
/// Translates LLVM type to descriptor for mangler.
/// \param Signed indicates integer type should be translated as signed.
/// \param VoidPtr indicates i8* should be translated as void*.
static SPIR::RefParamType transTypeDesc(Type *Ty,
                                        const BuiltinArgTypeMangleInfo &Info) {
  bool Signed = Info.IsSigned;
  unsigned Attr = Info.Attr;
  bool VoidPtr = Info.IsVoidPtr;
  if (Info.IsEnum)
    return SPIR::RefParamType(new SPIR::PrimitiveType(Info.Enum));
  if (Info.IsSampler)
    return SPIR::RefParamType(
        new SPIR::PrimitiveType(SPIR::PRIMITIVE_SAMPLER_T));
  if (Info.IsAtomic && !Ty->isPointerTy()) {
    BuiltinArgTypeMangleInfo DTInfo = Info;
    DTInfo.IsAtomic = false;
    return SPIR::RefParamType(new SPIR::AtomicType(transTypeDesc(Ty, DTInfo)));
  }
  if (auto *IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getBitWidth()) {
    case 1:
      return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_BOOL));
    case 8:
      return SPIR::RefParamType(new SPIR::PrimitiveType(
          Signed ? SPIR::PRIMITIVE_CHAR : SPIR::PRIMITIVE_UCHAR));
    case 16:
      return SPIR::RefParamType(new SPIR::PrimitiveType(
          Signed ? SPIR::PRIMITIVE_SHORT : SPIR::PRIMITIVE_USHORT));
    case 32:
      return SPIR::RefParamType(new SPIR::PrimitiveType(
          Signed ? SPIR::PRIMITIVE_INT : SPIR::PRIMITIVE_UINT));
    case 64:
      return SPIR::RefParamType(new SPIR::PrimitiveType(
          Signed ? SPIR::PRIMITIVE_LONG : SPIR::PRIMITIVE_ULONG));
    default:
      llvm_unreachable("invliad int size");
    }
  }
  if (Ty->isVoidTy())
    return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_VOID));
  if (Ty->isHalfTy())
    return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_HALF));
  if (Ty->isFloatTy())
    return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_FLOAT));
  if (Ty->isDoubleTy())
    return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_DOUBLE));
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    return SPIR::RefParamType(new SPIR::VectorType(
        transTypeDesc(VecTy->getElementType(), Info), VecTy->getNumElements()));
  }
  if (Ty->isArrayTy()) {
    return transTypeDesc(PointerType::get(Ty->getArrayElementType(), 0), Info);
  }
  if (Ty->isStructTy()) {
    auto Name = Ty->getStructName();
    std::string Tmp;

    if (Name.startswith(kLLVMTypeName::StructPrefix))
      Name = Name.drop_front(strlen(kLLVMTypeName::StructPrefix));
    if (Name.startswith(kSPIRVTypeName::PrefixAndDelim)) {
      Name = Name.substr(sizeof(kSPIRVTypeName::PrefixAndDelim) - 1);
      Tmp = Name.str();
      auto Pos = Tmp.find(kSPIRVTypeName::Delimiter); // first dot
      while (Pos != std::string::npos) {
        Tmp[Pos] = '_';
        Pos = Tmp.find(kSPIRVTypeName::Delimiter, Pos);
      }
      Name = Tmp = kSPIRVName::Prefix + Tmp;
    }
    // ToDo: Create a better unique Name for struct without Name
    if (Name.empty()) {
      std::ostringstream OS;
      OS << reinterpret_cast<size_t>(Ty);
      Name = Tmp = std::string("struct_") + OS.str();
    }
    return SPIR::RefParamType(new SPIR::UserDefinedType(Name.str()));
  }

  if (Ty->isPointerTy()) {
    auto ET = Ty->getPointerElementType();
    SPIR::ParamType *EPT = nullptr;
    if (isa<FunctionType>(ET)) {
      assert(isVoidFuncTy(cast<FunctionType>(ET)) && "Not supported");
      EPT = new SPIR::BlockType;
    } else if (auto StructTy = dyn_cast<StructType>(ET)) {
      LLVM_DEBUG(dbgs() << "ptr to struct: " << *Ty << '\n');
      auto TyName = StructTy->getStructName();
      if (TyName.startswith(kSPR2TypeName::OCLPrefix)) {
        auto DelimPos = TyName.find_first_of(kSPR2TypeName::Delimiter,
                                             strlen(kSPR2TypeName::OCLPrefix));
        if (DelimPos != StringRef::npos)
          TyName = TyName.substr(0, DelimPos);
      }
      LLVM_DEBUG(dbgs() << "  type Name: " << TyName << '\n');

      auto Prim = getOCLTypePrimitiveEnum(TyName);
      if (StructTy->isOpaque()) {
        if (TyName == "opencl.block") {
          auto BlockTy = new SPIR::BlockType;
          // Handle block with local memory arguments according to OpenCL 2.0
          // spec.
          if (Info.IsLocalArgBlock) {
            SPIR::RefParamType VoidTyRef(
                new SPIR::PrimitiveType(SPIR::PRIMITIVE_VOID));
            auto VoidPtrTy = new SPIR::PointerType(VoidTyRef);
            VoidPtrTy->setAddressSpace(SPIR::ATTR_LOCAL);
            // "__local void *"
            BlockTy->setParam(0, SPIR::RefParamType(VoidPtrTy));
            // "..."
            BlockTy->setParam(1, SPIR::RefParamType(new SPIR::PrimitiveType(
                                     SPIR::PRIMITIVE_VAR_ARG)));
          }
          EPT = BlockTy;
        } else if (Prim != SPIR::PRIMITIVE_NONE) {
          if (Prim == SPIR::PRIMITIVE_PIPE_RO_T ||
              Prim == SPIR::PRIMITIVE_PIPE_WO_T) {
            SPIR::RefParamType OpaqueTyRef(new SPIR::PrimitiveType(Prim));
            auto OpaquePtrTy = new SPIR::PointerType(OpaqueTyRef);
            OpaquePtrTy->setAddressSpace(getOCLOpaqueTypeAddrSpace(Prim));
            EPT = OpaquePtrTy;
          } else {
            EPT = new SPIR::PrimitiveType(Prim);
          }
        }
      } else if (Prim == SPIR::PRIMITIVE_NDRANGE_T)
        // ndrange_t is not opaque type
        EPT = new SPIR::PrimitiveType(SPIR::PRIMITIVE_NDRANGE_T);
    }
    if (EPT)
      return SPIR::RefParamType(EPT);

    if (VoidPtr && ET->isIntegerTy(8))
      ET = Type::getVoidTy(ET->getContext());
    auto PT = new SPIR::PointerType(transTypeDesc(ET, Info));
    PT->setAddressSpace(static_cast<SPIR::TypeAttributeEnum>(
        Ty->getPointerAddressSpace() + (unsigned)SPIR::ATTR_ADDR_SPACE_FIRST));
    for (unsigned I = SPIR::ATTR_QUALIFIER_FIRST, E = SPIR::ATTR_QUALIFIER_LAST;
         I <= E; ++I)
      PT->setQualifier(static_cast<SPIR::TypeAttributeEnum>(I), I & Attr);
    return SPIR::RefParamType(PT);
  }
  LLVM_DEBUG(dbgs() << "[transTypeDesc] " << *Ty << '\n');
  assert(0 && "not implemented");
  return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_INT));
}

Value *getScalarOrArray(Value *V, unsigned Size, Instruction *Pos) {
  if (!V->getType()->isPointerTy())
    return V;
  auto GEP = cast<GEPOperator>(V);
  assert(GEP->getNumOperands() == 3 && "must be a GEP from an array");
  assert(GEP->getSourceElementType()->getArrayNumElements() == Size);
  assert(dyn_cast<ConstantInt>(GEP->getOperand(1))->getZExtValue() == 0);
  assert(dyn_cast<ConstantInt>(GEP->getOperand(2))->getZExtValue() == 0);
  return new LoadInst(GEP->getSourceElementType(), GEP->getOperand(0), "", Pos);
}

Constant *getScalarOrVectorConstantInt(Type *T, uint64_t V, bool IsSigned) {
  if (auto IT = dyn_cast<IntegerType>(T))
    return ConstantInt::get(IT, V);
  if (auto VT = dyn_cast<FixedVectorType>(T)) {
    std::vector<Constant *> EV(
        VT->getNumElements(),
        getScalarOrVectorConstantInt(VT->getElementType(), V, IsSigned));
    return ConstantVector::get(EV);
  }
  llvm_unreachable("Invalid type");
  return nullptr;
}

Value *getScalarOrArrayConstantInt(Instruction *Pos, Type *T, unsigned Len,
                                   uint64_t V, bool IsSigned) {
  if (auto IT = dyn_cast<IntegerType>(T)) {
    assert(Len == 1 && "Invalid length");
    return ConstantInt::get(IT, V, IsSigned);
  }
  if (auto PT = dyn_cast<PointerType>(T)) {
    auto ET = PT->getPointerElementType();
    auto AT = ArrayType::get(ET, Len);
    std::vector<Constant *> EV(Len, ConstantInt::get(ET, V, IsSigned));
    auto CA = ConstantArray::get(AT, EV);
    auto Alloca = new AllocaInst(AT, 0, "", Pos);
    new StoreInst(CA, Alloca, Pos);
    auto Zero = ConstantInt::getNullValue(Type::getInt32Ty(T->getContext()));
    Value *Index[] = {Zero, Zero};
    auto Ret = GetElementPtrInst::CreateInBounds(Alloca, Index, "", Pos);
    LLVM_DEBUG(dbgs() << "[getScalarOrArrayConstantInt] Alloca: " << *Alloca
                      << ", Return: " << *Ret << '\n');
    return Ret;
  }
  if (auto AT = dyn_cast<ArrayType>(T)) {
    auto ET = AT->getArrayElementType();
    assert(AT->getArrayNumElements() == Len);
    std::vector<Constant *> EV(Len, ConstantInt::get(ET, V, IsSigned));
    auto Ret = ConstantArray::get(AT, EV);
    LLVM_DEBUG(dbgs() << "[getScalarOrArrayConstantInt] Array type: " << *AT
                      << ", Return: " << *Ret << '\n');
    return Ret;
  }
  llvm_unreachable("Invalid type");
  return nullptr;
}

void dumpUsers(Value *V, StringRef Prompt) {
  if (!V)
    return;
  LLVM_DEBUG(dbgs() << Prompt << " Users of " << *V << " :\n");
  for (auto UI = V->user_begin(), UE = V->user_end(); UI != UE; ++UI)
    LLVM_DEBUG(dbgs() << "  " << **UI << '\n');
}

std::string getSPIRVTypeName(StringRef BaseName, StringRef Postfixes) {
  assert(!BaseName.empty() && "Invalid SPIR-V type Name");
  auto TN = std::string(kSPIRVTypeName::PrefixAndDelim) + BaseName.str();
  if (Postfixes.empty())
    return TN;
  return TN + kSPIRVTypeName::Delimiter + Postfixes.str();
}

bool isSPIRVConstantName(StringRef TyName) {
  if (TyName == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler) ||
      TyName == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
    return true;

  return false;
}

Type *getSPIRVTypeByChangeBaseTypeName(Module *M, Type *T, StringRef OldName,
                                       StringRef NewName) {
  StringRef Postfixes;
  if (isSPIRVType(T, OldName, &Postfixes))
    return getOrCreateOpaquePtrType(M, getSPIRVTypeName(NewName, Postfixes));
  LLVM_DEBUG(dbgs() << " Invalid SPIR-V type " << *T << '\n');
  llvm_unreachable("Invalid SPIR-V type");
  return nullptr;
}

std::string getSPIRVImageTypePostfixes(StringRef SampledType,
                                       SPIRVTypeImageDescriptor Desc,
                                       SPIRVAccessQualifierKind Acc) {
  std::string S;
  raw_string_ostream OS(S);
  OS << kSPIRVTypeName::PostfixDelim << SampledType
     << kSPIRVTypeName::PostfixDelim << Desc.Dim << kSPIRVTypeName::PostfixDelim
     << Desc.Depth << kSPIRVTypeName::PostfixDelim << Desc.Arrayed
     << kSPIRVTypeName::PostfixDelim << Desc.MS << kSPIRVTypeName::PostfixDelim
     << Desc.Sampled << kSPIRVTypeName::PostfixDelim << Desc.Format
     << kSPIRVTypeName::PostfixDelim << Acc;
  return OS.str();
}

std::string getSPIRVImageSampledTypeName(SPIRVType *Ty) {
  switch (Ty->getOpCode()) {
  case OpTypeVoid:
    return kSPIRVImageSampledTypeName::Void;
  case OpTypeInt:
    if (Ty->getIntegerBitWidth() == 32) {
      if (static_cast<SPIRVTypeInt *>(Ty)->isSigned())
        return kSPIRVImageSampledTypeName::Int;
      else
        return kSPIRVImageSampledTypeName::UInt;
    }
    break;
  case OpTypeFloat:
    switch (Ty->getFloatBitWidth()) {
    case 16:
      return kSPIRVImageSampledTypeName::Half;
    case 32:
      return kSPIRVImageSampledTypeName::Float;
    default:
      break;
    }
    break;
  default:
    break;
  }
  llvm_unreachable("Invalid sampled type for image");
  return std::string();
}

// ToDo: Find a way to represent uint sampled type in LLVM, maybe an
//      opaque type.
Type *getLLVMTypeForSPIRVImageSampledTypePostfix(StringRef Postfix,
                                                 LLVMContext &Ctx) {
  if (Postfix == kSPIRVImageSampledTypeName::Void)
    return Type::getVoidTy(Ctx);
  if (Postfix == kSPIRVImageSampledTypeName::Float)
    return Type::getFloatTy(Ctx);
  if (Postfix == kSPIRVImageSampledTypeName::Half)
    return Type::getHalfTy(Ctx);
  if (Postfix == kSPIRVImageSampledTypeName::Int ||
      Postfix == kSPIRVImageSampledTypeName::UInt)
    return Type::getInt32Ty(Ctx);
  llvm_unreachable("Invalid sampled type postfix");
  return nullptr;
}

std::string getImageBaseTypeName(StringRef Name) {

  SmallVector<StringRef, 4> SubStrs;
  const char Delims[] = {kSPR2TypeName::Delimiter, 0};
  Name.split(SubStrs, Delims);
  if (Name.startswith(kSPR2TypeName::OCLPrefix)) {
    Name = SubStrs[1];
  } else {
    Name = SubStrs[0];
  }

  std::string ImageTyName{Name};
  if (hasAccessQualifiedName(Name))
    ImageTyName.erase(ImageTyName.size() - 5, 3);

  return ImageTyName;
}

std::string mapOCLTypeNameToSPIRV(StringRef Name, StringRef Acc) {
  std::string BaseTy;
  std::string Postfixes;
  raw_string_ostream OS(Postfixes);
  if (Name.startswith(kSPR2TypeName::ImagePrefix)) {
    std::string ImageTyName = getImageBaseTypeName(Name);
    auto Desc = map<SPIRVTypeImageDescriptor>(ImageTyName);
    LLVM_DEBUG(dbgs() << "[trans image type] " << Name << " => "
                      << "(" << (unsigned)Desc.Dim << ", " << Desc.Depth << ", "
                      << Desc.Arrayed << ", " << Desc.MS << ", " << Desc.Sampled
                      << ", " << Desc.Format << ")\n");

    BaseTy = kSPIRVTypeName::Image;
    OS << getSPIRVImageTypePostfixes(
        kSPIRVImageSampledTypeName::Void, Desc,
        SPIRSPIRVAccessQualifierMap::map(Acc.str()));
  } else {
    LLVM_DEBUG(dbgs() << "Mapping of " << Name << " is not implemented\n");
    llvm_unreachable("Not implemented");
  }
  return getSPIRVTypeName(BaseTy, OS.str());
}

bool eraseIfNoUse(Function *F) {
  bool Changed = false;
  if (!F)
    return Changed;
  if (!GlobalValue::isInternalLinkage(F->getLinkage()) && !F->isDeclaration())
    return Changed;

  dumpUsers(F, "[eraseIfNoUse] ");
  for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE;) {
    auto U = *UI++;
    if (auto CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->use_empty()) {
        CE->dropAllReferences();
        Changed = true;
      }
    }
  }
  if (F->use_empty()) {
    LLVM_DEBUG(dbgs() << "Erase "; F->printAsOperand(dbgs()); dbgs() << '\n');
    F->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

void eraseIfNoUse(Value *V) {
  if (!V->use_empty())
    return;
  if (Constant *C = dyn_cast<Constant>(V)) {
    C->destroyConstant();
    return;
  }
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (!I->mayHaveSideEffects())
      I->eraseFromParent();
  }
  eraseIfNoUse(dyn_cast<Function>(V));
}

bool eraseUselessFunctions(Module *M) {
  bool Changed = false;
  for (auto I = M->begin(), E = M->end(); I != E;)
    Changed |= eraseIfNoUse(&(*I++));
  return Changed;
}

// The mangling algorithm follows OpenCL pipe built-ins clang 3.8 CodeGen rules.
static SPIR::MangleError
manglePipeOrAddressSpaceCastBuiltin(const SPIR::FunctionDescriptor &Fd,
                                    std::string &MangledName) {
  assert(OCLUtil::isPipeOrAddressSpaceCastBI(Fd.Name) &&
         "Method is expected to be called only for pipe and address space cast "
         "builtins!");
  if (Fd.isNull()) {
    MangledName.assign(SPIR::FunctionDescriptor::nullString());
    return SPIR::MANGLE_NULL_FUNC_DESCRIPTOR;
  }
  MangledName.assign("__" + Fd.Name);
  return SPIR::MANGLE_SUCCESS;
}

std::string mangleBuiltin(StringRef UniqName, ArrayRef<Type *> ArgTypes,
                          BuiltinFuncMangleInfo *BtnInfo) {
  if (!BtnInfo)
    return std::string(UniqName);
  BtnInfo->init(UniqName);
  std::string MangledName;
  LLVM_DEBUG(dbgs() << "[mangle] " << UniqName << " => ");
  SPIR::FunctionDescriptor FD;
  FD.Name = BtnInfo->getUnmangledName();
  bool BIVarArgNegative = BtnInfo->getVarArg() < 0;

  if (ArgTypes.empty()) {
    // Function signature cannot be ()(void, ...) so if there is an ellipsis
    // it must be ()(...)
    if (BIVarArgNegative) {
      FD.Parameters.emplace_back(
          SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_VOID)));
    }
  } else {
    for (unsigned I = 0, E = BIVarArgNegative ? ArgTypes.size()
                                              : (unsigned)BtnInfo->getVarArg();
         I != E; ++I) {
      auto T = ArgTypes[I];
      FD.Parameters.emplace_back(
          transTypeDesc(T, BtnInfo->getTypeMangleInfo(I)));
    }
  }
  // Ellipsis must be the last argument of any function
  if (!BIVarArgNegative) {
    assert((unsigned)BtnInfo->getVarArg() <= ArgTypes.size() &&
           "invalid index of an ellipsis");
    FD.Parameters.emplace_back(
        SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_VAR_ARG)));
  }

#if defined(SPIRV_SPIR20_MANGLING_REQUIREMENTS)
  SPIR::NameMangler Mangler(SPIR::SPIR20);
  Mangler.mangle(FD, MangledName);
#else
  if (OCLUtil::isPipeOrAddressSpaceCastBI(BtnInfo->getUnmangledName())) {
    manglePipeOrAddressSpaceCastBuiltin(FD, MangledName);
  } else {
    SPIR::NameMangler Mangler(SPIR::SPIR20);
    Mangler.mangle(FD, MangledName);
  }
#endif

  LLVM_DEBUG(dbgs() << MangledName << '\n');
  return MangledName;
}

/// Check if access qualifier is encoded in the type Name.
bool hasAccessQualifiedName(StringRef TyName) {
  if (TyName.size() < 5)
    return false;
  auto Acc = TyName.substr(TyName.size() - 5, 3);
  return llvm::StringSwitch<bool>(Acc)
      .Case(kAccessQualPostfix::ReadOnly, true)
      .Case(kAccessQualPostfix::WriteOnly, true)
      .Case(kAccessQualPostfix::ReadWrite, true)
      .Default(false);
}

SPIRVAccessQualifierKind getAccessQualifier(StringRef TyName) {
  return SPIRSPIRVAccessQualifierMap::map(
      getAccessQualifierFullName(TyName).str());
}

StringRef getAccessQualifierPostfix(SPIRVAccessQualifierKind Access) {
  switch (Access) {
  case AccessQualifierReadOnly:
    return kAccessQualPostfix::ReadOnly;
  case AccessQualifierWriteOnly:
    return kAccessQualPostfix::WriteOnly;
  case AccessQualifierReadWrite:
    return kAccessQualPostfix::ReadWrite;
  default:
    assert(false && "Unrecognized access qualifier!");
    return kAccessQualPostfix::ReadWrite;
  }
}

/// Get access qualifier from the type Name.
StringRef getAccessQualifierFullName(StringRef TyName) {
  assert(hasAccessQualifiedName(TyName) &&
         "Type is not qualified with access.");
  auto Acc = TyName.substr(TyName.size() - 5, 3);
  return llvm::StringSwitch<StringRef>(Acc)
      .Case(kAccessQualPostfix::ReadOnly, kAccessQualName::ReadOnly)
      .Case(kAccessQualPostfix::WriteOnly, kAccessQualName::WriteOnly)
      .Case(kAccessQualPostfix::ReadWrite, kAccessQualName::ReadWrite);
}

/// Translates OpenCL image type names to SPIR-V.
Type *getSPIRVImageTypeFromOCL(Module *M, Type *ImageTy) {
  assert(isOCLImageType(ImageTy) && "Unsupported type");
  auto ImageTypeName = ImageTy->getPointerElementType()->getStructName();
  StringRef Acc = kAccessQualName::ReadOnly;
  if (hasAccessQualifiedName(ImageTypeName))
    Acc = getAccessQualifierFullName(ImageTypeName);
  return getOrCreateOpaquePtrType(M, mapOCLTypeNameToSPIRV(ImageTypeName, Acc));
}

llvm::PointerType *getOCLClkEventType(Module *M) {
  return getOrCreateOpaquePtrType(M, SPIR_TYPE_NAME_CLK_EVENT_T,
                                  SPIRAS_Private);
}

llvm::PointerType *getOCLClkEventPtrType(Module *M) {
  return PointerType::get(getOCLClkEventType(M), SPIRAS_Generic);
}

llvm::Constant *getOCLNullClkEventPtr(Module *M) {
  return Constant::getNullValue(getOCLClkEventPtrType(M));
}

bool hasLoopMetadata(const Module *M) {
  for (const Function &F : *M)
    for (const BasicBlock &BB : F) {
      const Instruction *Term = BB.getTerminator();
      if (Term && Term->getMetadata("llvm.loop"))
        return true;
    }
  return false;
}

bool isSPIRVOCLExtInst(const CallInst *CI, OCLExtOpKind *ExtOp) {
  StringRef DemangledName;
  if (!oclIsBuiltin(CI->getCalledFunction()->getName(), DemangledName))
    return false;
  StringRef S = DemangledName;
  if (!S.startswith(kSPIRVName::Prefix))
    return false;
  S = S.drop_front(strlen(kSPIRVName::Prefix));
  auto Loc = S.find(kSPIRVPostfix::Divider);
  auto ExtSetName = S.substr(0, Loc);
  SPIRVExtInstSetKind Set = SPIRVEIS_Count;
  if (!SPIRVExtSetShortNameMap::rfind(ExtSetName.str(), &Set))
    return false;

  if (Set != SPIRVEIS_OpenCL)
    return false;

  auto ExtOpName = S.substr(Loc + 1);
  auto PostFixPos = ExtOpName.find("_R");
  ExtOpName = ExtOpName.substr(0, PostFixPos);

  OCLExtOpKind EOC;
  if (!OCLExtOpMap::rfind(ExtOpName.str(), &EOC))
    return false;

  *ExtOp = EOC;
  return true;
}

// Returns true if type(s) and number of elements (if vector) is valid
bool checkTypeForSPIRVExtendedInstLowering(IntrinsicInst *II, SPIRVModule *BM) {
  switch (II->getIntrinsicID()) {
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
  case Intrinsic::minimum:
  case Intrinsic::minnum:
  case Intrinsic::nearbyint:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::sin:
  case Intrinsic::sqrt:
  case Intrinsic::trunc: {
    // Although some of the intrinsics above take multiple arguments, it is
    // sufficient to check arg 0 because the LLVM Verifier will have checked
    // that all floating point operands have the same type and the second
    // argument of powi is i32.
    Type *Ty = II->getType();
    if (II->getArgOperand(0)->getType() != Ty)
      return false;
    int NumElems = 1;
    if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
      NumElems = VecTy->getNumElements();
      Ty = VecTy->getElementType();
    }
    if ((!Ty->isFloatTy() && !Ty->isDoubleTy() && !Ty->isHalfTy()) ||
        ((NumElems > 4) && (NumElems != 8) && (NumElems != 16))) {
      BM->SPIRVCK(
          false, InvalidFunctionCall, II->getCalledOperand()->getName().str());
      return false;
    }
    break;
  }
  case Intrinsic::abs: {
    Type *Ty = II->getType();
    int NumElems = 1;
    if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
      NumElems = VecTy->getNumElements();
      Ty = VecTy->getElementType();
    }
    if ((!Ty->isIntegerTy()) ||
        ((NumElems > 4) && (NumElems != 8) && (NumElems != 16))) {
      BM->SPIRVCK(
          false, InvalidFunctionCall, II->getCalledOperand()->getName().str());
    }
    break;
  }
  default:
    break;
  }
  return true;
}
} // namespace SPIRV

namespace {
class SPIRVFriendlyIRMangleInfo : public BuiltinFuncMangleInfo {
public:
  SPIRVFriendlyIRMangleInfo(spv::Op OC, ArrayRef<Type *> ArgTys)
      : OC(OC), ArgTys(ArgTys) {}

  void init(StringRef UniqUnmangledName) override {
    UnmangledName = UniqUnmangledName.str();
    switch (OC) {
    case OpConvertUToF:
      LLVM_FALLTHROUGH;
    case OpUConvert:
      LLVM_FALLTHROUGH;
    case OpSatConvertUToS:
      // Treat all arguments as unsigned
      addUnsignedArg(-1);
      break;
    case OpSubgroupShuffleINTEL:
      LLVM_FALLTHROUGH;
    case OpSubgroupShuffleXorINTEL:
      addUnsignedArg(1);
      break;
    case OpSubgroupShuffleDownINTEL:
      LLVM_FALLTHROUGH;
    case OpSubgroupShuffleUpINTEL:
      addUnsignedArg(2);
      break;
    case OpSubgroupBlockWriteINTEL:
      addUnsignedArg(0);
      addUnsignedArg(1);
      break;
    case OpSubgroupImageBlockWriteINTEL:
      addUnsignedArg(2);
      break;
    case OpSubgroupBlockReadINTEL:
      setArgAttr(0, SPIR::ATTR_CONST);
      addUnsignedArg(0);
      break;
    case OpAtomicUMax:
      LLVM_FALLTHROUGH;
    case OpAtomicUMin:
      addUnsignedArg(0);
      addUnsignedArg(3);
      break;
    default:;
      // No special handling is needed
    }
  }

private:
  spv::Op OC;
  ArrayRef<Type *> ArgTys;
};
class OpenCLStdToSPIRVFriendlyIRMangleInfo : public BuiltinFuncMangleInfo {
public:
  OpenCLStdToSPIRVFriendlyIRMangleInfo(OCLExtOpKind ExtOpId,
                                       ArrayRef<Type *> ArgTys, Type *RetTy)
      : ExtOpId(ExtOpId), ArgTys(ArgTys) {

    std::string Postfix = "";
    if (needRetTypePostfix())
      Postfix = kSPIRVPostfix::Divider + getPostfixForReturnType(RetTy, true);

    UnmangledName = getSPIRVExtFuncName(SPIRVEIS_OpenCL, ExtOpId, Postfix);
  }

  bool needRetTypePostfix() {
    switch (ExtOpId) {
    case OpenCLLIB::Vload_half:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::Vload_halfn:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::Vloada_halfn:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::Vloadn:
      return true;
    default:
      return false;
    }
  }

  void init(StringRef) override {
    switch (ExtOpId) {
    case OpenCLLIB::UAbs:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UAbs_diff:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UAdd_sat:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UHadd:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::URhadd:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UClamp:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMad_hi:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMad_sat:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMax:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMin:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMul_hi:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::USub_sat:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::U_Upsample:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMad24:
      LLVM_FALLTHROUGH;
    case OpenCLLIB::UMul24:
      // Treat all arguments as unsigned
      addUnsignedArg(-1);
      break;
    case OpenCLLIB::S_Upsample:
      addUnsignedArg(1);
      break;
    default:;
      // No special handling is needed
    }
  }

private:
  OCLExtOpKind ExtOpId;
  ArrayRef<Type *> ArgTys;
};
} // namespace

namespace SPIRV {
std::string getSPIRVFriendlyIRFunctionName(OCLExtOpKind ExtOpId,
                                           ArrayRef<Type *> ArgTys,
                                           Type *RetTy) {
  OpenCLStdToSPIRVFriendlyIRMangleInfo MangleInfo(ExtOpId, ArgTys, RetTy);
  return mangleBuiltin(MangleInfo.getUnmangledName(), ArgTys, &MangleInfo);
}

std::string getSPIRVFriendlyIRFunctionName(const std::string &UniqName,
                                           spv::Op OC,
                                           ArrayRef<Type *> ArgTys) {
  SPIRVFriendlyIRMangleInfo MangleInfo(OC, ArgTys);
  return mangleBuiltin(UniqName, ArgTys, &MangleInfo);
}

} // namespace SPIRV
