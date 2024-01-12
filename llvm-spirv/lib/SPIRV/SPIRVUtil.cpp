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

// This file needs to be included before anything that declares
// llvm::PointerType to avoid a compilation bug on MSVC.
#include "llvm/Demangle/ItaniumDemangle.h"

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
#include "llvm/IR/Metadata.h"
#include "llvm/IR/TypedPointerType.h"
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

bool isSupportedTriple(Triple T) { return T.isSPIR() || T.isSPIRV(); }

void addFnAttr(CallInst *Call, Attribute::AttrKind Attr) {
  Call->addFnAttr(Attr);
}

void removeFnAttr(CallInst *Call, Attribute::AttrKind Attr) {
  Call->removeFnAttr(Attr);
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

std::string mapLLVMTypeToOCLType(const Type *Ty, bool Signed, Type *PET) {
  if (Ty->isHalfTy())
    return "half";
  if (Ty->isFloatTy())
    return "float";
  if (Ty->isDoubleTy())
    return "double";
  if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
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
  if (const auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getNumElements();
    std::stringstream Ss;
    Ss << mapLLVMTypeToOCLType(EleTy, Signed) << Size;
    return Ss.str();
  }
  // It is expected that `Ty` can be mapped to `ReturnType` from "Optional
  // Postfixes for SPIR-V Builtin Function Names" section of
  // SPIRVRepresentationInLLVM.rst document (aka SPIRV-friendly IR).
  // If `Ty` is not a scalar or vector type mentioned in the document (return
  // value of some SPIR-V instructions may be represented as pointer to a struct
  // in LLVM IR) we can mangle the type.
  BuiltinFuncMangleInfo MangleInfo;
  if (Ty->isPointerTy())
    Ty = TypedPointerType::get(PET, Ty->getPointerAddressSpace());
  std::string MangledName =
      mangleBuiltin("", const_cast<Type *>(Ty), &MangleInfo);
  // Remove "_Z0"(3 characters) from the front of the name
  return MangledName.erase(0, 3);
}

StructType *getOrCreateOpaqueStructType(Module *M, StringRef Name) {
  auto *OpaqueType = StructType::getTypeByName(M->getContext(), Name);
  if (!OpaqueType)
    OpaqueType = StructType::create(M->getContext(), Name);
  return OpaqueType;
}

void getFunctionTypeParameterTypes(llvm::FunctionType *FT,
                                   std::vector<Type *> &ArgTys) {
  for (auto I = FT->param_begin(), E = FT->param_end(); I != E; ++I) {
    ArgTys.push_back(*I);
  }
}

bool isVoidFuncTy(FunctionType *FT) { return FT->getReturnType()->isVoidTy(); }

bool isOCLImageType(llvm::Type *Ty, StringRef *Name) {
  if (auto *TPT = dyn_cast_or_null<TypedPointerType>(Ty))
    if (auto *ST = dyn_cast_or_null<StructType>(TPT->getElementType()))
      if (ST->isOpaque()) {
        auto FullName = ST->getName();
        if (FullName.find(kSPR2TypeName::ImagePrefix) == 0) {
          if (Name)
            *Name = FullName.drop_front(strlen(kSPR2TypeName::OCLPrefix));
          return true;
        }
      }
  if (auto *TET = dyn_cast_or_null<TargetExtType>(Ty)) {
    assert(!Name && "Cannot get the name for a target-extension type image");
    return TET->getName() == "spirv.Image";
  }
  return false;
}
/// \param BaseTyName is the type Name as in spirv.BaseTyName.Postfixes
/// \param Postfix contains postfixes extracted from the SPIR-V image
///   type Name as spirv.BaseTyName.Postfixes.
bool isSPIRVStructType(llvm::Type *Ty, StringRef BaseTyName,
                       StringRef *Postfix) {
  auto *ST = dyn_cast<StructType>(Ty);
  if (!ST)
    return false;
  if (ST->isOpaque()) {
    auto FullName = ST->getName();
    std::string N =
        std::string(kSPIRVTypeName::PrefixAndDelim) + BaseTyName.str();
    if (FullName != N)
      N = N + kSPIRVTypeName::Delimiter;
    if (FullName.starts_with(N)) {
      if (Postfix)
        *Postfix = FullName.drop_front(N.size());
      return true;
    }
  }
  return false;
}

bool isSYCLHalfType(llvm::Type *Ty) {
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (!ST->hasName())
      return false;
    StringRef Name = ST->getName();
    if (!Name.consume_front("class."))
      return false;
    if ((Name.starts_with("sycl::") || Name.starts_with("cl::sycl::") ||
         Name.starts_with("__sycl_internal::")) &&
        Name.ends_with("::half")) {
      return true;
    }
  }
  return false;
}

bool isSYCLBfloat16Type(llvm::Type *Ty) {
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (!ST->hasName())
      return false;
    StringRef Name = ST->getName();
    if (!Name.consume_front("class."))
      return false;
    if ((Name.starts_with("sycl::") || Name.starts_with("cl::sycl::") ||
         Name.starts_with("__sycl_internal::")) &&
        Name.ends_with("::bfloat16")) {
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
    report_fatal_error(llvm::Twine(SS.str()), false);
  }
  if (!F || F->getFunctionType() != FT) {
    auto *NewF =
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
    if (F)
      NewF->setDSOLocal(F->isDSOLocal());
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
    End = CI->arg_size();
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

std::string prefixSPIRVName(const std::string &S) {
  return std::string(kSPIRVName::Prefix) + S;
}

StringRef dePrefixSPIRVName(StringRef R, SmallVectorImpl<StringRef> &Postfix) {
  const size_t Start = strlen(kSPIRVName::Prefix);
  if (!R.starts_with(kSPIRVName::Prefix))
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

std::string getSPIRVFuncName(Op OC, const Type *PRetTy, bool IsSigned,
                             Type *PET) {
  return prefixSPIRVName(getName(OC) + kSPIRVPostfix::Divider +
                         getPostfixForReturnType(PRetTy, IsSigned, PET));
}

std::string getSPIRVFuncName(SPIRVBuiltinVariableKind BVKind) {
  return prefixSPIRVName(getName(BVKind));
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

  if (Postfix.starts_with(kSPIRVPostfix::Rt))
    return new SPIRVDecorate(spv::DecorationFPRoundingMode, Target,
                             map<SPIRVFPRoundingModeKind>(Postfix.str()));

  return nullptr;
}

SPIRVValue *addDecorations(SPIRVValue *Target,
                           const SmallVectorImpl<std::string> &Decs) {
  for (auto &I : Decs)
    if (auto *Dec = mapPostfixToDecorate(I, Target))
      Target->addDecorate(Dec);
  return Target;
}

std::string getPostfixForReturnType(CallInst *CI, bool IsSigned) {
  return getPostfixForReturnType(CI->getType(), IsSigned);
}

std::string getPostfixForReturnType(const Type *PRetTy, bool IsSigned,
                                    Type *PET) {
  return std::string(kSPIRVPostfix::Return) +
         mapLLVMTypeToOCLType(PRetTy, IsSigned, PET);
}

// Enqueue kernel, kernel query, pipe and address space cast built-ins
// are not mangled.
bool isNonMangledOCLBuiltin(StringRef Name) {
  if (!Name.starts_with("__"))
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
  if ((!Name.starts_with(kSPIRVName::Prefix) && !isNonMangledOCLBuiltin(S)) ||
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
  if (!Postfix.empty())
    return false;
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
  if (!Name.starts_with("_Z"))
    return false;
  // OpenCL C++ built-ins are declared in cl namespace.
  // TODO: consider using 'St' abbriviation for cl namespace mangling.
  // Similar to ::std:: in C++.
  if (IsCpp) {
    if (!Name.starts_with("_ZN"))
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
    if (!Name.substr(DemangledNameLenStart, Start - DemangledNameLenStart)
             .getAsInteger(10, Len)) {
      DemangledName = Name.substr(Start, Len);
      return true;
    }
    SPIRVDBG(errs() << "Error in extracting integer value");
    return false;
  }
  size_t Start = Name.find_first_not_of("0123456789", 2);
  size_t Len = 0;
  if (!Name.substr(2, Start - 2).getAsInteger(10, Len)) {
    DemangledName = Name.substr(Start, Len);
    return true;
  }
  SPIRVDBG(errs() << "Error in extracting integer value");
  return false;
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

bool hasArrayArg(Function *F) {
  for (auto I = F->arg_begin(), E = F->arg_end(); I != E; ++I) {
    LLVM_DEBUG(dbgs() << "[hasArrayArg] " << *I << '\n');
    if (I->getType()->isArrayTy()) {
      return true;
    }
  }
  return false;
}

/// Convert a struct name from the name given to it in Itanium name mangling to
/// the name given to it as an LLVM opaque struct.
static std::string demangleBuiltinOpenCLTypeName(StringRef MangledStructName) {
  assert(MangledStructName.starts_with("ocl_") &&
         "Not a valid builtin OpenCL mangled name");
  // Bare structure type that starts with ocl_ is a builtin opencl type.
  // See clang/lib/CodeGen/CGOpenCLRuntime for how these map to LLVM types
  // and clang/lib/AST/ItaniumMangle for how they are mangled.
  // In general, ocl_<foo> is mapped to pointer-to-%opencl.<foo>, but
  // there is some variance around whether or not _t is included in the
  // mangled name.
  std::string LlvmStructName = StringSwitch<StringRef>(MangledStructName)
                                   .Case("ocl_sampler", "opencl.sampler_t")
                                   .Case("ocl_event", "opencl.event_t")
                                   .Case("ocl_clkevent", "opencl.clk_event_t")
                                   .Case("ocl_queue", "opencl.queue_t")
                                   .Case("ocl_reserveid", "opencl.reserve_id_t")
                                   .Default("")
                                   .str();
  if (LlvmStructName.empty()) {
    LlvmStructName = "opencl.";
    LlvmStructName += MangledStructName.substr(4); // Strip off ocl_
    if (!MangledStructName.ends_with("_t"))
      LlvmStructName += "_t";
  }
  return LlvmStructName;
}

/// Convert a C/C++ type name into an LLVM type, if it's a basic integer or
/// floating point type.
static Type *parsePrimitiveType(LLVMContext &Ctx, StringRef Name) {
  return StringSwitch<Type *>(Name)
      .Cases("char", "signed char", "unsigned char", Type::getInt8Ty(Ctx))
      .Cases("short", "unsigned short", Type::getInt16Ty(Ctx))
      .Cases("int", "unsigned int", Type::getInt32Ty(Ctx))
      .Cases("long", "unsigned long", Type::getInt64Ty(Ctx))
      .Cases("long long", "unsigned long long", Type::getInt64Ty(Ctx))
      .Case("half", Type::getHalfTy(Ctx))
      .Case("float", Type::getFloatTy(Ctx))
      .Case("double", Type::getDoubleTy(Ctx))
      .Case("void", Type::getInt8Ty(Ctx))
      .Default(nullptr);
}

} // namespace SPIRV

// The demangler node hierarchy doesn't use LLVM's RTTI helper functions (as it
// also needs to live in libcxxabi). By specializing this implementation here,
// we can add support for these functions.
#define NODE(X)                                                                \
  template <typename From> struct llvm::isa_impl<itanium_demangle::X, From> {  \
    static inline bool doit(const From &Val) {                                 \
      return Val.getKind() == itanium_demangle::Node::K##X;                    \
    }                                                                          \
  };
#include "llvm/Demangle/ItaniumNodes.def"

namespace SPIRV {

namespace {
// An allocator to use with the demangler API.
class DefaultAllocator {
  BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...ArgList) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(ArgList)...);
  }

  void *allocateNodeArray(size_t Sz) {
    using namespace llvm::itanium_demangle;
    return Alloc.Allocate(sizeof(Node *) * Sz, alignof(Node *));
  }
};
} // unnamed namespace

static StringRef stringify(const itanium_demangle::NameType *Node) {
  return Node->getName();
}

/// Convert a mangled name that represents a basic integer, floating-point,
/// etc. type into the corresponding LLVM type.
static Type *getPrimitiveType(LLVMContext &Ctx,
                              const llvm::itanium_demangle::Node *N) {
  using namespace llvm::itanium_demangle;
  if (auto *Name = dyn_cast<NameType>(N)) {
    return parsePrimitiveType(Ctx, stringify(Name));
  }
  if (auto *BitInt = dyn_cast<BitIntType>(N)) {
    unsigned BitWidth = 0;
    BitInt->match([&](const Node *NodeSize, bool) {
      const StringRef SizeStr(stringify(cast<NameType>(NodeSize)));
      SizeStr.getAsInteger(10, BitWidth);
    });
    return Type::getIntNTy(Ctx, BitWidth);
  }
  if (auto *FP = dyn_cast<BinaryFPType>(N)) {
    StringRef SizeStr;
    FP->match([&](const Node *NodeDimension) {
      SizeStr = stringify(cast<NameType>(NodeDimension));
    });
    return StringSwitch<Type *>(SizeStr)
        .Case("16", Type::getHalfTy(Ctx))
        .Case("32", Type::getFloatTy(Ctx))
        .Case("64", Type::getDoubleTy(Ctx))
        .Case("128", Type::getFP128Ty(Ctx))
        .Default(nullptr);
  }
  return nullptr;
}

template <typename FnType>
static TypedPointerType *
parseNode(Module *M, const llvm::itanium_demangle::Node *ParamType,
          FnType GetStructType) {
  using namespace llvm::itanium_demangle;
  Type *PointeeTy = nullptr;
  unsigned AS = 0;

  if (auto *Name = dyn_cast<NameType>(ParamType)) {
    // This corresponds to a simple class name. Since we only care about
    // pointer element types, the only relevant names are those corresponding
    // to the OpenCL special types (which all begin with "ocl_").
    StringRef Arg(stringify(Name));
    if (Arg.starts_with("ocl_")) {
      const std::string StructName = demangleBuiltinOpenCLTypeName(Arg);
      PointeeTy = GetStructType(StructName);
    } else if (Arg.consume_front("__spirv_")) {
      // This is a pointer to a SPIR-V OpType* opaque struct. In general,
      // convert __spirv_<Type>[__Suffix] to %spirv.Type[._Suffix]
      auto NameSuffixPair = Arg.split('_');
      std::string StructName = "spirv.";
      StructName += NameSuffixPair.first;
      if (!NameSuffixPair.second.empty()) {
        StructName += ".";
        StructName += NameSuffixPair.second;
      }
      PointeeTy = GetStructType(StructName);
    } else if (Arg == "ndrange_t") {
      PointeeTy = GetStructType(Arg);
    }
  } else if (auto *P = dyn_cast<itanium_demangle::PointerType>(ParamType)) {
    const Node *Pointee = P->getPointee();

    // Strip through all of the qualifiers on the pointee type.
    while (true) {
      if (auto *VendorTy = dyn_cast<VendorExtQualType>(Pointee)) {
        Pointee = VendorTy->getTy();
        StringRef Qualifier(&*VendorTy->getExt().begin(),
                            VendorTy->getExt().size());
        if (Qualifier.consume_front("AS")) {
          Qualifier.getAsInteger(10, AS);
        }
      } else if (auto *Qual = dyn_cast<QualType>(Pointee)) {
        Pointee = Qual->getChild();
      } else {
        break;
      }
    }

    if (auto *Name = dyn_cast<NameType>(Pointee)) {
      StringRef MangledStructName(stringify(Name));
      if (MangledStructName.consume_front("__spirv_")) {
        // This is a pointer to a SPIR-V OpType* opaque struct. In general,
        // convert __spirv_<Type>[__Suffix] to %spirv.Type[._Suffix]
        auto NameSuffixPair = MangledStructName.split('_');
        std::string StructName = "spirv.";
        StructName += NameSuffixPair.first;
        if (!NameSuffixPair.second.empty()) {
          StructName += ".";
          StructName += NameSuffixPair.second;
        }
        PointeeTy = GetStructType(StructName);
      } else if (MangledStructName.starts_with("opencl.")) {
        PointeeTy = GetStructType(MangledStructName);
      } else if (MangledStructName.starts_with("ocl_")) {
        const std::string StructName =
            demangleBuiltinOpenCLTypeName(MangledStructName);
        PointeeTy = TypedPointerType::get(GetStructType(StructName), 0);
      } else {
        PointeeTy = parsePrimitiveType(M->getContext(), MangledStructName);
      }
    } else if (auto *Ty = getPrimitiveType(M->getContext(), Pointee)) {
      PointeeTy = Ty;
    } else if (auto *Vec = dyn_cast<itanium_demangle::VectorType>(Pointee)) {
      unsigned ElemCount = 0;
      const StringRef ElemCountStr(
          stringify(cast<NameType>(Vec->getDimension())));
      ElemCountStr.getAsInteger(10, ElemCount);
      if (auto *Ty = getPrimitiveType(M->getContext(), Vec->getBaseType())) {
        PointeeTy = llvm::VectorType::get(Ty, ElemCount, false);
      }
    } else if (llvm::isa<itanium_demangle::PointerType>(Pointee)) {
      PointeeTy = parseNode(M, Pointee, GetStructType);
    } else {
      // Other possible pointee types do not correspond to any of the special
      // struct types were are looking for here.
    }
  } else if (auto *VendorTy = dyn_cast<VendorExtQualType>(ParamType)) {
    // This is a block parameter. Decode the pointee type as if it were a
    // void (*)(void) function pointer type.
    if (VendorTy->getExt() == "block_pointer") {
      PointeeTy =
          llvm::FunctionType::get(Type::getVoidTy(M->getContext()), false);
    }
  } else {
    // Other parameter types are not likely to be pointer types, so we can
    // ignore these.
  }
  return PointeeTy ? TypedPointerType::get(PointeeTy, AS) : nullptr;
}

bool getParameterTypes(Function *F, SmallVectorImpl<Type *> &ArgTys,
                       std::function<std::string(StringRef)> NameMapFn) {
  using namespace llvm::itanium_demangle;
  // If there's no mangled name, we can't do anything. Also, if there's no
  // parameters, do nothing.
  StringRef Name = F->getName();
  if (!Name.starts_with("_Z") || F->arg_empty())
    return Name.starts_with("_Z");

  Module *M = F->getParent();
  auto GetStructType = [&](StringRef Name) {
    return getOrCreateOpaqueStructType(M, NameMapFn ? NameMapFn(Name) : Name);
  };

  // Start by filling in a skeleton of information we can get from the LLVM type
  // itself.
  ArgTys.clear();
  auto *FT = F->getFunctionType();
  ArgTys.reserve(FT->getNumParams());
  bool HasSret = false;
  for (Argument &Arg : F->args()) {
    if (!Arg.getType()->isPointerTy())
      ArgTys.push_back(Arg.getType());
    else if (Type *Ty = Arg.getParamStructRetType()) {
      assert(!HasSret && &Arg == F->getArg(0) &&
             "sret parameter should only appear on the first argument");
      HasSret = true;
      unsigned AS = Arg.getType()->getPointerAddressSpace();
      if (auto *STy = dyn_cast<StructType>(Ty))
        ArgTys.push_back(
            TypedPointerType::get(GetStructType(STy->getName()), AS));
      else
        ArgTys.push_back(TypedPointerType::get(Ty, AS));
    } else {
      ArgTys.push_back(Arg.getType());
    }
  }

  // Skip the first argument if it's an sret parameter--this would be an
  // implicit parameter not recognized as part of the function parameters.
  auto *ArgIter = ArgTys.begin();
  if (HasSret)
    ++ArgIter;

  // Demangle the function arguments. If we get an input name of
  // "_Z12write_imagei20ocl_image1d_array_woDv2_iiDv4_i", then we expect
  // that Demangler.getFunctionParameters will return
  // "(ocl_image1d_array_wo, int __vector(2), int, int __vector(4))" (in other
  // words, the stuff between the parentheses if you ran C++ filt, including
  // the parentheses itself).
  const StringRef MangledName(F->getName());
  ManglingParser<DefaultAllocator> Demangler(MangledName.begin(),
                                             MangledName.end());
  // We expect to see only function name encodings here. If it's not a function
  // name encoding, bail out.
  auto *RootNode = dyn_cast_or_null<FunctionEncoding>(Demangler.parse());
  if (!RootNode)
    return false;

  // Get the parameter list. If the function is a vararg function, drop the last
  // parameter.
  NodeArray Params = RootNode->getParams();
  if (F->isVarArg()) {
    bool HasVarArgParam = false;
    if (!Params.empty()) {
      if (auto *Name = dyn_cast<NameType>(Params[Params.size() - 1])) {
        if (stringify(Name) == "...")
          HasVarArgParam = true;
      }
    }
    if (HasVarArgParam) {
      Params = NodeArray(Params.begin(), Params.size() - 1);
    } else {
      LLVM_DEBUG(dbgs() << "[getParameterTypes] function " << MangledName
                        << " was expected to have a varargs parameter\n");
      return false;
    }
  }

  // Sanity check that the name mangling matches up to the expected number of
  // arguments.
  if (Params.size() != (size_t)(ArgTys.end() - ArgIter)) {
    LLVM_DEBUG(dbgs() << "[getParameterTypes] function " << MangledName
                      << " appears to have " << Params.size()
                      << " arguments but has " << (ArgTys.end() - ArgIter)
                      << "\n");
    return false;
  }

  // Overwrite the types of pointer-typed arguments with information from
  // demangling.
  bool DemangledSuccessfully = true;
  for (auto *ParamType : Params) {
    Type *ArgTy = *ArgIter;
    Type *DemangledTy = parseNode(M, ParamType, GetStructType);
    if (ArgTy->isPointerTy() && DemangledTy == nullptr) {
      DemangledTy = TypedPointerType::get(Type::getInt8Ty(ArgTy->getContext()),
                                          ArgTy->getPointerAddressSpace());
      LLVM_DEBUG(dbgs() << "Failed to recover type of argument " << *ArgTy
                        << " of function " << F->getName() << "\n");
      DemangledSuccessfully = false;
    } else if (ArgTy->isTargetExtTy() || !DemangledTy)
      DemangledTy = ArgTy;
    if (auto *TPT = dyn_cast<TypedPointerType>(DemangledTy))
      if (ArgTy->isPointerTy() &&
          TPT->getAddressSpace() != ArgTy->getPointerAddressSpace())
        DemangledTy = TypedPointerType::get(TPT->getElementType(),
                                            ArgTy->getPointerAddressSpace());
    *ArgIter++ = DemangledTy;
  }
  return DemangledSuccessfully;
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
  auto *NewCI = addCallInst(M, NewName, CI->getType(), Args, Attrs, CI, Mangle,
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
  auto *NewCI = addCallInst(M, NewName, RetTy, Args, Attrs, CI, Mangle, InstName,
                           TakeFuncName);
  auto *NewI = RetMutate(NewCI);
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
  auto *M = F->getParent();
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    if (auto *CI = dyn_cast<CallInst>(*I++))
      mutateCallInst(M, CI, ArgMutate, Mangle, Attrs, TakeFuncName);
  }
  if (F->use_empty())
    F->eraseFromParent();
}

void mutateFunction(
    Function *F,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate,
    BuiltinFuncMangleInfo *Mangle, AttributeList *Attrs, bool TakeName) {
  auto *M = F->getParent();
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    if (auto *CI = dyn_cast<CallInst>(*I++))
      mutateCallInst(M, CI, ArgMutate, RetMutate, Mangle, Attrs, TakeName);
  }
  if (F->use_empty())
    F->eraseFromParent();
}

CallInst *addCallInst(Module *M, StringRef FuncName, Type *RetTy,
                      ArrayRef<Value *> Args, AttributeList *Attrs,
                      Instruction *Pos, BuiltinFuncMangleInfo *Mangle,
                      StringRef InstName, bool TakeFuncName) {

  auto *F = getOrCreateFunction(M, RetTy, getTypes(Args), FuncName, Mangle,
                               Attrs, TakeFuncName);
  // Cannot assign a Name to void typed values
  auto *CI = CallInst::Create(F, Args, RetTy->isVoidTy() ? "" : InstName, Pos);
  CI->setCallingConv(F->getCallingConv());
  CI->setAttributes(F->getAttributes());
  return CI;
}

CallInst *addCallInstSPIRV(Module *M, StringRef FuncName, Type *RetTy,
                           ArrayRef<Value *> Args, AttributeList *Attrs,
                           ArrayRef<Type *> PointerElementTypes,
                           Instruction *Pos, StringRef InstName) {
  BuiltinFuncMangleInfo BtnInfo;
  for (unsigned I = 0; I < PointerElementTypes.size(); I++) {
    if (Args[I]->getType()->isPointerTy())
      BtnInfo.getTypeMangleInfo(I).PointerTy = TypedPointerType::get(
          PointerElementTypes[I], Args[I]->getType()->getPointerAddressSpace());
  }
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
  auto *Vec = Builder.CreateVectorSplat(VecSize, *Range.first);
  unsigned Index = 1;
  for (++Range.first; Range.first != Range.second; ++Range.first, ++Index)
    Vec = Builder.CreateInsertElement(
        Vec, *Range.first,
        ConstantInt::get(Type::getInt32Ty(InsPos->getContext()), Index, false));
  return Vec;
}

void makeVector(Instruction *InsPos, std::vector<Value *> &Ops,
                ValueVecRange Range) {
  auto *Vec = addVector(InsPos, Range);
  Ops.erase(Range.first, Range.second);
  Ops.push_back(Vec);
}

PointerType *getInt8PtrTy(PointerType *T) {
  return PointerType::get(T->getContext(), T->getAddressSpace());
}

Value *castToInt8Ptr(Value *V, Instruction *Pos) {
  return CastInst::CreatePointerCast(
      V, getInt8PtrTy(cast<PointerType>(V->getType())), "", Pos);
}

IntegerType *getSizetType(Module *M) {
  return IntegerType::getIntNTy(M->getContext(),
                                M->getDataLayout().getPointerSizeInBits(0));
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
int64_t getMDOperandAsInt(MDNode *N, unsigned I) {
  return mdconst::dyn_extract<ConstantInt>(N->getOperand(I))->getZExtValue();
}

// Additional helper function to be reused by getMDOperandAs* helpers
Metadata *getMDOperandOrNull(MDNode *N, unsigned I) {
  if (!N)
    return nullptr;
  return N->getOperand(I);
}

StringRef getMDOperandAsString(MDNode *N, unsigned I) {
  if (auto *Str = dyn_cast_or_null<MDString>(getMDOperandOrNull(N, I)))
    return Str->getString();
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
      StrSet.insert(getMDOperandAsString(MD, J).str());
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

bool isDecoratedSPIRVFunc(const Function *F, StringRef &UndecoratedName) {
  if (!F->hasName() || !F->getName().starts_with(kSPIRVName::Prefix))
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
  if (Ty->isPointerTy())
    Ty = TypedPointerType::get(Type::getInt8Ty(Ty->getContext()),
                               Ty->getPointerAddressSpace());
  if (Info.IsAtomic && !isa<TypedPointerType>(Ty)) {
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
      return SPIR::RefParamType(new SPIR::PrimitiveType(SPIR::PRIMITIVE_INT));
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
    return transTypeDesc(TypedPointerType::get(Ty->getArrayElementType(), 0),
                         Info);
  }
  if (Ty->isStructTy()) {
    auto Name = Ty->getStructName();
    std::string Tmp;

    if (Name.starts_with(kLLVMTypeName::StructPrefix))
      Name = Name.drop_front(strlen(kLLVMTypeName::StructPrefix));
    if (Name.starts_with(kSPIRVTypeName::PrefixAndDelim)) {
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
  if (auto *TargetTy = dyn_cast<TargetExtType>(Ty)) {
    std::string FullName;
    unsigned AS = 0;
    {
      raw_string_ostream OS(FullName);
      StringRef Name = TargetTy->getName();
      if (Name.consume_front(kSPIRVTypeName::PrefixAndDelim)) {
        OS << "__spirv_" << Name;
        AS = getOCLOpaqueTypeAddrSpace(
            SPIRVOpaqueTypeOpCodeMap::map(Name.str()));
      } else {
        OS << Name;
      }
      if (!TargetTy->int_params().empty())
        OS << "_";
      for (Type *InnerTy : TargetTy->type_params())
        OS << "_" << convertTypeToPostfix(InnerTy);
      for (unsigned Param : TargetTy->int_params())
        OS << "_" << Param;
    }
    // Translate as if it's a pointer to the named struct.
    auto *Inner = new SPIR::UserDefinedType(FullName);
    auto *PT = new SPIR::PointerType(Inner);
    PT->setAddressSpace(static_cast<SPIR::TypeAttributeEnum>(
        AS + (unsigned)SPIR::ATTR_ADDR_SPACE_FIRST));
    return SPIR::RefParamType(PT);
  }

  if (auto *TPT = dyn_cast<TypedPointerType>(Ty)) {
    auto *ET = TPT->getElementType();
    SPIR::ParamType *EPT = nullptr;
    if (isa<FunctionType>(ET)) {
      assert(isVoidFuncTy(cast<FunctionType>(ET)) && "Not supported");
      EPT = new SPIR::BlockType;
    } else if (auto *StructTy = dyn_cast<StructType>(ET)) {
      LLVM_DEBUG(dbgs() << "ptr to struct: " << *Ty << '\n');
      auto TyName = StructTy->getStructName();
      if (TyName.starts_with(kSPR2TypeName::OCLPrefix)) {
        auto DelimPos = TyName.find_first_of(kSPR2TypeName::Delimiter,
                                             strlen(kSPR2TypeName::OCLPrefix));
        if (DelimPos != StringRef::npos)
          TyName = TyName.substr(0, DelimPos);
      }
      LLVM_DEBUG(dbgs() << "  type Name: " << TyName << '\n');

      auto Prim = getOCLTypePrimitiveEnum(TyName);
      if (StructTy->isOpaque()) {
        if (TyName == "opencl.block") {
          auto *BlockTy = new SPIR::BlockType;
          // Handle block with local memory arguments according to OpenCL 2.0
          // spec.
          if (Info.IsLocalArgBlock) {
            SPIR::RefParamType VoidTyRef(
                new SPIR::PrimitiveType(SPIR::PRIMITIVE_VOID));
            auto *VoidPtrTy = new SPIR::PointerType(VoidTyRef);
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
            auto *OpaquePtrTy = new SPIR::PointerType(OpaqueTyRef);
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
    auto *PT = new SPIR::PointerType(transTypeDesc(ET, Info));
    PT->setAddressSpace(static_cast<SPIR::TypeAttributeEnum>(
        TPT->getAddressSpace() + (unsigned)SPIR::ATTR_ADDR_SPACE_FIRST));
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
  Type *SourceTy;
  Value *Addr;
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    SourceTy = GV->getValueType();
    Addr = GV;
  } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
    SourceTy = AI->getAllocatedType();
    Addr = AI;
  } else if (auto *GEP = dyn_cast<GEPOperator>(V)) {
    assert(GEP->getNumOperands() == 3 && "must be a GEP from an array");
    SourceTy = GEP->getSourceElementType();
    [[maybe_unused]] auto *OP1 = cast<ConstantInt>(GEP->getOperand(1));
    [[maybe_unused]] auto *OP2 = cast<ConstantInt>(GEP->getOperand(2));
    assert(OP1->getZExtValue() == 0);
    assert(OP2->getZExtValue() == 0);
    Addr = GEP->getOperand(0);
  } else {
    llvm_unreachable("Unknown array type");
  }
  assert(SourceTy->getArrayNumElements() == Size);
  return new LoadInst(SourceTy, Addr, "", Pos);
}

Constant *getScalarOrVectorConstantInt(Type *T, uint64_t V, bool IsSigned) {
  if (auto *IT = dyn_cast<IntegerType>(T))
    return ConstantInt::get(IT, V);
  if (auto *VT = dyn_cast<FixedVectorType>(T)) {
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
  if (auto *IT = dyn_cast<IntegerType>(T)) {
    assert(Len == 1 && "Invalid length");
    return ConstantInt::get(IT, V, IsSigned);
  }
  if (isa<PointerType>(T)) {
    unsigned PointerSize =
        Pos->getModule()->getDataLayout().getPointerTypeSizeInBits(T);
    auto *ET = Type::getIntNTy(T->getContext(), PointerSize);
    auto *AT = ArrayType::get(ET, Len);
    std::vector<Constant *> EV(Len, ConstantInt::get(ET, V, IsSigned));
    auto *CA = ConstantArray::get(AT, EV);
    auto *Alloca = new AllocaInst(AT, 0, "", Pos);
    new StoreInst(CA, Alloca, Pos);
    auto *Zero = ConstantInt::getNullValue(Type::getInt32Ty(T->getContext()));
    Value *Index[] = {Zero, Zero};
    auto *Ret = GetElementPtrInst::CreateInBounds(AT, Alloca, Index, "", Pos);
    LLVM_DEBUG(dbgs() << "[getScalarOrArrayConstantInt] Alloca: " << *Alloca
                      << ", Return: " << *Ret << '\n');
    return Ret;
  }
  if (auto *AT = dyn_cast<ArrayType>(T)) {
    auto *ET = AT->getArrayElementType();
    assert(AT->getArrayNumElements() == Len);
    std::vector<Constant *> EV(Len, ConstantInt::get(ET, V, IsSigned));
    auto *Ret = ConstantArray::get(AT, EV);
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

std::string convertTypeToPostfix(Type *Ty) {
  if (Ty->isIntegerTy()) {
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "uint";
    case 64:
      return "long";
    default:
      return (Twine("i") + Twine(Ty->getIntegerBitWidth())).str();
    }
  } else if (Ty->isHalfTy()) {
    return "half";
  } else if (Ty->isFloatTy()) {
    return "float";
  } else if (Ty->isDoubleTy()) {
    return "double";
  } else if (Ty->isBFloatTy()) {
    return "bfloat16";
  } else if (Ty->isVoidTy()) {
    return "void";
  } else {
    report_fatal_error("Unknown LLVM type for element type");
  }
}

std::string getImageBaseTypeName(StringRef Name) {

  SmallVector<StringRef, 4> SubStrs;
  const char Delims[] = {kSPR2TypeName::Delimiter, 0};
  Name.split(SubStrs, Delims);
  if (Name.starts_with(kSPR2TypeName::OCLPrefix)) {
    Name = SubStrs[1];
  } else {
    Name = SubStrs[0];
  }

  std::string ImageTyName{Name};
  if (hasAccessQualifiedName(Name))
    ImageTyName.erase(ImageTyName.size() - 5, 3);

  return ImageTyName;
}

size_t getImageOperandsIndex(Op OpCode) {
  switch (OpCode) {
  case OpImageRead:
  case OpImageSampleExplicitLod:
    return 2;
  case OpImageWrite:
    return 3;
  default:
    return ~0U;
  }
}

SPIRVTypeImageDescriptor getImageDescriptor(Type *Ty) {
  if (auto *TET = dyn_cast_or_null<TargetExtType>(Ty)) {
    auto IntParams = TET->int_params();
    assert(IntParams.size() > 6 && "Expected type to be an image type");
    return SPIRVTypeImageDescriptor(SPIRVImageDimKind(IntParams[0]),
                                    IntParams[1], IntParams[2], IntParams[3],
                                    IntParams[4], IntParams[5]);
  }
  StringRef TyName;
  [[maybe_unused]] bool IsImg = isOCLImageType(Ty, &TyName);
  assert(IsImg && "Must be an image type");
  return map<SPIRVTypeImageDescriptor>(getImageBaseTypeName(TyName));
}

bool eraseIfNoUse(Function *F) {
  bool Changed = false;
  if (!F)
    return Changed;
  if (!GlobalValue::isInternalLinkage(F->getLinkage()) && !F->isDeclaration())
    return Changed;

  dumpUsers(F, "[eraseIfNoUse] ");
  for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE;) {
    auto *U = *UI++;
    if (auto *CE = dyn_cast<ConstantExpr>(U)) {
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
  if (BtnInfo->avoidMangling())
    return std::string(UniqName);
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
      auto *T = ArgTypes[I];
      auto MangleInfo = BtnInfo->getTypeMangleInfo(I);
      if (MangleInfo.PointerTy && T->isPointerTy()) {
        T = MangleInfo.PointerTy;
      }
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
  assert(hasAccessQualifiedName(TyName) &&
         "Type is not qualified with access.");
  auto Acc = TyName.substr(TyName.size() - 5, 3);
  return llvm::StringSwitch<SPIRVAccessQualifierKind>(Acc)
      .Case(kAccessQualPostfix::ReadOnly, AccessQualifierReadOnly)
      .Case(kAccessQualPostfix::WriteOnly, AccessQualifierWriteOnly)
      .Case(kAccessQualPostfix::ReadWrite, AccessQualifierReadWrite);
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
  if (!S.starts_with(kSPIRVName::Prefix))
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

std::string decodeSPIRVTypeName(StringRef Name,
                                SmallVectorImpl<std::string> &Strs) {
  SmallVector<StringRef, 4> SubStrs;
  const char Delim[] = {kSPIRVTypeName::Delimiter, 0};
  Name.split(SubStrs, Delim, -1, true);
  assert(SubStrs.size() >= 2 && "Invalid SPIRV type name");
  assert(SubStrs[0] == kSPIRVTypeName::Prefix && "Invalid prefix");
  assert((SubStrs.size() == 2 || !SubStrs[2].empty()) && "Invalid postfix");

  if (SubStrs.size() > 2) {
    const char PostDelim[] = {kSPIRVTypeName::PostfixDelim, 0};
    SmallVector<StringRef, 4> Postfixes;
    SubStrs[2].split(Postfixes, PostDelim, -1, true);
    assert(Postfixes.size() > 1 && Postfixes[0].empty() && "Invalid postfix");
    for (unsigned I = 1, E = Postfixes.size(); I != E; ++I)
      Strs.push_back(std::string(Postfixes[I]).c_str());
  }
  return SubStrs[1].str();
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
        (!BM->hasCapability(CapabilityVectorAnyINTEL) &&
         ((NumElems > 4) && (NumElems != 8) && (NumElems != 16)))) {
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
        (!BM->hasCapability(CapabilityVectorAnyINTEL) &&
         ((NumElems > 4) && (NumElems != 8) && (NumElems != 16)))) {
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

CallInst *setAttrByCalledFunc(CallInst *Call) {
  Function *F = Call->getCalledFunction();
  assert(F);
  if (F->isIntrinsic()) {
    return Call;
  }
  Call->setCallingConv(F->getCallingConv());
  Call->setAttributes(F->getAttributes());
  return Call;
}

bool isSPIRVBuiltinVariable(GlobalVariable *GV,
                            SPIRVBuiltinVariableKind *Kind) {
  if (!GV->hasName() || !getSPIRVBuiltin(GV->getName().str(), *Kind))
    return false;
  return true;
}

// Variable like GlobalInvolcationId[x] -> get_global_id(x).
// Variable like WorkDim -> get_work_dim().
// Replace the following pattern:
// %a = addrspacecast i32 addrspace(1)* @__spirv_BuiltInSubgroupMaxSize to
// i32 addrspace(4)*
// %b = load i32, i32 addrspace(4)* %a, align 4
// %c = load i32, i32 addrspace(4)* %a, align 4
// With:
// %b = call spir_func i32 @_Z22get_max_sub_group_sizev()
// %c = call spir_func i32 @_Z22get_max_sub_group_sizev()

// And replace the following pattern:
// %a = addrspacecast <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to
// <3 x i64> addrspace(4)*
// %b = load <3 x i64>, <3 x i64> addrspace(4)* %a, align 32
// %c = extractelement <3 x i64> %b, i32 idx
// %d = extractelement <3 x i64> %b, i32 idx
// With:
// %0 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
// %1 = insertelement <3 x i64> undef, i64 %0, i32 0
// %2 = call spir_func i64 @_Z13get_global_idj(i32 1) #1
// %3 = insertelement <3 x i64> %1, i64 %2, i32 1
// %4 = call spir_func i64 @_Z13get_global_idj(i32 2) #1
// %5 = insertelement <3 x i64> %3, i64 %4, i32 2
// %c = extractelement <3 x i64> %5, i32 idx
// %d = extractelement <3 x i64> %5, i32 idx
//
// Replace the following pattern:
// %0 = addrspacecast <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to
// <3 x i64> addrspace(4)*
// %1 = getelementptr <3 x i64>, <3 x i64> addrspace(4)* %0, i64 0, i64 0
// %2 = load i64, i64 addrspace(4)* %1, align 32
// With:
// %0 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
// %1 = insertelement <3 x i64> undef, i64 %0, i32 0
// %2 = call spir_func i64 @_Z13get_global_idj(i32 1) #1
// %3 = insertelement <3 x i64> %1, i64 %2, i32 1
// %4 = call spir_func i64 @_Z13get_global_idj(i32 2) #1
// %5 = insertelement <3 x i64> %3, i64 %4, i32 2
// %6 = extractelement <3 x i64> %5, i32 0

/// Recursively look through the uses of a global variable, including casts or
/// gep offsets, to find all loads of the variable. Gep offsets that are non-0
/// are accumulated in the AccumulatedOffset parameter, which will eventually be
/// used to figure out which index of a variable is being used.
static void replaceUsesOfBuiltinVar(Value *V, const APInt &AccumulatedOffset,
                                    Function *ReplacementFunc,
                                    GlobalVariable *GV) {
  const DataLayout &DL = ReplacementFunc->getParent()->getDataLayout();
  SmallVector<Instruction *, 4> InstsToRemove;
  for (User *U : V->users()) {
    if (auto *Cast = dyn_cast<CastInst>(U)) {
      replaceUsesOfBuiltinVar(Cast, AccumulatedOffset, ReplacementFunc, GV);
      InstsToRemove.push_back(Cast);
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      APInt NewOffset = AccumulatedOffset.sextOrTrunc(
          DL.getIndexSizeInBits(GEP->getPointerAddressSpace()));
      if (!GEP->accumulateConstantOffset(DL, NewOffset))
        llvm_unreachable("Illegal GEP of a SPIR-V builtin variable");
      replaceUsesOfBuiltinVar(GEP, NewOffset, ReplacementFunc, GV);
      InstsToRemove.push_back(GEP);
    } else if (auto *Load = dyn_cast<LoadInst>(U)) {
      // Figure out which index the accumulated offset corresponds to. If we
      // have a weird offset (e.g., trying to load byte 7), bail out.
      Type *ScalarTy = ReplacementFunc->getReturnType();
      APInt Index;
      uint64_t Remainder;
      APInt::udivrem(AccumulatedOffset, ScalarTy->getScalarSizeInBits() / 8,
                     Index, Remainder);
      if (Remainder != 0)
        llvm_unreachable("Illegal GEP of a SPIR-V builtin variable");

      IRBuilder<> Builder(Load);
      Value *Replacement;
      if (ReplacementFunc->getFunctionType()->getNumParams() == 0) {
        if (Load->getType() != ScalarTy)
          llvm_unreachable("Illegal use of a SPIR-V builtin variable");
        Replacement =
            setAttrByCalledFunc(Builder.CreateCall(ReplacementFunc, {}));
      } else {
        // The function has an index parameter.
        if (auto *VecTy = dyn_cast<FixedVectorType>(Load->getType())) {
          // Reconstruct the original global variable vector because
          // the load type may not match.
          // global <3 x i64>, load <6 x i32>
          VecTy = cast<FixedVectorType>(GV->getValueType());
          if (!Index.isZero() || DL.getTypeSizeInBits(VecTy) !=
                                     DL.getTypeSizeInBits(Load->getType()))
            llvm_unreachable("Illegal use of a SPIR-V builtin variable");
          Replacement = UndefValue::get(VecTy);
          for (unsigned I = 0; I < VecTy->getNumElements(); I++) {
            Replacement = Builder.CreateInsertElement(
                Replacement,
                setAttrByCalledFunc(
                    Builder.CreateCall(ReplacementFunc, {Builder.getInt32(I)})),
                Builder.getInt32(I));
          }
          // Insert a bitcast from the reconstructed vector to the load vector
          // type in case they are different.
          // Input:
          // %1 = load <6 x i32>, ptr addrspace(1) %0, align 32
          // %2 = extractelement <6 x i32> %1, i32 0
          // %3 = add i32 5, %2
          // Modified:
          // < reconstruct global vector elements 0 and 1 >
          // %2 = insertelement <3 x i64> %0, i64 %1, i32 2
          // %3 = bitcast <3 x i64> %2 to <6 x i32>
          // %4 = extractelement <6 x i32> %3, i32 0
          // %5 = add i32 5, %4
          Replacement = Builder.CreateBitCast(Replacement, Load->getType());
        } else if (Load->getType() == ScalarTy) {
          Replacement = setAttrByCalledFunc(Builder.CreateCall(
              ReplacementFunc, {Builder.getInt32(Index.getZExtValue())}));
        } else {
          llvm_unreachable("Illegal load type of a SPIR-V builtin variable");
        }
      }
      Load->replaceAllUsesWith(Replacement);
      InstsToRemove.push_back(Load);
    } else {
      llvm_unreachable("Illegal use of a SPIR-V builtin variable");
    }
  }

  for (Instruction *I : InstsToRemove)
    I->eraseFromParent();
}

bool lowerBuiltinVariableToCall(GlobalVariable *GV,
                                SPIRVBuiltinVariableKind Kind) {
  // There might be dead constant users of GV (for example, SPIRVLowerConstExpr
  // replaces ConstExpr uses but those ConstExprs are not deleted, since LLVM
  // constants are created on demand as needed and never deleted).
  // Remove them first!
  GV->removeDeadConstantUsers();

  Module *M = GV->getParent();
  LLVMContext &C = M->getContext();
  std::string FuncName = GV->getName().str();
  Type *GVTy = GV->getValueType();
  Type *ReturnTy = GVTy;
  // Some SPIR-V builtin variables are translated to a function with an index
  // argument.
  bool HasIndexArg =
      ReturnTy->isVectorTy() &&
      !(BuiltInSubgroupEqMask <= Kind && Kind <= BuiltInSubgroupLtMask);
  if (HasIndexArg)
    ReturnTy = cast<VectorType>(ReturnTy)->getElementType();
  std::vector<Type *> ArgTy;
  if (HasIndexArg)
    ArgTy.push_back(Type::getInt32Ty(C));
  std::string MangledName;
  mangleOpenClBuiltin(FuncName, ArgTy, MangledName);
  Function *Func = M->getFunction(MangledName);
  if (!Func) {
    FunctionType *FT = FunctionType::get(ReturnTy, ArgTy, false);
    Func = Function::Create(FT, GlobalValue::ExternalLinkage, MangledName, M);
    Func->setCallingConv(CallingConv::SPIR_FUNC);
    Func->addFnAttr(Attribute::NoUnwind);
    Func->addFnAttr(Attribute::WillReturn);
    Func->setDoesNotAccessMemory();
  }

  replaceUsesOfBuiltinVar(GV, APInt(64, 0), Func, GV);
  return true;
}

bool lowerBuiltinVariablesToCalls(Module *M) {
  std::vector<GlobalVariable *> WorkList;
  for (auto I = M->global_begin(), E = M->global_end(); I != E; ++I) {
    SPIRVBuiltinVariableKind Kind;
    if (!isSPIRVBuiltinVariable(&(*I), &Kind))
      continue;
    if (!lowerBuiltinVariableToCall(&(*I), Kind))
      return false;
    WorkList.push_back(&(*I));
  }
  for (auto &I : WorkList) {
    I->eraseFromParent();
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
bool lowerBuiltinCallsToVariables(Module *M) {
  LLVM_DEBUG(dbgs() << "Enter lowerBuiltinCallsToVariables\n");
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

bool lowerBuiltins(SPIRVModule *BM, Module *M) {
  auto Format = BM->getBuiltinFormat();
  if (Format == BuiltinFormat::Function && !lowerBuiltinVariablesToCalls(M))
    return false;
  if (Format == BuiltinFormat::Global && !lowerBuiltinCallsToVariables(M))
    return false;
  return true;
}

bool postProcessBuiltinReturningStruct(Function *F) {
  Module *M = F->getParent();
  LLVMContext *Context = &M->getContext();
  std::string Name = F->getName().str();
  F->setName(Name + ".old");
  SmallVector<Instruction *, 32> InstToRemove;
  for (auto *U : F->users()) {
    if (auto *CI = dyn_cast<CallInst>(U)) {
      auto *ST = cast<StoreInst>(*(CI->user_begin()));
      std::vector<Type *> ArgTys;
      getFunctionTypeParameterTypes(F->getFunctionType(), ArgTys);
      ArgTys.insert(ArgTys.begin(),
                    PointerType::get(F->getReturnType(), SPIRAS_Private));
      auto *NewF =
          getOrCreateFunction(M, Type::getVoidTy(*Context), ArgTys, Name);
      auto SretAttr = Attribute::get(*Context, Attribute::AttrKind::StructRet,
                                     F->getReturnType());
      NewF->addParamAttr(0, SretAttr);
      NewF->setCallingConv(F->getCallingConv());
      auto Args = getArguments(CI);
      Args.insert(Args.begin(), ST->getPointerOperand());
      auto *NewCI = CallInst::Create(NewF, Args, CI->getName(), CI);
      NewCI->addParamAttr(0, SretAttr);
      NewCI->setCallingConv(CI->getCallingConv());
      InstToRemove.push_back(ST);
      InstToRemove.push_back(CI);
    }
  }
  for (auto *Inst : InstToRemove) {
    Inst->dropAllReferences();
    Inst->eraseFromParent();
  }
  F->dropAllReferences();
  F->eraseFromParent();
  return true;
}

bool postProcessBuiltinWithArrayArguments(Function *F,
                                          StringRef DemangledName) {
  LLVM_DEBUG(dbgs() << "[postProcessOCLBuiltinWithArrayArguments] " << *F
                    << '\n');
  auto Attrs = F->getAttributes();
  auto Name = F->getName();
  mutateFunction(
      F,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        auto FBegin = CI->getFunction()->begin()->getFirstInsertionPt();
        for (auto &I : Args) {
          auto *T = I->getType();
          if (!T->isArrayTy())
            continue;
          auto *Alloca = new AllocaInst(T, 0, "", &(*FBegin));
          new StoreInst(I, Alloca, false, CI);
          auto *Zero =
              ConstantInt::getNullValue(Type::getInt32Ty(T->getContext()));
          Value *Index[] = {Zero, Zero};
          I = GetElementPtrInst::CreateInBounds(T, Alloca, Index, "", CI);
        }
        return Name.str();
      },
      nullptr, &Attrs);
  return true;
}

bool postProcessBuiltinsReturningStruct(Module *M, bool IsCpp) {
  StringRef DemangledName;
  // postProcessBuiltinReturningStruct may remove some functions from the
  // module, so use make_early_inc_range
  for (auto &F : make_early_inc_range(M->functions())) {
    if (F.hasName() && F.isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[postProcess sret] " << F << '\n');
      if (F.getReturnType()->isStructTy() &&
          oclIsBuiltin(F.getName(), DemangledName, IsCpp)) {
        if (!postProcessBuiltinReturningStruct(&F))
          return false;
      }
    }
  }
  return true;
}

bool postProcessBuiltinsWithArrayArguments(Module *M, bool IsCpp) {
  StringRef DemangledName;
  // postProcessBuiltinWithArrayArguments may remove some functions from the
  // module, so use make_early_inc_range
  for (auto &F : make_early_inc_range(M->functions())) {
    if (F.hasName() && F.isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[postProcess array arg] " << F << '\n');
      if (hasArrayArg(&F) && oclIsBuiltin(F.getName(), DemangledName, IsCpp))
        if (!postProcessBuiltinWithArrayArguments(&F, DemangledName))
          return false;
    }
  }
  return true;
}

} // namespace SPIRV

namespace {
class SPIRVFriendlyIRMangleInfo : public BuiltinFuncMangleInfo {
public:
  SPIRVFriendlyIRMangleInfo(spv::Op OC, ArrayRef<Type *> ArgTys,
                            ArrayRef<SPIRVValue *> Ops)
      : OC(OC), ArgTys(ArgTys), Ops(Ops) {}

  void init(StringRef UniqUnmangledName) override {
    UnmangledName = UniqUnmangledName.str();
    switch (OC) {
    case OpConvertUToF:
    case OpUConvert:
    case OpSatConvertUToS:
      // Treat all arguments as unsigned
      addUnsignedArg(-1);
      break;
    case OpSubgroupShuffleINTEL:
    case OpSubgroupShuffleXorINTEL:
      addUnsignedArg(1);
      break;
    case OpSubgroupShuffleDownINTEL:
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
    case OpAtomicUMin:
      addUnsignedArg(0);
      addUnsignedArg(3);
      break;
    case OpGroupUMax:
    case OpGroupUMin:
    case OpGroupNonUniformBroadcast:
    case OpGroupNonUniformBallotBitCount:
    case OpGroupNonUniformShuffle:
    case OpGroupNonUniformShuffleXor:
    case OpGroupNonUniformShuffleUp:
    case OpGroupNonUniformShuffleDown:
      addUnsignedArg(2);
      break;
    case OpGroupNonUniformRotateKHR:
      if (ArgTys.size() == 4)
        addUnsignedArg(3);
      break;
    case OpGroupNonUniformInverseBallot:
    case OpGroupNonUniformBallotFindLSB:
    case OpGroupNonUniformBallotFindMSB:
      addUnsignedArg(1);
      break;
    case OpBitFieldSExtract:
    case OpGroupNonUniformBallotBitExtract:
      addUnsignedArg(1);
      addUnsignedArg(2);
      break;
    case OpGroupNonUniformIAdd:
    case OpGroupNonUniformFAdd:
    case OpGroupNonUniformIMul:
    case OpGroupNonUniformFMul:
    case OpGroupNonUniformSMin:
    case OpGroupNonUniformFMin:
    case OpGroupNonUniformSMax:
    case OpGroupNonUniformFMax:
    case OpGroupNonUniformBitwiseAnd:
    case OpGroupNonUniformBitwiseOr:
    case OpGroupNonUniformBitwiseXor:
    case OpGroupNonUniformLogicalAnd:
    case OpGroupNonUniformLogicalOr:
    case OpGroupNonUniformLogicalXor:
      addUnsignedArg(3);
      break;
    case OpBitFieldInsert:
    case OpGroupNonUniformUMax:
    case OpGroupNonUniformUMin:
      addUnsignedArg(2);
      addUnsignedArg(3);
      break;
    case OpEnqueueMarker:
      addUnsignedArg(1);
      break;
    case OpSubgroupAvcBmeInitializeINTEL:
      addUnsignedArgs(0, 7);
      break;
    case OpSubgroupAvcFmeInitializeINTEL:
    case OpSubgroupAvcSicConfigureIpeLumaINTEL:
      addUnsignedArgs(0, 6);
      break;
    case OpSubgroupAvcImeAdjustRefOffsetINTEL:
      addUnsignedArgs(1, 3);
      break;
    case OpSubgroupAvcImeGetBorderReachedINTEL:
    case OpSubgroupAvcImeRefWindowSizeINTEL:
    case OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL:
    case OpSubgroupAvcImeSetMaxMotionVectorCountINTEL:
    case OpSubgroupAvcImeSetWeightedSadINTEL:
    case OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL:
    case OpSubgroupAvcMceSetInterDirectionPenaltyINTEL:
    case OpSubgroupAvcMceSetInterShapePenaltyINTEL:
    case OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL:
    case OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL:
    case OpSubgroupAvcSicInitializeINTEL:
    case OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL:
    case OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL:
    case OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL:
    case OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL:
      addUnsignedArg(0);
      break;
    case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL:
    case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL:
    case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL:
    case OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL:
    case OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL:
    case OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL:
    case OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL:
      addUnsignedArgs(1, 2);
      break;
    case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL:
    case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL:
    case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL:
    case OpSubgroupAvcImeSetSingleReferenceINTEL:
      addUnsignedArg(1);
      break;
    case OpBitFieldUExtract:
    case OpSubgroupAvcImeInitializeINTEL:
    case OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL:
    case OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL:
      addUnsignedArgs(0, 2);
      break;
    case OpSubgroupAvcImeSetDualReferenceINTEL:
      addUnsignedArg(2);
      break;
    case OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL:
    case OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL:
    case OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL:
    case OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL:
    case OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL:
    case OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL:
    case OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL:
    case OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL:
    case OpSubgroupAvcSicGetMotionVectorMaskINTEL:
      addUnsignedArgs(0, 1);
      break;
    case OpSubgroupAvcSicConfigureIpeLumaChromaINTEL:
      addUnsignedArgs(0, 9);
      break;
    case OpSubgroupAvcSicConfigureSkcINTEL:
      addUnsignedArgs(0, 4);
      break;
    case OpUDotKHR:
    case OpUDotAccSatKHR:
      addUnsignedArg(-1);
      break;
    case OpSUDotKHR:
    case OpSUDotAccSatKHR:
      addUnsignedArg(1);
      break;
    case OpImageWrite: {
      size_t Idx = getImageOperandsIndex(OC);
      if (Ops.size() > Idx) {
        auto ImOp = static_cast<SPIRVConstant *>(Ops[Idx])->getZExtIntValue();
        if (ImOp & ImageOperandsMask::ImageOperandsZeroExtendMask)
          addUnsignedArg(2);
      }
      break;
    }
    default:;
      // No special handling is needed
    }
  }

private:
  spv::Op OC;
  ArrayRef<Type *> ArgTys;
  ArrayRef<SPIRVValue *> Ops;
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
    case OpenCLLIB::Vload_halfn:
    case OpenCLLIB::Vloada_halfn:
    case OpenCLLIB::Vloadn:
      return true;
    default:
      return false;
    }
  }

  void init(StringRef) override {
    switch (ExtOpId) {
    case OpenCLLIB::UAbs:
    case OpenCLLIB::UAbs_diff:
    case OpenCLLIB::UAdd_sat:
    case OpenCLLIB::UHadd:
    case OpenCLLIB::URhadd:
    case OpenCLLIB::UClamp:
    case OpenCLLIB::UMad_hi:
    case OpenCLLIB::UMad_sat:
    case OpenCLLIB::UMax:
    case OpenCLLIB::UMin:
    case OpenCLLIB::UMul_hi:
    case OpenCLLIB::USub_sat:
    case OpenCLLIB::U_Upsample:
    case OpenCLLIB::UMad24:
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
                                           spv::Op OC, ArrayRef<Type *> ArgTys,
                                           ArrayRef<SPIRVValue *> Ops) {
  SPIRVFriendlyIRMangleInfo MangleInfo(OC, ArgTys, Ops);
  return mangleBuiltin(UniqName, ArgTys, &MangleInfo);
}

} // namespace SPIRV
