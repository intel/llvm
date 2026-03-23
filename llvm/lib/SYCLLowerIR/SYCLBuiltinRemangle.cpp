//===-------- SYCLBuiltinRemangle.cpp - SYCL Builtin Remangler Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that remangles SYCL user device code function
// calls to match OpenCL C mangling conventions used in libclc/libspirv
// builtins. This is the INVERSE operation of libclc-remangler: instead of
// transforming library code to match SYCL types, we transform user code to
// match OpenCL types.
//
// The pass performs the following type transformations:
//
// 1. long long -> long
//    OpenCL C has no "long long" type; it only has "long" (always 64-bit).
//    SYCL's "long long" (always 64-bit) maps to OpenCL's "long".
//
//    Example: _Z17__spirv_ocl_s_absx -> _Z17__spirv_ocl_s_absl
//             (long long)           -> (long)
//
// 2. signed char -> char
//    In SYCL, 'signed char' is explicitly signed. In OpenCL C, 'char' is the
//    standard signed 8-bit type. The pass maps 'signed char' to 'char'.
//
//    Example: _Z17__spirv_ocl_s_absa -> _Z17__spirv_ocl_s_absc
//             (signed char)         -> (char)
//
// 2a. char -> unsigned char
//    In SYCL, signedness of 'char' is determined by host triple. In OpenCL C,
//    'char' is always signed.
//
//    When char is signed on host (e.g. X86):
//      SYCL's 'char' is signed -> OpenCL's 'char' (no transformation)
//      Example: _Z15__spirv_ocl_clzc (char) -> _Z15__spirv_ocl_clzc (char)
//
//    When char is unsigned on host (e.g. ARM, PowerPC, RISCV):
//      SYCL's 'char' is unsigned -> OpenCL's 'unsigned char'
//      Example: _Z15__spirv_ocl_clzc (char) -> _Z15__spirv_ocl_clzh (uchar)
//
//    Note: SPIR-V builtins containing `_s_` or `_Rchar` in the name are
//    dedicated to signed char types and must not be used when the char type is
//    unsigned. The restriction could be relaxed if SPV-IR support `_Rschar`
//    name in the future.
//
// 3. _Float16 -> half
//    SYCL uses _Float16 vendor extension. OpenCL C uses the 'half' type.
//
//    Example: _Z16__spirv_ocl_fabsDF16_ -> _Z16__spirv_ocl_fabsDh
//             (_Float16)               -> (half)
//
// 4. long -> int (Target-dependent: Windows and 32-bit platforms only)
//    On 64-bit Linux, 'long' is 64-bit and maps to OpenCL's 'long'.
//    On Windows (LLP64) and 32-bit platforms, 'long' is 32-bit and maps to
//    OpenCL's 'int'.
//
//    Windows x64: _Z17__spirv_ocl_s_absl -> _Z17__spirv_ocl_s_absi
//                 (long)                -> (int)
//
//    Linux x64:   _Z17__spirv_ocl_s_absl (no change)
//
// 5. Address Space transform (only if target's default addrspace is private)
//    The difference is in mangling conventions:
//    - SYCL: implicit pointers (mangled as "Pf") are generic (AS4)
//    - OpenCL C: implicit pointers (mangled as "Pf") are private (AS0)
//
//    SYCL implicit generic: _Z17__spirv_ocl_fractfPf (implicit = AS4) ->
//                  OpenCL: _Z17__spirv_ocl_fractfPU3AS4f (explicit AS4)
//
//    SYCL explicit private: _Z17__spirv_ocl_fractfPU3AS0f (explicit AS0) ->
//                  OpenCL: _Z17__spirv_ocl_fractfPf (implicit = AS0)
//
//    Other address spaces: _Z16__spirv_ocl_vloadlPU3AS1l (no change)
//
// 6. Vector Types
//    Vector transformations apply recursively to element types.
//
//    Example: _Z16__spirv_ocl_fabsDv16_DF16_ ->
//             _Z16__spirv_ocl_fabsDv16_Dh
//             (vector<_Float16, 16>)        -> (vector<half, 16>)
//
// Declaration Merging:
// When multiple SYCL function declarations remangle to the same name (e.g.,
// both signed char and char versions), the pass merges them into a single
// declaration. This is safe because they have the same IR function type.
//
//    Before: _Z17__spirv_ocl_s_absa (signed char)
//            _Z17__spirv_ocl_s_absc (char)
//    After:  _Z17__spirv_ocl_s_absc (merged to single declaration)
//
// The implementation includes:
// 1. SPIR type system
// 2. SPIR name mangler for generating OpenCL C mangled names
// 3. Bridge from itanium_demangle::Node to SPIR::ParamType
// 4. Type transformation logic for target-specific mappings
// 5. Address space remapping
//
// SAFETY NOTE - Why we transform ALL "__spirv_" functions:
//
// This pass transforms ALL functions containing "__spirv_" without validating
// against a whitelist of known SPIRV builtins. This is safe because "__" is a
// reserved namespace in C/C++. All legitimate "__spirv_*" functions come from
// libclc/libspirv. There will be a linker error if this rule is violated.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLBuiltinRemangle.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::itanium_demangle;

static cl::opt<bool> RemangleSPIRVTarget(
    "sycl-remangle-spirv-target",
    cl::desc(
        "Enable SYCL builtin remangling for SPIR/SPIRV targets "
        "(default: false, as these targets already have correct mangling)"),
    cl::init(false));

static cl::opt<bool> RemangleCharIsSigned(
    "sycl-remangle-char-is-signed",
    cl::desc("Char is signed for SYCL builtin remangling (default: true)"),
    cl::init(true));

namespace {

//===----------------------------------------------------------------------===//
// SPIR Type System (adapted from SPIRV-LLVM-Translator Mangler)
//===----------------------------------------------------------------------===//

namespace SPIR {

enum MangleError {
  MANGLE_SUCCESS,
  MANGLE_TYPE_NOT_SUPPORTED,
  MANGLE_NULL_FUNC_DESCRIPTOR
};

enum TypePrimitiveEnum {
  PRIMITIVE_BOOL,
  PRIMITIVE_UCHAR,
  PRIMITIVE_CHAR,
  PRIMITIVE_USHORT,
  PRIMITIVE_SHORT,
  PRIMITIVE_UINT,
  PRIMITIVE_INT,
  PRIMITIVE_ULONG,
  PRIMITIVE_LONG,
  PRIMITIVE_FLOAT16,
  PRIMITIVE_HALF,
  PRIMITIVE_FLOAT,
  PRIMITIVE_DOUBLE,
  PRIMITIVE_VOID,
  PRIMITIVE_NUM
};

enum TypeEnum {
  TYPE_ID_PRIMITIVE,
  TYPE_ID_POINTER,
  TYPE_ID_VECTOR,
  TYPE_ID_TEMPLATE_PARAMETER,
  TYPE_ID_STRUCTURE
};

enum TypeAttributeEnum {
  ATTR_RESTRICT,
  ATTR_VOLATILE,
  ATTR_CONST,
  ATTR_PRIVATE,
  ATTR_GLOBAL,
  ATTR_CONSTANT,
  ATTR_LOCAL,
  ATTR_GENERIC,
  ATTR_NONE,
  ATTR_NUM = ATTR_NONE
};

// Reference counting template
template <typename T> class RefCount {
public:
  RefCount() : Count(nullptr), Ptr(nullptr) {}
  RefCount(T *Ptr) : Count(new int(1)), Ptr(Ptr) {}
  RefCount(const RefCount<T> &Other) { cpy(Other); }
  ~RefCount() {
    if (Count)
      dispose();
  }

  RefCount &operator=(const RefCount<T> &Other) {
    if (this == &Other)
      return *this;
    if (Count)
      dispose();
    cpy(Other);
    return *this;
  }

  bool isNull() const { return !Ptr; }
  const T &operator*() const { return *Ptr; }
  T &operator*() { return *Ptr; }
  T *operator->() { return Ptr; }
  const T *operator->() const { return Ptr; }
  T *get() { return Ptr; }
  const T *get() const { return Ptr; }

private:
  void cpy(const RefCount<T> &Other) {
    Count = Other.Count;
    Ptr = Other.Ptr;
    if (Count)
      ++*Count;
  }

  void dispose() {
    if (0 == --*Count) {
      delete Count;
      delete Ptr;
      Ptr = nullptr;
      Count = nullptr;
    }
  }

  int *Count;
  T *Ptr;
};

// Forward declarations
struct ParamType;
typedef RefCount<ParamType> RefParamType;
struct TypeVisitor;

// Base type
struct ParamType {
  ParamType(TypeEnum TypeId) : TypeId(TypeId) {}
  virtual ~ParamType() {}
  virtual MangleError accept(TypeVisitor *) const = 0;
  virtual std::string toString() const = 0;
  virtual bool equals(const ParamType *) const = 0;
  TypeEnum getTypeId() const { return TypeId; }

protected:
  TypeEnum TypeId;
};

struct PrimitiveType : public ParamType {
  static const TypeEnum EnumTy;
  PrimitiveType(TypePrimitiveEnum P)
      : ParamType(TYPE_ID_PRIMITIVE), Primitive(P) {}
  MangleError accept(TypeVisitor *) const override;
  std::string toString() const override;
  bool equals(const ParamType *) const override;
  TypePrimitiveEnum getPrimitive() const { return Primitive; }

private:
  TypePrimitiveEnum Primitive;
};

struct PointerType : public ParamType {
  static const TypeEnum EnumTy;
  PointerType(const RefParamType Type);
  MangleError accept(TypeVisitor *) const override;
  std::string toString() const override;
  bool equals(const ParamType *) const override;
  const RefParamType &getPointee() const { return PType; }
  void setAddressSpace(TypeAttributeEnum Attr);
  TypeAttributeEnum getAddressSpace() const { return AddressSpace; }
  void setQualifier(TypeAttributeEnum Qual, bool Enabled);
  bool hasQualifier(TypeAttributeEnum Qual) const;

private:
  RefParamType PType;
  bool Qualifiers[3]; // restrict, volatile, const
  TypeAttributeEnum AddressSpace;
};

struct VectorType : public ParamType {
  static const TypeEnum EnumTy;
  VectorType(const RefParamType Type, int Len)
      : ParamType(TYPE_ID_VECTOR), PType(Type), Len(Len) {}
  MangleError accept(TypeVisitor *) const override;
  std::string toString() const override;
  bool equals(const ParamType *) const override;
  const RefParamType &getScalarType() const { return PType; }
  int getLength() const { return Len; }

private:
  RefParamType PType;
  int Len;
};

struct TemplateParameterType : public ParamType {
  static const TypeEnum EnumTy;
  TemplateParameterType(unsigned Index)
      : ParamType(TYPE_ID_TEMPLATE_PARAMETER), Index(Index) {}
  MangleError accept(TypeVisitor *) const override;
  std::string toString() const override;
  bool equals(const ParamType *) const override;
  unsigned getIndex() const { return Index; }

private:
  unsigned Index;
};

struct UserDefinedType : public ParamType {
  static const TypeEnum EnumTy;
  UserDefinedType(const std::string &Name)
      : ParamType(TYPE_ID_STRUCTURE), Name(Name) {}
  MangleError accept(TypeVisitor *) const override;
  std::string toString() const override;
  bool equals(const ParamType *) const override;
  StringRef getName() const { return Name; }

private:
  std::string Name;
};

struct TypeVisitor {
  virtual ~TypeVisitor() {}
  virtual MangleError visit(const PrimitiveType *) = 0;
  virtual MangleError visit(const VectorType *) = 0;
  virtual MangleError visit(const PointerType *) = 0;
  virtual MangleError visit(const TemplateParameterType *) = 0;
  virtual MangleError visit(const UserDefinedType *) = 0;
};

// Dynamic cast for SPIR types
template <typename T> const T *dynCast(const ParamType *PType) {
  return (T::EnumTy == PType->getTypeId()) ? static_cast<const T *>(PType)
                                           : nullptr;
}

template <typename T> T *dynCast(ParamType *PType) {
  return (T::EnumTy == PType->getTypeId()) ? static_cast<T *>(PType) : nullptr;
}

// Mangling utilities
// clang-format off
#define PRIMITIVE_TYPES_MAP(X) \
  X("bool",         "b",         PRIMITIVE_BOOL)                               \
  X("uchar",        "h",         PRIMITIVE_UCHAR)                              \
  X("char",         "c",         PRIMITIVE_CHAR)                               \
  X("ushort",       "t",         PRIMITIVE_USHORT)                             \
  X("short",        "s",         PRIMITIVE_SHORT)                              \
  X("uint",         "j",         PRIMITIVE_UINT)                               \
  X("int",          "i",         PRIMITIVE_INT)                                \
  X("ulong",        "m",         PRIMITIVE_ULONG)                              \
  X("long",         "l",         PRIMITIVE_LONG)                               \
  X("_Float16",     "DF16_",     PRIMITIVE_FLOAT16)                            \
  X("half",         "Dh",        PRIMITIVE_HALF)                               \
  X("float",        "f",         PRIMITIVE_FLOAT)                              \
  X("double",       "d",         PRIMITIVE_DOUBLE)                             \
  X("void",         "v",         PRIMITIVE_VOID)
// clang-format on

StringRef mangledPrimitiveString(TypePrimitiveEnum T) {
  switch (T) {
#define AS_ENUM_CASE(Name, Mangled, Enum)                                      \
  case Enum:                                                                   \
    return Mangled;
    PRIMITIVE_TYPES_MAP(AS_ENUM_CASE)
#undef AS_ENUM_CASE
  default:
    return {};
  }
}

void appendTemplateParameterMangling(unsigned Index, raw_ostream &Stream) {
  Stream << 'T';
  if (Index > 0)
    Stream << (Index - 1);
  Stream << '_';
}

std::string getLeafTypeMangling(const ParamType *Type) {
  if (const PrimitiveType *Prim = dynCast<PrimitiveType>(Type))
    return mangledPrimitiveString(Prim->getPrimitive()).str();
  if (const TemplateParameterType *TemplateParam =
          dynCast<TemplateParameterType>(Type)) {
    SmallString<8> Buffer;
    raw_svector_ostream Stream(Buffer);
    appendTemplateParameterMangling(TemplateParam->getIndex(), Stream);
    return std::string(Buffer);
  }
  return {};
}

// clang-format off
#define ATTRIBUTE_TYPES_MAP(X)                                                 \
  X(ATTR_RESTRICT, "r")                                                        \
  X(ATTR_VOLATILE, "V")                                                        \
  X(ATTR_CONST,    "K")                                                        \
  X(ATTR_PRIVATE,  "")                                                         \
  X(ATTR_GLOBAL,   "U3AS1")                                                    \
  X(ATTR_CONSTANT, "U3AS2")                                                    \
  X(ATTR_LOCAL,    "U3AS3")                                                    \
  X(ATTR_GENERIC,  "U3AS4")
// clang-format on

StringRef getMangledAttribute(TypeAttributeEnum Attribute) {
  switch (Attribute) {
#define AS_ENUM_CASE(Enum, Mangled)                                            \
  case Enum:                                                                   \
    return Mangled;
    ATTRIBUTE_TYPES_MAP(AS_ENUM_CASE)
#undef AS_ENUM_CASE
  default:
    return "";
  }
}

std::string getPointerAttributesMangling(const PointerType *P) {
  SmallString<8> QualStr(getMangledAttribute(P->getAddressSpace()));
  if (P->hasQualifier(ATTR_RESTRICT))
    QualStr += 'r';
  if (P->hasQualifier(ATTR_VOLATILE))
    QualStr += 'V';
  if (P->hasQualifier(ATTR_CONST))
    QualStr += 'K';
  return std::string(QualStr);
}

// Type implementations
const TypeEnum PrimitiveType::EnumTy = TYPE_ID_PRIMITIVE;
const TypeEnum PointerType::EnumTy = TYPE_ID_POINTER;
const TypeEnum VectorType::EnumTy = TYPE_ID_VECTOR;
const TypeEnum TemplateParameterType::EnumTy = TYPE_ID_TEMPLATE_PARAMETER;
const TypeEnum UserDefinedType::EnumTy = TYPE_ID_STRUCTURE;

MangleError PrimitiveType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string PrimitiveType::toString() const {
  switch (Primitive) {
#define AS_CASE(Name, Mangled, Enum)                                           \
  case Enum:                                                                   \
    return Name;
    PRIMITIVE_TYPES_MAP(AS_CASE)
#undef AS_CASE
  default:
    return "";
  }
}

bool PrimitiveType::equals(const ParamType *Type) const {
  const PrimitiveType *P = dynCast<PrimitiveType>(Type);
  return P && (Primitive == P->Primitive);
}

PointerType::PointerType(const RefParamType Type)
    : ParamType(TYPE_ID_POINTER), PType(Type), AddressSpace(ATTR_PRIVATE) {
  Qualifiers[0] = Qualifiers[1] = Qualifiers[2] = false;
}

MangleError PointerType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

void PointerType::setAddressSpace(TypeAttributeEnum Attr) {
  AddressSpace = Attr;
}

void PointerType::setQualifier(TypeAttributeEnum Qual, bool Enabled) {
  if (Qual == ATTR_RESTRICT)
    Qualifiers[0] = Enabled;
  else if (Qual == ATTR_VOLATILE)
    Qualifiers[1] = Enabled;
  else if (Qual == ATTR_CONST)
    Qualifiers[2] = Enabled;
}

bool PointerType::hasQualifier(TypeAttributeEnum Qual) const {
  if (Qual == ATTR_RESTRICT)
    return Qualifiers[0];
  if (Qual == ATTR_VOLATILE)
    return Qualifiers[1];
  if (Qual == ATTR_CONST)
    return Qualifiers[2];
  return false;
}

std::string PointerType::toString() const {
  SmallString<16> Buffer;
  raw_svector_ostream Stream(Buffer);
  Stream << getPointee()->toString() << '*';
  return std::string(Buffer);
}

bool PointerType::equals(const ParamType *Type) const {
  const PointerType *P = dynCast<PointerType>(Type);
  return P && getPointee()->equals(&*P->getPointee());
}

MangleError VectorType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string VectorType::toString() const {
  SmallString<16> Buffer;
  raw_svector_ostream Stream(Buffer);
  Stream << getScalarType()->toString() << Len;
  return std::string(Buffer);
}

bool VectorType::equals(const ParamType *Type) const {
  const VectorType *V = dynCast<VectorType>(Type);
  return V && (Len == V->Len) && getScalarType()->equals(&*V->getScalarType());
}

MangleError TemplateParameterType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string TemplateParameterType::toString() const {
  if (Index == 0)
    return "T_";
  return "T" + std::to_string(Index - 1) + "_";
}

bool TemplateParameterType::equals(const ParamType *Type) const {
  const TemplateParameterType *TemplateParam =
      dynCast<TemplateParameterType>(Type);
  return TemplateParam && (Index == TemplateParam->Index);
}

MangleError UserDefinedType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string UserDefinedType::toString() const { return Name; }

bool UserDefinedType::equals(const ParamType *Type) const {
  const UserDefinedType *U = dynCast<UserDefinedType>(Type);
  return U && (Name == U->Name);
}

class MangleVisitor : public TypeVisitor {
public:
  MangleVisitor(SmallVectorImpl<char> &Buffer, raw_ostream &Stream,
                unsigned InitialSeqId = 0)
      : Buffer(Buffer), Stream(Stream), SeqId(InitialSeqId) {}

  MangleError visit(const PrimitiveType *T) override {
    Stream << mangledPrimitiveString(T->getPrimitive());
    return MANGLE_SUCCESS;
  }

  MangleError visit(const PointerType *P) override {
    size_t Pos = Buffer.size();
    std::string AttrMangling = getPointerAttributesMangling(P);
    if (!mangleSubstitution(P, "P" + AttrMangling)) {
      Stream << "P" << AttrMangling;
      MangleError Err = P->getPointee()->accept(this);
      if (!AttrMangling.empty())
        recordSubstitution(currentBuffer().substr(Pos + 1).str());
      recordSubstitution(currentBuffer().substr(Pos).str());
      return Err;
    }
    return MANGLE_SUCCESS;
  }

  MangleError visit(const VectorType *V) override {
    size_t Index = Buffer.size();
    SmallString<16> TypeStorage;
    raw_svector_ostream TypeStream(TypeStorage);
    TypeStream << "Dv" << V->getLength() << '_';
    StringRef TypeStr(TypeStorage);
    if (!mangleSubstitution(V, TypeStr)) {
      Stream << TypeStr;
      MangleError Err = V->getScalarType()->accept(this);
      recordSubstitution(currentBuffer().substr(Index).str());
      return Err;
    }
    return MANGLE_SUCCESS;
  }

  MangleError visit(const TemplateParameterType *T) override {
    appendTemplateParameterMangling(T->getIndex(), Stream);
    return MANGLE_SUCCESS;
  }

  MangleError visit(const UserDefinedType *U) override {
    size_t Index = Buffer.size();
    StringRef Name = U->getName();
    if (!mangleSubstitution(U, Name)) {
      Stream << Name.size() << Name;
      recordSubstitution(currentBuffer().substr(Index).str());
    }
    return MANGLE_SUCCESS;
  }

private:
  StringRef currentBuffer() const {
    return StringRef(Buffer.data(), Buffer.size());
  }

  void mangleSequenceID(unsigned SeqID) {
    if (SeqID == 1)
      Stream << '0';
    else if (SeqID > 1) {
      SmallString<8> Bstr;
      constexpr StringRef Charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      SeqID--;
      Bstr.reserve(7);
      for (; SeqID != 0; SeqID /= 36)
        Bstr.push_back(Charset[SeqID % 36]);
      std::reverse(Bstr.begin(), Bstr.end());
      Stream << Bstr;
    }
    Stream << '_';
  }

  bool mangleSubstitution(const ParamType *Type, StringRef TypeStr) {
    if (currentBuffer().find(TypeStr) == StringRef::npos)
      return false;

    SmallString<32> ThisTypeStr(TypeStr);
    std::string NType;
    if (const PointerType *P = dynCast<PointerType>(Type)) {
      ThisTypeStr += getPointeeMangling(P->getPointee());
    } else if (const VectorType *PVec = dynCast<VectorType>(Type)) {
      NType = getLeafTypeMangling(PVec->getScalarType().get());
      if (!NType.empty())
        ThisTypeStr += NType;
    }
    auto It = Substitutions.find(StringRef(ThisTypeStr));
    if (It == Substitutions.end())
      return false;

    unsigned SeqID = It->getValue();
    Stream << 'S';
    mangleSequenceID(SeqID);
    return true;
  }

  std::string getPointeeMangling(RefParamType Pointee) {
    // Generates a simplified mangling of the pointee type for substitution
    // matching, without recursive substitutions.
    SmallString<32> Mangling;
    raw_svector_ostream ManglingStream(Mangling);

    while (const PointerType *P = SPIR::dynCast<PointerType>(Pointee.get())) {
      ManglingStream << 'P' << getPointerAttributesMangling(P);
      Pointee = P->getPointee();
    }

    if (const UserDefinedType *U =
            SPIR::dynCast<UserDefinedType>(Pointee.get())) {
      StringRef Name = U->getName();
      ManglingStream << Name.size() << Name;
    } else {
      std::string LeafMangling = getLeafTypeMangling(Pointee.get());
      if (!LeafMangling.empty())
        ManglingStream << LeafMangling;
    }
    return std::string(Mangling);
  }
  void recordSubstitution(StringRef Str) { Substitutions[Str] = SeqId++; }

  SmallVectorImpl<char> &Buffer;
  raw_ostream &Stream;
  unsigned SeqId;
  StringMap<unsigned> Substitutions;
};

// Name mangler
class NameMangler {
public:
  MangleError mangleTemplateName(StringRef Name,
                                 const std::vector<RefParamType> &TemplateArgs,
                                 std::string &MangledName) {
    SmallString<128> Buffer;
    raw_svector_ostream Stream(Buffer);
    Stream << "_Z" << Name.size() << Name;
    if (!TemplateArgs.empty()) {
      Stream << 'I';
      MangleVisitor Visitor(Buffer, Stream, 1);
      for (const auto &TemplateArg : TemplateArgs) {
        MangleError Err = TemplateArg->accept(&Visitor);
        if (Err != MANGLE_SUCCESS)
          return Err;
      }
      Stream << 'E';
    }
    MangledName = std::string(Buffer);
    return MANGLE_SUCCESS;
  }

  MangleError mangle(StringRef Name, const std::vector<RefParamType> &Params,
                     std::string &MangledName) {
    return mangle(Name, {}, Params, MangledName);
  }

  MangleError mangle(StringRef Name,
                     const std::vector<RefParamType> &TemplateArgs,
                     const std::vector<RefParamType> &Params,
                     std::string &MangledName) {
    SmallString<128> Buffer;
    raw_svector_ostream Stream(Buffer);
    Stream << "_Z" << Name.size() << Name;
    MangleVisitor Visitor(Buffer, Stream);
    if (!TemplateArgs.empty()) {
      Stream << 'I';
      for (const auto &TemplateArg : TemplateArgs) {
        MangleError Err = TemplateArg->accept(&Visitor);
        if (Err != MANGLE_SUCCESS)
          return Err;
      }
      Stream << 'E';
    }
    for (const auto &P : Params) {
      MangleError Err = P->accept(&Visitor);
      if (Err != MANGLE_SUCCESS)
        return Err;
    }
    MangledName = std::string(Buffer);
    return MANGLE_SUCCESS;
  }
};

} // namespace SPIR

// Bridge itanium_demangle::Node to SPIR::ParamType
class NodeToSPIRType {
  bool CharIsSigned;
  bool PreserveFloat16;
  bool IsNativeCPU;
  bool IsArch64;
  bool IsWindows;
  unsigned TargetDefaultAddrSpace;

public:
  NodeToSPIRType(const Triple &TT, bool CharIsSigned, bool PreserveFloat16)
      : CharIsSigned(CharIsSigned), PreserveFloat16(PreserveFloat16) {
    IsNativeCPU = TT.isNativeCPU();
    IsArch64 = TT.isArch64Bit() || IsNativeCPU;
    IsWindows = TT.isOSWindows();
    TargetDefaultAddrSpace = (TT.isSPIR() || TT.isSPIRV()) ? 4 : 0;
  }

  SPIR::RefParamType convert(const Node *N) {
    if (!N)
      return SPIR::RefParamType();

    switch (N->getKind()) {
    case Node::KNameType:
      return convertNameType(static_cast<const NameType *>(N));
    case Node::KPointerType:
      return convertPointerType(
          static_cast<const itanium_demangle::PointerType *>(N));
    case Node::KVectorType:
      return convertVectorType(
          static_cast<const itanium_demangle::VectorType *>(N));
    case Node::KQualType:
      return convert(static_cast<const QualType *>(N)->getChild());
    case Node::KTemplateParamQualifiedArg:
      return convertTemplateParamQualifiedArg(
          static_cast<const TemplateParamQualifiedArg *>(N));
    case Node::KSyntheticTemplateParamName:
      return convertSyntheticTemplateParamName(
          static_cast<const SyntheticTemplateParamName *>(N));
    case Node::KForwardTemplateReference:
      return convertForwardTemplateReference(
          static_cast<const ForwardTemplateReference *>(N));
    case Node::KVendorExtQualType:
      return convertVendorExtQualType(
          static_cast<const VendorExtQualType *>(N));
    case Node::KBinaryFPType:
      // BinaryFPType represents _Float16 (DF16_ in mangling).
      return SPIR::RefParamType(new SPIR::PrimitiveType(
          PreserveFloat16 ? SPIR::PRIMITIVE_FLOAT16 : SPIR::PRIMITIVE_HALF));
    default:
      return SPIR::RefParamType();
    }
  }

private:
  SPIR::RefParamType
  convertTemplateParamQualifiedArg(const TemplateParamQualifiedArg *TPQA) {
    const Node *Arg = nullptr;
    TPQA->match([&](Node *, Node *ArgNode) { Arg = ArgNode; });
    return convert(Arg);
  }

  SPIR::RefParamType
  convertSyntheticTemplateParamName(const SyntheticTemplateParamName *TPN) {
    TemplateParamKind Kind = TemplateParamKind::Type;
    unsigned Index = 0;
    TPN->match([&](TemplateParamKind K, unsigned I) {
      Kind = K;
      Index = I;
    });
    if (Kind != TemplateParamKind::Type)
      return SPIR::RefParamType();
    return SPIR::RefParamType(new SPIR::TemplateParameterType(Index));
  }

  SPIR::RefParamType
  convertForwardTemplateReference(const ForwardTemplateReference *FTR) {
    if (FTR->Ref)
      return convert(FTR->Ref);
    return SPIR::RefParamType(new SPIR::TemplateParameterType(FTR->Index));
  }

  // Convert a demangled type name to SPIR type representation.
  // Only C++ primitive types need explicit handling here. All other types
  // (OpenCL image types, event types, sampler, pipe, and user-defined structs)
  // automatically become UserDefinedType and pass through unchanged during
  // remangling. This design eliminates the need for 90+ type enums that
  // SPIRV-LLVM-Translator has - we only need the primitive types that can
  // appear in basic builtin signatures and may need transformation
  // (e.g., long long -> long, signed char -> char, _Float16 -> half).
  //
  // Examples of types handled as UserDefinedType:
  //   - ocl_event, ocl_sampler
  //   - ocl_image1d_ro, ocl_image2d_wo, ocl_image3d_rw, ...
  //   - pipe_ro_t, pipe_wo_t
  //   - Any user struct types
  SPIR::RefParamType convertNameType(const NameType *NT) {
    StringRef Name(NT->getName());
    SPIR::TypePrimitiveEnum Prim = SPIR::PRIMITIVE_INT;

    if (Name == "bool")
      Prim = SPIR::PRIMITIVE_BOOL;
    else if (Name == "char")
      Prim = CharIsSigned ? SPIR::PRIMITIVE_CHAR : SPIR::PRIMITIVE_UCHAR;
    else if (Name == "signed char")
      Prim = SPIR::PRIMITIVE_CHAR;
    else if (Name == "unsigned char")
      Prim = SPIR::PRIMITIVE_UCHAR;
    else if (Name == "short")
      Prim = SPIR::PRIMITIVE_SHORT;
    else if (Name == "unsigned short")
      Prim = SPIR::PRIMITIVE_USHORT;
    else if (Name == "int")
      Prim = SPIR::PRIMITIVE_INT;
    else if (Name == "unsigned int")
      Prim = SPIR::PRIMITIVE_UINT;
    else if (Name == "long")
      Prim =
          (IsArch64 && !IsWindows) ? SPIR::PRIMITIVE_LONG : SPIR::PRIMITIVE_INT;
    else if (Name == "unsigned long")
      Prim = (IsArch64 && !IsWindows) ? SPIR::PRIMITIVE_ULONG
                                      : SPIR::PRIMITIVE_UINT;
    else if (Name == "long long")
      Prim = SPIR::PRIMITIVE_LONG;
    else if (Name == "unsigned long long")
      Prim = SPIR::PRIMITIVE_ULONG;
    else if (Name == "_Float16")
      Prim = PreserveFloat16 ? SPIR::PRIMITIVE_FLOAT16 : SPIR::PRIMITIVE_HALF;
    else if (Name == "half")
      Prim = SPIR::PRIMITIVE_HALF;
    else if (Name == "float")
      Prim = SPIR::PRIMITIVE_FLOAT;
    else if (Name == "double")
      Prim = SPIR::PRIMITIVE_DOUBLE;
    else if (Name == "void")
      Prim = SPIR::PRIMITIVE_VOID;
    else
      return SPIR::RefParamType(new SPIR::UserDefinedType(Name.str()));

    return SPIR::RefParamType(new SPIR::PrimitiveType(Prim));
  }

  SPIR::RefParamType
  convertPointerType(const itanium_demangle::PointerType *PT) {
    const Node *PointeeNode = PT->getPointee();

    // Extract address space qualifier from pointer type.
    //
    // In Itanium mangling, address space qualifiers are encoded as vendor
    // extensions in the pointee type, not the pointer itself. We extract
    // the AS qualifier here and apply it to the pointer (which is how SPIR
    // represents it), then continue processing the unwrapped pointee type.
    //
    // Note: We must extract address space BEFORE CV-qualifiers because the
    // structure is: PointerType -> VendorExtQualType(AS) -> QualType(CV) ->
    // BaseType
    //
    // Initialize AS for implicit pointers (no AS qualifier in mangled name).
    // On SPIR targets: implicit pointers are generic (ATTR_GENERIC).
    //   Transform: "Pf" (implicit) -> "PU3AS4f" (explicit AS4)
    // On other targets: implicit pointers are private (ATTR_PRIVATE).
    //   No transform: "Pf" (implicit) -> "Pf" (stays implicit)
    SPIR::TypeAttributeEnum AS =
        (TargetDefaultAddrSpace == 4) ? SPIR::ATTR_GENERIC : SPIR::ATTR_PRIVATE;
    if (PointeeNode->getKind() == Node::KVendorExtQualType) {
      const auto *VT = static_cast<const VendorExtQualType *>(PointeeNode);
      StringRef Ext(VT->getExt());
      if (Ext.starts_with("AS")) {
        // Parse explicit AS qualifier from mangled name.
        // On SPIR targets, explicit AS0 becomes implicit (no qualifier).
        //   Transform: "PU3AS0f" (explicit AS0) -> "Pf" (implicit)
        int ASNum;
        if (!Ext.drop_front(2).getAsInteger(10, ASNum)) {
          if (ASNum == 0)
            AS = SPIR::ATTR_PRIVATE;
          else if (ASNum == 1)
            AS = SPIR::ATTR_GLOBAL;
          else if (ASNum == 2)
            AS = SPIR::ATTR_CONSTANT;
          else if (ASNum == 3)
            AS = SPIR::ATTR_LOCAL;
          else if (ASNum == 4)
            AS = SPIR::ATTR_GENERIC;
        }
        // Use the inner type, not the VendorExtQualType wrapper
        PointeeNode = VT->getTy();
      }
    }

    // Extract CV-qualifiers (const/volatile/restrict) from QualType nodes.
    // In Itanium mangling, qualifiers appear as QualType wrappers around the
    // actual pointee type. We need to extract and preserve these qualifiers
    // for correct OpenCL function signature matching.
    //
    // Example: pointer to const int with address space is represented as:
    //   PointerType -> VendorExtQualType(AS1) -> QualType(const) ->
    //   NameType("int")
    // And mangles as "PU3AS1Ki" where U3AS1=AS1, K=const, i=int
    bool IsConst = false;
    bool IsVolatile = false;
    bool IsRestrict = false;

    while (PointeeNode->getKind() == Node::KQualType) {
      const auto *QT = static_cast<const QualType *>(PointeeNode);
      Qualifiers Quals = QT->getQuals();
      if (Quals & QualConst)
        IsConst = true;
      if (Quals & QualVolatile)
        IsVolatile = true;
      if (Quals & QualRestrict)
        IsRestrict = true;
      PointeeNode = QT->getChild();
    }

    SPIR::RefParamType Pointee = convert(PointeeNode);
    if (Pointee.isNull())
      return SPIR::RefParamType();

    auto *Ptr = new SPIR::PointerType(Pointee);
    Ptr->setAddressSpace(AS);
    // Apply CV-qualifiers to the pointer
    Ptr->setQualifier(SPIR::ATTR_CONST, IsConst);
    Ptr->setQualifier(SPIR::ATTR_VOLATILE, IsVolatile);
    Ptr->setQualifier(SPIR::ATTR_RESTRICT, IsRestrict);
    return SPIR::RefParamType(Ptr);
  }

  SPIR::RefParamType convertVectorType(const itanium_demangle::VectorType *VT) {
    const Node *BaseNode = VT->getBaseType();
    SPIR::RefParamType Base = convert(BaseNode);
    if (Base.isNull())
      return SPIR::RefParamType();

    // Extract vector length from dimension node
    const Node *DimNode = VT->getDimension();
    if (DimNode->getKind() == Node::KNameType) {
      StringRef DimStr(static_cast<const NameType *>(DimNode)->getName());
      int Len;
      if (!DimStr.getAsInteger(10, Len))
        return SPIR::RefParamType(new SPIR::VectorType(Base, Len));
    }
    return SPIR::RefParamType();
  }

  SPIR::RefParamType convertVendorExtQualType(const VendorExtQualType *VT) {
    StringRef Ext(VT->getExt());
    SPIR::RefParamType Base = convert(VT->getTy());
    if (Base.isNull())
      return Base;

    // Handle address space qualifiers
    if (Ext.starts_with("AS")) {
      int AS;
      if (!Ext.drop_front(2).getAsInteger(10, AS)) {
        if (auto *PT = SPIR::dynCast<SPIR::PointerType>(Base.get())) {
          SPIR::TypeAttributeEnum Attr = SPIR::ATTR_PRIVATE;
          if (AS == 1)
            Attr = SPIR::ATTR_GLOBAL;
          else if (AS == 2)
            Attr = SPIR::ATTR_CONSTANT;
          else if (AS == 3)
            Attr = SPIR::ATTR_LOCAL;
          else if (AS == 4)
            Attr = SPIR::ATTR_GENERIC;
          PT->setAddressSpace(Attr);
        }
      }
    }
    return Base;
  }
};

class TypeTransformer {
public:
  SPIR::RefParamType transform(SPIR::RefParamType Type) {
    if (Type.isNull())
      return Type;

    if (const auto *Prim = SPIR::dynCast<SPIR::PrimitiveType>(Type.get())) {
      return transformPrimitive(Prim);
    } else if (const auto *Ptr = SPIR::dynCast<SPIR::PointerType>(Type.get())) {
      SPIR::RefParamType Pointee = transform(Ptr->getPointee());
      auto *NewPtr = new SPIR::PointerType(Pointee);
      NewPtr->setAddressSpace(Ptr->getAddressSpace());
      NewPtr->setQualifier(SPIR::ATTR_CONST,
                           Ptr->hasQualifier(SPIR::ATTR_CONST));
      NewPtr->setQualifier(SPIR::ATTR_VOLATILE,
                           Ptr->hasQualifier(SPIR::ATTR_VOLATILE));
      NewPtr->setQualifier(SPIR::ATTR_RESTRICT,
                           Ptr->hasQualifier(SPIR::ATTR_RESTRICT));
      return SPIR::RefParamType(NewPtr);
    } else if (const auto *Vec = SPIR::dynCast<SPIR::VectorType>(Type.get())) {
      SPIR::RefParamType Scalar = transform(Vec->getScalarType());
      return SPIR::RefParamType(new SPIR::VectorType(Scalar, Vec->getLength()));
    }
    return Type;
  }

private:
  SPIR::RefParamType transformPrimitive(const SPIR::PrimitiveType *Prim) {
    // All primitive type transformations are now handled in convertNameType.
    // This function just passes through the type unchanged.
    // See file header for transformation examples.
    return SPIR::RefParamType(new SPIR::PrimitiveType(Prim->getPrimitive()));
  }
};

// Simple allocator for demangling.
class SimpleAllocator {
  BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...A) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(A)...);
  }

  void *allocateNodeArray(size_t Sz) {
    return Alloc.Allocate(sizeof(Node *) * Sz, alignof(Node *));
  }
};

using Demangler = ManglingParser<SimpleAllocator>;

struct NameBoundaryParser : ManglingParser<SimpleAllocator> {
  using ManglingParser<SimpleAllocator>::ManglingParser;

  size_t getTemplateSuffixStart() const { return First - InputBegin; }

  const char *InputBegin = nullptr;
};

size_t findTemplateSuffixStart(StringRef MangledName) {
  if (!MangledName.starts_with("_Z"))
    return StringRef::npos;

  NameBoundaryParser Parser(MangledName.data() + 2,
                            MangledName.data() + MangledName.size());
  Parser.InputBegin = MangledName.data();
  const Node *ParsedName = Parser.parseName();
  if (!ParsedName || ParsedName->getKind() != Node::KNameWithTemplateArgs)
    return StringRef::npos;
  return Parser.getTemplateSuffixStart();
}

bool isValidRemangledBuiltinName(StringRef MangledName,
                                 StringRef ExpectedBaseName) {
  Demangler D(MangledName.data(), MangledName.data() + MangledName.size());
  const Node *Root = D.parse();
  if (!Root || Root->getKind() != Node::KFunctionEncoding)
    return false;

  const auto *Encoding = static_cast<const FunctionEncoding *>(Root);
  const Node *NameNode = Encoding->getName();
  return NameNode && StringRef(NameNode->getBaseName()) == ExpectedBaseName;
}

// NativeCPU libdevice implements a small set of subgroup and group builtins
// with _Float16 manglings (DF16_). Keep DF16_ for them.
// This workaround is not needed if they are moved to libspirv.
bool PreserveFloat16ForNativeCPU(StringRef BaseName) {
  return BaseName == "__spirv_GroupFMulKHR" ||
         BaseName == "__spirv_GroupFAdd" || BaseName == "__spirv_GroupFMin" ||
         BaseName == "__spirv_GroupFMax" ||
         BaseName == "__spirv_SubgroupShuffleINTEL" ||
         BaseName == "__spirv_SubgroupShuffleUpINTEL" ||
         BaseName == "__spirv_SubgroupShuffleDownINTEL" ||
         BaseName == "__spirv_SubgroupShuffleXorINTEL";
}

// Remangle a function if it's a SPIRV builtin.
// Returns the new mangled name if transformation is needed, or empty string if
// not. See SAFETY NOTE at the top of this file for why we transform all
// "__spirv_" functions.
std::string tryRemangleSPIRVBuiltin(StringRef MangledName, const Triple &TT,
                                    bool CharIsSigned) {
  // Check if it's a SPIRV builtin (simple string check)
  if (!MangledName.contains("__spirv_"))
    return "";

  // Demangle once
  Demangler D(MangledName.data(), MangledName.data() + MangledName.size());
  const Node *Root = D.parse();
  if (!Root || Root->getKind() != Node::KFunctionEncoding)
    return "";

  const auto *Encoding = static_cast<const FunctionEncoding *>(Root);

  // Check if function has template arguments.
  // For templated functions like __spirv_ImageRead<T>, the structure is:
  //   NameWithTemplateArgs -> Name (base name) + TemplateArgs (template params)
  // For non-templated functions, it's just a Name node.
  const Node *NameNode = Encoding->getName();
  StringRef BaseName = NameNode->getBaseName();
  bool HasTemplateArgs = NameNode->getKind() == Node::KNameWithTemplateArgs;

  // Validate no whitespace (would indicate malformed demangling).
  if (BaseName.contains(' '))
    return "";

  // Validate SPIRV builtin name.
  constexpr StringRef Prefix = "__spirv_";
  if (!BaseName.starts_with(Prefix) || BaseName.size() <= Prefix.size())
    return "";

  bool PreserveFloat16 =
      TT.isNativeCPU() && PreserveFloat16ForNativeCPU(BaseName);

  // Convert template arguments and function parameter types.
  NodeToSPIRType Converter(TT, CharIsSigned, PreserveFloat16);
  TypeTransformer Transformer;
  std::vector<SPIR::RefParamType> TemplateArgTypes;
  std::vector<SPIR::RefParamType> Params;

  if (HasTemplateArgs) {
    const auto *TemplatedName =
        static_cast<const NameWithTemplateArgs *>(NameNode);
    if (!TemplatedName->TemplateArgs ||
        TemplatedName->TemplateArgs->getKind() != Node::KTemplateArgs)
      return "";

    const auto *TemplateArgList =
        static_cast<const itanium_demangle::TemplateArgs *>(
            TemplatedName->TemplateArgs);
    for (const Node *TemplateArgNode : TemplateArgList->getParams()) {
      SPIR::RefParamType TemplateArgType = Converter.convert(TemplateArgNode);
      if (TemplateArgType.isNull())
        return "";
      TemplateArgType = Transformer.transform(TemplateArgType);
      TemplateArgTypes.push_back(TemplateArgType);
    }
  }

  NodeArray ParamNodes = Encoding->getParams();
  if (ParamNodes.empty() && TemplateArgTypes.empty())
    return "";

  for (const Node *ParamNode : ParamNodes) {
    SPIR::RefParamType ParamType = Converter.convert(ParamNode);
    if (ParamType.isNull()) {
      // If any parameter fails to convert, this is likely a
      // malformed/invalid mangling. Don't try to remangle it.
      return "";
    }
    ParamType = Transformer.transform(ParamType);
    Params.push_back(ParamType);
  }

  // Remangle using SPIR mangler. Templated functions are only handled
  // structurally when all template arguments can be represented as supported
  // type nodes; otherwise we conservatively skip remangling.
  std::string Result;
  SPIR::NameMangler Mangler;
  if (HasTemplateArgs) {
    size_t TemplateSuffixStart = findTemplateSuffixStart(MangledName);
    if (TemplateSuffixStart == StringRef::npos)
      return "";

    std::string RemangledTemplatePrefix;
    if (Mangler.mangleTemplateName(BaseName, TemplateArgTypes,
                                   RemangledTemplatePrefix) !=
        SPIR::MANGLE_SUCCESS)
      return "";

    Result =
        RemangledTemplatePrefix + MangledName.substr(TemplateSuffixStart).str();
  } else if (Mangler.mangle(BaseName, Params, Result) != SPIR::MANGLE_SUCCESS) {
    return "";
  }

  // Final validation: ensure the new mangled name is different from the
  // original. If they're the same or if Result is malformed, don't transform.
  if (Result == MangledName)
    return "";

  assert(isValidRemangledBuiltinName(Result, BaseName) &&
         "invalid remangled builtin name");

  return Result;
}

} // anonymous namespace

PreservedAnalyses SYCLBuiltinRemanglePass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  const Triple TT(M.getTargetTriple());

  // Skip SPIR/SPIRV targets by default unless explicitly enabled for testing.
  // Nevertheless, SPIR/SPIRV target that by-pass SPIR-V generation could enable
  // this pass to transform address space mangling.
  if ((TT.isSPIR() || TT.isSPIRV()) && !RemangleSPIRVTarget)
    return PreservedAnalyses::all();

  bool IsCharSigned = RemangleCharIsSigned.getNumOccurrences() > 0
                          ? RemangleCharIsSigned
                          : CharIsSigned;

  SmallVector<std::pair<Function *, std::string>, 16> ToRename;

  // Pass 1: Identify functions to rename and give them temporary names
  // to avoid collisions when function A renames to name of function B,
  // and function B also needs to be renamed.
  for (Function &F : M) {
    if (F.isIntrinsic() || !F.getName().starts_with("_Z"))
      continue;

    if (!F.isDeclaration())
      continue;

    StringRef Name = F.getName();
    std::string NewName = tryRemangleSPIRVBuiltin(Name, TT, IsCharSigned);
    if (!NewName.empty() && NewName != Name) {
      F.setName(Name + ".remangle");
      ToRename.push_back({&F, std::move(NewName)});
    }
  }

  // Pass 2: Rename from temporary names to final names and handle merging
  for (auto &[F, NewName] : ToRename) {
    Function *Existing = M.getFunction(NewName);
    if (Existing) {
      assert(F->getFunctionType() == Existing->getFunctionType() &&
             "Remangled function type mismatch with existing function");
      F->replaceAllUsesWith(Existing);
      F->eraseFromParent();
    } else {
      F->setName(NewName);
    }
  }

  return ToRename.empty() ? PreservedAnalyses::all()
                          : PreservedAnalyses::none();
}
