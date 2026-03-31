//===- MangleUtils.cpp - SPIR mangling helpers for SYCL -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contains code derived from SPIRV-LLVM Translator. License available at:
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/LICENSE.TXT
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/MangleUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::SPIR;

// Mangling utilities.
// clang-format off
#define PRIMITIVE_TYPES_MAP(X) \
  X(PRIMITIVE_BOOL,         "b")                                               \
  X(PRIMITIVE_SCHAR,        "a")                                               \
  X(PRIMITIVE_UCHAR,        "h")                                               \
  X(PRIMITIVE_CHAR,         "c")                                               \
  X(PRIMITIVE_USHORT,       "t")                                               \
  X(PRIMITIVE_SHORT,        "s")                                               \
  X(PRIMITIVE_UINT,         "j")                                               \
  X(PRIMITIVE_INT,          "i")                                               \
  X(PRIMITIVE_ULONG,        "m")                                               \
  X(PRIMITIVE_LONG,         "l")                                               \
  X(PRIMITIVE_ULONGLONG,    "y")                                               \
  X(PRIMITIVE_LONGLONG,     "x")                                               \
  X(PRIMITIVE_FLOAT16,      "DF16_")                                           \
  X(PRIMITIVE_HALF,         "Dh")                                              \
  X(PRIMITIVE_FLOAT,        "f")                                               \
  X(PRIMITIVE_DOUBLE,       "d")                                               \
  X(PRIMITIVE_VOID,         "v")
// clang-format on

StringRef mangledPrimitiveString(TypePrimitiveEnum T) {
  switch (T) {
#define TYPE_ENUM_CASE(Enum, Mangled)                                          \
  case Enum:                                                                   \
    return Mangled;
    PRIMITIVE_TYPES_MAP(TYPE_ENUM_CASE)
#undef TYPE_ENUM_CASE
  default:
    return {};
  }
}

#undef PRIMITIVE_TYPES_MAP

void appendTemplateParameterMangling(unsigned Index, raw_ostream &Stream) {
  Stream << 'T';
  if (Index > 0)
    Stream << (Index - 1);
  Stream << '_';
}

SmallString<8> getLeafTypeMangling(const ParamType *Type) {
  if (const auto *Prim = dyn_cast<PrimitiveType>(Type)) {
    SmallString<8> Buffer;
    Buffer += mangledPrimitiveString(Prim->getPrimitive());
    return Buffer;
  }
  if (const auto *TemplateParam = dyn_cast<TemplateParameterType>(Type)) {
    SmallString<8> Buffer;
    raw_svector_ostream Stream(Buffer);
    appendTemplateParameterMangling(TemplateParam->getIndex(), Stream);
    return Buffer;
  }
  return {};
}

// Address space mangling:
// - Private (AS0) is explicit (U3AS0) - matches SYCL conventions.
// clang-format off
#define ATTRIBUTE_TYPES_MAP(X)                                                 \
  X(ATTR_RESTRICT,          "r")                                               \
  X(ATTR_VOLATILE,          "V")                                               \
  X(ATTR_CONST,             "K")                                               \
  X(ATTR_PRIVATE,           "U3AS0")                                           \
  X(ATTR_GLOBAL,            "U3AS1")                                           \
  X(ATTR_CONSTANT,          "U3AS2")                                           \
  X(ATTR_LOCAL,             "U3AS3")                                           \
  X(ATTR_GENERIC,           "")                                                \
  X(ATTR_GENERIC_EXPLICIT,  "U3AS4")
// clang-format on

StringRef getMangledAttribute(TypeAttributeEnum Attribute) {
  switch (Attribute) {
#define ATTR_ENUM_CASE(Enum, Mangled)                                          \
  case Enum:                                                                   \
    return Mangled;
    ATTRIBUTE_TYPES_MAP(ATTR_ENUM_CASE)
#undef ATTR_ENUM_CASE
  default:
    return {};
  }
}

#undef ATTRIBUTE_TYPES_MAP

MangleError PrimitiveType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

MangleError PointerType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
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

MangleError VectorType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

MangleError TemplateParameterType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

MangleError UserDefinedType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
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
    SmallString<8> AttrMangling = getPointerAttributesManglingWithMode(P);
    SmallString<8> PtrTypePrefix("P");
    PtrTypePrefix += AttrMangling;
    if (!mangleSubstitution(P, PtrTypePrefix)) {
      Stream << PtrTypePrefix;
      MangleError Err =
          P->getPointee()->accept(static_cast<TypeVisitor *>(this));
      if (!AttrMangling.empty())
        recordSubstitution(currentBuffer().substr(Pos + 1));
      recordSubstitution(currentBuffer().substr(Pos));
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
      recordSubstitution(currentBuffer().substr(Index));
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
      recordSubstitution(currentBuffer().substr(Index));
    }
    return MANGLE_SUCCESS;
  }

private:
  StringRef currentBuffer() const {
    return StringRef(Buffer.data(), Buffer.size());
  }

  SmallString<8> getPointerAttributesManglingWithMode(const PointerType *P) {
    SmallString<8> QualStr;
    // Handle address space.
    TypeAttributeEnum AS = P->getAddressSpace();
    // Normal mangling (generic is implicit).
    QualStr += getMangledAttribute(AS);
    // Handle qualifiers.
    if (P->hasQualifier(ATTR_RESTRICT))
      QualStr += 'r';
    if (P->hasQualifier(ATTR_VOLATILE))
      QualStr += 'V';
    if (P->hasQualifier(ATTR_CONST))
      QualStr += 'K';
    return QualStr;
  }

  void mangleSequenceID(unsigned SeqID) {
    if (SeqID == 1)
      Stream << '0';
    else if (SeqID > 1) {
      SmallString<8> Bstr;
      constexpr StringRef Charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      SeqID--;
      for (; SeqID != 0; SeqID /= 36)
        Bstr.push_back(Charset[SeqID % 36]);
      for (char C : llvm::reverse(Bstr))
        Stream << C;
    }
    Stream << '_';
  }

  bool mangleSubstitution(const ParamType *Type, StringRef TypeStr) {
    if (currentBuffer().find(TypeStr) == StringRef::npos)
      return false;

    SmallString<32> ThisTypeStr(TypeStr);
    if (const auto *P = dyn_cast<PointerType>(Type)) {
      ThisTypeStr += getPointeeMangling(P->getPointee());
    } else if (const VectorType *PVec = dyn_cast<VectorType>(Type)) {
      SmallString<8> NType = getLeafTypeMangling(PVec->getScalarType().get());
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

  SmallString<32> getPointeeMangling(RefParamType Pointee) {
    // Generates a simplified mangling of the pointee type for substitution
    // matching, without recursive substitutions.
    SmallString<32> Mangling;
    raw_svector_ostream ManglingStream(Mangling);

    while (const auto *P = dyn_cast<PointerType>(Pointee.get())) {
      ManglingStream << 'P' << getPointerAttributesManglingWithMode(P);
      Pointee = P->getPointee();
    }

    if (const auto *U = dyn_cast<UserDefinedType>(Pointee.get())) {
      StringRef Name = U->getName();
      ManglingStream << Name.size() << Name;
    } else {
      SmallString<8> LeafMangling = getLeafTypeMangling(Pointee.get());
      if (!LeafMangling.empty())
        ManglingStream << LeafMangling;
    }
    return Mangling;
  }
  void recordSubstitution(StringRef Str) { Substitutions[Str] = SeqId++; }

  SmallVectorImpl<char> &Buffer;
  raw_ostream &Stream;
  unsigned SeqId;
  StringMap<unsigned> Substitutions;
};

MangleError
NameMangler::mangleTemplateName(StringRef Name,
                                ArrayRef<RefParamType> TemplateArgs,
                                SmallVectorImpl<char> &MangledName) {
  MangledName.clear();
  raw_svector_ostream Stream(MangledName);
  Stream << "_Z" << Name.size() << Name;
  if (!TemplateArgs.empty()) {
    Stream << 'I';
    MangleVisitor Visitor(MangledName, Stream, 1);
    for (const auto &TemplateArg : TemplateArgs) {
      MangleError Err = TemplateArg->accept(&Visitor);
      if (Err != MANGLE_SUCCESS)
        return Err;
    }
    Stream << 'E';
  }
  return MANGLE_SUCCESS;
}

MangleError NameMangler::mangle(StringRef Name, ArrayRef<RefParamType> Params,
                                SmallVectorImpl<char> &MangledName) {
  MangledName.clear();
  raw_svector_ostream Stream(MangledName);
  Stream << "_Z" << Name.size() << Name;
  MangleVisitor Visitor(MangledName, Stream, 0);
  for (const auto &P : Params) {
    MangleError Err = P->accept(&Visitor);
    if (Err != MANGLE_SUCCESS)
      return Err;
  }
  return MANGLE_SUCCESS;
}
