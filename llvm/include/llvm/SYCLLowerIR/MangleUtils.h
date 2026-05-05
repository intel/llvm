//===- MangleUtils.h - SPIR mangling helpers for SYCL -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contains code derived from SPIRV-LLVM Translator. License available at:
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/LICENSE.TXT
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_MANGLEUTILS_H
#define LLVM_SYCLLOWERIR_MANGLEUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

namespace SPIR {

static constexpr unsigned ADDRESS_SPACE_GENERIC = ~0u;

enum TypePrimitiveEnum {
  PRIMITIVE_BOOL,
  PRIMITIVE_SCHAR,
  PRIMITIVE_UCHAR,
  PRIMITIVE_CHAR,
  PRIMITIVE_USHORT,
  PRIMITIVE_SHORT,
  PRIMITIVE_UINT,
  PRIMITIVE_INT,
  PRIMITIVE_ULONG,
  PRIMITIVE_LONG,
  PRIMITIVE_ULONGLONG,
  PRIMITIVE_LONGLONG,
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

enum TypeAttributeEnum { ATTR_RESTRICT, ATTR_VOLATILE, ATTR_CONST };

struct TypeVisitor;
struct PrimitiveType;
struct VectorType;
struct PointerType;
struct TemplateParameterType;
struct UserDefinedType;

struct ParamType : public RefCountedBase<ParamType> {
  ParamType(TypeEnum TypeId) : TypeId(TypeId) {}
  virtual ~ParamType() {}
  virtual void accept(TypeVisitor *) const = 0;
  TypeEnum getTypeId() const { return TypeId; }

protected:
  TypeEnum TypeId;
};

typedef IntrusiveRefCntPtr<ParamType> RefParamType;

template <typename Derived, TypeEnum EnumVal>
struct ParamTypeBase : public ParamType {
  ParamTypeBase() : ParamType(EnumVal) {}

  static bool classof(const ParamType *P) { return P->getTypeId() == EnumVal; }
};

struct PrimitiveType : public ParamTypeBase<PrimitiveType, TYPE_ID_PRIMITIVE> {
  PrimitiveType(TypePrimitiveEnum P) : Primitive(P) {}
  void accept(TypeVisitor *Visitor) const override;
  TypePrimitiveEnum getPrimitive() const { return Primitive; }

private:
  TypePrimitiveEnum Primitive;
};

struct PointerType : public ParamTypeBase<PointerType, TYPE_ID_POINTER> {
  PointerType(const RefParamType Type) : PType(Type) {
    Qualifiers[0] = Qualifiers[1] = Qualifiers[2] = false;
  }
  void accept(TypeVisitor *Visitor) const override;
  const RefParamType &getPointee() const { return PType; }
  void setAddressSpace(unsigned AS) { AddressSpace = AS; }
  unsigned getAddressSpace() const { return AddressSpace; }
  void setQualifier(TypeAttributeEnum Qual, bool Enabled);
  bool hasQualifier(TypeAttributeEnum Qual) const;

private:
  RefParamType PType;
  bool Qualifiers[3];
  unsigned AddressSpace = ADDRESS_SPACE_GENERIC;
};

struct VectorType : public ParamTypeBase<VectorType, TYPE_ID_VECTOR> {
  VectorType(const RefParamType Type, int Len) : PType(Type), Len(Len) {}
  void accept(TypeVisitor *Visitor) const override;
  const RefParamType &getScalarType() const { return PType; }
  int getLength() const { return Len; }

private:
  RefParamType PType;
  int Len;
};

struct TemplateParameterType
    : public ParamTypeBase<TemplateParameterType, TYPE_ID_TEMPLATE_PARAMETER> {
  TemplateParameterType(unsigned Index) : Index(Index) {}
  void accept(TypeVisitor *Visitor) const override;
  unsigned getIndex() const { return Index; }

private:
  unsigned Index;
};

struct UserDefinedType
    : public ParamTypeBase<UserDefinedType, TYPE_ID_STRUCTURE> {
  UserDefinedType(StringRef Name) : Name(Name) {}
  void accept(TypeVisitor *Visitor) const override;
  StringRef getName() const { return Name; }

private:
  SmallString<32> Name;
};

struct TypeVisitor {
  virtual ~TypeVisitor() {}
  virtual void visit(const PrimitiveType *) = 0;
  virtual void visit(const VectorType *) = 0;
  virtual void visit(const PointerType *) = 0;
  virtual void visit(const TemplateParameterType *) = 0;
  virtual void visit(const UserDefinedType *) = 0;
};

class NameMangler {
public:
  void mangleTemplateName(StringRef Name, ArrayRef<RefParamType> TemplateArgs,
                          SmallVectorImpl<char> &MangledName);

  void mangle(StringRef Name, ArrayRef<RefParamType> Params,
              SmallVectorImpl<char> &MangledName);
};

} // namespace SPIR

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_MANGLEUTILS_H
