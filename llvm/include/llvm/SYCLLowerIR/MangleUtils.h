//===- MangleUtils.h - SPIR mangling helpers for SYCL -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

enum MangleError { MANGLE_SUCCESS };

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

enum TypeAttributeEnum {
  ATTR_RESTRICT,
  ATTR_VOLATILE,
  ATTR_CONST,
  ATTR_PRIVATE,
  ATTR_GLOBAL,
  ATTR_CONSTANT,
  ATTR_LOCAL,
  ATTR_GENERIC,
  ATTR_GENERIC_EXPLICIT,
  ATTR_NONE,
  ATTR_NUM = ATTR_NONE
};

struct TypeVisitor;
struct PrimitiveType;
struct VectorType;
struct PointerType;
struct TemplateParameterType;
struct UserDefinedType;

struct ParamType : public RefCountedBase<ParamType> {
  ParamType(TypeEnum TypeId) : TypeId(TypeId) {}
  virtual ~ParamType() {}
  virtual MangleError accept(TypeVisitor *) const = 0;
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
  MangleError accept(TypeVisitor *Visitor) const override;
  TypePrimitiveEnum getPrimitive() const { return Primitive; }

private:
  TypePrimitiveEnum Primitive;
};

struct PointerType : public ParamTypeBase<PointerType, TYPE_ID_POINTER> {
  static const TypeEnum EnumTy;
  PointerType(const RefParamType Type)
      : PType(Type), AddressSpace(ATTR_PRIVATE) {
    Qualifiers[0] = Qualifiers[1] = Qualifiers[2] = false;
  }
  MangleError accept(TypeVisitor *Visitor) const override;
  const RefParamType &getPointee() const { return PType; }
  void setAddressSpace(TypeAttributeEnum Attr) { AddressSpace = Attr; }
  TypeAttributeEnum getAddressSpace() const { return AddressSpace; }
  void setQualifier(TypeAttributeEnum Qual, bool Enabled);
  bool hasQualifier(TypeAttributeEnum Qual) const;

private:
  RefParamType PType;
  bool Qualifiers[3];
  TypeAttributeEnum AddressSpace;
};

struct VectorType : public ParamTypeBase<PointerType, TYPE_ID_VECTOR> {
  static const TypeEnum EnumTy;
  VectorType(const RefParamType Type, int Len) : PType(Type), Len(Len) {}
  MangleError accept(TypeVisitor *Visitor) const override;
  const RefParamType &getScalarType() const { return PType; }
  int getLength() const { return Len; }

private:
  RefParamType PType;
  int Len;
};

struct TemplateParameterType
    : public ParamTypeBase<PointerType, TYPE_ID_TEMPLATE_PARAMETER> {
  static const TypeEnum EnumTy;
  TemplateParameterType(unsigned Index) : Index(Index) {}
  MangleError accept(TypeVisitor *Visitor) const override;
  unsigned getIndex() const { return Index; }

private:
  unsigned Index;
};

struct UserDefinedType : public ParamTypeBase<PointerType, TYPE_ID_STRUCTURE> {
  static const TypeEnum EnumTy;
  UserDefinedType(StringRef Name) : Name(Name) {}
  MangleError accept(TypeVisitor *Visitor) const override;
  StringRef getName() const { return Name; }

private:
  SmallString<32> Name;
};

struct TypeVisitor {
  virtual ~TypeVisitor() {}
  virtual MangleError visit(const PrimitiveType *) = 0;
  virtual MangleError visit(const VectorType *) = 0;
  virtual MangleError visit(const PointerType *) = 0;
  virtual MangleError visit(const TemplateParameterType *) = 0;
  virtual MangleError visit(const UserDefinedType *) = 0;
};

class NameMangler {
public:
  MangleError mangleTemplateName(StringRef Name,
                                 ArrayRef<RefParamType> TemplateArgs,
                                 SmallVectorImpl<char> &MangledName);

  MangleError mangle(StringRef Name, ArrayRef<RefParamType> Params,
                     SmallVectorImpl<char> &MangledName);
};

} // namespace SPIR

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_MANGLEUTILS_H
