//===---- TargetInfo.cpp - Encapsulate target details -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "clang/../../lib/CodeGen/CGCXXABI.h"
#include "clang/../../lib/CodeGen/TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

namespace mlirclang {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static bool isAggregateTypeForABI(QualType Ty) {
  return !CodeGen::CodeGenFunction::hasScalarEvaluationKind(Ty) ||
         Ty->isMemberFunctionPointerType();
}

static CGCXXABI::RecordArgABI getRecordArgABI(const RecordType *RT,
                                              CGCXXABI &CXXABI) {
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD) {
    if (!RT->getDecl()->canPassInRegisters())
      return CGCXXABI::RAA_Indirect;
    return CGCXXABI::RAA_Default;
  }
  return CXXABI.getRecordArgABI(RD);
}

static CGCXXABI::RecordArgABI getRecordArgABI(QualType T, CGCXXABI &CXXABI) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return CGCXXABI::RAA_Default;
  return getRecordArgABI(RT, CXXABI);
}

static bool classifyReturnType(const CGCXXABI &CXXABI, CGFunctionInfo &FI,
                               const ABIInfo &Info) {
  QualType Ty = FI.getReturnType();

  if (const auto *RT = Ty->getAs<RecordType>())
    if (!isa<CXXRecordDecl>(RT->getDecl()) &&
        !RT->getDecl()->canPassInRegisters()) {
      FI.getReturnInfo() = Info.getNaturalAlignIndirect(Ty);
      return true;
    }

  return CXXABI.classifyReturnType(FI);
}

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
static QualType useFirstFieldIfTransparentUnion(QualType Ty) {
  if (const RecordType *UT = Ty->getAsUnionType()) {
    const RecordDecl *UD = UT->getDecl();
    if (UD->hasAttr<TransparentUnionAttr>()) {
      assert(!UD->field_empty() && "sema created an empty transparent union");
      return UD->field_begin()->getType();
    }
  }
  return Ty;
}

static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays);

/// isEmptyField - Return true iff a the field is "empty", that is it
/// is an unnamed bit-field or an (array of) empty record(s).
static bool isEmptyField(ASTContext &Context, const FieldDecl *FD,
                         bool AllowArrays) {
  if (FD->isUnnamedBitfield())
    return true;

  QualType FT = FD->getType();

  // Constant arrays of empty records count as empty, strip them off.
  // Constant arrays of zero length always count as empty.
  bool WasArray = false;
  if (AllowArrays)
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT)) {
      if (AT->getSize() == 0)
        return true;
      FT = AT->getElementType();
      // The [[no_unique_address]] special case below does not apply to
      // arrays of C++ empty records, so we need to remember this fact.
      WasArray = true;
    }

  const RecordType *RT = FT->getAs<RecordType>();
  if (!RT)
    return false;

  // C++ record fields are never empty, at least in the Itanium ABI.
  //
  // FIXME: We should use a predicate for whether this behavior is true in the
  // current ABI.
  //
  // The exception to the above rule are fields marked with the
  // [[no_unique_address]] attribute (since C++20).  Those do count as empty
  // according to the Itanium ABI.  The exception applies only to records,
  // not arrays of records, so we must also check whether we stripped off an
  // array type above.
  if (isa<CXXRecordDecl>(RT->getDecl()) &&
      (WasArray || !FD->hasAttr<NoUniqueAddressAttr>()))
    return false;

  return isEmptyRecord(Context, FT, AllowArrays);
}

/// isEmptyRecord - Return true iff a structure contains only empty
/// fields. Note that a structure with a flexible array member is not
/// considered empty.
static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (const auto &I : CXXRD->bases())
      if (!isEmptyRecord(Context, I.getType(), true))
        return false;

  for (const auto *I : RD->fields())
    if (!isEmptyField(Context, I, AllowArrays))
      return false;
  return true;
}

/// isSingleElementStruct - Determine if a structure is a "single
/// element struct", i.e. it has exactly one non-empty field or
/// exactly one field which is itself a single element
/// struct. Structures with flexible array members are never
/// considered single element structs.
///
/// \return The field declaration for the single non-empty field, if
/// it exists.
static const Type *isSingleElementStruct(QualType T, ASTContext &Context) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return nullptr;

  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return nullptr;

  const Type *Found = nullptr;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : CXXRD->bases()) {
      // Ignore empty records.
      if (isEmptyRecord(Context, I.getType(), true))
        continue;

      // If we already found an element then this isn't a single-element struct.
      if (Found)
        return nullptr;

      // If this is non-empty and not a single element struct, the composite
      // cannot be a single element struct.
      Found = isSingleElementStruct(I.getType(), Context);
      if (!Found)
        return nullptr;
    }
  }

  // Check for single element.
  for (const auto *FD : RD->fields()) {
    QualType FT = FD->getType();

    // Ignore empty fields.
    if (isEmptyField(Context, FD, true))
      continue;

    // If we already found an element then this isn't a single-element
    // struct.
    if (Found)
      return nullptr;

    // Treat single element arrays as the element.
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT)) {
      if (AT->getSize().getZExtValue() != 1)
        break;
      FT = AT->getElementType();
    }

    if (!isAggregateTypeForABI(FT)) {
      Found = FT.getTypePtr();
    } else {
      Found = isSingleElementStruct(FT, Context);
      if (!Found)
        return nullptr;
    }
  }

  // We don't consider a struct a single-element struct if it has
  // padding beyond the element type.
  if (Found && Context.getTypeSize(Found) != Context.getTypeSize(T))
    return nullptr;

  return Found;
}

//===----------------------------------------------------------------------===//
// DefaultABIInfo implementation
//===----------------------------------------------------------------------===//

ABIArgInfo DefaultABIInfo::classifyArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
      return getNaturalAlignIndirect(Ty, RAA == CGCXXABI::RAA_DirectInMemory);

    return getNaturalAlignIndirect(Ty);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  ASTContext &Context = getContext();
  if (const auto *EIT = Ty->getAs<BitIntType>())
    if (EIT->getNumBits() >
        Context.getTypeSize(Context.getTargetInfo().hasInt128Type()
                                ? Context.Int128Ty
                                : Context.LongLongTy))
      return getNaturalAlignIndirect(Ty);

  return (isPromotableIntegerTypeForABI(Ty) ? ABIArgInfo::getExtend(Ty)
                                            : ABIArgInfo::getDirect());
}

ABIArgInfo DefaultABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return getNaturalAlignIndirect(RetTy);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  if (const auto *EIT = RetTy->getAs<BitIntType>())
    if (EIT->getNumBits() >
        getContext().getTypeSize(getContext().getTargetInfo().hasInt128Type()
                                     ? getContext().Int128Ty
                                     : getContext().LongLongTy))
      return getNaturalAlignIndirect(RetTy);

  return (isPromotableIntegerTypeForABI(RetTy) ? ABIArgInfo::getExtend(RetTy)
                                               : ABIArgInfo::getDirect());
}

void DefaultABIInfo::computeInfo(CGFunctionInfo &FI) const {
  llvm_unreachable("Should never call this member function!");
}

Address DefaultABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                  QualType Ty) const {
  llvm_unreachable("Should never call this member function!");
}

//===----------------------------------------------------------------------===//
// CommonSPIRABIInfo implementation
//===----------------------------------------------------------------------===//

ABIArgInfo CommonSPIRABIInfo::classifyKernelArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (getContext().getLangOpts().SYCLIsDevice && isAggregateTypeForABI(Ty)) {
    // Pass all aggregate types allowed by Sema by value.
    return getNaturalAlignIndirect(Ty);
  }

  return DefaultABIInfo::classifyArgumentType(Ty);
}

// The two functions below are based on AMDGPUABIInfo, but without any
// restriction on the maximum number of arguments passed via registers.
// SPIRV BEs are expected to further adjust the calling convention as
// needed (use stack or byval-like passing) for some of the arguments.

ABIArgInfo CommonSPIRABIInfo::classifyRegcallReturnType(QualType RetTy) const {
  if (isAggregateTypeForABI(RetTy)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // returned by value.
    if (!getRecordArgABI(RetTy, getCXXABI())) {
      // Ignore empty structs/unions.
      if (isEmptyRecord(getContext(), RetTy, true))
        return ABIArgInfo::getIgnore();

      // Lower single-element structs to just return a regular value.
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

      if (const RecordType *RT = RetTy->getAs<RecordType>()) {
        const RecordDecl *RD = RT->getDecl();
        if (RD->hasFlexibleArrayMember())
          return classifyReturnType(RetTy);
      }

      // Pack aggregates <= 8 bytes into a single vector register or pair.
      // TODO make this parameterizeable/adjustable depending on spir target
      // triple abi component.
      uint64_t Size = getContext().getTypeSize(RetTy);
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

      if (Size <= 32)
        return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

      if (Size <= 64) {
        llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
        return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
      }
      return ABIArgInfo::getDirect();
    }
  }
  // Otherwise just do the default thing.
  return classifyReturnType(RetTy);
}

ABIArgInfo CommonSPIRABIInfo::classifyRegcallArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
      return getNaturalAlignIndirect(Ty, RAA == CGCXXABI::RAA_DirectInMemory);

    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using getExpand(),
    // though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
      return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      if (RD->hasFlexibleArrayMember())
        return classifyArgumentType(Ty);
    }

    // Pack aggregates <= 8 bytes into single vector register or pair.
    // TODO make this parameterizeable/adjustable depending on spir target
    // triple abi component.
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size <= 64) {
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

      if (Size <= 32)
        return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

      // XXX: Should this be i64 instead, and should the limit increase?
      llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
    }
    return ABIArgInfo::getDirect();
  }

  // Otherwise just do the default thing.
  return classifyArgumentType(Ty);
}

void CommonSPIRABIInfo::setCCs() {
  assert(getRuntimeCC() == llvm::CallingConv::C);
  RuntimeCC = llvm::CallingConv::SPIR_FUNC;
}

} // namespace mlirclang
