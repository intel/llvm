//===- type utils.cc ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "CodeGenTypes.h"

#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Types.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

namespace mlirclang {

using namespace llvm;

bool isRecursiveStruct(Type *T, Type *Meta, SmallPtrSetImpl<Type *> &Seen) {
  if (Seen.count(T))
    return false;
  Seen.insert(T);
  if (T->isVoidTy() || T->isFPOrFPVectorTy() || T->isIntOrIntVectorTy())
    return false;
  if (T == Meta)
    return true;
  for (Type *ST : T->subtypes()) {
    if (isRecursiveStruct(ST, Meta, Seen))
      return true;
  }
  return false;
}

Type *anonymize(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    return PointerType::get(anonymize(PT->getPointerElementType()),
                            PT->getAddressSpace());
  if (auto *AT = dyn_cast<ArrayType>(T))
    return ArrayType::get(anonymize(AT->getElementType()),
                          AT->getNumElements());
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    SmallVector<Type *, 4> V;
    for (auto *T : FT->params())
      V.push_back(anonymize(T));
    return FunctionType::get(anonymize(FT->getReturnType()), V, FT->isVarArg());
  }
  if (auto *ST = dyn_cast<StructType>(T)) {
    if (ST->isLiteral())
      return ST;

    SmallVector<Type *, 4> V;
    for (auto *T : ST->elements()) {
      SmallPtrSet<Type *, 4> Seen;
      if (isRecursiveStruct(T, ST, Seen))
        V.push_back(T);
      else
        V.push_back(anonymize(T));
    }
    return StructType::get(ST->getContext(), V, ST->isPacked());
  }
  return T;
}

mlir::IntegerAttr wrapIntegerMemorySpace(unsigned MemorySpace,
                                         mlir::MLIRContext *Ctx) {
  return MemorySpace ? mlir::IntegerAttr::get(mlir::IntegerType::get(Ctx, 64),
                                              MemorySpace)
                     : nullptr;
}

mlir::Type getSYCLType(const clang::RecordType *RT,
                       mlirclang::CodeGen::CodeGenTypes &CGT) {
  const auto *RD = RT->getAsRecordDecl();
  llvm::SmallVector<mlir::Type, 4> Body;

  for (const auto *Field : RD->fields())
    Body.push_back(CGT.getMLIRType(Field->getType()));

  if (const auto *CTS =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RD)) {
    if (CTS->getName() == "range") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::RangeType::get(CGT.getModule()->getContext(), Dim);
    }
    if (CTS->getName() == "nd_range") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::NdRangeType::get(CGT.getModule()->getContext(), Dim,
                                          Body);
    }
    if (CTS->getName() == "array") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::ArrayType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
    if (CTS->getName() == "id") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::IDType::get(CGT.getModule()->getContext(), Dim);
    }
    if (CTS->getName() == "accessor_common") {
      return mlir::sycl::AccessorCommonType::get(CGT.getModule()->getContext());
    }
    if (CTS->getName() == "accessor") {
      const auto Type =
          CGT.getMLIRType(CTS->getTemplateArgs().get(0).getAsType());
      const auto Dim =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      const auto MemAccessMode = static_cast<mlir::sycl::MemoryAccessMode>(
          CTS->getTemplateArgs().get(2).getAsIntegral().getExtValue());
      const auto MemTargetMode = static_cast<mlir::sycl::MemoryTargetMode>(
          CTS->getTemplateArgs().get(3).getAsIntegral().getExtValue());
      return mlir::sycl::AccessorType::get(CGT.getModule()->getContext(), Type,
                                           Dim, MemAccessMode, MemTargetMode,
                                           Body);
    }
    if (CTS->getName() == "AccessorImplDevice") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::AccessorImplDeviceType::get(
          CGT.getModule()->getContext(), Dim, Body);
    }
    if (CTS->getName() == "item") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemType::get(CGT.getModule()->getContext(), Dim,
                                       Offset, Body);
    }
    if (CTS->getName() == "ItemBase") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemBaseType::get(CGT.getModule()->getContext(), Dim,
                                           Offset, Body);
    }
    if (CTS->getName() == "nd_item") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::NdItemType::get(CGT.getModule()->getContext(), Dim,
                                         Body);
    }
    if (CTS->getName() == "group") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::GroupType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
  }

  llvm_unreachable("SYCL type not handle (yet)");
}

llvm::Type *getLLVMType(const clang::QualType QT,
                        clang::CodeGen::CodeGenModule &CGM) {
  if (QT->isVoidType())
    return llvm::Type::getVoidTy(CGM.getModule().getContext());

  return CGM.getTypes().ConvertType(QT);
}

} // namespace mlirclang
