//===- TypeUtils.cc ----------------------------------------------*- C++-*-===//
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

#include "mlir/Dialect/SYCL/IR/SYCLOpAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
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

unsigned getAddressSpace(mlir::Type Ty) {
  return llvm::TypeSwitch<mlir::Type, unsigned>(Ty)
      .Case<mlir::MemRefType>(
          [](auto MemRefTy) { return MemRefTy.getMemorySpaceAsInt(); })
      .Case<mlir::LLVM::LLVMPointerType>(
          [](auto PtrTy) { return PtrTy.getAddressSpace(); });
}

mlir::Type getPtrTyWithNewType(mlir::Type Orig, mlir::Type NewElementType) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(Orig)
      .Case<mlir::MemRefType>([NewElementType](auto Ty) {
        return mlir::MemRefType::get(Ty.getShape(), NewElementType,
                                     Ty.getLayout(), Ty.getMemorySpace());
      })
      .Case<mlir::LLVM::LLVMPointerType>([NewElementType](auto Ty) {
        return mlir::LLVM::LLVMPointerType::get(NewElementType,
                                                Ty.getAddressSpace());
      });
}

template <typename F>
static typename std::invoke_result_t<F, uint32_t>::value_type
symbolizeAttr(const clang::TemplateArgument &templArg, F symbolize) {
  auto optVal =
      symbolize(static_cast<uint32_t>(templArg.getAsIntegral().getZExtValue()));
  assert(optVal && "Invalid enum value");
  return *optVal;
}

mlir::Type getSYCLType(const clang::RecordType *RT,
                       mlirclang::CodeGen::CodeGenTypes &CGT) {

  enum TypeEnum {
    // Same order as in SYCLOps.td
    AccessorCommon,
    AccessorImplDevice,
    Accessor,
    AccessorSubscript,
    Array,
    Atomic,
    Group,
    HItem,
    ID,
    ItemBase,
    Item,
    KernelHandler,
    LocalAccessorBaseDevice,
    LocalAccessorBase,
    LocalAccessor,
    Maximum,
    Minimum,
    MultiPtr,
    NdItem,
    NdRange,
    OwnerLessBase,
    Range,
    Stream,
    SubGroup,
    SwizzleOp,
    Vec
  };

  std::map<std::string, TypeEnum> StrToTypeEnum = {
      // Same order as in SYCLOps.td
      {"accessor_common", TypeEnum::AccessorCommon},
      {"AccessorImplDevice", TypeEnum::AccessorImplDevice},
      {"accessor", TypeEnum::Accessor},
      {"AccessorSubscript", TypeEnum::AccessorSubscript},
      {"array", TypeEnum::Array},
      {"atomic", TypeEnum::Atomic},
      {"group", TypeEnum::Group},
      {"h_item", TypeEnum::HItem},
      {"id", TypeEnum::ID},
      {"ItemBase", TypeEnum::ItemBase},
      {"item", TypeEnum::Item},
      {"kernel_handler", TypeEnum::KernelHandler},
      {"LocalAccessorBaseDevice", TypeEnum::LocalAccessorBaseDevice},
      {"local_accessor_base", TypeEnum::LocalAccessorBase},
      {"local_accessor", TypeEnum::LocalAccessor},
      {"maximum", TypeEnum::Maximum},
      {"minimum", TypeEnum::Minimum},
      {"multi_ptr", MultiPtr},
      {"nd_item", TypeEnum::NdItem},
      {"nd_range", TypeEnum::NdRange},
      {"OwnerLessBase", TypeEnum::OwnerLessBase},
      {"range", TypeEnum::Range},
      {"stream", TypeEnum::Stream},
      {"sub_group", SubGroup},
      {"SwizzleOp", SwizzleOp},
      {"vec", TypeEnum::Vec},
  };

  const clang::RecordDecl *RD = RT->getAsRecordDecl();
  llvm::SmallVector<mlir::Type, 4> Body;

  for (const auto *Field : RD->fields())
    Body.push_back(CGT.getMLIRTypeForMem(Field->getType()));

  if (const auto *CTS =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RD)) {

    switch (StrToTypeEnum[CTS->getName().str()]) {
    // Keep in alphabetical order.
    case TypeEnum::AccessorCommon:
      return mlir::sycl::AccessorCommonType::get(CGT.getModule()->getContext());
    case TypeEnum::AccessorImplDevice: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::AccessorImplDeviceType::get(
          CGT.getModule()->getContext(), Dim, Body);
    }
    case TypeEnum::Accessor: {
      const auto Type =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      const auto Dim =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      const auto MemAccessMode =
          symbolizeAttr(CTS->getTemplateArgs().get(2), [](uint32_t val) {
            return mlir::sycl::symbolizeAccessMode(val);
          });
      const auto MemTargetMode =
          symbolizeAttr(CTS->getTemplateArgs().get(3), [](uint32_t val) {
            return mlir::sycl::symbolizeTarget(val);
          });

      // The SYCL RT specialize the accessor class for local memory accesses.
      // That specialization is derived from a non-empty base class, so push it.
      // TODO: we should push the non-empty base classes in a more general way.
      if (MemTargetMode == mlir::sycl::Target::Local) {
        assert(Body.empty());
        Body.push_back(CGT.getMLIRTypeForMem(CTS->bases_begin()->getType()));
      }

      return mlir::sycl::AccessorType::get(CGT.getModule()->getContext(), Type,
                                           Dim, MemAccessMode, MemTargetMode,
                                           Body);
    }
    case TypeEnum::AccessorSubscript: {
      const auto CurDim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::AccessorSubscriptType::get(
          CGT.getModule()->getContext(), CurDim, Body);
    }
    case TypeEnum::Array: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::ArrayType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
    case TypeEnum::Atomic: {
      const auto Type =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      return mlir::sycl::AtomicType::get(
          CGT.getModule()->getContext(), Type,
          symbolizeAttr(CTS->getTemplateArgs().get(1),
                        [](uint32_t val) {
                          return mlir::sycl::symbolizeAccessAddrSpace(val);
                        }),
          Body);
    }
    case TypeEnum::Group: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::GroupType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
    case TypeEnum::HItem: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::HItemType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
    case TypeEnum::ID: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      Body.push_back(CGT.getMLIRTypeForMem(CTS->bases_begin()->getType()));
      return mlir::sycl::IDType::get(CGT.getModule()->getContext(), Dim, Body);
    }
    case TypeEnum::ItemBase: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemBaseType::get(CGT.getModule()->getContext(), Dim,
                                           Offset, Body);
    }
    case TypeEnum::Item: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemType::get(CGT.getModule()->getContext(), Dim,
                                       Offset, Body);
    }
    case TypeEnum::LocalAccessorBaseDevice: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::LocalAccessorBaseDeviceType::get(
          CGT.getModule()->getContext(), Dim, Body);
    }
    case TypeEnum::LocalAccessorBase: {
      const auto Type =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      const auto Dim =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      const auto MemAccessMode =
          symbolizeAttr(CTS->getTemplateArgs().get(2), [](uint32_t val) {
            return mlir::sycl::symbolizeAccessMode(val);
          });
      return mlir::sycl::LocalAccessorBaseType::get(
          CGT.getModule()->getContext(), Type, Dim, MemAccessMode, Body);
    }
    case TypeEnum::LocalAccessor: {
      const auto Type =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      const auto Dim =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      Body.push_back(CGT.getMLIRTypeForMem(CTS->bases_begin()->getType()));
      return mlir::sycl::LocalAccessorType::get(CGT.getModule()->getContext(),
                                                Type, Dim, Body);
    }
    case TypeEnum::Maximum: {
      const auto Type =
          CGT.getMLIRType(CTS->getTemplateArgs().get(0).getAsType());
      return mlir::sycl::MaximumType::get(CGT.getModule()->getContext(), Type);
    }
    case TypeEnum::Minimum: {
      const auto Type =
          CGT.getMLIRType(CTS->getTemplateArgs().get(0).getAsType());
      return mlir::sycl::MinimumType::get(CGT.getModule()->getContext(), Type);
    }
    case TypeEnum::MultiPtr: {
      const auto Type =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      return mlir::sycl::MultiPtrType::get(
          CGT.getModule()->getContext(), Type,
          symbolizeAttr(CTS->getTemplateArgs().get(1),
                        [](uint32_t val) {
                          return mlir::sycl::symbolizeAccessAddrSpace(val);
                        }),
          symbolizeAttr(CTS->getTemplateArgs().get(2),
                        [](uint32_t val) {
                          return mlir::sycl::symbolizeAccessDecorated(val);
                        }),
          Body);
    }
    case TypeEnum::NdItem: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::NdItemType::get(CGT.getModule()->getContext(), Dim,
                                         Body);
    }
    case TypeEnum::NdRange: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::NdRangeType::get(CGT.getModule()->getContext(), Dim,
                                          Body);
    }
    case TypeEnum::OwnerLessBase:
      return mlir::sycl::OwnerLessBaseType::get(CGT.getModule()->getContext());
    case TypeEnum::Range: {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      Body.push_back(CGT.getMLIRTypeForMem(CTS->bases_begin()->getType()));
      return mlir::sycl::RangeType::get(CGT.getModule()->getContext(), Dim,
                                        Body);
    }
    case TypeEnum::Vec: {
      const auto ElemType =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType());
      const auto NumElems =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::VecType::get(CGT.getModule()->getContext(), ElemType,
                                      NumElems, Body);
    }
    case TypeEnum::SwizzleOp: {
      const auto VecType =
          CGT.getMLIRTypeForMem(CTS->getTemplateArgs().get(0).getAsType())
              .cast<mlir::sycl::VecType>();
      const auto IndexesArgs = CTS->getTemplateArgs().get(4).getPackAsArray();
      SmallVector<int> Indexes;
      Indexes.reserve(IndexesArgs.size());
      std::transform(
          IndexesArgs.begin(), IndexesArgs.end(), std::back_inserter(Indexes),
          [](const auto &Arg) { return Arg.getAsIntegral().getSExtValue(); });
      return mlir::sycl::SwizzledVecType::get(CGT.getModule()->getContext(),
                                              VecType, Indexes, Body);
    }
    default:
      llvm_unreachable(
          "ClassTemplateSpecializationDecl: SYCL type not handled (yet)");
    }
  }

  if (const auto *CXXRD = llvm::dyn_cast<clang::CXXRecordDecl>(RD)) {
    switch (StrToTypeEnum[CXXRD->getName().str()]) {
    // Keep in alphabetical order.
    case TypeEnum::KernelHandler:
      return mlir::sycl::KernelHandlerType::get(CGT.getModule()->getContext(),
                                                Body);
    case TypeEnum::SubGroup:
      return mlir::sycl::SubGroupType::get(CGT.getModule()->getContext());
    case TypeEnum::Stream:
      return mlir::sycl::StreamType::get(CGT.getModule()->getContext(), Body);
    default:
      llvm_unreachable("CXXRecordDecl: SYCL type not handled (yet)");
    }
  }

  llvm_unreachable("SYCL type not handled (yet)");
}

llvm::Type *getLLVMType(const clang::QualType QT,
                        clang::CodeGen::CodeGenModule &CGM) {
  if (QT->isVoidType())
    return llvm::Type::getVoidTy(CGM.getModule().getContext());

  return CGM.getTypes().ConvertType(QT);
}

template <typename MLIRTy> static bool isTyOrTyVectorTy(mlir::Type Ty) {
  if (Ty.isa<MLIRTy>())
    return true;
  const auto VecTy = Ty.dyn_cast<mlir::VectorType>();
  return VecTy && VecTy.getElementType().isa<MLIRTy>();
}

bool isFPOrFPVectorTy(mlir::Type Ty) {
  return isTyOrTyVectorTy<mlir::FloatType>(Ty);
}

bool isIntOrIntVectorTy(mlir::Type Ty) {
  return isTyOrTyVectorTy<mlir::IntegerType>(Ty);
}

unsigned getPrimitiveSizeInBits(mlir::Type Ty) {
  return llvm::TypeSwitch<mlir::Type, unsigned>(Ty)
      .Case<mlir::IntegerType, mlir::FloatType>(
          [](auto Ty) { return Ty.getWidth(); })
      .Case<mlir::IndexType>(
          [](auto) { return mlir::IndexType::kInternalStorageBitWidth; })
      .Case<mlir::VectorType>([](auto VecTy) {
        return VecTy.getNumElements() *
               getPrimitiveSizeInBits(VecTy.getElementType());
      });
}

/// There must be at least an argument and it must be a memref to a SYCL type.
bool areSYCLMemberFunctionOrConstructorArgs(mlir::TypeRange Types) {
  return !Types.empty() &&
         TypeSwitch<mlir::Type, bool>(Types[0])
             .Case<mlir::MemRefType>([](auto Ty) {
               return mlir::sycl::isSYCLType(Ty.getElementType());
             })
             .Default(false);
}

} // namespace mlirclang
