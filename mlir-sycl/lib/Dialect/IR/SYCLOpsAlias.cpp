// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsAlias.cpp -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsAlias.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"

SYCLOpAsmInterface::AliasResult
SYCLOpAsmInterface::getAlias(mlir::Attribute Attr,
                             llvm::raw_ostream &OS) const {
  return AliasResult::NoAlias;
}

SYCLOpAsmInterface::AliasResult
SYCLOpAsmInterface::getAlias(mlir::Type Type, llvm::raw_ostream &OS) const {
  if (const auto ID = Type.dyn_cast<mlir::sycl::IDType>()) {
    OS << "sycl_id_" << ID.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto Acc = Type.dyn_cast<mlir::sycl::AccessorType>()) {
    OS << "sycl_accessor"
       << "_" << Acc.getDimension() << "_" << Acc.getType() << "_"
       << Acc.getAccessModeAsString() << "_" << Acc.getTargetModeAsString();
    return AliasResult::FinalAlias;
  }
  if (const auto Range = Type.dyn_cast<mlir::sycl::RangeType>()) {
    OS << "sycl_range_" << Range.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto NdRange = Type.dyn_cast<mlir::sycl::NdRangeType>()) {
    OS << "sycl_nd_range_" << NdRange.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto AccDev = Type.dyn_cast<mlir::sycl::AccessorImplDeviceType>()) {
    OS << "sycl_accessor_impl_device_" << AccDev.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto AccSub = Type.dyn_cast<mlir::sycl::AccessorSubscriptType>()) {
    OS << "sycl_accessor_subscript_" << AccSub.getCurrentDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto Arr = Type.dyn_cast<mlir::sycl::ArrayType>()) {
    OS << "sycl_array_" << Arr.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto Item = Type.dyn_cast<mlir::sycl::ItemType>()) {
    OS << "sycl_item_" << Item.getDimension() << "_" << Item.getWithOffset();
    return AliasResult::FinalAlias;
  }
  if (const auto ItemBase = Type.dyn_cast<mlir::sycl::ItemBaseType>()) {
    OS << "sycl_item_base_" << ItemBase.getDimension() << "_"
       << ItemBase.getWithOffset();
    return AliasResult::FinalAlias;
  }
  if (const auto NDItem = Type.dyn_cast<mlir::sycl::NdItemType>()) {
    OS << "sycl_nd_item_" << NDItem.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto Group = Type.dyn_cast<mlir::sycl::GroupType>()) {
    OS << "sycl_group_" << Group.getDimension();
    return AliasResult::FinalAlias;
  }
  if (const auto Vec = Type.dyn_cast<mlir::sycl::VecType>()) {
    OS << "sycl_vec_" << Vec.getDataType() << "_" << Vec.getNumElements();
    return AliasResult::FinalAlias;
  }

  return AliasResult::NoAlias;
}
