// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsTypes.cpp -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

llvm::SmallVector<mlir::TypeID>
mlir::sycl::getDerivedTypes(mlir::TypeID TypeID) {
  if (TypeID == mlir::sycl::AccessorCommonType::getTypeID())
    return {mlir::sycl::AccessorType::getTypeID()};
  if (TypeID == mlir::sycl::ArrayType::getTypeID())
    return {mlir::sycl::IDType::getTypeID(),
            mlir::sycl::RangeType::getTypeID()};
  return {};
}

mlir::LogicalResult
mlir::sycl::VecType::verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
                            mlir::Type DataT, int NumElements,
                            llvm::ArrayRef<mlir::Type>) {
  if (!(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
        NumElements == 4 || NumElements == 8 || NumElements == 16)) {
    return EmitError() << "SYCL vector types can only hold 1, 2, 3, 4, 8 or 16 "
                          "elements. Got "
                       << NumElements << ".";
  }

  if (!DataT.isa<IntegerType, FloatType>()) {
    return EmitError()
           << "SYCL vector types can only hold basic scalar types. Got "
           << DataT << ".";
  }

  if (auto IntTy = DataT.dyn_cast<IntegerType>()) {
    const auto Width = IntTy.getWidth();
    if (!(Width == 1 || Width == 8 || Width == 16 || Width == 32 ||
          Width == 64)) {
      return EmitError() << "Integer SYCL vector element types can only be i1, "
                            "i8, i16, i32 or i64. Got "
                         << Width << ".";
    }
  } else {
    const auto Width = DataT.cast<FloatType>().getWidth();
    if (!(Width == 16 || Width == 32 || Width == 64)) {
      return EmitError()
             << "FP SYCL vector element types can only be f16, f32 or f64. Got "
             << Width << ".";
    }
  }

  return success();
}

unsigned mlir::sycl::getDimensions(mlir::Type Type) {
  if (auto MemRefTy = dyn_cast<mlir::MemRefType>(Type))
    Type = MemRefTy.getElementType();
  return TypeSwitch<mlir::Type, unsigned>(Type)
      .Case<AccessorType, GroupType, IDType, ItemType, NdItemType, NdRangeType,
            RangeType>([](auto Ty) { return Ty.getDimension(); });
}
