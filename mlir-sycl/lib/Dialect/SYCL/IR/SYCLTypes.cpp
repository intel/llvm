//===--- SYCLTypes.cpp ----------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace sycl {

static bool isMemRefWithExpectedShape(MemRefType mt) {
  return (mt && mt.hasRank() && (mt.getRank() == 1) &&
          ShapedType::isDynamic(mt.getShape()[0]) &&
          mt.getLayout().isIdentity());
}

bool SYCLType::classof(Type type) {
  return llvm::isa<SYCLDialect>(type.getDialect());
}

llvm::SmallVector<TypeID> getDerivedTypes(TypeID typeID) {
  if (typeID == AccessorCommonType::getTypeID())
    return {AccessorType::getTypeID()};
  if (typeID == ArrayType::getTypeID())
    return {IDType::getTypeID(), RangeType::getTypeID()};
  return {};
}

LogicalResult
VecType::verify(llvm::function_ref<InFlightDiagnostic()> emitError, Type dataT,
                int numElements, llvm::ArrayRef<Type>) {
  if (numElements != 1 && numElements != 2 && numElements != 3 &&
      numElements != 4 && numElements != 8 && numElements != 16) {
    return emitError() << "SYCL vector types can only hold 1, 2, 3, 4, 8 or 16 "
                          "elements. Got "
                       << numElements << ".";
  }

  return TypeSwitch<Type, LogicalResult>(dataT)
      .Case<IntegerType>([&](auto intTy) -> LogicalResult {
        const unsigned width = intTy.getWidth();
        if (width != 1 && width != 8 && width != 16 && width != 32 &&
            width != 64) {
          return emitError()
                 << "Integer SYCL vector element types can only be i1, "
                    "i8, i16, i32 or i64. Got "
                 << intTy << ".";
        }
        return success();
      })
      .Case<FloatType>([&](auto floatTy) -> LogicalResult {
        const unsigned width = dataT.cast<FloatType>().getWidth();
        if (width != 32 && width != 64) {
          return emitError() << "FP SYCL vector element types can only be "
                                "half, float or double. Got "
                             << floatTy << ".";
        }
        return success();
      })
      .Case<HalfType>([&](auto) { return success(); })
      .Default(
          [&](auto dataT) {
            return emitError()
                   << "SYCL vector types can only hold basic scalar types. Got "
                   << dataT << ".";
          });
}

unsigned getDimensions(Type type) {
  if (auto memRefTy = dyn_cast<MemRefType>(type))
    type = memRefTy.getElementType();
  return TypeSwitch<Type, unsigned>(type)
      .Case<AccessorType, BufferType, GroupType, IDType, ItemType, NdItemType,
            NdRangeType, RangeType>(
          [](auto type) { return type.getDimension(); });
}

template <typename T> bool isPtrOf(Type type) {
  auto mt = dyn_cast<MemRefType>(type);
  if (!isMemRefWithExpectedShape(mt))
    return false;

  return isa<T>(mt.getElementType());
}
template bool isPtrOf<AccessorType>(Type);
template bool isPtrOf<IDType>(Type);
template bool isPtrOf<ItemType>(Type);
template bool isPtrOf<NdItemType>(Type);

bool AccessorPtrValue::classof(Value v) {
  return isPtrOf<sycl::AccessorType>(v.getType());
}

bool isBaseClass(Type baseType, Type derivedType) {
  return TypeSwitch<Type, bool>(baseType)
      .Case<ArrayType, AccessorCommonType, LocalAccessorBaseType,
            OwnerLessBaseType>([&](auto bt) {
        return derivedType
            .hasTrait<SYCLInheritanceTypeTrait<decltype(bt)>::template Trait>();
      })
      .Default(false);
}

} // namespace sycl
} // namespace mlir
