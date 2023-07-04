//===--- SYCLOps.cpp ------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sycl;

bool SYCLCastOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  if (Inputs.size() != 1 || Outputs.size() != 1)
    return false;

  const auto Input = dyn_cast<MemRefType>(Inputs.front());
  const auto Output = dyn_cast<MemRefType>(Outputs.front());
  if (!Input || !Output)
    return false;

  // This is a hack - Since the sycl's CastOp takes as input/output MemRef, we
  // want to ensure that the cast is valid within MemRef's world.
  // In order to do that, we create a temporary Output that have the same
  // MemRef characteristic to check the MemRef cast without having the
  // ElementType triggering a condition like
  // (Input.getElementType() != Output.getElementType()).
  // This ElementType condition is checked later in this function.
  const auto TempOutput =
      MemRefType::get(Output.getShape(), Input.getElementType(),
                      Output.getLayout(), Output.getMemorySpace());
  if (!memref::CastOp::areCastCompatible(Input, TempOutput))
    return false;

  // Check whether the input element type is derived from the output element
  // type. If it is, the cast is legal.
  Type InputElemType = Input.getElementType();
  Type OutputElemType = Output.getElementType();

  return TypeSwitch<Type, bool>(OutputElemType)
      .template Case<ArrayType>([&](auto) {
        return InputElemType
            .hasTrait<SYCLInheritanceTypeTrait<ArrayType>::Trait>();
      })
      .template Case<AccessorCommonType>([&](auto) {
        return InputElemType
            .hasTrait<SYCLInheritanceTypeTrait<AccessorCommonType>::Trait>();
      })
      .template Case<LocalAccessorBaseType>([&](auto) {
        return InputElemType
            .hasTrait<SYCLInheritanceTypeTrait<LocalAccessorBaseType>::Trait>();
      })
      .template Case<OwnerLessBaseType>([&](auto) {
        return InputElemType
            .hasTrait<SYCLInheritanceTypeTrait<OwnerLessBaseType>::Trait>();
      })
      .Default(false);
}

bool SYCLAddrSpaceCastOp::areCastCompatible(TypeRange inputs,
                                            TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  const auto input = dyn_cast<MemRefType>(inputs.front());
  const auto output = dyn_cast<MemRefType>(outputs.front());
  if (!input || !output)
    return false;

  if (input.getShape() != output.getShape())
    return false;

  if (input.getElementType() != output.getElementType())
    return false;

  if (input.getLayout() != output.getLayout())
    return false;

  auto inputMS = dyn_cast_or_null<AccessAddrSpaceAttr>(input.getMemorySpace());
  auto outputMS =
      dyn_cast_or_null<AccessAddrSpaceAttr>(output.getMemorySpace());
  return (inputMS && outputMS &&
          (inputMS.getValue() == AccessAddrSpace::GenericAccess) !=
              (outputMS.getValue() == AccessAddrSpace::GenericAccess));
}

LogicalResult SYCLAccessorGetPointerOp::verify() {
  const auto accTy = cast<AccessorType>(
      cast<MemRefType>(getOperand().getType()).getElementType());
  const MemRefType resTy = getResult().getType();
  const Type resElemTy = resTy.getElementType();
  return (resElemTy != accTy.getType())
             ? emitOpError(
                   "Expecting a reference to this accessor's value type (")
                   << accTy.getType() << "). Got " << resTy
             : success();
}

LogicalResult SYCLAccessorGetRangeOp::verify() {
  const auto accTy = cast<AccessorType>(
      cast<MemRefType>(getOperand().getType()).getElementType());
  const RangeType resTy = getResult().getType();
  return (accTy.getDimension() != resTy.getDimension())
             ? emitOpError(
                   "Both the result and the accessor must have the same "
                   "number of dimensions, but the accessor has ")
                   << accTy.getDimension()
                   << " dimension(s) and the result has "
                   << resTy.getDimension() << " dimension(s)"
             : success();
}

LogicalResult SYCLAccessorSubscriptOp::verify() {
  // Available only when: (Dimensions > 0)
  // reference operator[](id<Dimensions> index) const;

  // Available only when: (Dimensions > 1)
  // __unspecified__ operator[](size_t index) const;

  // Available only when: (AccessMode != access_mode::atomic && Dimensions == 1)
  // reference operator[](size_t index) const;
  const auto AccessorTy = cast<AccessorType>(
      cast<MemRefType>(getOperand(0).getType()).getElementType());

  const unsigned Dimensions = AccessorTy.getDimension();
  if (Dimensions == 0)
    return emitOpError("Dimensions cannot be zero");

  const auto VerifyResultType = [&]() {
    const Type ResultType = getResult().getType();

    auto VerifyElemType = [&](const Type ElemType) {
      return (ElemType != AccessorTy.getType())
                 ? emitOpError(
                       "Expecting a reference to this accessor's value type (")
                       << AccessorTy.getType() << "). Got " << ResultType
                 : success();
    };

    return TypeSwitch<Type, LogicalResult>(ResultType)
        .Case<MemRefType>(
            [&](auto Ty) { return VerifyElemType(Ty.getElementType()); })
        .Case<LLVM::LLVMPointerType>([&](auto Ty) {
          if (!Ty.getElementType()) {
            // With opaque pointers, there is no element type to inspect.
            return success();
          }
          const Type ElemType = Ty.getElementType();
          return (!isa<LLVM::LLVMStructType>(ElemType))
                     ? emitOpError(
                           "Expecting pointer to struct return type. Got ")
                           << ResultType
                     : VerifyElemType(ElemType);
        })
        .Case<sycl::AtomicType>(
            [&](auto Ty) { return VerifyElemType(Ty.getDataType()); })
        .Default([this](auto Ty) {
          return emitOpError("Expecting memref/pointer return type. Got ")
                 << Ty;
        });
  };

  return TypeSwitch<Type, LogicalResult>(getOperand(1).getType())
      .Case<MemRefType>([&](auto MT) {
        const auto IDTy = MT.getElementType().template cast<IDType>();
        return (IDTy.getDimension() != Dimensions)
                   ? emitOpError(
                         "Both the index and the accessor must have the same "
                         "number of dimensions, but the accessor has ")
                         << Dimensions << " dimension(s) and the index has "
                         << IDTy.getDimension() << " dimension(s)"
                   : VerifyResultType();
      })
      .Case<IntegerType>([&](auto) {
        if (Dimensions != 1)
          return success(); // Implementation defined result type.

        return (AccessorTy.getAccessMode() == AccessMode::Atomic)
                   ? emitOpError(
                         "Cannot use this signature when the atomic access "
                         "mode is used")
                   : VerifyResultType();
      });
}

void SYCLConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // NOTE: This definition assumes only the first (`this`) argument is written
  // to. This is true for constructors run in the device, but not necessarily
  // true for constructors run in host code.
  auto *defaultResource = SideEffects::DefaultResource::get();
  // The `this` argument will always be written to
  effects.emplace_back(MemoryEffects::Write::get(), getDst(), defaultResource);
  // The rest of the arguments will be scalar or read from
  for (auto value : getArgs()) {
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
    }
  }
}

void SYCLIDConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto *defaultResource = SideEffects::DefaultResource::get();
  // All of the arguments will be scalar or read from
  for (auto value : getArgs()) {
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
    }
  }
  // The result will be allocated and written to
  Value id = getId();
  effects.emplace_back(MemoryEffects::Allocate::get(), id, defaultResource);
  effects.emplace_back(MemoryEffects::Write::get(), id, defaultResource);
}

LogicalResult SYCLIDConstructorOp::verify() {
  OperandRange::type_range argTypes = getArgs().getTypes();
  if (argTypes.empty()) {
    // sycl.id.constructor() -> memref<1x!sycl_id_N>
    return success();
  }
  MemRefType type = getId().getType();
  unsigned dimensions = getDimensions(type);
  if (llvm::all_of(argTypes, [](Type type) { return isa<IndexType>(type); })) {
    // sycl.id.constructor({index}N) -> memref<1x!sycl_id_N>
    if (argTypes.size() != dimensions) {
      return emitOpError("expects to be passed the same number of 'index' "
                         "numbers as the number of dimensions of the input: ")
             << argTypes.size() << " vs " << dimensions;
    }
    return success();
  }
  if (argTypes.size() == 1) {
    if (auto MT = dyn_cast<MemRefType>(argTypes.front());
        MT && isa<IDType, ItemType, RangeType>(MT.getElementType())) {
      // sycl.id.constructor(memref<?x!sycl_[id|item|range]_N>) ->
      // memref<1x!sycl_id_N>
      unsigned argDimensions = getDimensions(MT);
      if (dimensions != argDimensions) {
        return emitOpError("expects input and output to have the same number "
                           "of dimensions: ")
               << argDimensions << " vs " << dimensions;
      }
      return success();
    }
  }
  return emitOpError(
      "expects a different signature. Check documentation for details");
}

static Type getBodyType(Type type) {
  auto *ctx = type.getContext();
  return TypeSwitch<Type, Type>(type)
      .Case<HalfType>([&](auto) { return Float16Type::get(ctx); })
      .Default({});
}

bool SYCLWrapOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  Type sourceType = inputs.front();
  Type resultType = outputs.front();
  Type bodyType = getBodyType(resultType);

  return bodyType && sourceType == bodyType;
}

OpFoldResult SYCLWrapOp::fold(FoldAdaptor adaptor) {
  if (auto unwrap = dyn_cast_or_null<SYCLUnwrapOp>(getSource().getDefiningOp()))
    return unwrap.getSource();
  return nullptr;
}

bool SYCLUnwrapOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  Type sourceType = inputs.front();
  Type resultType = outputs.front();
  Type bodyType = getBodyType(sourceType);
  return bodyType && resultType == bodyType;
}

OpFoldResult SYCLUnwrapOp::fold(FoldAdaptor adaptor) {
  if (auto wrap = dyn_cast_or_null<SYCLWrapOp>(getSource().getDefiningOp()))
    return wrap.getSource();
  return nullptr;
}

static LogicalResult verifyReferencesKernel(SymbolUserOpInterface user,
                                            SymbolTableCollection &symbolTable,
                                            SymbolRefAttr symbol) {
  auto kernel =
      symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(user, symbol);
  if (!kernel || !kernel.isKernel())
    return user->emitOpError("'")
           << symbol << "' does not reference a valid kernel";
  return success();
}

LogicalResult
SYCLHostKernelNameOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

LogicalResult
SYCLHostGetKernelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

LogicalResult
SYCLHostHandlerSetKernel::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
