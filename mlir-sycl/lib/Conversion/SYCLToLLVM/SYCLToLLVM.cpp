//===- SYCLToLLVM.cpp - SYCL to LLVM Patterns -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/SYCLToGPU/SYCLToGPU.h"
#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/Conversion/SYCLToSPIRV/SYCLToSPIRV.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOLLVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOLLVM
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Returns true if the given type is 'memref<?xSYCLType>', and false otherwise.
template <typename SYCLType> static bool isMemRefOf(const Type &type) {
  if (!type.isa<MemRefType>())
    return false;

  MemRefType memRefTy = type.cast<MemRefType>();
  ArrayRef<int64_t> shape = memRefTy.getShape();
  if (shape.size() != 1 || shape[0] != -1)
    return false;

  return memRefTy.getElementType().isa<SYCLType>();
}

// Returns the element type of 'memref<?xSYCLType>'.
template <typename SYCLType> static SYCLType getElementType(const Type &type) {
  assert(isMemRefOf<SYCLType>(type) && "Expecting memref<?xsycl::<type>>");
  Type elemType = type.cast<MemRefType>().getElementType();
  return elemType.cast<SYCLType>();
}

// Get LLVM struct type with i8 as the body with name \p name.
static Optional<Type> getI8Struct(StringRef name,
                                  LLVMTypeConverter &converter) {
  auto convertedTy =
      LLVM::LLVMStructType::getIdentified(&converter.getContext(), name);
  if (!convertedTy.isInitialized())
    if (failed(convertedTy.setBody(IntegerType::get(&converter.getContext(), 8),
                                   /*isPacked=*/false)))
      return std::nullopt;
  return convertedTy;
}

//===----------------------------------------------------------------------===//
// Tags definitions.
//===----------------------------------------------------------------------===//

/// Base class for other offset tags.
///
/// Offset tags will specify the indices to use in a GEP operation to reference
/// a struct field. In order to do so, tags must provide a `static constexpr`
/// array field called `indices`. E.g., each dimension in a range is accessed
/// through the indices [0, 0, 0] (being the first one to dereference the
/// pointer argument), so the `indices` field should hold the values [0, 0].
struct OffsetTag {};

/// Get a dimension from a range.
struct RangeGetDim : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 0};
};

/// Get a dimension from an ID.
struct IDGetDim : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 0};
};

/// Get the underlying pointer from an accessor.
struct AccessorGetPtr : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{1, 0};
};

/// Get the MemRange field from an accessor.
struct AccessorGetMemRange : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 2};
};

/// Get the global range from an nd_range.
struct NDRangeGetGlobalRange : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{0};
};

/// Get the local range from an nd_range.
struct NDRangeGetLocalRange : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{1};
};

/// Get the offset from an nd_range.
struct NDRangeGetOffset : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{2};
};

/// Get the ID field from an item.
struct ItemGetID : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 1};
};

/// Get the range field from an item.
struct ItemGetRange : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 0};
};

/// Get the offset field from an item.
struct ItemGetOffset : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 2};
};

/// Get the ID field from a group.
struct GroupGetID : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{3};
};

/// Get the global range field from a group.
struct GroupGetGlobalRange : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{0};
};

/// Get the local range field from a group.
struct GroupGetLocalRange : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{1};
};

/// Get the group range field from a group.
struct GroupGetGroupRange : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{2};
};

/// Get the global item field from an nd_item.
struct NDItemGlobalItem : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{0};
};

/// Get the local item field from an nd_item.
struct NDItemLocalItem : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{1};
};

/// Get the group field from an nd_item.
struct NDItemGroup : public OffsetTag {
  static constexpr std::array<int32_t, 1> indices{2};
};

/// Auxiliary function to build an indices array from a tag.
template <typename Iter, typename Tag,
          typename = std::enable_if_t<std::is_base_of_v<OffsetTag, Tag>>>
constexpr Iter initIndicesEach(Iter it) {
  for (auto index : Tag::indices) {
    // Fill the indices from the tag.
    *it++ = index;
  }
  return it;
}

/// Auxiliary constant to find the required size of the array holding a sequence
/// of indices from the input tags.
template <typename... Tags>
static constexpr std::size_t indices_size{(Tags::indices.size() + ...)};

//===----------------------------------------------------------------------===//
// Utility patterns
//===----------------------------------------------------------------------===//

/// Helper type to find whether the input parameter pack is empty.
template <typename...> struct is_empty : public std::false_type {};
template <> struct is_empty<> : public std::true_type {};

template <typename... Args>
static constexpr bool is_empty_v{is_empty<Args...>::value};

/// Base class for patterns accessing struct members.
///
/// Each derived class is intended to access a given member of a given class,
/// e.g., the underlying pointer of an accessor.
///
/// Derived classes must implement getIndices().
class GetMemberPatternBase {
public:
  virtual ~GetMemberPatternBase() = default;

protected:
  constexpr GetMemberPatternBase() = default;

  /// Returns a reference of type \p ty to a member of the struct pointed by \p
  /// ptr.
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<LLVM::GEPArg, Args...> ||
                is_empty_v<Args...>>>
  Value getRef(OpBuilder &builder, Location loc, Type ty, Value ptr,
               Args &&...args) const {
    SmallVector<LLVM::GEPArg> indices{0};
    const auto staticIndices = getIndices();
    indices.append(staticIndices.begin(), staticIndices.end());
    if constexpr (!is_empty_v<Args...>) {
      // Add additional index if provided.
      indices.emplace_back(std::forward<Args>(args)...);
    }
    return builder.create<LLVM::GEPOp>(loc, ty, ptr, indices,
                                       /*inbounds*/ true);
  }

  /// Returns a value of the type pointed by \p ty to a member of the struct
  /// pointed by \p ptr.
  ///
  /// Effectively calls getRef() and loads the value.
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<LLVM::GEPArg, Args...> ||
                is_empty_v<Args...>>>
  Value loadValue(OpBuilder &builder, Location loc, Type ty, Value ptr,
                  Args &&...args) const {
    const auto gep =
        getRef<Args...>(builder, loc, ty, ptr, std::forward<Args>(args)...);
    return builder.create<LLVM::LoadOp>(loc, gep);
  }

  /// Return the indices needed to access the specific member this class is
  /// intended to access.
  virtual ArrayRef<int32_t> getIndices() const = 0;
};

template <typename Iter, typename... Tags>
constexpr void initIndices(Iter begin) {
  static_assert(llvm::are_base_of<OffsetTag, Tags...>::value,
                "All input types must be offset tags.");
  ((begin = initIndicesEach<Iter, Tags>(begin)), ...);
}

template <typename... Tags>
class GetMemberPattern : public GetMemberPatternBase {
  static_assert(llvm::are_base_of<OffsetTag, Tags...>::value,
                "All input types must be offset tags.");

protected:
  ArrayRef<int32_t> getIndices() const final { return *indices; }

  using GetMemberPatternBase::GetMemberPatternBase;

private:
  /// Struct definition to allow constexpr initialization of indices.
  static constexpr struct GetMemberPatternIndices {
    static constexpr std::size_t size{indices_size<Tags...>};

    constexpr GetMemberPatternIndices() {
      initIndices<typename std::array<int32_t, size>::iterator, Tags...>(
          indices.begin());
    }

    ArrayRef<int32_t> operator*() const { return indices; }

    std::array<int32_t, size> indices{0};
  } indices{};
};

/// Base pattern for operations getting a reference to a struct member.
template <typename Op, typename... Tags>
class GetRefToMemberPattern : public GetMemberPattern<Tags...>,
                              public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(),
                                  getTypeConverter()->convertType(op.getType()),
                                  operands[0]));
  }
};

/// Base pattern for operations getting a reference to a given dimension of a
/// struct member.
template <typename Op, typename... Tags>
class GetRefToMemberDimPattern : public GetMemberPattern<Tags...>,
                                 public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(),
                                  getTypeConverter()->convertType(op.getType()),
                                  operands[0], operands[1]));
  }
};

/// Base pattern for operations loading a struct member.
template <typename Op, typename... Tags>
class LoadMemberPattern : public GetMemberPattern<Tags...>,
                          public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using GetMemberPattern<Tags...>::loadValue;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto operands = adaptor.getOperands();
    const auto addressSpace = operands[0]
                                  .getType()
                                  .template cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    rewriter.replaceOp(
        op, loadValue(rewriter, op.getLoc(),
                      LLVM::LLVMPointerType::get(
                          getTypeConverter()->convertType(op.getType()),
                          addressSpace),
                      operands[0]));
  }
};

/// Base pattern for operations loading a given dimension of a struct member.
template <typename Op, typename... Tags>
class LoadMemberDimPattern : public GetMemberPattern<Tags...>,
                             public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using GetMemberPattern<Tags...>::loadValue;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto operands = adaptor.getOperands();
    const auto addressSpace = operands[0]
                                  .getType()
                                  .template cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    rewriter.replaceOp(
        op, loadValue(rewriter, op.getLoc(),
                      LLVM::LLVMPointerType::get(
                          getTypeConverter()->convertType(op.getType()),
                          addressSpace),
                      operands[0], operands[1]));
  }
};

/// Pattern replacing an operation with a single argument with an instance of
/// the same operation with an additional 0 i32 constant argument.
template <typename Op>
class AddZeroArgPattern : public ConvertOpToLLVMPattern<Op> {
public:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op op) const final {
    return success(op.getNumOperands() == 1 && isa<IntegerType>(op.getType()));
  }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [op, &rewriter] {
      constexpr unsigned indexWidth{32};
      const Value zero =
          rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, indexWidth);
      op->insertOperands(1, zero);
    });
  }
};

/// Base pattern for operations calculating the size of a range.
///
/// The result is the accumulation (mul) of all of each dimension of the input
/// range.
template <typename Op>
class GetRangeSizePattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

  virtual Value getRange(OpBuilder &builder, Location loc, Type ptrTy,
                         Value thisArg, int32_t index) const = 0;

public:
  virtual ~GetRangeSizePattern() = default;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto thisArg = adaptor.getOperands()[0];
    const auto ptrTy = LLVM::LLVMPointerType::get(
        op.getType(), thisArg.getType()
                          .template cast<LLVM::LLVMPointerType>()
                          .getAddressSpace());
    const auto loc = op.getLoc();
    const auto dimension = getDimensions(op.getOperand().getType());
    assert(1 <= dimension && dimension < 4 && "Invalid number of dimensions");
    Value newValue =
        rewriter.create<arith::ConstantIntOp>(loc, 1, op.getType());
    for (unsigned i = 0; i < dimension; ++i) {
      const auto size = getRange(rewriter, loc, ptrTy, thisArg, i);
      newValue = rewriter.create<arith::MulIOp>(loc, newValue, size);
    }
    rewriter.replaceOp(op, newValue);
  }
};

template <typename Op>
class GetLinearIDPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  virtual Value getID(OpBuilder &builder, Location loc, Type ptrTy,
                      Value thisArg, int32_t index) const = 0;
  virtual Value getRange(OpBuilder &builder, Location loc, Type ptrTy,
                         Value thisArg, int32_t index) const = 0;

public:
  virtual ~GetLinearIDPattern() = default;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto thisArg = adaptor.getOperands()[0];
    const auto ptrTy = LLVM::LLVMPointerType::get(
        op.getType(), thisArg.getType()
                          .template cast<LLVM::LLVMPointerType>()
                          .getAddressSpace());
    const auto loc = op.getLoc();
    const auto dimension = getDimensions(op.getOperand().getType());
    Value newValue;
    switch (dimension) {
    case 1:
      // get_id(0)
      newValue = getID(rewriter, loc, ptrTy, thisArg, 0);
      break;
    case 2: {
      // get_id(0) * get_range(1) + get_id(1)
      const auto id0 = getID(rewriter, loc, ptrTy, thisArg, 0);
      const auto r1 = getRange(rewriter, loc, ptrTy, thisArg, 1);
      const Value prod = rewriter.create<arith::MulIOp>(loc, id0, r1);
      const auto id1 = getID(rewriter, loc, ptrTy, thisArg, 1);
      newValue = rewriter.create<arith::AddIOp>(loc, prod, id1);
      break;
    }
    case 3: {
      // get_id(0) * get_range(1) * get_range(2) + get_id(1) * get_range(2) +
      // get_id(2)
      const auto id0 = getID(rewriter, loc, ptrTy, thisArg, 0);
      const auto r1 = getRange(rewriter, loc, ptrTy, thisArg, 1);
      const Value prod0 = rewriter.create<arith::MulIOp>(loc, id0, r1);
      const auto r2 = getRange(rewriter, loc, ptrTy, thisArg, 2);
      const Value prod1 = rewriter.create<arith::MulIOp>(loc, prod0, r2);
      const auto id1 = getID(rewriter, loc, ptrTy, thisArg, 1);
      const Value prod2 = rewriter.create<arith::MulIOp>(loc, id1, r2);
      const Value add = rewriter.create<arith::AddIOp>(loc, prod1, prod2);
      const auto id2 = getID(rewriter, loc, ptrTy, thisArg, 2);
      newValue = rewriter.create<arith::AddIOp>(loc, add, id2);
      break;
    }
    default:
      llvm_unreachable("Invalid number of dimensions");
    }
    rewriter.replaceOp(op, newValue);
  }
};

/// Base pattern for operations building a struct from a SYCL grid operation
/// \tparam GridOp.
template <typename Op, typename GridOp, typename... Tags>
class GridOpInitPattern : public ConvertOpToLLVMPattern<Op>,
                          public GetMemberPattern<Tags...> {
private:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    constexpr unsigned dimGetTyWidth{32};

    const auto loc = op.getLoc();
    const auto elTy = getTypeConverter()->convertType(op.getType());
    const auto ptrTy = LLVM::LLVMPointerType::get(elTy);
    const Value arraySize = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    const Value alloca =
        rewriter.create<LLVM::AllocaOp>(loc, ptrTy, elTy, arraySize);
    const auto structMemberTy = rewriter.getI64Type();
    const auto dimTy = rewriter.getIndexType();
    const auto innerPtrTy = LLVM::LLVMPointerType::get(structMemberTy);
    for (unsigned i = 0,
                  dimensions = getDimensions(op->getOperand(0).getType());
         i < dimensions; ++i) {
      const auto ref = getRef(rewriter, loc, innerPtrTy, alloca, i);
      const Value dim =
          rewriter.create<arith::ConstantIntOp>(loc, i, dimGetTyWidth);
      const Value val = rewriter.create<arith::IndexCastUIOp>(
          loc, structMemberTy, rewriter.create<GridOp>(loc, dimTy, dim));
      rewriter.create<LLVM::StoreOp>(loc, val, ref);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
  }
};

/// Base pattern for operations returning the result of a SYCL grid operation
/// \tparam GridOp.
template <typename Op, typename GridOp>
class GridOpInitDimPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const Value id = rewriter.create<GridOp>(
        op.getLoc(), rewriter.getIndexType(), adaptor.getOperands()[1]);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, op.getType(), id);
  }
};

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Create a LLVM struct type with name \p name, and the converted \p body as
/// the body.
static Optional<Type> convertBodyType(StringRef name,
                                      llvm::ArrayRef<mlir::Type> body,
                                      LLVMTypeConverter &converter) {
  SmallVector<Type> convertedElemTypes;
  convertedElemTypes.reserve(body.size());
  if (failed(converter.convertTypes(body, convertedElemTypes)))
    return std::nullopt;
  auto convertedTy =
      LLVM::LLVMStructType::getIdentified(&converter.getContext(), name);
  if (!convertedTy.isInitialized()) {
    if (failed(convertedTy.setBody(convertedElemTypes, /*isPacked=*/false)))
      return std::nullopt;
  } else if (convertedElemTypes != convertedTy.getBody()) {
    // If the name is already in use, create a new type.
    convertedTy = LLVM::LLVMStructType::getNewIdentified(
        &converter.getContext(), name, convertedElemTypes, /*isPacked=*/false);
  }

  return convertedTy;
}

/// Converts SYCL accessor common type to LLVM type.
static Optional<Type> convertAccessorCommonType(sycl::AccessorCommonType type,
                                                LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::accessor_common", converter);
}

/// Converts SYCL accessor implement device type to LLVM type.
static Optional<Type>
convertAccessorImplDeviceType(sycl::AccessorImplDeviceType type,
                              LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::AccessorImplDevice." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor type to LLVM type.
static Optional<Type> convertAccessorType(sycl::AccessorType type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::accessor." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor subscript type to LLVM type.
static Optional<Type>
convertAccessorSubscriptType(sycl::AccessorSubscriptType type,
                             LLVMTypeConverter &converter) {
  return convertBodyType(
      "class.sycl::_V1::detail::accessor_common.AccessorSubscript." +
          std::to_string(type.getCurrentDimension()),
      type.getBody(), converter);
}

/// Converts SYCL array type to LLVM type.
static Optional<Type> convertArrayType(sycl::ArrayType type,
                                       LLVMTypeConverter &converter) {
  assert(type.getBody().size() == 1 &&
         "Expecting SYCL array body to have size 1");
  assert(type.getBody()[0].isa<MemRefType>() &&
         "Expecting SYCL array body entry to be MemRefType");
  assert(type.getBody()[0].cast<MemRefType>().getElementType() ==
             converter.getIndexType() &&
         "Expecting SYCL array body entry element type to be the index type");
  auto structTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(),
      "class.sycl::_V1::detail::array." + std::to_string(type.getDimension()));
  if (!structTy.isInitialized()) {
    auto arrayTy =
        LLVM::LLVMArrayType::get(converter.getIndexType(), type.getDimension());
    if (failed(structTy.setBody(arrayTy, /*isPacked=*/false)))
      return std::nullopt;
  }
  return structTy;
}

/// Converts SYCL atomic type to LLVM type.
static Optional<Type> convertAtomicType(sycl::AtomicType type,
                                        LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::atomic", type.getBody(), converter);
}

/// Converts SYCL group type to LLVM type.
static Optional<Type> convertGroupType(sycl::GroupType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::group." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL h_item type to LLVM type.
static Optional<Type> convertHItemType(sycl::HItemType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::h_item", type.getBody(), converter);
}

/// Converts SYCL id type to LLVM type.
static Optional<Type> convertIDType(sycl::IDType type,
                                    LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::id." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL item base type to LLVM type.
static Optional<Type> convertItemBaseType(sycl::ItemBaseType type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::ItemBase." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL item type to LLVM type.
static Optional<Type> convertItemType(sycl::ItemType type,
                                      LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::item." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL kernel_handler type to LLVM type.
static Optional<Type> convertKernelHandlerType(sycl::KernelHandlerType type,
                                               LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::kernel_handler", type.getBody(),
                         converter);
}

/// Converts SYCL local accessor base device type to LLVM type.
static Optional<Type>
convertLocalAccessorBaseDeviceType(sycl::LocalAccessorBaseDeviceType type,
                                   LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::LocalAccessorBaseDevice." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL local accessor base type to LLVM type.
static Optional<Type>
convertLocalAccessorBaseType(sycl::LocalAccessorBaseType type,
                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::local_accessor_base." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL local accessor type to LLVM type.
static Optional<Type> convertLocalAccessorType(sycl::LocalAccessorType type,
                                               LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::local_accessor." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL maximum type to LLVM type.
static Optional<Type> convertMaximumType(sycl::MaximumType type,
                                         LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::maximum", converter);
}

/// Converts SYCL minimum type to LLVM type.
static Optional<Type> convertMinimumType(sycl::MinimumType type,
                                         LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::minimum", converter);
}

/// Converts SYCL multi_ptr type to LLVM type.
static Optional<Type> convertMultiPtrType(sycl::MultiPtrType type,
                                          LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::multi_ptr", type.getBody(),
                         converter);
}

/// Converts SYCL nd item type to LLVM type.
static Optional<Type> convertNdItemType(sycl::NdItemType type,
                                        LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::nd_item." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL nd_range type to LLVM type.
static Optional<Type> convertNdRangeType(sycl::NdRangeType type,
                                         LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::nd_range." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL owner less base type to LLVM type.
static Optional<Type> convertOwnerLessBaseType(sycl::OwnerLessBaseType type,
                                               LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::OwnerLessBase", converter);
}

/// Converts SYCL range type to LLVM type.
static Optional<Type> convertRangeType(sycl::RangeType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::range." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL stream type to LLVM type.
static Optional<Type> convertStreamType(sycl::StreamType type,
                                        LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::stream", type.getBody(), converter);
}

/// Converts SYCL sub_group type to LLVM type.
static Optional<Type> convertSubGroupType(sycl::SubGroupType type,
                                          LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::ext::oneapi::sub_group", converter);
}

/// Converts SYCL vec type to LLVM type.
static Optional<Type> convertSwizzledVecType(sycl::SwizzledVecType type,
                                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::SwizzleOp", type.getBody(),
                         converter);
}
/// Converts SYCL vec type to LLVM type.
static Optional<Type> convertVecType(sycl::VecType type,
                                     LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::vec", type.getBody(), converter);
}

//===----------------------------------------------------------------------===//
// CallPattern - Converts `sycl.call` to LLVM.
//===----------------------------------------------------------------------===//

class CallPattern final : public ConvertOpToLLVMPattern<sycl::SYCLCallOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLCallOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(sycl::SYCLCallOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return rewriteCall(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.call to a func call to the appropriate member function.
  LogicalResult rewriteCall(SYCLCallOp op, OpAdaptor opAdaptor,
                            ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "CallPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");
    assert(op.getNumResults() <= 1 && "Call should produce at most one result");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    FuncBuilder builder(rewriter, op.getLoc());

    bool producesResult = op.getNumResults() == 1;
    func::CallOp funcCall = builder.genCall(
        op.getMangledFunctionName(),
        producesResult ? TypeRange(op.getResult().getType()) : TypeRange(),
        op.getOperands(), module);

    rewriter.replaceOp(op.getOperation(),
                       producesResult ? funcCall->getResult(0) : ValueRange());

    LLVM_DEBUG({
      Operation *func = funcCall->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "CallPattern: Function after rewrite:\n" << *func << "\n";
    });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// CastPattern - Converts `sycl.cast` to LLVM.
//===----------------------------------------------------------------------===//

class CastPattern final : public ConvertOpToLLVMPattern<sycl::SYCLCastOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteCast(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.cast() to a LLVM bitcast operation.
  LogicalResult rewriteCast(SYCLCastOp op, OpAdaptor opAdaptor,
                            ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "CastPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    assert(op.getSource().getType().isa<MemRefType>() &&
           "The cast source type should be a memref type");
    assert(op.getResult().getType().isa<MemRefType>() &&
           "The result source type should be a memref type");

    // Ensure the input and result types are legal.
    auto srcType = op.getSource().getType().cast<MemRefType>();
    auto resType = op.getResult().getType().cast<MemRefType>();

    if (!isConvertibleAndHasIdentityMaps(srcType) ||
        !isConvertibleAndHasIdentityMaps(resType))
      return failure();

    // Cast the source memref descriptor's allocate & aligned pointers to the
    // type of those pointers in the results memref.
    Location loc = op.getLoc();
    LLVMBuilder builder(rewriter, loc);
    MemRefDescriptor srcMemRefDesc(opAdaptor.getSource());
    Value allocatedPtr = builder.genBitcast(
        getElementPtrType(resType), srcMemRefDesc.allocatedPtr(rewriter, loc));
    Value alignedPtr = builder.genBitcast(
        getElementPtrType(resType), srcMemRefDesc.alignedPtr(rewriter, loc));

    // Create the result memref descriptor.
    SmallVector<Value, 4> sizes, strides;
    for (int pos = 0; pos < resType.getRank(); ++pos) {
      sizes.push_back(srcMemRefDesc.size(rewriter, loc, pos));
      strides.push_back(srcMemRefDesc.stride(rewriter, loc, pos));
    }

    MemRefDescriptor resMemRefDesc = createMemRefDescriptor(
        loc, resType, allocatedPtr, alignedPtr, sizes, strides, rewriter);
    resMemRefDesc.setOffset(rewriter, loc, srcMemRefDesc.offset(rewriter, loc));

    rewriter.replaceOp(op.getOperation(), {resMemRefDesc});

    LLVM_DEBUG({
      Operation *func = op->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "CastPattern: Function after rewrite:\n" << *func << "\n";
    });

    return success();
  }
};

class BarePtrCastPattern final : public ConvertOpToLLVMPattern<SYCLCastOp> {
public:
  using ConvertOpToLLVMPattern<SYCLCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto srcType = op.getSource().getType().cast<MemRefType>();
    const auto resType = op.getResult().getType().cast<MemRefType>();
    const auto convSrcType = typeConverter->convertType(srcType);
    const auto convResType = typeConverter->convertType(resType);

    // Ensure the input and result types are legal.
    if (!canBeLoweredToBarePtr(srcType) || !canBeLoweredToBarePtr(resType) ||
        !convSrcType || !convResType)
      return failure();

    Location loc = op.getLoc();
    LLVMBuilder builder(rewriter, loc);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, convResType,
                                                 opAdaptor.getSource());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// CastPattern - Converts `sycl.addrspacecast` to LLVM.
//===----------------------------------------------------------------------===//

struct BarePtrAddrSpaceCastPattern
    : public ConvertOpToLLVMPattern<SYCLAddrSpaceCastOp> {
  using ConvertOpToLLVMPattern<SYCLAddrSpaceCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLAddrSpaceCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto newTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, newTy,
                                                       opAdaptor.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConstructorPattern - Converts `sycl.constructor` to LLVM.
//===----------------------------------------------------------------------===//
class ConstructorPattern final
    : public ConvertOpToLLVMPattern<sycl::SYCLConstructorOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLConstructorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLConstructorOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteConstructor(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.constructor to a func call to the appropriate constructor
  /// function.
  LogicalResult rewriteConstructor(SYCLConstructorOp op, OpAdaptor opAdaptor,
                                   ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConstructorPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    FuncBuilder builder(rewriter, op.getLoc());
    func::CallOp funcCall = builder.genCall(
        op.getMangledFunctionName(), TypeRange(), op.getOperands(), module);
    rewriter.eraseOp(op);
    (void)funcCall;

    LLVM_DEBUG({
      Operation *func = funcCall->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "ConstructorPattern: Function after rewrite:\n"
                   << *func << "\n";
    });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// AccessorSubscriptPattern - Convert `sycl.accessor.subscript` to LLVM.
//===----------------------------------------------------------------------===//

/// Base class for other patterns converting `sycl.accessor.subscript` to LLVM.
class AccessorSubscriptPattern
    : public ConvertOpToLLVMPattern<SYCLAccessorSubscriptOp>,
      public GetMemberPattern<AccessorGetPtr> {
public:
  using ConvertOpToLLVMPattern<SYCLAccessorSubscriptOp>::ConvertOpToLLVMPattern;

public:
  /// Whether the input accessor has atomic access mode.
  static bool hasAtomicAccessor(SYCLAccessorSubscriptOp op) {
    return op.getAcc()
               .getType()
               .getElementType()
               .cast<AccessorType>()
               .getAccessMode() == MemoryAccessMode::Atomic;
  }

  /// Whether the input accessor is 1-dimensional.
  static bool has1DAccessor(SYCLAccessorSubscriptOp op) {
    return op.getAcc()
               .getType()
               .getElementType()
               .cast<AccessorType>()
               .getDimension() == 1;
  }

  /// Whether the input offset is an id.
  static bool hasIDOffsetType(SYCLAccessorSubscriptOp op) {
    return op.getIndex().getType().isa<MemRefType>();
  }

  Value getRef(OpBuilder &builder, Location loc, Type ptrTy, Value acc,
               Value index) const {
    const auto ptr = GetMemberPattern<AccessorGetPtr>::loadValue(
        builder, loc, LLVM::LLVMPointerType::get(ptrTy), acc);
    return builder.create<LLVM::GEPOp>(loc, ptrTy, ptr, index,
                                       /*inbounds*/ true);
  }
};

class AccessorSubscriptIDIndexPattern
    : public AccessorSubscriptPattern,
      public GetMemberPattern<IDGetDim>,
      public GetMemberPattern<AccessorGetMemRange, RangeGetDim> {
  template <typename... Args> Value getID(Args &&...args) const {
    return GetMemberPattern<IDGetDim>::loadValue(std::forward<Args>(args)...);
  }

  template <typename... Args> Value getMemRange(Args &&...args) const {
    return GetMemberPattern<AccessorGetMemRange, RangeGetDim>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  /// Calculates the linear index out of an id.
  Value getLinearIndex(OpBuilder &builder, Location loc, AccessorType accTy,
                       OpAdaptor opAdaptor) const {
    const auto id = opAdaptor.getIndex();
    const auto mem = opAdaptor.getAcc();
    // int64_t Res{0};
    Value res = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    const auto memRangePtrTy = LLVM::LLVMPointerType::get(
        builder.getI64Type(),
        mem.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    const auto idPtrTy = LLVM::LLVMPointerType::get(
        builder.getI64Type(),
        id.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    for (unsigned i = 0, dim = accTy.getDimension(); i < dim; ++i) {
      // Res = Res * Mem[I] + Id[I]
      const auto memI = getMemRange(builder, loc, memRangePtrTy, mem, i);
      const auto idI = getID(builder, loc, idPtrTy, id, i);
      res = builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, res, memI), idI);
    }
    return res;
  }
};

/// Conversion pattern with non-atomic access mode and id offset type.
class SubscriptIDOffset : public AccessorSubscriptIDIndexPattern {
public:
  using AccessorSubscriptIDIndexPattern::AccessorSubscriptIDIndexPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   op.getType().isa<MemRefType>());
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ptrTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOp(
        op, AccessorSubscriptPattern::getRef(
                rewriter, loc, ptrTy, opAdaptor.getAcc(),
                getLinearIndex(
                    rewriter, loc,
                    op.getAcc().getType().getElementType().cast<AccessorType>(),
                    opAdaptor)));
  }
};

/// Conversion pattern with non-atomic access mode, scalar offset type and
/// 1-dimensional accessor.
class SubscriptScalarOffset1D : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   AccessorSubscriptPattern::has1DAccessor(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ptrTy = getTypeConverter()->convertType(op.getType());
    const Value ptr = GetMemberPattern<AccessorGetPtr>::loadValue(
        rewriter, loc, LLVM::LLVMPointerType::get(ptrTy), opAdaptor.getAcc());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrTy, ptr, opAdaptor.getIndex(), /*inbounds*/ true);
  }
};

/// Conversion pattern with non-atomic access mode, scalar offset type and
/// N-dimensional accessor.
///
/// Return type is implementation specific. Handling DPC++ case here: struct
/// with two fields:
/// - id<Dim - 1>: Current offset;
/// - accessor<Dim>: Original accessor.
class SubscriptScalarOffsetND : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   !AccessorSubscriptPattern::has1DAccessor(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    Value subscript = rewriter.create<LLVM::UndefOp>(
        loc, getTypeConverter()->convertType(op.getType()));
    // Insert initial offset in the first position
    subscript = rewriter.create<LLVM::InsertValueOp>(
        loc, subscript, opAdaptor.getIndex(), ArrayRef<int64_t>{0, 0, 0, 0});
    // Zero-initialize rest of the offset id<Dim - 1>
    const Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    for (unsigned i = 1, dim = getDimensions(op.getAcc().getType()) - 1;
         i < dim; ++i) {
      subscript = rewriter.create<LLVM::InsertValueOp>(
          loc, subscript, zero, ArrayRef<int64_t>{0, 0, 0, i});
    }
    // Insert original accessor
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, subscript, rewriter.create<LLVM::LoadOp>(loc, opAdaptor.getAcc()),
        1);
  }
};

/// Conversion pattern with atomic access mode and id offset type.
class AtomicSubscriptIDOffset : public AccessorSubscriptIDIndexPattern {
public:
  using AccessorSubscriptIDIndexPattern::AccessorSubscriptIDIndexPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   AccessorSubscriptPattern::hasIDOffsetType(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto atomicTy = op.getType().cast<AtomicType>();
    auto *typeConverter = getTypeConverter();
    const auto ptrTy = typeConverter->convertType(
        MemRefType::get(ShapedType::kDynamic, atomicTy.getDataType(), {},
                        static_cast<unsigned>(atomicTy.getAddrSpace())));
    const Value undef = rewriter.create<LLVM::UndefOp>(
        loc, typeConverter->convertType(atomicTy));
    const auto ptr = AccessorSubscriptPattern::getRef(
        rewriter, loc, ptrTy, opAdaptor.getAcc(),
        getLinearIndex(
            rewriter, loc,
            op.getAcc().getType().getElementType().cast<AccessorType>(),
            opAdaptor));
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undef, ptr, 0);
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeGetPattern - Convert `sycl.range.get` to LLVM.
//===----------------------------------------------------------------------===//

class RangeGetPattern
    : public LoadMemberDimPattern<SYCLRangeGetOp, RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLRangeGetOp, RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<IntegerType>());
  }
};

class RangeGetRefPattern
    : public GetRefToMemberDimPattern<SYCLRangeGetOp, RangeGetDim> {
public:
  using GetRefToMemberDimPattern<SYCLRangeGetOp,
                                 RangeGetDim>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<MemRefType>());
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeSizePattern - Convert `sycl.range.size` to LLVM.
//===----------------------------------------------------------------------===//

class RangeSizePattern : public GetRangeSizePattern<SYCLRangeSizeOp>,
                         public GetMemberPattern<RangeGetDim> {
public:
  using GetRangeSizePattern<SYCLRangeSizeOp>::GetRangeSizePattern;

  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return loadValue(builder, loc, ptrTy, thisArg, index);
  }
};

//===----------------------------------------------------------------------===//
// NDRangeGetGlobalRangePattern - Converts `sycl.nd_range.get_global_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Convert SYCLNdRangeGetGlobalRange to LLVM
///
/// For this pattern, we have to load the global range.
class NDRangeGetGlobalRangePattern
    : public LoadMemberPattern<SYCLNdRangeGetGlobalRange,
                               NDRangeGetGlobalRange> {
public:
  using LoadMemberPattern<SYCLNdRangeGetGlobalRange,
                          NDRangeGetGlobalRange>::LoadMemberPattern;
};

//===----------------------------------------------------------------------===//
// NDRangeGetLocalRangePattern - Converts `sycl.nd_range.get_local_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Convert SYCLNdRangeGetLocalRange to LLVM
///
/// For this pattern, we have to load the local range.
class NDRangeGetLocalRangePattern
    : public LoadMemberPattern<SYCLNdRangeGetLocalRange, NDRangeGetLocalRange> {
public:
  using LoadMemberPattern<SYCLNdRangeGetLocalRange,
                          NDRangeGetLocalRange>::LoadMemberPattern;
};

//===----------------------------------------------------------------------===//
// NDRangeGetGroupRangePattern - Converts `sycl.nd_range.get_group_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Convert SYCLNdRangeGetGroupRange to LLVM
///
/// For this pattern, we have to load both the global and local range and
/// perform an element-wise division.
class NDRangeGetGroupRangePattern
    : public ConvertOpToLLVMPattern<SYCLNdRangeGetGroupRange>,
      public GetMemberPattern<NDRangeGetGlobalRange, RangeGetDim>,
      public GetMemberPattern<NDRangeGetLocalRange, RangeGetDim>,
      public GetMemberPattern<RangeGetDim> {
public:
  using ConvertOpToLLVMPattern<
      SYCLNdRangeGetGroupRange>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNdRangeGetGroupRange op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto nd = opAdaptor.getND();
    const auto rangeTy = op.getType();
    Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc,
        LLVM::LLVMPointerType::get(getTypeConverter()->convertType(rangeTy)),
        rewriter.create<arith::ConstantIntOp>(loc, 1, 64),
        /*alignment*/ 0);
    const auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    for (int32_t i = 0, dim = rangeTy.getDimension(); i < dim; ++i) {
      const auto lhs =
          GetMemberPattern<NDRangeGetGlobalRange, RangeGetDim>::loadValue(
              rewriter, loc, ptrTy, nd, i);
      const auto rhs =
          GetMemberPattern<NDRangeGetLocalRange, RangeGetDim>::loadValue(
              rewriter, loc, ptrTy, nd, i);
      const Value val = rewriter.create<arith::DivUIOp>(loc, lhs, rhs);
      const auto ptr = GetMemberPattern<RangeGetDim>::getRef(rewriter, loc,
                                                             ptrTy, alloca, i);
      rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IDGetPattern - Converts `sycl.it.get` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLIDGet with a scalar return type to LLVM
class IDGetPattern : public LoadMemberDimPattern<SYCLIDGetOp, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLIDGetOp, IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLIDGetOp op) const final {
    return success(op.getNumOperands() > 1 && op.getType().isa<IntegerType>());
  }
};

/// Converts SYCLIDGet with a reference return type to LLVM
class IDGetRefPattern : public GetRefToMemberDimPattern<SYCLIDGetOp, IDGetDim> {
public:
  using GetRefToMemberDimPattern<SYCLIDGetOp,
                                 IDGetDim>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLIDGetOp op) const final {
    return success(op.getType().isa<MemRefType>());
  }
};

//===----------------------------------------------------------------------===//
// ItemGetIDPattern - Converts `sycl.item.get_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLItemGetIDOp with an id return type to LLVM
class ItemGetIDPattern : public LoadMemberPattern<SYCLItemGetIDOp, ItemGetID> {
public:
  using LoadMemberPattern<SYCLItemGetIDOp, ItemGetID>::LoadMemberPattern;

  LogicalResult match(SYCLItemGetIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLItemGetIDOp with an index return type to LLVM
class ItemGetIDDimPattern
    : public LoadMemberDimPattern<SYCLItemGetIDOp, ItemGetID, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLItemGetIDOp, ItemGetID,
                             IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLItemGetIDOp op) const final {
    return success(op.getNumOperands() > 1 &&
                   op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// ItemGetRangePattern - Converts `sycl.item.get_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLItemGetRangeOp with an range return type to LLVM
class ItemGetRangePattern
    : public LoadMemberPattern<SYCLItemGetRangeOp, ItemGetRange> {
public:
  using LoadMemberPattern<SYCLItemGetRangeOp, ItemGetRange>::LoadMemberPattern;

  LogicalResult match(SYCLItemGetRangeOp op) const final {
    return success(op.getRes().getType().isa<RangeType>());
  }
};

/// Converts SYCLItemGetIDOp with an index return type to LLVM
class ItemGetRangeDimPattern
    : public LoadMemberDimPattern<SYCLItemGetRangeOp, ItemGetRange,
                                  RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLItemGetRangeOp, ItemGetRange,
                             RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLItemGetRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// ItemGetLinearIDPattern - Converts `sycl.item.get_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

class ItemGetLinearIDPattern
    : public GetLinearIDPattern<SYCLItemGetLinearIDOp>,
      public GetMemberPattern<ItemGetID, IDGetDim>,
      public GetMemberPattern<ItemGetRange, RangeGetDim> {
protected:
  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<ItemGetRange, RangeGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }

public:
  using GetLinearIDPattern<SYCLItemGetLinearIDOp>::GetLinearIDPattern;
};

/// Converts SYCLItemGetLinearIDOp with no offset item to LLVM
class ItemNoOffsetGetLinearIDPattern : public ItemGetLinearIDPattern {
public:
  using ItemGetLinearIDPattern::ItemGetLinearIDPattern;

  LogicalResult match(SYCLItemGetLinearIDOp op) const final {
    return success(!op.getItem()
                        .getType()
                        .getElementType()
                        .cast<ItemType>()
                        .getWithOffset());
  }

  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
              int32_t index) const final {
    return GetMemberPattern<ItemGetID, IDGetDim>::loadValue(builder, loc, ptrTy,
                                                            thisArg, index);
  }
};

/// Converts SYCLItemGetLinearIDOp with no offset item to LLVM
class ItemOffsetGetLinearIDPattern
    : public ItemGetLinearIDPattern,
      public GetMemberPattern<ItemGetOffset, IDGetDim> {
protected:
  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
              int32_t index) const final {
    const auto id = GetMemberPattern<ItemGetID, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
    const auto offset = GetMemberPattern<ItemGetOffset, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
    return builder.create<arith::SubIOp>(loc, id, offset);
  }

public:
  using ItemGetLinearIDPattern::ItemGetLinearIDPattern;

  LogicalResult match(SYCLItemGetLinearIDOp op) const final {
    return success(op.getItem()
                       .getType()
                       .getElementType()
                       .cast<ItemType>()
                       .getWithOffset());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetGlobalID - Converts `sycl.nd_item.get_global_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetGlobalIDOp with an ID return type to LLVM
class NDItemGetGlobalIDPattern
    : public LoadMemberPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem,
                               ItemGetID> {
public:
  using LoadMemberPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem,
                          ItemGetID>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetGlobalIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLNDItemGetGlobalIDOp with an ID return type to LLVM
class NDItemGetGlobalIDDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem,
                                  ItemGetID, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem,
                             ItemGetID, IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetGlobalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetGlobalLinearIDPattern - Converts `sycl.nd_item.get_global_linear_id`
// to LLVM.
//===----------------------------------------------------------------------===//

class NDItemGetGlobalLinearIDPattern
    : public GetLinearIDPattern<SYCLNDItemGetGlobalLinearIDOp>,
      public GetMemberPattern<NDItemGlobalItem, ItemGetID, IDGetDim>,
      public GetMemberPattern<NDItemGlobalItem, ItemGetRange, RangeGetDim> {
protected:
  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<NDItemGlobalItem, ItemGetRange,
                            RangeGetDim>::loadValue(builder, loc, ptrTy,
                                                    thisArg, index);
  }

  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
              int32_t index) const final {
    return GetMemberPattern<NDItemGlobalItem, ItemGetID, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }

public:
  using GetLinearIDPattern<SYCLNDItemGetGlobalLinearIDOp>::GetLinearIDPattern;
};

//===----------------------------------------------------------------------===//
// NDItemGetLocalID - Converts `sycl.nd_item.get_local_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetLocalIDOp with an ID return type to LLVM
class NDItemGetLocalIDPattern
    : public LoadMemberPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem,
                               ItemGetID> {
public:
  using LoadMemberPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem,
                          ItemGetID>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLNDItemGetLocalIDOp with an ID return type to LLVM
class NDItemGetLocalIDDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem,
                                  ItemGetID, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem, ItemGetID,
                             IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetLocalLinearIDPattern - Converts `sycl.nd_item.get_local_linear_id`
// to LLVM.
//===----------------------------------------------------------------------===//

class NDItemGetLocalLinearIDPattern
    : public GetLinearIDPattern<SYCLNDItemGetLocalLinearIDOp>,
      public GetMemberPattern<NDItemLocalItem, ItemGetID, IDGetDim>,
      public GetMemberPattern<NDItemLocalItem, ItemGetRange, RangeGetDim> {
protected:
  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<NDItemLocalItem, ItemGetRange,
                            RangeGetDim>::loadValue(builder, loc, ptrTy,
                                                    thisArg, index);
  }

  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
              int32_t index) const final {
    return GetMemberPattern<NDItemLocalItem, ItemGetID, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }

public:
  using GetLinearIDPattern<SYCLNDItemGetLocalLinearIDOp>::GetLinearIDPattern;
};

//===----------------------------------------------------------------------===//
// NDItemGetGroup - Converts `sycl.nd_item.get_group` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetGroupOp with an ID return type to LLVM
class NDItemGetGroupPattern
    : public LoadMemberPattern<SYCLNDItemGetGroupOp, NDItemGroup> {
public:
  using LoadMemberPattern<SYCLNDItemGetGroupOp, NDItemGroup>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetGroupOp op) const final {
    return success(op.getRes().getType().isa<GroupType>());
  }
};

/// Converts SYCLNDItemGetGroupOp with an ID return type to LLVM
class NDItemGetGroupDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetGroupOp, NDItemGroup, GroupGetID,
                                  IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetGroupOp, NDItemGroup, GroupGetID,
                             IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetGroupOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetGroupRange - Converts `sycl.nd_item.get_group_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetGroupRangeOp with an ID return type to LLVM
class NDItemGetGroupRangePattern
    : public LoadMemberPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                               GroupGetGroupRange> {
public:
  using LoadMemberPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                          GroupGetGroupRange>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetGroupRangeOp op) const final {
    return success(op.getRes().getType().isa<RangeType>());
  }
};

/// Converts SYCLNDItemGetGroupOp with an ID return type to LLVM
class NDItemGetGroupRangeDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                                  GroupGetGroupRange, RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                             GroupGetGroupRange,
                             RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetGroupRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetLocalRange - Converts `sycl.nd_item.get_local_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetLocalRangeOp with an ID return type to LLVM
class NDItemGetLocalRangePattern
    : public LoadMemberPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                               ItemGetRange> {
public:
  using LoadMemberPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                          ItemGetRange>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetLocalRangeOp op) const final {
    return success(op.getRes().getType().isa<RangeType>());
  }
};

/// Converts SYCLNDItemGetLocalOp with an ID return type to LLVM
class NDItemGetLocalRangeDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                                  ItemGetRange, RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                             ItemGetRange, RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetLocalRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetNDRange - Converts `sycl.nd_item.get_nd_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetLocalRangeOp with an ID return type to LLVM
class NDItemGetNDRange
    : public ConvertOpToLLVMPattern<SYCLNDItemGetNdRangeOp>,
      public GetMemberPattern<NDItemGlobalItem, ItemGetRange>,
      public GetMemberPattern<NDItemLocalItem, ItemGetRange>,
      public GetMemberPattern<NDItemGlobalItem, ItemGetOffset>,
      public GetMemberPattern<NDRangeGetGlobalRange>,
      public GetMemberPattern<NDRangeGetLocalRange>,
      public GetMemberPattern<NDRangeGetOffset> {
public:
  using ConvertOpToLLVMPattern<SYCLNDItemGetNdRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNDItemGetNdRangeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ndItem = opAdaptor.getNDItem();

    const auto ndrTy = getTypeConverter()->convertType(op.getType());
    const Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(ndrTy), ndrTy,
        rewriter.create<arith::ConstantIntOp>(loc, 1, 32));

    const auto rangePtrTy = LLVM::LLVMPointerType::get(
        ndrTy.cast<LLVM::LLVMStructType>().getBody()[0]);
    const auto idPtrTy = LLVM::LLVMPointerType::get(
        ndrTy.cast<LLVM::LLVMStructType>().getBody()[2]);

    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemGlobalItem, ItemGetRange>::loadValue(
            rewriter, loc, rangePtrTy, ndItem),
        GetMemberPattern<NDRangeGetGlobalRange>::getRef(rewriter, loc,
                                                        rangePtrTy, alloca));
    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemLocalItem, ItemGetRange>::loadValue(
            rewriter, loc, rangePtrTy, ndItem),
        GetMemberPattern<NDRangeGetLocalRange>::getRef(rewriter, loc,
                                                       rangePtrTy, alloca));
    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemGlobalItem, ItemGetOffset>::loadValue(
            rewriter, loc, idPtrTy, ndItem),
        GetMemberPattern<NDRangeGetOffset>::getRef(rewriter, loc, idPtrTy,
                                                   alloca));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupGetID - Converts `sycl.group.get_group_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupID with an ID return type to LLVM
class GroupGetGroupIDPattern
    : public LoadMemberPattern<SYCLGroupGetGroupIDOp, GroupGetID> {
public:
  using LoadMemberPattern<SYCLGroupGetGroupIDOp, GroupGetID>::LoadMemberPattern;

  LogicalResult match(SYCLGroupGetGroupIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLGroupGetGroupID with a scalar return type to LLVM
class GroupGetGroupIDDimPattern
    : public LoadMemberDimPattern<SYCLGroupGetGroupIDOp, GroupGetID, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLGroupGetGroupIDOp, GroupGetID,
                             IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLGroupGetGroupIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetLocalID - Converts `sycl.group.get_local_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalID with an ID return type to LLVM
class GroupGetLocalIDPattern
    : public GridOpInitPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp, IDGetDim> {
public:
  using GridOpInitPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp,
                          IDGetDim>::GridOpInitPattern;

  LogicalResult match(SYCLGroupGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLGroupGetLocalID with a scalar return type to LLVM
class GroupGetLocalIDDimPattern
    : public GridOpInitDimPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp> {
public:
  using GridOpInitDimPattern<SYCLGroupGetLocalIDOp,
                             SYCLLocalIDOp>::GridOpInitDimPattern;

  LogicalResult match(SYCLGroupGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetLocalRange - Converts `sycl.group.get_local_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalRange with a range return type to LLVM
class GroupGetLocalRangePattern
    : public LoadMemberPattern<SYCLGroupGetLocalRangeOp, GroupGetLocalRange> {
public:
  using LoadMemberPattern<SYCLGroupGetLocalRangeOp,
                          GroupGetLocalRange>::LoadMemberPattern;

  LogicalResult match(SYCLGroupGetLocalRangeOp op) const final {
    return success(op.getRes().getType().isa<RangeType>());
  }
};

/// Converts SYCLGroupGetLocalRange with a scalar return type to LLVM
class GroupGetLocalRangeDimPattern
    : public LoadMemberDimPattern<SYCLGroupGetLocalRangeOp, GroupGetLocalRange,
                                  RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLGroupGetLocalRangeOp, GroupGetLocalRange,
                             RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLGroupGetLocalRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetGroupRange - Converts `sycl.group.get_group_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupRange with a range return type to LLVM
class GroupGetGroupRangePattern
    : public LoadMemberPattern<SYCLGroupGetGroupRangeOp, GroupGetGroupRange> {
public:
  using LoadMemberPattern<SYCLGroupGetGroupRangeOp,
                          GroupGetGroupRange>::LoadMemberPattern;

  LogicalResult match(SYCLGroupGetGroupRangeOp op) const final {
    return success(op.getRes().getType().isa<RangeType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetMaxLocalRange - Converts `sycl.group.get_max_local_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetMaxLocalRange
class GroupGetMaxLocalRangePattern
    : public LoadMemberPattern<SYCLGroupGetMaxLocalRangeOp,
                               GroupGetLocalRange> {
public:
  using LoadMemberPattern<SYCLGroupGetMaxLocalRangeOp,
                          GroupGetLocalRange>::LoadMemberPattern;
};

/// Converts SYCLGroupGetGroupRange with a scalar return type to LLVM
class GroupGetGroupRangeDimPattern
    : public LoadMemberDimPattern<SYCLGroupGetGroupRangeOp, GroupGetGroupRange,
                                  RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLGroupGetGroupRangeOp, GroupGetGroupRange,
                             RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLGroupGetGroupRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetGroupLinearID - Converts `sycl.group.get_group_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupLinearIDOp to LLVM
class GroupGetGroupLinearIDPattern
    : public GetLinearIDPattern<SYCLGroupGetGroupLinearIDOp>,
      public GetMemberPattern<GroupGetID, IDGetDim>,
      public GetMemberPattern<GroupGetGroupRange, IDGetDim> {
protected:
  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
              int32_t index) const final {
    return GetMemberPattern<GroupGetID, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }

  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<GroupGetGroupRange, IDGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }

public:
  using GetLinearIDPattern<SYCLGroupGetGroupLinearIDOp>::GetLinearIDPattern;
};

//===----------------------------------------------------------------------===//
// GroupGetLocalLinearID - Converts `sycl.group.get_local_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalLinearIDOp to LLVM
class GroupGetLocalLinearIDPattern
    : public GetLinearIDPattern<SYCLGroupGetLocalLinearIDOp>,
      public GetMemberPattern<GroupGetGroupRange, RangeGetDim> {
  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value,
              int32_t offset) const final {
    const Value dim = builder.create<arith::ConstantIntOp>(loc, offset, 32);
    const Value val =
        builder.create<SYCLLocalIDOp>(loc, builder.getIndexType(), dim);
    return builder.create<arith::IndexCastUIOp>(
        loc, ptrTy.cast<LLVM::LLVMPointerType>().getElementType(), val);
  }

  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t offset) const final {
    return GetMemberPattern<GroupGetGroupRange, RangeGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, offset);
  }

public:
  using GetLinearIDPattern<SYCLGroupGetLocalLinearIDOp>::GetLinearIDPattern;
};

//===----------------------------------------------------------------------===//
// GroupGetGroupLinearRange - Converts `sycl.group.get_group_linear_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupLinearRangeOp to LLVM
class GroupGetGroupLinearRangePattern
    : public GetRangeSizePattern<SYCLGroupGetGroupLinearRangeOp>,
      public GetMemberPattern<GroupGetGroupRange, RangeGetDim> {
public:
  using GetRangeSizePattern<
      SYCLGroupGetGroupLinearRangeOp>::GetRangeSizePattern;

protected:
  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<GroupGetGroupRange, RangeGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }
};

//===----------------------------------------------------------------------===//
// GroupGetLocalLinearRange - Converts `sycl.group.get_local_linear_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalLinearRangeOp to LLVM
class GroupGetLocalLinearRangePattern
    : public GetRangeSizePattern<SYCLGroupGetLocalLinearRangeOp>,
      public GetMemberPattern<GroupGetLocalRange, RangeGetDim> {
public:
  using GetRangeSizePattern<
      SYCLGroupGetLocalLinearRangeOp>::GetRangeSizePattern;

protected:
  Value getRange(OpBuilder &builder, Location loc, Type ptrTy, Value thisArg,
                 int32_t index) const final {
    return GetMemberPattern<GroupGetLocalRange, RangeGetDim>::loadValue(
        builder, loc, ptrTy, thisArg, index);
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateSYCLToLLVMTypeConversion(LLVMTypeConverter &typeConverter) {
  // Same order as in SYCLOps.td
  typeConverter.addConversion([&](sycl::AccessorCommonType type) {
    return convertAccessorCommonType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorImplDeviceType type) {
    return convertAccessorImplDeviceType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorType type) {
    return convertAccessorType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorSubscriptType type) {
    return convertAccessorSubscriptType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ArrayType type) {
    return convertArrayType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AtomicType type) {
    return convertAtomicType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GroupType type) {
    return convertGroupType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::HItemType type) {
    return convertHItemType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::IDType type) { return convertIDType(type, typeConverter); });
  typeConverter.addConversion([&](sycl::ItemBaseType type) {
    return convertItemBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ItemType type) {
    return convertItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::KernelHandlerType type) {
    return convertKernelHandlerType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorBaseDeviceType type) {
    return convertLocalAccessorBaseDeviceType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorBaseType type) {
    return convertLocalAccessorBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorType type) {
    return convertLocalAccessorType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MaximumType type) {
    return convertMaximumType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MinimumType type) {
    return convertMinimumType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MultiPtrType type) {
    return convertMultiPtrType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::NdItemType type) {
    return convertNdItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::NdRangeType type) {
    return convertNdRangeType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::OwnerLessBaseType type) {
    return convertOwnerLessBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::RangeType type) {
    return convertRangeType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::StreamType type) {
    return convertStreamType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::SubGroupType type) {
    return convertSubGroupType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::SwizzledVecType type) {
    return convertSwizzledVecType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::VecType type) { return convertVecType(type, typeConverter); });
}

void mlir::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  assert(typeConverter.getOptions().useBarePtrCallConv &&
         "These patterns only work with bare pointer calling convention");
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  patterns.add<BarePtrCastPattern>(typeConverter, /*benefit*/ 2);
  patterns
      .add<AddZeroArgPattern<SYCLIDGetOp>, AddZeroArgPattern<SYCLItemGetIDOp>,
           AtomicSubscriptIDOffset, BarePtrAddrSpaceCastPattern,
           GroupGetGroupIDPattern, GroupGetGroupLinearRangePattern,
           GroupGetGroupRangeDimPattern, GroupGetLocalIDPattern,
           GroupGetLocalLinearRangePattern, GroupGetLocalRangeDimPattern,
           IDGetPattern, IDGetRefPattern, ItemGetIDDimPattern,
           ItemGetRangeDimPattern, ItemGetRangePattern,
           NDItemGetGlobalIDDimPattern, NDItemGetGlobalIDPattern,
           NDItemGetGroupPattern, NDItemGetGroupRangeDimPattern,
           NDItemGetLocalIDDimPattern, NDItemGetLocalLinearIDPattern,
           NDItemGetNDRange, NDRangeGetGroupRangePattern,
           NDRangeGetLocalRangePattern, RangeGetRefPattern, RangeSizePattern,
           SubscriptScalarOffsetND, GroupGetGroupIDDimPattern,
           GroupGetGroupLinearIDPattern, GroupGetGroupRangePattern,
           GroupGetLocalIDDimPattern, GroupGetLocalLinearIDPattern,
           GroupGetLocalRangePattern, GroupGetMaxLocalRangePattern,
           ItemGetIDPattern, ItemNoOffsetGetLinearIDPattern,
           ItemOffsetGetLinearIDPattern, NDItemGetGlobalLinearIDPattern,
           NDItemGetGroupDimPattern, NDItemGetGroupRangePattern,
           NDItemGetLocalIDPattern, NDItemGetLocalRangeDimPattern,
           NDItemGetLocalRangePattern, NDRangeGetGlobalRangePattern,
           RangeGetPattern, SubscriptIDOffset, SubscriptScalarOffset1D>(
          typeConverter);
  patterns.add<ConstructorPattern>(typeConverter);
}

namespace {
/// A pass converting MLIR SYCL operations into LLVM dialect.
///
/// This pass relies on SYCL to GPU and target dialects, e.g., SPIRV, conversion
/// patterns. This pass is executed in 4 steps:
/// 1. Lower all of the operations to LLVM. As some of the operations will yield
///    SYCL grid ops, we need to run this step first;
/// 2. Convert grid ops to a target dialect, e.g., SPIRV;
/// 3. Lower remaining operations to LLVM. Same as 1, but no operation will
///    yield SYCL grid ops, so we can mark these as illegal;
/// 4. Resolve UnrealizedConversionCastOps appearing due to the fact that this
///    pass is performed in several steps.
class ConvertSYCLToLLVMPass
    : public impl::ConvertSYCLToLLVMBase<ConvertSYCLToLLVMPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult convertToSPIRV();
  LogicalResult convertToLLVM(bool lastRun);
  void cleanUnrealizedConversionCasts();
};
} // namespace

LogicalResult ConvertSYCLToLLVMPass::convertToSPIRV() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering to SPIRV...\n");

  auto &context = getContext();
  auto module = getOperation();

  const auto res = failure(
      module
          .walk([&context](gpu::GPUModuleOp gpuModule) {
            // We walk the different GPU modules looking for different SPIRV
            // target environment definitions. Currently, this does not affect
            // the behavior of this pass.
            RewritePatternSet patterns(&context);

            auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
            auto target = SPIRVConversionTarget::get(targetAttr);
            SPIRVConversionOptions options;
            // TODO: Add 32 bits support.
            options.use64bitIndex = true;
            SPIRVTypeConverter typeConverter{targetAttr, options};

            populateGPUToSPIRVPatterns(typeConverter, patterns);
            populateSYCLToSPIRVConversionPatterns(typeConverter, patterns);
            populateSYCLToGPUConversionPatterns(patterns);

            target->addLegalDialect<arith::ArithDialect, spirv::SPIRVDialect,
                                    memref::MemRefDialect,
                                    vector::VectorDialect>();

            target->addDynamicallyLegalDialect<gpu::GPUDialect>(
                [](Operation *op) {
                  return isa<gpu::GPUModuleOp, gpu::ModuleEndOp>(op);
                });

            target->addDynamicallyLegalDialect<SYCLDialect>([](Operation *op) {
              return !isa<
                  // Convertible to GPU dialect
                  SYCLWorkGroupIDOp, SYCLNumWorkItemsOp, SYCLWorkGroupSizeOp,
                  SYCLLocalIDOp, SYCLGlobalIDOp, SYCLSubGroupIDOp,
                  SYCLNumSubGroupsOp, SYCLSubGroupSizeOp,
                  // Not convertible to GPU dialect
                  SYCLGlobalOffsetOp, SYCLNumWorkGroupsOp,
                  SYCLSubGroupLocalIDOp, SYCLSubGroupMaxSizeOp>(op);
            });

            // Add generic source and target materializations to handle cases
            // where non-LLVM types persist after an LLVM conversion.
            typeConverter.addSourceMaterialization(
                [&](OpBuilder &builder, Type resultType, ValueRange inputs,
                    Location loc) -> std::optional<Value> {
                  if (inputs.size() != 1)
                    return std::nullopt;

                  return builder
                      .create<UnrealizedConversionCastOp>(loc, resultType,
                                                          inputs)
                      .getResult(0);
                });
            typeConverter.addTargetMaterialization(
                [&](OpBuilder &builder, Type resultType, ValueRange inputs,
                    Location loc) -> std::optional<Value> {
                  if (inputs.size() != 1)
                    return std::nullopt;

                  return builder
                      .create<UnrealizedConversionCastOp>(loc, resultType,
                                                          inputs)
                      .getResult(0);
                });

            return applyPartialConversion(gpuModule, *target,
                                          std::move(patterns))
                           .failed()
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted());

  LLVM_DEBUG(llvm::dbgs() << "Module after SPIRV lowering:\n"; module.dump(););

  return res;
}

LogicalResult ConvertSYCLToLLVMPass::convertToLLVM(bool lastRun) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering to LLVM...\n");

  auto &context = getContext();
  auto module = getOperation();

  // TODO: As we may have device modules with different index widths, we may
  // need to revamp how we run this.

  LowerToLLVMOptions options(&context);
  options.useBarePtrCallConv = true;
  LLVMTypeConverter converter(&context, options);

  RewritePatternSet patterns(&context);

  // Keep these at the top; these should be run before the rest of
  // function conversion patterns.
  populateReturnOpTypeConversionPattern(patterns, converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
  polygeist::populateBareMemRefToLLVMConversionPatterns(converter, patterns);

  populateSYCLToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  populateVectorToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateSPIRVToLLVMTypeConversion(converter);
  populateSPIRVToLLVMConversionPatterns(converter, patterns);
  populateSPIRVToLLVMFunctionConversionPatterns(converter, patterns);

  LLVMConversionTarget target(context);
  if (!lastRun) {
    target.addDynamicallyLegalDialect<sycl::SYCLDialect>([](Operation *op) {
      return isa< // Convertible to GPU dialect
          SYCLWorkGroupIDOp, SYCLNumWorkItemsOp, SYCLWorkGroupSizeOp,
          SYCLLocalIDOp, SYCLGlobalIDOp, SYCLSubGroupIDOp, SYCLNumSubGroupsOp,
          SYCLSubGroupSizeOp,
          // Not convertible to GPU dialect
          SYCLGlobalOffsetOp, SYCLNumWorkGroupsOp, SYCLSubGroupLocalIDOp,
          SYCLSubGroupMaxSizeOp>(op);
    });
  }

  const auto res = applyPartialConversion(module, target, std::move(patterns));

  LLVM_DEBUG(llvm::dbgs() << "Module after LLVM lowering:\n"; module.dump(););

  return res;
}

void ConvertSYCLToLLVMPass::cleanUnrealizedConversionCasts() {
  LLVM_DEBUG(
      llvm::dbgs() << "Reconciling UnrealizedConversionCasts operations...\n");

  auto &context = getContext();
  auto module = getOperation();

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  populateReconcileUnrealizedCastsPatterns(patterns);

  if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
    LLVM_DEBUG(llvm::dbgs() << "WARNING: Could not remove every "
                               "UnrealizedConversionCast operation\n");
  }

  LLVM_DEBUG(
      llvm::dbgs()
          << "Module after reconciling UnrealizedConversionCasts operations:\n";
      module.dump(););
}

void ConvertSYCLToLLVMPass::runOnOperation() {
  if (convertToLLVM(/*lastRun*/ false).failed() || convertToSPIRV().failed() ||
      convertToLLVM(/*lastRun*/ true).failed()) {
    signalPassFailure();
    return;
  }
  // We will not signal pass failure here as the operations causing the failure
  // may come from previous conversions.
  cleanUnrealizedConversionCasts();
}
