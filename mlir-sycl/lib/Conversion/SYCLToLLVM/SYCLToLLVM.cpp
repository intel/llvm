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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-to-llvm"

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
  static constexpr std::array<int32_t, 2> indices{0, 1};
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
// IDGetPattern - Converts `sycl.it.get` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLIDGet with a scalar return type to LLVM
class IDGetPattern : public LoadMemberDimPattern<SYCLIDGetOp, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLIDGetOp, IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLIDGetOp op) const final {
    return success(op.getType().isa<IntegerType>());
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
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::sycl::populateSYCLToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
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

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  if (typeConverter.getOptions().useBarePtrCallConv) {
    patterns.add<BarePtrCastPattern>(typeConverter, /*benefit*/ 2);
    patterns.add<AtomicSubscriptIDOffset, IDGetPattern, IDGetRefPattern,
                 RangeGetPattern, RangeGetRefPattern, RangeSizePattern,
                 SubscriptIDOffset, SubscriptScalarOffset1D,
                 SubscriptScalarOffsetND>(typeConverter);
  }
  patterns.add<ConstructorPattern>(typeConverter);
}
