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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
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

struct OffsetTag {};

struct RangeGet : public OffsetTag {};
struct IDGet : public OffsetTag {};
struct NDRangeGetGlobalRange : public OffsetTag {};
struct NDRangeGetLocalRange : public OffsetTag {};
struct NDRangeGetOffset : public OffsetTag {};
struct AccessorSubscript : public OffsetTag {};
struct AccessorGetMemRange : public OffsetTag {};
struct ItemGetID : public OffsetTag {};
struct ItemGetRange : public OffsetTag {};
struct ItemGetOffset : public OffsetTag {};
struct GroupGetID : public OffsetTag {};
struct GroupGetGlobalRange : public OffsetTag {};
struct GroupGetLocalRange : public OffsetTag {};
struct GroupGetGroupRange : public OffsetTag {};
struct NDItemGlobalItem : public OffsetTag {};
struct NDItemLocalItem : public OffsetTag {};
struct NDItemGroup : public OffsetTag {};

/// Workaround to mimic static_assert(false), which is is illegal.
template <typename Tag,
          typename = std::enable_if_t<std::is_base_of_v<OffsetTag, Tag>>>
struct unhandled_tag : public std::false_type {};

template <typename Tag,
          typename = std::enable_if_t<std::is_base_of_v<OffsetTag, Tag>>>
constexpr static bool unhandled_tag_v{unhandled_tag<Tag>::value};

template <typename Iter, typename Tag,
          typename = std::enable_if_t<std::is_base_of_v<OffsetTag, Tag>>>
constexpr void initIndicesEach(Iter &it) {
  if constexpr (llvm::is_one_of<Tag, RangeGet, IDGet, ItemGetRange>::value) {
    *it++ = 0;
    *it++ = 0;
  } else if constexpr (llvm::is_one_of<Tag, NDRangeGetGlobalRange,
                                       GroupGetGlobalRange,
                                       NDItemGlobalItem>::value) {
    *it++ = 0;
  } else if constexpr (llvm::is_one_of<Tag, NDRangeGetLocalRange,
                                       GroupGetLocalRange,
                                       NDItemLocalItem>::value) {
    *it++ = 1;
  } else if constexpr (llvm::is_one_of<Tag, AccessorSubscript>::value) {
    *it++ = 1;
    *it++ = 0;
  } else if constexpr (llvm::is_one_of<Tag, AccessorGetMemRange,
                                       ItemGetID>::value) {
    *it++ = 0;
    *it++ = 1;
  } else if constexpr (llvm::is_one_of<Tag, ItemGetOffset>::value) {
    *it++ = 0;
    *it++ = 2;
  } else if constexpr (llvm::is_one_of<Tag, GroupGetGroupRange, NDItemGroup,
                                       NDRangeGetOffset>::value) {
    *it++ = 2;
  } else if constexpr (llvm::is_one_of<Tag, GroupGetID>::value) {
    *it++ = 3;
  } else {
    static_assert(unhandled_tag_v<Tag>, "Unhandled tag");
  }
}

template <typename... Tags> struct indices_size {
  static_assert(llvm::are_base_of<OffsetTag, Tags...>::value,
                "Only previously defined tags can be used in this context");
  static constexpr std::size_t value{(indices_size<Tags>::value + ...)};
};

template <> struct indices_size<RangeGet> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<IDGet> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<ItemGetRange> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<AccessorSubscript> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<AccessorGetMemRange> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<ItemGetID> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<ItemGetOffset> {
  static constexpr std::size_t value{2};
};

template <> struct indices_size<NDRangeGetGlobalRange> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<NDRangeGetLocalRange> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<NDRangeGetOffset> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<GroupGetID> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<GroupGetGlobalRange> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<GroupGetLocalRange> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<GroupGetGroupRange> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<NDItemGlobalItem> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<NDItemLocalItem> {
  static constexpr std::size_t value{1};
};

template <> struct indices_size<NDItemGroup> {
  static constexpr std::size_t value{1};
};

template <typename... Tags>
constexpr std::size_t indices_size_v{indices_size<Tags...>::value};

//===----------------------------------------------------------------------===//
// Utility patterns
//===----------------------------------------------------------------------===//

template <typename...> struct is_empty : public std::false_type {};
template <> struct is_empty<> : public std::true_type {};

template <typename... Args>
static constexpr bool is_empty_v{is_empty<Args...>::value};

class GetMemberPatternBase {
protected:
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<LLVM::GEPArg, Args...> ||
                is_empty_v<Args...>>>
  Value getRef(OpBuilder &builder, Location loc, Type ty, Value ptr,
               Args &&...args) const {
    SmallVector<LLVM::GEPArg> indices{0};
    const auto staticIndices = getIndices();
    indices.append(staticIndices.begin(), staticIndices.end());
    if constexpr (std::is_constructible_v<LLVM::GEPArg, Args...>) {
      // Add additional index if provided.
      indices.emplace_back(std::forward<Args>(args)...);
    }
    return builder.create<LLVM::GEPOp>(loc, ty, ptr, indices,
                                       /*inbounds*/ true);
  }

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

  virtual ArrayRef<int32_t> getIndices() const = 0;
};

template <typename Iter, typename... Tags>
constexpr void initIndices(Iter begin) {
  ((initIndicesEach<Iter, Tags>(begin)), ...);
}

template <typename... Tags>
class GetMemberPattern : public GetMemberPatternBase {
protected:
  constexpr ArrayRef<int32_t> getIndices() const final { return *indices; }

private:
  /// Struct definition to allow constexpr initialization of indices.
  static constexpr struct GetMemberPatternIndices {
    static constexpr std::size_t N{indices_size_v<Tags...>};

    constexpr GetMemberPatternIndices() {
      initIndices<typename std::array<int32_t, N>::iterator, Tags...>(
          indices.begin());
    }

    constexpr ArrayRef<int32_t> operator*() const { return indices; }

    std::array<int32_t, N> indices{0};
  } indices{};
};

template <typename Op, typename... Tags>
class GetRefToMemberPattern : public GetMemberPattern<Tags...>,
                              public ConvertOpToLLVMPattern<Op> {
protected:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(),
                                  getTypeConverter()->convertType(op.getType()),
                                  operands[0]));
  }
};

template <typename Op, typename... Tags>
class GetRefToMemberDimPattern : public GetMemberPattern<Tags...>,
                                 public ConvertOpToLLVMPattern<Op> {
protected:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(),
                                  getTypeConverter()->convertType(op.getType()),
                                  operands[0], operands[1]));
  }
};

template <typename Op, typename... Tags>
class LoadMemberPattern : public GetMemberPattern<Tags...>,
                          public ConvertOpToLLVMPattern<Op> {
protected:
  using GetMemberPattern<Tags...>::loadValue;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
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

template <typename Op, typename... Tags>
class LoadMemberDimPattern : public GetMemberPattern<Tags...>,
                             public ConvertOpToLLVMPattern<Op> {
protected:
  using GetMemberPattern<Tags...>::loadValue;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
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

template <typename Op, typename GridOp, typename... Tags>
class SPIRVInitPattern : public ConvertOpToLLVMPattern<Op>,
                         public GetMemberPattern<Tags...> {
protected:
  using GetMemberPattern<Tags...>::getRef;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    const auto elTy = getTypeConverter()->convertType(op.getType());
    const auto ptrTy = LLVM::LLVMPointerType::get(elTy);
    const Value arraySize =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    const Value alloca =
        rewriter.create<LLVM::AllocaOp>(loc, ptrTy, elTy, arraySize);
    const auto structMemberTy = rewriter.getI64Type();
    const auto dimTy = rewriter.getIndexType();
    const auto dimGetTy = rewriter.getI32Type();
    const auto innerPtrTy = LLVM::LLVMPointerType::get(structMemberTy);
    for (unsigned i = 0,
                  dimensions = getDimensions(op->getOperand(0).getType());
         i < dimensions; ++i) {
      const auto ref = getRef(rewriter, loc, innerPtrTy, alloca, i);
      const Value dim = rewriter.create<LLVM::ConstantOp>(loc, dimGetTy, i);
      const Value val = rewriter.create<arith::IndexCastUIOp>(
          loc, structMemberTy, rewriter.create<GridOp>(loc, dimTy, dim));
      rewriter.create<LLVM::StoreOp>(loc, ref, val);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
  }
};

template <typename Op, typename GridOp>
class SPIRVInitDimPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

public:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
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

/// Converts SYCL AssertHappened type to LLVM type.
static Optional<Type> convertAssertHappenedType(sycl::AssertHappenedType type,
                                                LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::AssertHappened",
                         type.getBody(), converter);
}

/// Converts SYCL atomic type to LLVM type.
static Optional<Type> convertAtomicType(sycl::AtomicType type,
                                        LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::atomic", type.getBody(), converter);
}

/// Converts SYCL bfloat16 type to LLVM type.
static Optional<Type> convertBFloat16Type(sycl::BFloat16Type type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::ext::oneapi::bfloat16",
                         type.getBody(), converter);
}

/// Converts SYCL GetOp type to LLVM type.
static Optional<Type> convertGetOpType(sycl::GetOpType type,
                                       LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::GetOp", converter);
}

/// Converts SYCL GetScalarOp type to LLVM type.
static Optional<Type> convertGetScalarOpType(sycl::GetScalarOpType type,
                                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::GetScalarOp", type.getBody(),
                         converter);
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

/// Converts SYCL TupleCopyAssignableValueHolder type to LLVM type.
static Optional<Type> convertTupleCopyAssignableValueHolderType(
    sycl::TupleCopyAssignableValueHolderType type,
    LLVMTypeConverter &converter) {
  return convertBodyType(
      "struct.sycl::_V1::detail::TupleCopyAssignableValueHolder",
      type.getBody(), converter);
}

/// Converts SYCL TupleValueHolder type to LLVM type.
static Optional<Type>
convertTupleValueHolderType(sycl::TupleValueHolderType type,
                            LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::TupleValueHolder",
                         type.getBody(), converter);
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
// SYCLRangeGetPattern - Convert `sycl.range.get` to LLVM.
//===----------------------------------------------------------------------===//

class RangeGetPattern : public LoadMemberDimPattern<SYCLRangeGetOp, RangeGet> {
public:
  using LoadMemberDimPattern<SYCLRangeGetOp, RangeGet>::LoadMemberDimPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<IntegerType>());
  }
};

class RangeGetRefPattern
    : public GetRefToMemberDimPattern<SYCLRangeGetOp, RangeGet> {
public:
  using GetRefToMemberDimPattern<SYCLRangeGetOp,
                                 RangeGet>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<MemRefType>());
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeSizePattern - Convert `sycl.range.size` to LLVM.
//===----------------------------------------------------------------------===//

class RangeSizePattern : public ConvertOpToLLVMPattern<SYCLRangeSizeOp>,
                         public GetMemberPattern<RangeGet> {
public:
  using ConvertOpToLLVMPattern<SYCLRangeSizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLRangeSizeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ty = rewriter.getI64Type();
    Value size = rewriter.create<LLVM::ConstantOp>(loc, ty, 1);
    const auto range = opAdaptor.getRange();
    const auto ptrTy = LLVM::LLVMPointerType::get(
        ty, range.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    for (unsigned i = 0, dim = getDimensions(op.getRange().getType()); i < dim;
         ++i) {
      const auto val = loadValue(rewriter, loc, ptrTy, range, i);
      size = rewriter.create<LLVM::MulOp>(loc, size, val);
    }
    rewriter.replaceOp(op, size);
    return success();
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
      public GetMemberPattern<NDRangeGetGlobalRange, RangeGet>,
      public GetMemberPattern<NDRangeGetLocalRange, RangeGet>,
      public GetMemberPattern<RangeGet> {
  template <typename... Args> Value loadGlobal(Args &&...args) const {
    return GetMemberPattern<NDRangeGetGlobalRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value loadLocal(Args &&...args) const {
    return GetMemberPattern<NDRangeGetLocalRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value refRange(Args &&...args) const {
    return GetMemberPattern<RangeGet>::getRef(std::forward<Args>(args)...);
  }

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
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1),
        /*alignment*/ 0);
    const auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    for (int32_t i = 0, dim = rangeTy.getDimension(); i < dim; ++i) {
      const auto lhs = loadGlobal(rewriter, loc, ptrTy, nd, i);
      const auto rhs = loadLocal(rewriter, loc, ptrTy, nd, i);
      const Value val = rewriter.create<LLVM::UDivOp>(loc, lhs, rhs);
      const auto ptr = refRange(rewriter, loc, ptrTy, alloca, i);
      rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AccessorSubscriptPattern - Convert `sycl.accessor.subscript` to LLVM.
//===----------------------------------------------------------------------===//

/// Base class for other patterns converting `sycl.accessor.subscript` to LLVM.
class AccessorSubscriptPattern
    : public ConvertOpToLLVMPattern<SYCLAccessorSubscriptOp>,
      public GetMemberPattern<AccessorSubscript> {
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
};

class AccessorSubscriptIDIndexPattern
    : public AccessorSubscriptPattern,
      public GetMemberPattern<IDGet>,
      public GetMemberPattern<AccessorGetMemRange, RangeGet> {
  template <typename... Args> Value getID(Args &&...args) const {
    return GetMemberPattern<IDGet>::loadValue(std::forward<Args>(args)...);
  }

  template <typename... Args> Value getMemRange(Args &&...args) const {
    return GetMemberPattern<AccessorGetMemRange, RangeGet>::loadValue(
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
    Value res = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), 0);
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
      res = builder.create<LLVM::AddOp>(
          loc, builder.create<LLVM::MulOp>(loc, res, memI), idI);
    }
    return res;
  }
};

/// Conversion pattern with non-atomic access mode and id offset type.
class SubscriptIDOffset : public AccessorSubscriptIDIndexPattern {
public:
  using AccessorSubscriptIDIndexPattern::AccessorSubscriptIDIndexPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   AccessorSubscriptPattern::hasIDOffsetType(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ptrTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOp(
        op, GetMemberPattern<AccessorSubscript>::getRef(
                rewriter, loc, ptrTy, opAdaptor.getAcc(),
                getLinearIndex(
                    rewriter, loc,
                    op.getAcc().getType().getElementType().cast<AccessorType>(),
                    opAdaptor)));
  }
};

/// Conversion pattern with non-atomic access mode, scalar offset type and
/// 1-dimensional accessor.
class SubscriptScalarOffset1D
    : public GetRefToMemberDimPattern<SYCLAccessorSubscriptOp,
                                      AccessorSubscript> {
public:
  using GetRefToMemberDimPattern<SYCLAccessorSubscriptOp,
                                 AccessorSubscript>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   AccessorSubscriptPattern::has1DAccessor(op));
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
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op) &&
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
    const Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
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
    const Value ptr = GetMemberPattern<AccessorSubscript>::getRef(
        rewriter, loc, ptrTy, opAdaptor.getAcc(),
        getLinearIndex(
            rewriter, loc,
            op.getAcc().getType().getElementType().cast<AccessorType>(),
            opAdaptor));
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undef, ptr, 0);
  }
};

/// Conversion pattern with atomic access mode and scalar offset type.
class AtomicSubscriptScalarOffset : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op));
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
    const Value ptr = GetMemberPattern<AccessorSubscript>::getRef(
        rewriter, loc, ptrTy, opAdaptor.getAcc(), opAdaptor.getIndex());
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undef, ptr, 0);
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
    : public LoadMemberDimPattern<SYCLItemGetIDOp, ItemGetID, IDGet> {
public:
  using LoadMemberDimPattern<SYCLItemGetIDOp, ItemGetID,
                             IDGet>::LoadMemberDimPattern;

  LogicalResult match(SYCLItemGetIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
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
    : public LoadMemberDimPattern<SYCLItemGetRangeOp, ItemGetRange, RangeGet> {
public:
  using LoadMemberDimPattern<SYCLItemGetRangeOp, ItemGetRange,
                             RangeGet>::LoadMemberDimPattern;

  LogicalResult match(SYCLItemGetRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// ItemGetLinearIDPattern - Converts `sycl.item.get_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

class ItemGetLinearIDPattern
    : public ConvertOpToLLVMPattern<SYCLItemGetLinearIDOp>,
      public GetMemberPattern<ItemGetID, IDGet>,
      public GetMemberPattern<ItemGetRange, RangeGet> {
protected:
  template <typename... Args> Value getID(Args &&...args) const {
    return GetMemberPattern<ItemGetID, IDGet>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getRange(Args &&...args) const {
    return GetMemberPattern<ItemGetRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<SYCLItemGetLinearIDOp>::ConvertOpToLLVMPattern;
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

  void rewrite(SYCLItemGetLinearIDOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto ptrTy = LLVM::LLVMPointerType::get(
        op.getType(), opAdaptor.getItem()
                          .getType()
                          .cast<LLVM::LLVMPointerType>()
                          .getAddressSpace());
    const auto newValue = [this, loc = op.getLoc(), item = opAdaptor.getItem(),
                           ptrTy,
                           dimension = getDimensions(op.getItem().getType()),
                           &builder = rewriter]() -> Value {
      switch (dimension) {
      case 1:
        // get_id(0)
        return getID(builder, loc, ptrTy, item, 0);
      case 2: {
        // get_id(0) * get_range(1) + get_id(1)
        const auto id0 = getID(builder, loc, ptrTy, item, 0);
        const auto r1 = getRange(builder, loc, ptrTy, item, 1);
        const auto prod =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, id0, r1));
        const auto id1 = getID(builder, loc, ptrTy, item, 1);
        return builder.create<LLVM::AddOp>(loc, prod, id1);
      }
      case 3: {
        // get_id(0) * get_range(1) * get_range(2) + get_id(1) * get_range(2) +
        // get_id(2)
        const auto id0 = getID(builder, loc, ptrTy, item, 0);
        const auto r1 = getRange(builder, loc, ptrTy, item, 1);
        const auto prod0 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, id0, r1));
        const auto r2 = getRange(builder, loc, ptrTy, item, 2);
        const auto prod1 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, prod0, r2));
        const auto id1 = getID(builder, loc, ptrTy, item, 1);
        const auto prod2 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, id1, r2));
        const auto add =
            static_cast<Value>(builder.create<LLVM::AddOp>(loc, prod1, prod2));
        const auto id2 = getID(builder, loc, ptrTy, item, 2);
        return builder.create<LLVM::AddOp>(loc, add, id2);
      }
      default:
        llvm_unreachable("Invalid number of dimensions");
      }
    }();
    rewriter.replaceOp(op, newValue);
  }
};

/// Converts SYCLItemGetLinearIDOp with no offset item to LLVM
class ItemOffsetGetLinearIDPattern
    : public ItemGetLinearIDPattern,
      public GetMemberPattern<ItemGetOffset, IDGet> {
  template <typename... Args> Value getOffset(Args &&...args) const {
    return GetMemberPattern<ItemGetOffset, IDGet>::loadValue(
        std::forward<Args>(args)...);
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

  void rewrite(SYCLItemGetLinearIDOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto ptrTy = LLVM::LLVMPointerType::get(
        op.getType(), opAdaptor.getItem()
                          .getType()
                          .cast<LLVM::LLVMPointerType>()
                          .getAddressSpace());
    const auto newValue = [this, ptrTy, loc = op.getLoc(),
                           item = opAdaptor.getItem(),
                           dimension = getDimensions(op.getItem().getType()),
                           &builder = rewriter]() -> Value {
      const auto getDim = [this, ptrTy, loc, item,
                           &builder](int32_t dim) -> Value {
        const auto id = getID(builder, loc, ptrTy, item, dim);
        const auto off = getOffset(builder, loc, ptrTy, item, dim);
        return builder.create<LLVM::SubOp>(loc, id, off);
      };
      switch (dimension) {
      case 1:
        // get_id(0) - get_offset(0)
        return getDim(0);
      case 2: {
        // (get_id(0) - get_offset(0)) * get_range(1) + (get_id(1) -
        // get_offset(1))
        const auto diff0 = getDim(0);
        const auto r1 = getRange(builder, loc, ptrTy, item, 1);
        const auto prod =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, diff0, r1));
        const auto diff1 = getDim(1);
        return builder.create<LLVM::AddOp>(loc, prod, diff1);
      }
      case 3: {
        // (get_id(0) - get_offset(0)) * get_range(1) * get_range(2) +
        // (get_id(1) - get_offset(1)) * get_range(2) + (get_id(2) -
        // get_offset(2))
        const auto diff0 = getDim(0);
        const auto r1 = getRange(builder, loc, ptrTy, item, 1);
        const auto prod0 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, diff0, r1));
        const auto r2 = getRange(builder, loc, ptrTy, item, 2);
        const auto prod1 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, prod0, r2));
        const auto diff1 = getDim(1);
        const auto prod2 =
            static_cast<Value>(builder.create<LLVM::MulOp>(loc, diff1, r2));
        const auto add =
            static_cast<Value>(builder.create<LLVM::AddOp>(loc, prod1, prod2));
        const auto diff2 = getDim(2);
        return builder.create<LLVM::AddOp>(loc, add, diff2);
      }
      default:
        llvm_unreachable("Invalid number of dimensions");
      }
    }();
    rewriter.replaceOp(op, newValue);
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
    : public LoadMemberPattern<SYCLGroupGetGroupIDOp, GroupGetID, IDGet> {
public:
  using LoadMemberPattern<SYCLGroupGetGroupIDOp, GroupGetID,
                          IDGet>::LoadMemberPattern;

  LogicalResult match(SYCLGroupGetGroupIDOp op) const final {
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
                                  RangeGet> {
public:
  using LoadMemberDimPattern<SYCLGroupGetLocalRangeOp, GroupGetLocalRange,
                             RangeGet>::LoadMemberDimPattern;

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

/// Converts SYCLGroupGetGroupRange with a scalar return type to LLVM
class GroupGetGroupRangeDimPattern
    : public LoadMemberDimPattern<SYCLGroupGetGroupRangeOp, GroupGetGroupRange,
                                  RangeGet> {
public:
  using LoadMemberDimPattern<SYCLGroupGetGroupRangeOp, GroupGetGroupRange,
                             RangeGet>::LoadMemberDimPattern;

  LogicalResult match(SYCLGroupGetGroupRangeOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
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

//===----------------------------------------------------------------------===//
// GroupGetLocalID - Converts `sycl.group.get_local_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalID with an ID return type to LLVM
class GroupGetLocalIDPattern
    : public SPIRVInitPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp, IDGet> {
public:
  using SPIRVInitPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp,
                         IDGet>::SPIRVInitPattern;

  LogicalResult match(SYCLGroupGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IDType>());
  }
};

/// Converts SYCLGroupGetLocalID with a scalar return type to LLVM
class GroupGetLocalIDDimPattern
    : public SPIRVInitDimPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp> {
public:
  using SPIRVInitDimPattern<SYCLGroupGetLocalIDOp,
                            SYCLLocalIDOp>::SPIRVInitDimPattern;

  LogicalResult match(SYCLGroupGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
};

//===----------------------------------------------------------------------===//
// GroupGetGroupLinearID - Converts `sycl.group.get_group_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupLinearIDOp to LLVM
class GroupGetGroupLinearIDPattern
    : public ConvertOpToLLVMPattern<SYCLGroupGetGroupLinearIDOp>,
      public GetMemberPattern<GroupGetGroupRange, RangeGet>,
      public GetMemberPattern<GroupGetID, IDGet> {
  template <typename... Args> Value getID(Args &&...args) const {
    return GetMemberPattern<GroupGetID, IDGet>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getRange(Args &&...args) const {
    return GetMemberPattern<GroupGetGroupRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<
      SYCLGroupGetGroupLinearIDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLGroupGetGroupLinearIDOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto addressSpace = opAdaptor.getGroup()
                                  .getType()
                                  .cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    const auto ptrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(op.getType()), addressSpace);
    const auto newValue =
        [this, ptrTy, group = opAdaptor.getGroup(),
         dimensions = getDimensions(op->getOperand(0).getType()),
         loc = op.getLoc(), opAdaptor, &builder = rewriter]() -> Value {
      switch (dimensions) {
      case 1:
        return getID(builder, loc, ptrTy, group, 0);
      case 2: {
        const auto id0 = getID(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const Value m0 = builder.create<LLVM::MulOp>(loc, id0, r1);
        const auto id1 = getID(builder, loc, ptrTy, group, 1);
        return builder.create<LLVM::AddOp>(loc, m0, id1);
      }
      case 3: {
        const auto id0 = getID(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const auto r2 = getRange(builder, loc, ptrTy, group, 2);
        const Value m0 = builder.create<LLVM::MulOp>(loc, id0, r1);
        const Value m1 = builder.create<LLVM::MulOp>(loc, m0, r2);
        const auto id1 = getID(builder, loc, ptrTy, group, 1);
        const Value m2 = builder.create<LLVM::MulOp>(loc, id1, r2);
        const Value add0 = builder.create<LLVM::AddOp>(loc, m1, m2);
        const auto id2 = getID(builder, loc, ptrTy, group, 2);
        return builder.create<LLVM::AddOp>(loc, add0, id2);
      }
      }
      llvm_unreachable("Invalid number of dimensions");
    }();
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupGetLocalLinearID - Converts `sycl.group.get_local_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalLinearIDOp to LLVM
class GroupGetLocalLinearIDPattern
    : public ConvertOpToLLVMPattern<SYCLGroupGetLocalLinearIDOp>,
      public GetMemberPattern<GroupGetGroupRange, RangeGet> {
  Value getID(OpBuilder &builder, Location loc, Type ptrTy, Value,
              int32_t offset) const {
    const Value dim =
        builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), offset);
    const Value val =
        builder.create<SYCLLocalIDOp>(loc, builder.getI64Type(), dim);
    return builder.create<arith::IndexCastUIOp>(
        loc, ptrTy.cast<LLVM::LLVMPointerType>().getElementType(), val);
  }

  template <typename... Args> Value getRange(Args &&...args) const {
    return GetMemberPattern<GroupGetGroupRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<
      SYCLGroupGetLocalLinearIDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLGroupGetLocalLinearIDOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto addressSpace = opAdaptor.getGroup()
                                  .getType()
                                  .cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    const auto ptrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(op.getType()), addressSpace);
    const auto newValue =
        [this, ptrTy, group = opAdaptor.getGroup(),
         dimensions = getDimensions(op->getOperand(0).getType()),
         loc = op.getLoc(), opAdaptor, &builder = rewriter]() -> Value {
      switch (dimensions) {
      case 1:
        return getID(builder, loc, ptrTy, group, 0);
      case 2: {
        const auto id0 = getID(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const Value m0 = builder.create<LLVM::MulOp>(loc, id0, r1);
        const auto id1 = getID(builder, loc, ptrTy, group, 1);
        return builder.create<LLVM::AddOp>(loc, m0, id1);
      }
      case 3: {
        const auto id0 = getID(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const auto r2 = getRange(builder, loc, ptrTy, group, 2);
        const Value m0 = builder.create<LLVM::MulOp>(loc, id0, r1);
        const Value m1 = builder.create<LLVM::MulOp>(loc, m0, r2);
        const auto id1 = getID(builder, loc, ptrTy, group, 1);
        const Value m2 = builder.create<LLVM::MulOp>(loc, id1, r2);
        const Value add0 = builder.create<LLVM::AddOp>(loc, m1, m2);
        const auto id2 = getID(builder, loc, ptrTy, group, 2);
        return builder.create<LLVM::AddOp>(loc, add0, id2);
      }
      }
      llvm_unreachable("Invalid number of dimensions");
    }();
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupGetGroupLinearRange - Converts `sycl.group.get_group_linear_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetGroupLinearRangeOp to LLVM
class GroupGetGroupLinearRangePattern
    : public ConvertOpToLLVMPattern<SYCLGroupGetGroupLinearRangeOp>,
      public GetMemberPattern<GroupGetGroupRange, RangeGet> {

  template <typename... Args> Value getRange(Args &&...args) const {
    return GetMemberPattern<GroupGetGroupRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<
      SYCLGroupGetGroupLinearRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLGroupGetGroupLinearRangeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto addressSpace = opAdaptor.getGroup()
                                  .getType()
                                  .cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    const auto ptrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(op.getType()), addressSpace);
    const auto newValue =
        [this, ptrTy, group = opAdaptor.getGroup(),
         dimensions = getDimensions(op->getOperand(0).getType()),
         loc = op.getLoc(), opAdaptor, &builder = rewriter]() -> Value {
      switch (dimensions) {
      case 1:
        return getRange(builder, loc, ptrTy, group, 0);
      case 2: {
        const auto r0 = getRange(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        return builder.create<LLVM::MulOp>(loc, r0, r1);
      }
      case 3: {
        const auto r0 = getRange(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const Value m = builder.create<LLVM::MulOp>(loc, r0, r1);
        const auto r2 = getRange(builder, loc, ptrTy, group, 2);
        return builder.create<LLVM::MulOp>(loc, m, r2);
      }
      }
      llvm_unreachable("Invalid number of dimensions");
    }();
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupGetLocalLinearRange - Converts `sycl.group.get_local_linear_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalLinearRangeOp to LLVM
class GroupGetLocalLinearRangePattern
    : public ConvertOpToLLVMPattern<SYCLGroupGetLocalLinearRangeOp>,
      public GetMemberPattern<GroupGetLocalRange, RangeGet> {
  template <typename... Args> Value getRange(Args &&...args) const {
    return GetMemberPattern<GroupGetLocalRange, RangeGet>::loadValue(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<
      SYCLGroupGetLocalLinearRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLGroupGetLocalLinearRangeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto addressSpace = opAdaptor.getGroup()
                                  .getType()
                                  .cast<LLVM::LLVMPointerType>()
                                  .getAddressSpace();
    const auto ptrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(op.getType()), addressSpace);
    const auto newValue =
        [this, ptrTy, group = opAdaptor.getGroup(),
         dimensions = getDimensions(op->getOperand(0).getType()),
         loc = op.getLoc(), opAdaptor, &builder = rewriter]() -> Value {
      switch (dimensions) {
      case 1:
        return getRange(builder, loc, ptrTy, group, 0);
      case 2: {
        const auto r0 = getRange(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        return builder.create<LLVM::MulOp>(loc, r0, r1);
      }
      case 3: {
        const auto r0 = getRange(builder, loc, ptrTy, group, 0);
        const auto r1 = getRange(builder, loc, ptrTy, group, 1);
        const Value m = builder.create<LLVM::MulOp>(loc, r0, r1);
        const auto r2 = getRange(builder, loc, ptrTy, group, 2);
        return builder.create<LLVM::MulOp>(loc, m, r2);
      }
      }
      llvm_unreachable("Invalid number of dimensions");
    }();
    rewriter.replaceOp(op, newValue);
    return success();
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
    : public LoadMemberPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem,
                               ItemGetID, IDGet> {
public:
  using LoadMemberPattern<SYCLNDItemGetGlobalIDOp, NDItemGlobalItem, ItemGetID,
                          IDGet>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetGlobalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
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
    : public LoadMemberPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem,
                               ItemGetID, IDGet> {
public:
  using LoadMemberPattern<SYCLNDItemGetLocalIDOp, NDItemLocalItem, ItemGetID,
                          IDGet>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetLocalIDOp op) const final {
    return success(op.getRes().getType().isa<IntegerType>());
  }
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
    : public LoadMemberPattern<SYCLNDItemGetGroupOp, NDItemGroup, GroupGetID,
                               IDGet> {
public:
  using LoadMemberPattern<SYCLNDItemGetGroupOp, NDItemGroup, GroupGetID,
                          IDGet>::LoadMemberPattern;

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
    : public LoadMemberPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                               GroupGetGroupRange, RangeGet> {
public:
  using LoadMemberPattern<SYCLNDItemGetGroupRangeOp, NDItemGroup,
                          GroupGetGroupRange, RangeGet>::LoadMemberPattern;

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
    : public LoadMemberPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                               ItemGetRange, RangeGet> {
public:
  using LoadMemberPattern<SYCLNDItemGetLocalRangeOp, NDItemLocalItem,
                          ItemGetRange, RangeGet>::LoadMemberPattern;

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
  template <typename... Args> Value getGlobalRange(Args &&...args) const {
    return GetMemberPattern<NDItemGlobalItem, ItemGetRange>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getLocalRange(Args &&...args) const {
    return GetMemberPattern<NDItemLocalItem, ItemGetRange>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getOffset(Args &&...args) const {
    return GetMemberPattern<NDItemGlobalItem, ItemGetOffset>::loadValue(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getGlobalRangeRef(Args &&...args) const {
    return GetMemberPattern<NDRangeGetGlobalRange>::getRef(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getLocalRangeRef(Args &&...args) const {
    return GetMemberPattern<NDRangeGetLocalRange>::getRef(
        std::forward<Args>(args)...);
  }

  template <typename... Args> Value getOffsetRef(Args &&...args) const {
    return GetMemberPattern<NDRangeGetOffset>::getRef(
        std::forward<Args>(args)...);
  }

public:
  using ConvertOpToLLVMPattern<SYCLNDItemGetNdRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNDItemGetNdRangeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto ndItem = opAdaptor.getNDItem();
    const auto ndPtrTy = LLVM::LLVMPointerType::get(
        rewriter.getI64Type(),
        ndItem.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    const auto allocaPtrTy = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    const Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc,
        LLVM::LLVMPointerType::get(
            getTypeConverter()->convertType(op.getType())),
        getTypeConverter()->convertType(op.getType()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1));
    rewriter.create<LLVM::StoreOp>(
        loc, getGlobalRange(rewriter, loc, ndPtrTy, ndItem),
        getGlobalRangeRef(rewriter, loc, allocaPtrTy, alloca));
    rewriter.create<LLVM::StoreOp>(
        loc, getLocalRange(rewriter, loc, ndPtrTy, ndItem),
        getLocalRangeRef(rewriter, loc, allocaPtrTy, alloca));
    rewriter.create<LLVM::StoreOp>(
        loc, getOffset(rewriter, loc, ndPtrTy, ndItem),
        getOffsetRef(rewriter, loc, allocaPtrTy, alloca));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
    return success();
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
  typeConverter.addConversion([&](sycl::AssertHappenedType type) {
    return convertAssertHappenedType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AtomicType type) {
    return convertAtomicType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::BFloat16Type type) {
    return convertBFloat16Type(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GetOpType type) {
    return convertGetOpType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GetScalarOpType type) {
    return convertGetScalarOpType(type, typeConverter);
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
      [&](sycl::TupleCopyAssignableValueHolderType type) {
        return convertTupleCopyAssignableValueHolderType(type, typeConverter);
      });
  typeConverter.addConversion([&](sycl::TupleValueHolderType type) {
    return convertTupleValueHolderType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::VecType type) { return convertVecType(type, typeConverter); });
}

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  if (typeConverter.getOptions().useBarePtrCallConv)
    patterns.add<BarePtrCastPattern>(typeConverter, /*benefit*/ 2);
  patterns.add<ConstructorPattern>(typeConverter);
  if (typeConverter.getOptions().useBarePtrCallConv)
    patterns.add<AtomicSubscriptIDOffset, AtomicSubscriptScalarOffset,
                 GroupGetGroupIDPattern, GroupGetGroupIDDimPattern,
                 GroupGetLocalIDPattern, GroupGetLocalIDDimPattern,
                 GroupGetLocalRangePattern, GroupGetLocalRangeDimPattern,
                 GroupGetGroupRangePattern, GroupGetGroupRangeDimPattern,
                 GroupGetMaxLocalRangePattern, GroupGetGroupLinearIDPattern,
                 GroupGetLocalLinearIDPattern, GroupGetGroupLinearRangePattern,
                 GroupGetLocalLinearRangePattern, ItemGetIDDimPattern,
                 ItemGetIDPattern, ItemGetRangeDimPattern, ItemGetRangePattern,
                 ItemNoOffsetGetLinearIDPattern, ItemOffsetGetLinearIDPattern,
                 NDItemGetGlobalIDPattern, NDItemGetGlobalIDDimPattern,
                 NDItemGetLocalIDPattern, NDItemGetLocalIDDimPattern,
                 NDItemGetGroupPattern, NDItemGetGroupDimPattern,
                 NDItemGetGroupRangePattern, NDItemGetGroupRangeDimPattern,
                 NDItemGetLocalRangePattern, NDItemGetLocalRangeDimPattern,
                 NDItemGetNDRange, NDRangeGetGlobalRangePattern,
                 NDRangeGetGroupRangePattern, NDRangeGetLocalRangePattern,
                 RangeGetPattern, RangeGetRefPattern, RangeSizePattern,
                 SubscriptIDOffset, SubscriptScalarOffset1D,
                 SubscriptScalarOffsetND>(typeConverter);
}
