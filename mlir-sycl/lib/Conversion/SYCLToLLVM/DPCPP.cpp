//===- DPCPP.cpp - SYCL to LLVM Patterns for the DPC++ implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DPCPP.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::sycl;

#define DEBUG_TYPE "sycl-to-llvm"

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Returns true if the given type is 'memref<?xSYCLType>', and false otherwise.
template <typename SYCLType> static bool isMemRefOf(const Type &type) {
  if (!isa<MemRefType>(type))
    return false;

  MemRefType memRefTy = cast<MemRefType>(type);
  ArrayRef<int64_t> shape = memRefTy.getShape();
  if (shape.size() != 1 || shape[0] != -1)
    return false;

  return isa<SYCLType>(memRefTy.getElementType());
}

// Returns the element type of 'memref<?xSYCLType>'.
template <typename SYCLType> static SYCLType getElementType(const Type &type) {
  assert(isMemRefOf<SYCLType>(type) && "Expecting memref<?xsycl::<type>>");
  Type elemType = cast<MemRefType>(type).getElementType();
  return cast<SYCLType>(elemType);
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

/// Get the ID field from an accessor.
struct AccessorGetID : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 0};
};

/// Get the MAccessRange field from an accessor.
struct AccessorGetMAccessRange : public OffsetTag {
  static constexpr std::array<int32_t, 2> indices{0, 1};
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

  /// Returns a reference to type \p ty to a member of the struct pointed by \p
  /// ptr in address space \p targetAddressSpace.
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<LLVM::GEPArg, Args...> ||
                is_empty_v<Args...>>>
  Value getRef(OpBuilder &builder, Location loc, Type baseTy, Value ptr,
               std::optional<unsigned> targetAddressSpace,
               Args &&...args) const {
    SmallVector<LLVM::GEPArg> indices{0};
    const auto staticIndices = getIndices();
    indices.append(staticIndices.begin(), staticIndices.end());
    if constexpr (!is_empty_v<Args...>) {
      // Add additional index if provided.
      indices.emplace_back(std::forward<Args>(args)...);
    }
    const auto origAddressSpace =
        cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();

    const auto addressSpace = targetAddressSpace.value_or(origAddressSpace);

    if (origAddressSpace != addressSpace) {
      auto ptrTy =
          LLVM::LLVMPointerType::get(baseTy.getContext(), addressSpace);
      ptr = builder.create<LLVM::AddrSpaceCastOp>(loc, ptrTy, ptr);
    }

    const auto ptrTy =
        LLVM::LLVMPointerType::get(baseTy.getContext(), addressSpace);
    return builder.create<LLVM::GEPOp>(loc, ptrTy, baseTy, ptr, indices,
                                       /*inbounds*/ true);
  }

  /// Returns a value of type \p ty being a member of the struct pointed by \p
  /// ptr.
  ///
  /// Effectively calls getRef() and loads the value.
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<LLVM::GEPArg, Args...> ||
                is_empty_v<Args...>>>
  Value loadValue(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                  Value ptr, Args &&...args) const {
    const auto elemTy = baseTy;
    const auto gep = getRef<Args...>(builder, loc, elemTy, ptr, std::nullopt,
                                     std::forward<Args>(args)...);
    return builder.create<LLVM::LoadOp>(loc, ty, gep);
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
    const auto ptrTy = getTypeConverter()
                           ->convertType(op.getType())
                           .template cast<LLVM::LLVMPointerType>();

    auto baseTy = getTypeConverter()->convertType(
        cast<MemRefType>(op.getOperands()[0].getType()).getElementType());
    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(), baseTy,
                                  ptrTy.getAddressSpace(), operands[0]));
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
    const auto ptrTy = getTypeConverter()
                           ->convertType(op.getType())
                           .template cast<LLVM::LLVMPointerType>();

    auto baseTy = getTypeConverter()->convertType(
        cast<MemRefType>(op.getOperands()[0].getType()).getElementType());

    rewriter.replaceOp(op, getRef(rewriter, op.getLoc(), baseTy, operands[0],
                                  ptrTy.getAddressSpace(), operands[1]));
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
    // Assume the first operand is the 'this' argument.
    const auto baseTy = getTypeConverter()->convertType(
        cast<MemRefType>(op->getOperand(0).getType()).getElementType());
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op,
                       loadValue(rewriter, op.getLoc(), baseTy,
                                 getTypeConverter()->convertType(op.getType()),
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
    // Assume the first operand is the 'this' argument.
    const auto baseTy = getTypeConverter()->convertType(
        cast<MemRefType>(op->getOperand(0).getType()).getElementType());
    const auto operands = adaptor.getOperands();
    rewriter.replaceOp(op,
                       loadValue(rewriter, op.getLoc(), baseTy,
                                 getTypeConverter()->convertType(op.getType()),
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
    return success(op.getNumOperands() == 1 && op.getType().isIntOrIndex());
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
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

  virtual Value getRange(OpBuilder &builder, Location loc, Type baseTy,
                         Type ptrTy, Value thisArg, int32_t index) const = 0;

  virtual Type getBaseType(Op op) const = 0;

public:
  virtual ~GetRangeSizePattern() = default;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto thisArg = adaptor.getOperands()[0];
    const auto elTy = getTypeConverter()->convertType(op.getType());
    const auto loc = op.getLoc();
    const auto dimension = getDimensions(op.getOperand().getType());
    const auto convBaseTy = getTypeConverter()->convertType(getBaseType(op));
    assert(1 <= dimension && dimension < 4 && "Invalid number of dimensions");
    Value newValue =
        rewriter.create<arith::ConstantIntOp>(loc, 1, op.getType());
    for (unsigned i = 0; i < dimension; ++i) {
      const auto size = getRange(rewriter, loc, convBaseTy, elTy, thisArg, i);
      newValue = rewriter.create<arith::MulIOp>(loc, newValue, size);
    }
    rewriter.replaceOp(op, newValue);
  }
};

/// Helper function to calculate the linear ID using ID and range getters.
template <typename IDGetter, typename RangeGetter>
static void getLinearIDRewriter(Operation *op, unsigned dimension,
                                IDGetter getID, RangeGetter getRange,
                                ConversionPatternRewriter &rewriter) {
  const auto loc = op->getLoc();
  Value linearID;
  assert(1 <= dimension && dimension < 4 && "Invalid number of dimensions");
  std::array<Value, 3> ranges;
  const auto getRangeCached = [&](unsigned dim) {
    auto &r = ranges[dim];
    if (!r)
      r = getRange(rewriter, loc, dim);
    return r;
  };
  for (unsigned i = 0; i < dimension; ++i) {
    Value id = getID(rewriter, loc, i);
    for (unsigned j = i + 1; j < dimension; ++j)
      id = rewriter.create<arith::MulIOp>(loc, id, getRangeCached(j));
    linearID =
        linearID ? rewriter.create<arith::AddIOp>(loc, linearID, id) : id;
  }
  rewriter.replaceOp(op, linearID);
}

template <typename Op>
class GetLinearIDPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::getTypeConverter;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  virtual Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ptrTy,
                      Value thisArg, int32_t index) const = 0;
  virtual Value getRange(OpBuilder &builder, Location loc, Type baseTy,
                         Type ptrTy, Value thisArg, int32_t index) const = 0;

  virtual Type getBaseType(Op op) const = 0;

public:
  virtual ~GetLinearIDPattern() = default;

  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto thisArg = adaptor.getOperands()[0];
    const auto elTy = getTypeConverter()->convertType(op.getType());
    const auto dimension = getDimensions(op.getOperand().getType());
    const auto convBaseTy = getTypeConverter()->convertType(getBaseType(op));
    getLinearIDRewriter(
        op, dimension,
        [&](OpBuilder &builder, Location loc, int32_t index) {
          return getID(builder, loc, convBaseTy, elTy, thisArg, index);
        },
        [&](OpBuilder &builder, Location loc, int32_t index) {
          return getRange(builder, loc, convBaseTy, elTy, thisArg, index);
        },
        rewriter);
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
    rewriter.replaceOpWithNewOp<GridOp>(op, op.getType(), ValueRange{});
  }
};

/// Base pattern for operations returning the result of a SYCL grid operation
/// \tparam GridOp.
template <typename Op, typename GridOp, typename Getter,
          typename = std::enable_if_t<
              llvm::is_one_of<Getter, SYCLIDGetOp, SYCLRangeGetOp>::value>>
class GridOpInitDimPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

private:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

public:
  LogicalResult match(Op) const override { return success(); }

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto indexType =
        ConvertOpToLLVMPattern<Op>::getTypeConverter()->getIndexType();
    const auto dimensions = getDimensions(op.getOperands()[0].getType());
    const auto arrayType = rewriter.getType<ArrayType>(
        dimensions, MemRefType::get(dimensions, indexType));
    const auto idType = rewriter.getType<IDType>(dimensions, arrayType);
    const auto loc = op.getLoc();
    Value idVal = rewriter.create<GridOp>(loc, idType);
    Value id =
        rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, idType));
    rewriter.create<memref::StoreOp>(loc, idVal, id);
    const auto offset = adaptor.getOperands()[1];
    Value res = rewriter.create<Getter>(op.getLoc(), indexType, id, offset);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, op.getType(), res);
  }
};

/// Base pattern for operations implementing a specific constructor for a type,
/// i.e., `sycl.ty.constructor`.
template <typename Op>
class BaseConstructorPattern : public ConvertOpToLLVMPattern<Op> {
protected:
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  virtual void initialize(Value alloca, Op op, OpAdaptor adaptor,
                          OpBuilder &builder) const = 0;

public:
  virtual ~BaseConstructorPattern() = default;

  void rewrite(Op op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    const LLVMTypeConverter *typeConverter =
        ConvertOpToLLVMPattern<Op>::getTypeConverter();
    Type ET = typeConverter->convertType(op.getType().getElementType());
    // The constructor value corresponds with the value defined by the alloca
    // operation.
    Value alloca = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, typeConverter->getPointerType(ET), ET,
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32));
    initialize(alloca, op, adaptor, rewriter);
  }
};

template <typename Op>
class CopyConstructorPattern : public BaseConstructorPattern<Op> {
public:
  using BaseConstructorPattern<Op>::BaseConstructorPattern;

  LogicalResult match(Op op) const final {
    OperandRange::type_range types = op.getOperands().getTypes();
    if (types.size() != 1)
      return failure();
    auto MT = dyn_cast<MemRefType>(types.front());
    Type resElTy = op.getType().getElementType();
    return success(MT && MT.getElementType() == resElTy);
  }

protected:
  /// Returns the size to be passed to `llvm.intr.memcpy`
  int64_t getSize(Op op) const;

  // We can simply copy the source array into the new one using
  // `llvm.intr.memcpy`.
  void initialize(Value alloca, Op op, typename Op::Adaptor adaptor,
                  OpBuilder &builder) const final {
    Location loc = op.getLoc();
    Value src = adaptor.getOperands()[0];
    Value len = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                 getSize(op));
    builder.create<LLVM::MemcpyOp>(loc, alloca, src, len, /*isVolatile*/ false);
  }
};

template <typename Op>
class BaseInitializeConstructorPattern : public BaseConstructorPattern<Op> {
public:
  using BaseConstructorPattern<Op>::BaseConstructorPattern;

protected:
  /// Returns a vector with the values the struct should be initialized with.
  virtual SmallVector<Value> getValues(Op op, typename Op::Adaptor adaptor,
                                       OpBuilder &builder) const = 0;
  /// Returns a pointer to the i-th element.
  virtual Value getElementPtr(Op op, Value alloca, int64_t i,
                              OpBuilder &builder) const = 0;
  /// Transforms the initial alloca value
  virtual Value transform([[maybe_unused]] Op op, Value alloca,
                          [[maybe_unused]] OpBuilder &builder) const {
    return alloca;
  }

  void initialize(Value alloca, Op op, typename Op::Adaptor adaptor,
                  OpBuilder &builder) const final {
    Location loc = op.getLoc();
    // Initial transformation
    alloca = transform(op, alloca, builder);
    // GEP + store loop
    SmallVector<Value> values = getValues(op, adaptor, builder);
    for (const auto &[i, value] : llvm::enumerate(values)) {
      Value ptr = getElementPtr(op, alloca, i, builder);
      builder.create<LLVM::StoreOp>(loc, value, ptr);
    }
  }
};

template <typename Op>
class BaseUnrealizedConversionCastInitializeConstructorPattern
    : public BaseInitializeConstructorPattern<Op> {
public:
  using BaseInitializeConstructorPattern<Op>::BaseInitializeConstructorPattern;

protected:
  Value transform(Op op, Value alloca, OpBuilder &builder) const final {
    return builder
        .create<UnrealizedConversionCastOp>(op.getLoc(), op.getType(), alloca)
        .getOutputs()[0];
  }
};

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Create a LLVM struct type with name \p name and the \p body.
/// In case of collision, generate a new name.
static Optional<Type> buildStructType(StringRef Name,
                                      llvm::ArrayRef<mlir::Type> Body,
                                      LLVMTypeConverter &Converter) {
  auto ConvertedTy =
      LLVM::LLVMStructType::getIdentified(&Converter.getContext(), Name);
  if (!ConvertedTy.isInitialized()) {
    if (failed(ConvertedTy.setBody(Body, /*isPacked=*/false)))
      return std::nullopt;
  } else if (Body != ConvertedTy.getBody()) {
    // If the name is already in use, create a new type.
    ConvertedTy = LLVM::LLVMStructType::getNewIdentified(
        &Converter.getContext(), Name, Body, /*isPacked=*/false);
  }

  return ConvertedTy;
}

/// Create a LLVM struct type with name \p name, and the converted \p body as
/// the body.
static Optional<Type> convertBodyType(StringRef name,
                                      llvm::ArrayRef<mlir::Type> body,
                                      LLVMTypeConverter &converter) {
  SmallVector<Type> convertedElemTypes;
  convertedElemTypes.reserve(body.size());
  if (failed(converter.convertTypes(body, convertedElemTypes)))
    return std::nullopt;
  return buildStructType(name, convertedElemTypes, converter);
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
  assert(isa<MemRefType>(type.getBody()[0]) &&
         "Expecting SYCL array body entry to be MemRefType");
  assert(converter.convertType(
             cast<MemRefType>(type.getBody()[0]).getElementType()) ==
             converter.getIndexType() &&
         "Expecting SYCL array body entry element type to be the index type");
  auto arrayTy =
      LLVM::LLVMArrayType::get(converter.getIndexType(), type.getDimension());
  return buildStructType("class.sycl::_V1::detail::array." +
                             std::to_string(type.getDimension()),
                         {arrayTy}, converter);
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

/// Converts SYCL half type to LLVM type.
static Optional<Type> convertHalfType(sycl::HalfType type,
                                      LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::half", type.getBody(), converter);
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

    assert(isa<MemRefType>(op.getSource().getType()) &&
           "The cast source type should be a memref type");
    assert(isa<MemRefType>(op.getResult().getType()) &&
           "The result source type should be a memref type");

    // Ensure the input and result types are legal.
    auto srcType = cast<MemRefType>(op.getSource().getType());
    auto resType = cast<MemRefType>(op.getResult().getType());

    if (!isConvertibleAndHasIdentityMaps(srcType) ||
        !isConvertibleAndHasIdentityMaps(resType))
      return failure();

    // Bitcasts with opaque pointers are just no-ops, so no need to create
    // them here.
    rewriter.replaceOp(op, opAdaptor.getSource());
    return success();
  }
};

class BarePtrCastPattern final : public ConvertOpToLLVMPattern<SYCLCastOp> {
public:
  using ConvertOpToLLVMPattern<SYCLCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto srcType = cast<MemRefType>(op.getSource().getType());
    const auto resType = cast<MemRefType>(op.getResult().getType());
    const auto convSrcType = typeConverter->convertType(srcType);
    const auto convResType = typeConverter->convertType(resType);

    // Ensure the input and result types are legal.
    if (!canBeLoweredToBarePtr(srcType) || !canBeLoweredToBarePtr(resType) ||
        !convSrcType || !convResType)
      return failure();

    // Bitcasts with opaque pointers are just no-ops, so no need to create
    // them here.
    rewriter.replaceOp(op, opAdaptor.getSource());
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
// AccessorGetPointerPattern - Convert `sycl.accessor.get_pointer` to LLVM.
//===----------------------------------------------------------------------===//

class AccessorGetPointerPattern
    : public ConvertOpToLLVMPattern<SYCLAccessorGetPointerOp>,
      public GetMemberPattern<AccessorGetPtr>,
      public GetMemberPattern<AccessorGetID, IDGetDim>,
      public GetMemberPattern<AccessorGetMemRange, RangeGetDim> {
public:
  using ConvertOpToLLVMPattern<
      SYCLAccessorGetPointerOp>::ConvertOpToLLVMPattern;

private:
  template <typename... Args> Value getID(Args &&...args) const {
    return GetMemberPattern<AccessorGetID, IDGetDim>::loadValue(
        std::forward<Args>(args)...);
  }
  template <typename... Args> Value getMemRange(Args &&...args) const {
    return GetMemberPattern<AccessorGetMemRange, RangeGetDim>::loadValue(
        std::forward<Args>(args)...);
  }

  Value getTotalOffset(OpBuilder &builder, Location loc, AccessorType accTy,
                       OpAdaptor opAdaptor) const {
    const auto acc = opAdaptor.getAcc();
    const auto resTy = getTypeConverter()->getIndexType();
    Value res = builder.create<arith::ConstantIntOp>(loc, 0, resTy);
    for (unsigned i = 0; i < accTy.getDimension(); ++i) {
      // Res = Res * Mem[I] + Id[I]
      const auto memI = getMemRange(builder, loc, accTy, resTy, acc, i);
      const auto idI = getID(builder, loc, accTy, resTy, acc, i);
      res = builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, res, memI), idI);
    }
    return res;
  }

public:
  LogicalResult
  matchAndRewrite(SYCLAccessorGetPointerOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto indexTy = getTypeConverter()->getIndexType();
    const Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, indexTy);
    const auto accTy =
        cast<AccessorType>(op.getAcc().getType().getElementType());
    const auto convAccTy = getTypeConverter()->convertType(accTy);
    Value index = rewriter.create<arith::SubIOp>(
        loc, zero,
        getTotalOffset(
            rewriter, loc,
            cast<AccessorType>(op.getAcc().getType().getElementType()),
            opAdaptor));
    const auto ptrTy = cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(op.getType()));
    Value ptr = GetMemberPattern<AccessorGetPtr>::loadValue(
        rewriter, loc, convAccTy, ptrTy, opAdaptor.getAcc());
    auto elemType =
        getTypeConverter()->convertType(op.getType().getElementType());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, ptrTy, elemType, ptr, index,
                                             /*inbounds*/ true);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AccessorSizePattern - Convert `sycl.accessor.get_range` to LLVM.
//===----------------------------------------------------------------------===//

class AccessorGetRangePattern
    : public LoadMemberPattern<SYCLAccessorGetRangeOp,
                               AccessorGetMAccessRange> {
public:
  using LoadMemberPattern<SYCLAccessorGetRangeOp,
                          AccessorGetMAccessRange>::LoadMemberPattern;
};

//===----------------------------------------------------------------------===//
// AccessorSizePattern - Convert `sycl.accessor.size` to LLVM.
//===----------------------------------------------------------------------===//

class AccessorSizePattern
    : public GetRangeSizePattern<SYCLAccessorSizeOp>,
      public GetMemberPattern<AccessorGetMAccessRange, RangeGetDim> {
public:
  using GetRangeSizePattern<SYCLAccessorSizeOp>::GetRangeSizePattern;

protected:
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<AccessorGetMAccessRange, RangeGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLAccessorSizeOp op) const final {
    return op.getAcc().getType().getElementType();
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
    return cast<AccessorType>(op.getAcc().getType().getElementType())
               .getAccessMode() == AccessMode::Atomic;
  }

  /// Whether the input accessor is 1-dimensional.
  static bool has1DAccessor(SYCLAccessorSubscriptOp op) {
    return cast<AccessorType>(op.getAcc().getType().getElementType())
               .getDimension() == 1;
  }

  /// Whether the input offset is an id.
  static bool hasIDOffsetType(SYCLAccessorSubscriptOp op) {
    return isa<MemRefType>(op.getIndex().getType());
  }

  Value getRef(OpBuilder &builder, Location loc, SYCLAccessorSubscriptOp orig,
               LLVM::LLVMPointerType ptrTy, Value acc, Value index) const {
    auto accTy = cast<AccessorType>(orig.getAcc().getType().getElementType());
    const auto addressSpace = targetToAddressSpace(accTy.getTargetMode());
    auto convAccTy = getTypeConverter()->convertType(accTy);

    const auto gepPtrTy =
        LLVM::LLVMPointerType::get(ptrTy.getContext(), addressSpace);
    const auto ptr = GetMemberPattern<AccessorGetPtr>::loadValue(
        builder, loc, convAccTy, gepPtrTy, acc);
    const Value gep = builder.create<LLVM::GEPOp>(
        loc, gepPtrTy, accTy.getType(), ptr, index, /*inbounds*/ true);
    return (ptrTy.getAddressSpace() == addressSpace)
               ? gep
               : builder.create<LLVM::AddrSpaceCastOp>(loc, ptrTy, gep);
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
                       Type idTy, OpAdaptor opAdaptor) const {
    const auto id = opAdaptor.getIndex();
    const auto acc = opAdaptor.getAcc();
    // size_t Res{0};
    const auto resTy = getTypeConverter()->getIndexType();
    Value res = builder.create<arith::ConstantIntOp>(loc, 0, resTy);
    auto convAccTy = getTypeConverter()->convertType(accTy);
    for (unsigned i = 0, dim = accTy.getDimension(); i < dim; ++i) {
      // Res = Res * Mem[I] + Id[I]
      const auto memI = getMemRange(builder, loc, convAccTy, resTy, acc, i);
      const auto idI = getID(builder, loc, idTy, resTy, id, i);
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
    return success(
        AccessorSubscriptPattern::hasIDOffsetType(op) &&
        getDimensions(op.getAcc().getType().getElementType()) ==
            getDimensions(
                cast<MemRefType>(op.getIndex().getType()).getElementType()));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto convIDTy = getTypeConverter()->convertType(
        cast<MemRefType>(op.getIndex().getType()).getElementType());
    const auto loc = op.getLoc();
    const auto ptrTy = cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(op.getType()));
    rewriter.replaceOp(
        op, AccessorSubscriptPattern::getRef(
                rewriter, loc, op, ptrTy, opAdaptor.getAcc(),
                getLinearIndex(
                    rewriter, loc,
                    cast<AccessorType>(op.getAcc().getType().getElementType()),
                    convIDTy, opAdaptor)));
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
    const auto ptrTy = cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(op.getType()));
    rewriter.replaceOp(op, AccessorSubscriptPattern::getRef(
                               rewriter, op.getLoc(), op, ptrTy,
                               opAdaptor.getAcc(), opAdaptor.getIndex()));
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
    const auto indexTy = getTypeConverter()->getIndexType();
    const Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, indexTy);
    for (unsigned i = 1, dim = getDimensions(op.getAcc().getType()) - 1;
         i < dim; ++i) {
      subscript = rewriter.create<LLVM::InsertValueOp>(
          loc, subscript, zero, ArrayRef<int64_t>{0, 0, 0, i});
    }
    // Insert original accessor
    auto accTy =
        getTypeConverter()->convertType(op.getAcc().getType().getElementType());
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, subscript,
        rewriter.create<LLVM::LoadOp>(loc, accTy, opAdaptor.getAcc()), 1);
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
    const auto convIDTy = getTypeConverter()->convertType(
        cast<MemRefType>(op.getIndex().getType()).getElementType());
    const auto loc = op.getLoc();
    const auto atomicTy = cast<AtomicType>(op.getType());
    auto *typeConverter = getTypeConverter();
    auto *context = op.getContext();
    const auto ptrTy =
        cast<LLVM::LLVMPointerType>(typeConverter->convertType(MemRefType::get(
            ShapedType::kDynamic, atomicTy.getDataType(), AffineMap{},
            AccessAddrSpaceAttr::get(context, atomicTy.getAddrSpace()))));
    const Value undef = rewriter.create<LLVM::UndefOp>(
        loc, typeConverter->convertType(atomicTy));
    const auto ptr = AccessorSubscriptPattern::getRef(
        rewriter, loc, op, ptrTy, opAdaptor.getAcc(),
        getLinearIndex(
            rewriter, loc,
            cast<AccessorType>(op.getAcc().getType().getElementType()),
            convIDTy, opAdaptor));
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
    return success(op.getType().isIntOrIndex());
  }
};

class RangeGetRefPattern
    : public GetRefToMemberDimPattern<SYCLRangeGetOp, RangeGetDim> {
public:
  using GetRefToMemberDimPattern<SYCLRangeGetOp,
                                 RangeGetDim>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(isa<MemRefType>(op.getType()));
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeSizePattern - Convert `sycl.range.size` to LLVM.
//===----------------------------------------------------------------------===//

class RangeSizePattern : public GetRangeSizePattern<SYCLRangeSizeOp>,
                         public GetMemberPattern<RangeGetDim> {
public:
  using GetRangeSizePattern<SYCLRangeSizeOp>::GetRangeSizePattern;

  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return loadValue(builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLRangeSizeOp op) const final {
    return op.getRange().getType().getElementType();
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
    const auto convNDRangeTy =
        getTypeConverter()->convertType(op.getND().getType().getElementType());
    const auto rangeTy = op.getType();
    const auto convRangeTy = getTypeConverter()->convertType(rangeTy);
    const auto indexTy = getTypeConverter()->getIndexType();
    const auto allocaTy = getTypeConverter()->getPointerType(convRangeTy);
    const auto baseTy = convRangeTy;
    Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc, allocaTy, convRangeTy,
        rewriter.create<arith::ConstantIntOp>(loc, 1, indexTy),
        /*alignment*/ 0);
    for (int32_t i = 0, dim = rangeTy.getDimension(); i < dim; ++i) {
      const auto lhs =
          GetMemberPattern<NDRangeGetGlobalRange, RangeGetDim>::loadValue(
              rewriter, loc, convNDRangeTy, indexTy, nd, i);
      const auto rhs =
          GetMemberPattern<NDRangeGetLocalRange, RangeGetDim>::loadValue(
              rewriter, loc, convNDRangeTy, indexTy, nd, i);
      const Value val = rewriter.create<arith::DivUIOp>(loc, lhs, rhs);
      auto ptr = GetMemberPattern<RangeGetDim>::getRef(rewriter, loc, baseTy,
                                                       alloca, std::nullopt, i);
      rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, convRangeTy, alloca);
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
    return success(op.getNumOperands() > 1 && op.getType().isIntOrIndex());
  }
};

/// Converts SYCLIDGet with a reference return type to LLVM
class IDGetRefPattern : public GetRefToMemberDimPattern<SYCLIDGetOp, IDGetDim> {
public:
  using GetRefToMemberDimPattern<SYCLIDGetOp,
                                 IDGetDim>::GetRefToMemberDimPattern;

  LogicalResult match(SYCLIDGetOp op) const final {
    return success(isa<MemRefType>(op.getType()));
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
    return success(isa<IDType>(op.getRes().getType()));
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
                   isa<IntegerType>(op.getRes().getType()));
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
    return success(isa<RangeType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
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
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<ItemGetRange, RangeGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLItemGetLinearIDOp op) const final {
    return op.getItem().getType().getElementType();
  }

public:
  using GetLinearIDPattern<SYCLItemGetLinearIDOp>::GetLinearIDPattern;
};

/// Converts SYCLItemGetLinearIDOp with no offset item to LLVM
class ItemNoOffsetGetLinearIDPattern : public ItemGetLinearIDPattern {
public:
  using ItemGetLinearIDPattern::ItemGetLinearIDPattern;

  LogicalResult match(SYCLItemGetLinearIDOp op) const final {
    return success(!cast<ItemType>(op.getItem().getType().getElementType())
                        .getWithOffset());
  }

  Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ty,
              Value thisArg, int32_t index) const final {
    return GetMemberPattern<ItemGetID, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }
};

/// Converts SYCLItemGetLinearIDOp with no offset item to LLVM
class ItemOffsetGetLinearIDPattern
    : public ItemGetLinearIDPattern,
      public GetMemberPattern<ItemGetOffset, IDGetDim> {
protected:
  Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ty,
              Value thisArg, int32_t index) const final {
    const auto id = GetMemberPattern<ItemGetID, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
    const auto offset = GetMemberPattern<ItemGetOffset, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
    return builder.create<arith::SubIOp>(loc, id, offset);
  }

public:
  using ItemGetLinearIDPattern::ItemGetLinearIDPattern;

  LogicalResult match(SYCLItemGetLinearIDOp op) const final {
    return success(cast<ItemType>(op.getItem().getType().getElementType())
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
    return success(isa<IDType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetGlobalRange - Converts `sycl.nd_item.get_global_range` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetGlobalRangeOp with a Range return type to LLVM.
class NDItemGetGlobalRangePattern
    : public LoadMemberPattern<SYCLNDItemGetGlobalRangeOp, NDItemGlobalItem,
                               ItemGetRange> {
public:
  using LoadMemberPattern<SYCLNDItemGetGlobalRangeOp, NDItemGlobalItem,
                          ItemGetRange>::LoadMemberPattern;

  LogicalResult match(SYCLNDItemGetGlobalRangeOp op) const final {
    return success(isa<RangeType>(op.getRes().getType()));
  }
};

/// Converts SYCLNDItemGetGlobalRangeOp with a scalar return type to LLVM.
class NDItemGetGlobalRangeDimPattern
    : public LoadMemberDimPattern<SYCLNDItemGetGlobalRangeOp, NDItemGlobalItem,
                                  ItemGetRange, RangeGetDim> {
public:
  using LoadMemberDimPattern<SYCLNDItemGetGlobalRangeOp, NDItemGlobalItem,
                             ItemGetRange, RangeGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLNDItemGetGlobalRangeOp op) const final {
    return success(isa<IntegerType>(op.getRes().getType()));
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
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<NDItemGlobalItem, ItemGetRange,
                            RangeGetDim>::loadValue(builder, loc, baseTy, ty,
                                                    thisArg, index);
  }

  Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ty,
              Value thisArg, int32_t index) const final {
    return GetMemberPattern<NDItemGlobalItem, ItemGetID, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLNDItemGetGlobalLinearIDOp op) const final {
    return op.getNDItem().getType().getElementType();
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
    return success(isa<IDType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
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
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<NDItemLocalItem, ItemGetRange,
                            RangeGetDim>::loadValue(builder, loc, baseTy, ty,
                                                    thisArg, index);
  }

  Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ty,
              Value thisArg, int32_t index) const final {
    return GetMemberPattern<NDItemLocalItem, ItemGetID, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLNDItemGetLocalLinearIDOp op) const final {
    return op.getNDItem().getType().getElementType();
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
    return success(isa<GroupType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
  }
};

//===----------------------------------------------------------------------===//
// NDItemGetGroupLinearID - Converts `sycl.nd_item.get_group_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLNDItemGetGroupLinearIDOp to LLVM.
class NDItemGetGroupLinearIDPattern
    : public ConvertOpToLLVMPattern<SYCLNDItemGetGroupLinearIDOp>,
      public GetMemberPattern<NDItemGroup> {
public:
  using ConvertOpToLLVMPattern<
      SYCLNDItemGetGroupLinearIDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNDItemGetGroupLinearIDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    // TODO: We query the group type from the body. Not ideal. Drop this when
    // body is dropped and we can create group type more easily.
    const auto indices = GetMemberPattern<NDItemGroup>::getIndices();
    assert(indices.size() == 1 && "Expecting a single index");
    auto ndItemTy = cast<NdItemType>(op.getNDItem().getType().getElementType());
    const auto groupTy = ndItemTy.getBody()[indices[0]];
    Type convNDItemTy = getTypeConverter()->convertType(ndItemTy);
    auto group = GetMemberPattern<NDItemGroup>::getRef(
        rewriter, loc, convNDItemTy, adaptor.getNDItem(), std::nullopt);
    const auto thisTy = MemRefType::get(ShapedType::kDynamic, groupTy);
    // We have the already converted group, but, in order to not replicate
    // `sycl.group.get_group_linear_id` conversion to LLVM, we just reuse that
    // using `builtin.unrealized_conversion_cast` to convert the pointer into a
    // memref to sycl type.
    const auto syclGroup =
        rewriter.create<UnrealizedConversionCastOp>(loc, thisTy, group)
            .getResult(0);
    rewriter.replaceOpWithNewOp<SYCLGroupGetGroupLinearIDOp>(
        op, rewriter.getI64Type(), syclGroup);
    return success();
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
    return success(isa<RangeType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
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
    return success(isa<RangeType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
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
    const auto convNDItemType = getTypeConverter()->convertType(
        op.getNDItem().getType().getElementType());

    const auto ndrTy = getTypeConverter()->convertType(op.getType());
    const auto allocaTy = getTypeConverter()->getPointerType(ndrTy);
    const Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc, allocaTy, ndrTy,
        rewriter.create<arith::ConstantIntOp>(loc, 1, 32));

    const auto rangeTy = cast<LLVM::LLVMStructType>(ndrTy).getBody()[0];
    const auto idTy = cast<LLVM::LLVMStructType>(ndrTy).getBody()[2];

    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemGlobalItem, ItemGetRange>::loadValue(
            rewriter, loc, convNDItemType, rangeTy, ndItem),
        GetMemberPattern<NDRangeGetGlobalRange>::getRef(rewriter, loc, ndrTy,
                                                        alloca, std::nullopt));
    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemLocalItem, ItemGetRange>::loadValue(
            rewriter, loc, convNDItemType, rangeTy, ndItem),
        GetMemberPattern<NDRangeGetLocalRange>::getRef(rewriter, loc, ndrTy,
                                                       alloca, std::nullopt));
    rewriter.create<LLVM::StoreOp>(
        loc,
        GetMemberPattern<NDItemGlobalItem, ItemGetOffset>::loadValue(
            rewriter, loc, convNDItemType, idTy, ndItem),
        GetMemberPattern<NDRangeGetOffset>::getRef(rewriter, loc, ndrTy, alloca,
                                                   std::nullopt));

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, ndrTy, alloca);
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
    return success(isa<IDType>(op.getRes().getType()));
  }
};

/// Converts SYCLGroupGetGroupID with a scalar return type to LLVM
class GroupGetGroupIDDimPattern
    : public LoadMemberDimPattern<SYCLGroupGetGroupIDOp, GroupGetID, IDGetDim> {
public:
  using LoadMemberDimPattern<SYCLGroupGetGroupIDOp, GroupGetID,
                             IDGetDim>::LoadMemberDimPattern;

  LogicalResult match(SYCLGroupGetGroupIDOp op) const final {
    return success(isa<IntegerType>(op.getRes().getType()));
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
    return success(isa<IDType>(op.getRes().getType()));
  }
};

/// Converts SYCLGroupGetLocalID with a scalar return type to LLVM
class GroupGetLocalIDDimPattern
    : public GridOpInitDimPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp,
                                  SYCLIDGetOp> {
public:
  using GridOpInitDimPattern<SYCLGroupGetLocalIDOp, SYCLLocalIDOp,
                             SYCLIDGetOp>::GridOpInitDimPattern;

  LogicalResult match(SYCLGroupGetLocalIDOp op) const final {
    return success(isa<IntegerType>(op.getRes().getType()));
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
    return success(isa<RangeType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
  }
};

//===----------------------------------------------------------------------===//
// IDConstructor - Convert `sycl.id.constructor` to LLVM.
//===----------------------------------------------------------------------===//

// () -> memref<1x!sycl_id_N_>
class IDDefaultConstructorPattern
    : public BaseConstructorPattern<SYCLIDConstructorOp> {
public:
  using BaseConstructorPattern<SYCLIDConstructorOp>::BaseConstructorPattern;

  LogicalResult match(SYCLIDConstructorOp op) const final {
    return success(op.getOperands().empty());
  }

protected:
  // We can simply zero-initialize the array using `llvm.intr.memset`.
  void initialize(Value alloca, SYCLIDConstructorOp op, OpAdaptor adaptor,
                  OpBuilder &builder) const final {
    Location loc = op.getLoc();
    Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI8Type(), 0);
    unsigned dimension = getDimensions(op.getId().getType());
    unsigned indexWidth =
        BaseConstructorPattern<SYCLIDConstructorOp>::getTypeConverter()
            ->getIndexTypeBitwidth();
    Value len = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                 dimension * indexWidth / 8);
    builder.create<LLVM::MemsetOp>(loc, alloca, zero, len,
                                   /*isVolatile*/ false);
  }
};

// (!memref<?x!sycl_id_N_) -> !memref<?x!sycl_id_N_>
using IDCopyConstructorPattern = CopyConstructorPattern<SYCLIDConstructorOp>;

template <>
int64_t IDCopyConstructorPattern::getSize(SYCLIDConstructorOp op) const {
  unsigned dimension = getDimensions(op.getId().getType());
  unsigned indexWidth =
      BaseConstructorPattern<SYCLIDConstructorOp>::getTypeConverter()
          ->getIndexTypeBitwidth();
  return dimension * indexWidth / 8;
}

class IDConstructorBase
    : public BaseUnrealizedConversionCastInitializeConstructorPattern<
          SYCLIDConstructorOp> {
public:
  using BaseUnrealizedConversionCastInitializeConstructorPattern<
      SYCLIDConstructorOp>::
      BaseUnrealizedConversionCastInitializeConstructorPattern;

protected:
  /// Returns a pointer to the i-th element.
  virtual Value getElementPtr(SYCLIDConstructorOp op, Value alloca, int64_t i,
                              OpBuilder &builder) const final {
    Location loc = op.getLoc();
    auto MT =
        MemRefType::get(/*shape=*/ShapedType::kDynamic, builder.getIndexType());
    Value offset =
        builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), i);
    Value ptr = builder.create<SYCLIDGetOp>(loc, MT, alloca, offset);
    /// Avoid inserting operations from the `memref` dialect by inserting this
    /// `unrealized_conversion_cast`:
    Type PT = ConvertOpToLLVMPattern<SYCLIDConstructorOp>::getTypeConverter()
                  ->convertType(MT);
    return builder.create<UnrealizedConversionCastOp>(loc, PT, ptr)
        .getOutputs()[0];
  }
};

// ({index}N) -> memref<1x!sycl_id_N_>
class IDIndexConstructorPattern : public IDConstructorBase {
public:
  using IDConstructorBase::IDConstructorBase;

  LogicalResult match(SYCLIDConstructorOp op) const final {
    OperandRange::type_range types = op.getOperands().getTypes();
    return success(!types.empty() && llvm::all_of(types, [](Type ty) {
      return isa<IndexType>(ty);
    }));
  }

protected:
  SmallVector<Value> getValues(SYCLIDConstructorOp op, OpAdaptor adaptor,
                               OpBuilder &builder) const final {
    Location loc = op.getLoc();
    SmallVector<Value> values;
    Type indexTy =
        ConvertOpToLLVMPattern<SYCLIDConstructorOp>::getTypeConverter()
            ->getIndexType();
    llvm::transform(
        adaptor.getOperands(), std::back_inserter(values), [&](Value val) {
          return builder.create<UnrealizedConversionCastOp>(loc, indexTy, val)
              .getOutputs()[0];
        });
    return values;
  }
};

/// Base class for `sycl.id.constructor` signatures receiving a pointer to an
/// object.
template <typename SrcTy, typename OpTy>
class IDStructConstructorPattern : public IDConstructorBase {
public:
  using IDConstructorBase::IDConstructorBase;

  LogicalResult match(SYCLIDConstructorOp op) const final {
    OperandRange::type_range types = op.getOperands().getTypes();
    if (types.size() != 1)
      return failure();
    auto MT = dyn_cast<MemRefType>(types.front());
    return success(MT && isa<SrcTy>(MT.getElementType()));
  }

protected:
  SmallVector<Value> getValues(SYCLIDConstructorOp op, OpAdaptor adaptor,
                               OpBuilder &builder) const final {
    Location loc = op.getLoc();
    SmallVector<Value> values;
    Type i32Ty = builder.getI32Type();
    Type indexTy =
        ConvertOpToLLVMPattern<SYCLIDConstructorOp>::getTypeConverter()
            ->getIndexType();
    auto srcTy = cast<SrcTy>(
        cast<MemRefType>(op.getOperands()[0].getType()).getElementType());
    Value orig =
        builder
            .create<UnrealizedConversionCastOp>(
                loc, op.getOperands()[0].getType(), adaptor.getOperands()[0])
            .getOutputs()[0];
    for (unsigned i = 0, dimension = srcTy.getDimension(); i < dimension; ++i) {
      Value offset = builder.create<LLVM::ConstantOp>(loc, i32Ty, i);
      Value val = builder.create<OpTy>(loc, indexTy, orig, offset);
      values.push_back(val);
    }
    return values;
  }
};

// (!memref<?x!sycl_item_N_) -> !memref<?x!sycl_item_N_>
class IDItemConstructorPattern
    : public IDStructConstructorPattern<ItemType, SYCLItemGetIDOp> {
public:
  using IDStructConstructorPattern<ItemType,
                                   SYCLItemGetIDOp>::IDStructConstructorPattern;
};

// (!memref<?x!sycl_range_N_) -> !memref<?x!sycl_range_N_>
class IDRangeConstructorPattern
    : public IDStructConstructorPattern<RangeType, SYCLRangeGetOp> {
public:
  using IDStructConstructorPattern<RangeType,
                                   SYCLRangeGetOp>::IDStructConstructorPattern;
};

//===----------------------------------------------------------------------===//
// RangeConstructor - Convert `sycl.range.constructor` to LLVM.
//===----------------------------------------------------------------------===//

// (!memref<?x!sycl_range_N_) -> !memref<?x!sycl_range_N_>
using RangeCopyConstructorPattern =
    CopyConstructorPattern<SYCLRangeConstructorOp>;

template <>
int64_t RangeCopyConstructorPattern::getSize(SYCLRangeConstructorOp op) const {
  unsigned dimension = getDimensions(op.getRange().getType());
  unsigned indexWidth =
      BaseConstructorPattern<SYCLRangeConstructorOp>::getTypeConverter()
          ->getIndexTypeBitwidth();
  return dimension * indexWidth / 8;
}

class RangeConstructorBase
    : public BaseUnrealizedConversionCastInitializeConstructorPattern<
          SYCLRangeConstructorOp> {
public:
  using BaseUnrealizedConversionCastInitializeConstructorPattern<
      SYCLRangeConstructorOp>::
      BaseUnrealizedConversionCastInitializeConstructorPattern;

protected:
  /// Returns a pointer to the i-th element.
  virtual Value getElementPtr(SYCLRangeConstructorOp op, Value alloca,
                              int64_t i, OpBuilder &builder) const final {
    Location loc = op.getLoc();
    auto MT =
        MemRefType::get(/*shape=*/ShapedType::kDynamic, builder.getIndexType());
    Value offset =
        builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), i);
    Value ptr = builder.create<SYCLRangeGetOp>(loc, MT, alloca, offset);
    /// Avoid inserting operations from the `memref` dialect by inserting this
    /// `unrealized_conversion_cast`:
    Type PT = ConvertOpToLLVMPattern<SYCLRangeConstructorOp>::getTypeConverter()
                  ->convertType(MT);
    return builder.create<UnrealizedConversionCastOp>(loc, PT, ptr)
        .getOutputs()[0];
  }
};

// ({index}N) -> memref<1x!sycl_id_N_>
class RangeIndexConstructorPattern : public RangeConstructorBase {
public:
  using RangeConstructorBase::RangeConstructorBase;

  LogicalResult match(SYCLRangeConstructorOp op) const final {
    OperandRange::type_range types = op.getOperands().getTypes();
    return success(!types.empty() && llvm::all_of(types, [](Type ty) {
      return isa<IndexType>(ty);
    }));
  }

protected:
  SmallVector<Value> getValues(SYCLRangeConstructorOp op, OpAdaptor adaptor,
                               OpBuilder &builder) const final {
    Location loc = op.getLoc();
    SmallVector<Value> values;
    Type indexTy =
        ConvertOpToLLVMPattern<SYCLRangeConstructorOp>::getTypeConverter()
            ->getIndexType();
    llvm::transform(
        adaptor.getOperands(), std::back_inserter(values), [&](Value val) {
          return builder.create<UnrealizedConversionCastOp>(loc, indexTy, val)
              .getOutputs()[0];
        });
    return values;
  }
};

//===----------------------------------------------------------------------===//
// NDRangeConstructor - Convert `sycl.nd_range.constructor` to LLVM.
//===----------------------------------------------------------------------===//

// (!memref<?x!sycl_range_N_) -> !memref<?x!sycl_range_N_>
using NDRangeCopyConstructorPattern =
    CopyConstructorPattern<SYCLNDRangeConstructorOp>;

template <>
int64_t
NDRangeCopyConstructorPattern::getSize(SYCLNDRangeConstructorOp op) const {
  unsigned dimension = getDimensions(op.getNDRange().getType());
  unsigned indexWidth =
      BaseConstructorPattern<SYCLNDRangeConstructorOp>::getTypeConverter()
          ->getIndexTypeBitwidth();
  // 2 ranges + 1 id
  return 3 * dimension * indexWidth / 8;
}

class NDRangeConstructorBase
    : public BaseConstructorPattern<SYCLNDRangeConstructorOp>,
      public GetMemberPattern<NDRangeGetGlobalRange>,
      public GetMemberPattern<NDRangeGetLocalRange>,
      public GetMemberPattern<NDRangeGetOffset> {
public:
  using BaseConstructorPattern<
      SYCLNDRangeConstructorOp>::BaseConstructorPattern;

  ~NDRangeConstructorBase() = default;

protected:
  /// Returns a pointer to the i-th element.
  void initialize(Value alloca, SYCLNDRangeConstructorOp op, OpAdaptor adaptor,
                  OpBuilder &builder) const final {
    Location loc = op.getLoc();
    const LLVMTypeConverter *typeConverter =
        ConvertOpToLLVMPattern<SYCLNDRangeConstructorOp>::getTypeConverter();
    unsigned dimensions = getDimensions(op.getNDRange().getType());
    auto ndrTy = cast<LLVM::LLVMStructType>(
        typeConverter->convertType(op.getNDRange().getType().getElementType()));

    // Always copy-initialize global size
    Value globalSizePtr = GetMemberPattern<NDRangeGetGlobalRange>::getRef(
        builder, loc, ndrTy, alloca, std::nullopt);
    memcpy(builder, loc, globalSizePtr, adaptor.getOperands()[0], dimensions);

    // Always copy-initialize local size
    Value localSizePtr = GetMemberPattern<NDRangeGetLocalRange>::getRef(
        builder, loc, ndrTy, alloca, std::nullopt);
    memcpy(builder, loc, localSizePtr, adaptor.getOperands()[1], dimensions);

    // Offset initialization will depend on the constructor being handled
    Value offsetPtr = GetMemberPattern<NDRangeGetOffset>::getRef(
        builder, loc, ndrTy, alloca, std::nullopt);
    initializeOffset(builder, loc, offsetPtr, adaptor, dimensions);
  }

  void memcpy(OpBuilder &builder, Location loc, Value dst, Value src,
              unsigned dimensions) const {
    Value len = getFieldsSize(builder, loc, dimensions);
    builder.create<LLVM::MemcpyOp>(loc, dst, src, len, /*isVolatile*/ false);
  }

  /// Returns the size of this struct's fields498
  Value getFieldsSize(OpBuilder &builder, Location loc,
                      unsigned dimensions) const {
    unsigned indexWidth =
        ConvertOpToLLVMPattern<SYCLNDRangeConstructorOp>::getTypeConverter()
            ->getIndexTypeBitwidth();
    return builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                            dimensions * indexWidth / 8);
  }

  virtual void initializeOffset(OpBuilder &builder, Location loc, Value dst,
                                OpAdaptor adaptor,
                                unsigned dimensions) const = 0;
};

class NDRangeNoOffsetConstructor : public NDRangeConstructorBase {
public:
  using NDRangeConstructorBase::NDRangeConstructorBase;

  LogicalResult match(SYCLNDRangeConstructorOp op) const final {
    return success(op.getNumOperands() == 2);
  }

protected:
  void initializeOffset(OpBuilder &builder, Location loc, Value dst,
                        [[maybe_unused]] OpAdaptor adaptor,
                        unsigned dimensions) const final {
    Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI8Type(), 0);
    Value len = getFieldsSize(builder, loc, dimensions);
    builder.create<LLVM::MemsetOp>(loc, dst, zero, len, /*isVolatile*/ false);
  }
};

class NDRangeConstructorWithOffset : public NDRangeConstructorBase {
public:
  using NDRangeConstructorBase::NDRangeConstructorBase;

  LogicalResult match(SYCLNDRangeConstructorOp op) const final {
    return success(op.getNumOperands() == 3);
  }

protected:
  void initializeOffset(OpBuilder &builder, Location loc, Value dst,
                        OpAdaptor adaptor, unsigned dimensions) const final {
    memcpy(builder, loc, dst, adaptor.getOperands()[2], dimensions);
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
    return success(isa<RangeType>(op.getRes().getType()));
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
    return success(isa<IntegerType>(op.getRes().getType()));
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
  Value getID(OpBuilder &builder, Location loc, Type baseTy, Type ty,
              Value thisArg, int32_t index) const final {
    return GetMemberPattern<GroupGetID, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<GroupGetGroupRange, IDGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLGroupGetGroupLinearIDOp op) const final {
    return op.getGroup().getType().getElementType();
  }

public:
  using GetLinearIDPattern<SYCLGroupGetGroupLinearIDOp>::GetLinearIDPattern;
};

//===----------------------------------------------------------------------===//
// GroupGetLocalLinearID - Converts `sycl.group.get_local_linear_id` to LLVM.
//===----------------------------------------------------------------------===//

/// Converts SYCLGroupGetLocalLinearIDOp to LLVM
class GroupGetLocalLinearIDPattern
    : public ConvertOpToLLVMPattern<SYCLGroupGetLocalLinearIDOp>,
      public GetMemberPattern<GroupGetLocalRange, RangeGetDim> {
public:
  using ConvertOpToLLVMPattern<
      SYCLGroupGetLocalLinearIDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLGroupGetLocalLinearIDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto groupTy =
        cast<GroupType>(op.getGroup().getType().getElementType());
    const auto dimension = groupTy.getDimension();
    auto indexType = getTypeConverter()->getIndexType();
    const auto arrayType = rewriter.getType<ArrayType>(
        dimension, MemRefType::get(dimension, indexType));
    const auto idType = rewriter.getType<IDType>(dimension, arrayType);
    // Obtain the local ID (already mirrored)
    Value localIDVal = rewriter.create<SYCLLocalIDOp>(loc, idType);
    // Store it in a memref to access its elements
    Value localID =
        rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, idType));
    rewriter.create<memref::StoreOp>(loc, localIDVal, localID);
    const auto group = adaptor.getGroup();
    const auto convGroupTy =
        typeConverter->convertType(op.getGroup().getType().getElementType());
    // The local linear ID is calculated from the local ID and the group's local
    // range.
    getLinearIDRewriter(
        op, dimension,
        [&](OpBuilder &builder, Location loc, int32_t index) -> Value {
          return builder.create<SYCLIDGetOp>(
              loc, indexType, localID,
              builder.create<arith::ConstantIntOp>(loc, index,
                                                   /*bitwidth=*/32));
        },
        [&](OpBuilder &builder, Location loc, int32_t index) -> Value {
          return GetMemberPattern<GroupGetLocalRange, RangeGetDim>::loadValue(
              builder, loc, convGroupTy, indexType, group, index);
        },
        rewriter);
    return success();
  }
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
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<GroupGetGroupRange, RangeGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLGroupGetGroupLinearRangeOp op) const final {
    return op.getGroup().getType().getElementType();
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
  Value getRange(OpBuilder &builder, Location loc, Type baseTy, Type ty,
                 Value thisArg, int32_t index) const final {
    return GetMemberPattern<GroupGetLocalRange, RangeGetDim>::loadValue(
        builder, loc, baseTy, ty, thisArg, index);
  }

  Type getBaseType(SYCLGroupGetLocalLinearRangeOp op) const final {
    return op.getGroup().getType().getElementType();
  }
};

//===----------------------------------------------------------------------===//
// Wrap/UnwrapPattern - Converts `sycl.mlir.wrap` and `.unwrap` to LLVM.
//===----------------------------------------------------------------------===//

struct WrapPattern : public ConvertOpToLLVMPattern<SYCLWrapOp> {
  using ConvertOpToLLVMPattern<SYCLWrapOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLWrapOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type structType = typeConverter->convertType(op.getType());
    Value undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), structType);
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, undef, opAdaptor.getSource(), rewriter.getDenseI64ArrayAttr(0));
    return success();
  }
};

struct UnwrapPattern : public ConvertOpToLLVMPattern<SYCLUnwrapOp> {
  using ConvertOpToLLVMPattern<SYCLUnwrapOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLUnwrapOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, opAdaptor.getSource(), rewriter.getDenseI64ArrayAttr(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::dpcpp::populateSYCLToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType, AccessAddrSpaceAttr addrSpace)
          -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(
            IntegerType::get(addrSpace.getContext(), 64),
            // SPIR-V mapping. Will need to change for other targets.
            static_cast<int64_t>(addrSpace.getValue()));
      });

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
  typeConverter.addConversion([&](sycl::HalfType type) {
    return convertHalfType(type, typeConverter);
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

static void
populateSYCLToLLVMSPIRConversionPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  assert(typeConverter.getOptions().useBarePtrCallConv &&
         "These patterns only work with bare pointer calling convention");

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  patterns.add<BarePtrCastPattern>(typeConverter, /*benefit*/ 2);
  patterns.add<
      AccessorGetPointerPattern, AccessorGetRangePattern, AccessorSizePattern,
      AddZeroArgPattern<SYCLIDGetOp>, AddZeroArgPattern<SYCLItemGetIDOp>,
      AtomicSubscriptIDOffset, BarePtrAddrSpaceCastPattern,
      GroupGetGroupIDPattern, GroupGetGroupLinearRangePattern,
      GroupGetGroupRangeDimPattern, GroupGetLocalIDPattern,
      GroupGetLocalLinearRangePattern, GroupGetLocalRangeDimPattern,
      IDCopyConstructorPattern, IDDefaultConstructorPattern,
      IDIndexConstructorPattern, IDItemConstructorPattern,
      IDRangeConstructorPattern, IDGetPattern, IDGetRefPattern,
      ItemGetIDDimPattern, ItemGetRangeDimPattern, ItemGetRangePattern,
      NDItemGetGlobalIDDimPattern, NDItemGetGlobalIDPattern,
      NDItemGetGlobalRangeDimPattern, NDItemGetGlobalRangePattern,
      NDItemGetGroupPattern, NDItemGetGroupRangeDimPattern,
      NDItemGetLocalIDDimPattern, NDItemGetLocalLinearIDPattern,
      NDItemGetNDRange, NDRangeGetGroupRangePattern,
      NDRangeGetLocalRangePattern, NDRangeCopyConstructorPattern,
      NDRangeNoOffsetConstructor, NDRangeConstructorWithOffset,
      RangeGetRefPattern, RangeSizePattern, SubscriptScalarOffsetND,
      GroupGetGroupIDDimPattern, GroupGetGroupLinearIDPattern,
      GroupGetGroupRangePattern, GroupGetLocalIDDimPattern,
      GroupGetLocalLinearIDPattern, GroupGetLocalRangePattern,
      GroupGetMaxLocalRangePattern, ItemGetIDPattern,
      ItemNoOffsetGetLinearIDPattern, ItemOffsetGetLinearIDPattern,
      NDItemGetGlobalLinearIDPattern, NDItemGetGroupDimPattern,
      NDItemGetGroupLinearIDPattern, NDItemGetGroupRangePattern,
      NDItemGetLocalIDPattern, NDItemGetLocalRangeDimPattern,
      NDItemGetLocalRangePattern, NDRangeGetGlobalRangePattern, RangeGetPattern,
      RangeCopyConstructorPattern, RangeIndexConstructorPattern,
      SubscriptIDOffset, SubscriptScalarOffset1D, UnwrapPattern, WrapPattern>(
      typeConverter);
  patterns.add<ConstructorPattern>(typeConverter);
}

void mlir::dpcpp::populateSYCLToLLVMConversionPatterns(
    sycl::LoweringTarget target, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  switch (target) {
  case sycl::LoweringTarget::SPIR:
    populateSYCLToLLVMSPIRConversionPatterns(typeConverter, patterns);
    break;
  }
}
