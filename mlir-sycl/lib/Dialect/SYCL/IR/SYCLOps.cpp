//===--- SYCLOps.cpp ------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sycl;

static LogicalResult checkNumIndices(Operation *op, unsigned numArgs,
                                     unsigned dimensions) {
  return numArgs == dimensions
             ? success()
             : op->emitOpError(
                   "expects to be passed the same number of 'index' numbers as "
                   "the number of dimensions of the input: ")
                   << numArgs << " vs " << dimensions;
}

static LogicalResult emitBadSignatureError(Operation *op) {
  return op->emitOpError(
      "expects a different signature. Check documentation for details");
}

static LogicalResult checkSameDimensions(Operation *op, Type thisType,
                                         Type inputType) {
  unsigned thisDims = getDimensions(thisType);
  unsigned inputDims = getDimensions(inputType);
  return thisDims == inputDims
             ? success()
             : op->emitOpError("expects input and output to have the same "
                               "number of dimensions: ")
                   << inputDims << " vs " << thisDims;
}

template <typename Ty> static Ty getMemRefElementType(Type ty) {
  auto mt = dyn_cast<MemRefType>(ty);
  if (!mt)
    return {};
  return dyn_cast<Ty>(mt.getElementType());
}

/// Test \p id is default constructed and it is not modified.
static bool isDefaultConstructedID(Value id) {
  // Check it is default constructed
  auto constructor = id.getDefiningOp<sycl::SYCLIDConstructorOp>();
  if (!constructor || !constructor.getArgs().empty())
    return false;

  // Conservatively check it has not been modified after construction
  return llvm::all_of(id.getUsers(), [=](Operation *op) {
    if (auto get = dyn_cast<SYCLIDGetOp>(op))
      // sycl.id.get should only be used when returning a scalar
      return !isa<MemRefType>(get.getRes().getType());
    return isa<SYCLNDRangeConstructorOp, SYCLIDConstructorOp, SYCLConstructorOp,
               affine::AffineLoadOp, memref::LoadOp>(op);
  });
}

namespace {
/// Transform:
/// ```
/// %off = sycl.id.constructor() : () -> memref<?x!sycl_id_X_>
/// sycl.nd_range.constructor(%gs, %ls, %off)
///   : (memref<?x!sycl_range_X_>,
///      memref<?x!sycl_range_X_>,
///      memref<?x!sycl_id_X_>)
///   -> memref<?x!sycl_nd_range_X_>
/// ```
/// To:
/// ```
/// sycl.nd_range.constructor(%gs, %ls)
///   : (memref<?x!sycl_range_X_>, memref<?x!sycl_range_X_>)
///   -> memref<?x!sycl_nd_range_X_>
/// ```
class SYCLNDRangeConstructorEraseDefaultOffset
    : public OpRewritePattern<SYCLNDRangeConstructorOp> {
public:
  using OpRewritePattern<SYCLNDRangeConstructorOp>::OpRewritePattern;

  LogicalResult match(SYCLNDRangeConstructorOp op) const final {
    OperandRange args = op.getArgs();
    return success(args.size() == offsetOperandIndex + 1 &&
                   isDefaultConstructedID(args[offsetOperandIndex]));
  }

  void rewrite(SYCLNDRangeConstructorOp op,
               PatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [=] { op->eraseOperand(offsetOperandIndex); });
  }

private:
  constexpr static unsigned offsetOperandIndex = 2;
};
} // namespace

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

  return isBaseClass(OutputElemType, InputElemType);
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
          // With opaque pointers, there is no element type to inspect.
          return success();
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
    return checkNumIndices(*this, argTypes.size(), dimensions);
  }
  if (argTypes.size() == 1) {
    if (auto MT = dyn_cast<MemRefType>(argTypes.front());
        MT && isa<IDType, ItemType, RangeType>(MT.getElementType()))
      // sycl.id.constructor(memref<?x!sycl_[id|item|range]_N>) ->
      // memref<1x!sycl_id_N>
      return checkSameDimensions(*this, type, MT);
  }
  return emitBadSignatureError(*this);
}

/// Fold:
/// ```
/// %c0 = arith.constant 0 : index
/// sycl.id.constructor(%c0+) : (index+) -> memref<?x!sycl_id_X_>
/// ```
/// To:
/// ```
/// sycl.id.constructor() : () -> memref<?x!sycl_id_X_>
/// ```
OpFoldResult SYCLIDConstructorOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> args = adaptor.getArgs();
  if (args.empty())
    // Already dealing with the default constructor
    return {};

  // Check all arguments are a constant 0
  if (!llvm::all_of(args, [](Attribute attr) {
        auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(attr);
        return intAttr && intAttr.getValue() == 0;
      }))
    return {};

  // Erase arguments
  (*this)->setOperands({});
  return getId();
}

LogicalResult SYCLRangeConstructorOp::verify() {
  OperandRange::type_range argTypes = getArgs().getTypes();
  MemRefType thisType = getRange().getType();
  unsigned dimensions = getDimensions(thisType);
  unsigned numArgs = getNumOperands();
  switch (numArgs) {
  case 1:
    if (auto mt = dyn_cast<MemRefType>(argTypes.front())) {
      if (auto rangeType = dyn_cast<RangeType>(mt.getElementType()))
        // sycl.range.constructor(memref<?xsycl_range_N>) ->
        // memref<1xsycl_range_N>
        return checkSameDimensions(*this, thisType, rangeType);
      break;
    }
    [[fallthrough]];
  case 2:
  case 3:
    // sycl.range.constructor(index) -> memref<1xsycl_range_1>
    // sycl.range.constructor(index, index) -> memref<1xsycl_range_2>
    // sycl.range.constructor(index, index, index) -> memref<1xsycl_range_3>
    if (llvm::all_of(argTypes, [](Type type) { return isa<IndexType>(type); }))
      return checkNumIndices(*this, numArgs, dimensions);
    [[fallthrough]];
  default:
    break;
  }
  return emitBadSignatureError(*this);
}

void SYCLRangeConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto *defaultResource = SideEffects::DefaultResource::get();
  // All of the arguments will be scalars or read from
  for (auto value : getArgs()) {
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
  }
  // The result will be allocated and written to
  Value range = getRange();
  effects.emplace_back(MemoryEffects::Allocate::get(), range, defaultResource);
  effects.emplace_back(MemoryEffects::Write::get(), range, defaultResource);
}

static Type getBodyType(Type type) {
  auto *ctx = type.getContext();
  return TypeSwitch<Type, Type>(type)
      .Case<HalfType>([&](auto) { return Float16Type::get(ctx); })
      .Case<VecType>([&](auto vecTy) {
        if (isa<HalfType>(vecTy.getDataType()))
          return VectorType::get({vecTy.getNumElements()},
                                 Float16Type::get(ctx));
        return VectorType::get({vecTy.getNumElements()}, vecTy.getDataType());
      })
      .Default({});
}

LogicalResult SYCLNDRangeConstructorOp::verify() {
  OperandRange::type_range argTypes = getArgs().getTypes();
  MemRefType thisType = getNDRange().getType();
  switch (argTypes.size()) {
  case 1:
    // sycl.nd_range.constructor(memref<?xsycl_nd_range_N>) ->
    // memref<1x!sycl_nd_range_N>
    if (auto nd = getMemRefElementType<NdRangeType>(argTypes.front()))
      return checkSameDimensions(*this, thisType, nd);
    break;
  case 3:
    // (memref<?xsycl_range_N>, memref<?xsycl_range_N>, memref<?xsycl_id_N>) ->
    // memref<1x!sycl_nd_range_N>
    if (auto id = getMemRefElementType<IDType>(argTypes[2])) {
      if (LogicalResult sameDims = checkSameDimensions(*this, thisType, id);
          failed(sameDims))
        return sameDims;
    } else {
      break;
    }
    [[fallthrough]];
  case 2:
    // (memref<?xsycl_range_N>, memref<?xsycl_range_N>, ...) ->
    // memref<1x!sycl_nd_range_N>
    if (std::all_of(argTypes.begin(), argTypes.begin() + 2, [](Type type) {
          return static_cast<bool>(getMemRefElementType<RangeType>(type));
        })) {
      // Check all inputs have the same number of dimensions as the output
      for (Type type : argTypes) {
        LogicalResult result = checkSameDimensions(*this, thisType, type);
        if (failed(result))
          return result;
      }
      return success();
    }
    [[fallthrough]];
  default:
    break;
  }
  return emitBadSignatureError(*this);
}

void SYCLNDRangeConstructorOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<SYCLNDRangeConstructorEraseDefaultOffset>(context);
}

void SYCLNDRangeConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto *defaultResource = SideEffects::DefaultResource::get();
  // All of the arguments will be read from
  for (auto value : getArgs())
    effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
  // The result will be allocated and written to
  Value ndRange = getNDRange();
  effects.emplace_back(MemoryEffects::Allocate::get(), ndRange,
                       defaultResource);
  effects.emplace_back(MemoryEffects::Write::get(), ndRange, defaultResource);
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

LogicalResult SYCLHostConstructorOp::verify() {
  Type type = getType().getValue();
  if (!isa<SYCLType>(type))
    return emitOpError("expecting a sycl type as constructed type. Got ")
           << type;
  return success();
}

void SYCLHostConstructorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // NOTE: This definition assumes only the first (`this`) argument is written
  // to and the remaining arguments, if they are memory arguments, are only read
  // from. This is optimistic assumption, but enables better analysis and is
  // true for the types and properties we are most interested in right now.
  auto *defaultResource = SideEffects::DefaultResource::get();
  // The `this` argument will always be written to
  effects.emplace_back(MemoryEffects::Write::get(), getDst(), defaultResource);
  // The rest of the arguments will be scalar or read from
  for (auto value : getArgs())
    if (isa<MemRefType, LLVM::LLVMPointerType>(value.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
}

LogicalResult
SYCLHostGetKernelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

LogicalResult
SYCLHostHandlerSetKernel::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

static LogicalResult verifyNdRange(Operation *op, Value firstArg,
                                   Value secondArg, bool ndRange) {
  return ndRange && secondArg ? op->emitOpError("expects no offset argument if "
                                                "the nd_range attribute is set")
                              : success();
}

LogicalResult SYCLHostHandlerSetNDRange::verify() {
  return verifyNdRange(*this, getRange(), getOffset(), getNdRange());
}

static LogicalResult verifySYCLTypeAttribute(Operation *op, Value value,
                                             Type type) {
  if (!isa<LLVM::LLVMPointerType>(value.getType()))
    return op->emitOpError(
        "does not expect a type attribute for a non-pointer value");
  if (!isa<SYCLType>(type))
    return op->emitOpError(
        "expects the type attribute to reference a SYCL type");

  return success();
}

LogicalResult SYCLHostSetCaptured::verify() {
  if (auto syclType = getSyclType())
    return verifySYCLTypeAttribute(*this, getValue(), syclType.value());
  return success();
}

static mlir::ParseResult
parseArgsWithSYCLTypes(mlir::OpAsmParser &parser,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand> &args,
                       ArrayAttr &syclTypes) {
  auto &builder = parser.getBuilder();
  SmallVector<Type> types;

  if (parser.parseOptionalLParen()) {
    // no argument list
    syclTypes = builder.getTypeArrayAttr(types);
    return success();
  }

  auto parseOperandAndOptionalType = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();

    args.push_back(operand);

    Type type = builder.getNoneType();
    if (parser.parseOptionalColon()) {
      // no type attribute
      types.push_back(type);
      return success();
    }

    if (parser.parseType(type))
      return failure();

    types.push_back(type);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOperandAndOptionalType) ||
      parser.parseRParen())
    return failure();

  syclTypes = builder.getTypeArrayAttr(types);
  return success();
}

static void printArgsWithSYCLTypes(mlir::OpAsmPrinter &printer,
                                   sycl::SYCLHostScheduleKernel op,
                                   OperandRange args, ArrayAttr syclTypes) {
  if (args.empty())
    return;
  printer << '(';
  llvm::interleaveComma(llvm::zip(args, syclTypes.getAsRange<TypeAttr>()),
                        printer, [&](auto it) {
                          printer.printOperand(std::get<0>(it));
                          TypeAttr attr = std::get<1>(it);
                          if (!isa<NoneType>(attr.getValue())) {
                            printer << ": ";
                            printer.printType(attr.getValue());
                          }
                        });
  printer << ')';
}

LogicalResult
SYCLHostScheduleKernel::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyReferencesKernel(*this, symbolTable, getKernelNameAttr());
}

LogicalResult SYCLHostScheduleKernel::verify() {
  // TODO: verify that the given args match the kernel's signature.
  Value range = getRange();
  Value offset = getOffset();
  bool ndRange = getNdRange();
  if (!range) {
    if (offset)
      return emitOpError("expects no offset when no range is present");
    if (ndRange)
      return emitOpError(
          "expects nd_range to be unset when a range is not present");
  }

  if (failed(verifyNdRange(*this, range, offset, ndRange)))
    return failure();

  if (getArgs().size() != getSyclTypes().size())
    return emitOpError("has inconsistent SYCL type attributes");

  for (auto it :
       llvm::zip(getArgs(), getSyclTypes().getAsValueRange<TypeAttr>())) {
    Value arg = std::get<0>(it);
    Type syclType = std::get<1>(it);
    if (!isa<NoneType>(syclType) &&
        failed(verifySYCLTypeAttribute(*this, arg, syclType)))
      return failure();
  }

  return success();
}

void SYCLHostScheduleKernel::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto *defaultResource = SideEffects::DefaultResource::get();
  // Unconditionally add a write effect on the handler to prevent the op from
  // being trivially dead when all other operands are read-only.
  effects.emplace_back(MemoryEffects::Write::get(), getHandler(),
                       defaultResource);
  // The nd-range arguments will be read from
  if (Value range = getRange())
    effects.emplace_back(MemoryEffects::Read::get(), range, defaultResource);
  if (Value offset = getOffset())
    effects.emplace_back(MemoryEffects::Read::get(), offset, defaultResource);
  // The rest of the arguments will be scalar, accessors with read access or
  // have read-write access mode.
  for (auto iter :
       llvm::zip(getArgs(), getSyclTypes().getAsValueRange<TypeAttr>())) {
    auto [value, type] = iter;
    if (!isa<LLVM::LLVMPointerType>(value.getType()))
      // Scalars
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), value, defaultResource);
    if (!llvm::isa_and_nonnull<AccessorType>(type))
      // Not sycl::accessor
      effects.emplace_back(MemoryEffects::Write::get(), value, defaultResource);
  }
}

LogicalResult
SYCLHostSubmitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  FlatSymbolRefAttr symbol = getCGFNameAttr();
  auto cgf =
      symbolTable.lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(*this, symbol);
  if (!cgf)
    return emitOpError("'")
           << symbol << "' does not reference a valid CGF function";
  if (LLVM::Linkage linkage = cgf.getLinkage();
      linkage != LLVM::Linkage::Internal) {
    auto diag = emitOpError()
                << "expects CGF function to have internal linkage";
    diag.attachNote() << "got: '" << stringifyEnum(linkage) << "'";
    return diag;
  }

  if (cgf.isVarArg())
    return emitOpError("expects CGF function to not have variadic arguments");

  constexpr unsigned numInputs = 2;
  LLVM::LLVMFunctionType fnType = cgf.getFunctionType();
  if (fnType.getNumParams() != numInputs)
    return emitOpError("incorrect number of operands for CGF");

  for (unsigned i = 0; i < numInputs; ++i) {
    Type type = fnType.getParamType(i);
    if (!isa<LLVM::LLVMPointerType>(type))
      return emitOpError("expecting CGF's operand type '!llvm.ptr', but got ")
             << type << " for operand number " << i;
  }

  if (Type returnType = fnType.getReturnType();
      !isa<LLVM::LLVMVoidType>(returnType))
    return emitOpError("expecting CGF's result type '!llvm.void', but got ")
           << returnType;

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
