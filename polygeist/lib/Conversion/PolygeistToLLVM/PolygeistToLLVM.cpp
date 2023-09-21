//===- PolygeistToLLVM.cpp -------------------------------------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PolygeistToLLVM/PolygeistToLLVM.h"

#include <numeric>

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"
#include "mlir/Conversion/SYCLToMath/SYCLToMath.h"
#include "mlir/Conversion/SYCLToSPIRV/SYCLToSPIRV.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-polygeist-to-llvm"

using namespace mlir;
using namespace polygeist;

namespace mlir {
#define GEN_PASS_DEF_CONVERTPOLYGEISTTOLLVM
#include "mlir/Conversion/PolygeistPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTPOLYGEISTTOLLVM
} // namespace mlir

/// Return \p mrTy's address space as an integer (being 0 the default).
static std::optional<unsigned>
getMemorySpaceAsInt(const TypeConverter &converter, MemRefType mrTy) {
  Attribute memorySpace = mrTy.getMemorySpace();
  if (!memorySpace)
    return 0;
  std::optional<Attribute> convAttr =
      converter.convertTypeAttribute(mrTy, memorySpace);
  if (!convAttr)
    return std::nullopt;
  return cast<IntegerAttr>(*convAttr).getInt();
}

/// Returns failure if \p mrTy memory space is not compatible with \p ptrTy's
/// address space.
///
/// This is actually an extension to `polygeist.pointer2memref` and
/// `polygeist.memref2pointer` verification, as we do not have the required
/// information to check non-integer memory spaces compatibility with llvm.ptr
/// address spaces.
static LogicalResult verifyPtrMemrefConversion(const TypeConverter &converter,
                                               Operation *op,
                                               LLVM::LLVMPointerType ptrTy,
                                               MemRefType mrTy) {
  std::optional<unsigned> memorySpaceAsInt =
      getMemorySpaceAsInt(converter, mrTy);
  if (!memorySpaceAsInt)
    return op->emitOpError()
           << "Cannot convert memory space '" << mrTy.getMemorySpace() << "'";
  if (*memorySpaceAsInt == ptrTy.getAddressSpace())
    return success();
  return op->emitOpError() << "Incompatible address space '"
                           << ptrTy.getAddressSpace() << "' and memory space '"
                           << mrTy.getMemorySpace() << "'";
}

struct BaseSubIndexOpLowering : public ConvertOpToLLVMPattern<SubIndexOp> {
  using ConvertOpToLLVMPattern<SubIndexOp>::ConvertOpToLLVMPattern;

protected:
  // Compute the indices of the GEP operation we lower the SubIndexOp to.
  // The indices are computed based on:
  //   a) the (converted) source element type, and
  //   b) the (converted) result element type that is requested
  // Examples:
  //  - src ty: ptr<struct<array<1xi64>>>, res ty: ptr<i64>
  //      -> idxs = [0, 0, SubIndexOp's index]
  //  - src ty: ptr<struct<array<1xi64>>>, res ty: ptr<array<1xi64>>
  //      -> idxs = [0, SubIndexOp's index]
  //
  // Note: when the source element type is a struct with more than one member
  // type, the result type that is requested is deemed illegal unless it is one
  // of the source member types. For example assume:
  //   - src ty: ptr<struct<array<1xi64>,i32>>
  //   - res ty: ptr<i64>
  // This is illegal because res ty can only be either ptr<i32> or
  // ptr<array<1xi64>>
  static void computeIndices(const LLVM::LLVMStructType &srcElemType,
                             const Type &resElemType,
                             SmallVectorImpl<Value> &indices, SubIndexOp op,
                             OpAdaptor transformed,
                             ConversionPatternRewriter &rewriter) {
    assert(indices.empty() && "Expecting an empty vector");

    ArrayRef<Type> memTypes = srcElemType.getBody();
    unsigned numMembers = memTypes.size();
    assert((numMembers == 1 ||
            any_of(memTypes, [=](Type t) { return resElemType == t; })) &&
           "The requested result memref element type is illegal");

    Type indexType = transformed.getIndex().getType();
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), indexType, rewriter.getIntegerAttr(indexType, 0));
    indices.push_back(zero);

    if (numMembers == 1) {
      Type currType = srcElemType.getBody()[0];
      while (currType != resElemType) {
        indices.push_back(zero);

        TypeSwitch<Type>(currType)
            .Case<LLVM::LLVMStructType, polygeist::StructType>([&](auto t) {
              assert(t.getBody().size() == 1 && "Expecting single member type");
              currType = t.getBody()[0];
            })
            .Case<LLVM::LLVMArrayType>(
                [&](auto t) { currType = t.getElementType(); })
            .Case<LLVM::LLVMPointerType>(
                [&](auto t) { llvm_unreachable("Pointer type not allowed"); })
            .Default([&](Type t) {
              currType = t;
              assert(currType == resElemType &&
                     "requested result type is illegal");
            });
      }
    }

    indices.push_back(transformed.getIndex());
  }
};

struct SubIndexOpLowering : public BaseSubIndexOpLowering {
  using BaseSubIndexOpLowering::BaseSubIndexOpLowering;

  LogicalResult
  matchAndRewrite(SubIndexOp subViewOp, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<MemRefType>(subViewOp.getSource().getType()) &&
           "Source operand should be a memref type");
    assert(isa<MemRefType>(subViewOp.getType()) &&
           "Result should be a memref type");

    auto sourceMemRefType = cast<MemRefType>(subViewOp.getSource().getType());
    auto viewMemRefType = cast<MemRefType>(subViewOp.getType());
    if (!canBeLoweredToBarePtr(sourceMemRefType) ||
        !canBeLoweredToBarePtr(viewMemRefType))
      return failure();

    const auto loc = subViewOp.getLoc();
    const auto target = transformed.getSource();
    auto idx = transformed.getIndex();

    if (sourceMemRefType.getRank() != viewMemRefType.getRank()) {
      if (sourceMemRefType.getRank() != viewMemRefType.getRank() + 1)
        return failure();

      size_t sz = 1;
      for (int64_t i = 1; i < sourceMemRefType.getRank(); i++) {
        if (sourceMemRefType.getShape()[i] == ShapedType::kDynamic)
          return failure();
        sz *= sourceMemRefType.getShape()[i];
      }
      Value cop = rewriter.create<LLVM::ConstantOp>(
          loc, idx.getType(), rewriter.getIntegerAttr(idx.getType(), sz));
      idx = rewriter.create<LLVM::MulOp>(loc, idx, cop);
    }

    Type sourceElemType = sourceMemRefType.getElementType();
    Type convSourceElemType = getTypeConverter()->convertType(sourceElemType);
    if (!convSourceElemType)
      return failure();
    Type viewElemType = viewMemRefType.getElementType();
    Type convViewElemType = getTypeConverter()->convertType(viewElemType);
    Type resType = getTypeConverter()->convertType(subViewOp.getType());

    // Handle the general (non-SYCL) case first.
    if (convViewElemType == convSourceElemType) {
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(subViewOp, resType,
                                               convViewElemType, target, idx);
      return success();
    }
    assert(isa<LLVM::LLVMStructType>(convSourceElemType) &&
           "Expecting struct type");

    // SYCL case
    assert(sourceMemRefType.getRank() == viewMemRefType.getRank() &&
           "Expecting the input and output MemRef ranks to be the same");

    SmallVector<Value> indices;
    computeIndices(cast<LLVM::LLVMStructType>(convSourceElemType),
                   convViewElemType, indices, subViewOp, transformed, rewriter);
    assert(!indices.empty() && "Expecting a least one index");

    // Note: MLIRScanner::InitializeValueByInitListExpr() in clang-mlir.cc, when
    // a memref element type is a struct type, the return type of a
    // polygeist.subindex operation should be a memref of the element type of
    // the struct.

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        subViewOp, resType, convSourceElemType, target, indices);

    return success();
  }
};

struct StreamToTokenOpLowering
    : public ConvertOpToLLVMPattern<StreamToTokenOp> {
  using ConvertOpToLLVMPattern<StreamToTokenOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StreamToTokenOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Value v[] = {transformed.getSource()};
    rewriter.replaceOp(op, v);
    return success();
  }
};

/// Lowers to a bitcast operation
struct Memref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(op.getSource().getType()))
      return failure();

    if (LogicalResult verifyResult = verifyPtrMemrefConversion(
            *getTypeConverter(), op, op.getType(), op.getSource().getType());
        failed(verifyResult))
      return verifyResult;

    const auto target = transformed.getSource();
    // In an opaque pointer world, a bitcast is a no-op, so no need to insert
    // one here.
    rewriter.replaceOp(op, target);

    return success();
  }
};

/// Lowers to a bitcast operation
struct Pointer2MemrefOpLowering
    : public ConvertOpToLLVMPattern<Pointer2MemrefOp> {
  using ConvertOpToLLVMPattern<Pointer2MemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(op.getType()))
      return failure();

    if (LogicalResult verifyResult = verifyPtrMemrefConversion(
            *getTypeConverter(), op, op.getSource().getType(), op.getType());
        failed(verifyResult))
      return verifyResult;

    const auto convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();
    // In an opaque pointer world, a bitcast is a no-op, so no need to insert
    // one here.
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

struct TypeSizeOpLowering : public ConvertOpToLLVMPattern<TypeSizeOp> {
  using ConvertOpToLLVMPattern<TypeSizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeSizeOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Type NT = op.getSourceAttr().getValue();
    if (auto T = getTypeConverter()->convertType(NT)) {
      NT = T;
    }
    assert(NT);

    auto type = getTypeConverter()->convertType(op.getType());

    if (isa<IntegerType, FloatType>(NT) || LLVM::isCompatibleType(NT)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, DLI.getTypeSize(NT)));
      return success();
    }

    if (NT != op.getSourceAttr().getValue() || type != op.getType()) {
      rewriter.replaceOpWithNewOp<TypeSizeOp>(op, type, NT);
      return success();
    }
    return failure();
  }
};

struct TypeAlignOpLowering : public ConvertOpToLLVMPattern<TypeAlignOp> {
  using ConvertOpToLLVMPattern<TypeAlignOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeAlignOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Type NT = op.getSourceAttr().getValue();
    if (auto T = getTypeConverter()->convertType(NT)) {
      NT = T;
    }
    assert(NT);

    auto type = getTypeConverter()->convertType(op.getType());

    if (isa<IntegerType, FloatType>(NT) || LLVM::isCompatibleType(NT)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, DLI.getTypeABIAlignment(NT)));
      return success();
    }

    if (NT != op.getSourceAttr().getValue() || type != op.getType()) {
      rewriter.replaceOpWithNewOp<TypeAlignOp>(op, type, NT);
      return success();
    }
    return failure();
  }
};

static void
populatePolygeistToLLVMTypeConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](polygeist::StructType type) -> Optional<Type> {
        llvm::ArrayRef<Type> body = type.getBody();
        SmallVector<Type> convertedElemTypes;
        convertedElemTypes.reserve(body.size());
        if (failed(typeConverter.convertTypes(body, convertedElemTypes)))
          return std::nullopt;
        if (type.getName().has_value()) {
          if (type.isOpaque())
            return LLVM::LLVMStructType::getOpaque(*type.getName(),
                                                   &typeConverter.getContext());
          auto ST = LLVM::LLVMStructType::getIdentified(
              &typeConverter.getContext(), *type.getName());
          if (!ST.isInitialized()) {
            if (failed(ST.setBody(convertedElemTypes, type.isPacked())))
              return std::nullopt;
          } else if (convertedElemTypes != ST.getBody()) {
            ST = LLVM::LLVMStructType::getNewIdentified(
                &typeConverter.getContext(), *type.getName(),
                convertedElemTypes, type.isPacked());
          }
          return ST;
        }
        return LLVM::LLVMStructType::getLiteral(&typeConverter.getContext(),
                                                convertedElemTypes);
      });
}

void populatePolygeistToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  assert(converter.getOptions().useBarePtrCallConv &&
         "These patterns only work with bare pointer calling convention");
  populatePolygeistToLLVMTypeConversion(converter);

  patterns.add<TypeSizeOpLowering, TypeAlignOpLowering>(converter);
  patterns.add<SubIndexOpLowering, Memref2PointerOpLowering,
               Pointer2MemrefOpLowering>(converter);
}

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();
    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes))) {
      return failure();
    }
    SmallVector<Type> convertedOperandTypes;
    if (failed(converter->convertTypes(op->getOperandTypes(),
                                       convertedOperandTypes))) {
      return failure();
    }

    // With opaque pointers, type attributes also might need to be
    // translated to the corresponding LLVM types, for example the element
    // type attribute of GEP or alloca.
    bool needsTyAttrConversion =
        llvm::any_of(op->getAttrs(), [&](const NamedAttribute &Attr) {
          if (auto TyAttr = dyn_cast<TypeAttr>(Attr.getValue()))
            return !converter->isLegal(TyAttr.getValue());

          return false;
        });

    if (convertedResultTypes == op->getResultTypes() &&
        convertedOperandTypes == op->getOperandTypes() &&
        !needsTyAttrConversion) {
      return failure();
    }
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    if (needsTyAttrConversion) {
      SmallVector<NamedAttribute> Attrs;
      for (const auto &NA : op->getAttrs()) {
        if (auto tyAttr = dyn_cast<TypeAttr>(NA.getValue())) {
          auto convTy = converter->convertType(tyAttr.getValue());
          assert(convTy);
          Attrs.emplace_back(NA.getName(), TypeAttr::get(convTy));
        } else {
          Attrs.push_back(NA);
        }
      }
      state.addAttributes(Attrs);
    } else {
      state.addAttributes(op->getAttrs());
    }
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.create(state);
    rewriter.replaceOp(op, rewritten->getResults());

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    return success();
  }
};

// FIXME: The following function and pattern with the "Old" suffix should be
// removed once we drop typed pointer support.

// TODO lock this wrt module
static LLVM::LLVMFuncOp addMocCUDAFunctionOld(ModuleOp module, Type streamTy) {
  const char fname[] = "fake_cuda_dispatch";

  MLIRContext *ctx = module.getContext();
  auto loc = module.getLoc();
  auto moduleBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  for (auto fn : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (fn.getName() == fname)
      return fn;
  }

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      fname, LLVM::LLVMFunctionType::get(
                 voidTy, {i8Ptr,
                          LLVM::LLVMPointerType::get(
                              LLVM::LLVMFunctionType::get(voidTy, {i8Ptr})),
                          streamTy}));
  resumeOp.setPrivate();

  return resumeOp;
}

struct AsyncOpLoweringOld : public ConvertOpToLLVMPattern<async::ExecuteOp> {
  using ConvertOpToLLVMPattern<async::ExecuteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::ExecuteOp execute, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = execute->getParentOfType<ModuleOp>();

    MLIRContext *ctx = module.getContext();
    Location loc = execute.getLoc();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    Type voidPtr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

    // Make sure that all constants will be inside the outlined async function
    // to reduce the number of function arguments.
    Region &funcReg = execute.getRegion();

    // Collect all outlined function inputs.
    SetVector<mlir::Value> functionInputs;

    getUsedValuesDefinedAbove(execute.getRegion(), funcReg, functionInputs);
    SmallVector<Value> toErase;
    for (auto a : functionInputs) {
      Operation *op = a.getDefiningOp();
      if (op && op->hasTrait<OpTrait::ConstantLike>())
        toErase.push_back(a);
    }
    for (auto a : toErase) {
      functionInputs.remove(a);
    }

    // Collect types for the outlined function inputs and outputs.
    const TypeConverter *converter = getTypeConverter();
    auto typesRange = llvm::map_range(functionInputs, [&](Value value) {
      return converter->convertType(value.getType());
    });
    SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());

    Type ftypes[] = {voidPtr};
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, ftypes);

    // TODO: Derive outlined function name from the parent FuncOp (support
    // multiple nested async.execute operations).
    auto moduleBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

    static int off = 0;
    off++;
    auto func = moduleBuilder.create<LLVM::LLVMFuncOp>(
        execute.getLoc(),
        "kernelbody." + std::to_string((long long int)&execute) + "." +
            std::to_string(off),
        funcType);

    rewriter.setInsertionPointToStart(func.addEntryBlock());
    IRMapping valueMapping;
    for (Value capture : toErase) {
      Operation *op = capture.getDefiningOp();
      for (auto r :
           llvm::zip(op->getResults(),
                     rewriter.clone(*op, valueMapping)->getResults())) {
        valueMapping.map(rewriter.getRemappedValue(std::get<0>(r)),
                         std::get<1>(r));
      }
    }
    // Prepare for coroutine conversion by creating the body of the function.
    {
      // Map from function inputs defined above the execute op to the function
      // arguments.
      auto arg = func.getArgument(0);

      if (functionInputs.size() == 0) {
      } else if (functionInputs.size() == 1 &&
                 isa<LLVM::LLVMPointerType>(
                     converter->convertType(functionInputs[0].getType()))) {
        valueMapping.map(
            functionInputs[0],
            rewriter.create<LLVM::BitcastOp>(
                execute.getLoc(),
                converter->convertType(functionInputs[0].getType()), arg));
      } else if (functionInputs.size() == 1 &&
                 isa<IntegerType>(
                     converter->convertType(functionInputs[0].getType()))) {
        valueMapping.map(
            functionInputs[0],
            rewriter.create<LLVM::PtrToIntOp>(
                execute.getLoc(),
                converter->convertType(functionInputs[0].getType()), arg));
      } else {
        SmallVector<Type> types;
        for (auto v : functionInputs)
          types.push_back(converter->convertType(v.getType()));
        auto ST = LLVM::LLVMStructType::getLiteral(ctx, types);
        auto alloc = rewriter.create<LLVM::BitcastOp>(
            execute.getLoc(), LLVM::LLVMPointerType::get(ST), arg);
        for (auto idx : llvm::enumerate(functionInputs)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          Value next = rewriter.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(idx.value().getType()), alloc,
              idxs);
          valueMapping.map(idx.value(),
                           rewriter.create<LLVM::LoadOp>(loc, next));
        }
        auto freef = getFreeFn(*getTypeConverter(), module);
        Value args[] = {arg};
        rewriter.create<LLVM::CallOp>(loc, freef, args);
      }

      // Clone all operations from the execute operation body into the outlined
      // function body.
      for (Operation &op : execute.getBody()->without_terminator())
        rewriter.clone(op, valueMapping);

      rewriter.create<LLVM::ReturnOp>(execute.getLoc(), ValueRange());
    }

    // Replace the original `async.execute` with a call to outlined function.
    {
      rewriter.setInsertionPoint(execute);
      SmallVector<Value> crossing;
      for (auto tup : llvm::zip(functionInputs, inputTypes)) {
        Value val = std::get<0>(tup);
        crossing.push_back(val);
      }

      SmallVector<Value> vals;
      if (crossing.size() == 0) {
        vals.push_back(
            rewriter.create<LLVM::NullOp>(execute.getLoc(), voidPtr));
      } else if (crossing.size() == 1 &&
                 isa<LLVM::LLVMPointerType>(
                     converter->convertType(crossing[0].getType()))) {
        vals.push_back(rewriter.create<LLVM::BitcastOp>(execute.getLoc(),
                                                        voidPtr, crossing[0]));
      } else if (crossing.size() == 1 &&
                 isa<IntegerType>(
                     converter->convertType(crossing[0].getType()))) {
        vals.push_back(rewriter.create<LLVM::IntToPtrOp>(execute.getLoc(),
                                                         voidPtr, crossing[0]));
      } else {
        SmallVector<Type> types;
        for (auto v : crossing)
          types.push_back(v.getType());
        auto ST = LLVM::LLVMStructType::getLiteral(ctx, types);

        auto mallocf = getAllocFn(*getTypeConverter(), module, getIndexType());

        Value args[] = {rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(),
            rewriter.create<polygeist::TypeSizeOp>(loc, rewriter.getIndexType(),
                                                   ST))};
        mlir::Value alloc = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(ST),
            rewriter.create<mlir::LLVM::CallOp>(loc, mallocf, args)
                .getResult());
        rewriter.setInsertionPoint(execute);
        for (auto idx : llvm::enumerate(crossing)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          Value next = rewriter.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(idx.value().getType()), alloc,
              idxs);
          rewriter.create<LLVM::StoreOp>(loc, idx.value(), next);
        }
        vals.push_back(
            rewriter.create<LLVM::BitcastOp>(execute.getLoc(), voidPtr, alloc));
      }
      vals.push_back(
          rewriter.create<LLVM::AddressOfOp>(execute.getLoc(), func));
      for (auto dep : execute.getDependencies()) {
        auto ctx = dep.getDefiningOp<polygeist::StreamToTokenOp>();
        vals.push_back(ctx.getSource());
      }
      assert(vals.size() == 3);

      auto f = addMocCUDAFunctionOld(execute->getParentOfType<ModuleOp>(),
                                     vals.back().getType());

      rewriter.create<LLVM::CallOp>(execute.getLoc(), f, vals);
      rewriter.eraseOp(execute);
    }

    return success();
  }
};

// TODO lock this wrt module
static LLVM::LLVMFuncOp addMocCUDAFunction(ModuleOp module, Type streamTy) {
  const char fname[] = "fake_cuda_dispatch";

  MLIRContext *ctx = module.getContext();
  auto loc = module.getLoc();
  auto moduleBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  for (auto fn : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (fn.getName() == fname)
      return fn;
  }

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i8Ptr = LLVM::LLVMPointerType::get(ctx);

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      fname, LLVM::LLVMFunctionType::get(
                 voidTy, {i8Ptr, LLVM::LLVMPointerType::get(ctx), streamTy}));
  resumeOp.setPrivate();

  return resumeOp;
}

struct AsyncOpLowering : public ConvertOpToLLVMPattern<async::ExecuteOp> {
  using ConvertOpToLLVMPattern<async::ExecuteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::ExecuteOp execute, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = execute->getParentOfType<ModuleOp>();

    MLIRContext *ctx = module.getContext();
    Location loc = execute.getLoc();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    Type voidPtr = LLVM::LLVMPointerType::get(ctx);

    // Make sure that all constants will be inside the outlined async function
    // to reduce the number of function arguments.
    Region &funcReg = execute.getRegion();

    // Collect all outlined function inputs.
    SetVector<mlir::Value> functionInputs;

    getUsedValuesDefinedAbove(execute.getRegion(), funcReg, functionInputs);
    SmallVector<Value> toErase;
    for (auto a : functionInputs) {
      Operation *op = a.getDefiningOp();
      if (op && op->hasTrait<OpTrait::ConstantLike>())
        toErase.push_back(a);
    }
    for (auto a : toErase) {
      functionInputs.remove(a);
    }

    // Collect types for the outlined function inputs and outputs.
    const TypeConverter *converter = getTypeConverter();
    auto typesRange = llvm::map_range(functionInputs, [&](Value value) {
      return converter->convertType(value.getType());
    });
    SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());

    Type ftypes[] = {voidPtr};
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, ftypes);

    // TODO: Derive outlined function name from the parent FuncOp (support
    // multiple nested async.execute operations).
    auto moduleBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

    static int off = 0;
    off++;
    auto func = moduleBuilder.create<LLVM::LLVMFuncOp>(
        execute.getLoc(),
        "kernelbody." + std::to_string((long long int)&execute) + "." +
            std::to_string(off),
        funcType);

    rewriter.setInsertionPointToStart(func.addEntryBlock());
    IRMapping valueMapping;
    for (Value capture : toErase) {
      Operation *op = capture.getDefiningOp();
      for (auto r :
           llvm::zip(op->getResults(),
                     rewriter.clone(*op, valueMapping)->getResults())) {
        valueMapping.map(rewriter.getRemappedValue(std::get<0>(r)),
                         std::get<1>(r));
      }
    }
    // Prepare for coroutine conversion by creating the body of the function.
    {
      // Map from function inputs defined above the execute op to the function
      // arguments.
      auto arg = func.getArgument(0);

      if (functionInputs.size() == 0) {
      } else if (functionInputs.size() == 1 &&
                 isa<LLVM::LLVMPointerType>(
                     converter->convertType(functionInputs[0].getType()))) {
        valueMapping.map(functionInputs[0], arg);
      } else if (functionInputs.size() == 1 &&
                 isa<IntegerType>(
                     converter->convertType(functionInputs[0].getType()))) {
        valueMapping.map(
            functionInputs[0],
            rewriter.create<LLVM::PtrToIntOp>(
                execute.getLoc(),
                converter->convertType(functionInputs[0].getType()), arg));
      } else {
        SmallVector<Type> types;
        for (auto v : functionInputs)
          types.push_back(converter->convertType(v.getType()));

        for (auto idx : llvm::enumerate(functionInputs)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          auto nextTy = types[idx.index()];
          Value next = rewriter.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(ctx), nextTy, arg, idxs);
          valueMapping.map(idx.value(),
                           rewriter.create<LLVM::LoadOp>(loc, nextTy, next));
        }
        auto freef = getFreeFn(*getTypeConverter(), module);
        Value args[] = {arg};
        rewriter.create<LLVM::CallOp>(loc, freef, args);
      }

      // Clone all operations from the execute operation body into the outlined
      // function body.
      for (Operation &op : execute.getBody()->without_terminator())
        rewriter.clone(op, valueMapping);

      rewriter.create<LLVM::ReturnOp>(execute.getLoc(), ValueRange());
    }

    // Replace the original `async.execute` with a call to outlined function.
    {
      rewriter.setInsertionPoint(execute);
      SmallVector<Value> crossing;
      for (auto tup : llvm::zip(functionInputs, inputTypes)) {
        Value val = std::get<0>(tup);
        crossing.push_back(val);
      }

      SmallVector<Value> vals;
      if (crossing.size() == 0) {
        vals.push_back(
            rewriter.create<LLVM::NullOp>(execute.getLoc(), voidPtr));
      } else if (crossing.size() == 1 &&
                 isa<LLVM::LLVMPointerType>(
                     converter->convertType(crossing[0].getType()))) {
        vals.push_back(crossing[0]);
      } else if (crossing.size() == 1 &&
                 isa<IntegerType>(
                     converter->convertType(crossing[0].getType()))) {
        vals.push_back(rewriter.create<LLVM::IntToPtrOp>(execute.getLoc(),
                                                         voidPtr, crossing[0]));
      } else {
        SmallVector<Type> types;
        for (auto v : crossing)
          types.push_back(v.getType());
        auto ST = LLVM::LLVMStructType::getLiteral(ctx, types);

        auto mallocf = getAllocFn(*getTypeConverter(), module, getIndexType());

        Value args[] = {rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(),
            rewriter.create<polygeist::TypeSizeOp>(loc, rewriter.getIndexType(),
                                                   ST))};
        mlir::Value alloc =
            rewriter.create<mlir::LLVM::CallOp>(loc, mallocf, args).getResult();
        rewriter.setInsertionPoint(execute);
        for (auto idx : llvm::enumerate(crossing)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          Value next =
              rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                           idx.value().getType(), alloc, idxs);
          rewriter.create<LLVM::StoreOp>(loc, idx.value(), next);
        }
        vals.push_back(alloc);
      }
      vals.push_back(
          rewriter.create<LLVM::AddressOfOp>(execute.getLoc(), func));
      for (auto dep : execute.getDependencies()) {
        auto ctx = dep.getDefiningOp<polygeist::StreamToTokenOp>();
        vals.push_back(ctx.getSource());
      }
      assert(vals.size() == 3);

      auto f = addMocCUDAFunction(execute->getParentOfType<ModuleOp>(),
                                  vals.back().getType());

      rewriter.create<LLVM::CallOp>(execute.getLoc(), f, vals);
      rewriter.eraseOp(execute);
    }

    return success();
  }
};

struct GlobalOpTypeConversion : public OpConversionPattern<LLVM::GlobalOp> {
  explicit GlobalOpTypeConversion(LLVMTypeConverter &converter)
      : OpConversionPattern<LLVM::GlobalOp>(converter,
                                            &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, LLVM::GlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();
    Type globalType = adaptor.getGlobalType();
    Type convertedType = converter->convertType(globalType);
    if (!convertedType)
      return failure();
    if (convertedType == globalType)
      return failure();

    rewriter.updateRootInPlace(
        op, [&]() { op.setGlobalTypeAttr(TypeAttr::get(convertedType)); });
    return success();
  }
};

struct ReturnOpTypeConversion : public ConvertOpToLLVMPattern<LLVM::ReturnOp> {
  using ConvertOpToLLVMPattern<LLVM::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, LLVM::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getArg());
    replacement->setAttrs(adaptor.getAttributes());
    return success();
  }
};

struct GPUModuleOpToModuleOpConversion
    : public ConvertOpToLLVMPattern<gpu::GPUModuleOp> {
  using ConvertOpToLLVMPattern<gpu::GPUModuleOp>::ConvertOpToLLVMPattern;

  /// Remove operations already defined in the parent module.
  static void
  removeAlreadyDefinedFunctions(gpu::GPUModuleOp deviceModule,
                                ConversionPatternRewriter &rewriter) {
    auto Module = deviceModule->getParentOfType<ModuleOp>();
    assert(Module && "Module not found");
    const auto Operations = deviceModule.getOps();
    SmallVector<std::reference_wrapper<Operation>> AlreadyDefined;
    std::copy_if(
        Operations.begin(), Operations.end(),
        std::back_inserter(AlreadyDefined), [&](Operation &Op) -> bool {
          if (isa<gpu::ModuleEndOp>(Op)) {
            // Erase GPUEndOp.
            return true;
          }
          // Erase operations already defined in the parent module.
          auto *Other = Module.lookupSymbol(SymbolTable::getSymbolName(&Op));
          return Other && Other->getParentOp() == Module;
        });
    for (auto Op : AlreadyDefined) {
      rewriter.eraseOp(&Op.get());
    }
  }

  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp deviceModule, gpu::GPUModuleOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Erase functions already present in the parent module.
    removeAlreadyDefinedFunctions(deviceModule, rewriter);
    // Copy contents to the parent module and erase the operation.
    auto module = deviceModule->getParentOfType<ModuleOp>();
    assert(module && "Module not found");
    rewriter.mergeBlocks(deviceModule.getBody(), module.getBody(), {});
    rewriter.eraseOp(deviceModule);
    return success();
  }
};

struct GPUFuncOpToFuncOpConversion
    : public ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  using ConvertOpToLLVMPattern<gpu::GPUFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, gpu::GPUFuncOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = gpuFuncOp->getParentOfType<ModuleOp>();

    rewriter.setInsertionPointToEnd(module.getBody());
    auto NewFuncOp = rewriter.create<func::FuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), gpuFuncOp.getFunctionType());
    NewFuncOp->setAttrs(gpuFuncOp->getAttrs());
    rewriter.notifyOperationInserted(NewFuncOp);

    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), NewFuncOp.getBody(),
                                NewFuncOp.getBody().end());

    rewriter.eraseOp(gpuFuncOp);

    return success();
  }
};

struct GPUModuleEndOpLowering
    : public ConvertOpToLLVMPattern<gpu::ModuleEndOp> {
  using ConvertOpToLLVMPattern<gpu::ModuleEndOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ModuleEndOp op, gpu::ModuleEndOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct EraseSPIRVBuiltinPattern
    : public OpRewritePattern<spirv::GlobalVariableOp> {
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::GlobalVariableOp global,
                                PatternRewriter &rewriter) const final {
    auto *parent = global->getParentWithTrait<OpTrait::SymbolTable>();
    if (!parent)
      return failure();
    const auto name = global.getName();
    bool alreadyDefined = llvm::any_of(parent->getRegion(0), [&](auto &block) {
      return llvm::any_of(block, [&](auto &op) {
        if (&op == global.getOperation())
          return false;
        auto nameAttr = op.template getAttrOfType<StringAttr>(
            mlir::SymbolTable::getSymbolAttrName());
        return nameAttr && nameAttr.getValue() == name;
      });
    });
    if (!alreadyDefined)
      return failure();
    rewriter.eraseOp(global);
    return success();
  }
};

class DuplicateFuncCleaner {
public:
  struct Entry {
    Entry(StringRef funcName, FunctionType funcType)
        : funcName(funcName), signature(funcType) {}

    StringRef funcName;
    FunctionType signature;
    bool anyHadConflictingSignature = false;
    bool found = false;
  };

  explicit DuplicateFuncCleaner(std::initializer_list<Entry> init)
      : entries(init) {
    llvm::sort(entries, [](const Entry &lhs, const Entry &rhs) {
      return lhs.funcName < rhs.funcName;
    });
  }

  LogicalResult run(ModuleOp module) {
    std::vector<LLVM::LLVMFuncOp> toRemove;
    const auto getEntry = [this](LLVM::LLVMFuncOp func) {
      auto iter = llvm::lower_bound(entries, func.getName(),
                                    [](const Entry &entry, StringRef name) {
                                      return name < entry.funcName;
                                    });
      if (iter == entries.end())
        return iter;
      return iter->funcName == func.getName() ? iter : entries.end();
    };
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      auto iter = getEntry(func);
      if (iter == entries.end())
        continue;
      // If the type is different, annotate it in the entry. This will lead to
      // an error only if more than one occurrence of the function is found.
      iter->anyHadConflictingSignature |=
          iter->signature != func.getFunctionType();
      // If it was already found, the current function should be removed.
      // Otherwise, set the function as already found.
      if (iter->found)
        toRemove.push_back(func);
      else
        iter->found = true;
    }
    for (LLVM::LLVMFuncOp func : toRemove) {
      auto iter = getEntry(func);
      assert(iter != entries.end() && "Should always be found");
      // If this is a duplicate and a signature difference occurred, signal
      // error.
      if (iter->anyHadConflictingSignature)
        return func.emitError()
               << "'" << func.getName()
               << "' defined with conflicting signature w.r.t. stdlib function";
      func.erase();
    }
    return success();
  }

private:
  std::vector<Entry> entries;
};

class ConvertPolygeistToLLVMPass
    : public impl::ConvertPolygeistToLLVMBase<ConvertPolygeistToLLVMPass> {
private:
  // During conversion, duplicate malloc and free functions can be generated if
  // the user calls either of these functions and also uses `new` while using
  // "non-generic" allocation functions. Remove potential duplicates here.
  void cleanDuplicateFunctions(unsigned bitwidth) {
    constexpr StringLiteral freeFn("free");
    constexpr StringLiteral mallocFn("malloc");
    Builder builder(&getContext());
    DuplicateFuncCleaner cleaner{
        {freeFn,
         builder.getFunctionType(builder.getType<LLVM::LLVMPointerType>(), {})},
        {mallocFn,
         builder.getFunctionType(builder.getIntegerType(bitwidth),
                                 builder.getType<LLVM::LLVMPointerType>())}};
    if (failed(cleaner.run(getOperation())))
      signalPassFailure();
  }

public:
  using impl::ConvertPolygeistToLLVMBase<
      ConvertPolygeistToLLVMPass>::ConvertPolygeistToLLVMBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = true;
    options.useOpaquePointers = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    options.dataLayout = llvm::DataLayout(this->dataLayout);

    for (int i = 0; i < 2; i++) {

      LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
      RewritePatternSet patterns(&getContext());
      // Keep these at the top; these should be run before the rest of
      // function conversion patterns.
      populateReturnOpTypeConversionPattern(patterns, converter);
      populateCallOpTypeConversionPattern(patterns, converter);
      populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

      constexpr auto clientAPI = spirv::ClientAPI::OpenCL;

      populateSPIRVToLLVMConversionPatterns(converter, patterns, clientAPI);
      populateSPIRVToLLVMTypeConversion(converter, clientAPI);

      populateSYCLToLLVMConversionPatterns(syclImplementation, syclTarget,
                                           converter, patterns);
      populateSYCLToMathConversionPatterns(patterns);
      populateSYCLToSPIRVConversionPatterns(converter, patterns);
      populatePolygeistToLLVMConversionPatterns(converter, patterns);
      populateSCFToControlFlowConversionPatterns(patterns);
      cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
      populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
      populateFuncToLLVMConversionPatterns(converter, patterns);
      populateMathToLLVMConversionPatterns(converter, patterns);
      populateOpenMPToLLVMConversionPatterns(converter, patterns);
      arith::populateArithToLLVMConversionPatterns(converter, patterns);
      populateVectorToLLVMConversionPatterns(converter, patterns);

      converter.addConversion([&](async::TokenType type) { return type; });

      patterns
          .add<LLVMOpLowering, GlobalOpTypeConversion, ReturnOpTypeConversion>(
              converter);

      // TODO: This is a temporary solution. In the future, we might want to
      // handle GPUDialect lowering by extending the GpuToLLVMConversionPass.
      patterns.add<GPUFuncOpToFuncOpConversion, GPUModuleOpToModuleOpConversion,
                   GPUReturnOpLowering, GPUModuleEndOpLowering>(converter);

      // cgeist already introduces these globals, so we can drop the ones coming
      // from the sycl to llvm conversion patterns. Add with a higher benefit so
      // that this is applied before the conversion to llvm pattern.
      patterns.add<EraseSPIRVBuiltinPattern>(&getContext(), /*benefit=*/2);

      // Run these instead of the ones provided by the dialect to avoid lowering
      // memrefs to a struct.
      populateBareMemRefToLLVMConversionPatterns(converter, patterns);

      // Legality callback for operations that checks whether their operand and
      // results types are converted.
      auto areAllTypesConverted = [&](Operation *op) -> Optional<bool> {
        SmallVector<Type> convertedResultTypes;
        if (failed(converter.convertTypes(op->getResultTypes(),
                                          convertedResultTypes)))
          return std::nullopt;
        SmallVector<Type> convertedOperandTypes;
        if (failed(converter.convertTypes(op->getOperandTypes(),
                                          convertedOperandTypes)))
          return std::nullopt;

        // With opaque pointers, type attributes also might need to be
        // translated to the corresponding LLVM types, for example the element
        // type attribute of GEP or alloca.
        std::optional<bool> noTyAttrConversion = std::transform_reduce(
            op->getAttrs().begin(), op->getAttrs().end(),
            std::optional<bool>{true},
            [](std::optional<bool> b1,
               std::optional<bool> b2) -> std::optional<bool> {
              if (!b1.has_value() || !b2.has_value())
                return std::nullopt;

              return b1.value() && b2.value();
            },
            [&](const NamedAttribute &Attr) -> std::optional<bool> {
              if (auto TyAttr = dyn_cast<TypeAttr>(Attr.getValue())) {
                auto ConvTy = converter.convertType(TyAttr.getValue());
                if (!ConvTy) {
                  return std::nullopt;
                }
                return ConvTy == TyAttr.getValue();
              }
              return true;
            });

        // Type conversion of at least one type attribute failed.
        if (!noTyAttrConversion) {
          return std::nullopt;
        }

        return convertedResultTypes == op->getResultTypes() &&
               convertedOperandTypes == op->getOperandTypes() &&
               noTyAttrConversion.value();
      };

      LLVMConversionTarget target(getContext());
      target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
          [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
      target.addIllegalDialect<gpu::GPUDialect>();
      target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                          scf::ExecuteRegionOp, func::FuncOp>();
      target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                        omp::YieldOp, omp::BarrierOp, omp::TaskwaitOp>();
      target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
          areAllTypesConverted);
      target.addDynamicallyLegalOp<LLVM::GlobalOp>(
          [&](LLVM::GlobalOp op) -> Optional<bool> {
            if (converter.convertType(op.getGlobalType()) == op.getGlobalType())
              return true;
            return std::nullopt;
          });
      target.addDynamicallyLegalOp<LLVM::ReturnOp>(
          [&](LLVM::ReturnOp op) -> Optional<bool> {
            // Outside global ops, defer to the normal type-based check. Note
            // that the infrastructure will not do it automatically because
            // per-op checks override dialect-level checks unconditionally.
            if (!isa<LLVM::GlobalOp>(op->getParentOp()))
              return areAllTypesConverted(op);

            SmallVector<Type> convertedOperandTypes;
            if (failed(converter.convertTypes(op->getOperandTypes(),
                                              convertedOperandTypes)))
              return std::nullopt;
            return convertedOperandTypes == op->getOperandTypes();
          });

      if (i == 1) {
        patterns.add<AsyncOpLowering>(converter);
        patterns.add<StreamToTokenOpLowering>(converter);
      }
      if (failed(applyPartialConversion(m, target, std::move(patterns))))
        signalPassFailure();

      LLVM_DEBUG(llvm::dbgs() << "ConvertPolygeistToLLVMPass: Module after:\n";
                 m->dump(); llvm::dbgs() << "\n";);
    }

    if (!options.useGenericFunctions)
      cleanDuplicateFunctions(options.getIndexBitwidth());
  }
};
} // namespace
