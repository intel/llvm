//=== SYCLRaiseHost.cpp - Raise host constructs to SYCL dialect operations ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to detect instruction sequences of interest in the MLIR
// (mostly LLVM dialect) for the SYCL host side and raise them to types and
// operations from the SYCL dialect to facilitate analysis in other passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SYCLRAISEHOSTCONSTRUCTS
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {

class SYCLRaiseHostConstructsPass
    : public polygeist::impl::SYCLRaiseHostConstructsBase<
          SYCLRaiseHostConstructsPass> {
public:
  using polygeist::impl::SYCLRaiseHostConstructsBase<
      SYCLRaiseHostConstructsPass>::SYCLRaiseHostConstructsBase;

  void runOnOperation() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

namespace {
struct RaiseKernelName : public OpRewritePattern<LLVM::AddressOfOp> {
public:
  using OpRewritePattern<LLVM::AddressOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp op,
                                PatternRewriter &rewriter) const final {
    // Get the global this operation uses
    SymbolTableCollection symbolTable;
    auto global = symbolTable.lookupNearestSymbolFrom<LLVM::GlobalOp>(
        op, op.getGlobalNameAttr());
    if (!global)
      return failure();

    // Get a reference to the kernel the global references
    std::optional<SymbolRefAttr> ref = getKernelRef(global, symbolTable);
    if (!ref)
      return failure();

    rewriter.replaceOpWithNewOp<sycl::SYCLHostGetKernelOp>(op, op.getType(),
                                                           *ref);
    return success();
  }

private:
  /// If the input global contains a constant string representing the name of a
  /// SYCL kernel, returns a reference to the `gpu.func` implementing this
  /// kernel.
  ///
  /// The string will contain a trailing character we need to get rid of before
  /// searching.
  static std::optional<SymbolRefAttr>
  getKernelRef(LLVM::GlobalOp op, SymbolTableCollection &symbolTable) {
    // Check the operation has a value
    std::optional<Attribute> attr = op.getValue();
    if (!attr)
      return std::nullopt;

    // Check it is a string
    auto strAttr = dyn_cast<StringAttr>(*attr);
    if (!strAttr)
      return std::nullopt;

    // Drop the trailing `0` character
    StringRef name = strAttr.getValue().drop_back();

    // Search the `gpu.func` in the device module
    auto ref =
        SymbolRefAttr::get(op->getContext(), DeviceModuleName,
                           FlatSymbolRefAttr::get(op->getContext(), name));
    auto kernel = symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, ref);

    // If it was found and it is a kernel, return the reference
    return kernel && kernel.isKernel() ? std::optional<SymbolRefAttr>(ref)
                                       : std::nullopt;
  }
};

template <typename Derived, typename ConstructorOp, typename TypeTag,
          bool PostProcess = false>
class RaiseConstructorBasePattern : public OpRewritePattern<ConstructorOp> {
public:
  using OpRewritePattern<ConstructorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstructorOp constructor,
                                PatternRewriter &rewriter) const final {

    if (constructor.getArgOperands().empty())
      return failure();

    // 'this*' is the first argument to the constructor call, if it is a
    // constructor.
    auto alloc = dyn_cast_or_null<LLVM::AllocaOp>(
        constructor.getArgOperands().front().getDefiningOp());

    if (!alloc)
      return failure();

    assert(alloc.getElemType().has_value() &&
           "Expecting element type attribute for opaque alloca");

    // Check whether the type allocated for the first argument ('this') matches
    // the expected type.
    auto allocTy = *alloc.getElemType();
    auto structAllocTy = dyn_cast<LLVM::LLVMStructType>(allocTy);
    if (!structAllocTy || structAllocTy.getName() != TypeTag::getTypeName())
      return failure();

    if (constructor.getNumResults())
      // Constructor should not return anything.
      return failure();

    if (!isConstructor(constructor))
      // Invoke is not a constructor call.
      return failure();

    rewriter.create<sycl::SYCLHostConstructorOp>(
        constructor->getLoc(), constructor.getArgOperands().front(),
        constructor.getArgOperands().drop_front(1),
        TypeAttr::get(TypeTag::getTypeFromConstructor(constructor)));

    if constexpr (PostProcess)
      static_cast<const Derived *>(this)->postprocess(constructor, rewriter);

    rewriter.eraseOp(constructor);
    return success();
  }

private:
  bool isConstructor(ConstructorOp call) const {
    CallInterfaceCallable callableOp = call.getCallableForCallee();
    StringRef funcName = callableOp.get<SymbolRefAttr>().getLeafReference();

    llvm::ItaniumPartialDemangler Demangler;
    Demangler.partialDemangle(funcName.data());
    if (!Demangler.isCtorOrDtor())
      return false;

    char *demangled = Demangler.finishDemangle(nullptr, 0);
    if (!demangled)
      // Demangling failed
      return false;

    llvm::StringRef demangledName{demangled};
    bool isDestructor = demangledName.contains('~');
    free(demangled);
    return !isDestructor;
  }
};

template <typename TypeTag>
class RaiseInvokeConstructorBasePattern
    : public RaiseConstructorBasePattern<
          RaiseInvokeConstructorBasePattern<TypeTag>, LLVM::InvokeOp, TypeTag,
          true> {
public:
  using RaiseConstructorBasePattern<RaiseInvokeConstructorBasePattern<TypeTag>,
                                    LLVM::InvokeOp, TypeTag,
                                    true>::RaiseConstructorBasePattern;

  void postprocess(LLVM::InvokeOp invoke, PatternRewriter &rewriter) const {
    rewriter.create<LLVM::BrOp>(invoke->getLoc(),
                                invoke.getNormalDestOperands(),
                                invoke.getNormalDest());
  }
};

struct BufferTypeTag {

  static llvm::StringRef getTypeName() { return "class.sycl::_V1::buffer"; }

  static mlir::Type getTypeFromConstructor(CallOpInterface constructor) {
    CallInterfaceCallable callableOp = constructor.getCallableForCallee();
    StringRef constructorName =
        callableOp.get<SymbolRefAttr>().getLeafReference();

    auto demangledName = llvm::demangle(constructorName);

    // Try to determine the dimensions of the buffer by parsing the template
    // parameter from the demangled name of the constructor.
    llvm::Regex bufferTemplate("buffer<.*, ([0-9]+)");
    llvm::SmallVector<StringRef> matches;
    bool regexMatch = bufferTemplate.match(demangledName, &matches);
    unsigned dimensions = 1;
    if (regexMatch)
      dimensions = std::stoul(matches[1].str());

    // FIXME: There's currently no good way to obtain the element type of the
    // buffer from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for 'dimensions' above, would require translation
    // from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

    return sycl::BufferType::get(constructor->getContext(), elemTy, dimensions);
  }
};

struct BufferInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<BufferTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      BufferTypeTag>::RaiseInvokeConstructorBasePattern;
};

struct AccessorTypeTag {

  static llvm::StringRef getTypeName() { return "class.sycl::_V1::accessor"; }

  static mlir::Type getTypeFromConstructor(CallOpInterface constructor) {
    CallInterfaceCallable callableOp = constructor.getCallableForCallee();
    StringRef constructorName =
        callableOp.get<SymbolRefAttr>().getLeafReference();

    auto demangledName = llvm::demangle(constructorName);

    // Try to determine the parameters of the accessor by parsing the template
    // parameter from the demangled name of the constructor.
    llvm::Regex accessorTemplate(
        "accessor<.*, ([0-9]+), \\(sycl::_V1::access::mode\\)([0-9]+), "
        "\\(sycl::_V1::access::target\\)([0-9]+)");
    llvm::SmallVector<StringRef> matches;
    bool regexMatch = accessorTemplate.match(demangledName, &matches);
    unsigned dimensions = 1;
    unsigned target = 2014;
    unsigned mode = 1026;
    if (regexMatch) {
      dimensions = std::stoul(matches[1].str());
      mode = std::stoul(matches[2].str());
      target = std::stoul(matches[3].str());
    }

    sycl::AccessMode accessMode = static_cast<sycl::AccessMode>(mode);
    sycl::Target accessTarget = static_cast<sycl::Target>(target);

    // FIXME: There's currently no good way to obtain the element type of the
    // accessor from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for other parameters above, would require
    // translation from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

    return sycl::AccessorType::get(constructor.getContext(), elemTy, dimensions,
                                   accessMode, accessTarget, {});
  }
};

struct AccessorInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<AccessorTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      AccessorTypeTag>::RaiseInvokeConstructorBasePattern;
};

} // namespace

//===----------------------------------------------------------------------===//
// SYCLRaiseHostConstructsPass
//===----------------------------------------------------------------------===//

void SYCLRaiseHostConstructsPass::runOnOperation() {
  Operation *scopeOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet rewritePatterns{context};
  rewritePatterns.add<RaiseKernelName, BufferInvokeConstructorPattern,
                      AccessorInvokeConstructorPattern>(context);
  FrozenRewritePatternSet frozen(std::move(rewritePatterns));

  if (failed(applyPatternsAndFoldGreedily(scopeOp, frozen)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass() {
  return std::make_unique<SYCLRaiseHostConstructsPass>();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass(
    const polygeist::SYCLRaiseHostConstructsOptions &options) {
  return std::make_unique<SYCLRaiseHostConstructsPass>(options);
}
