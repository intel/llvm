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
// Helper
//===----------------------------------------------------------------------===//

namespace {

bool isConstructor(CallOpInterface call, LLVM::AllocaOp alloc) {
  if (call->getOperand(0) != alloc) {
    // Allocation ('this*') must be first argument for constructor call.
    return false;
  }

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

bool anyUsesBetween(LLVM::AllocaOp alloc, Operation *end) {
  assert(alloc->getBlock() && end->getBlock() &&
         "Expecting operations to be aprt of a block");

  if (alloc->getBlock() != end->getBlock())
    return true;

  llvm::SmallPtrSet<Operation *, 8> uses(alloc->user_begin(),
                                         alloc->user_end());

  bool started = false;
  for (auto &op : *alloc->getBlock()) {
    if (&op == alloc)
      started = true;

    if (started) {
      if (&op == end)
        return false;

      if (uses.contains(&op))
        return true;
    }
  }
  llvm_unreachable("'end' operation before allocation");
}

sycl::BufferType getBufferTypeFromConstructor(CallOpInterface constructor) {
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
    std::stoul(matches[1].str());

  // FIXME: There's currently no good way to obtain the element type of the
  // buffer from the constructor call (or allocation). Parsing it from the
  // demangled name, as done for 'dimensions' above, would require translation
  // from C++ types to MLIR types, which is not available here.
  Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

  return sycl::BufferType::get(constructor->getContext(), elemTy, dimensions);
}

} // namespace

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

class BufferConstructorPattern : public OpRewritePattern<LLVM::InvokeOp> {

public:
  using OpRewritePattern<LLVM::InvokeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::InvokeOp invoke,
                                PatternRewriter &rewriter) const final {

    if (invoke.getArgOperands().empty())
      return failure();

    // 'this*' is the first argument to the constructor call, if it is a
    // constructor.
    auto alloc = dyn_cast_or_null<LLVM::AllocaOp>(
        invoke.getArgOperands().front().getDefiningOp());

    if (!alloc)
      return failure();

    assert(alloc.getElemType().has_value() &&
           "Expecting element type attribute for opaque alloca");

    auto allocTy = *alloc.getElemType();
    auto structAllocTy = dyn_cast<LLVM::LLVMStructType>(allocTy);
    if (!structAllocTy || structAllocTy.getName() != "class.sycl::_V1::buffer")
      return failure();

    if (invoke.getNumResults())
      // Constructor should not return anything.
      return failure();

    if (!isConstructor(invoke, alloc))
      // Invoke is not a constructor call.
      return failure();

    rewriter.create<sycl::SYCLHostConstructorOp>(
        invoke->getLoc(), invoke.getArgOperands().front(),
        invoke.getArgOperands().drop_front(1),
        TypeAttr::get(getBufferTypeFromConstructor(invoke)));

    rewriter.create<LLVM::BrOp>(invoke->getLoc(),
                                invoke.getNormalDestOperands(),
                                invoke.getNormalDest());
    rewriter.eraseOp(invoke);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SYCLRaiseHostConstructsPass
//===----------------------------------------------------------------------===//

void SYCLRaiseHostConstructsPass::runOnOperation() {
  Operation *scopeOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet rewritePatterns{context};
  rewritePatterns.add<RaiseKernelName, BufferConstructorPattern>(context);
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
