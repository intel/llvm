//===- Utils.cpp - Polygeist Dialect Utilities  ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Utils/Utils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
static bool isNonBareConvertibleMemref(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<MemRefType>(std::not_fn(canBeLoweredToBarePtr))
      .Default(false);
}

static constexpr bool linkageCouldBreakABI(LLVM::Linkage linkage) {
  return linkage != LLVM::Linkage::Private &&
         linkage != LLVM::Linkage::Internal;
}

static bool couldBreakABI(SymbolOpInterface symbol) {
  return TypeSwitch<Operation *, bool>(symbol)
      .Case<FunctionOpInterface>([](auto func) {
        // We use llvm.linkage to specify linkage.
        static constexpr StringLiteral linkageAttrName = "llvm.linkage";
        // Default "external" value for linkage;
        LLVM::Linkage linkage = LLVM::Linkage::External;
        if (auto linkageAttr = dyn_cast_or_null<LLVM::LinkageAttr>(
                func->getAttr(linkageAttrName)))
          linkage = linkageAttr.getLinkage();
        if (!linkageCouldBreakABI(linkage))
          return false;
        return llvm::any_of(func.getArgumentTypes(),
                            isNonBareConvertibleMemref) ||
               llvm::any_of(func.getResultTypes(), isNonBareConvertibleMemref);
      })
      .Case<memref::GlobalOp>([](auto globalOp) {
        if (static_cast<SymbolOpInterface>(globalOp).isPrivate())
          return false;
        return isNonBareConvertibleMemref(globalOp.getType());
      })
      .Case<LLVM::GlobalOp>([](auto globalOp) {
        if (!linkageCouldBreakABI(globalOp.getLinkage()))
          return false;
        return isNonBareConvertibleMemref(globalOp.getGlobalType());
      })
      // Safe operations, no types involved
      .Case<gpu::GPUModuleOp, LLVM::ComdatSelectorOp, LLVM::ComdatOp>(
          [](auto) { return false; })
      // Safely assume operations can break ABI by default.
      .Default(true);
}

LogicalResult verifyABI(ModuleOp module) {
  return failure(module
                     .walk([](SymbolOpInterface symbol) -> WalkResult {
                       if (couldBreakABI(symbol))
                         return symbol->emitOpError()
                                << "could break ABI when converting to LLVM";
                       return WalkResult::advance();
                     })
                     .wasInterrupted());
}
} // namespace mlir
