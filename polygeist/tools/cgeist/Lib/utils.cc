//===- utils.cc -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "clang-mlir.h"

#include "clang/AST/Expr.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlirclang {

Operation *buildLinalgOp(llvm::StringRef Name, OpBuilder &B,
                         llvm::SmallVectorImpl<Value> &Input,
                         llvm::SmallVectorImpl<Value> &Output) {
  if (Name.compare("memref.copy") == 0) {
    assert(Input.size() == 1 && "memref::copyOp requires 1 input");
    assert(Output.size() == 1 && "memref::CopyOp requires 1 output");
    return B.create<memref::CopyOp>(B.getUnknownLoc(), Input[0], Output[0]);
  }

  llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + Name);
  return nullptr;
}

Operation *replaceFuncByOperation(func::FuncOp F, llvm::StringRef OpName,
                                  OpBuilder &B,
                                  llvm::SmallVectorImpl<Value> &Input,
                                  llvm::SmallVectorImpl<Value> &Output) {
  MLIRContext *Ctx = F->getContext();
  assert(Ctx->isOperationRegistered(OpName) &&
         "Provided lower_to opName should be registered.");

  if (OpName.startswith("memref"))
    return buildLinalgOp(OpName, B, Input, Output);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState OpState(B.getUnknownLoc(), OpName, Input,
                         F.getCallableResults(), {});
  return B.create(OpState);
}

NamespaceKind getNamespaceKind(const clang::DeclContext *DC) {
  if (!DC)
    return NamespaceKind::Other;

  if (const auto *ND = dyn_cast<clang::NamespaceDecl>(DC)) {
    if (const auto *II = ND->getIdentifier()) {
      if (II->isStr("sycl"))
        return NamespaceKind::SYCL;
    }
  }

  if (DC->getParent() &&
      getNamespaceKind(DC->getParent()) != NamespaceKind::Other)
    return NamespaceKind::WithinSYCL;

  return NamespaceKind::Other;
}

FunctionContext getInputContext(const OpBuilder &B) {
  return B.getInsertionBlock()
                 ->getParentOp()
                 ->getParentOfType<gpu::GPUModuleOp>()
             ? FunctionContext::SYCLDevice
             : FunctionContext::Host;
}

gpu::GPUModuleOp getDeviceModule(ModuleOp Module) {
  return cast<gpu::GPUModuleOp>(
      Module.lookupSymbol(MLIRASTConsumer::DeviceModuleName));
}

FunctionContext getFuncContext(FunctionOpInterface Function) {
  return isa<mlir::gpu::GPUModuleOp>(Function->getParentOp())
             ? FunctionContext::SYCLDevice
             : FunctionContext::Host;
}

void setInsertionPoint(OpBuilder &Builder, FunctionContext FuncContext,
                       ModuleOp Module) {
  switch (FuncContext) {
  case FunctionContext::SYCLDevice:
    Builder.setInsertionPointToStart(
        mlirclang::getDeviceModule(Module).getBody());
    break;
  case FunctionContext::Host:
    Builder.setInsertionPointToStart(Module.getBody());
    break;
  }
}

} // namespace mlirclang
