// Copyright (C) Codeplay Software Limited

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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace clang;

extern llvm::cl::opt<bool> SuppressWarnings;

Operation *buildLinalgOp(StringRef name, OpBuilder &b,
                         SmallVectorImpl<mlir::Value> &input,
                         SmallVectorImpl<mlir::Value> &output) {
  if (name.compare("memref.copy") == 0) {
    assert(input.size() == 1 && "memref::copyOp requires 1 input");
    assert(output.size() == 1 && "memref::CopyOp requires 1 output");
    return b.create<memref::CopyOp>(b.getUnknownLoc(), input[0], output[0]);
  } else {
    llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + name);
    return nullptr;
  }
}

Operation *mlirclang::replaceFuncByOperation(
    func::FuncOp f, StringRef opName, OpBuilder &b,
    SmallVectorImpl<mlir::Value> &input, SmallVectorImpl<mlir::Value> &output) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  if (opName.startswith("memref"))
    return buildLinalgOp(opName, b, input, output);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), opName, input,
                         f.getCallableResults(), {});
  return b.create(opState);
}

bool mlirclang::isNamespaceSYCL(const clang::DeclContext *DC) {
  if (!DC) {
    return false;
  }

  if (const auto *ND = dyn_cast<clang::NamespaceDecl>(DC)) {
    if (const auto *II = ND->getIdentifier()) {
      if (II->isStr("sycl")) {
        return true;
      }
    }
  }

  if (DC->getParent()) {
    return mlirclang::isNamespaceSYCL(DC->getParent());
  }

  return false;
}

FunctionContext mlirclang::getInputContext(const OpBuilder &Builder) {
  return Builder.getInsertionBlock()
                 ->getParentOp()
                 ->getParentOfType<gpu::GPUModuleOp>()
             ? FunctionContext::SYCLDevice
             : FunctionContext::Host;
}

mlir::gpu::GPUModuleOp mlirclang::getDeviceModule(mlir::ModuleOp Module) {
  return cast<mlir::gpu::GPUModuleOp>(
      Module.lookupSymbol(MLIRASTConsumer::DeviceModuleName));
}

llvm::raw_ostream &mlirclang::warning() {
  if (SuppressWarnings)
    return llvm::nulls();
  return llvm::WithColor::warning();
}