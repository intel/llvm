//===-- Passes.td - SYCL pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
#define MLIR_DIALECT_SYCL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SYCLMethodToSYCLCall : Pass<"sycl-method-to-sycl-call", "ModuleOp"> {
  let summary = "Convert SYCLMethodOpInterface instances to SYCLCallOps";
  let description = [{
    Replace SYCLMethodOpInterface instances by SYCLCallOps.
  }];
  let constructor = "mlir::sycl::createSYCLMethodToSYCLCallPass()";
}

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
