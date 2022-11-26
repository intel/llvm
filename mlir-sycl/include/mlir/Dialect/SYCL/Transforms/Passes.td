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

def AlwaysInlinePass : Pass<"sycl-always-inline"> {
  let summary = "Inline SYCL calls to functions marked 'alwaysinline'";
  let description = [{
    Replace a sycl.call operation with the body of the callee if the callee has 
    the 'always_inline' attribute. 
  }];  
  let constructor = "mlir::sycl::createAlwaysInlinePass()";
}

def InlinePass : Pass<"sycl-inline"> {
  let summary = "Inline SYCL calls";
  let description = [{
    Replace a sylc.call operation with the body of the callee using simple heuristics.
  }];  
  let constructor = "mlir::sycl::createInlinePass()";
}

def SYCLMethodToSYCLCall : Pass<"sycl-method-to-sycl-call", "ModuleOp"> {
  let summary = "Convert SYCLMethodOpInterface instances to SYCLCallOps";
  let description = [{
    Replace SYCLMethodOpInterface instances by SYCLCallOps.
  }];
  let constructor = "mlir::sycl::createSYCLMethodToSYCLCallPass()";
}

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
