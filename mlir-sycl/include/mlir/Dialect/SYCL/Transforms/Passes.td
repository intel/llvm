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

def InlinePass : Pass<"inliner"> {
  let summary = "Inline function calls";
  let description = [{
    Replace a function call operation with the body of the callee.
  }];
  let constructor = "mlir::sycl::createInlinePass()";
  let options = [
    Option<"InlineMode", "mode", "enum InlineMode", 
      "mlir::sycl::InlineMode::Simple", "Inline mode to attempt", 
      "llvm::cl::values("
        "clEnumValN(mlir::sycl::InlineMode::AlwaysInline, "
          "\"alwaysinline\", \"Inline callees with 'alwaysinline' attribute\"), "
        "clEnumValN(mlir::sycl::InlineMode::Simple, "
          "\"simple\", \"Inline callees using simple heuristics\"), "
        "clEnumValN(mlir::sycl::InlineMode::Aggressive, "
          "\"aggressive\", \"Inline callees using aggressive heuristics\"))">,
    Option<"RemoveDeadCallees", "remove-dead-callees", "bool", /*default=*/"true",
           "Remove callees that become unreachable after inlining them">,
    Option<"MaxIterationCount", "max-num-iters", "bool", /*default=*/"3",
           "Maximum number of inlining iterations for each SCC">,
  ];

  let statistics = [
    Statistic<"NumInlinedCalls", "num-inlined-calls", "Number of inlined calls">
  ];
}

def SYCLMethodToSYCLCall : Pass<"sycl-method-to-sycl-call", "ModuleOp"> {
  let summary = "Convert SYCLMethodOpInterface instances to SYCLCallOps";
  let description = [{
    Replace SYCLMethodOpInterface instances by SYCLCallOps.
  }];
  let constructor = "mlir::sycl::createSYCLMethodToSYCLCallPass()";
}

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
