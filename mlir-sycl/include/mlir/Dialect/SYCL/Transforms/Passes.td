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

def ConstantPropagationPass
    : Pass<"sycl-constant-propagation", "::mlir::ModuleOp"> {
  let summary = "SYCL host-device constant propagation";
  let description = [{
    This pass implements SYCL host-device constant propagation. In order to do
    so, it tracks constant values, e.g., those defined by `arith.constant`,
    passed as an argument to `sycl.host.schedule_kernel`, and replaces the
    corresponding `gpu.func` argument with the constant value.

    This pass is also capable of propagating constant members of
    `sycl::accessor` arguments, i.e., offset and range.

    No further change is needed to the `sycl.host.schedule_kernel` operation or
    the kernel, as the Dead Argument Elimination SYCL LLVM pass (`DAESYCL`) will
    remove the kernel arguments that become unused as the result of this pass.
  }];
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool",
            /*default=*/"false",
            "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
  ];
  let constructor = "mlir::sycl::createConstantPropagationPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect"];

  let statistics = [
    Statistic<"NumPropagatedConstants", "num-propagated-constants",
              "Number of propagated constants">,
    Statistic<"NumReplacedImplicitArguments", "num-replaced-implicit-args",
              "Number of replaced implicit arguments">,
    Statistic<"NumReplacedExplicitArguments", "num-replaced-explicit-args",
              "Number of replaced explicit arguments">
  ];
}

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
    Option<"MaxIterationCount", "max-num-iters", "unsigned", /*default=*/"3",
           "Maximum number of inlining iterations for each SCC">,
  ];
  let dependentDialects = ["memref::MemRefDialect"];

  let statistics = [
    Statistic<"NumInlinedCalls", "num-inlined-calls", "Number of inlined calls">
  ];
}

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
