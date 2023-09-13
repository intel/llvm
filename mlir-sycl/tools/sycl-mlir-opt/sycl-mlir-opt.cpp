//===- sycl-mlir-opt.cpp - SYCL MLIR Optimizer Driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for sycl-mlir-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SYCLPasses.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
namespace test {
void registerTestAccessorAnalysisPass();
void registerTestBufferAnalysisPass();
void registerTestIDAndRangeAnalysisPass();
void registerTestNDRangeAnalysisPass();
} // namespace test
} // namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  mlir::test::registerTestAccessorAnalysisPass();
  mlir::test::registerTestBufferAnalysisPass();
  mlir::test::registerTestIDAndRangeAnalysisPass();
  mlir::test::registerTestNDRangeAnalysisPass();
}
#endif

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<sycl::SYCLDialect>();
  mlir::func::registerInlinerExtension(registry);

  // Register passes.
  registerAllPasses();
  sycl::registerSYCLPasses();
  sycl::registerConversionPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SYCL MLIR optimizer driver\n", registry));
}
