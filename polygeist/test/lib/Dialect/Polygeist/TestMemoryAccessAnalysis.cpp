//===- TestMemoryAcessAnalysis.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

struct TestMemoryAccessAnalysisPass
    : public PassWrapper<TestMemoryAccessAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemoryAccessAnalysisPass)

  StringRef getArgument() const override { return "test-memory-access"; }

  StringRef getDescription() const final {
    return "Print result of the memory access analysis.";
  }

  void runOnOperation() override {
#if 0
    AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(
        sycl::AliasAnalysis(false /* relaxedAliasing*/));

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<polygeist::ReachingDefinitionAnalysis>(aliasAnalysis);

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();
#endif

    Operation *op = getOperation();
    ModuleAnalysisManager mam(op, /*passInstrumentor=*/nullptr);
    AnalysisManager am = mam;

    polygeist::MemoryAccessAnalysis &memAccessAnalysis =
        am.getAnalysis<polygeist::MemoryAccessAnalysis>();

    llvm::errs() << "Testing : " << *op << "\n";
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMemoryAccessAnalysisPass() {
  PassRegistration<TestMemoryAccessAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
