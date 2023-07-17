//===- TestUniformityAnalysis.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/UniformityAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::polygeist;

namespace {

struct TestUniformityAnalysisPass
    : public PassWrapper<TestUniformityAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUniformityAnalysisPass)

  StringRef getArgument() const final { return "test-uniformity"; }
  StringRef getDescription() const final {
    return "Test the uniformity analysis";
  }

  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(
        sycl::AliasAnalysis(false /* relaxedAliasing*/));

    DataFlowSolverWrapper solver(aliasAnalysis);
    solver.loadWithRequiredAnalysis<UniformityAnalysis>();

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;

      const UniformityLattice *uniformity =
          solver.lookupState<UniformityLattice>(op->getResult(0));
      assert(uniformity && "expected uniformity information");
      assert(!uniformity->getValue().isUninitialized() &&
             "lattice element should be initialized");

      llvm::errs() << tag.getValue() << ", uniformity: " << *uniformity << "\n";
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestUniformityAnalysisPass() {
  PassRegistration<TestUniformityAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
