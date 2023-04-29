//===- TestReachingDefinitionsAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

struct TestReachingDefinitionAnalysisPass
    : public PassWrapper<TestReachingDefinitionAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestReachingDefinitionAnalysisPass)

  StringRef getArgument() const override { return "test-reaching-definition"; }

  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(
        sycl::AliasAnalysis(false /* relaxedAliasing*/));

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<sycl::UnderlyingValueAnalysis>(aliasAnalysis);
    solver.load<sycl::ReachingDefinitionAnalysis>(aliasAnalysis);

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

#if 1
    op->walk([&](Operation *op) {
      // Only operations with the "tag" attribute are interesting.
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      const sycl::ReachingDefinition *reachingDef =
          solver.lookupState<sycl::ReachingDefinition>(op);
      assert(reachingDef && "expected a reaching definition");

      // Print the reaching definitions for each operand.
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs() << " operand #" << index << "\n";
        Value val = sycl::UnderlyingValue::getUnderlyingValue(
            operand,
            [&](Value value) {
              return solver.lookupState<sycl::UnderlyingValueLattice>(value);
            },
            aliasAnalysis);
        assert(val && "expected an underlying value");

        if (std::optional<ArrayRef<Operation *>> lastMods =
                reachingDef->getModifiers(val)) {
          for (Operation *lastMod : *lastMods) {
            if (auto tagName = lastMod->getAttrOfType<StringAttr>("tag_name"))
              llvm::errs() << "  - " << tagName.getValue() << "\n";
            else
              llvm::errs() << "  - " << lastMod->getName() << "\n";
          }
        } else {
          llvm::errs() << "  - <unknown>\n";
        }
      }
    });
#endif
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestReachingDefinitionAnalysisPass() {
  PassRegistration<TestReachingDefinitionAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
