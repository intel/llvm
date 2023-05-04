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
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

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
    solver.load<polygeist::ReachingDefinitionAnalysis>(aliasAnalysis);

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    auto printOps = [](std::optional<ArrayRef<Operation *>> ops,
                       StringRef title) {
      if (!ops) {
        llvm::errs() << title << "<unknown>\n";
        return;
      }

      llvm::errs() << title;
      llvm::interleave(
          *ops, llvm::errs(),
          [](Operation *op) {
            if (auto tagName = op->getAttrOfType<StringAttr>("tag_name"))
              llvm::errs() << tagName.getValue();
            else
              llvm::errs() << "'" << *op << "'";
          },
          " ");
      llvm::errs() << "\n";
    };

    op->walk([&](Operation *op) {
      // Only operations with the "tag" attribute are interesting.
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      const polygeist::ReachingDefinition *reachingDef =
          solver.lookupState<polygeist::ReachingDefinition>(op);
      assert(reachingDef && "expected a reaching definition");

      // Print the reaching definitions for each operand.
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs() << " operand #" << index << "\n";
        std::optional<ArrayRef<Operation *>> mods =
            reachingDef->getModifiers(operand);
        std::optional<ArrayRef<Operation *>> pMods =
            reachingDef->getPotentialModifiers(operand);
        printOps(mods, " - mods: ");
        printOps(pMods, " - pMods: ");
      }
    });
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
