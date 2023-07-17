//===- TestReachingDefinitionsAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::polygeist;

namespace {

struct TestReachingDefinitionAnalysisPass
    : public PassWrapper<TestReachingDefinitionAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestReachingDefinitionAnalysisPass)

  StringRef getArgument() const override { return "test-reaching-definition"; }

  void runOnOperation() override {
    using Definition = polygeist::Definition;
    using ModifiersTy = polygeist::ReachingDefinition::ModifiersTy;

    AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(
        sycl::AliasAnalysis(false /* relaxedAliasing*/));

    DataFlowSolverWrapper solver(aliasAnalysis);
    solver.loadWithRequiredAnalysis<ReachingDefinitionAnalysis>(aliasAnalysis);

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    auto printOps = [](std::optional<ModifiersTy> defs, StringRef title) {
      if (!defs) {
        llvm::errs() << title << "<unknown>\n";
        return;
      }

      llvm::errs() << title;
      llvm::interleave(
          *defs, llvm::errs(),
          [](const Definition &def) {
            if (!def.isOperation()) {
              llvm::errs() << def;
              return;
            }
            if (auto tagName =
                    def.getOperation()->getAttrOfType<StringAttr>("tag_name"))
              llvm::errs() << tagName.getValue();
            else
              llvm::errs() << "'" << def << "'";
          },
          " ");
      if (defs->empty())
        llvm::errs() << "<none>";
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
        auto mods = reachingDef->getModifiers(operand, solver);
        auto pMods = reachingDef->getPotentialModifiers(operand, solver);
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
