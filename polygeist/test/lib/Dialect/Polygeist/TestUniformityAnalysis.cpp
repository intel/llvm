//===- TestUniformityAnalysis.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
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
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<UniformityAnalysis>();

    Operation *op = getOperation();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    auto printUniformity = [](const Uniformity &info, raw_ostream &os) {
      switch (info.getKind()) {
      case Uniformity::Kind::Unknown:
        os << "unknown\n";
        break;
      case Uniformity::Kind::Uniform:
        os << "uniform\n";
        break;
      case Uniformity::Kind::NonUniform:
        os << "non-uniform\n";
        break;
      }
    };

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;

      const UniformityLattice *uniformity =
          solver.lookupState<UniformityLattice>(op->getResult(0));
      assert(uniformity && "expected uniformity information");
      assert(!uniformity->getValue().isUninitialized() &&
             "lattice element should be initialized");

      llvm::errs() << tag.getValue() << ", " << *uniformity << "\n";

      //      printUniformity(uniformity->getValue(), llvm::errs());
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
