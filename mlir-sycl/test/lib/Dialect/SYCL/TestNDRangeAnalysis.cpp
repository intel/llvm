//===----------- TestNDRangeAnalysis.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestNDRangeAnalysisPass
    : public PassWrapper<TestNDRangeAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNDRangeAnalysisPass)

  StringRef getArgument() const override { return "test-nd-range-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool relaxedAliasing = true;
    auto &NDRangeAnalysis =
        getAnalysis<sycl::SYCLNDRangeAnalysis>().initialize(relaxedAliasing);

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return WalkResult::skip();

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs().indent(2) << "operand #" << index << "\n";
        auto ndrResult =
            NDRangeAnalysis.getNDRangeInformationFromConstruction(op, operand);
        if (ndrResult) {
          llvm::errs().indent(2) << "nd_range:\n";
          llvm::errs().indent(4) << *ndrResult << "\n";
        }
      }
      return WalkResult::advance();
    });
  };
};

} // namespace

namespace mlir {
namespace test {
void registerTestNDRangeAnalysisPass() {
  PassRegistration<TestNDRangeAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
