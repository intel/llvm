//===----------- TestIDAndRangeAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestIDAndRangeAnalysisPass
    : public PassWrapper<TestIDAndRangeAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIDAndRangeAnalysisPass)

  StringRef getArgument() const override { return "test-id-range-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool relaxedAliasing = true;
    auto &IDRangeAnalysis =
        getAnalysis<sycl::SYCLIDAndRangeAnalysis>().initialize(relaxedAliasing);

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return WalkResult::skip();

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs().indent(2) << "operand #" << index << "\n";
        auto idResult =
            IDRangeAnalysis.getIDRangeInformationFromConstruction<sycl::IDType>(
                op, operand);
        if (idResult) {
          llvm::errs().indent(2) << "id:\n";
          llvm::errs().indent(4) << *idResult << "\n";
        }
        auto rangeResult =
            IDRangeAnalysis
                .getIDRangeInformationFromConstruction<sycl::RangeType>(
                    op, operand);
        if (rangeResult) {
          llvm::errs().indent(2) << "range:\n";
          llvm::errs().indent(4) << *rangeResult << "\n";
        }
      }
      return WalkResult::advance();
    });
  };
};

} // namespace

namespace mlir {
namespace test {
void registerTestIDAndRangeAnalysisPass() {
  PassRegistration<TestIDAndRangeAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
