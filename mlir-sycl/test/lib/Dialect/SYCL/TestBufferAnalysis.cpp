//===--------------- TestBufferAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLBufferAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestBufferAnalysisPass
    : public PassWrapper<TestBufferAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferAnalysisPass)

  StringRef getArgument() const override { return "test-buffer-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool relaxedAliasing = true;
    auto &IDRangeAnalysis =
        getAnalysis<sycl::SYCLBufferAnalysis>().initialize(relaxedAliasing);

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return WalkResult::skip();

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs().indent(2) << "operand #" << index << "\n";
        auto result =
            IDRangeAnalysis.getBufferInformationFromConstruction(op, operand);
        if (result) {
          llvm::errs().indent(2) << "buffer:\n";
          llvm::errs() << *result;
        }
      }
      return WalkResult::advance();
    });
  };
};

} // namespace

namespace mlir {
namespace test {
void registerTestBufferAnalysisPass() {
  PassRegistration<TestBufferAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
