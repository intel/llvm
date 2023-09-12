//===------------- TestAccessorAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestAccessorAnalysisPass
    : public PassWrapper<TestAccessorAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAccessorAnalysisPass)

  StringRef getArgument() const override { return "test-accessor-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool relaxedAliasing = true;
    AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(
        sycl::AliasAnalysis(relaxedAliasing));
    auto &AccessorAnalysis =
        getAnalysis<sycl::SYCLAccessorAnalysis>().initialize(relaxedAliasing);

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return WalkResult::skip();

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        llvm::errs().indent(2) << "operand #" << index << "\n";
        auto accResult =
            AccessorAnalysis.getAccessorInformationFromConstruction(op,
                                                                    operand);
        if (accResult) {
          llvm::errs().indent(2) << "accessor:\n";
          llvm::errs() << *accResult;

          // Check aliasing with other operands.
          for (size_t otherIdx = index + 1; otherIdx < op->getNumOperands();
               ++otherIdx) {
            auto otherResult =
                AccessorAnalysis.getAccessorInformationFromConstruction(
                    op, op->getOperand(otherIdx));

            if (otherResult) {
              auto aliasing = accResult->alias(*otherResult, aliasAnalysis);
              llvm::errs().indent(2) << "Alias (op#" << index << " x op#"
                                     << otherIdx << "): " << aliasing << "\n";
            }
          }
        }
      }
      return WalkResult::advance();
    });
  };
};

} // namespace

namespace mlir {
namespace test {
void registerTestAccessorAnalysisPass() {
  PassRegistration<TestAccessorAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
