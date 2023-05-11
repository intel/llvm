//===- TestMemoryAcessAnalysis.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

struct TestMemoryAccessAnalysisPass
    : public PassWrapper<TestMemoryAccessAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemoryAccessAnalysisPass)

  StringRef getArgument() const override { return "test-memory-access"; }

  StringRef getDescription() const final {
    return "Print result of the memory access analysis.";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    ModuleAnalysisManager mam(op, /*passInstrumentor=*/nullptr);
    AnalysisManager am = mam;
    auto &memAccessAnalysis = am.getAnalysis<MemoryAccessAnalysis>();

    op->walk([&](Operation *op) {
      // Only operations with the "tag" attribute are interesting.
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;

      assert(
          (isa<affine::AffineLoadOp>(op) || isa<affine::AffineStoreOp>(op)) &&
          "expecting affine load/store operation");

      llvm::errs() << "test_tag: " << tag.getValue() << ":\n";

      affine::MemRefAccess access(op);
      const std::optional<MemoryAccessMatrix> matrix =
          memAccessAnalysis.getMemoryAccessMatrix(access);
      assert(matrix.has_value() && "expected a m,emory access matrix");

      llvm::errs() << "matrix:\n" << *matrix << "\n";
    });
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
