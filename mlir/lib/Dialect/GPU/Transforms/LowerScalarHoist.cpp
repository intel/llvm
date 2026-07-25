//===- LowerScalarHoist.cpp - Lower scalar_hoist dialect to standard ops ---===//
//
// Inlines scalar_hoist.precompute regions into the parent block and erases
// the wrapper ops, leaving only standard arith/math operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/ScalarHoistDialect.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERSCALARHOISTPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::scalar_hoist::ScalarHoistDialect)

using namespace mlir;

namespace {

struct LowerScalarHoistPass
    : public impl::LowerScalarHoistPassBase<LowerScalarHoistPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<Operation *> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "scalar_hoist.precompute")
        return;

      Region &body = op->getRegion(0);
      if (body.empty())
        return;
      Block &bodyBlock = body.front();

      // Replace region block args with the precompute op's operands
      for (auto [bodyArg, operand] :
           llvm::zip(bodyBlock.getArguments(), op->getOperands()))
        bodyArg.replaceAllUsesWith(operand);

      // Find the yield terminator
      Operation *yieldOp = bodyBlock.getTerminator();

      // Replace precompute results with yield operands
      for (auto [result, yieldVal] :
           llvm::zip(op->getResults(), yieldOp->getOperands()))
        result.replaceAllUsesWith(yieldVal);

      // Move body ops (except yield) before the precompute op
      Block *parentBlock = op->getBlock();
      auto insertPt = Block::iterator(op);
      for (auto &bodyOp :
           llvm::make_early_inc_range(bodyBlock.getOperations())) {
        if (&bodyOp == yieldOp)
          continue;
        bodyOp.moveBefore(parentBlock, insertPt);
      }

      // Erase yield and mark precompute for erasure
      yieldOp->erase();
      toErase.push_back(op);
    });

    for (auto *op : toErase)
      op->erase();
  }
};

} // namespace
