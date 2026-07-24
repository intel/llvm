//===- ScalarHoist.cpp - Hoist int division magic/shift from GPU to host --===//
//
// Identifies arith.divui/remui by scalar kernel args in gpu.func bodies.
// Hoists magic/shift precomputation to the host side (before gpu.launch_func).
// Adds magic/shift as new kernel arguments and replaces division in kernel.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DEF_GPUSCALARHOISTPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

// Trace back through index_cast to find the original block argument.
// Returns the gpu.func block argument or nullptr.
static BlockArgument getKernelArg(Value val) {
  auto arg = dyn_cast<BlockArgument>(val);
  if (!arg) return nullptr;
  return arg;
}

static bool isIntOrIndex(Type t) { return isa<IntegerType, IndexType>(t); }

namespace {

struct GpuScalarHoistPass
    : public impl::GpuScalarHoistPassBase<GpuScalarHoistPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<std::tuple<func::FuncOp, gpu::LaunchFuncOp, gpu::GPUFuncOp>> work;
    module.walk([&](gpu::LaunchFuncOp launch) {
      auto gm = module.lookupSymbol<gpu::GPUModuleOp>(launch.getKernelModuleName());
      if (!gm) return;
      auto gf = gm.lookupSymbol<gpu::GPUFuncOp>(launch.getKernelName());
      if (!gf) return;
      auto hf = launch->getParentOfType<func::FuncOp>();
      if (!hf) return;
      work.emplace_back(hf, launch, gf);
    });

    for (auto &[hostFunc, launch, gpuFunc] : work) {
      // Find divisors that are scalar kernel args and need hoisting
      DenseMap<unsigned, Operation *> divOps;
      DenseMap<unsigned, Operation *> remOps;

      gpuFunc.walk([&](arith::DivUIOp op) {
        BlockArgument arg = getKernelArg(op.getRhs());
        if (arg && isIntOrIndex(arg.getType()))
          divOps[arg.getArgNumber()] = op;
      });
      gpuFunc.walk([&](arith::RemUIOp op) {
        BlockArgument arg = getKernelArg(op.getRhs());
        if (arg && isIntOrIndex(arg.getType()))
          remOps[arg.getArgNumber()] = op;
      });

      // Combine: any divisor used in div or rem
      DenseSet<unsigned> hoistArgs;
      for (auto &kv : divOps) hoistArgs.insert(kv.first);
      for (auto &kv : remOps) hoistArgs.insert(kv.first);

      if (hoistArgs.empty()) return;

      // HOST side: compute magic/shift for each scalar divisor
      // Insert before the first gpu.launch_func (which is inside scf.for).
      // Need to insert BEFORE scf.for — find the insertion point:
      // last arith.constant or memref.store before the first scf.for.
      Block &entry = hostFunc.getBody().front();
      Operation *insertPt = &entry.front();
      for (auto &op : entry.getOperations()) {
        if (isa<scf::ForOp>(op)) break;
        insertPt = &op;
      }
      OpBuilder hostB(insertPt);
      Type i32Ty = hostB.getI32Type();
      Type i64Ty = hostB.getI64Type();
      Location loc = launch.getLoc();
      DenseMap<unsigned, std::pair<Value, Value>> magicShift; // argIdx → (magic, shift)

      for (unsigned argIdx : hoistArgs) {
        Value hostDivisor = launch.getKernelOperand(argIdx);
        if (!isIntOrIndex(hostDivisor.getType()))
          hostDivisor = hostB.create<arith::IndexCastUIOp>(loc, i32Ty, hostDivisor);

        // shift = 32 - ctlz(divisor - 1)
        Value one = hostB.create<arith::ConstantIntOp>(loc, 1, 32);
        Value dm1 = hostB.create<arith::SubIOp>(loc, hostDivisor, one);
        Value clz = hostB.create<math::CountLeadingZerosOp>(loc, dm1);
        Value c32 = hostB.create<arith::ConstantIntOp>(loc, 32, 32);
        Value shift = hostB.create<arith::SubIOp>(loc, c32, clz);

        // magic64 = ((1 << (32+shift)) / d + 1), adjusted
        Value d64 = hostB.create<arith::ExtUIOp>(loc, i64Ty, hostDivisor);
        Value s64 = hostB.create<arith::ExtUIOp>(loc, i64Ty, shift);
        Value c32_64 = hostB.create<arith::ConstantIntOp>(loc, 32, 64);
        Value ts = hostB.create<arith::AddIOp>(loc, s64, c32_64);
        Value one64 = hostB.create<arith::ConstantIntOp>(loc, 1, 64);
        Value p2 = hostB.create<arith::ShLIOp>(loc, one64, ts);
        Value m0 = hostB.create<arith::DivUIOp>(loc, p2, d64);
        Value m1 = hostB.create<arith::AddIOp>(loc, m0, one64);
        Value pr = hostB.create<arith::MulIOp>(loc, m1, d64);
        Value ov = hostB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, pr, p2);
        Value m64 = hostB.create<arith::SelectOp>(loc, ov, m0, m1);
        Value magic = hostB.create<arith::TruncIOp>(loc, i32Ty, m64);

        magicShift[argIdx] = {magic, shift};
      }

      // Add magic/shift as new kernel args + launch operands
      for (unsigned argIdx : hoistArgs) {
        auto [magic, shift] = magicShift[argIdx];
        gpuFunc.getBody().front().addArgument(i32Ty, loc);
        gpuFunc.getBody().front().addArgument(i32Ty, loc);
        launch.getKernelOperandsMutable().append(magic);
        launch.getKernelOperandsMutable().append(shift);
      }

      // Update ALL gpu.launch_func ops calling this kernel
      auto gpuMod = launch->getParentOfType<ModuleOp>().lookupSymbol<gpu::GPUModuleOp>(
          launch.getKernelModuleName());
      if (gpuMod) {
        hostFunc.walk([&](gpu::LaunchFuncOp otherLaunch) {
          if (otherLaunch.getKernelName() != launch.getKernelName()) return;
          if (otherLaunch == launch) return; // already updated
          for (unsigned argIdx : hoistArgs) {
            otherLaunch.getKernelOperandsMutable().append(magicShift[argIdx].first);
            otherLaunch.getKernelOperandsMutable().append(magicShift[argIdx].second);
          }
        });
      }

      // Update gpu.func function type
      SmallVector<Type> newArgTypes;
      for (auto arg : gpuFunc.getBody().getArguments())
        newArgTypes.push_back(arg.getType());
      auto newFnTy = FunctionType::get(gpuFunc.getContext(), newArgTypes, {});
      gpuFunc.setFunctionType(newFnTy);

      // KERNEL side: replace div/rem using new magic/shift block args
      unsigned oldNumArgs = gpuFunc.getNumArguments() - 2 * hoistArgs.size();
      SmallVector<unsigned, 8> hoistOrder(hoistArgs.begin(), hoistArgs.end());
      llvm::sort(hoistOrder);

      for (auto &[argIdx, divOp] : divOps) {
        auto dOp = cast<arith::DivUIOp>(divOp);
        unsigned pos = llvm::find(hoistOrder, argIdx) - hoistOrder.begin();
        BlockArgument magicArg = gpuFunc.getBody().getArgument(oldNumArgs + 2 * pos);
        BlockArgument shiftArg = gpuFunc.getBody().getArgument(oldNumArgs + 2 * pos + 1);

        OpBuilder b(dOp);
        Value n = dOp.getLhs();
        // mul_hi via explicit i64 multiply: (zext(magic)*zext(n)) >> 32
        auto i64Ty = b.getI64Type();
        Value m64 = b.create<arith::ExtUIOp>(loc, i64Ty, magicArg);
        Value n64 = b.create<arith::ExtUIOp>(loc, i64Ty, n);
        Value full = b.create<arith::MulIOp>(loc, m64, n64);
        Value c32 = b.create<arith::ConstantIntOp>(loc, 32, 64);
        Value hi = b.create<arith::ShRUIOp>(loc, full, c32);
        Value hi32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), hi);
        Value sum = b.create<arith::AddIOp>(loc, hi32, n);
        Value res = b.create<arith::ShRUIOp>(loc, sum, shiftArg);
        dOp->getResult(0).replaceAllUsesWith(res);
        dOp->erase();
      }

      for (auto &[argIdx, remOp] : remOps) {
        auto rOp = cast<arith::RemUIOp>(remOp);
        unsigned pos = llvm::find(hoistOrder, argIdx) - hoistOrder.begin();
        BlockArgument magicArg = gpuFunc.getBody().getArgument(oldNumArgs + 2 * pos);
        BlockArgument shiftArg = gpuFunc.getBody().getArgument(oldNumArgs + 2 * pos + 1);

        OpBuilder b(rOp);
        Value n2 = rOp.getLhs();
        auto i64Ty = b.getI64Type();
        Value m64_2 = b.create<arith::ExtUIOp>(loc, i64Ty, magicArg);
        Value n64_2 = b.create<arith::ExtUIOp>(loc, i64Ty, n2);
        Value full2 = b.create<arith::MulIOp>(loc, m64_2, n64_2);
        Value c32_2 = b.create<arith::ConstantIntOp>(loc, 32, 64);
        Value hi2 = b.create<arith::ShRUIOp>(loc, full2, c32_2);
        Value hi32_2 = b.create<arith::TruncIOp>(loc, b.getI32Type(), hi2);
        Value sum2 = b.create<arith::AddIOp>(loc, hi32_2, n2);
        Value quot = b.create<arith::ShRUIOp>(loc, sum2, shiftArg);
        Value prod = b.create<arith::MulIOp>(loc, quot, rOp.getRhs());
        Value res = b.create<arith::SubIOp>(loc, n2, prod);
        rOp->getResult(0).replaceAllUsesWith(res);
        rOp->erase();
      }
    }
  }
};
} // namespace