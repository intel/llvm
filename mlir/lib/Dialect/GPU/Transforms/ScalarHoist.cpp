//===- ScalarHoist.cpp - Hoist int division magic/shift from GPU to host --===//
//
// Implements the patent's Phase 1 (Dependency Classification) and the
// cost-driven hoisting of scalar integer division.
//
// Phase 1: Forward dataflow analysis tags each SSA value in the GPU kernel
// as SCALAR_ONLY (uniform across work-items), INDEX_ONLY (varies per
// work-item), MIXED (depends on both), or CONSTANT (compile-time known).
//
// Hoisting: For each arith.divui / arith.remui where the divisor (or a
// scalar operand of a MIXED divisor) is SCALAR_ONLY, the pass:
//   - Precomputes magic/shift on the HOST side (before gpu.launch_func)
//   - Adds magic/shift as extra kernel arguments
//   - Replaces division in the kernel with magic multiply.
//
// The cost model is simple: all integer divisions by scalar values are
// hoisted (integer division is expensive, ~20-80 cycles on GPU vs ~6-8
// for the replacement).
//
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

//===----------------------------------------------------------------------===//
// Phase 1: Dependency Classification
//===----------------------------------------------------------------------===//

enum class DepClass : uint8_t { SCALAR_ONLY, INDEX_ONLY, MIXED, CONSTANT };

/// Classify every SSA value in the gpu.func body.
/// For OUTLINED kernels (gpu.module/gpu.func), the block arguments are the
/// actual kernel parameters (memrefs + scalars). GPU index values are
/// generated inside the body via gpu.thread_id / gpu.block_id ops.
static void classifyValues(gpu::GPUFuncOp kernel,
                           DenseMap<Value, DepClass> &depMap) {
  Block &entry = kernel.getBody().front();

  // For outlined kernels: all block args are kernel params.
  // MemRef → INDEX_ONLY (data buffers), scalars → SCALAR_ONLY.
  for (auto arg : entry.getArguments()) {
    Type t = arg.getType();
    if (isa<MemRefType>(t))
      depMap[arg] = DepClass::INDEX_ONLY;
    else
      depMap[arg] = DepClass::SCALAR_ONLY;
  }

  // --- Forward propagation: iterate to fixed point ---
  SmallVector<Operation *> allOps;
  kernel.walk([&](Operation *op) {
    if (!op->getResults().empty())
      allOps.push_back(op);
  });

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : allOps) {
      // GPU index queries → INDEX_ONLY
      if (isa<gpu::BlockIdOp, gpu::ThreadIdOp, gpu::LaneIdOp, gpu::GlobalIdOp,
              gpu::SubgroupIdOp>(op)) {
        for (Value r : op->getResults()) {
          if (depMap.lookup(r) != DepClass::INDEX_ONLY) {
            depMap[r] = DepClass::INDEX_ONLY;
            changed = true;
          }
        }
        continue;
      }

      // gpu.block_dim / gpu.grid_dim → SCALAR_ONLY (uniform per launch)
      if (isa<gpu::BlockDimOp, gpu::GridDimOp>(op)) {
        for (Value r : op->getResults()) {
          if (depMap.lookup(r) != DepClass::SCALAR_ONLY) {
            depMap[r] = DepClass::SCALAR_ONLY;
            changed = true;
          }
        }
        continue;
      }

      // Memory loads → INDEX_ONLY
      if (isa<memref::LoadOp>(op)) {
        for (Value r : op->getResults()) {
          if (depMap.lookup(r) != DepClass::INDEX_ONLY) {
            depMap[r] = DepClass::INDEX_ONLY;
            changed = true;
          }
        }
        continue;
      }

      // Only classify arithmetic / math operations
      if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
        continue;

      // --- Lattice join ---
      bool hasScalar = false, hasIndex = false;
      for (Value operand : op->getOperands()) {
        auto it = depMap.find(operand);
        if (it == depMap.end()) continue;
        DepClass dc = it->second;
        if (dc == DepClass::SCALAR_ONLY || dc == DepClass::CONSTANT)
          hasScalar = true;
        if (dc == DepClass::INDEX_ONLY || dc == DepClass::MIXED)
          hasIndex = true;
      }

      DepClass newClass;
      if (!hasScalar && !hasIndex)
        newClass = DepClass::CONSTANT;
      else if (hasScalar && !hasIndex)
        newClass = DepClass::SCALAR_ONLY;
      else if (!hasScalar && hasIndex)
        newClass = DepClass::INDEX_ONLY;
      else
        newClass = DepClass::MIXED;

      for (Value r : op->getResults()) {
        if (depMap.lookup(r) != newClass) {
          depMap[r] = newClass;
          changed = true;
        }
      }
    }
  }
}

/// Check whether a value is SCALAR_ONLY by looking it up in the depMap.
static bool isScalarOnly(Value val, const DenseMap<Value, DepClass> &depMap) {
  auto it = depMap.find(val);
  return it != depMap.end() && it->second == DepClass::SCALAR_ONLY;
}

/// Return the SCALAR_ONLY operand of a MIXED value if exactly one operand
/// is SCALAR_ONLY and the other is INDEX_ONLY. This is useful for MIXED
/// decomposition: arith.divui(n, scalar_op) where scalar_op may itself be
/// a SCALAR_ONLY computation chain (not necessarily a block argument).
/// Returns nullptr if no usable scalar operand.
static Value getScalarOperand(Value val,
                               const DenseMap<Value, DepClass> &depMap) {
  if (isScalarOnly(val, depMap))
    return val;                                    // directly SCALAR_ONLY

  // Check if this is a MIXED binary op with one SCALAR_ONLY operand
  Operation *defOp = val.getDefiningOp();
  if (!defOp || defOp->getNumOperands() < 1)
    return nullptr;

  for (Value operand : defOp->getOperands()) {
    if (isScalarOnly(operand, depMap))
      return operand;                              // found scalar operand
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Utility: check if type is integer or index (for magic/shift computation).
//===----------------------------------------------------------------------===//
static bool isIntOrIndex(Type t) { return isa<IntegerType, IndexType>(t); }

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//

namespace {

struct GpuScalarHoistPass
    : public impl::GpuScalarHoistPassBase<GpuScalarHoistPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<std::tuple<func::FuncOp, gpu::LaunchFuncOp, gpu::GPUFuncOp>>
        work;
    module.walk([&](gpu::LaunchFuncOp launch) {
      auto gm =
          module.lookupSymbol<gpu::GPUModuleOp>(launch.getKernelModuleName());
      if (!gm) return;
      auto gf = gm.lookupSymbol<gpu::GPUFuncOp>(launch.getKernelName());
      if (!gf) return;
      auto hf = launch->getParentOfType<func::FuncOp>();
      if (!hf) return;
      work.emplace_back(hf, launch, gf);
    });

    for (auto &[hostFunc, launch, gpuFunc] : work) {

      // ---- Phase 1: classify all values in the kernel ----
      DenseMap<Value, DepClass> depMap;
      classifyValues(gpuFunc, depMap);

      // ---- Phase 2: find divisions by SCALAR_ONLY operands ----
      struct HoistCandidate {
        Operation *op;
        Value scalarOperand; // the SCALAR_ONLY value to hoist
        Value divisor;       // the actual divisor in the operation
      };
      SmallVector<HoistCandidate> candidates;

      gpuFunc.walk([&](arith::DivUIOp op) {
        Value divisor = op.getRhs();
        Value scalarOp = getScalarOperand(divisor, depMap);
        if (scalarOp && isIntOrIndex(divisor.getType()))
          candidates.push_back({op, scalarOp, divisor});
      });
      gpuFunc.walk([&](arith::RemUIOp op) {
        Value divisor = op.getRhs();
        Value scalarOp = getScalarOperand(divisor, depMap);
        if (scalarOp && isIntOrIndex(divisor.getType()))
          candidates.push_back({op, scalarOp, divisor});
      });

      if (candidates.empty()) continue;

      // ---- Phase 3+4: Hoist to host & replace in kernel ----
      // Insertion point: before the first scf.for loop
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

      // Deduplicate: same scalar operand → same magic/shift
      DenseMap<Value, std::pair<Value, Value>> scalarMagicShift;
      SmallVector<Value> orderedScalars;

      for (auto &c : candidates) {
        if (scalarMagicShift.count(c.scalarOperand)) continue;
        orderedScalars.push_back(c.scalarOperand);

        // Get the host-side value corresponding to this kernel arg
        Value hostDivisor;
        // Check if it's a block argument — then get launch operand
        if (auto arg = dyn_cast<BlockArgument>(c.scalarOperand)) {
          hostDivisor = launch.getKernelOperand(arg.getArgNumber());
        } else {
          // SCALAR_ONLY but not a block arg → compute in kernel?
          // For pure block-arg SCALAR_ONLY patterns, this shouldn't happen.
          // If the scalar is defined inside the kernel (e.g., arith.add of
          // two scalar args), we can't easily get its host equivalent.
          // Skip for now.
          continue;
        }

        if (!isIntOrIndex(hostDivisor.getType()))
          hostDivisor = hostB.create<arith::IndexCastUIOp>(loc, i32Ty, hostDivisor);

        // --- Magic/shift computation (host side) ---
        Value one   = hostB.create<arith::ConstantIntOp>(loc, 1, 32);
        Value dm1   = hostB.create<arith::SubIOp>(loc, hostDivisor, one);
        Value clz   = hostB.create<math::CountLeadingZerosOp>(loc, dm1);
        Value c32   = hostB.create<arith::ConstantIntOp>(loc, 32, 32);
        Value shift = hostB.create<arith::SubIOp>(loc, c32, clz);

        Value d64     = hostB.create<arith::ExtUIOp>(loc, i64Ty, hostDivisor);
        Value s64     = hostB.create<arith::ExtUIOp>(loc, i64Ty, shift);
        Value c32_64  = hostB.create<arith::ConstantIntOp>(loc, 32, 64);
        Value ts      = hostB.create<arith::AddIOp>(loc, s64, c32_64);
        Value one64   = hostB.create<arith::ConstantIntOp>(loc, 1, 64);
        Value p2      = hostB.create<arith::ShLIOp>(loc, one64, ts);
        Value m0      = hostB.create<arith::DivUIOp>(loc, p2, d64);
        Value m1      = hostB.create<arith::AddIOp>(loc, m0, one64);
        Value pr      = hostB.create<arith::MulIOp>(loc, m1, d64);
        Value ov      = hostB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, pr, p2);
        Value m64     = hostB.create<arith::SelectOp>(loc, ov, m0, m1);
        Value magic   = hostB.create<arith::TruncIOp>(loc, i32Ty, m64);

        scalarMagicShift[c.scalarOperand] = {magic, shift};
      }

      if (orderedScalars.empty()) continue;

      // --- Add magic/shift as kernel args & launch operands ---
      SmallVector<std::pair<unsigned, std::pair<Value, Value>>> newArgs;
      unsigned newArgIdx = gpuFunc.getNumArguments();
      for (Value sv : orderedScalars) {
        auto [magic, shift] = scalarMagicShift[sv];
        gpuFunc.getBody().front().addArgument(i32Ty, loc);
        gpuFunc.getBody().front().addArgument(i32Ty, loc);
        launch.getKernelOperandsMutable().append(magic);
        launch.getKernelOperandsMutable().append(shift);
        newArgs.push_back({newArgIdx, {magic, shift}});
        newArgIdx += 2;
      }

      // Update ALL gpu.launch_func ops calling this kernel
      auto gpuMod = module.lookupSymbol<gpu::GPUModuleOp>(
          launch.getKernelModuleName());
      if (gpuMod) {
        hostFunc.walk([&](gpu::LaunchFuncOp otherLaunch) {
          if (otherLaunch.getKernelName() != launch.getKernelName()) return;
          if (otherLaunch == launch) return;
          for (unsigned i = 0; i < orderedScalars.size(); ++i) {
            auto [magic, shift] = scalarMagicShift[orderedScalars[i]];
            otherLaunch.getKernelOperandsMutable().append(magic);
            otherLaunch.getKernelOperandsMutable().append(shift);
          }
        });
      }

      // Update gpu.func function type
      SmallVector<Type> newArgTypes;
      for (auto arg : gpuFunc.getBody().getArguments())
        newArgTypes.push_back(arg.getType());
      gpuFunc.setFunctionType(
          FunctionType::get(gpuFunc.getContext(), newArgTypes, {}));

      // --- Replace division ops in kernel body ---
      for (auto &c : candidates) {
        // Find the corresponding new block args
        unsigned idx = llvm::find(orderedScalars, c.scalarOperand) -
                       orderedScalars.begin();
        if (idx >= orderedScalars.size()) continue;

        unsigned oldCount = gpuFunc.getNumArguments() -
                            2 * orderedScalars.size();
        BlockArgument magicArg =
            gpuFunc.getBody().getArgument(oldCount + 2 * idx);
        BlockArgument shiftArg =
            gpuFunc.getBody().getArgument(oldCount + 2 * idx + 1);

        OpBuilder b(c.op);
        Value n;
        if (auto dOp = dyn_cast<arith::DivUIOp>(c.op))
          n = dOp.getLhs();
        else if (auto rOp = dyn_cast<arith::RemUIOp>(c.op))
          n = rOp.getLhs();
        else
          continue;

        // mul_hi(magic, n) → zext both to i64, mul, lshr 32
        Value m64_2 = b.create<arith::ExtUIOp>(loc, i64Ty, magicArg);
        Value n64_2 = b.create<arith::ExtUIOp>(loc, i64Ty, n);
        Value full   = b.create<arith::MulIOp>(loc, m64_2, n64_2);
        Value c32_2  = b.create<arith::ConstantIntOp>(loc, 32, 64);
        Value hi     = b.create<arith::ShRUIOp>(loc, full, c32_2);
        Value hi32   = b.create<arith::TruncIOp>(loc, i32Ty, hi);
        Value sum    = b.create<arith::AddIOp>(loc, hi32, n);

        if (isa<arith::DivUIOp>(c.op)) {
          Value res = b.create<arith::ShRUIOp>(loc, sum, shiftArg);
          c.op->getResult(0).replaceAllUsesWith(res);
        } else {
          // rem: n - (quot * divisor)
          Value quot = b.create<arith::ShRUIOp>(loc, sum, shiftArg);
          Value divVal = cast<arith::RemUIOp>(c.op).getRhs();
          Value prod = b.create<arith::MulIOp>(loc, quot, divVal);
          Value res  = b.create<arith::SubIOp>(loc, n, prod);
          c.op->getResult(0).replaceAllUsesWith(res);
        }
        c.op->erase();
      }
    }
  }
};
} // namespace