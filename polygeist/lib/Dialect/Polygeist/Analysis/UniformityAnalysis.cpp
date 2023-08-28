//===- UniformityAnalysis.cpp - Uniformity Analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "uniformity-analysis"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Uniformity
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

bool isDivergent(Operation *op, DataFlowSolver &solver) {
  assert(op && op->getBlock() && "Expecting a valid operation in a block");

  SetVector<RegionBranchOpInterface> enclosingBranches =
      getParentsOfType<RegionBranchOpInterface>(*op->getBlock());
  if (enclosingBranches.empty())
    return false;

  auto isUniform = [&](Value val) {
    const auto *lattice = solver.lookupState<UniformityLattice>(val);
    assert(lattice && "expected uniformity information");
    assert(!lattice->getValue().isUninitialized() &&
           "lattice element should be initialized");
    Uniformity uniformity = lattice->getValue();
    return uniformity.isUniform();
  };

  bool mayBeDivergent =
      llvm::any_of(enclosingBranches, [&](RegionBranchOpInterface branchOp) {
        std::optional<IfCondition> cond = IfCondition::getCondition(branchOp);
        if (!cond)
          return false;

        return cond->perform([&](ValueRange values) {
          return llvm::any_of(values,
                              [&](Value val) { return !isUniform(val); });
        });
      });

  return mayBeDivergent;
}

raw_ostream &operator<<(raw_ostream &os, const Uniformity &uniformity) {
  if (uniformity.isUninitialized())
    return os << "<UNINITIALIZED>";

  switch (uniformity.getKind()) {
  case Uniformity::Kind::Unknown:
    return os << "unknown\n";
  case Uniformity::Kind::Uniform:
    return os << "uniform\n";
  case Uniformity::Kind::NonUniform:
    return os << "non-uniform\n";
  }
  llvm_unreachable("Unhandled case");
}

} // namespace polygeist
} // namespace mlir

void UniformityLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  auto value = point.get<Value>();
  if (Operation *op = value.getDefiningOp()) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      for (Value operand : branch->getOperands()) {
        auto *uniformity = solver->lookupState<UniformityLattice>(operand);
        assert(uniformity && "Expecting a valid uniformity");

        auto *lattice = solver->getOrCreateState<UniformityLattice>(value);
        return solver->propagateIfChanged(lattice, lattice->join(*uniformity));
      }
    }
  }

  // Note: required to get the dataflow framework to visit nested regions.
  auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
  return solver->propagateIfChanged(
      cv, cv->join(ConstantValue::getUnknownConstant()));
}

void Uniformity::print(raw_ostream &os) const { os << *this; }

//===----------------------------------------------------------------------===//
// UniformityAnalysis
//===----------------------------------------------------------------------===//

void UniformityAnalysis::setToEntryState(UniformityLattice *lattice) {
  // The arguments of a gpu kernel are uniform by default.
  if (Region *region = lattice->getPoint().getParentRegion()) {
    if (auto gpuFuncOp = dyn_cast<gpu::GPUFuncOp>(region->getParentOp())) {
      if (gpuFuncOp.isKernel()) {
        for (Value arg : gpuFuncOp.front().getArguments()) {
          UniformityLattice *lattice = getLatticeElement(arg);
          propagateIfChanged(lattice, lattice->join(Uniformity::getUniform()));
        }
        return;
      }
    }
  }

  // Other arguments have unknown uniformity.
  propagateIfChanged(lattice, lattice->join(Uniformity::getUnknown()));
}

void UniformityAnalysis::visitOperation(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting operation: " << *op << "\n");

  // If the lattice on any operand isn't yet initialized, bail out.
  if (anyOfUniformityIsUninitialized(operands)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operand(s) uniformity not yet initialized\n");
    return;
  }

  // Operations that always yield non-uniform result(s).
  if (op->hasTrait<OpTrait::ResultsNonUniform>()) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operation yields non-uniform result(s)\n");
    return propagateAllIfChanged(results, Uniformity::getNonUniform());
  }

  // If any operand uniformity is unknown/non-uniform the result(s) are also
  // unknown/non-uniform.
  if (anyOfUniformityIs(operands, Uniformity::Kind::Unknown)) {
    LLVM_DEBUG(llvm::dbgs() << "UA: Operand(s) uniformity is unknown\n");
    return propagateAllIfChanged(results, Uniformity::getUnknown());
  }
  if (anyOfUniformityIs(operands, Uniformity::Kind::NonUniform)) {
    LLVM_DEBUG(llvm::dbgs() << "UA: Operand(s) are non-uniform\n");
    return propagateAllIfChanged(results, Uniformity::getNonUniform());
  }
  assert(llvm::all_of(operands,
                      [](const UniformityLattice *lattice) {
                        return Uniformity::isUniform(
                            lattice->getValue().getKind());
                      }) &&
         "Expecting all operands to be uniform");

  // A memory side effects free operation that has uniform operands yields
  // uniform result(s).
  if (isMemoryEffectFree(op)) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Operation is memory effect free\n");
    return propagateAllIfChanged(results, Uniformity::getUniform());
  }

  return analyzeMemoryEffects(op, operands, results);
}

void UniformityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<UniformityLattice *> argLattices, unsigned firstIndex) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting non-ctrl-flow args of operation: "
                          << *op << "\n");

  // If the lattice on any operand isn't yet initialized, bail out.
  if (anyOfUniformityIsUninitialized(op->getOperands())) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operand(s) uniformity not yet initialized\n");
    return;
  }

  // Infer the uniformity of the loop IV by analyzing the loop bounds and step.
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    if (auto uniformity = getInductionVariableUniformity(loop))
      return propagateAllIfChanged(*loop.getSingleInductionVar(), *uniformity);
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

void UniformityAnalysis::analyzeMemoryEffects(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  assert(!isMemoryEffectFree(op) &&
         "Expecting operation to have memory effects");
  assert(llvm::all_of(operands,
                      [](const UniformityLattice *lattice) {
                        return !lattice->getValue().isUninitialized();
                      }) &&
         "Expecting all operands to be initialized");
  assert(llvm::all_of(operands,
                      [](const UniformityLattice *lattice) {
                        return Uniformity::isUniform(
                            lattice->getValue().getKind());
                      }) &&
         "Expecting all operands to be uniform");

  // If the operation has unknown memory side effects assume its result(s)
  // have unknown uniformity.
  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffectOp) {
    LLVM_DEBUG(llvm::dbgs() << "UA: Operation has unknown memory effects\n");
    return propagateAllIfChanged(results, Uniformity::getUnknown());
  }

  // If the operation only allocates memory, the value yielded is uniform.
  if (hasSingleEffect<MemoryEffects::Allocate>(memoryEffectOp)) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Operation allocates a value\n");
    return propagateAllIfChanged(results, Uniformity::getUniform());
  }

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Analyzing memory effects\n");
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectOp.getEffects(effects);

  for (const auto &EI : effects) {
    Value val = EI.getValue();
    if (!val) {
      // Memory effect on anything other than a value: conservatively assume
      // the result(s) uniformity is unknown.
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Memory Effect on non-values found\n");
      return propagateAllIfChanged(results, Uniformity::getUnknown());
    }

    if (!isa<MemoryEffects::Read>(EI.getEffect()))
      continue;

    if (!getOrCreateFor<ReachingDefinition>(op, op))
      return;

    // Get the reaching def. and potential reaching def. of the value 'val'
    // and analyze them to determine its uniformity.
    const ReachingDefinition *rdef = solver.lookupState<ReachingDefinition>(op);

    if (!rdef) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Unable to find reaching definition\n");
      return propagateAllIfChanged(op->getResults(), Uniformity::getUnknown());
    }

    LLVM_DEBUG(llvm::dbgs().indent(2) << "rdef: " << *rdef << "\n";);

    using ModifiersTy = ReachingDefinition::ModifiersTy;
    auto merge = [](std::optional<ModifiersTy> mods,
                    std::optional<ModifiersTy> pMods) {
      if (!mods && !pMods)
        return mods;
      if (mods && !pMods)
        return mods;
      if (pMods && !mods)
        return pMods;
      mods->merge(*pMods);
      return mods;
    };

    // Merge mods and pMods together.
    std::optional<ModifiersTy> mods =
        merge(rdef->getModifiers(val, solver),
              rdef->getPotentialModifiers(val, solver));
    if (!mods)
      continue;

    LLVM_DEBUG({
      llvm::dbgs().indent(2) << "val: " << val << "\n";
      llvm::dbgs().indent(2) << "merged mods:\n";
      for (Definition mod : *mods)
        llvm::dbgs().indent(4) << mod << "\n";
    });

    // Collect the branch conditions that dominate the modifiers.
    SmallVector<IfCondition> branchConditions = collectBranchConditions(*mods);

    // If we haven't yet computed the uniformity of the branch conditions, bail
    // out.
    if (!isUniformityInitialized(branchConditions, op)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Reaching def operand(s) uniformity not yet initialized\n");
      return;
    }

    // If any of mods/pMods are dominated by a branch with a condition that is
    // unknown/non-uniform the loaded value has the same uniformity.
    if (anyOfUniformityIs(branchConditions, Uniformity::Kind::Unknown)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Branch condition has unknown uniformity\n");
      return propagateAllIfChanged(op->getResults(), Uniformity::getUnknown());
    }
    if (anyOfUniformityIs(branchConditions, Uniformity::Kind::NonUniform)) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Branch condition non-uniform\n");
      return propagateAllIfChanged(op->getResults(),
                                   Uniformity::getNonUniform());
    }

    // If we haven't yet computed the mods/pMods operands uniformity, bail out.
    if (!isUniformityInitialized(*mods, op)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Reaching def operand(s) uniformity not yet initialized\n");
      return;
    }

    // If any modifiers or potential modifiers of the value loaded store a
    // value that is unknown/non-uniform the result(s) of the load are also
    // unknown/non-uniform.
    if (anyModifierUniformityIs(*mods, Uniformity::Kind::Unknown)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Reaching def has unknown uniformity\n");
      return propagateAllIfChanged(op->getResults(), Uniformity::getUnknown());
    }
    if (anyModifierUniformityIs(*mods, Uniformity::Kind::NonUniform)) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Reaching def is non-uniform\n");
      return propagateAllIfChanged(op->getResults(),
                                   Uniformity::getNonUniform());
    }
  }

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Memory effects analyzed\n");
  return propagateAllIfChanged(op->getResults(), Uniformity::getUniform());
}

std::optional<Uniformity>
UniformityAnalysis::getInductionVariableUniformity(LoopLikeOpInterface loop) {
  if (std::optional<Value> iv = loop.getSingleInductionVar()) {
    // Collect the bounds and step if they aren't constant.
    SmallVector<Value> boundsAndStep;
    Operation *loopOp = loop;
    if (auto affineLoop = dyn_cast<affine::AffineForOp>(loopOp)) {
      // Non-constant lower/upper bounds for affine loops use a map so special
      // case them here. Note that the step is always a constant.
      ValueRange lbOperands = affineLoop.getLowerBoundOperands();
      ValueRange ubOperands = affineLoop.getUpperBoundOperands();
      boundsAndStep.append(lbOperands.begin(), lbOperands.end());
      boundsAndStep.append(ubOperands.begin(), ubOperands.end());
    } else {
      std::optional<OpFoldResult> lb = loop.getSingleLowerBound();
      std::optional<OpFoldResult> ub = loop.getSingleUpperBound();
      std::optional<OpFoldResult> st = loop.getSingleStep();

      // The result yielded by the calls above may be:
      //  - std::nullopt is an OpFoldResult cannot be computed
      //  - PointerUnion<Attribute, Value> containing:
      //     + an Attribute if the loop bound/step is constant
      //     + a Value if the bound/step is not immediately known
      if (!lb || !ub || !st)
        return std::nullopt;

      if (auto lbVal = llvm::dyn_cast_if_present<Value>(*lb))
        boundsAndStep.push_back(lbVal);
      if (auto ubVal = llvm::dyn_cast_if_present<Value>(*ub))
        boundsAndStep.push_back(ubVal);
      if (auto stVal = llvm::dyn_cast_if_present<Value>(*st))
        boundsAndStep.push_back(stVal);
    }

    // The loop IV uniformity matches the one of the values collected.
    if (anyOfUniformityIs(boundsAndStep, Uniformity::Kind::Unknown)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "UA: Loop bound(s) or step have unknown uniformity\n");
      return Uniformity::getUnknown();
    }
    if (anyOfUniformityIs(boundsAndStep, Uniformity::Kind::NonUniform)) {
      LLVM_DEBUG(llvm::dbgs() << "UA: Loop bound(s) or step are non-uniform\n");
      return Uniformity::getNonUniform();
    }

    LLVM_DEBUG(llvm::dbgs() << "UA: Loop bounds and step are uniform\n");
    return Uniformity::getUniform();
  }

  return std::nullopt;
}

SmallVector<IfCondition> UniformityAnalysis::collectBranchConditions(
    const ReachingDefinition::ModifiersTy &mods) {
  SmallVector<IfCondition> conditions;
  for (const Definition &mod : mods) {
    if (!mod.isOperation())
      continue;

    SetVector<RegionBranchOpInterface> enclosingBranches =
        getParentsOfType<RegionBranchOpInterface>(
            *mod.getOperation()->getBlock());
    for (RegionBranchOpInterface branchOp : enclosingBranches) {
      if (isa<affine::AffineForOp, scf::ForOp>(branchOp))
        continue;

      std::optional<IfCondition> condition =
          IfCondition::getCondition(branchOp);
      assert(condition && "Failed to find condition");
      conditions.push_back(*condition);
    }
  }

  LLVM_DEBUG({
    if (!conditions.empty()) {
      llvm::dbgs().indent(2) << "branch conditions:\n";
      for (auto cond : conditions)
        llvm::dbgs().indent(4) << cond << "\n";
    }
  });

  return conditions;
}

bool UniformityAnalysis::isUniformityInitialized(
    ArrayRef<IfCondition> conditions, Operation *op) {
  // Determine whether any condition has uniformity that is not yet known.
  bool uniformityIsKnown =
      llvm::all_of(conditions, [&](const IfCondition &cond) {
        return cond.perform([&](ValueRange values) {
          return !anyOfUniformityIsUninitialized(values);
        });
      });

  // Inject lattice nodes if necessary.
  if (!uniformityIsKnown) {
    auto getOrCreateLatticeFor = [&](Value val) {
      UniformityLattice *lattice = getLatticeElement(val);
      if (lattice->getValue().isUninitialized())
        getOrCreateFor<UniformityLattice>(op, val);
    };

    for (const IfCondition &cond : conditions) {
      cond.perform([&](ValueRange values) {
        for (Value val : values)
          getOrCreateLatticeFor(val);
        return true;
      });
    }
  }

  return uniformityIsKnown;
}

bool UniformityAnalysis::isUniformityInitialized(
    const ReachingDefinition::ModifiersTy &mods, Operation *op) {
  assert(op && "Expecting a valid operation");

  // Determine whether any modifier has operands with uniformity that is not
  // yet known.
  bool uniformityIsKnown = llvm::all_of(mods, [&](const Definition &def) {
    if (!def.isOperation())
      return true;

    Operation *defOp = def.getOperation();
    return !anyOfUniformityIsUninitialized(defOp->getOperands());
  });

  // Inject lattice nodes if necessary.
  if (!uniformityIsKnown) {
    for (const Definition &def : mods) {
      if (!def.isOperation())
        continue;

      Operation *defOp = def.getOperation();
      for (Value operand : defOp->getOperands()) {
        // If the operand uniformity is not yet initialized we need to create
        // a dependency between the operand state and 'op', the operation that
        // uses values the modifiers potentially define, so that when the
        // uniformity of the modifier(s) has been computed, the dataflow
        // framework revisits 'op'.
        UniformityLattice *lattice = getLatticeElement(operand);
        if (lattice->getValue().isUninitialized())
          getOrCreateFor<UniformityLattice>(op, operand);
      }
    }
  }

  return uniformityIsKnown;
}

bool UniformityAnalysis::anyModifierUniformityIs(
    const ReachingDefinition::ModifiersTy &mods, Uniformity::Kind kind) {
  return llvm::any_of(mods, [&](const Definition &def) {
    // The initial definition (the one for pointer args to a function)
    // has unknown uniformity.
    if (def.isInitialDefinition())
      return Uniformity::isUnknown(kind);

    assert(!anyOfUniformityIsUninitialized(def.getOperation()->getOperands()) &&
           "Expecting the uniformity of all operands to be initialized");

    // Handle a concrete definition.
    return TypeSwitch<Operation *, bool>(def.getOperation())
        .Case<memref::AllocaOp, LLVM::AllocaOp>(
            [&](auto) { return Uniformity::isUniform(kind); })
        .Case<memref::StoreOp, affine::AffineStoreOp, sycl::SYCLConstructorOp,
              sycl::SYCLIDConstructorOp, sycl::SYCLRangeConstructorOp,
              sycl::SYCLNDRangeConstructorOp, LLVM::StoreOp, LLVM::MemsetOp>(
            [&](auto op) { return anyOfUniformityIs(op.getOperands(), kind); })
        .Default([](auto *op) {
          llvm::errs() << "op: " << *op << "\n";
          llvm_unreachable("Unhandled operation");
          return false;
        });
  });
}

void UniformityAnalysis::propagateAllIfChanged(
    ArrayRef<UniformityLattice *> results, const Uniformity &uniformity) {
  for (UniformityLattice *result : results)
    propagateIfChanged(result, result->join(uniformity));
  LLVM_DEBUG(llvm::dbgs() << "UA: Result(s) are: " << uniformity << "\n");
}

void UniformityAnalysis::propagateAllIfChanged(const ValueRange values,
                                               const Uniformity &uniformity) {
  for (Value value : values) {
    UniformityLattice *lattice = getLatticeElement(value);
    propagateIfChanged(lattice, lattice->join(uniformity));
  }
  LLVM_DEBUG(llvm::dbgs() << "UA: Result(s) are " << uniformity << "\n");
}
