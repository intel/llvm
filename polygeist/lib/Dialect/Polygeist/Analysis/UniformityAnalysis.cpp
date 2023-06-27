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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
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

UniformityAnalysis::UniformityAnalysis(DataFlowSolver &solver,
                                       AliasAnalysis &aliasAnalysis)
    : SparseDataFlowAnalysis<UniformityLattice>(solver) {
  // Load the reaching definition analysis (and the analysis it depends on).
  // Reaching definition information are required by this analysis to reason
  // about the uniformity of values loaded from memory.
  internalSolver.load<DeadCodeAnalysis>();
  internalSolver.load<SparseConstantPropagation>();
  internalSolver.load<polygeist::ReachingDefinitionAnalysis>(aliasAnalysis);
}

LogicalResult UniformityAnalysis::initialize(Operation *top) {
  // Run the dataflow analysis loaded in the internal solver.
  if (failed(internalSolver.initializeAndRun(top)))
    return failure();

  return SparseDataFlowAnalysis::initialize(top);
}

void UniformityAnalysis::visitOperation(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting operation: " << *op << "\n");

  // If the lattice on any operand isn't yet initialized, bail out.
  if (llvm::any_of(operands, [](const UniformityLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
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
    LLVM_DEBUG(llvm::dbgs() << "Operand(s) uniformity is unknown\n");
    return propagateAllIfChanged(results, Uniformity::getNonUniform());
  }
  if (anyOfUniformityIs(operands, Uniformity::Kind::NonUniform)) {
    LLVM_DEBUG(llvm::dbgs() << "Operand(s) are non-uniform\n");
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
    LLVM_DEBUG(llvm::dbgs() << "Operation is memory effect free\n");
    return propagateAllIfChanged(results, Uniformity::getUniform());
  }

  return analyzeMemoryEffects(op, operands, results);
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
    LLVM_DEBUG(llvm::dbgs() << "Operation has unknown memory effects\n");
    return propagateAllIfChanged(results, Uniformity::getUnknown());
  }

  // If the operation only allocates memory, the value yielded is uniform.
  if (hasSingleEffect<MemoryEffects::Allocate>(memoryEffectOp)) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Operation allocates a value\n");
    return propagateAllIfChanged(results, Uniformity::getUniform());
  }

  // Given the reaching definitions of a value determine whether any value
  // stored into memory by any of the reaching definitions has uniformity that
  // catches the given kind.
  auto anyModifierUniformityIs =
      [&](const ReachingDefinition::ModifiersTy &mods, Uniformity::Kind kind) {
        return llvm::any_of(mods, [&](const Definition &def) {
          // The initial definition (the one for pointer args to a function) has
          // unknown uniformity.
          if (def.isInitialDefinition())
            return Uniformity::isUnknown(kind);

          // Handle a concrete definition.
          return TypeSwitch<Operation *, bool>(def.getOperation())
              .Case<memref::AllocaOp>(
                  [&](auto) { return Uniformity::isUniform(kind); })
              .Case<memref::StoreOp, affine::AffineStoreOp>([&](auto storeOp) {
                return anyOfUniformityIs(storeOp.getValueToStore(), kind);
              })
              .Default([](auto op) {
                llvm::errs() << "op: " << *op << "\n";
                llvm_unreachable("Unhandled operation type");
                return false;
              });
        });
      };

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Analyzing memory effects\n");
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectOp.getEffects(effects);

  for (const auto &EI : effects) {
    Value val = EI.getValue();
    if (!val) {
      // Memory effect on anything other than a value: conservatively assume
      // the result(s) uniformity is unknown.
      LLVM_DEBUG(llvm::dbgs() << "Memory Effect on non-values found\n");
      return propagateAllIfChanged(results, Uniformity::getUnknown());
    }

    if (!isa<MemoryEffects::Read>(EI.getEffect()))
      continue;

    // Get the reaching def. and potential reaching def. of the value 'val' and
    // analyze them to determine its uniformity.
    const ReachingDefinition *rdef =
        internalSolver.lookupState<ReachingDefinition>(op);
    if (!rdef) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to find reaching definition\n");
      return propagateAllIfChanged(op->getResults(), Uniformity::getUnknown());
    }
    LLVM_DEBUG(llvm::dbgs().indent(2) << "rdef:\n" << *rdef << "\n");

    auto rdefUniformityIs = [&](const ReachingDefinition &rdef,
                                Uniformity::Kind kind) {
      using ModifiersTy = ReachingDefinition::ModifiersTy;
      std::optional<ModifiersTy> mods = rdef.getModifiers(val);
      std::optional<ModifiersTy> pMods = rdef.getPotentialModifiers(val);
      assert((mods || pMods) && "Should have at least a (potential) modifier");
      return (mods && anyModifierUniformityIs(*mods, kind)) ||
             (pMods && anyModifierUniformityIs(*pMods, kind));
    };

    // If any modifiers or potential modifiers of the value are
    // unknown/non-uniform the result(s) are also unknown/non-uniform.
    if (rdefUniformityIs(*rdef, Uniformity::Kind::Unknown)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Reaching def. has unknown uniformity\n");
      return propagateAllIfChanged(op->getResults(), Uniformity::getUnknown());
    }
    if (rdefUniformityIs(*rdef, Uniformity::Kind::NonUniform)) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Reaching def. is non-uniform\n");
      return propagateAllIfChanged(op->getResults(),
                                   Uniformity::getNonUniform());
    }
  }

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Memory effects analyzed\n");
  return propagateAllIfChanged(op->getResults(), Uniformity::getUniform());
}

void UniformityAnalysis::propagateAllIfChanged(
    ArrayRef<UniformityLattice *> results, Uniformity &&uniformity) {
  for (UniformityLattice *result : results)
    propagateIfChanged(result, result->join(uniformity));
  LLVM_DEBUG(llvm::dbgs() << "Results are: " << uniformity << "\n");
}

void UniformityAnalysis::propagateAllIfChanged(const ValueRange values,
                                               Uniformity &&uniformity) {
  for (Value value : values) {
    UniformityLattice *lattice = getLatticeElement(value);
    propagateIfChanged(lattice, lattice->join(uniformity));
  }
  LLVM_DEBUG(llvm::dbgs().indent(2) << "Result(s) are " << uniformity << "\n");
}
