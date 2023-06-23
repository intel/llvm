//===- UniformityAnalysis.cpp - Uniformity Analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
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

raw_ostream &operator<<(raw_ostream &os, const Uniformity &uniformity) {
  if (uniformity.isUninitialized())
    return os << "<UNINITIALIZED>";

  os << "uniformity: ";
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

void UniformityAnalysis::visitOperation(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting operation: " << *op << "\n");

  // If the lattice on any operand is uninitialized, bail out.
  if (llvm::any_of(operands, [](const UniformityLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
    return;
  }

  // These operations always yield non-uniform result(s).
  if (isa<sycl::SYCLItemGetIDOp, sycl::SYCLNDItemGetGlobalIDOp>(op)) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(Uniformity::getNonUniform()));
    return;
  }

  auto anyOperandUniformityIs = [&](Uniformity::Kind kind) {
    return llvm::any_of(operands, [&](const UniformityLattice *lattice) {
      return lattice->getValue().getKind() == kind;
    });
  };

  // If any operand uniformity is unknown/non-uniform the result(s) are also
  // unknown/non-uniform.
  if (anyOperandUniformityIs(Uniformity::Kind::Unknown)) {
    setAllToEntryStates(results);
    return;
  }
  if (anyOperandUniformityIs(Uniformity::Kind::NonUniform)) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(Uniformity::getNonUniform()));
    return;
  }

  assert(llvm::all_of(operands,
                      [&](const UniformityLattice *lattice) {
                        return lattice->getValue().getKind() ==
                               Uniformity::Kind::Uniform;
                      }) &&
         "Expecting all operands to be uniform");

  // A memory side effects free operation that has uniform operands yields
  // uniform result(s).
  if (isMemoryEffectFree(op)) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(Uniformity::getUniform()));
    return;
  }

  // If an operation has unknown memory side effects assume its result(s)
  // have unknown uniformity.
  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffectOp) {
    for (UniformityLattice *result : results)
      propagateIfChanged(result, result->join(Uniformity::getUnknown()));
    return;
  }

  // Analyze operations with memory side effects that have uniform operands.
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectOp.getEffects(effects);
  for (const auto &effect : effects) {
    Value val = effect.getValue();
    if (!val) {
      // Memory effect on anything other than a value: conservatively assume
      // the result(s) uniformity is unknown.
      LLVM_DEBUG(llvm::dbgs() << "Memory Effect on non-values found\n");
      for (UniformityLattice *result : results)
        propagateIfChanged(result, result->join(Uniformity::getUnknown()));
      return;
    }

    TypeSwitch<MemoryEffects::Effect *>(effect.getEffect())
        .Case<MemoryEffects::Read>([](auto) {
          // A read operation yields uniform result(s) iff the (potentially)
          // reaching definitions of all its operands are uniform.
          assert(false && "TODO");
        })
        .Case<MemoryEffects::Write>([](auto) {
          // A write operation store a value to a memory location. Anything
          // to do here ?
          assert(false && "TODO");
        });
  }

  return;
}
