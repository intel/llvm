//===- ReachingDefinitionAnalysis.cpp - Reaching Defs Analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "reaching-definition-analysis"

namespace mlir {
namespace sycl {

//===----------------------------------------------------------------------===//
// UnderlyingValue
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const UnderlyingValue &val) {
  return os << val.underlyingValue;
}

bool UnderlyingValue::isUninitialized() const {
  return !underlyingValue.has_value();
}

Value UnderlyingValue::getUnderlyingValue() const {
  assert(!isUninitialized());
  return *underlyingValue;
}

UnderlyingValue UnderlyingValue::join(const UnderlyingValue &lhs,
                                      const UnderlyingValue &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return (lhs == rhs) ? lhs : UnderlyingValue({});
}

Value UnderlyingValue::getUnderlyingValue(
    Value val,
    function_ref<const UnderlyingValueLattice *(Value)> getLatticeForValue) {
  do {
    const UnderlyingValueLattice *lattice = getLatticeForValue(val);
    if (!lattice || lattice->getValue().isUninitialized())
      return {};
    Value underlyingValue = lattice->getValue().getUnderlyingValue();
    if (underlyingValue == val)
      break;
    val = underlyingValue;
  } while (true);

  return val;
}

//===----------------------------------------------------------------------===//
// ReachingDefinition
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const ReachingDefinition &lastDef) {
  for (const auto &entry : lastDef.valueToModifiers) {
    Value val = entry.first;
    const ReachingDefinition::ModifiersTy &valModifiers = entry.second;

    os << val << ":\n";
    for (Operation *op : valModifiers)
      os.indent(2) << *op << "\n";
  }
  return os;
}

ChangeResult ReachingDefinition::join(const AbstractDenseLattice &lattice) {
  ChangeResult result = ChangeResult::NoChange;
  auto otherReachingDef = static_cast<const ReachingDefinition &>(lattice);

  for (const auto &entry : otherReachingDef.valueToModifiers) {
    Value val = entry.first;
    const ModifiersTy &newModifiers = entry.second;
    ModifiersTy &currentModifiers = valueToModifiers[val];
    if (currentModifiers != newModifiers) {
      currentModifiers.insert(newModifiers.begin(), newModifiers.end());
      result = ChangeResult::Change;
    }
  }
  return result;
}

ChangeResult ReachingDefinition::reset() {
  if (valueToModifiers.empty())
    return ChangeResult::NoChange;

  valueToModifiers.clear();
  return ChangeResult::Change;
}

ChangeResult ReachingDefinition::set(Value val, Operation *op) {
  ReachingDefinition::ModifiersTy &valModifiers = valueToModifiers[val];
  if (valModifiers.size() == 1 && valModifiers.front() == op)
    return ChangeResult::NoChange;

  valModifiers.clear();
  valModifiers.insert(op);
  return ChangeResult::Change;
}

std::optional<ArrayRef<Operation *>>
ReachingDefinition::getLastModifiers(Value val) const {
  auto it = valueToModifiers.find(val);
  return (it != valueToModifiers.end()) ? it->second.getArrayRef()
                                        : std::nullopt;
}

//===----------------------------------------------------------------------===//
// ReachingDefinitionAnalysis
//===----------------------------------------------------------------------===//

void ReachingDefinitionAnalysis::visitOperation(
    Operation *op, const ReachingDefinition &before,
    ReachingDefinition *after) {
  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffectOp) {
    // If an operation has unknown memory effects then nothing can be deduced
    // about the last modifications.
    return setToEntryState(after);
  }

  ChangeResult result = after->join(before);
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectOp.getEffects(effects);

  // Get the analysis state at the program point corresponding to this
  // operation.
  auto getAnalysisState = [&](Value val) {
    Operation *programPoint = op;
    return getOrCreateFor<UnderlyingValueLattice>(programPoint, val);
  };

  for (const auto &effect : effects) {
    Value val = effect.getValue();
    if (!val) {
      // Memory effect on anything other than a value: conservatively assume we
      // can't deduce anything about the last modifications.
      return setToEntryState(after);
    }

    // Read operations do not modify the reaching definitions.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    Value underlyingVal =
        UnderlyingValue::getUnderlyingValue(val, getAnalysisState);
    if (!underlyingVal)
      return;

    result = after->set(underlyingVal, op);
  }

  propagateIfChanged(after, result);
}

} // namespace sycl
} // namespace mlir
