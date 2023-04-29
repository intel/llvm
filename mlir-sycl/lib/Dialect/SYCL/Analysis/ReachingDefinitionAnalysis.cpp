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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "reaching-definition-analysis"

namespace mlir {
namespace sycl {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

class AliasUtilities {
public:
  /// \p op is the operation to analyze.
  AliasUtilities(AliasAnalysis &aliasAnalysis, Operation &op)
      : aliasAnalysis(aliasAnalysis) {
    SmallVector<Value> memoryResources = collectMemoryResourcesIn(op);
    for (Value val1 : memoryResources) {
      for (Value val2 : memoryResources) {
        if (val1 == val2)
          continue;

        AliasResult aliasResult = aliasAnalysis.alias(val1, val2);
        if (aliasResult.isMust())
          valueToMustAliasValues[val1].push_back(val2);
        else if (aliasResult.isMay())
          valueToMayAliasValues[val1].push_back(val2);
      }
    }
  }

  /// TODO: merge these 2 function and pass the kind of alised (must, may)
  /// values to get ? Create a map from a value to the set of values that are
  /// "must" alias.
  const SmallVector<Value> &getMustAlias(Value val) const;

  /// Create a map from a value to the set of values that are "may" alias.
  const SmallVector<Value> &getMayAlias(Value val) const;

private:
  /// Collect values that reference a memory resource within operation \p op.
  /// TODO: should pass a functionLikeInterface here ? Or a Region ?
  SmallVector<Value> collectMemoryResourcesIn(Operation &op) {
    SmallVector<Value> memoryResources;
    assert(false); // TODO
    return memoryResources;
  }

private:
  // val -> values that are definitely aliased.
  DenseMap<Value, SmallVector<Value>> valueToMustAliasValues;

  // val -> values that might be aliased.
  DenseMap<Value, SmallVector<Value>> valueToMayAliasValues;

  AliasAnalysis &aliasAnalysis;
};

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
  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "lhs: " << lhs << ", rhs: " << rhs << "\n");

  UnderlyingValue res;
  if (lhs.isUninitialized())
    res = rhs;
  else if (rhs.isUninitialized())
    res = lhs;
  else
    res = (lhs == rhs) ? lhs : UnderlyingValue();

  LLVM_DEBUG(llvm::dbgs().indent(4) << "res: " << res << "\n");
  return res;
}

Value UnderlyingValue::getUnderlyingValue(
    Value val,
    function_ref<const UnderlyingValueLattice *(Value)> getLatticeForValue,
    AliasAnalysis &aliasAnalysis) {
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
// UnderlyingValueAnalysis
//===----------------------------------------------------------------------===//

void UnderlyingValueAnalysis::setToEntryState(UnderlyingValueLattice *lattice) {
  LLVM_DEBUG(llvm::dbgs() << "UnderlyingValueAnalysis - setToEntryState\n");

  /// At an entry point, the underlying value of a value is itself.
  propagateIfChanged(lattice,
                     lattice->join(UnderlyingValue{lattice->getPoint()}));
}

void UnderlyingValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const UnderlyingValueLattice *> operands,
    ArrayRef<UnderlyingValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "UnderlyingValueAnalysis - Visiting: " << *op
                          << "\n");

  /// The underlying value of the results of an operation are not known.
  setAllToEntryStates(results);
}

//===----------------------------------------------------------------------===//
// ReachingDefinition
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const ReachingDefinition &lastDef) {
  if (lastDef.valueToModifiers.empty())
    return os.indent(4) << "<empty>\n";

  for (const auto &entry : lastDef.valueToModifiers) {
    Value val = entry.first;
    const ReachingDefinition::ModifiersTy &valModifiers = entry.second;

    os.indent(4) << "val: " << val << "\n";
    os.indent(4) << "modifiers:\n";
    if (valModifiers.empty())
      os.indent(4) << "<none>\n";
    else {
      for (Operation *op : valModifiers)
        os.indent(6) << *op << "\n";
    }
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

ChangeResult ReachingDefinition::setModifier(Value val, Operation *op) {
  ReachingDefinition::ModifiersTy &modifiers = valueToModifiers[val];
  if (modifiers.size() == 1 && modifiers.front() == op)
    return ChangeResult::NoChange;

  modifiers.clear();
  modifiers.insert(op);
  return ChangeResult::Change;
}

ChangeResult ReachingDefinition::addPotentialModifier(Value val,
                                                      Operation *op) {
  ReachingDefinition::ModifiersTy &modifiers = valueToPotentialModifiers[val];
  modifiers.insert(op);
  return ChangeResult::Change;
}

std::optional<ArrayRef<Operation *>>
ReachingDefinition::getModifiers(Value val) const {
  auto it = valueToModifiers.find(val);
  return (it != valueToModifiers.end()) ? it->second.getArrayRef()
                                        : std::nullopt;
}

std::optional<ArrayRef<Operation *>>
ReachingDefinition::getPotentialModifiers(Value val) const {
  auto it = valueToModifiers.find(val);
  return (it != valueToPotentialModifiers.end()) ? it->second.getArrayRef()
                                                 : std::nullopt;
}

//===----------------------------------------------------------------------===//
// ReachingDefinitionAnalysis
//===----------------------------------------------------------------------===//

void ReachingDefinitionAnalysis::setToEntryState(ReachingDefinition *lattice) {
  /// Set the initial state (nothing is known about reaching definitions).
  LLVM_DEBUG(llvm::dbgs() << "ReachingDefinitionAnalysis - setToEntryState\n");
  propagateIfChanged(lattice, lattice->reset());
}

void ReachingDefinitionAnalysis::visitOperation(
    Operation *op, const ReachingDefinition &before,
    ReachingDefinition *after) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << *op << "\n");

  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffectOp) {
    // If an operation has unknown memory effects then nothing can be
    // deduced about the last modifications.
    LLVM_DEBUG(llvm::dbgs() << "Operation has unknown side effects\n");
    return setToEntryState(after);
  }

  ChangeResult result = ChangeResult::NoChange;
  after->join(before);

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
      // Memory effect on anything other than a value: conservatively
      // assume we can't deduce anything about the last modifications.
      LLVM_DEBUG(llvm::dbgs() << "Memory Effect on non-values found\n");
      return setToEntryState(after);
    }

    // Read operations do not modify the reaching definitions.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    //    LLVM_DEBUG(llvm::dbgs() << "val: " << val << "\n";);

    Value underlyingVal = UnderlyingValue::getUnderlyingValue(
        val, getAnalysisState, aliasAnalysis);
    if (!underlyingVal) {
      LLVM_DEBUG(llvm::dbgs() << "underlyingVal not known\n");
      return;
    }

    //  LLVM_DEBUG({ llvm::dbgs() << "underlyingVal: " << underlyingVal <<
    //  "\n";
    //  });

    // TODO: op (DEF) modifies underlyingVal, the following instruction
    // modifies the value '%cast':
    //   %cast: memref.store %arg0, %cast[%1]
    // However the same instruction also potentially modifies any other value
    // that "may_alias" with %cast.
    result = after->setModifier(underlyingVal, op);

    // So here we need to have a loop that
    // "after->addPotentialModifier(aliasedVal, op)" for each aliasedVal.

    /// TODO: Construct AliasUtilities an the start of the analysis.
    AliasUtilities aliasUtils(aliasAnalysis, *op);

    const SmallVector<Value> &mayAlias = aliasUtils.getMayAlias(underlyingVal);
    for (Value aliasedVal : mayAlias)
      after->addPotentialModifier(aliasedVal, op);
  }

  propagateIfChanged(after, result);

  LLVM_DEBUG({
    if (result == ChangeResult::Change) {
      llvm::dbgs().indent(2) << "Updated ReachingDef:\n";
      llvm::dbgs() << *after;
    }
  });
}

} // namespace sycl
} // namespace mlir
