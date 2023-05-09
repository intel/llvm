//===- ReachingDefinitionAnalysis.cpp - Reaching Defs Analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "reaching-definition-analysis"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// ReachingDefinition
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const ReachingDefinition &lastDef) {
  if (lastDef.valueToModifiers.empty() &&
      lastDef.valueToPotentialModifiers.empty())
    return os.indent(4) << "<empty>\n";

  auto printMap =
      [&os](const DenseMap<Value, ReachingDefinition::ModifiersTy> &map,
            StringRef title) {
        for (const auto &entry : map) {
          Value val = entry.first;
          const ReachingDefinition::ModifiersTy &valModifiers = entry.second;

          os.indent(4) << val << "\n";
          os.indent(4) << title << "\n";
          if (valModifiers.empty())
            os.indent(6) << "<none>\n";
          else {
            for (Operation *op : valModifiers)
              os.indent(6) << *op << "\n";
          }
          os << "\n";
        }
      };

  printMap(lastDef.valueToModifiers, "mods:");
  printMap(lastDef.valueToPotentialModifiers, "pMods:");
  return os;
}

ChangeResult ReachingDefinition::join(const AbstractDenseLattice &lattice) {
  ChangeResult result = ChangeResult::NoChange;
  auto otherReachingDef = static_cast<const ReachingDefinition &>(lattice);

  auto join = [&result](DenseMap<Value, ModifiersTy> &currentMap,
                        DenseMap<Value, ModifiersTy> &otherMap) {
    for (const auto &entry : otherMap) {
      Value val = entry.first;
      const ModifiersTy &newModifiers = entry.second;

      ModifiersTy &currentModifiers = currentMap[val];
      const std::size_t size = currentModifiers.size();
      currentModifiers.insert(newModifiers.begin(), newModifiers.end());
      if (currentModifiers.size() != size)
        result |= ChangeResult::Change;
    }
  };

  join(valueToModifiers, otherReachingDef.valueToModifiers);
  join(valueToPotentialModifiers, otherReachingDef.valueToPotentialModifiers);
  return result;
}

ChangeResult ReachingDefinition::reset() {
  if (valueToModifiers.empty() && valueToPotentialModifiers.empty())
    return ChangeResult::NoChange;

  valueToModifiers.clear();
  valueToPotentialModifiers.clear();
  return ChangeResult::Change;
}

ChangeResult ReachingDefinition::setModifier(Value val, Operation *op) {
  ReachingDefinition::ModifiersTy &mods = valueToModifiers[val];
  assert((mods.size() != 1 || mods.front() != op) &&
         "seen this modifier already");

  // Set the new modifier and clear out all previous definitions.
  mods.clear();
  mods.insert(op);
  valueToPotentialModifiers[val].clear();
  return ChangeResult::Change;
}

ChangeResult ReachingDefinition::removeModifiers(Value val) {
  if (!valueToModifiers.contains(val))
    return ChangeResult::NoChange;
  if (!valueToModifiers[val].empty()) {
    valueToModifiers[val].clear();
    return ChangeResult::Change;
  }
  return ChangeResult::NoChange;
}

ChangeResult ReachingDefinition::addPotentialModifier(Value val,
                                                      Operation *op) {
  return (valueToPotentialModifiers[val].insert(op)) ? ChangeResult::Change
                                                     : ChangeResult::NoChange;
}

ChangeResult ReachingDefinition::removePotentialModifiers(Value val) {
  if (!valueToPotentialModifiers.contains(val))
    return ChangeResult::NoChange;
  if (!valueToPotentialModifiers[val].empty()) {
    valueToPotentialModifiers[val].clear();
    return ChangeResult::Change;
  }
  return ChangeResult::NoChange;
}

std::optional<ArrayRef<Operation *>>
ReachingDefinition::getModifiers(Value val) const {
  if (valueToModifiers.contains(val))
    return valueToModifiers.at(val).getArrayRef();
  return std::nullopt;
}

std::optional<ArrayRef<Operation *>>
ReachingDefinition::getPotentialModifiers(Value val) const {
  if (valueToPotentialModifiers.contains(val))
    return valueToPotentialModifiers.at(val).getArrayRef();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ReachingDefinitionAnalysis
//===----------------------------------------------------------------------===//

void ReachingDefinitionAnalysis::setToEntryState(ReachingDefinition *lattice) {
  /// Set the initial state (nothing is known about reaching definitions).
  propagateIfChanged(lattice, lattice->reset());
}

void ReachingDefinitionAnalysis::visitOperation(
    Operation *op, const ReachingDefinition &before,
    ReachingDefinition *after) {
  LLVM_DEBUG(llvm::dbgs() << "ReachingDefinitionAnalysis - Visit: " << *op
                          << "\n");

  // Initialize the alias queries for the current function.
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    aliasQueriesMap[funcOp] =
        std::make_unique<AliasQueries>(funcOp, aliasAnalysis);
    return;
  }

  // If an operation has unknown memory effects assume we can't deduce
  // anything about reaching definitions.
  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffectOp) {
    LLVM_DEBUG(llvm::dbgs() << "Operation has unknown side effects\n");
    return setToEntryState(after);
  }

  // Transfer the input state.
  ChangeResult result = ChangeResult::NoChange;
  propagateIfChanged(after, after->join(before));

  // Retrieve the alias utilities for the function this operation belongs to.
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  AliasQueries &aliasQueries = *aliasQueriesMap[funcOp];

  // Analyze the operation's memory effects.
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectOp.getEffects(effects);
  for (const auto &effect : effects) {
    Value val = effect.getValue();
    if (!val) {
      // Memory effect on anything other than a value: conservatively assume we
      // can't deduce anything about reaching definitions.
      LLVM_DEBUG(llvm::dbgs() << "Memory Effect on non-values found\n");
      return setToEntryState(after);
    }

    // Read operations do not modify the reaching definitions state.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    TypeSwitch<MemoryEffects::Effect *>(effect.getEffect())
        .Case<MemoryEffects::Allocate>([&](auto) {
          // An allocate operation creates a definition for the current value.
          result |= after->setModifier(val, op);
        })
        .Case<MemoryEffects::Write>([&](auto) {
          // A write operation updates the definition of the current value
          // and the definition of its definitely aliased values. It also
          // updates the potential definitions of values that may alias the
          // current value.
          result |= after->setModifier(val, op);
          for (Value aliasedVal : aliasQueries.getMustAlias(val))
            result |= after->setModifier(aliasedVal, op);
          for (Value aliasedVal : aliasQueries.getMayAlias(val))
            result |= after->addPotentialModifier(aliasedVal, op);
        })
        .Case<MemoryEffects::Free>([&](auto) {
          // A deallocate operation kills reaching definitions of the
          // current value and of its definitely aliased values. It also
          // kills the potential definitions of values that may alias the
          // current value.
          result |= after->removeModifiers(val);
          for (Value aliasedVal : aliasQueries.getMustAlias(val))
            result |= after->removeModifiers(aliasedVal);
          for (Value aliasedVal : aliasQueries.getMayAlias(val))
            result |= after->removePotentialModifiers(aliasedVal);
        });
  }

  propagateIfChanged(after, result);

  LLVM_DEBUG({
    if (result == ChangeResult::Change) {
      llvm::dbgs().indent(2) << "Updated ReachingDef:\n";
      llvm::dbgs() << *after;
    }
  });
}

} // namespace polygeist
} // namespace mlir
