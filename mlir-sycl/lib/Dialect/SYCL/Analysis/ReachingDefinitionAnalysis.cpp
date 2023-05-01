//===- ReachingDefinitionAnalysis.cpp - Reaching Defs Analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "reaching-definition-analysis"

namespace mlir {
namespace sycl {

//===----------------------------------------------------------------------===//
// AliasUtilities
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const AliasUtilities &aliasUtilities) {
  auto printMap = [&os](const DenseMap<Value, SetVector<Value>> &map,
                        StringRef title) {
    for (const auto &entry : map) {
      Value val = entry.first;
      const SetVector<Value> &mustAlias = entry.second;
      os.indent(4) << val << "\n";
      os.indent(4) << title << ":\n";
      if (mustAlias.empty())
        os.indent(6) << "<none>\n";
      else {
        for (Value aliasedVal : mustAlias)
          os.indent(6) << aliasedVal << "\n";
      }
      os << "\n";
    }
  };

  os.indent(4) << aliasUtilities.funcOp.getName() << ":\n";
  printMap(aliasUtilities.valueToMustAliasValues, "mustAliases");
  printMap(aliasUtilities.valueToMayAliasValues, "mayAliases");
  return os;
}

AliasUtilities::AliasUtilities(FunctionOpInterface &funcOp,
                               mlir::AliasAnalysis &aliasAnalysis)
    : funcOp(funcOp), aliasAnalysis(aliasAnalysis) {
  /// Collect all operations that reference a memory resource in the given
  /// function and initialize the maps.
  SetVector<Value> memoryResources = collectMemoryResourcesIn(funcOp);
  for (Value val1 : memoryResources) {
    for (Value val2 : memoryResources) {
      if (val1 == val2)
        continue;

      AliasResult aliasResult = aliasAnalysis.alias(val1, val2);
      if (aliasResult.isMust())
        valueToMustAliasValues[val1].insert(val2);
      else if (aliasResult.isMay())
        valueToMayAliasValues[val1].insert(val2);
    }
  }
}

SetVector<Value>
AliasUtilities::collectMemoryResourcesIn(FunctionOpInterface funcOp) {
  SetVector<Value> memoryResources;
  for (BlockArgument arg : funcOp.getArguments()) {
    if (isa<MemRefType>(arg.getType()))
      memoryResources.insert(arg);
  }

  for (Block &block : funcOp) {
    for (Operation &op : block.without_terminator()) {
      auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
      if (!memoryEffectOp)
        continue;

      SmallVector<MemoryEffects::EffectInstance> effects;
      memoryEffectOp.getEffects(effects);
      for (const auto &effect : effects) {
        if (Value val = effect.getValue())
          memoryResources.insert(val);
      }
    }
  }
  return memoryResources;
}

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
      if (currentModifiers != newModifiers) {
        currentModifiers.insert(newModifiers.begin(), newModifiers.end());
        result |= ChangeResult::Change;
      }
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
  if (mods.size() == 1 && mods.front() == op)
    return ChangeResult::NoChange;

  // Set the new modifier and clear out all previous definitions.
  mods.clear();
  mods.insert(op);
  valueToPotentialModifiers[val].clear();
  return ChangeResult::Change;
}

ChangeResult ReachingDefinition::addPotentialModifier(Value val,
                                                      Operation *op) {
  valueToPotentialModifiers[val].insert(op);
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
  auto it = valueToPotentialModifiers.find(val);
  return (it != valueToPotentialModifiers.end()) ? it->second.getArrayRef()
                                                 : std::nullopt;
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

  // Initialize the analysis for the current function.
  if (isa<FunctionOpInterface>(op)) {
    auto funcOp = cast<FunctionOpInterface>(*op);
    aliasUtilities[funcOp] =
        std::make_unique<AliasUtilities>(funcOp, aliasAnalysis);
    LLVM_DEBUG(llvm::dbgs() << "Initialized aliasUtilities for "
                            << funcOp.getName() << ":\n");
    return setToEntryState(after);
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
  after->join(before);

  // Retrieve the alias utilities for the function this operation belongs to.
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  AliasUtilities &aliasUtils = *aliasUtilities[funcOp];

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

    // Update the reaching definition for the current value.
    result |= after->setModifier(val, op);

    // Write operations might also update the reaching definitions of values
    // that are aliased to the current one.
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      for (Value aliasedVal : aliasUtils.getMustAlias(val))
        result |= after->setModifier(aliasedVal, op);
      for (Value aliasedVal : aliasUtils.getMayAlias(val))
        result |= after->addPotentialModifier(aliasedVal, op);
    }
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
