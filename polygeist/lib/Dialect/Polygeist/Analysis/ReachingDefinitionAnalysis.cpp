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
// Initial Definition
//===----------------------------------------------------------------------===//

InitialDefinition *InitialDefinition::singleton = nullptr;

InitialDefinition *InitialDefinition::getInstance() {
  if (singleton == nullptr)
    singleton = new InitialDefinition();
  return singleton;
}

raw_ostream &operator<<(raw_ostream &os, const InitialDefinition &def) {
  os << "<initial>"
     << "(" << &def << ")";
  return os;
}

//===----------------------------------------------------------------------===//
// Definition
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const Definition &def) {
  if (def.isOperation())
    os << *def.getOperation();
  if (def.isInitialDefinition())
    os << *def.getInitialDefinition();
  return os;
}

bool Definition::operator==(const Definition &other) const {
  llvm::dbgs() << "at line" << __LINE__ << "\n";
  if (isOperation() && other.isOperation())
    return getOperation() == other.getOperation();
  if (isInitialDefinition() && other.isInitialDefinition())
    return true;
  return false;
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
            for (const Definition *def : valModifiers)
              os.indent(6) << *def << "\n";
          }
          os << "\n";
        }
      };

  printMap(lastDef.valueToModifiers, "mods:");
  printMap(lastDef.valueToPotentialModifiers, "pMods:");
  return os;
}

ReachingDefinition::ReachingDefinition(ProgramPoint p)
    : AbstractDenseLattice(p) {
  // Upon creating a new reaching definition at the start of a function, each
  // memory argument is set to have an unknown initial value.
  if (auto *block = p.dyn_cast<Block *>()) {
    if (block->isEntryBlock()) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(block->getParentOp())) {
        for (Value arg : funcOp.getArguments()) {
          if (isa<MemRefType>(arg.getType()))
            setModifier(arg, new Definition());
        }
      }
    }
  }
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
      // TODO: the <initial> definition is added more than once.
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

ChangeResult ReachingDefinition::setModifier(Value val, Definition *def) {
  ReachingDefinition::ModifiersTy &mods = valueToModifiers[val];
  assert((mods.size() != 1 || mods.front() != def) &&
         "seen this modifier already");

  // Set the new modifier and clear out all previous definitions.
  mods.clear();
  mods.insert(def);
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
                                                      Definition *def) {
  return (valueToPotentialModifiers[val].insert(def)) ? ChangeResult::Change
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

std::optional<ArrayRef<Definition *>>
ReachingDefinition::getModifiers(Value val) const {
  if (valueToModifiers.contains(val))
    return valueToModifiers.at(val).getArrayRef();
  return std::nullopt;
}

std::optional<ArrayRef<Definition *>>
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
  ProgramPoint p = lattice->getPoint();

  llvm::dbgs().indent(2) << "Before:\n";
  llvm::dbgs() << *lattice;

  propagateIfChanged(
      lattice, lattice->join(ReachingDefinition::getUnknownDefinition(p)));
  llvm::dbgs().indent(2) << "Updated ReachingDef:\n";
  llvm::dbgs() << *lattice;
}

void ReachingDefinitionAnalysis::visitOperation(
    Operation *op, const ReachingDefinition &before,
    ReachingDefinition *after) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");

  // Upon entering a function we need to create the alias oracle for the
  // current function and transfer the input state.
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    aliasOracles[funcOp] = std::make_unique<AliasOracle>(funcOp, aliasAnalysis);
    auto result = after->join(before);
    propagateIfChanged(after, result);
    LLVM_DEBUG({
      if (result == ChangeResult::Change) {
        llvm::dbgs().indent(2) << "Updated ReachingDef:\n";
        llvm::dbgs() << *after;
      }
    });
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
  after->join(before);

  // Retrieve the alias oracle for the function this operation belongs to.
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  AliasOracle &aliasOracle = *aliasOracles[funcOp];

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
          result |= after->setModifier(val, new Definition(op));
        })
        .Case<MemoryEffects::Write>([&](auto) {
          // A write operation updates the definition of the current value
          // and the definition of its definitely aliased values. It also
          // updates the potential definitions of values that may alias the
          // current value.
          result |= after->setModifier(val, new Definition(op));
          for (Value aliasedVal : aliasOracle.getMustAlias(val))
            result |= after->setModifier(aliasedVal, new Definition(op));
          for (Value aliasedVal : aliasOracle.getMayAlias(val))
            result |=
                after->addPotentialModifier(aliasedVal, new Definition(op));
        })
        .Case<MemoryEffects::Free>([&](auto) {
          // A deallocate operation kills reaching definitions of the
          // current value and of its definitely aliased values. It also
          // kills the potential definitions of values that may alias the
          // current value.
          result |= after->removeModifiers(val);
          for (Value aliasedVal : aliasOracle.getMustAlias(val))
            result |= after->removeModifiers(aliasedVal);
          for (Value aliasedVal : aliasOracle.getMayAlias(val))
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
