//===- ReachingDefinitionAnalysis.h - Reaching Definitions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a simple reaching definition analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_REACHINGDEFINITIONANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Polygeist/Utils/AliasUtils.h"
#include <set>
#include <variant>

namespace mlir {
namespace polygeist {

/// Represents the initial definition of a memory resource.
class InitialDefinition {};

/// Represents either an operation that modifies a memory resource, or the
/// initial definition.
class Definition {
  friend raw_ostream &operator<<(raw_ostream &, const Definition &);

public:
  Definition(Operation *op) : def(op) {}
  Definition() : def(InitialDefinition()) {}

  bool operator==(const Definition &other) const;
  bool operator!=(const Definition &other) const { return !(*this == other); }
  bool operator<(const Definition &other) const;

  bool isOperation() const { return std::holds_alternative<Operation *>(def); }
  bool isInitialDefinition() const {
    return std::holds_alternative<InitialDefinition>(def);
  }

  Operation *getOperation() const {
    assert(isOperation() && "expecting operation");
    return std::get<Operation *>(def);
  }

private:
  std::variant<Operation *, InitialDefinition> def;
};

/// This lattice represents the set of operations that might have modified a
/// memory resource last (in at least one control flow path).
/// Note: Two sets of definitions are tracked:
///   - the set of operations that have modified a memory resource, and
///   - the set of (may) aliased operations that might have modified a memory
///     resource
/// For example given:
///   func.func @foo(%v1: i32, %v2: i32, %ptr1: memref<i32>, %ptr2: memref<i32>)
///     scf.if %cond {
///       memref.store %v1, %ptr1[] {tag_name = "a"}: memref<i32>
///     } else {
///       memref.store %v2, %ptr2[] {tag_name = "b"} : memref<i32>
///     }
///     ... = memref.load %ptr1[] {tag = "load"} : memref<?xi32>
///   }
///
/// The 2 sets reaching the load of 'ptr1' are:
///   - definitions: {a}
///   - potential definitions: {b}
///
class ReachingDefinition : public dataflow::AbstractDenseLattice {
  friend raw_ostream &operator<<(raw_ostream &, const ReachingDefinition &);

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  using ModifiersTy = std::set<Definition>;

  explicit ReachingDefinition(ProgramPoint p);

  /// Construct an unknown definition (represents the incoming definition at an
  /// entry point).
  static ReachingDefinition getUnknownDefinition(ProgramPoint p) {
    return ReachingDefinition(p);
  }

  /// Union the reaching definitions from \p lattice.
  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Reset the state.
  ChangeResult reset();

  /// Set definition \p def as a definite modifier of value \p val.
  ChangeResult setModifier(Value val, Definition def);

  /// Remove all definite modifiers of value \p val.
  ChangeResult removeModifiers(Value val);

  /// Add definition \p def as a possible modifier of value \p val.
  ChangeResult addPotentialModifier(Value val, Definition def);

  /// Remove all potential modifiers of value \p val.
  ChangeResult removePotentialModifiers(Value val);

  /// Get the definitions that have modified \p val.
  std::optional<ModifiersTy> getModifiers(Value val) const;

  /// Get the definition that have possibly modified \p val.
  std::optional<ModifiersTy> getPotentialModifiers(Value val) const;

  /// Return the unique definition for the operand at index \p opIndex in
  /// operation \p op, or std::nullopt if it does not have a unique definition.
  static std::optional<Definition>
  getUniqueDefinition(unsigned opIndex, Operation *op, DataFlowSolver &solver);

  void print(raw_ostream &os) const override { os << *this; }

private:
  /// A map between a memory resource (Value) and the definitions that have
  /// modified the memory resource last.
  DenseMap<Value, ModifiersTy> valueToModifiers;

  /// A map between a memory resource (Value) and the definitions that
  /// might have modified the memory resource last because of aliasing.
  DenseMap<Value, ModifiersTy> valueToPotentialModifiers;
};

class ReachingDefinitionAnalysis
    : public dataflow::DenseDataFlowAnalysis<ReachingDefinition> {
public:
  using DenseDataFlowAnalysis::DenseDataFlowAnalysis;

  ReachingDefinitionAnalysis(DataFlowSolver &solver,
                             mlir::AliasAnalysis &aliasAnalysis)
      : DenseDataFlowAnalysis<ReachingDefinition>(solver),
        aliasAnalysis(aliasAnalysis) {}

  /// Visit operation \p op and update the output state \p after with the
  /// contributions of this operation:
  /// - if the operation has no memory effects, no changes are made
  /// - if the operation allocates a resource, its reaching definitions are
  ///   set to empty
  /// - if the operation writes to a resource, its reaching definition is set
  ///   to the written value.
  void visitOperation(Operation *op, const ReachingDefinition &before,
                      ReachingDefinition *after) override;

  /// Set the initial state (nothing is known about reaching definitions).
  void setToEntryState(ReachingDefinition *lattice) override;

private:
  DenseMap<FunctionOpInterface, std::unique_ptr<AliasOracle>> aliasOracles;
  AliasAnalysis &aliasAnalysis;
};

class UnderlyingValue {
public:
  /// Create an underlying value state with a known underlying value.
  explicit UnderlyingValue(std::optional<Value> underlyingValue = std::nullopt)
      : underlyingValue(underlyingValue) {}

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !underlyingValue.has_value(); }

  /// Returns the underlying value.
  Value getUnderlyingValue() const {
    assert(!isUninitialized());
    return *underlyingValue;
  }

  /// Join two underlying values. If there are conflicting underlying values,
  /// go to the pessimistic value.
  static UnderlyingValue join(const UnderlyingValue &lhs,
                              const UnderlyingValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return lhs.underlyingValue == rhs.underlyingValue
               ? lhs
               : UnderlyingValue(Value{});
  }

  /// Compare underlying values.
  bool operator==(const UnderlyingValue &rhs) const {
    return underlyingValue == rhs.underlyingValue;
  }

  void print(raw_ostream &os) const { os << underlyingValue; }

private:
  std::optional<Value> underlyingValue;
};

struct UnderlyingValueLattice : public dataflow::Lattice<UnderlyingValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnderlyingValueLattice)
  using Lattice::Lattice;
};

/// An analysis that uses forwarding of values along control-flow and callgraph
/// edges to determine single underlying values for block arguments.
class UnderlyingValueAnalysis
    : public dataflow::SparseDataFlowAnalysis<UnderlyingValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  /// Retrieve the underlying value for \p value given the function
  /// \p getUnderlyingValueFn.
  static Value getUnderlyingValue(
      Value value,
      function_ref<const UnderlyingValueLattice *(Value)> getUnderlyingValueFn);

  /// The underlying value of the results of an operation are not known.
  void visitOperation(Operation *op,
                      ArrayRef<const UnderlyingValueLattice *> operands,
                      ArrayRef<UnderlyingValueLattice *> results) override {
    setAllToEntryStates(results);
  }

  /// At an entry point, the underlying value of a value is itself.
  void setToEntryState(UnderlyingValueLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(UnderlyingValue{lattice->getPoint()}));
  }
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
