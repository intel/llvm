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

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_REACHINGDEFINITIONANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {
namespace sycl {

struct UnderlyingValueLattice;

/// This lattice represents a single underlying value for an SSA value.
class UnderlyingValue {
  friend raw_ostream &operator<<(raw_ostream &, const UnderlyingValue &);

public:
  /// Create an underlying value with a known value.
  explicit UnderlyingValue(std::optional<Value> optVal = std::nullopt)
      : underlyingValue(optVal) {}

  bool operator==(const UnderlyingValue &rhs) const {
    return underlyingValue == rhs.underlyingValue;
  }

  /// Return true is the underlying value is not std::nullopt.
  bool isUninitialized() const;

  /// Retrieve the underlying value;
  Value getUnderlyingValue() const;

  /// Join two underlying values, in case of conflict use a pessimistic value.
  static UnderlyingValue join(const UnderlyingValue &lhs,
                              const UnderlyingValue &rhs);

  /// Look for the underlying value of a \p val.
  static Value getUnderlyingValue(
      Value val,

      function_ref<const UnderlyingValueLattice *(Value)> getLatticeForValue,
      mlir::AliasAnalysis &aliasAnalysis);

  void print(raw_ostream &os) const { os << *this; }

private:
  std::optional<Value> underlyingValue;
};

struct UnderlyingValueLattice : public dataflow::Lattice<UnderlyingValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnderlyingValueLattice)
  using Lattice::Lattice;
};

/// This lattice represents the potential operations that might have modified a
/// memory resource last.
class ReachingDefinition : public dataflow::AbstractDenseLattice {
  friend raw_ostream &operator<<(raw_ostream &, const ReachingDefinition &);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReachingDefinition)

  using AbstractDenseLattice::AbstractDenseLattice;
  using ModifiersTy = SetVector<Operation *, SmallVector<Operation *>,
                                SmallPtrSet<Operation *, 2>>;

  /// Union the reaching definitions from \p lattice.
  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Reset the state.
  ChangeResult reset();

  /// Set the operation \p op as a definite modifier of value \p val.
  ChangeResult setModifier(Value val, Operation *op);

  /// Add operation \p op as a possible modifier of value \p val.
  ChangeResult addPotentialModifier(Value val, Operation *op);

  /// Get the operations that have modified \p val.
  std::optional<ArrayRef<Operation *>> getModifiers(Value val) const;

  /// Get the operations that have possibly modified \p val.
  std::optional<ArrayRef<Operation *>> getPotentialModifiers(Value val) const;

  void print(raw_ostream &os) const override { os << *this; }

private:
  /// A map between a memory resource (Value) and the operations that have
  /// modified the memory resource last.
  DenseMap<Value, ModifiersTy> valueToModifiers;

  /// A map between a memory resource (Value) and the operations that
  /// might have modified the memory resource last because of aliasing.
  DenseMap<Value, ModifiersTy> valueToPotentialModifiers;
};

class UnderlyingValueAnalysis
    : public dataflow::SparseDataFlowAnalysis<UnderlyingValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  UnderlyingValueAnalysis(DataFlowSolver &solver,
                          mlir::AliasAnalysis &aliasAnalysis)
      : SparseDataFlowAnalysis<UnderlyingValueLattice>(solver),
        aliasAnalysis(aliasAnalysis) {}

  void visitOperation(Operation *op,
                      ArrayRef<const UnderlyingValueLattice *> operands,
                      ArrayRef<UnderlyingValueLattice *> results) override;

  void setToEntryState(UnderlyingValueLattice *lattice) override;

private:
  mlir::AliasAnalysis &aliasAnalysis;
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
  mlir::AliasAnalysis &aliasAnalysis;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
