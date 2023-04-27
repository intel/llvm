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
      function_ref<const UnderlyingValueLattice *(Value)> getLatticeForValue);

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

  /// Add the reaching definitions from \p lattice.
  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Reset the state.
  ChangeResult reset();

  /// Set the last modifier of value \p val to be operation \p op.
  ChangeResult set(Value val, Operation *op);

  /// Get the operations that have potentially modified \p val last.
  std::optional<ArrayRef<Operation *>> getLastModifiers(Value val) const;

  void print(raw_ostream &os) const override { os << *this; }

private:
  /// A map between a memory resource (Value) and the potential operations that
  /// might have modified the memory resource last.
  DenseMap<Value, ModifiersTy> valueToModifiers;
};

class UnderlyingValueAnalysis
    : public dataflow::SparseDataFlowAnalysis<UnderlyingValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

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

class ReachingDefinitionAnalysis
    : public dataflow::DenseDataFlowAnalysis<ReachingDefinition> {
public:
  using DenseDataFlowAnalysis::DenseDataFlowAnalysis;

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
  void setToEntryState(ReachingDefinition *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
