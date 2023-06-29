//===- UniformityAnalysis.h - Uniformity Analysis ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a simple intra-procedural uniformity analysis which
// attempts to classify values as either uniform (all threads agree on the
// content of the value) or not uniform.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// Uniformity
//===----------------------------------------------------------------------===//

/// This lattice value represents the uniformity of a value.
class Uniformity {
  friend raw_ostream &operator<<(raw_ostream &, const Uniformity &);

public:
  enum class Kind { Unknown, Uniform, NonUniform };

  /// Construct an uninitialized uniformity.
  explicit Uniformity() = default;

  /// Construct a uniformity of a known kind.
  Uniformity(Kind kind) : kind(kind) {}

  bool operator==(const Uniformity &rhs) const { return kind == rhs.kind; }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !kind.has_value(); }

  /// Get the uniformity of the value associated with this object.
  Kind getKind() const {
    assert(!isUninitialized() && "Uniformity is not initialized");
    return *kind;
  }

  /// Whether the state is [unknown | uniform | non-uniform].
  static bool isUnknown(Kind kind) { return kind == Kind::Unknown; }
  static bool isUniform(Kind kind) { return kind == Kind::Uniform; }
  static bool isNonUniform(Kind kind) { return kind == Kind::NonUniform; }

  bool isUnknown() const { return Uniformity::isUnknown(getKind()); }
  bool isUniform() const { return Uniformity::isUniform(getKind()); }
  bool isNonUniform() const { return Uniformity::isNonUniform(getKind()); }

  /// Create a state where the uniformity info is uninitialized. This happens
  /// when the state hasn't been set during the analysis.
  static Uniformity getUninitialized() { return Uniformity(); }

  /// Create a state for \p val where the uniformity info is unknown.
  static Uniformity getUnknown() { return Uniformity(Kind::Unknown); }

  /// Create a state for \p val where the uniformity info is uniform.
  static Uniformity getUniform() { return Uniformity(Kind::Uniform); }

  /// Create a state for \p val where the uniformity info is non-uniform.
  static Uniformity getNonUniform() { return Uniformity(Kind::NonUniform); }

  /// Join two uniformity info.
  static Uniformity join(const Uniformity &lhs, const Uniformity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    if (lhs.isNonUniform() && rhs.isUniform())
      return lhs;
    if (rhs.isNonUniform() && lhs.isUniform())
      return rhs;
    return getUnknown();
  }

  void print(raw_ostream &) const;

private:
  std::optional<Kind> kind;
};

class UniformityLattice : public dataflow::Lattice<Uniformity> {
public:
  using Lattice::Lattice;

  void onUpdate(DataFlowSolver *solver) const override;
};

//===----------------------------------------------------------------------===//
// UniformityAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements sparse uniformity analysis, which attempts to
/// determine whether values are uniform or not w.r.t. threads. A uniform value
/// is a value for which all threads agree on its content.
class UniformityAnalysis
    : public dataflow::SparseDataFlowAnalysis<UniformityLattice> {
  friend raw_ostream &operator<<(raw_ostream &, const UniformityAnalysis &);

public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  UniformityAnalysis(DataFlowSolver &solver, AliasAnalysis &aliasAnalysis);

  LogicalResult initialize(Operation *top) override;

  /// At an entry point, we cannot reason about uniformity.
  void setToEntryState(UniformityLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(Uniformity::getUnknown()));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const UniformityLattice *> operands,
                      ArrayRef<UniformityLattice *> results) override;

private:
  /// Analyze an operation \p op that has memory side effects and uniform
  /// operands.
  void analyzeMemoryEffects(Operation *op,
                            ArrayRef<const UniformityLattice *> operands,
                            ArrayRef<UniformityLattice *> results);

  /// Collect the branch conditions that dominate each of the modifiers \p mods.
  SetVector<Value>
  collectBranchConditions(const ReachingDefinition::ModifiersTy &mods);

  /// Return true if all the modifiers \p mods have operands with known uniform
  /// that is initialized. The \p op argument is the operation the modifiers
  /// are for.
  bool canComputeUniformity(const ReachingDefinition::ModifiersTy &mods,
                            Operation *op);

  /// Return true if any of the modifiers \p mods store a value with uniformity
  /// equal to \p kind.
  bool anyModifierUniformityIs(const ReachingDefinition::ModifiersTy &mods,
                               Uniformity::Kind kind);

  /// Return true is any of the \p operands uniformity is uninitialized.
  bool
  anyOfUniformityIsUninitialized(ArrayRef<const UniformityLattice *> operands) {
    return llvm::any_of(operands, [&](const UniformityLattice *lattice) {
      return lattice->getValue().isUninitialized();
    });
  }

  /// Return true is any of the \p values uniformity is uninitialized.
  bool anyOfUniformityIsUninitialized(const ValueRange values) {
    return llvm::any_of(values, [&](Value value) {
      UniformityLattice *lattice = getLatticeElement(value);
      return lattice->getValue().isUninitialized();
    });
  }

  /// Return true if any of the \p operands has uniformity of the given \p kind.
  bool anyOfUniformityIs(ArrayRef<const UniformityLattice *> operands,
                         Uniformity::Kind kind) {
    return llvm::any_of(operands, [&](const UniformityLattice *lattice) {
      return lattice->getValue().getKind() == kind;
    });
  }

  /// Return true if any of the \p values has uniformity of the given \p kind.
  bool anyOfUniformityIs(const ValueRange values, Uniformity::Kind kind) {
    return llvm::any_of(values, [&](Value value) {
      UniformityLattice *lattice = getLatticeElement(value);
      return lattice->getValue().getKind() == kind;
    });
  }

  /// Propagate \p uniformity to all \p results if necessary.
  void propagateAllIfChanged(ArrayRef<UniformityLattice *> results,
                             Uniformity &&uniformity);

  /// Propagate \p uniformity to all \p values if necessary.
  void propagateAllIfChanged(const ValueRange values, Uniformity &&uniformity);

private:
  DataFlowSolver internalSolver;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H
