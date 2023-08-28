//===- UniformityAnalysis.h - Uniformity Analysis ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a dataflow analysis which attempts to classify values as
// either uniform (all threads agree on the content of the value) or not
// uniform.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"

namespace mlir {

class LoopLikeOpInterface;
namespace polygeist {

/// Returns true if the given operation \p op may not be executed by all
/// threads.
bool isDivergent(Operation *op, DataFlowSolver &solver);

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
    : public dataflow::SparseForwardDataFlowAnalysis<UniformityLattice>,
      public RequiredDataFlowAnalyses<UniformityAnalysis> {
  friend class RequiredDataFlowAnalyses;
  friend raw_ostream &operator<<(raw_ostream &, const UniformityAnalysis &);

public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  UniformityAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis<UniformityLattice>(solver),
        solver(solver) {}

  /// Set the initial state at an entry point. If the entry point is a kernel
  /// its argument are uniform. Otherwise the arguments have unknown uniformity.
  void setToEntryState(UniformityLattice *lattice) override;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const UniformityLattice *> operands,
                      ArrayRef<UniformityLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow.
  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<UniformityLattice *> argLattices,
                                    unsigned firstIndex) override;

private:
  /// Load required dataflow analyses.
  static void loadRequiredAnalysesImpl(DataFlowSolverWrapper &solver) {
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.loadWithRequiredAnalysis<ReachingDefinitionAnalysis>(
        solver.getAliasAnalysis());
  }

  /// Analyze an operation \p op that has memory side effects and uniform
  /// operands.
  void analyzeMemoryEffects(Operation *op,
                            ArrayRef<const UniformityLattice *> operands,
                            ArrayRef<UniformityLattice *> results);

  std::optional<Uniformity>
  getInductionVariableUniformity(LoopLikeOpInterface loop);

  /// Collect the branch conditions that dominate each of the modifiers \p
  /// mods.
  SmallVector<IfCondition>
  collectBranchConditions(const ReachingDefinition::ModifiersTy &mods);

  /// Return true if all the \p conditions have uniformity that is initialized.
  /// The \p op argument is the operation the conditions are for.
  bool isUniformityInitialized(ArrayRef<IfCondition> conditions, Operation *op);

  /// Return true if all the modifiers \p mods have operands with uniformity
  /// that is initialized. The \p op argument is the operation the modifiers are
  /// for.
  bool isUniformityInitialized(const ReachingDefinition::ModifiersTy &mods,
                               Operation *op);

  /// Return true if any of the modifiers \p mods store a value with
  /// uniformity equal to \p kind.
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

  /// Return true if any of the \p operands has uniformity of the given \p
  /// kind.
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

  /// Return true if any of the \p conditions has uniformity of the given \p
  /// kind.
  bool anyOfUniformityIs(ArrayRef<IfCondition> conditions,
                         Uniformity::Kind kind) {
    return llvm::any_of(conditions, [&](const IfCondition &cond) {
      return cond.perform(
          [&](ValueRange values) { return anyOfUniformityIs(values, kind); });
    });
  }

  /// Propagate \p uniformity to all \p results if necessary.
  void propagateAllIfChanged(ArrayRef<UniformityLattice *> results,
                             const Uniformity &uniformity);

  /// Propagate \p uniformity to all \p values if necessary.
  void propagateAllIfChanged(const ValueRange values,
                             const Uniformity &uniformity);

private:
  DataFlowSolver &solver;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H
