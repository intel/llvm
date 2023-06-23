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

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

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

  /// Whether the state is [unknown | uniform | non-uniform].
  bool isUnknown() const { return kind == Kind::Unknown; }
  bool isUniform() const { return kind == Kind::Uniform; }
  bool isNonUniform() const { return kind == Kind::NonUniform; }

  /// Get the uniformity of the value associated with this object.
  Kind getKind() const {
    assert(!isUninitialized() && "Uniformity is not initialized");
    return *kind;
  }

  /// Create a state where the uniformity info is uninitialized. This happens
  /// when the state hasn't been set during the analysis.
  static Uniformity getUninitialized() { return Uniformity(); }

  /// Create a state for \p val where the uniformity info is unknown.
  static Uniformity getUnknown() { return Uniformity(Kind::Unknown); }

  /// Create a state for \p val where the uniformity info is uniform.
  static Uniformity getUniform() { return Uniformity(Kind::Uniform); }

  /// Create a state for \p val where the uniformity info is non-uniform.
  static Uniformity getNonUniform() { return Uniformity(Kind::NonUniform); }

  /// Join two uniformity info
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

  UniformityAnalysis(DataFlowSolver &solver)
      : SparseDataFlowAnalysis<UniformityLattice>(solver) {}

  /// At an entry point, we cannot reason about uniformity.
  void setToEntryState(UniformityLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(Uniformity::getUnknown()));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const UniformityLattice *> operands,
                      ArrayRef<UniformityLattice *> results) override;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_UNIFORMITYANALYSIS_H
