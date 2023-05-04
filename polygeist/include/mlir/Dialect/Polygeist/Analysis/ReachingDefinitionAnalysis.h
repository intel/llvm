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
#include "mlir/Dialect/Polygeist/Utils/AliasUtils.h"

namespace mlir {
namespace polygeist {

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReachingDefinition)

  using AbstractDenseLattice::AbstractDenseLattice;
  using ModifiersTy = SetVector<Operation *, SmallVector<Operation *>,
                                SmallPtrSet<Operation *, 2>>;

  /// Union the reaching definitions from \p lattice.
  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Reset the state.
  ChangeResult reset();

  /// Set operation \p op as a definite modifier of value \p val.
  ChangeResult setModifier(Value val, Operation *op);

  /// Remove all definite modifiers of value \p val.
  ChangeResult removeModifiers(Value val);

  /// Add operation \p op as a possible modifier of value \p val.
  ChangeResult addPotentialModifier(Value val, Operation *op);

  /// Remove all potential modifiers of value \p val.
  ChangeResult removePotentialModifiers(Value val);

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
  DenseMap<FunctionOpInterface, std::unique_ptr<AliasQueries>> aliasQueriesMap;
  mlir::AliasAnalysis &aliasAnalysis;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_REACHINGDEFINITIONANALYSIS_H
