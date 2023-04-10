//===- TransformUtils.h - Polygeist Transform Utilities -----------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for the Polygeist transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H
#define MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H

#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
class AffineForOp;
class AffineParallelOp;
class DominanceInfo;
class IntegerSet;
class LoopLikeOpInterface;
class PatternRewriter;
class RegionBranchOpInterface;

namespace scf {
class ParallelOp;
class ForOp;
} // namespace scf

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

void fully2ComposeAffineMapAndOperands(PatternRewriter &rewriter,
                                       AffineMap *map,
                                       SmallVectorImpl<Value> *operands,
                                       DominanceInfo &DI);
bool isValidIndex(Value val);

//===----------------------------------------------------------------------===//
// Interface classes
//===----------------------------------------------------------------------===//
class IfThenElseBodyInterface {
protected:
  virtual ~IfThenElseBodyInterface() = default;
  virtual void createThenBody(RegionBranchOpInterface) const = 0;
  virtual void createElseBody(RegionBranchOpInterface) const = 0;
};

//===----------------------------------------------------------------------===//
// Loop Versioning Utilities
//===----------------------------------------------------------------------===//

/// Abstract base class to version a loop like operation.
class LoopVersionBuilder : public IfThenElseBodyInterface {
public:
  static std::unique_ptr<LoopVersionBuilder> create(LoopLikeOpInterface);

  /// Version the loop using the \p condition supplied.
  /// Note: this is legal for SCF loops.
  virtual void versionLoop(Value condition) const;

  /// Version the loop using the \p ifCondSet and \p setOperands supplied.
  /// Note: This is legal for affine loops.
  virtual void versionLoop(IntegerSet ifCondSet,
                           SmallVectorImpl<Value> &setOperands) const;

protected:
  LoopVersionBuilder(LoopLikeOpInterface loop) : loop(loop) {}

  void versionLoop(RegionBranchOpInterface) const;

  mutable LoopLikeOpInterface loop; /// The loop to version.
};

/// Concrete class to version a SCF loop operation.
class SCFLoopVersionBuilder : public LoopVersionBuilder {
public:
  SCFLoopVersionBuilder(LoopLikeOpInterface loop) : LoopVersionBuilder(loop) {}

  /// Version the loop using the \p condition supplied.
  void versionLoop(Value condition) const final;

private:
  void createThenBody(RegionBranchOpInterface) const final;
  void createElseBody(RegionBranchOpInterface) const final;
};

/// Concrete class to version an affine loop operation.
class AffineLoopVersionBuilder : public LoopVersionBuilder {
public:
  AffineLoopVersionBuilder(LoopLikeOpInterface loop)
      : LoopVersionBuilder(loop) {}

  /// Version the loop using the \p ifCondSet and \p setOperands supplied.
  void versionLoop(IntegerSet ifCondSet,
                   SmallVectorImpl<Value> &setOperands) const final;

private:
  void createThenBody(RegionBranchOpInterface) const final;
  void createElseBody(RegionBranchOpInterface) const final;
};

//===----------------------------------------------------------------------===//
// Loop Guarding Utilities
//===----------------------------------------------------------------------===//

/// Abstract base class to guard a loop like operation.
class LoopGuardBuilder : public IfThenElseBodyInterface {
public:
  static std::unique_ptr<LoopGuardBuilder> create(LoopLikeOpInterface);

  virtual void guardLoop() const = 0;

protected:
  LoopGuardBuilder(LoopLikeOpInterface loop) : loop(loop) {}

  virtual OperandRange getInitVals() const = 0;
  void guardLoop(RegionBranchOpInterface) const;

  mutable LoopLikeOpInterface loop; /// The loop to guard.
};

/// Abstract class to guard an SCF loop operation.
class SCFLoopGuardBuilder : public LoopGuardBuilder {
public:
  SCFLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

  /// Guard the loop associated with this class.
  void guardLoop() const final;

private:
  Value createCondition() const;
  void createThenBody(RegionBranchOpInterface) const final;
  void createElseBody(RegionBranchOpInterface) const final;
  virtual OperandRange getLowerBounds() const = 0;
  virtual OperandRange getUpperBounds() const = 0;
};

/// Abstract class to guard an affine loop operation.
class AffineLoopGuardBuilder : public LoopGuardBuilder {
public:
  AffineLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createCondition(SmallVectorImpl<Value> &) const;
  void createThenBody(RegionBranchOpInterface) const final;
  void createElseBody(RegionBranchOpInterface) const final;
  virtual void getConstraints(SmallVectorImpl<AffineExpr> &,
                              ArrayRef<AffineExpr>,
                              ArrayRef<AffineExpr>) const = 0;
  virtual OperandRange getLowerBoundsOperands() const = 0;
  virtual OperandRange getUpperBoundsOperands() const = 0;
  virtual AffineMap getLowerBoundsMap() const = 0;
  virtual AffineMap getUpperBoundsMap() const = 0;
};

/// Concrete class to guard an scf:ForOp operation.
class SCFForGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFForGuardBuilder(scf::ForOp loop);

private:
  scf::ForOp getLoop() const;
  OperandRange getInitVals() const final;
  OperandRange getLowerBounds() const final;
  OperandRange getUpperBounds() const final;
};

/// Concrete class to guard an scf:ParallelOp operation.
class SCFParallelGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFParallelGuardBuilder(scf::ParallelOp loop);

private:
  scf::ParallelOp getLoop() const;
  OperandRange getInitVals() const final;
  OperandRange getLowerBounds() const final;
  OperandRange getUpperBounds() const final;
};

/// Concrete class to guard an AffineForOp operation.
class AffineForGuardBuilder : public AffineLoopGuardBuilder {
public:
  AffineForGuardBuilder(AffineForOp loop);

private:
  AffineForOp getLoop() const;
  void getConstraints(SmallVectorImpl<AffineExpr> &, ArrayRef<AffineExpr>,
                      ArrayRef<AffineExpr>) const final;
  OperandRange getInitVals() const final;
  OperandRange getLowerBoundsOperands() const final;
  OperandRange getUpperBoundsOperands() const final;
  AffineMap getLowerBoundsMap() const final;
  AffineMap getUpperBoundsMap() const final;
};

/// Concrete class to guard an AffineParallelOp operation.
class AffineParallelGuardBuilder : public AffineLoopGuardBuilder {
public:
  AffineParallelGuardBuilder(AffineParallelOp loop);

private:
  AffineParallelOp getLoop() const;
  void getConstraints(SmallVectorImpl<AffineExpr> &, ArrayRef<AffineExpr>,
                      ArrayRef<AffineExpr>) const final;
  mlir::Operation::operand_range getInitVals() const final;
  OperandRange getLowerBoundsOperands() const final;
  OperandRange getUpperBoundsOperands() const final;
  AffineMap getLowerBoundsMap() const final;
  AffineMap getUpperBoundsMap() const final;
};

} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H
