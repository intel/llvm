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

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
class AffineIfOp;
class AffineMap;
class AffineParallelOp;
class Block;
class DominanceInfo;
class IntegerSet;
class LoopLikeOpInterface;
class OperandRange;
class Operation;
class PatternRewriter;
class RegionBranchOpInterface;
class Value;

namespace scf {
class IfOp;
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
// If/then/else builder
//===----------------------------------------------------------------------===//

/// Abstract class to build an if operation.
class IfThenElseBuilder {
public:
  virtual ~IfThenElseBuilder() = default;

  virtual RegionBranchOpInterface createIfOp(Operation::result_range results,
                                             OpBuilder &builder,
                                             Location loc) const = 0;
};

/// Abstract class to build an scf::IfOp operation.
class SCFIfBuilder : public IfThenElseBuilder {
public:
  RegionBranchOpInterface createIfOp(Operation::result_range results,
                                     OpBuilder &builder,
                                     Location loc) const final;
  virtual Value createCondition() const = 0;
};

/// Abstract class to build an AffineIfOp operation.
class AffineIfBuilder : public IfThenElseBuilder {
public:
  RegionBranchOpInterface createIfOp(Operation::result_range results,
                                     OpBuilder &builder,
                                     Location loc) const final;
  virtual IntegerSet createCondition(SmallVectorImpl<Value> &) const = 0;
};

//===----------------------------------------------------------------------===//
// Loop Versioning Utilities
//===----------------------------------------------------------------------===//

/// Abstract class to version a loop like operation.
class LoopVersionBuilder {
public:
  LoopVersionBuilder(LoopLikeOpInterface loop) : loop(loop) {}
  virtual ~LoopVersionBuilder() = default;

  virtual void versionLoop() const = 0;

protected:
  void versionLoop(RegionBranchOpInterface) const;

  mutable LoopLikeOpInterface loop;

private:
  virtual void createThenBody(RegionBranchOpInterface) const = 0;
  virtual void createElseBody(RegionBranchOpInterface) const = 0;
};

/// Abstract class to version a SCF loop operation.
class SCFLoopVersionBuilder : public LoopVersionBuilder, public SCFIfBuilder {
public:
  SCFLoopVersionBuilder(LoopLikeOpInterface loop) : LoopVersionBuilder(loop) {}
  virtual ~SCFLoopVersionBuilder() = 0;

  void versionLoop() const final;

private:
  void createThenBody(RegionBranchOpInterface) const override;
  void createElseBody(RegionBranchOpInterface) const override;
};

/// Abstract class to version a affine loop operation.
class AffineLoopVersionBuilder : public LoopVersionBuilder,
                                 public AffineIfBuilder {
public:
  AffineLoopVersionBuilder(LoopLikeOpInterface loop)
      : LoopVersionBuilder(loop) {}
  virtual ~AffineLoopVersionBuilder() = 0;

  void versionLoop() const final;

private:
  void createThenBody(RegionBranchOpInterface) const override;
  void createElseBody(RegionBranchOpInterface) const override;
};

//===----------------------------------------------------------------------===//
// Loop Guarding Utilities
//===----------------------------------------------------------------------===//

/// Abstract class to guard loop like operation.
class LoopGuardBuilder {
public:
  static std::unique_ptr<LoopGuardBuilder> create(LoopLikeOpInterface loop);

  LoopGuardBuilder(LoopLikeOpInterface loop) : loop(loop) {}
  virtual ~LoopGuardBuilder() = default;

  virtual void guardLoop() const = 0;

protected:
  void guardLoop(RegionBranchOpInterface) const;
  virtual OperandRange getInitVals() const = 0;

  mutable LoopLikeOpInterface loop;

private:
  virtual void createThenBody(RegionBranchOpInterface) const = 0;
  virtual void createElseBody(RegionBranchOpInterface) const = 0;
};

/// Abstract class to guard an SCF loop operation.
class SCFLoopGuardBuilder : public LoopGuardBuilder, public SCFIfBuilder {
public:
  SCFLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

  void guardLoop() const final;

private:
  void createThenBody(RegionBranchOpInterface) const final;
  void createElseBody(RegionBranchOpInterface) const final;
};

/// Abstract class to guard an affine loop operation.
class AffineLoopGuardBuilder : public LoopGuardBuilder, public AffineIfBuilder {
public:
  AffineLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createCondition(SmallVectorImpl<Value> &) const final;
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
  Value createCondition() const final;
  OperandRange getInitVals() const final;
};

/// Concrete class to guard an scf:ParallelOp operation.
class SCFParallelGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFParallelGuardBuilder(scf::ParallelOp loop);

private:
  scf::ParallelOp getLoop() const;
  Value createCondition() const final;
  OperandRange getInitVals() const final;
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
