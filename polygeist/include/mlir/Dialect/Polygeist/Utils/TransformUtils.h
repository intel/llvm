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
// Loop Versioning Utilities
//===----------------------------------------------------------------------===//

class LoopVersionBuilder {
public:
  LoopVersionBuilder(LoopLikeOpInterface loop);
  LoopVersionBuilder(const LoopVersionBuilder &) = delete;
  LoopVersionBuilder(LoopVersionBuilder &&) = delete;
  void operator=(const LoopVersionBuilder &) = delete;
  void operator=(LoopVersionBuilder &&) = delete;
  virtual ~LoopVersionBuilder() = default;

  void versionLoop();

protected:
  static Block &getThenBlock(RegionBranchOpInterface ifOp) {
    return ifOp->getRegion(0).front();
  }
  static Block &getElseBlock(RegionBranchOpInterface ifOp) {
    return ifOp->getRegion(1).front();
  }

  RegionBranchOpInterface ifOp;
  mutable OpBuilder builder;
  mutable LoopLikeOpInterface loop;

private:
  void replaceUsesOfLoopReturnValues() const;
  virtual void createIfOp() = 0;
  virtual void createThenBody() const = 0;
  virtual void createElseBody() const = 0;
};

class SCFLoopVersionBuilder : public LoopVersionBuilder {
public:
  SCFLoopVersionBuilder(LoopLikeOpInterface loop);

protected:
  scf::IfOp getIfOp() const;

private:
  virtual Value createCondition() const = 0;
  void createIfOp() final;
  void createThenBody() const override;
  void createElseBody() const override;
};

class AffineLoopVersionBuilder : public LoopVersionBuilder {
public:
  AffineLoopVersionBuilder(LoopLikeOpInterface loop);

protected:
  AffineIfOp getIfOp() const;

private:
  virtual IntegerSet createCondition(SmallVectorImpl<Value> &) const = 0;
  void createIfOp() final;
  void createThenBody() const override;
  void createElseBody() const override;
};

//===----------------------------------------------------------------------===//
// Loop Guarding Utilities
//===----------------------------------------------------------------------===//

class LoopGuardBuilder {
public:
  static std::unique_ptr<LoopGuardBuilder> create(LoopLikeOpInterface loop);

  LoopGuardBuilder() = default;
  LoopGuardBuilder(const LoopGuardBuilder &) = delete;
  LoopGuardBuilder(LoopGuardBuilder &&) = delete;
  void operator=(const LoopGuardBuilder &) = delete;
  void operator=(LoopGuardBuilder &&) = delete;
  virtual ~LoopGuardBuilder() = default;

  virtual void guardLoop() = 0;

protected:
  virtual OperandRange getInitVals() const = 0;
};

class SCFLoopGuardBuilder : public LoopGuardBuilder,
                            public SCFLoopVersionBuilder {
public:
  SCFLoopGuardBuilder(LoopLikeOpInterface loop);

  void guardLoop() final { versionLoop(); }

private:
  void createElseBody() const final;
};

class SCFForGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFForGuardBuilder(scf::ForOp loop);

private:
  scf::ForOp getLoop() const;
  Value createCondition() const final;
  OperandRange getInitVals() const final;
};

class SCFParallelGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFParallelGuardBuilder(scf::ParallelOp loop);

private:
  scf::ParallelOp getLoop() const;
  Value createCondition() const final;
  OperandRange getInitVals() const final;
};

class AffineLoopGuardBuilder : public LoopGuardBuilder,
                               public AffineLoopVersionBuilder {
public:
  AffineLoopGuardBuilder(LoopLikeOpInterface loop)
      : LoopGuardBuilder(), AffineLoopVersionBuilder(loop) {}
  void guardLoop() final { versionLoop(); }

private:
  void createElseBody() const final;
  IntegerSet createCondition(SmallVectorImpl<Value> &) const final;
  virtual void getConstraints(SmallVectorImpl<AffineExpr> &,
                              ArrayRef<AffineExpr>,
                              ArrayRef<AffineExpr>) const = 0;
  virtual OperandRange getLowerBoundsOperands() const = 0;
  virtual OperandRange getUpperBoundsOperands() const = 0;
  virtual AffineMap getLowerBoundsMap() const = 0;
  virtual AffineMap getUpperBoundsMap() const = 0;
};

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
