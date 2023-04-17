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
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <variant>

namespace mlir {
class AffineForOp;
class AffineIfOp;
class AffineParallelOp;
class DominanceInfo;
class FunctionOpInterface;
class LoopLikeOpInterface;
class PatternRewriter;
class RegionBranchOpInterface;

namespace scf {
class ForOp;
class IfOp;
class ParallelOp;
} // namespace scf

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

void fully2ComposeAffineMapAndOperands(PatternRewriter &rewriter,
                                       AffineMap *map,
                                       SmallVectorImpl<Value> *operands,
                                       DominanceInfo &DI);
bool isValidIndex(Value val);

/// Return true if the given function is potentially a SYCL kernel body
/// function.
bool isPotentialKernelBodyFunc(FunctionOpInterface);

//===----------------------------------------------------------------------===//
// Loop Versioning Utilities
//===----------------------------------------------------------------------===//

/// Represents a loop versioning condition.
class LoopVersionCondition {
public:
  using SCFCondition = Value;

  struct AffineCondition {
    IntegerSet ifCondSet;
    SmallVectorImpl<Value> &setOperands;
  };

  /// Create a loop versioning condition for SCF loops.
  LoopVersionCondition(SCFCondition scfCond) : versionCondition(scfCond) {}

  LoopVersionCondition(AffineCondition affineCond)
      : versionCondition(affineCond) {}

  bool hasSCFCondition() const {
    return std::holds_alternative<SCFCondition>(versionCondition);
  }

  SCFCondition getSCFCondition() const {
    assert(hasSCFCondition() && "expecting valid SCF condition");
    return std::get<SCFCondition>(versionCondition);
  }

  bool hasAffineCondition() const {
    return std::holds_alternative<AffineCondition>(versionCondition);
  }

  AffineCondition getAffineCondition() const {
    assert(hasAffineCondition() && "expecting valid affine condition");
    return std::get<AffineCondition>(versionCondition);
  }

private:
  std::variant<SCFCondition, AffineCondition> versionCondition;
};

/// Version a loop like operation.
class LoopVersionBuilder {
public:
  LoopVersionBuilder(LoopLikeOpInterface loop) : loop(loop) {}

  void versionLoop(const LoopVersionCondition &) const;

protected:
  void createElseBody(scf::IfOp) const;
  void createElseBody(AffineIfOp) const;

  mutable LoopLikeOpInterface loop; /// The loop to version.
};

//===----------------------------------------------------------------------===//
// Loop Guarding Utilities
//===----------------------------------------------------------------------===//

/// Abstract base class to guard a loop like operation.
class LoopGuardBuilder {
public:
  static std::unique_ptr<LoopGuardBuilder> create(LoopLikeOpInterface);
  virtual ~LoopGuardBuilder() = default;

  virtual void guardLoop() const = 0;

protected:
  LoopGuardBuilder(LoopLikeOpInterface loop) : loop(loop) {}

  virtual OperandRange getInitVals() const = 0;
  virtual void createThenBody(RegionBranchOpInterface) const = 0;
  virtual void createElseBody(RegionBranchOpInterface) const = 0;

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

//===----------------------------------------------------------------------===//
// Loop Tools
//===----------------------------------------------------------------------===//

/// A collection of tools for loop transformations.
class LoopTools {
public:
  /// Guard the given loop \p loop.
  void guardLoop(LoopLikeOpInterface loop) const;

  /// Version the given loop \p loop using the condition \p versionCond.
  void versionLoop(LoopLikeOpInterface loop,
                   const LoopVersionCondition &versionCond) const;
};

//===----------------------------------------------------------------------===//
// VersionConditionBuilder
//===----------------------------------------------------------------------===//

/// Build a version condition to check if the given list of accessor pairs
/// overlap.
class VersionConditionBuilder {
public:
  using AccessorType = TypedValue<MemRefType>;
  using AccessorPairType = std::pair<AccessorType, AccessorType>;
  using SCFCondition = LoopVersionCondition::SCFCondition;
  using AffineCondition = LoopVersionCondition::AffineCondition;

  VersionConditionBuilder(
      LoopLikeOpInterface loop,
      ArrayRef<AccessorPairType> requireNoOverlapAccessorPairs)
      : loop(loop), accessorPairs(requireNoOverlapAccessorPairs) {}

  std::unique_ptr<LoopVersionCondition> createCondition() const {
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    SCFCondition scfCond = createSCFCondition(builder, loc);
    return std::make_unique<LoopVersionCondition>(scfCond);
  }

private:
  /// Create a versioning condition suitable for scf::IfOp.
  SCFCondition createSCFCondition(OpBuilder builder, Location loc) const;

  mutable LoopLikeOpInterface loop;
  ArrayRef<AccessorPairType> accessorPairs;
};

} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H
