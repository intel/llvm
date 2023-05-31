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

#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <set>
#include <variant>

namespace mlir {
class CallOpInterface;
class DataFlowSolver;
class DominanceInfo;
class FunctionOpInterface;
class LoopLikeOpInterface;
class PatternRewriter;
class RegionBranchOpInterface;

namespace affine {
class AffineForOp;
class AffineIfOp;
class AffineParallelOp;
} // namespace affine

namespace scf {
class ForOp;
class IfOp;
class ParallelOp;
} // namespace scf

namespace sycl {
class AccessorPtrValue;
} // namespace sycl

namespace polygeist {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

void fully2ComposeAffineMapAndOperands(PatternRewriter &rewriter,
                                       AffineMap *map,
                                       SmallVectorImpl<Value> *operands,
                                       DominanceInfo &DI);
bool isValidIndex(Value val);

/// Returns true if the given function has 'linkonce_odr' LLVM  linkage.
bool isLinkonceODR(FunctionOpInterface);

/// Change the linkage and visibility of the given function to private.
void privatize(FunctionOpInterface);

/// Returns true if the given call is a tail call.
bool isTailCall(CallOpInterface);

/// Returns the maximum depth from any GPU kernel.
/// Returns std::nullopt if the call is not called from a GPU kernel.
/// For example:
/// Call chains:
///   GPUKernel1 -> func1 (depth 1) -> func2 (depth 2)
///   GPUKernel2 -> func2 (depth 1)
/// =>
///   getMaxDepthFromAnyGPUKernel(func1) returns 1.
///   getMaxDepthFromAnyGPUKernel(func2) returns 2.
Optional<unsigned> getMaxDepthFromAnyGPUKernel(FunctionOpInterface);

/// Returns true if the given function is potentially a SYCL kernel body
/// function. The SYCL kernel body function is created by SemaSYCL in clang for
/// the body of the SYCL kernel, e.g., code in parallel_for.
/// TODO: add an attribute to the call operator of the SYCL kernel functor in
/// SemaSYCL in clang, to identify SYCL kernel body function accurately.
bool isPotentialKernelBodyFunc(FunctionOpInterface);

/// Return the accessor used by \p op if found, and std::nullopt otherwise.
Optional<Value> getAccessorUsedByOperation(const Operation &op);

/// Determine whether a value is a known integer value.
std::optional<APInt> getConstIntegerValue(Value val, DataFlowSolver &solver);

/// Record the \p block parent operations with the specified type \tparam T.
template <typename T> SetVector<T> getParentsOfType(Block &block);

/// Retrieve operations with type \tparam T in \p funcOp.
template <typename T>
SetVector<T> getOperationsOfType(FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Versioning Utilities
//===----------------------------------------------------------------------===//

/// Represents a versioning condition.
class VersionCondition {
public:
  using SCFCondition = Value;

  struct AffineCondition {
    IntegerSet ifCondSet;
    SmallVectorImpl<Value> &setOperands;
  };

  /// Create a versioning condition suitable for scf::IfOp.
  VersionCondition(SCFCondition scfCond) : versionCondition(scfCond) {}

  /// Create a versioning condition suitable for AffineIfOp.
  VersionCondition(AffineCondition affineCond) : versionCondition(affineCond) {}

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

/// Version an operation.
class VersionBuilder {
public:
  VersionBuilder(Operation *op) : op(op) {
    assert(op && "Expecting valid operation");
  }

  void version(const VersionCondition &) const;

protected:
  void createElseBody(scf::IfOp) const;
  void createElseBody(affine::AffineIfOp) const;

  mutable Operation *op; // The operation to version.
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
  AffineForGuardBuilder(affine::AffineForOp loop);

private:
  affine::AffineForOp getLoop() const;
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
  AffineParallelGuardBuilder(affine::AffineParallelOp loop);

private:
  affine::AffineParallelOp getLoop() const;
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

/// A collection of tools for loop analysis and transformations.
class LoopTools {
public:
  /// Guard the given loop \p loop.
  static void guardLoop(LoopLikeOpInterface loop);

  /// Version the given loop \p loop using the condition \p versionCond.
  static void versionLoop(LoopLikeOpInterface loop,
                          const VersionCondition &versionCond);

  /// Return true is \p loop is the outermost loop in a loop nest and false
  /// otherwise.
  static bool isOutermostLoop(LoopLikeOpInterface loop);

  /// Collect perfectly nested loops starting from \p root. Loops are perfectly
  /// nested if each loop is the first and only non-terminator operation in the
  /// parent loop.
  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, affine::AffineForOp, scf::ForOp>::value>>
  static void getPerfectlyNestedLoops(SmallVector<T> &nestedLoops, T root) {
    LoopLikeOpInterface previousLoop = root;
    root->template walk<WalkOrder::PreOrder>([&](T loop) {
      if (!arePerfectlyNested(previousLoop, loop))
        return WalkResult::interrupt();
      nestedLoops.push_back(loop);
      previousLoop = loop;
      return WalkResult::advance();
    });
  }

  /// Return true if the loop nest rooted at \p root is perfectly nested.
  /// Note that \p root can, but it need not, be the outermost loop in a loop
  /// nest.
  static bool isPerfectLoopNest(LoopLikeOpInterface root);

  /// Return the innermost loop in the perfect loop nest rooted by \p root or
  /// std::nullopt if the loop nest is not perfect.
  static std::optional<LoopLikeOpInterface>
  getInnermostLoop(LoopLikeOpInterface root);

private:
  /// Return true if \p outer and \p inner are perfectly nested with respect to
  /// each other and false otherwise.
  static bool arePerfectlyNested(LoopLikeOpInterface outer,
                                 LoopLikeOpInterface inner);
};

//===----------------------------------------------------------------------===//
// VersionConditionBuilder
//===----------------------------------------------------------------------===//

/// Build a version condition to check if the given list of accessor pairs
/// overlap.
class VersionConditionBuilder {
public:
  using SCFCondition = VersionCondition::SCFCondition;
  using AffineCondition = VersionCondition::AffineCondition;

  VersionConditionBuilder(
      std::set<sycl::AccessorPtrPair> requireNoOverlapAccessorPairs,
      OpBuilder builder, Location loc);

  std::unique_ptr<VersionCondition>
  createCondition(bool useOpaquePointers) const {
    SCFCondition scfCond = createSCFCondition(builder, loc, useOpaquePointers);
    return std::make_unique<VersionCondition>(scfCond);
  }

private:
  /// Create a versioning condition suitable for scf::IfOp.
  SCFCondition createSCFCondition(OpBuilder builder, Location loc,
                                  bool useOpaquePointers) const;

  std::set<sycl::AccessorPtrPair> accessorPairs;
  mutable OpBuilder builder;
  mutable Location loc;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H
