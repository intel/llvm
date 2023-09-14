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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallSet.h"
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

/// Updates \p newName to a unique function/global name if it is already
/// defined.
void getUniqueSymbolName(std::string &newName, Operation *symbolTable);

/// Returns true if the given function has 'linkonce_odr' LLVM  linkage.
bool isLinkonceODR(FunctionOpInterface);

/// Change the linkage and visibility of the given function to private.
void privatize(FunctionOpInterface);

/// Returns true if the given call is a tail call.
bool isTailCall(CallOpInterface);

/// Return the grid dimension for \p func (e.g the dimension of the
/// sycl::nd_item or sycl::item argument passed to the function).
/// Return zero if it cannot identify the grid dimension.
unsigned getGridDimension(FunctionOpInterface func);

/// Return a vector containing thread values corresponding to the global
/// thread declarations in \p funcOp. The returned vector has size equal to
/// the thread grid dimension and is sorted based on the dimension.
/// For example given:
///   %c1_i32 = arith.constant 1 : i32
///   %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32)
/// The corresponding vector entry is %ty and the vector has size equal to 2
/// where the first element is a null value.
SmallVector<Value> getThreadVector(FunctionOpInterface funcOp,
                                   DataFlowSolver &solver);

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
// FunctionKernelInfo
//===----------------------------------------------------------------------===//

/// Create a map from each function to a list of all the kernel that can reach
/// the function, and its associated depth from the kernel to the function.
class FunctionKernelInfo {
public:
  FunctionKernelInfo() = delete;
  FunctionKernelInfo(gpu::GPUModuleOp);

  struct KernelInfo {
    gpu::GPUFuncOp kernel;
    unsigned depth; // Depth from the associated kernel
  };

  /// Returns true if the given function is potentially a SYCL kernel body
  /// function. The SYCL kernel body function is created by SemaSYCL in clang
  /// for the body of the SYCL kernel, e.g., code in parallel_for.
  /// TODO: add an attribute to the call operator of the SYCL kernel functor in
  /// SemaSYCL in clang, to identify SYCL kernel body function accurately.
  bool isPotentialKernelBodyFunction(FunctionOpInterface func) const;

  /// Returns the maximum depth from any GPU kernel.
  /// Returns std::nullopt if the call is not called from a GPU kernel.
  /// For example:
  /// Call chains:
  ///   GPUKernel1 -> func1 (depth 1) -> func2 (depth 2)
  ///   GPUKernel2 -> func2 (depth 1)
  /// =>
  ///   getMaxDepthFromAnyGPUKernel(func1) returns 1.
  ///   getMaxDepthFromAnyGPUKernel(func2) returns 2.
  Optional<unsigned>
  getMaxDepthFromAnyGPUKernel(FunctionOpInterface func) const;

  /// Returns the maximum depth from \p kernel to \p func.
  /// For example:
  /// Call chains:
  ///   GPUKernel1 -> func1 (depth 1) -> func2 (depth 2)
  ///   GPUKernel1 -> func2 (depth 1)
  ///   GPUKernel2 -> func0 (depth 1) -> func1 (depth 2) -> func2 (depth 3)
  /// =>
  ///   getMaxDepthFromGPUKernel(func2, GPUKernel1) returns 2.
  ///   getMaxDepthFromGPUKernel(func2, GPUKernel2) returns 3.
  ///   getMaxDepthFromAnyGPUKernel(func2) returns 3.
  Optional<unsigned> getMaxDepthFromGPUKernel(FunctionOpInterface func,
                                              gpu::GPUFuncOp kernel) const;

  /// Populates \p kernels with GPU kernels that can reach \p func.
  void getKernelCallers(FunctionOpInterface func,
                        SmallVectorImpl<gpu::GPUFuncOp> &kernels) const;

  /// Returns the potential 'kernel body functions' of \p kernel. A 'kernel body
  /// function' is the lambda/functor associated with the SYCL kernel construct
  /// (e.g., parallel_for). Note that transformation passes might have cloned
  /// the kernel body to specialize it. This function returns all possible
  /// kernel body functions, including specializations.
  llvm::SmallSet<FunctionOpInterface, 4>
  getPotentialKernelBodyFunctions(gpu::GPUFuncOp kernel) const;

private:
  /// Populate funcKernelInfosMap with the list of GPU kernels that can reach
  /// \p func and their associated depth.
  void populateGPUKernelInfo(FunctionOpInterface func);

  /// Map from a function to all kernels that can reach it and their
  /// corresponding depths.
  DenseMap<FunctionOpInterface, SmallVector<KernelInfo>> funcKernelInfosMap;
  /// Map from a kernel to all functions that can be reached from it.
  DenseMap<gpu::GPUFuncOp, std::set<FunctionOpInterface>> kernelFuncsMap;
};

//===----------------------------------------------------------------------===//
// Versioning Utilities
//===----------------------------------------------------------------------===//

/// Represents a condition used in a if operation.
class IfCondition {
  friend raw_ostream &operator<<(raw_ostream &, const IfCondition &);

public:
  struct AffineCondition {
    AffineCondition(IntegerSet iSet, OperandRange operands)
        : ifCondSet(iSet), setOperands(operands) {}
    IntegerSet ifCondSet;
    ValueRange setOperands;
  };

  IfCondition(Value scfCond) : ifCondition(scfCond) {}
  IfCondition(AffineCondition affineCond) : ifCondition(affineCond) {}

  bool hasSCFCondition() const {
    return std::holds_alternative<Value>(ifCondition);
  }

  Value getSCFCondition() const {
    assert(hasSCFCondition() && "expecting valid SCF condition");
    return std::get<Value>(ifCondition);
  }

  bool hasAffineCondition() const {
    return std::holds_alternative<AffineCondition>(ifCondition);
  }

  AffineCondition getAffineCondition() const {
    assert(hasAffineCondition() && "expecting valid affine condition");
    return std::get<AffineCondition>(ifCondition);
  }

  /// Perform \p operation on SCF condition or affine condition operands.
  template <typename OpTy, typename = std::enable_if_t<std::is_same_v<
                               bool, std::invoke_result_t<OpTy, ValueRange>>>>
  bool perform(OpTy operation) const {
    if (hasSCFCondition())
      return operation(getSCFCondition());
    assert(hasAffineCondition() && "Expecting affine condition");
    return operation(getAffineCondition().setOperands);
  }

  /// Return the condition used by \p op if one is found.
  static std::optional<IfCondition> getCondition(Operation *op);

private:
  std::variant<Value, AffineCondition> ifCondition;
};

/// Version an operation.
class VersionBuilder {
public:
  VersionBuilder(Operation *op) : op(op) {
    assert(op && "Expecting valid operation");
  }

  void version(const IfCondition &) const;

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
                          const IfCondition &versionCond);

  /// Return true if \p loop is the outermost loop in a loop nest and false
  /// otherwise.
  static bool isOutermostLoop(LoopLikeOpInterface loop);

  /// Return true if \p loop is an innermost loop in a loop nest and false
  /// otherwise.
  static bool isInnermostLoop(LoopLikeOpInterface loop);

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
  using AffineCondition = IfCondition::AffineCondition;

  VersionConditionBuilder(
      llvm::SmallSet<sycl::AccessorPtrPair, 4> requireNoOverlapAccessorPairs,
      OpBuilder builder, Location loc);

  std::unique_ptr<IfCondition> createCondition() const {
    Value scfCond = createSCFCondition(builder, loc);
    return std::make_unique<IfCondition>(scfCond);
  }

private:
  /// Create a versioning condition suitable for scf::IfOp.
  Value createSCFCondition(OpBuilder builder, Location loc) const;

  llvm::SmallSet<sycl::AccessorPtrPair, 4> accessorPairs;
  mutable OpBuilder builder;
  mutable Location loc;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_TRANSFORMUTILS_H
