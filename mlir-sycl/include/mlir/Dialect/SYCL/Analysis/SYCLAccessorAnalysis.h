//===- SYCLAccessorAnalysis.h - Analysis for sycl::accessor -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis to derive interesting properties about
// sycl::accessor in SYCL host code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_SYCLACCESSORANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_SYCLACCESSORANALYSIS_H

#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/ConstructorBaseAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLBufferAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace sycl {

/// Represents information about a `sycl::accessor` gathered from its
/// construction.
class AccessorInformation {
public:
  AccessorInformation() : isTopAcc{true} {}

  AccessorInformation(Value buf, std::optional<BufferInformation> bufInfo,
                      bool hasRange, llvm::ArrayRef<size_t> constRange,
                      bool hasOffset, llvm::ArrayRef<size_t> constOffset)
      : buffer{buf}, bufferInfo{bufInfo}, needRange{hasRange},
        constantRange{constRange}, needOffset{hasOffset},
        constantOffset{constOffset} {}

  AccessorInformation(bool hasRange, llvm::ArrayRef<size_t> constRange)
      : isLocal{true}, needRange{hasRange}, constantRange{constRange},
        needOffset{false} {}

  /// Return whether the accessor is local.
  bool isLocalAccessor() const { return isLocal; }

  /// Returns true if the underlying buffer of this accessor is known.
  bool hasKnownBuffer() const { return buffer != nullptr; }

  /// Returns the underlying buffer of this accessor.
  Value getBuffer() const {
    assert(hasKnownBuffer() && "Buffer unknown");
    return buffer;
  }

  /// Returns true if buffer information is available for the buffer underlying
  /// this accessor.
  bool hasBufferInformation() const { return bufferInfo.has_value(); }

  /// Returns the buffer information for the buffer underlying this accessor.
  const BufferInformation &getBufferInfo() const {
    assert(hasBufferInformation() && "No buffer information available");
    return *bufferInfo;
  }

  /// Returns false if the range can be omitted for this accessor, true
  /// otherwise.
  bool needsSubRange() const { return needRange; }

  /// Returns true if the range of this accessor is known to be constant.
  bool hasConstantSubRange() const { return !constantRange.empty(); }

  /// Returns the constant values for the range of this accessor.
  ArrayRef<size_t> getConstantRange() const {
    assert(hasConstantSubRange() && "Range not constant");
    return constantRange;
  }

  /// Returns false if the offset can be omitted for this accessor, true
  /// otherwise.
  bool needsOffset() const { return needOffset; }

  /// Return true if the offset of this accessor is known to be constant.
  bool hasConstantOffset() const { return !constantOffset.empty(); }

  /// Returns the constant values for the offset of this accessor.
  ArrayRef<size_t> getConstantOffset() const {
    assert(hasConstantOffset() && "Offset not constant");
    return constantOffset;
  }

  /// Returns true if the top of the lattice has been reached, i.e., it is not
  /// possible to further refine the information known about this accessor.
  bool isTop() const { return isTopAcc; }

  const AccessorInformation join(const AccessorInformation &other,
                                 mlir::AliasAnalysis &aliasAnalysis) const;

  /// Returns an AliasResult indicating whether this accessor and the given
  /// accessor must, may or do not alias.
  AliasResult alias(const AccessorInformation &other,
                    mlir::AliasAnalysis &aliasAnalysis) const;

private:
  bool isTopAcc = false;
  bool isLocal = false;

  Value buffer;

  std::optional<BufferInformation> bufferInfo;

  bool needRange;

  llvm::SmallVector<size_t, 3> constantRange;

  bool needOffset;

  llvm::SmallVector<size_t, 3> constantOffset;

  friend raw_ostream &operator<<(raw_ostream &, const AccessorInformation &);
};

/// Analysis to determine properties of interest about a `sycl::accessor` from
/// its construction.
class SYCLAccessorAnalysis
    : public ConstructorBaseAnalysis<SYCLAccessorAnalysis,
                                     AccessorInformation> {
public:
  SYCLAccessorAnalysis(Operation *op, AnalysisManager &mgr);

  void finalizeInitialization(bool useRelaxedAliasing = false);

  std::optional<AccessorInformation>
  getAccessorInformationFromConstruction(Operation *op, Value operand);

  template <typename... SYCLType>
  AccessorInformation getInformationImpl(const polygeist::Definition &def);

private:
  template <typename OperandType>
  std::optional<IDRangeInformation>
  getOperandInfo(sycl::SYCLHostConstructorOp constructor,
                 ArrayRef<size_t> possibleIndices);

  SYCLIDAndRangeAnalysis idRangeAnalysis;

  SYCLBufferAnalysis bufferAnalysis;
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_SYCLACCESSORANALYSIS_H
