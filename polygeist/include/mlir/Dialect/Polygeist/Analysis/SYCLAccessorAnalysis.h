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

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLACCESSORANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLACCESSORANALYSIS_H

#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLBufferAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace polygeist {

/// YES indicates that the buffer is definitely a sub-buffer
/// NO indicates that the buffer is definitely not a sub-buffer.
/// MAYBE indicates that there are multiple constructions with different
/// sub-buffer property or insufficient information is available.
enum class SYCLAcessorAliasResult { YES, NO, MAYBE };

/// Represents information about a `sycl::accessor` gathered from its
/// construction.
class AccessorInformation {
public:
  AccessorInformation() {}

  AccessorInformation(Value buf, std::optional<BufferInformation> bufInfo,
                      bool hasRange, llvm::ArrayRef<size_t> constRange,
                      bool hasOffset, llvm::ArrayRef<size_t> constOffset)
      : buffer{buf}, bufferInfo{bufInfo}, needRange{hasRange},
        constantRange{constRange}, needOffset{hasOffset}, constantOffset{
                                                              constOffset} {}

  bool hasKnownBuffer() const { return buffer != nullptr; }

  Value getBuffer() const { return buffer; }

  bool hasBufferInformation() const { return bufferInfo.has_value(); }

  const BufferInformation &getBufferInfo() const { return *bufferInfo; }

  bool needsRange() const { return needRange; }

  bool hasConstantRange() const { return !constantRange.empty(); }

  const llvm::SmallVector<size_t, 3> &getConstantRange() const {
    assert(hasConstantRange() && "Range not constant");
    return constantRange;
  }

  bool needsOffset() const { return needOffset; }

  bool hasConstantOffset() const { return !constantOffset.empty(); }

  const llvm::SmallVector<size_t, 3> &getConstantOffset() const {
    assert(hasConstantOffset() && "Offset not constant");
    return constantOffset;
  }

  const AccessorInformation join(const AccessorInformation &other,
                                 AliasAnalysis &aliasAnalysis) const;

private:
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
class SYCLAccessorAnalysis {
public:
  SYCLAccessorAnalysis(Operation *op, AnalysisManager &am);

  /// Consumers of the analysis must call this member function immediately after
  /// construction and before requesting any information from the analysis.
  SYCLAccessorAnalysis &initialize(bool useRelaxedAliasing = false);

  std::optional<AccessorInformation>
  getAccessorInformationFromConstruction(Operation *op, Value operand);

private:
  void initialize();

  bool isConstructor(const Definition &def);

  AccessorInformation getInformation(const Definition &def);

  template <typename OperandType>
  std::optional<IDRangeInformation>
  getOperandInfo(sycl::SYCLHostConstructorOp constructor, size_t possibleIndex1,
                 size_t possibleIndex2);

  Operation *operation;

  AnalysisManager &am;

  std::unique_ptr<DataFlowSolverWrapper> solver;

  bool initialized = false;

  AliasAnalysis *aliasAnalysis;

  SYCLIDAndRangeAnalysis idRangeAnalysis;

  SYCLBufferAnalysis bufferAnalysis;
};
} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLACCESSORANALYSIS_H
