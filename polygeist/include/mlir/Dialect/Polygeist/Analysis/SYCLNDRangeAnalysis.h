//===- SYCLNDRangeAnalysis.h - Analysis for sycl::nd_range ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis to derive interesting properties about
// sycl::nd_range in SYCL host code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLNDRANGEANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLNDRANGEANALYSIS_H

#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace polygeist {
class Definition;
/// Represents information about a `sycl::nd_range` gathered from its
/// construction.
class NDRangeInformation {
public:
  static NDRangeInformation join(const NDRangeInformation &lhs,
                                 const NDRangeInformation &rhs);

  NDRangeInformation() = default;

  explicit NDRangeInformation(size_t dimensions);

  NDRangeInformation(const IDRangeInformation &globalSizeInfo,
                     const IDRangeInformation &localSizeInfo,
                     const IDRangeInformation &offsetInfo);

  /// Returns the global size information.
  const IDRangeInformation &getGlobalSizeInfo() const;

  /// Returns the local size information.
  const IDRangeInformation &getLocalSizeInfo() const;

  /// Returns the offset information.
  const IDRangeInformation &getOffsetInfo() const;

  bool isTop() const;

private:
  friend raw_ostream &operator<<(raw_ostream &, const NDRangeInformation &);

  IDRangeInformation globalSizeInfo;
  IDRangeInformation localSizeInfo;
  IDRangeInformation offsetInfo;
};

/// Analysis to determine properties of interest about `sycl::nd_range` from its
/// construction.
class SYCLNDRangeAnalysis {
public:
  SYCLNDRangeAnalysis(Operation *op, AnalysisManager &am);

  /// Consumers of the analysis must call this member function immediately after
  /// construction and before requesting any information from the analysis.
  SYCLNDRangeAnalysis &initialize(bool useRelaxedAliasing);

  std::optional<NDRangeInformation>
  getNDRangeInformationFromConstruction(Operation *op, Value operand);

private:
  NDRangeInformation getInformation(const Definition &def);

  Operation *operation;

  AnalysisManager &am;

  std::unique_ptr<DataFlowSolverWrapper> solver;

  bool initialized = false;

  SYCLIDAndRangeAnalysis idRangeAnalysis;
};
} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLNDRANGEANALYSIS_H
