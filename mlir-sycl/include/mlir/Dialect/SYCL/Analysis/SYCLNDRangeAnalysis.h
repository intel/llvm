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

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_SYCLNDRANGEANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_SYCLNDRANGEANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/ConstructorBaseAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace sycl {
/// Represents information about a `sycl::nd_range` gathered from its
/// construction.
class NDRangeInformation {
public:
  const NDRangeInformation join(const NDRangeInformation &other,
                                mlir::AliasAnalysis &);

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
class SYCLNDRangeAnalysis
    : public ConstructorBaseAnalysis<SYCLNDRangeAnalysis, NDRangeInformation> {
public:
  SYCLNDRangeAnalysis(Operation *op, AnalysisManager &am);

  void finalizeInitialization(bool useRelaxedAliasing);

  std::optional<NDRangeInformation>
  getNDRangeInformationFromConstruction(Operation *op, Value operand);

  template <typename SYCLType>
  NDRangeInformation getInformationImpl(const polygeist::Definition &def);

private:
  SYCLIDAndRangeAnalysis idRangeAnalysis;
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_SYCLNDRANGEANALYSIS_H
