//===- SYCLIDAndRangeAnalysis.h - Analysis for sycl::id/range ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis to derive interesting properties about
// sycl::id and sycl::range in SYCL host code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLIDANDRANGEANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLIDANDRANGEANALYSIS_H

#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace polygeist {

/// Represents information about a `sycl::id` or `sycl::range` gathered from its
/// construction.
class IDRangeInformation {
public:
  IDRangeInformation();

  explicit IDRangeInformation(size_t dim);

  explicit IDRangeInformation(llvm::ArrayRef<size_t> constVals);

  /// Returns true if the id/range is always constructed with the same number of
  /// dimensions.
  bool hasFixedDimensions() const;

  /// Returns the number of dimensions for an id or range, in case it is always
  /// constructed with the same number of dimensions.
  size_t getNumDimensions() const;

  /// Returns true if the id/range is always constructed with the same constant
  /// values.
  bool isConstant() const;

  /// Returns the constant values with which this id/range is constructed, in
  /// case it is always constructed with the same constant values.
  const llvm::SmallVector<size_t, 3> &getConstantValues() const;

  const IDRangeInformation join(const IDRangeInformation &other) const;

private:
  std::optional<size_t> dimensions;

  std::optional<llvm::SmallVector<size_t, 3>> constantValues;

  friend raw_ostream &operator<<(raw_ostream &, const IDRangeInformation &);
};

/// Analysis to determine properties of interest about `sycl::id` or
/// `sycl::range` from their construction.
class SYCLIDAndRangeAnalysis {
public:
  SYCLIDAndRangeAnalysis(Operation *op, AnalysisManager &am);

  /// Consumers of the analysis must call this member function immediately after
  /// construction and before requesting any information from the analysis.
  SYCLIDAndRangeAnalysis &initialize(bool useRelaxedAliasing);

  template <typename Type>
  std::optional<IDRangeInformation>
  getIDRangeInformationFromConstruction(Operation *op, Value operand);

private:
  Operation *operation;

  AnalysisManager &am;

  std::unique_ptr<DataFlowSolverWrapper> solver;

  bool initialized = false;
};
} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLIDANDRANGEANALYSIS_H
