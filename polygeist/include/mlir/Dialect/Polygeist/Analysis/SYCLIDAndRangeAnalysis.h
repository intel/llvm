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

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace polygeist {

class IDRangeInformation {
public:
  IDRangeInformation();

  explicit IDRangeInformation(size_t dim);

  explicit IDRangeInformation(llvm::ArrayRef<size_t> constVals);

  bool hasFixedDimensions() const;

  size_t getNumDimensions() const;

  bool isConstant() const;

  const llvm::SmallVector<size_t, 3> &getConstantValues() const;

  const IDRangeInformation join(const IDRangeInformation &other) const;

private:
  std::optional<size_t> dimensions;

  std::optional<llvm::SmallVector<size_t, 3>> constantValues;

  friend raw_ostream &operator<<(raw_ostream &, const IDRangeInformation &);
};

class SYCLIDAndRangeAnalysis {
public:
  SYCLIDAndRangeAnalysis(Operation *op, AnalysisManager &am);

  template <typename Type>
  std::optional<IDRangeInformation> getIDRangeInformation(Operation *op,
                                                          Value operand);

private:
  void initialize();

  Operation *operation;

  AnalysisManager &am;

  DataFlowSolver solver;

  bool initialized = false;
};
} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLIDANDRANGEANALYSIS_H
