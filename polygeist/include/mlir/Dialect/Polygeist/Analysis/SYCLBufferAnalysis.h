//===- SYCLBufferAnalysis.h - Analysis for sycl::buffer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis to derive interesting properties about
// sycl::buffer in SYCL host code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLBUFFERANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLBUFFERANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace polygeist {

enum class SubBufferLattice { YES, NO, MAYBE };

class BufferInformation {
public:
  BufferInformation();

  BufferInformation(llvm::ArrayRef<size_t> constRange,
                    SubBufferLattice IsSubBuffer, Value baseBuffer,
                    llvm::ArrayRef<size_t> constBaseBufferSize,
                    llvm::ArrayRef<size_t> constSubBufferOffset);

  bool hasConstantSize() const { return !constantSize.empty(); }

  const llvm::SmallVector<size_t, 3> &getConstantSize() const {
    return constantSize;
  }

  SubBufferLattice isSubBuffer() const { return subBuf; }

  const BufferInformation join(const BufferInformation &other,
                               AliasAnalysis &aliasAnalysis) const;

private:
  SmallVector<size_t, 3> constantSize;

  SubBufferLattice subBuf;

  Value baseBuffer;

  SmallVector<size_t, 3> baseBufferSize;

  SmallVector<size_t, 3> subBufOffset;

  friend raw_ostream &operator<<(raw_ostream &, const BufferInformation &);
};

class SYCLBufferAnalysis {
public:
  SYCLBufferAnalysis(Operation *op, AnalysisManager &am);

  void initialize(bool useRelaxedAliasing = false);

  std::optional<BufferInformation>
  getBufferInformationFromConstruction(Operation *op, Value operand);

private:
  void initialize();

  bool isConstructor(const Definition &def);

  BufferInformation getInformation(const Definition &def);

  Operation *operation;

  AnalysisManager &am;

  DataFlowSolver solver;

  bool initialized = false;

  AliasAnalysis* aliasAnalysis;

  SYCLIDAndRangeAnalysis idRangeAnalysis;
};
} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_SYCLBUFFERANALYSIS_H
