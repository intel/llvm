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

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_SYCLBUFFERANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_SYCLBUFFERANALYSIS_H

#include "ConstructorBaseAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/DataFlowSolverWrapper.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
namespace sycl {

/// YES indicates that the buffer is definitely a sub-buffer
/// NO indicates that the buffer is definitely not a sub-buffer.
/// MAYBE indicates that there are multiple constructions with different
/// sub-buffer property or insufficient information is available.
enum class SubBufferLattice { YES, NO, MAYBE };

/// Represents information about a `sycl::buffer` gathered from its
/// construction.
class BufferInformation {
public:
  BufferInformation();

  BufferInformation(ArrayRef<size_t> constRange, SubBufferLattice isSubBuffer,
                    Value baseBuffer, ArrayRef<size_t> constBaseBufferSize,
                    ArrayRef<size_t> constSubBufferOffset);

  /// Returns true if this buffer is always constructed with the same range as
  /// size.
  bool hasConstantSize() const { return !constantSize.empty(); }

  /// Returns the constant values for the size this buffer is constructed with.
  ArrayRef<size_t> getConstantSize() const { return constantSize; }

  /// Returns information about whether this buffer is a sub-buffer of another
  /// buffer.
  SubBufferLattice getSubBuffer() const { return subBuf; }

  /// Returns true if this sub-buffer is always constructed with the same base
  /// buffer.
  bool hasKnownBaseBuffer() const { return baseBuffer != nullptr; }

  /// Returns the base buffer if this sub-buffer is always constructed with the
  /// same base buffer.
  Value getKnownBaseBuffer() const {
    assert(hasKnownBaseBuffer() && "Base buffer unknown or not unique");
    return baseBuffer;
  }

  /// Returns true if the the base buffer of this sub-buffer is known and that
  /// base buffer is always constructed with the same size.
  bool hasKnownBaseBufferSize() const { return !baseBufferSize.empty(); }

  /// Returns the constant values for the size of the base buffer if known.
  ArrayRef<size_t> getKnownBaseBufferSize() const {
    assert(hasKnownBaseBufferSize() &&
           "Base buffer size unknown or not constant");
    return baseBufferSize;
  }

  /// Returns true if this sub-buffer is always constructed with the same
  /// offset.
  bool hasConstantOffset() const { return !subBufOffset.empty(); }

  /// Returns the constant values for the offset of this sub-buffer.
  ArrayRef<size_t> getConstantOffset() const {
    assert(hasConstantOffset() && "Offset unknown or not constant");
    return subBufOffset;
  }

  /// Returns true if the top of the lattice has been reached.
  bool isTop() const {
    return !hasConstantSize() && getSubBuffer() == SubBufferLattice::MAYBE;
  }

  const BufferInformation join(const BufferInformation &other,
                               mlir::AliasAnalysis &aliasAnalysis) const;

private:
  SmallVector<size_t, 3> constantSize;

  SubBufferLattice subBuf;

  Value baseBuffer;

  SmallVector<size_t, 3> baseBufferSize;

  SmallVector<size_t, 3> subBufOffset;

  friend raw_ostream &operator<<(raw_ostream &, const BufferInformation &);
};

/// Analysis to determine properties of interest about a `sycl::buffer` from its
/// construction.
class SYCLBufferAnalysis
    : public ConstructorBaseAnalysis<SYCLBufferAnalysis, BufferInformation> {
public:
  SYCLBufferAnalysis(Operation *op, AnalysisManager &mgr);

  /// Consumers of the analysis must call this member function immediately after
  /// construction and before requesting any information from the analysis.
  void finalizeInitialization(bool useRelaxedAliasing = false);

  std::optional<BufferInformation>
  getBufferInformationFromConstruction(Operation *op, Value operand);

  template <typename SYCLType>
  BufferInformation getInformationImpl(const polygeist::Definition &def);

private:
  SYCLIDAndRangeAnalysis idRangeAnalysis;
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_SYCLBUFFERANALYSIS_H
