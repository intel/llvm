//===- MemoryAccessAnalysis.h - SYCL Memory Access Analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a memory access analysis for the SYCL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H

namespace mlir {
namespace sycl {

/// Classify array accesses.
enum MemoryAccessPattern {
  Unkown = 0,

  /// Array accessed contiguously, in increasing offset order:
  ///    |-------------------->|
  ///    |-------------------->|
  Linear = 1 << 0,
  Reverse = 1 << 1, /// Array accessed in decreasing offset order.

  /// Array accessed contiguously, in decreasing offset order:
  ///    |<-------------------|
  ///    |<-------------------|
  ReverseLinear = Reverse | Linear,
  Shifted = 1 << 2, /// Array accessed starting at an offset.

  /// Array accessed contiguously, increasing offset order, starting at an
  /// offset:
  ///    |  ----------------->|
  ///    |  ----------------->|
  LinearShifted = Linear | Shifted,

  /// Array accessed contiguously, decreasing offset order, starting at an
  /// offset:
  ///    |<-----------------  |
  ///    |<-----------------  |
  ReverseLinearShifted = ReverseLinear | Shifted,
  Overlapped = 1 << 3, // Array accessed starting at different row offsets.

  /// Array accessed contiguously, increasing offset order, starting at
  /// different row offsets:
  ///    |  ----------------->|
  ///    |    --------------->|
  LinearOverlapped = Linear | Overlapped,

  /// Array accessed contiguously, decreasing offset order, starting at
  /// different row offsets:
  ///    |<------------------ |
  ///    |<----------------   |
  ReverseLinearOverlapped = ReverseLinear | Overlapped,

  /// Array accessed in strided fashion, increasing offset order:
  ///    |x-->x-->x-->x-->x-->|
  ///    |x-->x-->x-->x-->x-->|
  Strided = 1 << 4,

  /// Array accessed in strided fashion, decreasing offset order:
  ///    |<--x<--x<--x<--x<--x|
  ///    |<--x<--x<--x<--x<--x|
  ReverseStrided = Reverse | Strided,

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// an offset:
  ///    |    x-->x-->x-->x-->|
  ///    |x-->x-->x-->x-->x-->|
  StridedShifted = Strided | Shifted,

  /// Array accessed in strided fashion, decreasing offset order:
  ///    |<--x<--x<--x<--x<--x|
  ///    |<--x<--x<--x<--x<--x|
  ReverseStridedShifted = Reverse | StridedShifted,

  /// Array accessed in random order.
  Random = 1 << 8

};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H
which clang