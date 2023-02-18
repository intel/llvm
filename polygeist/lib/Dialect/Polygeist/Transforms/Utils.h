//===- Utils.h - Utility functions ------------------------------------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace llvm {
template <typename T> struct SmallVectorImpl;
} // namespace llvm

namespace mlir {
class Affinemap;
class DominanceInfo;
class PatternRewriter;
class Value;

void fully2ComposeAffineMapAndOperands(PatternRewriter &rewriter,
                                       AffineMap *map,
                                       llvm::SmallVectorImpl<Value> *operands,
                                       DominanceInfo &DI);
bool isValidIndex(Value val);
} // namespace mlir
