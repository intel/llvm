//===- AliasAnalysis.h - SYCL Alias Analysis -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of an alias analysis for the SYCL
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_ALIASANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

namespace mlir {
namespace sycl {

/// Specialized alias analysis for SYCL dialect operations.
class AliasAnalysis : public LocalAliasAnalysis {
public:
  AliasAnalysis(bool relaxedAliasing)
      : LocalAliasAnalysis(), relaxedAliasing(relaxedAliasing) {}

protected:
  AliasResult aliasImpl(Value lhs, Value rhs) override;

private:
  /// Return 'NoAlias' if \p lhs or \p rhs are function arguments and any of
  /// them have attribute 'llvm.noalias', and 'MayAlias' otherwise.
  AliasResult handleNoAliasArguments(Value lhs, Value rhs);

  /// This function attempts to refine aliasing for values produced by SYCL
  /// operations. It returns 'NoAlias' if it can prove that values do not alias
  /// and 'MayAlias' otherwise.
  AliasResult handleSYCLAlias(Value lhs, Value rhs);

  /// This function attempts to refine aliasing for values produced by SYCL
  /// 'accessor.subscript' operations. It returns 'NoAlias' if it can prove that
  /// values do not alias and 'MayAlias' otherwise.
  AliasResult handleAccessorSubscriptAlias(Value lhs, Value rhs);

  /// Whether to assume the program abides to strict aliasing rules (i.e type
  /// based aliasing) or not.
  const bool relaxedAliasing = false;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_ALIASANALYSIS_H
