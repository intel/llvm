//===- AliasUtils.h - Aliasing queries utilities -----------------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_UTILS_ALIASUTILS_H
#define MLIR_DIALECT_POLYGEIST_UTILS_ALIASUTILS_H

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
class AliasAnalysis;

namespace polygeist {

/// A class to perform queries related to aliased values.
class AliasOracle {
  friend raw_ostream &operator<<(raw_ostream &, const AliasOracle &);

public:
  AliasOracle(FunctionOpInterface &funcOp, mlir::AliasAnalysis &aliasAnalysis);

  /// Return the set of values that are definitely aliased to \p val.
  SetVector<Value> getMustAlias(Value val) const {
    return valueToMustAliasValues.lookup(val);
  }

  /// Return the set of values that are possibly aliased to \p val.
  SetVector<Value> getMayAlias(Value val) const {
    return valueToMayAliasValues.lookup(val);
  }

  mlir::AliasAnalysis &getAliasAnalysis() const { return aliasAnalysis; }

private:
  /// Collect values that reference a memory resource within function \p funcOp.
  SetVector<Value> collectMemoryResourcesIn(FunctionOpInterface funcOp);

private:
  // val -> set of values that are definitely aliased to val.
  DenseMap<Value, SetVector<Value>> valueToMustAliasValues;
  // val -> set of values that might be aliased to val.
  DenseMap<Value, SetVector<Value>> valueToMayAliasValues;
  FunctionOpInterface &funcOp; // The function associated with this object.
  mlir::AliasAnalysis &aliasAnalysis; // The alias analysis.
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_ALIASUTILS_H
