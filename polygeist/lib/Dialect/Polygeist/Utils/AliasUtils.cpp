//===- AliasUtils.cpp - Alias Queries Utilities  --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Utils/AliasUtils.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace polygeist {

//===----------------------------------------------------------------------===//
// AliasOracle
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const AliasOracle &AliasOracle) {
  auto printMap = [&os](const DenseMap<Value, SetVector<Value>> &map,
                        StringRef title) {
    for (const auto &entry : map) {
      Value val = entry.first;
      const SetVector<Value> &aliasSet = entry.second;
      os.indent(4) << val << "\n";
      os.indent(4) << title << ":\n";
      if (aliasSet.empty())
        os.indent(6) << "<none>\n";
      else {
        for (Value aliasedVal : aliasSet)
          os.indent(6) << aliasedVal << "\n";
      }
      os << "\n";
    }
  };

  os.indent(4) << AliasOracle.funcOp.getName() << ":\n";
  printMap(AliasOracle.valueToMustAliasValues, "mustAliases");
  printMap(AliasOracle.valueToMayAliasValues, "mayAliases");
  return os;
}

AliasOracle::AliasOracle(FunctionOpInterface &funcOp,
                         mlir::AliasAnalysis &aliasAnalysis)
    : funcOp(funcOp), aliasAnalysis(aliasAnalysis) {
  /// Collect all operations that reference a memory resource in the given
  /// function and initialize the maps.
  SetVector<Value> memoryResources = collectMemoryResourcesIn(funcOp);
  for (Value val1 : memoryResources) {
    for (Value val2 : memoryResources) {
      if (val1 == val2)
        continue;

      AliasResult aliasResult = aliasAnalysis.alias(val1, val2);
      if (aliasResult.isMust())
        valueToMustAliasValues[val1].insert(val2);
      else if (aliasResult.isMay())
        valueToMayAliasValues[val1].insert(val2);
    }
  }
}

SetVector<Value>
AliasOracle::collectMemoryResourcesIn(FunctionOpInterface funcOp) {
  SetVector<Value> memoryResources;
  funcOp->walk([&](MemoryEffectOpInterface memoryEffectOp) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffectOp.getEffects(effects);
    for (const auto &effect : effects) {
      if (Value val = effect.getValue())
        memoryResources.insert(val);
    }
    return WalkResult::advance();
  });
  return memoryResources;
}

} // namespace polygeist
} // namespace mlir
