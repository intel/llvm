//===---- ModuleSplitter.h - split a module into callgraphs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module into call graphs. A callgraph here is a set
// of entry points with all functions reachable from them via a call. The result
// of the split is new modules containing corresponding callgraph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H
#define LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

#include <vector>
#include <functional>

namespace llvm {

namespace module_split {
using EntryPointsVec = std::vector<const Function*>;

struct ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointsVec EntryPoints;
};

void split(std::unique_ptr<Module> M, ModuleDesc /*inout*/&MA, ModuleDesc /*inout*/&MB, StringRef RenameSharedSuff="");
void extractCallgraph(const Module &Src, ModuleDesc /*inout*/&M);

} // module_split

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H
