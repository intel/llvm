//===--------- ModuleSplitter.h - split a module into callgraphs ----------===//
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

#pragma once

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace llvm {

class Function;
class Module;

namespace module_split {

enum IRSplitMode {
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_AUTO,       // automatically select split mode
  SPLIT_NONE        // no splitting
};

// A vector that contains all entry point functions in a split module.
using EntryPointVec = std::vector<const Function *>;

struct EntryPointGroup {
  StringRef GroupId;
  EntryPointVec Functions;
};

using EntryPointGroupVec = std::vector<EntryPointGroup>;

struct ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;

  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }

  bool isEsimd();
};

// Module split support interface.
// It gets a module and a collection of entry points groups. Each group
// specifies subset entry points from input module that should be included in
// a split module.
class ModuleSplitterBase {
  std::unique_ptr<Module> InputModule{nullptr};
  const EntryPointGroupVec Groups;
  EntryPointGroupVec::const_iterator GroupsIt;

protected:
  const EntryPointGroup &nextGroup() {
    assert(hasMoreSplits() && "Reached end of entry point groups list.");
    return *(GroupsIt++);
  }

  Module &getInputModule() {
    assert(InputModule && "No module to access to.");
    return *InputModule;
  }
  std::unique_ptr<Module> releaseInputModule() {
    assert(InputModule && "No module to release.");
    return std::move(InputModule);
  }

public:
  explicit ModuleSplitterBase(std::unique_ptr<Module> M,
                              EntryPointGroupVec GroupVec)
      : InputModule(std::move(M)), Groups(std::move(GroupVec)) {
    assert(InputModule && "Module is absent.");
    assert(!Groups.empty() && "Entry points groups collection is empty!");
    GroupsIt = Groups.cbegin();
  }

  virtual ~ModuleSplitterBase() = default;

  // Gets next subsequence of entry points in an input module and provides split
  // submodule containing these entry points and their dependencies.
  virtual ModuleDesc nextSplit() = 0;

  size_t totalSplits() const { return Groups.size(); }

  // Check that there are still submodules to split.
  bool hasMoreSplits() const { return GroupsIt != Groups.cend(); }
};

std::unique_ptr<ModuleSplitterBase>
getSplitterByKernelType(std::unique_ptr<Module> M, bool SplitEsimd,
                        bool EmitOnlyKernelsAsEntryPoints);

std::unique_ptr<ModuleSplitterBase>
getSplitterByMode(std::unique_ptr<Module> M, IRSplitMode Mode,
                  bool IROutputOnly, bool EmitOnlyKernelsAsEntryPoints,
                  bool DeviceGlobals);

} // namespace module_split

} // namespace llvm
