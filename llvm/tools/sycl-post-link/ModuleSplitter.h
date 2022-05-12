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
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"

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

public:
  EntryPointVec Functions;

  EntryPointGroup(StringRef GroupId = "") : GroupId(GroupId) {}
  EntryPointGroup(StringRef GroupId, EntryPointVec &&Functions)
      : GroupId(GroupId), Functions(std::move(Functions)) {}

  const StringRef getId() const { return GroupId; }
  bool isEsimd() const;
  bool isSycl() const;
  void saveNames(std::vector<std::string> &Dest) const;
  void rebuildFromNames(const std::vector<std::string> &Names, const Module &M);
  void rebuild(const Module &M);
};

using EntryPointGroupVec = std::vector<EntryPointGroup>;

enum class ESIMDStatus { SYCL_ONLY, ESIMD_ONLY, SYCL_AND_ESIMD };

class ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;

public:
  struct Properties {
    ESIMDStatus HasEsimd = ESIMDStatus::SYCL_AND_ESIMD;
    bool SpecConstsMet = true;
  };
  std::string Name = "";
  Properties Props;

  ModuleDesc(std::unique_ptr<Module> &&M, EntryPointGroup &&EntryPoints)
      : M(std::move(M)), EntryPoints(std::move(EntryPoints)) {
    Name = this->EntryPoints.getId().str();
  }

  ModuleDesc(std::unique_ptr<Module> &&M, const std::vector<std::string> &Names)
      : M(std::move(M)) {
    rebuildEntryPoints(Names);
  }

  const EntryPointVec &entries() const { return EntryPoints.Functions; }
  EntryPointVec &entries() { return EntryPoints.Functions; }
  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }
  std::unique_ptr<Module> releaseModulePtr() { return std::move(M); }

  // Sometimes, during module transformations, some Function objects within the
  // module are replaced with different Function objects with the same name (for
  // example, GenXSPIRVWriterAdaptor). Entry points need to be updated to
  // include the replacement function. save/rebuild pair of functions is
  // provided to automate this process.
  void saveEntryPointNames(std::vector<std::string> &Dest) {
    EntryPoints.saveNames(Dest);
  }

  void rebuildEntryPoints(const std::vector<std::string> &Names) {
    EntryPoints.rebuildFromNames(Names, getModule());
  }

  void rebuildEntryPoints(const Module &M) { EntryPoints.rebuild(M); }

  // Updates Props.HasEsimd to ESIMDStatus::ESIMD_ONLY/SYCL_ONLY if this module
  // descriptor is a ESIMD/SYCL part of the ESIMD/SYCL module split. Otherwise
  // assumes the module has both SYCL and ESIMD.
  void assignESIMDProperty();
#ifndef _NDEBUG
  void verifyESIMDProperty() const;
#endif // _NDEBUG

#ifndef _NDEBUG
  void dump();
#endif // _NDEBUG
};

// Module split support interface.
// It gets a module and a collection of entry points groups. Each group
// specifies subset entry points from input module that should be included in
// a split module.
class ModuleSplitterBase {
  std::unique_ptr<Module> InputModule{nullptr};
  EntryPointGroupVec Groups;

protected:
  EntryPointGroup nextGroup() {
    assert(hasMoreSplits() && "Reached end of entry point groups list.");
    EntryPointGroup Res = std::move(Groups.back());
    Groups.pop_back();
    return Res;
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
  ModuleSplitterBase(std::unique_ptr<Module> M, EntryPointGroupVec &&GroupVec)
      : InputModule(std::move(M)), Groups(std::move(GroupVec)) {
    assert(InputModule && "Module is absent.");
    assert(!Groups.empty() && "Entry points groups collection is empty!");
  }

  // For device global variables with the 'device_image_scope' property,
  // the function checks that there are no usages of a single device global
  // variable from kernels grouped to different modules. Otherwise, an error is
  // issued and the tool is aborted.
  void verifyNoCrossModuleDeviceGlobalUsage();

  virtual ~ModuleSplitterBase() = default;

  // Gets next subsequence of entry points in an input module and provides split
  // submodule containing these entry points and their dependencies.
  virtual ModuleDesc nextSplit() = 0;

  size_t totalSplits() const { return Groups.size(); }

  // Check that there are still submodules to split.
  bool hasMoreSplits() const { return totalSplits() > 0; }
};

std::unique_ptr<ModuleSplitterBase>
getSplitterByKernelType(std::unique_ptr<Module> M,
                        bool EmitOnlyKernelsAsEntryPoints,
                        EntryPointVec *AllowedEntries = nullptr);

std::unique_ptr<ModuleSplitterBase>
getSplitterByMode(std::unique_ptr<Module> M, IRSplitMode Mode,
                  bool EmitOnlyKernelsAsEntryPoints);

#ifndef _NDEBUG
void dumpEntryPoints(const EntryPointVec &C, const char *msg = "", int Tab = 0);
void dumpEntryPoints(const Module &M, bool OnlyKernelsAreEntryPoints = false,
                     const char *msg = "", int Tab = 0);
#endif // _NDEBUG

} // namespace module_split

} // namespace llvm
