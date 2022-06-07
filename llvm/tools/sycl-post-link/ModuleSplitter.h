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

enum class SyclEsimdSplitStatus { SYCL_ONLY, ESIMD_ONLY, SYCL_AND_ESIMD };

// Describes scope covered by each entry in the module-entry points map
// populated by the groupEntryPointsByScope function.
enum EntryPointsGroupScope {
  Scope_PerKernel, // one entry per kernel
  Scope_PerModule, // one entry per module
  Scope_Global     // single entry in the map for all kernels
};

// Represents a named group of device code entry points - kernels and
// SYCL_EXTERNAL functions. There are special group names - "<SYCL>" and
// "<ESIMD>" - which have effect on group processing.
struct EntryPointGroup {
  // Properties an entry point (EP) group
  struct Properties {
    // Whether all EPs are ESIMD, SYCL or there are both kinds.
    SyclEsimdSplitStatus HasESIMD = SyclEsimdSplitStatus::SYCL_AND_ESIMD;
    // Whether any of the EPs use double GRF mode.
    bool UsesDoubleGRF = false;
    // Scope represented by EPs in a group
    EntryPointsGroupScope Scope = Scope_Global;

    Properties merge(const Properties& Other) const {
      Properties Res;
      Res.HasESIMD = HasESIMD == Other.HasESIMD ? HasESIMD : SyclEsimdSplitStatus::SYCL_AND_ESIMD;
      Res.UsesDoubleGRF = UsesDoubleGRF || Other.UsesDoubleGRF;
      // Scope remains global
      return Res;
    }
  };

  StringRef GroupId;
  EntryPointVec Functions;
  Properties Props;

  EntryPointGroup(StringRef GroupId = "") : GroupId(GroupId) {}
  EntryPointGroup(StringRef GroupId, EntryPointVec &&Functions)
      : GroupId(GroupId), Functions(std::move(Functions)) {}

  // Tells if this group has only ESIMD entry points (based on GroupId).
  bool isEsimd() const { return Props.HasESIMD == SyclEsimdSplitStatus::ESIMD_ONLY; }
  // Tells if this group has only SYCL entry points (based on GroupId).
  bool isSycl() const { return Props.HasESIMD == SyclEsimdSplitStatus::SYCL_ONLY; }
  // Tells if some entry points use double GRF mode.
  bool isDoubleGRF() const { return Props.UsesDoubleGRF; }

  void saveNames(std::vector<std::string> &Dest) const;
  void rebuildFromNames(const std::vector<std::string> &Names, const Module &M);
  void rebuild(const Module &M);
};

using EntryPointGroupVec = std::vector<EntryPointGroup>;

class ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;

public:
  struct Properties {
    bool SpecConstsMet = true;
  };
  std::string Name = "";
  Properties Props;

  ModuleDesc(std::unique_ptr<Module> &&M, EntryPointGroup &&EntryPoints)
      : M(std::move(M)), EntryPoints(std::move(EntryPoints)) {
    Name = this->EntryPoints.GroupId.str();
  }

  ModuleDesc(std::unique_ptr<Module> &&M, const std::vector<std::string> &Names)
      : M(std::move(M)) {
    rebuildEntryPoints(Names);
  }

  void assignMergedProperties(const ModuleDesc& MD1, const ModuleDesc& MD2);

  bool isESIMD() const { return EntryPoints.isEsimd(); }
  bool isSYCL() const { return EntryPoints.isSycl(); }
  bool isDoubleGRF() const { return EntryPoints.isDoubleGRF(); }

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

  void renameDuplicatesOf(const Module &M, StringRef Suff);

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
                  bool AutoSplitIsGlobalScope,
                  bool EmitOnlyKernelsAsEntryPoints);

std::unique_ptr<ModuleSplitterBase>
getESIMDDoubleGRFSplitter(std::unique_ptr<Module> M,
                          bool EmitOnlyKernelsAsEntryPoints);

#ifndef _NDEBUG
void dumpEntryPoints(const EntryPointVec &C, const char *msg = "", int Tab = 0);
void dumpEntryPoints(const Module &M, bool OnlyKernelsAreEntryPoints = false,
                     const char *msg = "", int Tab = 0);
#endif // _NDEBUG

} // namespace module_split

} // namespace llvm
