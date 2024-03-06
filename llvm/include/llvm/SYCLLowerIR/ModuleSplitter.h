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

#ifndef LLVM_SYCLLOWERIR_MODULE_SPLITTER_H
#define LLVM_SYCLLOWERIR_MODULE_SPLITTER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
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
using EntryPointSet = SetVector<Function *>;

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
    // Scope represented by EPs in a group
    EntryPointsGroupScope Scope = Scope_Global;

    Properties merge(const Properties &Other) const {
      Properties Res;
      Res.HasESIMD = HasESIMD == Other.HasESIMD
                         ? HasESIMD
                         : SyclEsimdSplitStatus::SYCL_AND_ESIMD;
      // Scope remains global
      return Res;
    }
  };

  std::string GroupId;
  EntryPointSet Functions;
  Properties Props;

  EntryPointGroup(StringRef GroupId = "") : GroupId(GroupId) {}
  EntryPointGroup(StringRef GroupId, EntryPointSet &&Functions)
      : GroupId(GroupId), Functions(std::move(Functions)) {}
  EntryPointGroup(StringRef GroupId, EntryPointSet &&Functions,
                  const Properties &Props)
      : GroupId(GroupId), Functions(std::move(Functions)), Props(Props) {}

  // Tells if this group has only ESIMD entry points.
  bool isEsimd() const {
    return Props.HasESIMD == SyclEsimdSplitStatus::ESIMD_ONLY;
  }
  // Tells if this group has only SYCL entry points.
  bool isSycl() const {
    return Props.HasESIMD == SyclEsimdSplitStatus::SYCL_ONLY;
  }

  void saveNames(std::vector<std::string> &Dest) const;
  void rebuildFromNames(const std::vector<std::string> &Names, const Module &M);
  void rebuild(const Module &M);
};

using EntryPointGroupVec = std::vector<EntryPointGroup>;

// Annotates an llvm::Module with information necessary to perform and track
// result of device code (llvm::Module instances) splitting:
// - entry points of the module determined e.g. by a module splitter, as well
//   as information about entry point origin (e.g. result of a scoped split)
// - its properties, such as whether it has specialization constants uses
// It also provides convenience functions for entry point set transformation
// between llvm::Function object and string representations.
class ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;
  bool IsTopLevel = false;

public:
  struct Properties {
    bool IsSpecConstantDefault = false;
    bool SpecConstsMet = false;
  };
  std::string Name = "";
  Properties Props;

  ModuleDesc(std::unique_ptr<Module> &&M, StringRef Name = "TOP-LEVEL")
      : M(std::move(M)), IsTopLevel(true), Name(Name) {}

  ModuleDesc(std::unique_ptr<Module> &&M, EntryPointGroup &&EntryPoints,
             const Properties &Props)
      : M(std::move(M)), EntryPoints(std::move(EntryPoints)), Props(Props) {
    Name = this->EntryPoints.GroupId;
  }

  ModuleDesc(std::unique_ptr<Module> &&M, const std::vector<std::string> &Names,
             StringRef Name = "NoName")
      : M(std::move(M)), Name(Name) {
    rebuildEntryPoints(Names);
  }

  // Filters out functions which are not part of this module's entry point set.
  bool isEntryPointCandidate(const Function &F) const {
    if (EntryPoints.Functions.size() > 0) {
      return EntryPoints.Functions.contains(const_cast<Function *>(&F));
    }
    return IsTopLevel; // Top level module does not limit entry points set.
  }

  void assignMergedProperties(const ModuleDesc &MD1, const ModuleDesc &MD2);

  bool isESIMD() const { return EntryPoints.isEsimd(); }
  bool isSYCL() const { return EntryPoints.isSycl(); }

  const EntryPointSet &entries() const { return EntryPoints.Functions; }
  const EntryPointGroup &getEntryPointGroup() const { return EntryPoints; }
  EntryPointSet &entries() { return EntryPoints.Functions; }
  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }
  std::unique_ptr<Module> releaseModulePtr() { return std::move(M); }

  // Sometimes, during module transformations, some Function objects within the
  // module are replaced with different Function objects with the same name (for
  // example, GenXSPIRVWriterAdaptor). Entry points need to be updated to
  // include the replacement function. save/rebuild pair of functions is
  // provided to automate this process.
  // TODO: this scheme is unnecessarily complex. The simpler and easier
  // maintainable one would be using a special function attribute for the
  // duration of post-link transformations.
  void saveEntryPointNames(std::vector<std::string> &Dest) {
    EntryPoints.saveNames(Dest);
  }

  void rebuildEntryPoints(const std::vector<std::string> &Names) {
    EntryPoints.rebuildFromNames(Names, getModule());
  }

  void rebuildEntryPoints(const Module &M) { EntryPoints.rebuild(M); }

  void rebuildEntryPoints() { EntryPoints.rebuild(*M); }

  void renameDuplicatesOf(const Module &M, StringRef Suff);

  // Fixups an invoke_simd target linkage so that it is not dropped by global
  // DCE performed on an ESIMD module after it splits out. If SimdF can't be
  // deduced, then we have real function pointer, and user code is assumed to
  // define proper linkage for the potential target functions.
  // Also saves old linkage into a function attribute.
  void fixupLinkageOfDirectInvokeSimdTargets();

  // Restores original linkage of invoke_simd targets. This effectively
  // re-enables DCE on invoke_simd targets with linkonce linkage.
  void restoreLinkageOfDirectInvokeSimdTargets();

  // Cleans up module IR - removes dead globals, debug info etc.
  void cleanup();

  bool isSpecConstantDefault() const;
  void setSpecConstantDefault(bool Value);

  ModuleDesc clone() const;

#ifndef NDEBUG
  void verifyESIMDProperty() const;
  void dump() const;
#endif // NDEBUG
};

// Module split support interface.
// It gets a module (in a form of module descriptor, to get additional info) and
// a collection of entry points groups. Each group specifies subset entry points
// from input module that should be included in a split module.
class ModuleSplitterBase {
protected:
  ModuleDesc Input;
  EntryPointGroupVec Groups;

protected:
  EntryPointGroup nextGroup() {
    assert(hasMoreSplits() && "Reached end of entry point groups list.");
    EntryPointGroup Res = std::move(Groups.back());
    Groups.pop_back();
    return Res;
  }

  Module &getInputModule() { return Input.getModule(); }

  std::unique_ptr<Module> releaseInputModule() {
    return std::move(Input.releaseModulePtr());
  }

public:
  ModuleSplitterBase(ModuleDesc &&MD, EntryPointGroupVec &&GroupVec)
      : Input(std::move(MD)), Groups(std::move(GroupVec)) {
    assert(!Groups.empty() && "Entry points groups collection is empty!");
  }

  // For device global variables with the 'device_image_scope' property,
  // the function checks that there are no usages of a single device global
  // variable from kernels grouped to different modules. Otherwise, an error is
  // returned.
  Error verifyNoCrossModuleDeviceGlobalUsage();

  virtual ~ModuleSplitterBase() = default;

  // Gets next subsequence of entry points in an input module and provides split
  // submodule containing these entry points and their dependencies.
  virtual ModuleDesc nextSplit() = 0;

  // Returns a number of remaining modules, which can be split out using this
  // splitter. The value is reduced by 1 each time nextSplit is called.
  size_t remainingSplits() const { return Groups.size(); }

  // Check that there are still submodules to split.
  bool hasMoreSplits() const { return remainingSplits() > 0; }
};

SmallVector<ModuleDesc, 2> splitByESIMD(ModuleDesc &&MD,
                                        bool EmitOnlyKernelsAsEntryPoints);

std::unique_ptr<ModuleSplitterBase>
getDeviceCodeSplitter(ModuleDesc &&MD, IRSplitMode Mode, bool IROutputOnly,
                      bool EmitOnlyKernelsAsEntryPoints);

#ifndef NDEBUG
void dumpEntryPoints(const EntryPointSet &C, const char *msg = "", int Tab = 0);
void dumpEntryPoints(const Module &M, bool OnlyKernelsAreEntryPoints = false,
                     const char *msg = "", int Tab = 0);
#endif // NDEBUG

} // namespace module_split

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_MODULE_SPLITTER_H
