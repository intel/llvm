//===- Action.h - Abstract compilation steps --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_ACTION_H
#define LLVM_CLANG_DRIVER_ACTION_H

#include "clang/Basic/LLVM.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include <initializer_list>
#include <string>

namespace llvm {
namespace opt {

class Arg;

} // namespace opt
} // namespace llvm

namespace clang {
namespace driver {

class ToolChain;

/// Action - Represent an abstract compilation step to perform.
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
///
/// The current driver is hard wired to expect actions which produce a
/// single primary output, at least in terms of controlling the
/// compilation. Actions can produce auxiliary files, but can only
/// produce a single output to feed into subsequent actions.
///
/// Actions are usually owned by a Compilation, which creates new
/// actions via MakeAction().
class Action {
public:
  using size_type = ActionList::size_type;
  using input_iterator = ActionList::iterator;
  using input_const_iterator = ActionList::const_iterator;
  using input_range = llvm::iterator_range<input_iterator>;
  using input_const_range = llvm::iterator_range<input_const_iterator>;

  enum ActionClass {
    InputClass = 0,
    BindArchClass,
    OffloadClass,
    ForEachWrappingClass,
    PreprocessJobClass,
    PrecompileJobClass,
    ExtractAPIJobClass,
    AnalyzeJobClass,
    CompileJobClass,
    BackendJobClass,
    AssembleJobClass,
    LinkJobClass,
    IfsMergeJobClass,
    LipoJobClass,
    DsymutilJobClass,
    VerifyDebugInfoJobClass,
    VerifyPCHJobClass,
    OffloadBundlingJobClass,
    OffloadUnbundlingJobClass,
    OffloadWrapperJobClass,
    OffloadPackagerJobClass,
    OffloadDepsJobClass,
    SPIRVTranslatorJobClass,
    SYCLPostLinkJobClass,
    BackendCompileJobClass,
    FileTableTformJobClass,
    AppendFooterJobClass,
    SpirvToIrWrapperJobClass,
    LinkerWrapperJobClass,
    StaticLibJobClass,
    BinaryAnalyzeJobClass,
    BinaryTranslatorJobClass,

    JobClassFirst = PreprocessJobClass,
    JobClassLast = BinaryTranslatorJobClass
  };

  // The offloading kind determines if this action is binded to a particular
  // programming model. Each entry reserves one bit. We also have a special kind
  // to designate the host offloading tool chain.
  enum OffloadKind {
    OFK_None = 0x00,

    // The host offloading tool chain.
    OFK_Host = 0x01,

    // The device offloading tool chains - one bit for each programming model.
    OFK_Cuda = 0x02,
    OFK_OpenMP = 0x04,
    OFK_HIP = 0x08,
    OFK_SYCL = 0x10,
  };

  static const char *getClassName(ActionClass AC);

private:
  ActionClass Kind;

  /// The output type of this action.
  types::ID Type;

  ActionList Inputs;

  /// Flag that is set to true if this action can be collapsed with others
  /// actions that depend on it. This is true by default and set to false when
  /// the action is used by two different tool chains, which is enabled by the
  /// offloading support implementation.
  bool CanBeCollapsedWithNextDependentAction = true;

protected:
  ///
  /// Offload information.
  ///

  /// The host offloading kind - a combination of kinds encoded in a mask.
  /// Multiple programming models may be supported simultaneously by the same
  /// host.
  unsigned ActiveOffloadKindMask = 0u;

  /// Offloading kind of the device.
  OffloadKind OffloadingDeviceKind = OFK_None;

  /// The Offloading architecture associated with this action.
  const char *OffloadingArch = nullptr;

  /// The Offloading toolchain associated with this device action.
  const ToolChain *OffloadingToolChain = nullptr;

  Action(ActionClass Kind, types::ID Type) : Action(Kind, ActionList(), Type) {}
  Action(ActionClass Kind, Action *Input, types::ID Type)
      : Action(Kind, ActionList({Input}), Type) {}
  Action(ActionClass Kind, Action *Input)
      : Action(Kind, ActionList({Input}), Input->getType()) {}
  Action(ActionClass Kind, const ActionList &Inputs, types::ID Type)
      : Kind(Kind), Type(Type), Inputs(Inputs) {}

public:
  virtual ~Action();

  const char *getClassName() const { return Action::getClassName(getKind()); }

  ActionClass getKind() const { return Kind; }
  types::ID getType() const { return Type; }

  ActionList &getInputs() { return Inputs; }
  const ActionList &getInputs() const { return Inputs; }

  size_type size() const { return Inputs.size(); }

  input_iterator input_begin() { return Inputs.begin(); }
  input_iterator input_end() { return Inputs.end(); }
  input_range inputs() { return input_range(input_begin(), input_end()); }
  input_const_iterator input_begin() const { return Inputs.begin(); }
  input_const_iterator input_end() const { return Inputs.end(); }
  input_const_range inputs() const {
    return input_const_range(input_begin(), input_end());
  }

  /// Mark this action as not legal to collapse.
  void setCannotBeCollapsedWithNextDependentAction() {
    CanBeCollapsedWithNextDependentAction = false;
  }

  /// Return true if this function can be collapsed with others.
  bool isCollapsingWithNextDependentActionLegal() const {
    return CanBeCollapsedWithNextDependentAction;
  }

  /// Return a string containing the offload kind of the action.
  std::string getOffloadingKindPrefix() const;

  /// Return a string that can be used as prefix in order to generate unique
  /// files for each offloading kind. By default, no prefix is used for
  /// non-device kinds, except if \a CreatePrefixForHost is set.
  static std::string
  GetOffloadingFileNamePrefix(OffloadKind Kind,
                              StringRef NormalizedTriple,
                              bool CreatePrefixForHost = false);

  /// Return a string containing a offload kind name.
  static StringRef GetOffloadKindName(OffloadKind Kind);

  /// Set the device offload info of this action and propagate it to its
  /// dependences.
  void propagateDeviceOffloadInfo(OffloadKind OKind, const char *OArch,
                                  const ToolChain *OToolChain);

  /// Append the host offload info of this action and propagate it to its
  /// dependences.
  void propagateHostOffloadInfo(unsigned OKinds, const char *OArch);

  void setHostOffloadInfo(unsigned OKinds, const char *OArch) {
    ActiveOffloadKindMask |= OKinds;
    OffloadingArch = OArch;
  }

  /// Set the offload info of this action to be the same as the provided action,
  /// and propagate it to its dependences.
  void propagateOffloadInfo(const Action *A);

  unsigned getOffloadingHostActiveKinds() const {
    return ActiveOffloadKindMask;
  }

  OffloadKind getOffloadingDeviceKind() const { return OffloadingDeviceKind; }
  const char *getOffloadingArch() const { return OffloadingArch; }
  const ToolChain *getOffloadingToolChain() const {
    return OffloadingToolChain;
  }

  /// Check if this action have any offload kinds. Note that host offload kinds
  /// are only set if the action is a dependence to a host offload action.
  bool isHostOffloading(unsigned int OKind) const {
    return ActiveOffloadKindMask & OKind;
  }
  bool isDeviceOffloading(OffloadKind OKind) const {
    return OffloadingDeviceKind == OKind;
  }
  bool isOffloading(OffloadKind OKind) const {
    return isHostOffloading(OKind) || isDeviceOffloading(OKind);
  }
};

class InputAction : public Action {
  const llvm::opt::Arg &Input;
  std::string Id;
  virtual void anchor();

public:
  InputAction(const llvm::opt::Arg &Input, types::ID Type,
              StringRef Id = StringRef());

  const llvm::opt::Arg &getInputArg() const { return Input; }

  void setId(StringRef _Id) { Id = _Id.str(); }
  StringRef getId() const { return Id; }

  static bool classof(const Action *A) {
    return A->getKind() == InputClass;
  }
};

class BindArchAction : public Action {
  virtual void anchor();

  /// The architecture to bind, or 0 if the default architecture
  /// should be bound.
  StringRef ArchName;

public:
  BindArchAction(Action *Input, StringRef ArchName);

  StringRef getArchName() const { return ArchName; }

  static bool classof(const Action *A) {
    return A->getKind() == BindArchClass;
  }
};

/// An offload action combines host or/and device actions according to the
/// programming model implementation needs and propagates the offloading kind to
/// its dependences.
class OffloadAction final : public Action {
  LLVM_DECLARE_VIRTUAL_ANCHOR_FUNCTION();

public:
  /// Type used to communicate device actions. It associates bound architecture,
  /// toolchain, and offload kind to each action.
  class DeviceDependences final {
  public:
    using ToolChainList = SmallVector<const ToolChain *, 3>;
    using BoundArchList = SmallVector<const char *, 3>;
    using OffloadKindList = SmallVector<OffloadKind, 3>;

  private:
    // Lists that keep the information for each dependency. All the lists are
    // meant to be updated in sync. We are adopting separate lists instead of a
    // list of structs, because that simplifies forwarding the actions list to
    // initialize the inputs of the base Action class.

    /// The dependence actions.
    ActionList DeviceActions;

    /// The offloading toolchains that should be used with the action.
    ToolChainList DeviceToolChains;

    /// The architectures that should be used with this action.
    BoundArchList DeviceBoundArchs;

    /// The offload kind of each dependence.
    OffloadKindList DeviceOffloadKinds;

  public:
    /// Add an action along with the associated toolchain, bound arch, and
    /// offload kind.
    void add(Action &A, const ToolChain &TC, const char *BoundArch,
             OffloadKind OKind);

    /// Add an action along with the associated toolchain, bound arch, and
    /// offload kinds.
    void add(Action &A, const ToolChain &TC, const char *BoundArch,
             unsigned OffloadKindMask);

    /// Get each of the individual arrays.
    const ActionList &getActions() const { return DeviceActions; }
    const ToolChainList &getToolChains() const { return DeviceToolChains; }
    const BoundArchList &getBoundArchs() const { return DeviceBoundArchs; }
    const OffloadKindList &getOffloadKinds() const {
      return DeviceOffloadKinds;
    }
  };

  /// Type used to communicate host actions. It associates bound architecture,
  /// toolchain, and offload kinds to the host action.
  class HostDependence final {
    /// The dependence action.
    Action &HostAction;

    /// The offloading toolchain that should be used with the action.
    const ToolChain &HostToolChain;

    /// The architectures that should be used with this action.
    const char *HostBoundArch = nullptr;

    /// The offload kind of each dependence.
    unsigned HostOffloadKinds = 0u;

  public:
    HostDependence(Action &A, const ToolChain &TC, const char *BoundArch,
                   const unsigned OffloadKinds)
        : HostAction(A), HostToolChain(TC), HostBoundArch(BoundArch),
          HostOffloadKinds(OffloadKinds) {}

    /// Constructor version that obtains the offload kinds from the device
    /// dependencies.
    HostDependence(Action &A, const ToolChain &TC, const char *BoundArch,
                   const DeviceDependences &DDeps);
    Action *getAction() const { return &HostAction; }
    const ToolChain *getToolChain() const { return &HostToolChain; }
    const char *getBoundArch() const { return HostBoundArch; }
    unsigned getOffloadKinds() const { return HostOffloadKinds; }
  };

  using OffloadActionWorkTy =
      llvm::function_ref<void(Action *, const ToolChain *, const char *)>;

private:
  /// The host offloading toolchain that should be used with the action.
  const ToolChain *HostTC = nullptr;

  /// The tool chains associated with the list of actions.
  DeviceDependences::ToolChainList DevToolChains;

public:
  OffloadAction(const HostDependence &HDep);
  OffloadAction(const DeviceDependences &DDeps, types::ID Ty);
  OffloadAction(const HostDependence &HDep, const DeviceDependences &DDeps);

  /// Execute the work specified in \a Work on the host dependence.
  void doOnHostDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each device dependence.
  void doOnEachDeviceDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each dependence.
  void doOnEachDependence(const OffloadActionWorkTy &Work) const;

  /// Execute the work specified in \a Work on each host or device dependence if
  /// \a IsHostDependenceto is true or false, respectively.
  void doOnEachDependence(bool IsHostDependence,
                          const OffloadActionWorkTy &Work) const;

  /// Return true if the action has a host dependence.
  bool hasHostDependence() const;

  /// Return the host dependence of this action. This function is only expected
  /// to be called if the host dependence exists.
  Action *getHostDependence() const;

  /// Return true if the action has a single device dependence. If \a
  /// DoNotConsiderHostActions is set, ignore the host dependence, if any, while
  /// accounting for the number of dependences.
  bool hasSingleDeviceDependence(bool DoNotConsiderHostActions = false) const;

  /// Return the single device dependence of this action. This function is only
  /// expected to be called if a single device dependence exists. If \a
  /// DoNotConsiderHostActions is set, a host dependence is allowed.
  Action *
  getSingleDeviceDependence(bool DoNotConsiderHostActions = false) const;

  static bool classof(const Action *A) { return A->getKind() == OffloadClass; }
};

class JobAction : public Action {
  virtual void anchor();

protected:
  JobAction(ActionClass Kind, Action *Input, types::ID Type);
  JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type);

public:
  static bool classof(const Action *A) {
    return (A->getKind() >= JobClassFirst &&
            A->getKind() <= JobClassLast);
  }
};

class PreprocessJobAction : public JobAction {
  void anchor() override;

public:
  PreprocessJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PreprocessJobClass;
  }
};

class PrecompileJobAction : public JobAction {
  void anchor() override;

protected:
  PrecompileJobAction(ActionClass Kind, Action *Input, types::ID OutputType);

public:
  PrecompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == PrecompileJobClass;
  }
};

class ExtractAPIJobAction : public JobAction {
  void anchor() override;

public:
  ExtractAPIJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == ExtractAPIJobClass;
  }

  void addHeaderInput(Action *Input) { getInputs().push_back(Input); }
};

class AnalyzeJobAction : public JobAction {
  void anchor() override;

public:
  AnalyzeJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AnalyzeJobClass;
  }
};

class CompileJobAction : public JobAction {
  void anchor() override;

public:
  CompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == CompileJobClass;
  }
};

class BackendJobAction : public JobAction {
  void anchor() override;

public:
  BackendJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == BackendJobClass;
  }
};

class AssembleJobAction : public JobAction {
  void anchor() override;

public:
  AssembleJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == AssembleJobClass;
  }
};

class IfsMergeJobAction : public JobAction {
  void anchor() override;

public:
  IfsMergeJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == IfsMergeJobClass;
  }
};

class LinkJobAction : public JobAction {
  void anchor() override;

public:
  LinkJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LinkJobClass;
  }
};

class LipoJobAction : public JobAction {
  void anchor() override;

public:
  LipoJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LipoJobClass;
  }
};

class DsymutilJobAction : public JobAction {
  void anchor() override;

public:
  DsymutilJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == DsymutilJobClass;
  }
};

class VerifyJobAction : public JobAction {
  void anchor() override;

public:
  VerifyJobAction(ActionClass Kind, Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass ||
           A->getKind() == VerifyPCHJobClass;
  }
};

class VerifyDebugInfoJobAction : public VerifyJobAction {
  void anchor() override;

public:
  VerifyDebugInfoJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == VerifyDebugInfoJobClass;
  }
};

class VerifyPCHJobAction : public VerifyJobAction {
  void anchor() override;

public:
  VerifyPCHJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == VerifyPCHJobClass;
  }
};

class OffloadBundlingJobAction : public JobAction {
  void anchor() override;

public:
  // Offloading bundling doesn't change the type of output.
  OffloadBundlingJobAction(ActionList &Inputs);

  static bool classof(const Action *A) {
    return A->getKind() == OffloadBundlingJobClass;
  }
};

class OffloadUnbundlingJobAction final : public JobAction {
  void anchor() override;

public:
  /// Type that provides information about the actions that depend on this
  /// unbundling action.
  struct DependentActionInfo final {
    /// The tool chain of the dependent action.
    const ToolChain *DependentToolChain = nullptr;

    /// The bound architecture of the dependent action.
    StringRef DependentBoundArch;

    /// The offload kind of the dependent action.
    const OffloadKind DependentOffloadKind = OFK_None;

    DependentActionInfo(const ToolChain *DependentToolChain,
                        StringRef DependentBoundArch,
                        const OffloadKind DependentOffloadKind)
        : DependentToolChain(DependentToolChain),
          DependentBoundArch(DependentBoundArch),
          DependentOffloadKind(DependentOffloadKind) {}
  };

  /// Allow for a complete override of the target to unbundle.
  /// This is used for specific unbundles used for SYCL AOT when generating full
  /// device files that are bundled with the host object.
  void setTargetString(std::string Target) { TargetString = Target; }

  std::string getTargetString() const { return TargetString; }

private:
  /// Container that keeps information about each dependence of this unbundling
  /// action.
  SmallVector<DependentActionInfo, 6> DependentActionInfoArray;

  /// Provides a specific type to be used that overrides the input type.
  types::ID DependentType = types::TY_Nothing;

  std::string TargetString;

public:
  // Offloading unbundling doesn't change the type of output.
  OffloadUnbundlingJobAction(Action *Input);
  OffloadUnbundlingJobAction(Action *Input, types::ID Type);
  OffloadUnbundlingJobAction(ActionList &Inputs, types::ID Type);

  /// Register information about a dependent action.
  void registerDependentActionInfo(const ToolChain *TC, StringRef BoundArch,
                                   OffloadKind Kind) {
    DependentActionInfoArray.push_back({TC, BoundArch, Kind});
  }

  /// Return the information about all depending actions.
  ArrayRef<DependentActionInfo> getDependentActionsInfo() const {
    return DependentActionInfoArray;
  }

  static bool classof(const Action *A) {
    return A->getKind() == OffloadUnbundlingJobClass;
  }

  /// Set the dependent type.
  void setDependentType(types::ID Type) { DependentType = Type; }

  /// Get the dependent type.
  types::ID getDependentType() const { return DependentType; }
};

class OffloadWrapperJobAction : public JobAction {
  void anchor() override;

  bool EmbedIR;

public:
  OffloadWrapperJobAction(ActionList &Inputs, types::ID Type);
  OffloadWrapperJobAction(Action *Input, types::ID OutputType,
                          bool EmbedIR = false);

  bool isEmbeddedIR() const { return EmbedIR; }

  static bool classof(const Action *A) {
    return A->getKind() == OffloadWrapperJobClass;
  }

  // Set the compilation step setting.  This is used to tell the wrapper job
  // action that the compilation step to create the object should be performed
  // after the wrapping step is complete.
  void setCompileStep(bool SetValue) { CompileStep = SetValue; }

  // Get the compilation step setting.
  bool getCompileStep() const { return CompileStep; }

  // Set the individual wrapping setting.  This is used to tell the wrapper job
  // action that the wrapping (and subsequent compile step) should be done
  // with for-each instead of using -batch.
  void setWrapIndividualFiles() { WrapIndividualFiles = true; }

  // Get the individual wrapping setting.
  bool getWrapIndividualFiles() const { return WrapIndividualFiles; }

  // Set the offload kind for the current wrapping job action.  Default usage
  // is to use the kind of the current toolchain.
  void setOffloadKind(OffloadKind SetKind) { Kind = SetKind; }

  // Get the offload kind.
  OffloadKind getOffloadKind() const { return Kind; }

private:
  bool CompileStep = true;
  bool WrapIndividualFiles = false;
  OffloadKind Kind = OFK_None;
};

class OffloadPackagerJobAction : public JobAction {
  void anchor() override;

public:
  OffloadPackagerJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == OffloadPackagerJobClass;
  }
};

class OffloadDepsJobAction final : public JobAction {
  void anchor() override;

public:
  /// Type that provides information about the actions that depend on this
  /// offload deps action.
  struct DependentActionInfo final {
    /// The tool chain of the dependent action.
    const ToolChain *DependentToolChain = nullptr;

    /// The bound architecture of the dependent action.
    StringRef DependentBoundArch;

    /// The offload kind of the dependent action.
    const OffloadKind DependentOffloadKind = OFK_None;

    DependentActionInfo(const ToolChain *DependentToolChain,
                        StringRef DependentBoundArch,
                        const OffloadKind DependentOffloadKind)
        : DependentToolChain(DependentToolChain),
          DependentBoundArch(DependentBoundArch),
          DependentOffloadKind(DependentOffloadKind) {}
  };

private:
  /// The host offloading toolchain that should be used with the action.
  const ToolChain *HostTC = nullptr;

  /// Container that keeps information about each dependence of this deps
  /// action.
  SmallVector<DependentActionInfo, 6> DependentActionInfoArray;

public:
  OffloadDepsJobAction(const OffloadAction::HostDependence &HDep,
                       types::ID Type);

  /// Register information about a dependent action.
  void registerDependentActionInfo(const ToolChain *TC, StringRef BoundArch,
                                   OffloadKind Kind) {
    DependentActionInfoArray.push_back({TC, BoundArch, Kind});
  }

  /// Return the information about all depending actions.
  ArrayRef<DependentActionInfo> getDependentActionsInfo() const {
    return DependentActionInfoArray;
  }

  const ToolChain *getHostTC() const { return HostTC; }

  static bool classof(const Action *A) {
    return A->getKind() == OffloadDepsJobClass;
  }
};

class SPIRVTranslatorJobAction : public JobAction {
  void anchor() override;

public:
  SPIRVTranslatorJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == SPIRVTranslatorJobClass;
  }
};

class SYCLPostLinkJobAction : public JobAction {
  void anchor() override;

public:
  // The tempfiletable management relies on shadowing the main file type by
  // types::TY_Tempfiletable. The problem of shadowing is it prevents its
  // integration with clang tools that relies on the file type to properly set
  // args.
  // We "trick" the driver by declaring the underlying file type and set a
  // "true output type" which will be used by the SYCLPostLinkJobAction
  // to properly set the job.
  SYCLPostLinkJobAction(Action *Input, types::ID ShadowOutputType,
                        types::ID TrueOutputType);

  static bool classof(const Action *A) {
    return A->getKind() == SYCLPostLinkJobClass;
  }

  void setRTSetsSpecConstants(bool Val) { RTSetsSpecConsts = Val; }

  bool getRTSetsSpecConstants() const { return RTSetsSpecConsts; }

  types::ID getTrueType() const { return TrueOutputType; }

private:
  bool RTSetsSpecConsts = true;
  types::ID TrueOutputType;
};

class BackendCompileJobAction : public JobAction {
  void anchor() override;

public:
  BackendCompileJobAction(ActionList &Inputs, types::ID OutputType);
  BackendCompileJobAction(Action *Input, types::ID OutputType);

  static bool classof(const Action *A) {
    return A->getKind() == BackendCompileJobClass;
  }
};

// Represents a file table transformation action. The order of inputs to a
// FileTableTformJobAction at construction time must accord with the tforms
// added later - some tforms "consume" inputs. For example, "replace column"
// needs another file to read the replacement column from.
class FileTableTformJobAction : public JobAction {
  void anchor() override;

public:
  static constexpr const char *COL_CODE = "Code";
  static constexpr const char *COL_ZERO = "0";
  static constexpr const char *COL_SYM_AND_PROPS = "SymAndProps";

  struct Tform {
    enum Kind {
      EXTRACT,
      EXTRACT_DROP_TITLE,
      REPLACE,
      REPLACE_CELL,
      RENAME,
      COPY_SINGLE_FILE,
      MERGE
    };

    Tform() = default;
    Tform(Kind K, std::initializer_list<StringRef> Args) : TheKind(K) {
      for (auto &A : Args)
        TheArgs.emplace_back(A.str());
    }

    Kind TheKind;
    SmallVector<std::string, 2> TheArgs;
  };

  FileTableTformJobAction(Action *Input, types::ID ShadowOutputType,
                          types::ID TrueOutputType);
  FileTableTformJobAction(ActionList &Inputs, types::ID ShadowOutputType,
                          types::ID TrueOutputType);

  // Deletes all columns except the one with given name.
  void addExtractColumnTform(StringRef ColumnName, bool WithColTitle = true);

  // Replaces a column with title <From> in this table with a column with title
  // <To> from another file table passed as input to this action.
  void addReplaceColumnTform(StringRef From, StringRef To);

  // Replaces a cell in this table with column title <ColumnName> and row <Row>
  // with the file name passed as input to this action.
  void addReplaceCellTform(StringRef ColumnName, int Row);

  // Renames a column with title <From> in this table with a column with title
  // <To> passed as input to this action.
  void addRenameColumnTform(StringRef From, StringRef To);

  // Specifies that, instead of generating a new table, the transformation
  // should copy the file at column <ColumnName> and row <Row> into the
  // output file.
  void addCopySingleFileTform(StringRef ColumnName, int Row);

  // Merges all tables from filename listed at column <ColumnName> into a
  // single output table.
  void addMergeTform(StringRef ColumnName);

  static bool classof(const Action *A) {
    return A->getKind() == FileTableTformJobClass;
  }

  const ArrayRef<Tform> getTforms() const { return Tforms; }

  types::ID getTrueType() const { return TrueOutputType; }

private:
  types::ID TrueOutputType;
  SmallVector<Tform, 2> Tforms; // transformation actions requested

  // column to copy single file from if requested
  std::string CopySingleFileColumnName;
};

class AppendFooterJobAction : public JobAction {
  void anchor() override;

public:
  AppendFooterJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == AppendFooterJobClass;
  }
};

class SpirvToIrWrapperJobAction : public JobAction {
  void anchor() override;

public:
  SpirvToIrWrapperJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == SpirvToIrWrapperJobClass;
  }
};

class LinkerWrapperJobAction : public JobAction {
  void anchor() override;

public:
  LinkerWrapperJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == LinkerWrapperJobClass;
  }
};

class StaticLibJobAction : public JobAction {
  void anchor() override;

public:
  StaticLibJobAction(ActionList &Inputs, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == StaticLibJobClass;
  }
};

/// Wrap all jobs performed between TFormInput (excluded) and Job (included)
/// behind a `llvm-foreach` call.
///
/// Assumptions:
///   - No change of toolchain, boundarch and offloading kind should occur
///     within the sub-region;
///   - No job should produce multiple outputs;
///   - Results of action within the sub-region should not be used outside the
///     wrapped region.
/// Note: this doesn't bind to a tool directly and this need special casing
/// anyhow. Hence why this is an Action and not a JobAction, even if there is a
/// command behind.
class ForEachWrappingAction : public Action {
public:
  ForEachWrappingAction(JobAction *TFormInput, JobAction *Job);

  JobAction *getTFormInput() const;
  JobAction *getJobAction() const;

  static bool classof(const Action *A) {
    return A->getKind() == ForEachWrappingClass;
  }

  void addSerialAction(const Action *A) { SerialActions.insert(A); }
  const llvm::SmallSetVector<const Action *, 2> &getSerialActions() const {
    return SerialActions;
  }

private:
  llvm::SmallSetVector<const Action *, 2> SerialActions;
};

class BinaryAnalyzeJobAction : public JobAction {
  void anchor() override;

public:
  BinaryAnalyzeJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == BinaryAnalyzeJobClass;
  }
};

class BinaryTranslatorJobAction : public JobAction {
  void anchor() override;

public:
  BinaryTranslatorJobAction(Action *Input, types::ID Type);

  static bool classof(const Action *A) {
    return A->getKind() == BinaryTranslatorJobClass;
  }
};

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_ACTION_H
