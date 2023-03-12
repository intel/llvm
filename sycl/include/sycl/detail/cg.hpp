//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/host_profiling_info.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/group.hpp>
#include <sycl/id.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/interop_handler.hpp>
#include <sycl/kernel.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// Forward declarations
class queue;

namespace detail {

class event_impl;
using EventImplPtr = std::shared_ptr<event_impl>;

class stream_impl;
class queue_impl;
class kernel_bundle_impl;

// If there's a need to add new members to CG classes without breaking ABI
// compatibility, we can bring back the extended members mechanism. See
// https://github.com/intel/llvm/pull/6759
/// Base class for all types of command groups.
class CG {
public:
  /// Type of the command group.
  enum CGTYPE : unsigned int {
    None = 0,
    Kernel = 1,
    CopyAccToPtr = 2,
    CopyPtrToAcc = 3,
    CopyAccToAcc = 4,
    Barrier = 5,
    BarrierWaitlist = 6,
    Fill = 7,
    UpdateHost = 8,
    RunOnHostIntel = 9,
    CopyUSM = 10,
    FillUSM = 11,
    PrefetchUSM = 12,
    CodeplayInteropTask = 13,
    CodeplayHostTask = 14,
    AdviseUSM = 15,
    Copy2DUSM = 16,
    Fill2DUSM = 17,
    Memset2DUSM = 18,
    CopyToDeviceGlobal = 19,
    CopyFromDeviceGlobal = 20,
  };

  CG(CGTYPE Type, std::vector<std::vector<char>> ArgsStorage,
     std::vector<detail::AccessorImplPtr> AccStorage,
     std::vector<std::shared_ptr<const void>> SharedPtrStorage,
     std::vector<AccessorImplHost *> Requirements,
     std::vector<detail::EventImplPtr> Events, detail::code_location loc = {})
      : MType(Type), MArgsStorage(std::move(ArgsStorage)),
        MAccStorage(std::move(AccStorage)),
        MSharedPtrStorage(std::move(SharedPtrStorage)),
        MRequirements(std::move(Requirements)), MEvents(std::move(Events)) {
    // Capture the user code-location from Q.submit(), Q.parallel_for()
    // etc for later use; if code location information is not available,
    // the file name and function name members will be empty strings
    if (loc.functionName())
      MFunctionName = loc.functionName();
    if (loc.fileName())
      MFileName = loc.fileName();
    MLine = loc.lineNumber();
    MColumn = loc.columnNumber();
  }

  CG(CG &&CommandGroup) = default;

  CGTYPE getType() { return MType; }

  std::vector<std::vector<char>> &getArgsStorage() { return MArgsStorage; }

  std::vector<detail::AccessorImplPtr> &getAccStorage() { return MAccStorage; }

  virtual ~CG() = default;

private:
  CGTYPE MType;
  // The following storages are needed to ensure that arguments won't die while
  // we are using them.
  /// Storage for standard layout arguments.
  std::vector<std::vector<char>> MArgsStorage;
  /// Storage for accessors.
  std::vector<detail::AccessorImplPtr> MAccStorage;
  /// Storage for shared_ptrs.
  std::vector<std::shared_ptr<const void>> MSharedPtrStorage;

public:
  /// List of requirements that specify which memory is needed for the command
  /// group to be executed.
  std::vector<AccessorImplHost *> MRequirements;
  /// List of events that order the execution of this CG
  std::vector<detail::EventImplPtr> MEvents;
  // Member variables to capture the user code-location
  // information from Q.submit(), Q.parallel_for() etc
  // Storage for function name and source file name
  std::string MFunctionName, MFileName;
  // Storage for line and column of code location
  int32_t MLine, MColumn;
};

/// "Execute kernel" command group class.
class CGExecKernel : public CG {
public:
  /// Stores ND-range description.
  NDRDescT MNDRDesc;
  std::unique_ptr<HostKernelBase> MHostKernel;
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  std::shared_ptr<detail::kernel_bundle_impl> MKernelBundle;
  std::vector<ArgDesc> MArgs;
  std::string MKernelName;
  detail::OSModuleHandle MOSModuleHandle;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreams;
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;

  CGExecKernel(NDRDescT NDRDesc, std::unique_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::shared_ptr<detail::kernel_bundle_impl> KernelBundle,
               std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<AccessorImplHost *> Requirements,
               std::vector<detail::EventImplPtr> Events,
               std::vector<ArgDesc> Args, std::string KernelName,
               detail::OSModuleHandle OSModuleHandle,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams,
               std::vector<std::shared_ptr<const void>> AuxiliaryResources,
               CGTYPE Type, detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)),
        MKernelBundle(std::move(KernelBundle)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MOSModuleHandle(OSModuleHandle),
        MStreams(std::move(Streams)),
        MAuxiliaryResources(std::move(AuxiliaryResources)) {
    assert((getType() == RunOnHostIntel || getType() == Kernel) &&
           "Wrong type of exec kernel CG.");
  }

  std::vector<ArgDesc> getArguments() const { return MArgs; }
  std::string getKernelName() const { return MKernelName; }
  std::vector<std::shared_ptr<detail::stream_impl>> getStreams() const {
    return MStreams;
  }

  std::vector<std::shared_ptr<const void>> getAuxiliaryResources() const {
    return MAuxiliaryResources;
  }

  std::shared_ptr<detail::kernel_bundle_impl> getKernelBundle() {
    return MKernelBundle;
  }

  void clearStreams() { MStreams.clear(); }
  bool hasStreams() { return !MStreams.empty(); }

  void clearAuxiliaryResources() { MAuxiliaryResources.clear(); }
  bool hasAuxiliaryResources() { return !MAuxiliaryResources.empty(); }
};

/// "Copy memory" command group class.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<AccessorImplHost *> Requirements,
         std::vector<detail::EventImplPtr> Events,
         detail::code_location loc = {})
      : CG(CopyType, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst) {}
  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
};

/// "Fill memory" command group class.
class CGFill : public CG {
public:
  std::vector<char> MPattern;
  AccessorImplHost *MPtr;

  CGFill(std::vector<char> Pattern, void *Ptr,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<AccessorImplHost *> Requirements,
         std::vector<detail::EventImplPtr> Events,
         detail::code_location loc = {})
      : CG(Fill, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MPtr((AccessorImplHost *)Ptr) {}
  AccessorImplHost *getReqToFill() { return MPtr; }
};

/// "Update host" command group class.
class CGUpdateHost : public CG {
  AccessorImplHost *MPtr;

public:
  CGUpdateHost(void *Ptr, std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<AccessorImplHost *> Requirements,
               std::vector<detail::EventImplPtr> Events,
               detail::code_location loc = {})
      : CG(UpdateHost, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPtr((AccessorImplHost *)Ptr) {}

  AccessorImplHost *getReqToUpdate() { return MPtr; }
};

/// "Copy USM" command group class.
class CGCopyUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MLength;

public:
  CGCopyUSM(void *Src, void *Dst, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<AccessorImplHost *> Requirements,
            std::vector<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(CopyUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst), MLength(Length) {}

  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Fill USM" command group class.
class CGFillUSM : public CG {
  std::vector<char> MPattern;
  void *MDst;
  size_t MLength;

public:
  CGFillUSM(std::vector<char> Pattern, void *DstPtr, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<AccessorImplHost *> Requirements,
            std::vector<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(FillUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  int getFill() { return MPattern[0]; }
};

/// "Prefetch USM" command group class.
class CGPrefetchUSM : public CG {
  void *MDst;
  size_t MLength;

public:
  CGPrefetchUSM(void *DstPtr, size_t Length,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<AccessorImplHost *> Requirements,
                std::vector<detail::EventImplPtr> Events,
                detail::code_location loc = {})
      : CG(PrefetchUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Advise USM" command group class.
class CGAdviseUSM : public CG {
  void *MDst;
  size_t MLength;
  pi_mem_advice MAdvice;

public:
  CGAdviseUSM(void *DstPtr, size_t Length, pi_mem_advice Advice,
              std::vector<std::vector<char>> ArgsStorage,
              std::vector<detail::AccessorImplPtr> AccStorage,
              std::vector<std::shared_ptr<const void>> SharedPtrStorage,
              std::vector<AccessorImplHost *> Requirements,
              std::vector<detail::EventImplPtr> Events, CGTYPE Type,
              detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDst(DstPtr), MLength(Length), MAdvice(Advice) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  pi_mem_advice getAdvice() { return MAdvice; }
};

class CGInteropTask : public CG {
public:
  std::unique_ptr<InteropTask> MInteropTask;

  CGInteropTask(std::unique_ptr<InteropTask> InteropTask,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<AccessorImplHost *> Requirements,
                std::vector<detail::EventImplPtr> Events, CGTYPE Type,
                detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MInteropTask(std::move(InteropTask)) {}
};

class CGHostTask : public CG {
public:
  std::unique_ptr<HostTask> MHostTask;
  // queue for host-interop task
  std::shared_ptr<detail::queue_impl> MQueue;
  // context for host-interop task
  std::shared_ptr<detail::context_impl> MContext;
  std::vector<ArgDesc> MArgs;

  CGHostTask(std::unique_ptr<HostTask> HostTask,
             std::shared_ptr<detail::queue_impl> Queue,
             std::shared_ptr<detail::context_impl> Context,
             std::vector<ArgDesc> Args,
             std::vector<std::vector<char>> ArgsStorage,
             std::vector<detail::AccessorImplPtr> AccStorage,
             std::vector<std::shared_ptr<const void>> SharedPtrStorage,
             std::vector<AccessorImplHost *> Requirements,
             std::vector<detail::EventImplPtr> Events, CGTYPE Type,
             detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MHostTask(std::move(HostTask)), MQueue(Queue), MContext(Context),
        MArgs(std::move(Args)) {}
};

class CGBarrier : public CG {
public:
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  CGBarrier(std::vector<detail::EventImplPtr> EventsWaitWithBarrier,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<AccessorImplHost *> Requirements,
            std::vector<detail::EventImplPtr> Events, CGTYPE Type,
            detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MEventsWaitWithBarrier(std::move(EventsWaitWithBarrier)) {}
};

/// "Copy 2D USM" command group class.
class CGCopy2DUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MSrcPitch;
  size_t MDstPitch;
  size_t MWidth;
  size_t MHeight;

public:
  CGCopy2DUSM(void *Src, void *Dst, size_t SrcPitch, size_t DstPitch,
              size_t Width, size_t Height,
              std::vector<std::vector<char>> ArgsStorage,
              std::vector<detail::AccessorImplPtr> AccStorage,
              std::vector<std::shared_ptr<const void>> SharedPtrStorage,
              std::vector<AccessorImplHost *> Requirements,
              std::vector<detail::EventImplPtr> Events,
              detail::code_location loc = {})
      : CG(Copy2DUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst), MSrcPitch(SrcPitch), MDstPitch(DstPitch),
        MWidth(Width), MHeight(Height) {}

  void *getSrc() const { return MSrc; }
  void *getDst() const { return MDst; }
  size_t getSrcPitch() const { return MSrcPitch; }
  size_t getDstPitch() const { return MDstPitch; }
  size_t getWidth() const { return MWidth; }
  size_t getHeight() const { return MHeight; }
};

/// "Fill 2D USM" command group class.
class CGFill2DUSM : public CG {
  std::vector<char> MPattern;
  void *MDst;
  size_t MPitch;
  size_t MWidth;
  size_t MHeight;

public:
  CGFill2DUSM(std::vector<char> Pattern, void *DstPtr, size_t Pitch,
              size_t Width, size_t Height,
              std::vector<std::vector<char>> ArgsStorage,
              std::vector<detail::AccessorImplPtr> AccStorage,
              std::vector<std::shared_ptr<const void>> SharedPtrStorage,
              std::vector<AccessorImplHost *> Requirements,
              std::vector<detail::EventImplPtr> Events,
              detail::code_location loc = {})
      : CG(Fill2DUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MPitch(Pitch),
        MWidth(Width), MHeight(Height) {}
  void *getDst() const { return MDst; }
  size_t getPitch() const { return MPitch; }
  size_t getWidth() const { return MWidth; }
  size_t getHeight() const { return MHeight; }
  const std::vector<char> &getPattern() const { return MPattern; }
};

/// "Memset 2D USM" command group class.
class CGMemset2DUSM : public CG {
  char MValue;
  void *MDst;
  size_t MPitch;
  size_t MWidth;
  size_t MHeight;

public:
  CGMemset2DUSM(char Value, void *DstPtr, size_t Pitch, size_t Width,
                size_t Height, std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<AccessorImplHost *> Requirements,
                std::vector<detail::EventImplPtr> Events,
                detail::code_location loc = {})
      : CG(Memset2DUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MValue(Value), MDst(DstPtr), MPitch(Pitch), MWidth(Width),
        MHeight(Height) {}
  void *getDst() const { return MDst; }
  size_t getPitch() const { return MPitch; }
  size_t getWidth() const { return MWidth; }
  size_t getHeight() const { return MHeight; }
  char getValue() const { return MValue; }
};

/// "Copy to device_global" command group class.
class CGCopyToDeviceGlobal : public CG {
  void *MSrc;
  void *MDeviceGlobalPtr;
  bool MIsDeviceImageScoped;
  size_t MNumBytes;
  size_t MOffset;
  detail::OSModuleHandle MOSModuleHandle;

public:
  CGCopyToDeviceGlobal(
      void *Src, void *DeviceGlobalPtr, bool IsDeviceImageScoped,
      size_t NumBytes, size_t Offset,
      std::vector<std::vector<char>> ArgsStorage,
      std::vector<detail::AccessorImplPtr> AccStorage,
      std::vector<std::shared_ptr<const void>> SharedPtrStorage,
      std::vector<AccessorImplHost *> Requirements,
      std::vector<detail::EventImplPtr> Events,
      detail::OSModuleHandle OSModuleHandle, detail::code_location loc = {})
      : CG(CopyToDeviceGlobal, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDeviceGlobalPtr(DeviceGlobalPtr),
        MIsDeviceImageScoped(IsDeviceImageScoped), MNumBytes(NumBytes),
        MOffset(Offset), MOSModuleHandle(OSModuleHandle) {}

  void *getSrc() { return MSrc; }
  void *getDeviceGlobalPtr() { return MDeviceGlobalPtr; }
  bool isDeviceImageScoped() { return MIsDeviceImageScoped; }
  size_t getNumBytes() { return MNumBytes; }
  size_t getOffset() { return MOffset; }
  detail::OSModuleHandle getOSModuleHandle() { return MOSModuleHandle; }
};

/// "Copy to device_global" command group class.
class CGCopyFromDeviceGlobal : public CG {
  void *MDeviceGlobalPtr;
  void *MDest;
  bool MIsDeviceImageScoped;
  size_t MNumBytes;
  size_t MOffset;
  detail::OSModuleHandle MOSModuleHandle;

public:
  CGCopyFromDeviceGlobal(
      void *DeviceGlobalPtr, void *Dest, bool IsDeviceImageScoped,
      size_t NumBytes, size_t Offset,
      std::vector<std::vector<char>> ArgsStorage,
      std::vector<detail::AccessorImplPtr> AccStorage,
      std::vector<std::shared_ptr<const void>> SharedPtrStorage,
      std::vector<AccessorImplHost *> Requirements,
      std::vector<detail::EventImplPtr> Events,
      detail::OSModuleHandle OSModuleHandle, detail::code_location loc = {})
      : CG(CopyFromDeviceGlobal, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDeviceGlobalPtr(DeviceGlobalPtr), MDest(Dest),
        MIsDeviceImageScoped(IsDeviceImageScoped), MNumBytes(NumBytes),
        MOffset(Offset), MOSModuleHandle(OSModuleHandle) {}

  void *getDeviceGlobalPtr() { return MDeviceGlobalPtr; }
  void *getDest() { return MDest; }
  bool isDeviceImageScoped() { return MIsDeviceImageScoped; }
  size_t getNumBytes() { return MNumBytes; }
  size_t getOffset() { return MOffset; }
  detail::OSModuleHandle getOSModuleHandle() { return MOSModuleHandle; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
