//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>        // for AccessorImplHost, AccessorImplPtr
#include <sycl/detail/cg_types.hpp> // for ArgDesc, HostTask, HostKernelBase
#include <sycl/detail/common.hpp>   // for code_location
#include <sycl/detail/helpers.hpp>  // for context_impl
#include <sycl/detail/pi.h>         // for pi_mem_advice, _pi_ext_command_b...
#include <sycl/detail/pi.hpp>       // for PiImageOffset, PiImageRegion
#include <sycl/event.hpp>           // for event_impl
#include <sycl/exception_list.hpp>  // for queue_impl
#include <sycl/kernel.hpp>          // for kernel_impl
#include <sycl/kernel_bundle.hpp>   // for kernel_bundle_impl

#include <assert.h> // for assert
#include <memory>   // for shared_ptr, unique_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for int32_t
#include <string>   // for string
#include <utility>  // for move
#include <vector>   // for vector

namespace sycl {
inline namespace _V1 {

// Forward declarations
class queue;

namespace ext::oneapi::experimental::detail {
class exec_graph_impl;
}

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
    CopyUSM = 10,
    FillUSM = 11,
    PrefetchUSM = 12,
    CodeplayHostTask = 14,
    AdviseUSM = 15,
    Copy2DUSM = 16,
    Fill2DUSM = 17,
    Memset2DUSM = 18,
    CopyToDeviceGlobal = 19,
    CopyFromDeviceGlobal = 20,
    ReadWriteHostPipe = 21,
    ExecCommandBuffer = 22,
    CopyImage = 23,
    SemaphoreWait = 24,
    SemaphoreSignal = 25,
  };

  struct StorageInitHelper {
    StorageInitHelper() = default;
    StorageInitHelper(std::vector<std::vector<char>> ArgsStorage,
                      std::vector<detail::AccessorImplPtr> AccStorage,
                      std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                      std::vector<AccessorImplHost *> Requirements,
                      std::vector<detail::EventImplPtr> Events)
        : MArgsStorage(std::move(ArgsStorage)),
          MAccStorage(std::move(AccStorage)),
          MSharedPtrStorage(std::move(SharedPtrStorage)),
          MRequirements(std::move(Requirements)), MEvents(std::move(Events)) {}
    StorageInitHelper(StorageInitHelper &&) = default;
    StorageInitHelper(const StorageInitHelper &) = default;
    // The following storages are needed to ensure that arguments won't die
    // while we are using them.
    /// Storage for standard layout arguments.
    std::vector<std::vector<char>> MArgsStorage;
    /// Storage for accessors.
    std::vector<detail::AccessorImplPtr> MAccStorage;
    /// Storage for shared_ptrs.
    std::vector<std::shared_ptr<const void>> MSharedPtrStorage;

    /// List of requirements that specify which memory is needed for the command
    /// group to be executed.
    std::vector<AccessorImplHost *> MRequirements;
    /// List of events that order the execution of this CG
    std::vector<detail::EventImplPtr> MEvents;
  };

  CG(CGTYPE Type, StorageInitHelper D, detail::code_location loc = {})
      : MType(Type), MData(std::move(D)) {
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
  CG(const CG &CommandGroup) = default;

  CGTYPE getType() const { return MType; }

  std::vector<std::vector<char>> &getArgsStorage() {
    return MData.MArgsStorage;
  }
  std::vector<detail::AccessorImplPtr> &getAccStorage() {
    return MData.MAccStorage;
  }
  std::vector<std::shared_ptr<const void>> &getSharedPtrStorage() {
    return MData.MSharedPtrStorage;
  }

  std::vector<AccessorImplHost *> &getRequirements() {
    return MData.MRequirements;
  }
  std::vector<detail::EventImplPtr> &getEvents() { return MData.MEvents; }

  virtual std::vector<std::shared_ptr<const void>>
  getAuxiliaryResources() const {
    return {};
  }
  virtual void clearAuxiliaryResources(){};

  virtual ~CG() = default;

private:
  CGTYPE MType;
  StorageInitHelper MData;

public:
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
  std::shared_ptr<HostKernelBase> MHostKernel;
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  std::shared_ptr<detail::kernel_bundle_impl> MKernelBundle;
  std::vector<ArgDesc> MArgs;
  std::string MKernelName;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreams;
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;
  sycl::detail::pi::PiKernelCacheConfig MKernelCacheConfig;
  bool MKernelIsCooperative = false;

  CGExecKernel(NDRDescT NDRDesc, std::shared_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::shared_ptr<detail::kernel_bundle_impl> KernelBundle,
               CG::StorageInitHelper CGData, std::vector<ArgDesc> Args,
               std::string KernelName,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams,
               std::vector<std::shared_ptr<const void>> AuxiliaryResources,
               CGTYPE Type,
               sycl::detail::pi::PiKernelCacheConfig KernelCacheConfig,
               bool KernelIsCooperative, detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)),
        MKernelBundle(std::move(KernelBundle)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MStreams(std::move(Streams)),
        MAuxiliaryResources(std::move(AuxiliaryResources)),
        MKernelCacheConfig(std::move(KernelCacheConfig)),
        MKernelIsCooperative(KernelIsCooperative) {
    assert(getType() == Kernel && "Wrong type of exec kernel CG.");
  }

  CGExecKernel(const CGExecKernel &CGExec) = default;

  std::vector<ArgDesc> getArguments() const { return MArgs; }
  std::string getKernelName() const { return MKernelName; }
  std::vector<std::shared_ptr<detail::stream_impl>> getStreams() const {
    return MStreams;
  }

  std::vector<std::shared_ptr<const void>>
  getAuxiliaryResources() const override {
    return MAuxiliaryResources;
  }
  void clearAuxiliaryResources() override { MAuxiliaryResources.clear(); }

  std::shared_ptr<detail::kernel_bundle_impl> getKernelBundle() {
    return MKernelBundle;
  }

  void clearStreams() { MStreams.clear(); }
  bool hasStreams() { return !MStreams.empty(); }
};

/// "Copy memory" command group class.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst, CG::StorageInitHelper CGData,
         std::vector<std::shared_ptr<const void>> AuxiliaryResources,
         detail::code_location loc = {})
      : CG(CopyType, std::move(CGData), std::move(loc)), MSrc(Src), MDst(Dst),
        MAuxiliaryResources{AuxiliaryResources} {}
  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }

  std::vector<std::shared_ptr<const void>>
  getAuxiliaryResources() const override {
    return MAuxiliaryResources;
  }
  void clearAuxiliaryResources() override { MAuxiliaryResources.clear(); }
};

/// "Fill memory" command group class.
class CGFill : public CG {
public:
  std::vector<char> MPattern;
  AccessorImplHost *MPtr;

  CGFill(std::vector<char> Pattern, void *Ptr, CG::StorageInitHelper CGData,
         detail::code_location loc = {})
      : CG(Fill, std::move(CGData), std::move(loc)),
        MPattern(std::move(Pattern)), MPtr((AccessorImplHost *)Ptr) {}
  AccessorImplHost *getReqToFill() { return MPtr; }
};

/// "Update host" command group class.
class CGUpdateHost : public CG {
  AccessorImplHost *MPtr;

public:
  CGUpdateHost(void *Ptr, CG::StorageInitHelper CGData,
               detail::code_location loc = {})
      : CG(UpdateHost, std::move(CGData), std::move(loc)),
        MPtr((AccessorImplHost *)Ptr) {}

  AccessorImplHost *getReqToUpdate() { return MPtr; }
};

/// "Copy USM" command group class.
class CGCopyUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MLength;

public:
  CGCopyUSM(void *Src, void *Dst, size_t Length, CG::StorageInitHelper CGData,
            detail::code_location loc = {})
      : CG(CopyUSM, std::move(CGData), std::move(loc)), MSrc(Src), MDst(Dst),
        MLength(Length) {}

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
            CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(FillUSM, std::move(CGData), std::move(loc)),
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
  CGPrefetchUSM(void *DstPtr, size_t Length, CG::StorageInitHelper CGData,
                detail::code_location loc = {})
      : CG(PrefetchUSM, std::move(CGData), std::move(loc)), MDst(DstPtr),
        MLength(Length) {}
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
              CG::StorageInitHelper CGData, CGTYPE Type,
              detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)), MDst(DstPtr),
        MLength(Length), MAdvice(Advice) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  pi_mem_advice getAdvice() { return MAdvice; }
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
             std::vector<ArgDesc> Args, CG::StorageInitHelper CGData,
             CGTYPE Type, detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MHostTask(std::move(HostTask)), MQueue(Queue), MContext(Context),
        MArgs(std::move(Args)) {}
};

class CGBarrier : public CG {
public:
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  CGBarrier(std::vector<detail::EventImplPtr> EventsWaitWithBarrier,
            CG::StorageInitHelper CGData, CGTYPE Type,
            detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
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
              size_t Width, size_t Height, CG::StorageInitHelper CGData,
              detail::code_location loc = {})
      : CG(Copy2DUSM, std::move(CGData), std::move(loc)), MSrc(Src), MDst(Dst),
        MSrcPitch(SrcPitch), MDstPitch(DstPitch), MWidth(Width),
        MHeight(Height) {}

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
              size_t Width, size_t Height, CG::StorageInitHelper CGData,
              detail::code_location loc = {})
      : CG(Fill2DUSM, std::move(CGData), std::move(loc)),
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
                size_t Height, CG::StorageInitHelper CGData,
                detail::code_location loc = {})
      : CG(Memset2DUSM, std::move(CGData), std::move(loc)), MValue(Value),
        MDst(DstPtr), MPitch(Pitch), MWidth(Width), MHeight(Height) {}
  void *getDst() const { return MDst; }
  size_t getPitch() const { return MPitch; }
  size_t getWidth() const { return MWidth; }
  size_t getHeight() const { return MHeight; }
  char getValue() const { return MValue; }
};

/// "ReadWriteHostPipe" command group class.
class CGReadWriteHostPipe : public CG {
  std::string PipeName;
  bool Blocking;
  void *HostPtr;
  size_t TypeSize;
  bool IsReadOp;

public:
  CGReadWriteHostPipe(const std::string &Name, bool Block, void *Ptr,
                      size_t Size, bool Read, CG::StorageInitHelper CGData,
                      detail::code_location loc = {})
      : CG(ReadWriteHostPipe, std::move(CGData), std::move(loc)),
        PipeName(Name), Blocking(Block), HostPtr(Ptr), TypeSize(Size),
        IsReadOp(Read) {}

  std::string getPipeName() { return PipeName; }
  void *getHostPtr() { return HostPtr; }
  size_t getTypeSize() { return TypeSize; }
  bool isBlocking() { return Blocking; }
  bool isReadHostPipe() { return IsReadOp; }
};

/// "Copy to device_global" command group class.
class CGCopyToDeviceGlobal : public CG {
  void *MSrc;
  void *MDeviceGlobalPtr;
  bool MIsDeviceImageScoped;
  size_t MNumBytes;
  size_t MOffset;

public:
  CGCopyToDeviceGlobal(void *Src, void *DeviceGlobalPtr,
                       bool IsDeviceImageScoped, size_t NumBytes, size_t Offset,
                       CG::StorageInitHelper CGData,
                       detail::code_location loc = {})
      : CG(CopyToDeviceGlobal, std::move(CGData), std::move(loc)), MSrc(Src),
        MDeviceGlobalPtr(DeviceGlobalPtr),
        MIsDeviceImageScoped(IsDeviceImageScoped), MNumBytes(NumBytes),
        MOffset(Offset) {}

  void *getSrc() { return MSrc; }
  void *getDeviceGlobalPtr() { return MDeviceGlobalPtr; }
  bool isDeviceImageScoped() { return MIsDeviceImageScoped; }
  size_t getNumBytes() { return MNumBytes; }
  size_t getOffset() { return MOffset; }
};

/// "Copy to device_global" command group class.
class CGCopyFromDeviceGlobal : public CG {
  void *MDeviceGlobalPtr;
  void *MDest;
  bool MIsDeviceImageScoped;
  size_t MNumBytes;
  size_t MOffset;

public:
  CGCopyFromDeviceGlobal(void *DeviceGlobalPtr, void *Dest,
                         bool IsDeviceImageScoped, size_t NumBytes,
                         size_t Offset, CG::StorageInitHelper CGData,
                         detail::code_location loc = {})
      : CG(CopyFromDeviceGlobal, std::move(CGData), std::move(loc)),
        MDeviceGlobalPtr(DeviceGlobalPtr), MDest(Dest),
        MIsDeviceImageScoped(IsDeviceImageScoped), MNumBytes(NumBytes),
        MOffset(Offset) {}

  void *getDeviceGlobalPtr() { return MDeviceGlobalPtr; }
  void *getDest() { return MDest; }
  bool isDeviceImageScoped() { return MIsDeviceImageScoped; }
  size_t getNumBytes() { return MNumBytes; }
  size_t getOffset() { return MOffset; }
};
/// "Copy Image" command group class.
class CGCopyImage : public CG {
  void *MSrc;
  void *MDst;
  sycl::detail::pi::PiMemImageDesc MImageDesc;
  sycl::detail::pi::PiMemImageFormat MImageFormat;
  sycl::detail::pi::PiImageCopyFlags MImageCopyFlags;
  sycl::detail::pi::PiImageOffset MSrcOffset;
  sycl::detail::pi::PiImageOffset MDstOffset;
  sycl::detail::pi::PiImageRegion MHostExtent;
  sycl::detail::pi::PiImageRegion MCopyExtent;

public:
  CGCopyImage(void *Src, void *Dst, sycl::detail::pi::PiMemImageDesc ImageDesc,
              sycl::detail::pi::PiMemImageFormat ImageFormat,
              sycl::detail::pi::PiImageCopyFlags ImageCopyFlags,
              sycl::detail::pi::PiImageOffset SrcOffset,
              sycl::detail::pi::PiImageOffset DstOffset,
              sycl::detail::pi::PiImageRegion HostExtent,
              sycl::detail::pi::PiImageRegion CopyExtent,
              CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(CopyImage, std::move(CGData), std::move(loc)), MSrc(Src), MDst(Dst),
        MImageDesc(ImageDesc), MImageFormat(ImageFormat),
        MImageCopyFlags(ImageCopyFlags), MSrcOffset(SrcOffset),
        MDstOffset(DstOffset), MHostExtent(HostExtent),
        MCopyExtent(CopyExtent) {}

  void *getSrc() const { return MSrc; }
  void *getDst() const { return MDst; }
  sycl::detail::pi::PiMemImageDesc getDesc() const { return MImageDesc; }
  sycl::detail::pi::PiMemImageFormat getFormat() const { return MImageFormat; }
  sycl::detail::pi::PiImageCopyFlags getCopyFlags() const {
    return MImageCopyFlags;
  }
  sycl::detail::pi::PiImageOffset getSrcOffset() const { return MSrcOffset; }
  sycl::detail::pi::PiImageOffset getDstOffset() const { return MDstOffset; }
  sycl::detail::pi::PiImageRegion getHostExtent() const { return MHostExtent; }
  sycl::detail::pi::PiImageRegion getCopyExtent() const { return MCopyExtent; }
};

/// "Semaphore Wait" command group class.
class CGSemaphoreWait : public CG {
  sycl::detail::pi::PiInteropSemaphoreHandle MInteropSemaphoreHandle;

public:
  CGSemaphoreWait(
      sycl::detail::pi::PiInteropSemaphoreHandle InteropSemaphoreHandle,
      CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(SemaphoreWait, std::move(CGData), std::move(loc)),
        MInteropSemaphoreHandle(InteropSemaphoreHandle) {}

  sycl::detail::pi::PiInteropSemaphoreHandle getInteropSemaphoreHandle() const {
    return MInteropSemaphoreHandle;
  }
};

/// "Semaphore Signal" command group class.
class CGSemaphoreSignal : public CG {
  sycl::detail::pi::PiInteropSemaphoreHandle MInteropSemaphoreHandle;

public:
  CGSemaphoreSignal(
      sycl::detail::pi::PiInteropSemaphoreHandle InteropSemaphoreHandle,
      CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(SemaphoreSignal, std::move(CGData), std::move(loc)),
        MInteropSemaphoreHandle(InteropSemaphoreHandle) {}

  sycl::detail::pi::PiInteropSemaphoreHandle getInteropSemaphoreHandle() const {
    return MInteropSemaphoreHandle;
  }
};

/// "Execute command-buffer" command group class.
class CGExecCommandBuffer : public CG {
public:
  sycl::detail::pi::PiExtCommandBuffer MCommandBuffer;
  std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
      MExecGraph;

  CGExecCommandBuffer(
      const sycl::detail::pi::PiExtCommandBuffer &CommandBuffer,
      const std::shared_ptr<
          sycl::ext::oneapi::experimental::detail::exec_graph_impl> &ExecGraph,
      CG::StorageInitHelper CGData)
      : CG(CGTYPE::ExecCommandBuffer, std::move(CGData)),
        MCommandBuffer(CommandBuffer), MExecGraph(ExecGraph) {}
};

} // namespace detail
} // namespace _V1
} // namespace sycl
