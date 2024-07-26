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

// The structure represents kernel argument.
class ArgDesc {
public:
  ArgDesc(sycl::detail::kernel_param_kind_t Type, void *Ptr, int Size,
          int Index)
      : MType(Type), MPtr(Ptr), MSize(Size), MIndex(Index) {}

  sycl::detail::kernel_param_kind_t MType;
  void *MPtr;
  int MSize;
  int MIndex;
};

// The structure represents NDRange - global, local sizes, global offset and
// number of dimensions.
class NDRDescT {
  // The method initializes all sizes for dimensions greater than the passed one
  // to the default values, so they will not affect execution.
  void setNDRangeLeftover() {
    for (int I = Dims; I < 3; ++I) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = 0;
    }
  }

  template <int Dims> static sycl::range<3> padRange(sycl::range<Dims> Range) {
    if constexpr (Dims == 3) {
      return Range;
    } else {
      sycl::range<3> Res{0, 0, 0};
      for (int I = 0; I < Dims; ++I)
        Res[I] = Range[I];
      return Res;
    }
  }

  template <int Dims> static sycl::id<3> padId(sycl::id<Dims> Id) {
    if constexpr (Dims == 3) {
      return Id;
    } else {
      sycl::id<3> Res{0, 0, 0};
      for (int I = 0; I < Dims; ++I)
        Res[I] = Id[I];
      return Res;
    }
  }

public:
  NDRDescT() = default;
  NDRDescT(const NDRDescT &Desc) = default;
  NDRDescT(NDRDescT &&Desc) = default;

  NDRDescT(sycl::range<3> N, bool SetNumWorkGroups, int DimsArg)
      : GlobalSize{SetNumWorkGroups ? sycl::range<3>{0, 0, 0} : N},
        NumWorkGroups{SetNumWorkGroups ? N : sycl::range<3>{0, 0, 0}},
        Dims{size_t(DimsArg)} {
    setNDRangeLeftover();
  }

  NDRDescT(sycl::range<3> NumWorkItems, sycl::id<3> Offset, int DimsArg)
      : GlobalSize{NumWorkItems}, GlobalOffset{Offset}, Dims{size_t(DimsArg)} {}

  NDRDescT(sycl::range<3> NumWorkItems, sycl::range<3> LocalSize,
           sycl::id<3> Offset, int DimsArg)
      : GlobalSize{NumWorkItems}, LocalSize{LocalSize}, GlobalOffset{Offset},
        Dims{size_t(DimsArg)} {
    setNDRangeLeftover();
  }

  template <int Dims_>
  NDRDescT(sycl::nd_range<Dims_> ExecutionRange, int DimsArg)
      : NDRDescT(padRange(ExecutionRange.get_global_range()),
                 padRange(ExecutionRange.get_local_range()),
                 padId(ExecutionRange.get_offset()), size_t(DimsArg)) {
    setNDRangeLeftover();
  }

  template <int Dims_>
  NDRDescT(sycl::nd_range<Dims_> ExecutionRange)
      : NDRDescT(ExecutionRange, Dims_) {}

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> Range)
      : NDRDescT(padRange(Range), /*SetNumWorkGroups=*/false, Dims_) {}

  void setClusterDimensions(sycl::range<3> N, int Dims) {
    if (this->Dims != size_t(Dims)) {
      throw std::runtime_error(
          "Dimensionality of cluster, global and local ranges must be same");
    }

    for (int I = 0; I < 3; ++I)
      ClusterDimensions[I] = (I < Dims) ? N[I] : 1;
  }

  NDRDescT &operator=(const NDRDescT &Desc) = default;
  NDRDescT &operator=(NDRDescT &&Desc) = default;

  sycl::range<3> GlobalSize{0, 0, 0};
  sycl::range<3> LocalSize{0, 0, 0};
  sycl::id<3> GlobalOffset{0, 0, 0};
  /// Number of workgroups, used to record the number of workgroups from the
  /// simplest form of parallel_for_work_group. If set, all other fields must be
  /// zero
  sycl::range<3> NumWorkGroups{0, 0, 0};
  sycl::range<3> ClusterDimensions{1, 1, 1};
  size_t Dims = 0;
};

/// Base class for all types of command groups.
class CG {
public:
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

  CG(CGType Type, StorageInitHelper D, detail::code_location loc = {})
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

  CGType getType() const { return MType; }

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
  virtual void clearAuxiliaryResources() {};

  virtual ~CG() = default;

private:
  CGType MType;
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
  bool MKernelUsesClusterLaunch = false;

  CGExecKernel(NDRDescT NDRDesc, std::shared_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::shared_ptr<detail::kernel_bundle_impl> KernelBundle,
               CG::StorageInitHelper CGData, std::vector<ArgDesc> Args,
               std::string KernelName,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams,
               std::vector<std::shared_ptr<const void>> AuxiliaryResources,
               CGType Type,
               sycl::detail::pi::PiKernelCacheConfig KernelCacheConfig,
               bool KernelIsCooperative, bool MKernelUsesClusterLaunch,
               detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)),
        MKernelBundle(std::move(KernelBundle)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MStreams(std::move(Streams)),
        MAuxiliaryResources(std::move(AuxiliaryResources)),
        MKernelCacheConfig(std::move(KernelCacheConfig)),
        MKernelIsCooperative(KernelIsCooperative),
        MKernelUsesClusterLaunch(MKernelUsesClusterLaunch) {
    assert(getType() == CGType::Kernel && "Wrong type of exec kernel CG.");
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
  CGCopy(CGType CopyType, void *Src, void *Dst, CG::StorageInitHelper CGData,
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
  std::vector<unsigned char> MPattern;
  AccessorImplHost *MPtr;

  CGFill(std::vector<unsigned char> Pattern, void *Ptr,
         CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(CGType::Fill, std::move(CGData), std::move(loc)),
        MPattern(std::move(Pattern)), MPtr((AccessorImplHost *)Ptr) {}
  AccessorImplHost *getReqToFill() { return MPtr; }
};

/// "Update host" command group class.
class CGUpdateHost : public CG {
  AccessorImplHost *MPtr;

public:
  CGUpdateHost(void *Ptr, CG::StorageInitHelper CGData,
               detail::code_location loc = {})
      : CG(CGType::UpdateHost, std::move(CGData), std::move(loc)),
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
      : CG(CGType::CopyUSM, std::move(CGData), std::move(loc)), MSrc(Src),
        MDst(Dst), MLength(Length) {}

  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Fill USM" command group class.
class CGFillUSM : public CG {
  std::vector<unsigned char> MPattern;
  void *MDst;
  size_t MLength;

public:
  CGFillUSM(std::vector<unsigned char> Pattern, void *DstPtr, size_t Length,
            CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(CGType::FillUSM, std::move(CGData), std::move(loc)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  const std::vector<unsigned char> &getPattern() { return MPattern; }
};

/// "Prefetch USM" command group class.
class CGPrefetchUSM : public CG {
  void *MDst;
  size_t MLength;

public:
  CGPrefetchUSM(void *DstPtr, size_t Length, CG::StorageInitHelper CGData,
                detail::code_location loc = {})
      : CG(CGType::PrefetchUSM, std::move(CGData), std::move(loc)),
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
              CG::StorageInitHelper CGData, CGType Type,
              detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)), MDst(DstPtr),
        MLength(Length), MAdvice(Advice) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  pi_mem_advice getAdvice() { return MAdvice; }
};

class CGBarrier : public CG {
public:
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  CGBarrier(std::vector<detail::EventImplPtr> EventsWaitWithBarrier,
            CG::StorageInitHelper CGData, CGType Type,
            detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MEventsWaitWithBarrier(std::move(EventsWaitWithBarrier)) {}
};

class CGProfilingTag : public CG {
public:
  CGProfilingTag(CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(CGType::ProfilingTag, std::move(CGData), std::move(loc)) {}
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
      : CG(CGType::Copy2DUSM, std::move(CGData), std::move(loc)), MSrc(Src),
        MDst(Dst), MSrcPitch(SrcPitch), MDstPitch(DstPitch), MWidth(Width),
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
  std::vector<unsigned char> MPattern;
  void *MDst;
  size_t MPitch;
  size_t MWidth;
  size_t MHeight;

public:
  CGFill2DUSM(std::vector<unsigned char> Pattern, void *DstPtr, size_t Pitch,
              size_t Width, size_t Height, CG::StorageInitHelper CGData,
              detail::code_location loc = {})
      : CG(CGType::Fill2DUSM, std::move(CGData), std::move(loc)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MPitch(Pitch),
        MWidth(Width), MHeight(Height) {}
  void *getDst() const { return MDst; }
  size_t getPitch() const { return MPitch; }
  size_t getWidth() const { return MWidth; }
  size_t getHeight() const { return MHeight; }
  const std::vector<unsigned char> &getPattern() const { return MPattern; }
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
      : CG(CGType::Memset2DUSM, std::move(CGData), std::move(loc)),
        MValue(Value), MDst(DstPtr), MPitch(Pitch), MWidth(Width),
        MHeight(Height) {}
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
      : CG(CGType::ReadWriteHostPipe, std::move(CGData), std::move(loc)),
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
      : CG(CGType::CopyToDeviceGlobal, std::move(CGData), std::move(loc)),
        MSrc(Src), MDeviceGlobalPtr(DeviceGlobalPtr),
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
      : CG(CGType::CopyFromDeviceGlobal, std::move(CGData), std::move(loc)),
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
  sycl::detail::pi::PiMemImageDesc MSrcImageDesc;
  sycl::detail::pi::PiMemImageDesc MDstImageDesc;
  sycl::detail::pi::PiMemImageFormat MSrcImageFormat;
  sycl::detail::pi::PiMemImageFormat MDstImageFormat;
  sycl::detail::pi::PiImageCopyFlags MImageCopyFlags;
  sycl::detail::pi::PiImageOffset MSrcOffset;
  sycl::detail::pi::PiImageOffset MDstOffset;
  sycl::detail::pi::PiImageRegion MCopyExtent;

public:
  CGCopyImage(void *Src, void *Dst,
              sycl::detail::pi::PiMemImageDesc SrcImageDesc,
              sycl::detail::pi::PiMemImageDesc DstImageDesc,
              sycl::detail::pi::PiMemImageFormat SrcImageFormat,
              sycl::detail::pi::PiMemImageFormat DstImageFormat,
              sycl::detail::pi::PiImageCopyFlags ImageCopyFlags,
              sycl::detail::pi::PiImageOffset SrcOffset,
              sycl::detail::pi::PiImageOffset DstOffset,
              sycl::detail::pi::PiImageRegion CopyExtent,
              CG::StorageInitHelper CGData, detail::code_location loc = {})
      : CG(CGType::CopyImage, std::move(CGData), std::move(loc)), MSrc(Src),
        MDst(Dst), MSrcImageDesc(SrcImageDesc), MDstImageDesc(DstImageDesc),
        MSrcImageFormat(SrcImageFormat), MDstImageFormat(DstImageFormat),
        MImageCopyFlags(ImageCopyFlags), MSrcOffset(SrcOffset),
        MDstOffset(DstOffset), MCopyExtent(CopyExtent) {}

  void *getSrc() const { return MSrc; }
  void *getDst() const { return MDst; }
  sycl::detail::pi::PiMemImageDesc getSrcDesc() const { return MSrcImageDesc; }
  sycl::detail::pi::PiMemImageDesc getDstDesc() const { return MDstImageDesc; }
  sycl::detail::pi::PiMemImageFormat getSrcFormat() const {
    return MSrcImageFormat;
  }
  sycl::detail::pi::PiMemImageFormat getDstFormat() const {
    return MDstImageFormat;
  }
  sycl::detail::pi::PiImageCopyFlags getCopyFlags() const {
    return MImageCopyFlags;
  }
  sycl::detail::pi::PiImageOffset getSrcOffset() const { return MSrcOffset; }
  sycl::detail::pi::PiImageOffset getDstOffset() const { return MDstOffset; }
  sycl::detail::pi::PiImageRegion getCopyExtent() const { return MCopyExtent; }
};

/// "Semaphore Wait" command group class.
class CGSemaphoreWait : public CG {
  sycl::detail::pi::PiInteropSemaphoreHandle MInteropSemaphoreHandle;
  std::optional<uint64_t> MWaitValue;

public:
  CGSemaphoreWait(
      sycl::detail::pi::PiInteropSemaphoreHandle InteropSemaphoreHandle,
      std::optional<uint64_t> WaitValue, CG::StorageInitHelper CGData,
      detail::code_location loc = {})
      : CG(CGType::SemaphoreWait, std::move(CGData), std::move(loc)),
        MInteropSemaphoreHandle(InteropSemaphoreHandle), MWaitValue(WaitValue) {
  }

  sycl::detail::pi::PiInteropSemaphoreHandle getInteropSemaphoreHandle() const {
    return MInteropSemaphoreHandle;
  }
  std::optional<uint64_t> getWaitValue() const { return MWaitValue; }
};

/// "Semaphore Signal" command group class.
class CGSemaphoreSignal : public CG {
  sycl::detail::pi::PiInteropSemaphoreHandle MInteropSemaphoreHandle;
  std::optional<uint64_t> MSignalValue;

public:
  CGSemaphoreSignal(
      sycl::detail::pi::PiInteropSemaphoreHandle InteropSemaphoreHandle,
      std::optional<uint64_t> SignalValue, CG::StorageInitHelper CGData,
      detail::code_location loc = {})
      : CG(CGType::SemaphoreSignal, std::move(CGData), std::move(loc)),
        MInteropSemaphoreHandle(InteropSemaphoreHandle),
        MSignalValue(SignalValue) {}

  sycl::detail::pi::PiInteropSemaphoreHandle getInteropSemaphoreHandle() const {
    return MInteropSemaphoreHandle;
  }
  std::optional<uint64_t> getSignalValue() const { return MSignalValue; }
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
      : CG(CGType::ExecCommandBuffer, std::move(CGData)),
        MCommandBuffer(CommandBuffer), MExecGraph(ExecGraph) {}
};

class CGHostTask : public CG {
public:
  std::shared_ptr<HostTask> MHostTask;
  // queue for host-interop task
  std::shared_ptr<detail::queue_impl> MQueue;
  // context for host-interop task
  std::shared_ptr<detail::context_impl> MContext;
  std::vector<ArgDesc> MArgs;

  CGHostTask(std::shared_ptr<HostTask> HostTask,
             std::shared_ptr<detail::queue_impl> Queue,
             std::shared_ptr<detail::context_impl> Context,
             std::vector<ArgDesc> Args, CG::StorageInitHelper CGData,
             CGType Type, detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MHostTask(std::move(HostTask)), MQueue(Queue), MContext(Context),
        MArgs(std::move(Args)) {}
};

} // namespace detail
} // namespace _V1
} // namespace sycl
