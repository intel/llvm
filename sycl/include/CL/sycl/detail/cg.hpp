//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/cg_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/interop_handle.hpp>
#include <CL/sycl/interop_handler.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class queue;
namespace detail {
class queue_impl;
} // namespace detail

namespace detail {

class stream_impl;
/// Base class for all types of command groups.
class CG {
public:
  /// Type of the command group.
  enum CGTYPE {
    NONE = 0,
    KERNEL = 1,
    COPY_ACC_TO_PTR = 2,
    COPY_PTR_TO_ACC = 3,
    COPY_ACC_TO_ACC = 4,
    BARRIER = 5,
    BARRIER_WAITLIST = 6,
    FILL = 7,
    UPDATE_HOST = 8,
    RUN_ON_HOST_INTEL = 9,
    COPY_USM = 10,
    FILL_USM = 11,
    PREFETCH_USM = 12,
    CODEPLAY_INTEROP_TASK = 13,
    CODEPLAY_HOST_TASK = 14
  };

  CG(CGTYPE Type, vector_class<vector_class<char>> ArgsStorage,
     vector_class<detail::AccessorImplPtr> AccStorage,
     vector_class<shared_ptr_class<const void>> SharedPtrStorage,
     vector_class<Requirement *> Requirements,
     vector_class<detail::EventImplPtr> Events, detail::code_location loc = {})
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

  virtual ~CG() = default;

private:
  CGTYPE MType;
  // The following storages are needed to ensure that arguments won't die while
  // we are using them.
  /// Storage for standard layout arguments.
  vector_class<vector_class<char>> MArgsStorage;
  /// Storage for accessors.
  vector_class<detail::AccessorImplPtr> MAccStorage;
  /// Storage for shared_ptrs.
  vector_class<shared_ptr_class<const void>> MSharedPtrStorage;

public:
  /// List of requirements that specify which memory is needed for the command
  /// group to be executed.
  vector_class<Requirement *> MRequirements;
  /// List of events that order the execution of this CG
  vector_class<detail::EventImplPtr> MEvents;
  // Member variables to capture the user code-location
  // information from Q.submit(), Q.parallel_for() etc
  // Storage for function name and source file name
  string_class MFunctionName, MFileName;
  // Storage for line and column of code location
  int32_t MLine, MColumn;
};

/// "Execute kernel" command group class.
class CGExecKernel : public CG {
public:
  /// Stores ND-range description.
  NDRDescT MNDRDesc;
  unique_ptr_class<HostKernelBase> MHostKernel;
  shared_ptr_class<detail::kernel_impl> MSyclKernel;
  vector_class<ArgDesc> MArgs;
  string_class MKernelName;
  detail::OSModuleHandle MOSModuleHandle;
  vector_class<shared_ptr_class<detail::stream_impl>> MStreams;

  CGExecKernel(NDRDescT NDRDesc, unique_ptr_class<HostKernelBase> HKernel,
               shared_ptr_class<detail::kernel_impl> SyclKernel,
               vector_class<vector_class<char>> ArgsStorage,
               vector_class<detail::AccessorImplPtr> AccStorage,
               vector_class<shared_ptr_class<const void>> SharedPtrStorage,
               vector_class<Requirement *> Requirements,
               vector_class<detail::EventImplPtr> Events,
               vector_class<ArgDesc> Args, string_class KernelName,
               detail::OSModuleHandle OSModuleHandle,
               vector_class<shared_ptr_class<detail::stream_impl>> Streams,
               CGTYPE Type, detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MOSModuleHandle(OSModuleHandle),
        MStreams(std::move(Streams)) {
    assert((getType() == RUN_ON_HOST_INTEL || getType() == KERNEL) &&
           "Wrong type of exec kernel CG.");
  }

  vector_class<ArgDesc> getArguments() const { return MArgs; }
  string_class getKernelName() const { return MKernelName; }
  vector_class<shared_ptr_class<detail::stream_impl>> getStreams() const {
    return MStreams;
  }
  void clearStreams() { MStreams.clear(); }
};

/// "Copy memory" command group class.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst,
         vector_class<vector_class<char>> ArgsStorage,
         vector_class<detail::AccessorImplPtr> AccStorage,
         vector_class<shared_ptr_class<const void>> SharedPtrStorage,
         vector_class<Requirement *> Requirements,
         vector_class<detail::EventImplPtr> Events,
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
  vector_class<char> MPattern;
  Requirement *MPtr;

  CGFill(vector_class<char> Pattern, void *Ptr,
         vector_class<vector_class<char>> ArgsStorage,
         vector_class<detail::AccessorImplPtr> AccStorage,
         vector_class<shared_ptr_class<const void>> SharedPtrStorage,
         vector_class<Requirement *> Requirements,
         vector_class<detail::EventImplPtr> Events,
         detail::code_location loc = {})
      : CG(FILL, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MPtr((Requirement *)Ptr) {}
  Requirement *getReqToFill() { return MPtr; }
};

/// "Update host" command group class.
class CGUpdateHost : public CG {
  Requirement *MPtr;

public:
  CGUpdateHost(void *Ptr, vector_class<vector_class<char>> ArgsStorage,
               vector_class<detail::AccessorImplPtr> AccStorage,
               vector_class<shared_ptr_class<const void>> SharedPtrStorage,
               vector_class<Requirement *> Requirements,
               vector_class<detail::EventImplPtr> Events,
               detail::code_location loc = {})
      : CG(UPDATE_HOST, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPtr((Requirement *)Ptr) {}

  Requirement *getReqToUpdate() { return MPtr; }
};

/// "Copy USM" command group class.
class CGCopyUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MLength;

public:
  CGCopyUSM(void *Src, void *Dst, size_t Length,
            vector_class<vector_class<char>> ArgsStorage,
            vector_class<detail::AccessorImplPtr> AccStorage,
            vector_class<shared_ptr_class<const void>> SharedPtrStorage,
            vector_class<Requirement *> Requirements,
            vector_class<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(COPY_USM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst), MLength(Length) {}

  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Fill USM" command group class.
class CGFillUSM : public CG {
  vector_class<char> MPattern;
  void *MDst;
  size_t MLength;

public:
  CGFillUSM(vector_class<char> Pattern, void *DstPtr, size_t Length,
            vector_class<vector_class<char>> ArgsStorage,
            vector_class<detail::AccessorImplPtr> AccStorage,
            vector_class<shared_ptr_class<const void>> SharedPtrStorage,
            vector_class<Requirement *> Requirements,
            vector_class<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(FILL_USM, std::move(ArgsStorage), std::move(AccStorage),
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
                vector_class<vector_class<char>> ArgsStorage,
                vector_class<detail::AccessorImplPtr> AccStorage,
                vector_class<shared_ptr_class<const void>> SharedPtrStorage,
                vector_class<Requirement *> Requirements,
                vector_class<detail::EventImplPtr> Events,
                detail::code_location loc = {})
      : CG(PREFETCH_USM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

class CGInteropTask : public CG {
public:
  std::unique_ptr<InteropTask> MInteropTask;

  CGInteropTask(std::unique_ptr<InteropTask> InteropTask,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<Requirement *> Requirements,
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
  shared_ptr_class<detail::queue_impl> MQueue;
  // context for host-interop task
  shared_ptr_class<detail::context_impl> MContext;
  vector_class<ArgDesc> MArgs;

  CGHostTask(std::unique_ptr<HostTask> HostTask,
             std::shared_ptr<detail::queue_impl> Queue,
             std::shared_ptr<detail::context_impl> Context,
             vector_class<ArgDesc> Args,
             std::vector<std::vector<char>> ArgsStorage,
             std::vector<detail::AccessorImplPtr> AccStorage,
             std::vector<std::shared_ptr<const void>> SharedPtrStorage,
             std::vector<Requirement *> Requirements,
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
  vector_class<detail::EventImplPtr> MEventsWaitWithBarrier;

  CGBarrier(vector_class<detail::EventImplPtr> EventsWaitWithBarrier,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events, CGTYPE Type,
            detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MEventsWaitWithBarrier(std::move(EventsWaitWithBarrier)) {}
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
