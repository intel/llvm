//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
class queue;
namespace detail {
class queue_impl;
} // namespace detail

// Interoperability handler
//
class interop_handler {
  // Make accessor class friend to access the detail mem objects
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  friend class accessor;

public:
  using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
  using ReqToMem = std::pair<detail::Requirement *, pi_mem>;

  interop_handler(std::vector<ReqToMem> MemObjs, QueueImplPtr Queue)
      : MQueue(std::move(Queue)), MMemObjs(std::move(MemObjs)) {}

  template <backend BackendName = backend::opencl>
  auto get_queue() const -> typename interop<BackendName, queue>::type {
    return reinterpret_cast<typename interop<BackendName, queue>::type>(
        GetNativeQueue());
  }

  template <backend BackendName = backend::opencl, typename DataT, int Dims,
            access::mode AccessMode, access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  auto get_mem(accessor<DataT, Dims, AccessMode, AccessTarget,
                        access::placeholder::false_t>
                   Acc) const ->
      typename interop<BackendName,
                       accessor<DataT, Dims, AccessMode, AccessTarget,
                                access::placeholder::false_t>>::type {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    return getMemImpl<BackendName, DataT, Dims, AccessMode, AccessTarget,
                      access::placeholder::false_t>(
        detail::getSyclObjImpl(*AccBase).get());
  }

private:
  QueueImplPtr MQueue;
  std::vector<ReqToMem> MMemObjs;

  template <backend BackendName, typename DataT, int Dims,
            access::mode AccessMode, access::target AccessTarget,
            access::placeholder IsPlaceholder>
  auto getMemImpl(detail::Requirement *Req) const -> typename interop<
      BackendName,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder>>::type {
    return (typename interop<BackendName,
                             accessor<DataT, Dims, AccessMode, AccessTarget,
                                      IsPlaceholder>>::type)GetNativeMem(Req);
  }

  __SYCL_EXPORT pi_native_handle GetNativeMem(detail::Requirement *Req) const;
  __SYCL_EXPORT pi_native_handle GetNativeQueue() const;
};

namespace detail {

using namespace cl;

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
  void setNDRangeLeftover(int Dims_) {
    for (int I = Dims_; I < 3; ++I) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = 0;
    }
  }

public:
  NDRDescT()
      : GlobalSize{0, 0, 0}, LocalSize{0, 0, 0}, NumWorkGroups{0, 0, 0} {}

  template <int Dims_> void set(sycl::range<Dims_> NumWorkItems) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  // Initializes this ND range descriptor with given range of work items and
  // offset.
  template <int Dims_>
  void set(sycl::range<Dims_> NumWorkItems, sycl::id<Dims_> Offset) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = 0;
      GlobalOffset[I] = Offset[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  template <int Dims_> void set(sycl::nd_range<Dims_> ExecutionRange) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = ExecutionRange.get_global_range()[I];
      LocalSize[I] = ExecutionRange.get_local_range()[I];
      GlobalOffset[I] = ExecutionRange.get_offset()[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  void set(int Dims_, sycl::nd_range<3> ExecutionRange) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = ExecutionRange.get_global_range()[I];
      LocalSize[I] = ExecutionRange.get_local_range()[I];
      GlobalOffset[I] = ExecutionRange.get_offset()[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  template <int Dims_> void setNumWorkGroups(sycl::range<Dims_> N) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = 0;
      // '0' is a mark to adjust before kernel launch when there is enough info:
      LocalSize[I] = 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = N[I];
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  sycl::range<3> GlobalSize;
  sycl::range<3> LocalSize;
  sycl::id<3> GlobalOffset;
  /// Number of workgroups, used to record the number of workgroups from the
  /// simplest form of parallel_for_work_group. If set, all other fields must be
  /// zero
  sycl::range<3> NumWorkGroups;
  size_t Dims;
};

// The pure virtual class aimed to store lambda/functors of any type.
class HostKernelBase {
public:
  // The method executes lambda stored using NDRange passed.
  virtual void call(const NDRDescT &NDRDesc, HostProfilingInfo *HPI) = 0;
  // Return pointer to the lambda object.
  // Used to extract captured variables.
  virtual char *getPtr() = 0;
  virtual ~HostKernelBase() = default;
};

class InteropTask {
  std::function<void(cl::sycl::interop_handler)> MFunc;

public:
  InteropTask(function_class<void(cl::sycl::interop_handler)> Func)
      : MFunc(Func) {}
  void call(cl::sycl::interop_handler &h) { MFunc(h); }
};

class HostTask {
  std::function<void()> MHostTask;

public:
  HostTask() : MHostTask([]() {}) {}
  HostTask(std::function<void()> &&Func) : MHostTask(Func) {}

  void call() { MHostTask(); }
};

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims>
class HostKernel : public HostKernelBase {
  using IDBuilder = sycl::detail::Builder;
  KernelType MKernel;

public:
  HostKernel(KernelType Kernel) : MKernel(Kernel) {}
  void call(const NDRDescT &NDRDesc, HostProfilingInfo *HPI) override {
    // adjust ND range for serial host:
    NDRDescT AdjustedRange = NDRDesc;

    if (NDRDesc.GlobalSize[0] == 0 && NDRDesc.NumWorkGroups[0] != 0) {
      // This is a special case - NDRange information is not complete, only the
      // desired number of work groups is set by the user. Choose work group
      // size (LocalSize), calculate the missing NDRange characteristics
      // needed to invoke the kernel and adjust the NDRange descriptor
      // accordingly. For some devices the work group size selection requires
      // access to the device's properties, hence such late "adjustment".
      range<3> WGsize{1, 1, 1}; // no better alternative for serial host?
      AdjustedRange.set(NDRDesc.Dims,
                        nd_range<3>(NDRDesc.NumWorkGroups * WGsize, WGsize));
    }
    // If local size for host is not set explicitly, let's adjust it to 1,
    // so nd_range_error for zero local size is not thrown.
    if (AdjustedRange.LocalSize[0] == 0)
      for (int I = 0; I < AdjustedRange.Dims; ++I)
        AdjustedRange.LocalSize[I] = 1;
    if (HPI)
      HPI->start();
    runOnHost(AdjustedRange);
    if (HPI)
      HPI->end();
  }

  char *getPtr() override { return reinterpret_cast<char *>(&MKernel); }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, void>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    MKernel();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<Dims>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    detail::NDLoop<Dims>::iterate(
        Range, [&](const sycl::id<Dims> &ID) { MKernel(ID); });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, item<Dims, /*Offset=*/false>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::id<Dims> ID;
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    detail::NDLoop<Dims>::iterate(Range, [&](const sycl::id<Dims> ID) {
      sycl::item<Dims, /*Offset=*/false> Item =
          IDBuilder::createItem<Dims, false>(Range, ID);
      MKernel(Item);
    });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, item<Dims, /*Offset=*/true>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    sycl::id<Dims> Offset;
    for (int I = 0; I < Dims; ++I) {
      Range[I] = NDRDesc.GlobalSize[I];
      Offset[I] = NDRDesc.GlobalOffset[I];
    }

    detail::NDLoop<Dims>::iterate(Range, [&](const sycl::id<Dims> &ID) {
      sycl::id<Dims> OffsetID = ID + Offset;
      sycl::item<Dims, /*Offset=*/true> Item =
          IDBuilder::createItem<Dims, true>(Range, OffsetID, Offset);
      MKernel(Item);
    });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, nd_item<Dims>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> GroupSize(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_INVALID_WORK_GROUP_SIZE);
      GroupSize[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(InitializedVal<Dims, range>::template get<0>());
    sycl::range<Dims> GlobalSize(
        InitializedVal<Dims, range>::template get<0>());
    sycl::id<Dims> GlobalOffset;
    for (int I = 0; I < Dims; ++I) {
      GlobalOffset[I] = NDRDesc.GlobalOffset[I];
      LocalSize[I] = NDRDesc.LocalSize[I];
      GlobalSize[I] = NDRDesc.GlobalSize[I];
    }

    detail::NDLoop<Dims>::iterate(GroupSize, [&](const id<Dims> &GroupID) {
      sycl::group<Dims> Group = IDBuilder::createGroup<Dims>(
          GlobalSize, LocalSize, GroupSize, GroupID);

      detail::NDLoop<Dims>::iterate(LocalSize, [&](const id<Dims> &LocalID) {
        id<Dims> GlobalID = GroupID * LocalSize + LocalID + GlobalOffset;
        const sycl::item<Dims, /*Offset=*/true> GlobalItem =
            IDBuilder::createItem<Dims, true>(GlobalSize, GlobalID,
                                              GlobalOffset);
        const sycl::item<Dims, /*Offset=*/false> LocalItem =
            IDBuilder::createItem<Dims, false>(LocalSize, LocalID);
        const sycl::nd_item<Dims> NDItem =
            IDBuilder::createNDItem<Dims>(GlobalItem, LocalItem, Group);
        MKernel(NDItem);
      });
    });
  }

  template <typename ArgT = KernelArgType>
  enable_if_t<std::is_same<ArgT, cl::sycl::group<Dims>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> NGroups(InitializedVal<Dims, range>::template get<0>());

    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_INVALID_WORK_GROUP_SIZE);
      NGroups[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(InitializedVal<Dims, range>::template get<0>());
    sycl::range<Dims> GlobalSize(
        InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      LocalSize[I] = NDRDesc.LocalSize[I];
      GlobalSize[I] = NDRDesc.GlobalSize[I];
    }
    detail::NDLoop<Dims>::iterate(NGroups, [&](const id<Dims> &GroupID) {
      sycl::group<Dims> Group =
          IDBuilder::createGroup<Dims>(GlobalSize, LocalSize, NGroups, GroupID);
      MKernel(Group);
    });
  }

  ~HostKernel() = default;
};

class stream_impl;
/// Base class for all types of command groups.
class CG {
public:
  /// Type of the command group.
  enum CGTYPE {
    NONE,
    KERNEL,
    COPY_ACC_TO_PTR,
    COPY_PTR_TO_ACC,
    COPY_ACC_TO_ACC,
    FILL,
    UPDATE_HOST,
    RUN_ON_HOST_INTEL,
    COPY_USM,
    FILL_USM,
    PREFETCH_USM,
    CODEPLAY_INTEROP_TASK,
    CODEPLAY_HOST_TASK
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
  vector_class<ArgDesc> MArgs;

  CGHostTask(std::unique_ptr<HostTask> HostTask, vector_class<ArgDesc> Args,
             std::vector<std::vector<char>> ArgsStorage,
             std::vector<detail::AccessorImplPtr> AccStorage,
             std::vector<std::shared_ptr<const void>> SharedPtrStorage,
             std::vector<Requirement *> Requirements,
             std::vector<detail::EventImplPtr> Events, CGTYPE Type,
             detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MHostTask(std::move(HostTask)), MArgs(std::move(Args)) {}
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
