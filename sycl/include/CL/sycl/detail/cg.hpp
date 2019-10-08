//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
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

namespace cl {
namespace sycl {
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

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims>
class HostKernel : public HostKernelBase {
  using IDBuilder = sycl::detail::Builder;
  KernelType MKernel;

public:
  HostKernel(KernelType Kernel) : MKernel(Kernel) {}
  void call(const NDRDescT &NDRDesc, HostProfilingInfo *HPI) override {
    // adjust ND range for serial host:
    NDRDescT AdjustedRange;
    bool Adjust = false;

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
      Adjust = true;
    }
    const NDRDescT &R = Adjust ? AdjustedRange : NDRDesc;
    if (HPI)
      HPI->start();
    runOnHost(R);
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
    size_t XYZ[3] = {0};
    sycl::id<Dims> ID;
    for (; XYZ[2] < NDRDesc.GlobalSize[2]; ++XYZ[2]) {
      XYZ[1] = 0;
      for (; XYZ[1] < NDRDesc.GlobalSize[1]; ++XYZ[1]) {
        XYZ[0] = 0;
        for (; XYZ[0] < NDRDesc.GlobalSize[0]; ++XYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            ID[I] = XYZ[I];
          MKernel(ID);
        }
      }
    }
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, item<Dims, /*Offset=*/false>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    size_t XYZ[3] = {0};
    sycl::id<Dims> ID;
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    for (; XYZ[2] < NDRDesc.GlobalSize[2]; ++XYZ[2]) {
      XYZ[1] = 0;
      for (; XYZ[1] < NDRDesc.GlobalSize[1]; ++XYZ[1]) {
        XYZ[0] = 0;
        for (; XYZ[0] < NDRDesc.GlobalSize[0]; ++XYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            ID[I] = XYZ[I];

          sycl::item<Dims, /*Offset=*/false> Item =
              IDBuilder::createItem<Dims, false>(Range, ID);
          MKernel(Item);
        }
      }
    }
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
    size_t XYZ[3] = {0};
    sycl::id<Dims> ID;
    for (; XYZ[2] < NDRDesc.GlobalSize[2]; ++XYZ[2]) {
      XYZ[1] = 0;
      for (; XYZ[1] < NDRDesc.GlobalSize[1]; ++XYZ[1]) {
        XYZ[0] = 0;
        for (; XYZ[0] < NDRDesc.GlobalSize[0]; ++XYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            ID[I] = XYZ[I] + Offset[I];

          sycl::item<Dims, /*Offset=*/true> Item =
              IDBuilder::createItem<Dims, true>(Range, ID, Offset);
          MKernel(Item);
        }
      }
    }
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, nd_item<Dims>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> GroupSize(
        InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::runtime_error("Invalid local size for global size");
      GroupSize[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(
        InitializedVal<Dims, range>::template get<0>());
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
        throw sycl::runtime_error("Invalid local size for global size");
      NGroups[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(
      InitializedVal<Dims, range>::template get<0>());
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
// The base class for all types of command groups.
class CG {
public:
  // Type of the command group.
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
    PREFETCH_USM
  };

  CG(CGTYPE Type, std::vector<std::vector<char>> ArgsStorage,
     std::vector<detail::AccessorImplPtr> AccStorage,
     std::vector<std::shared_ptr<const void>> SharedPtrStorage,
     std::vector<Requirement *> Requirements,
     std::vector<detail::EventImplPtr> Events)
      : MType(Type), MArgsStorage(std::move(ArgsStorage)),
        MAccStorage(std::move(AccStorage)),
        MSharedPtrStorage(std::move(SharedPtrStorage)),
        MRequirements(std::move(Requirements)), MEvents(std::move(Events)) {}

  CG(CG &&CommandGroup) = default;

  CGTYPE getType() { return MType; }

  virtual ~CG() = default;

private:
  CGTYPE MType;
  // The following storages needed to ensure that arguments won't die while
  // we are using them.
  // Storage for standard layout arguments.
  std::vector<std::vector<char>> MArgsStorage;
  // Storage for accessors.
  std::vector<detail::AccessorImplPtr> MAccStorage;
  // Storage for shared_ptrs.
  std::vector<std::shared_ptr<const void>> MSharedPtrStorage;

public:
  // List of requirements that specify which memory is needed for the command
  // group to be executed.
  std::vector<Requirement *> MRequirements;
  // List of events that order the execution of this CG
  std::vector<detail::EventImplPtr> MEvents;
};

// The class which represents "execute kernel" command group.
class CGExecKernel : public CG {
public:
  NDRDescT MNDRDesc;
  std::unique_ptr<HostKernelBase> MHostKernel;
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  std::vector<ArgDesc> MArgs;
  std::string MKernelName;
  detail::OSModuleHandle MOSModuleHandle;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreams;

  CGExecKernel(NDRDescT NDRDesc, std::unique_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements,
               std::vector<detail::EventImplPtr> Events,
               std::vector<ArgDesc> Args, std::string KernelName,
               detail::OSModuleHandle OSModuleHandle,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams,
               CGTYPE Type)
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MOSModuleHandle(OSModuleHandle),
        MStreams(std::move(Streams)) {
    assert((getType() == RUN_ON_HOST_INTEL || getType() == KERNEL) &&
           "Wrong type of exec kernel CG.");

    if (MNDRDesc.LocalSize.size() > 0) {
      range<3> Excess = (MNDRDesc.GlobalSize % MNDRDesc.LocalSize);
      for (int I = 0; I < 3; I++) {
        if (Excess[I] != 0)
          throw nd_range_error("Global size is not a multiple of local size",
              CL_INVALID_WORK_GROUP_SIZE);
      }
    }
  }

  std::vector<ArgDesc> getArguments() const { return MArgs; }
  std::string getKernelName() const { return MKernelName; }
  std::vector<std::shared_ptr<detail::stream_impl>> getStreams() const {
    return MStreams;
  }
};

// The class which represents "copy" command group.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements,
         std::vector<detail::EventImplPtr> Events)
      : CG(CopyType, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MSrc(Src), MDst(Dst) {}
  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
};

// The class which represents "fill" command group.
class CGFill : public CG {
public:
  std::vector<char> MPattern;
  Requirement *MPtr;

  CGFill(std::vector<char> Pattern, void *Ptr,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements,
         std::vector<detail::EventImplPtr> Events)
      : CG(FILL, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MPattern(std::move(Pattern)), MPtr((Requirement *)Ptr) {}
  Requirement *getReqToFill() { return MPtr; }
};

// The class which represents "update host" command group.
class CGUpdateHost : public CG {
  Requirement *MPtr;

public:
  CGUpdateHost(void *Ptr, std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements,
               std::vector<detail::EventImplPtr> Events)
      : CG(UPDATE_HOST, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MPtr((Requirement *)Ptr) {}

  Requirement *getReqToUpdate() { return MPtr; }
};

// The class which represents "copy" command group for USM pointers.
class CGCopyUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MLength;

public:
  CGCopyUSM(void *Src, void *Dst, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events)
      : CG(COPY_USM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MSrc(Src), MDst(Dst), MLength(Length) {}

  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

// The class which represents "fill" command group for USM pointers.
class CGFillUSM : public CG {
  std::vector<char> MPattern;
  void *MDst;
  size_t MLength;

public:
  CGFillUSM(std::vector<char> Pattern, void *DstPtr, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events)
      : CG(FILL_USM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  int getFill() { return MPattern[0]; }
};

// The class which represents "prefetch" command group for USM pointers.
class CGPrefetchUSM : public CG {
  void *MDst;
  size_t MLength;

public:
  CGPrefetchUSM(void *DstPtr, size_t Length,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<Requirement *> Requirements,
                std::vector<detail::EventImplPtr> Events)
      : CG(PREFETCH_USM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events)),
        MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

} // namespace detail
} // namespace sycl
} // namespace cl
