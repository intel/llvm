//==-------- handler.hpp --- SYCL command group handler --------*- C++ -*---==//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive property
// of Intel Corporation and may not be disclosed, examined or reproduced in
// whole or in part without explicit written authorization from the company.
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/sampler.hpp>

#include <CL/sycl/stl.hpp>

#include <functional>
#include <memory>
#include <type_traits>

template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __fill;

template <typename T_Src, typename T_Dst, int Dims,
          cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __copyAcc2Ptr;

template <typename T_Src, typename T_Dst, int Dims,
          cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __copyPtr2Acc;

template <typename T_Src, int Dims_Src, cl::sycl::access::mode AccessMode_Src,
          cl::sycl::access::target AccessTarget_Src, typename T_Dst,
          int Dims_Dst, cl::sycl::access::mode AccessMode_Dst,
          cl::sycl::access::target AccessTarget_Dst,
          cl::sycl::access::placeholder IsPlaceholder_Src,
          cl::sycl::access::placeholder IsPlaceholder_Dst>
class __copyAcc2Acc;

namespace cl {
namespace sycl {

namespace csd = cl::sycl::detail;

// Forward declaration

template <typename T, int Dimensions, typename AllocatorT> class buffer;
namespace detail {

/// This class is the default KernelName template parameter type for kernel
/// invocation APIs such as single_task.
class auto_name {};

class queue_impl;
class stream_impl;
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor;
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

// Non-const version of the above template to match functors whose 'operator()'
// is declared w/o the 'const' qualifier.
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg));

// template <typename RetType, typename Func>
// static void member_ptr_helper(RetType (Func::*)() const);

// template <typename RetType, typename Func>
// static void member_ptr_helper(RetType (Func::*)());

template <typename F>
decltype(member_ptr_helper(&F::operator())) argument_helper(F);

template <typename T>
using lambda_arg_type = decltype(argument_helper(std::declval<T>()));

/// Helper struct to get a kernel name type based on given \c Name and \c Type
/// types: if \c Name is undefined (is a \c auto_name) then \c Type becomes
/// the \c Name.
template <typename Name, typename Type> struct get_kernel_name_t {
  using name = Name;
};

/// Specialization for the case when \c Name is undefined.
template <typename Type> struct get_kernel_name_t<csd::auto_name, Type> {
  using name = Type;
};

} // namespace detail

// Objects of the handler class collect information about command group, such as
// kernel, requirements to the memory, arguments for the kernel.
//
// sycl::queue::submit([](handler &CGH){
//   CGH.require(Accessor1);   // Adds a requirement to the memory object.
//   CGH.setArg(0, Accessor2); // Registers accessor given as an argument to the
//                             // kernel + adds a requirement to the memory
//                             // object.
//   CGH.setArg(1, N);         // Registers value given as an argument to the
//                             // kernel.
//   // The following registers KernelFunctor to be a kernel that will be
//   // executed in case of queue is bound to the host device, SyclKernel - for
//   // an OpenCL device. This function clearly indicates that command group
//   // represents kernel execution.
//   CGH.parallel_for(KernelFunctor, SyclKernel);
//  });
//
// The command group can represent absolutely different operations. Depending
// on the operation we need to store different data. But, in most cases, it's
// impossible to say what kind of operation we need to perform until the very
// end. So, handler class contains all fields simultaneously, then during
// "finalization" it constructs CG object, that represents specific operation,
// passing fields that are required only.

// 4.8.3 Command group handler class
class handler {
  std::shared_ptr<detail::queue_impl> MQueue;
  // The storage for the arguments passed.
  // We need to store a copy of values that are passed explicitly through
  // set_arg, require and so on, because we need them to be alive after
  // we exit the method they are passed in.
  std::vector<std::vector<char>> MArgsStorage;
  std::vector<detail::AccessorImplPtr> MAccStorage;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreamStorage;
  std::vector<std::shared_ptr<const void>> MSharedPtrStorage;
  // The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;
  // The list of associated accessors with this handler.
  // These accessors were created with this handler as argument or
  // have become required for this handler via require method.
  std::vector<detail::ArgDesc> MAssociatedAccesors;
  // The list of requirements to the memory objects for the scheduling.
  std::vector<detail::Requirement *> MRequirements;
  // Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;
  std::string MKernelName;
  // Storage for a sycl::kernel object.
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  // Type of the command group, e.g. kernel, fill.
  detail::CG::CGTYPE MCGType = detail::CG::NONE;
  // Pointer to the source host memory or accessor(depending on command type).
  void *MSrcPtr = nullptr;
  // Pointer to the dest host memory or accessor(depends on command type).
  void *MDstPtr = nullptr;
  // Length to copy or fill (for USM operations).
  size_t MLength = 0;
  // Pattern that is used to fill memory object in case command type is fill.
  std::vector<char> MPattern;
  // Storage for a lambda or function object.
  std::unique_ptr<detail::HostKernelBase> MHostKernel;
  detail::OSModuleHandle MOSModuleHandle;
  // The list of events that order this operation
  std::vector<detail::EventImplPtr> MEvents;

  bool MIsHost = false;

private:
  handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
      : MQueue(std::move(Queue)), MIsHost(IsHost) {}

  // Method stores copy of Arg passed to the MArgsStorage.
  template <typename T, typename F = typename std::remove_const<
                            typename std::remove_reference<T>::type>::type>
  F *storePlainArg(T &&Arg) {
    MArgsStorage.emplace_back(sizeof(T));
    F *Storage = (F *)MArgsStorage.back().data();
    *Storage = Arg;
    return Storage;
  }

  // The method extracts and prepares kernel arguments from the lambda using
  // integration header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs) {
    const bool IsKernelCreatedFromSource = false;
    size_t IndexShift = 0;
    for (size_t I = 0; I < KernelArgsNum; ++I) {
      void *Ptr = LambdaPtr + KernelArgs[I].offset;
      const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
      const int &Size = KernelArgs[I].info;
      if (Kind == detail::kernel_param_kind_t::kind_accessor) {
        // For args kind of accessor Size is information about accessor.
        // The first 11 bits of Size encodes the accessor target.
        const access::target AccTarget =
            static_cast<access::target>(Size & 0x7ff);
        if ((AccTarget == access::target::global_buffer ||
             AccTarget == access::target::constant_buffer) ||
            (AccTarget == access::target::image ||
             AccTarget == access::target::image_array)) {
          detail::AccessorBaseHost *AccBase =
              static_cast<detail::AccessorBaseHost *>(Ptr);
          Ptr = detail::getSyclObjImpl(*AccBase).get();
        }
      }
      processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource);
    }
  }

  // The method extracts and prepares kernel arguments that were set
  // via set_arg(s)
  void extractArgsAndReqs() {
    assert(MSyclKernel && "MSyclKernel is not initialized");
    std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
    MArgs.clear();

    std::sort(UnPreparedArgs.begin(), UnPreparedArgs.end(),
              [](const detail::ArgDesc &first, const detail::ArgDesc &second)
                  -> bool { return (first.MIndex < second.MIndex); });

    const bool IsKernelCreatedFromSource = MSyclKernel->isCreatedFromSource();

    size_t IndexShift = 0;
    for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
      void *Ptr = UnPreparedArgs[I].MPtr;
      const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
      const int &Size = UnPreparedArgs[I].MSize;
      const int Index = UnPreparedArgs[I].MIndex;
      processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource);
    }
  }

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource) {
    const auto kind_std_layout = detail::kernel_param_kind_t::kind_std_layout;
    const auto kind_accessor = detail::kernel_param_kind_t::kind_accessor;
    const auto kind_sampler = detail::kernel_param_kind_t::kind_sampler;
    const auto kind_pointer = detail::kernel_param_kind_t::kind_pointer;

    switch (Kind) {
    case kind_std_layout:
    case kind_pointer: {
      MArgs.emplace_back(Kind, Ptr, Size, Index + IndexShift);
      break;
    }
    case kind_accessor: {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & 0x7ff);
      switch (AccTarget) {
      case access::target::global_buffer:
      case access::target::constant_buffer: {
        detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
        MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
        if (!IsKernelCreatedFromSource) {
          // Dimensionality of the buffer is 1 when dimensionality of the
          // accessor is 0.
          const size_t SizeAccField =
              sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
          ++IndexShift;
          MArgs.emplace_back(kind_std_layout, &AccImpl->MAccessRange[0],
                             SizeAccField, Index + IndexShift);
          ++IndexShift;
          MArgs.emplace_back(kind_std_layout, &AccImpl->MMemoryRange[0],
                             SizeAccField, Index + IndexShift);
          ++IndexShift;
          MArgs.emplace_back(kind_std_layout, &AccImpl->MOffset[0],
                             SizeAccField, Index + IndexShift);
        }
        break;
      }
      case access::target::local: {
        detail::LocalAccessorBaseHost *LAcc =
            static_cast<detail::LocalAccessorBaseHost *>(Ptr);
        range<3> &Size = LAcc->getSize();
        const int Dims = LAcc->getNumOfDims();
        int SizeInBytes = LAcc->getElementSize();
        for (int I = 0; I < Dims; ++I)
          SizeInBytes *= Size[I];
        MArgs.emplace_back(kind_std_layout, nullptr, SizeInBytes,
                           Index + IndexShift);
        if (!IsKernelCreatedFromSource) {
          ++IndexShift;
          const size_t SizeAccField = Dims * sizeof(Size[0]);
          MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                             Index + IndexShift);
          ++IndexShift;
          MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                             Index + IndexShift);
          ++IndexShift;
          MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                             Index + IndexShift);
        }
        break;
      }
      case access::target::image:
      case access::target::image_array: {
        detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
        MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
        if (!IsKernelCreatedFromSource) {
          // TODO Handle additional kernel arguments for image class
          // if the compiler front-end adds them.
        }
        break;
      }
      case access::target::host_image:
      case access::target::host_buffer: {
        throw cl::sycl::invalid_parameter_error(
            "Unsupported accessor target case.");
        break;
      }
      }
      break;
    }
    case kind_sampler: {
      MArgs.emplace_back(kind_sampler, Ptr, sizeof(sampler),
                         Index + IndexShift);
      break;
    }
    }
  }

  template <typename LambdaName> bool lambdaAndKernelHaveEqualName() {
    // TODO It is unclear a kernel and a lambda/functor must to be equal or not
    // for parallel_for with sycl::kernel and lambda/functor together
    // Now if they are equal we extract argumets from lambda/functor for the
    // kernel. Else it is necessary use set_atg(s) for resolve the order and
    // values of arguments for the kernel.
    assert(MSyclKernel && "MSyclKernel is not initialized");
    const std::string lambdaName = detail::KernelInfo<LambdaName>::getName();
    const std::string kernelName =
        MSyclKernel->get_info<info::kernel::function_name>();
    return lambdaName == kernelName;
  }

  // The method constructs CG object of specific type, pass it to Scheduler and
  // returns sycl::event object representing the command group.
  // It's expected that the method is the latest method executed before
  // object destruction.
  event finalize() {
    sycl::event EventRet;
    std::unique_ptr<detail::CG> CommandGroup;
    switch (MCGType) {
    case detail::CG::KERNEL:
    case detail::CG::RUN_ON_HOST_INTEL:
      CommandGroup.reset(new detail::CGExecKernel(
          std::move(MNDRDesc), std::move(MHostKernel), std::move(MSyclKernel),
          std::move(MArgsStorage), std::move(MAccStorage),
          std::move(MSharedPtrStorage), std::move(MRequirements),
          std::move(MEvents), std::move(MArgs), std::move(MKernelName),
          std::move(MOSModuleHandle), std::move(MStreamStorage), MCGType));
      break;
    case detail::CG::COPY_ACC_TO_PTR:
    case detail::CG::COPY_PTR_TO_ACC:
    case detail::CG::COPY_ACC_TO_ACC:
      CommandGroup.reset(new detail::CGCopy(
          MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents)));
      break;
    case detail::CG::FILL:
      CommandGroup.reset(new detail::CGFill(
          std::move(MPattern), MDstPtr, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents)));
      break;
    case detail::CG::UPDATE_HOST:
      CommandGroup.reset(new detail::CGUpdateHost(
          MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
          std::move(MSharedPtrStorage), std::move(MRequirements),
          std::move(MEvents)));
      break;
    case detail::CG::COPY_USM:
      CommandGroup.reset(new detail::CGCopyUSM(
          MSrcPtr, MDstPtr, MLength, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents)));
      break;
    case detail::CG::FILL_USM:
      CommandGroup.reset(new detail::CGFillUSM(
          std::move(MPattern), MDstPtr, MLength, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents)));
      break;
    case detail::CG::PREFETCH_USM:
      CommandGroup.reset(new detail::CGPrefetchUSM(
          MDstPtr, MLength, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents)));
      break;
    case detail::CG::NONE:
      throw runtime_error("Command group submitted without a kernel or a "
                          "explicit memory operation.");
    default:
      throw runtime_error("Unhandled type of command group");
    }

    detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
        std::move(CommandGroup), std::move(MQueue));

    EventRet = detail::createSyclObjFromImpl<event>(Event);
    return EventRet;
  }

  // Save streams associated with this handler. Streams are then forwarded to
  // command group and flushed in the scheduler.
  void addStream(std::shared_ptr<detail::stream_impl> s) {
    MStreamStorage.push_back(std::move(s));
  }

  ~handler() = default;

  bool is_host() { return MIsHost; }

  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget>
  void associateWithHandler(accessor<DataT, Dims, AccessMode, AccessTarget,
                                     access::placeholder::false_t>
                                Acc) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::Requirement *Req = AccImpl.get();
    // Add accessor to the list of requirements.
    MRequirements.push_back(Req);
    // Store copy of the accessor.
    MAccStorage.push_back(std::move(AccImpl));
    // Add an accessor to the handler list of associated accessors.
    // For associated accessors index does not means nothing.
    MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                     Req, static_cast<int>(AccessTarget),
                                     /*index*/ 0);
  }

  // Recursively calls itself until arguments pack is fully processed.
  // The version for regular(standard layout) argument.
  template <typename T, typename... Ts>
  void setArgsHelper(int ArgIndex, T &&Arg, Ts &&... Args) {
    set_arg(ArgIndex, std::move(Arg));
    setArgsHelper(++ArgIndex, std::move(Args)...);
  }

  void setArgsHelper(int ArgIndex) {}

  // setArgHelper for local accessor argument.
  template <typename DataT, int Dims, access::mode AccessMode,
            access::placeholder IsPlaceholder>
  void setArgHelper(int ArgIndex,
                    accessor<DataT, Dims, AccessMode, access::target::local,
                             IsPlaceholder> &&Arg) {
    detail::LocalAccessorBaseHost *LocalAccBase =
        (detail::LocalAccessorBaseHost *)&Arg;
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_accessor, LocalAccBase,
                       static_cast<int>(access::target::local), ArgIndex);
  }

  // setArgHelper for non local accessor argument.
  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  typename std::enable_if<AccessTarget != access::target::local, void>::type
  setArgHelper(
      int ArgIndex,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder> &&Arg) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Arg;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::Requirement *Req = AccImpl.get();
    // Add accessor to the list of requirements.
    MRequirements.push_back(Req);
    // Store copy of the accessor.
    MAccStorage.push_back(std::move(AccImpl));
    // Add accessor to the list of arguments.
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_accessor, Req,
                       static_cast<int>(AccessTarget), ArgIndex);
  }

  template <typename T> void setArgHelper(int ArgIndex, T &&Arg) {
    void *StoredArg = (void *)storePlainArg(Arg);

    if (!std::is_same<cl_mem, T>::value && std::is_pointer<T>::value) {
      MArgs.emplace_back(detail::kernel_param_kind_t::kind_pointer, StoredArg,
                         sizeof(T), ArgIndex);
    } else {
      MArgs.emplace_back(detail::kernel_param_kind_t::kind_std_layout,
                         StoredArg, sizeof(T), ArgIndex);
    }
  }

  void setArgHelper(int ArgIndex, sampler &&Arg) {
    void *StoredArg = (void *)storePlainArg(Arg);
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_sampler, StoredArg,
                       sizeof(sampler), ArgIndex);
  }

  void verifySyclKernelInvoc(const kernel &SyclKernel) {
    if (is_host()) {
      throw invalid_object_error(
          "This kernel invocation method cannot be used on the host");
    }
    if (SyclKernel.is_host()) {
      throw invalid_object_error("Invalid kernel type, OpenCL expected");
    }
  }

  // Make queue_impl class friend to be able to call finalize method.
  friend class detail::queue_impl;
  // Make accessor class friend to keep the list of associated accessors.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  friend class accessor;

  template <typename DataT, int Dimensions, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  friend class detail::image_accessor;
  // Make stream class friend to be able to keep the list of associated streams
  friend class stream;

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  // The method registers requirement to the memory. So, the command group has a
  // requirement to gain access to the given memory object before executing.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget>
  void
  require(accessor<DataT, Dims, AccMode, AccTarget, access::placeholder::true_t>
              Acc) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::Requirement *Req = AccImpl.get();
    // Add accessor to the list of requirements.
    MRequirements.push_back(Req);
    // Store copy of the accessor.
    MAccStorage.push_back(std::move(AccImpl));
    // Add an accessor to the handler list of associated accessors.
    // For associated accessors index does not means nothing.
    MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                     Req, static_cast<int>(AccTarget),
                                     /*index*/ 0);
  }

  // This method registers event dependencies on this command group.
  void depends_on(event e) {
    MEvents.push_back(std::move(detail::getSyclObjImpl(e)));
  }

  void depends_on(std::vector<event> Events) {
    for (event e : Events) {
      depends_on(e);
    }
  }

  // OpenCL interoperability interface
  // Registers Arg passed as argument # ArgIndex.
  template <typename T> void set_arg(int ArgIndex, T &&Arg) {
    setArgHelper(ArgIndex, std::move(Arg));
  }

  // Registers pack of arguments(Args) with indexes starting from 0.
  template <typename... Ts> void set_args(Ts &&... Args) {
    setArgsHelper(0, std::move(Args)...);
  }

#ifdef __SYCL_DEVICE_ONLY__

  template <typename KernelT, typename IndexerT>
  using EnableIfIndexer = detail::enable_if_t<
      std::is_same<detail::lambda_arg_type<KernelT>, IndexerT>::value>;

  template <typename KernelT, int Dims>
  using EnableIfId = EnableIfIndexer<KernelT, id<Dims>>;

  template <typename KernelT, int Dims>
  using EnableIfItemWithOffset = EnableIfIndexer<KernelT, item<Dims, true>>;

  template <typename KernelT, int Dims>
  using EnableIfItemWithoutOffset = EnableIfIndexer<KernelT, item<Dims, false>>;

  template <typename KernelT, int Dims>
  using EnableIfNDItem = EnableIfIndexer<KernelT, nd_item<Dims>>;

  // NOTE: the name of this function - "kernel_single_task" - is used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType>
  __attribute__((sycl_kernel)) void kernel_single_task(KernelType KernelFunc) {
    KernelFunc();
  }

  // NOTE: the name of these functions - "kernel_parallel_for" - are used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) EnableIfId<KernelType, Dims>
  kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getId<Dims>());
  }

  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) EnableIfItemWithoutOffset<KernelType, Dims>
  kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getItem<Dims, false>());
  }

  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) EnableIfItemWithOffset<KernelType, Dims>
  kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getItem<Dims, true>());
  }

  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) EnableIfNDItem<KernelType, Dims>
  kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getNDItem<Dims>());
  }

  // NOTE: the name of this function - "kernel_parallel_for_work_group" - is
  // used by the Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) void
  kernel_parallel_for_work_group(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getGroup<Dims>());
  }

#endif

  // The method stores lambda to the template-free object and initializes
  // kernel name, list of arguments and requirements using information from
  // integration header.
  template <typename KernelName, typename KernelType, int Dims,
            typename LambdaArgType = sycl::detail::lambda_arg_type<KernelType>>
  void StoreLambda(KernelType KernelFunc) {
    MHostKernel.reset(
        new detail::HostKernel<KernelType, LambdaArgType, Dims>(KernelFunc));

    using KI = sycl::detail::KernelInfo<KernelName>;
    // Empty name indicates that the compilation happens without integration
    // header, so don't perform things that require it.
    if (KI::getName() != "") {
      MArgs.clear();
      extractArgsAndReqsFromLambda(MHostKernel->getPtr(), KI::getNumParams(),
                                   &KI::getParamDesc(0));
      MKernelName = KI::getName();
      MOSModuleHandle = csd::OSUtil::getOSModuleHandle(KI::getName());
    } else {
      // In case w/o the integration header it is necessary to process
      // accessors from the list(which are associated with this handler) as
      // arguments.
      MArgs = std::move(MAssociatedAccesors);
    }
  }

  // single_task version with a kernel represented as a lambda.
  template <typename KernelName = csd::auto_name, typename KernelType>
  void single_task(KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(KernelFunc);
#else
    MNDRDesc.set(range<1>{1});

    StoreLambda<NameT, KernelType, /*Dims*/ 0, void>(KernelFunc);
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version with a kernel represented as a lambda + range that
  // specifies global size only.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // Similar to single_task, but passed lambda will be executed on host.
  template <typename FuncT> void run_on_host_intel(FuncT Func) {
    MNDRDesc.set(range<1>{1});

    MArgs = std::move(MAssociatedAccesors);
    MHostKernel.reset(
        new detail::HostKernel<FuncT, void, 1>(std::move(Func)));
    MCGType = detail::CG::RUN_ON_HOST_INTEL;
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version with a kernel represented as a lambda + nd_range that
  // specifies global, local sizes and offset.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  // single_task version with a kernel represented as a sycl::kernel.
  // The kernel invocation method has no functors and cannot be called on host.
  void single_task(kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MNDRDesc.set(range<1>{1});
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  // parallel_for version with a kernel represented as a sycl::kernel + range
  // that specifies global size only. The kernel invocation method has no
  // functors and cannot be called on host.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NumWorkItems));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  // parallel_for version with a kernel represented as a sycl::kernel + range
  // and offset that specify global size and global offset correspondingly.
  // The kernel invocation method has no functors and cannot be called on host.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> workItemOffset,
                    kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NumWorkItems), std::move(workItemOffset));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  // parallel_for version with a kernel represented as a sycl::kernel + nd_range
  // that specifies global, local sizes and offset. The kernel invocation
  // method has no functors and cannot be called on host.
  template <int Dims>
  void parallel_for(nd_range<Dims> NDRange, kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NDRange));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  // Note: the kernel invocation methods below are only planned to be added
  // to the spec as of v1.2.1 rev. 3, despite already being present in SYCL
  // conformance tests.

  // single_task version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise.
  template <typename KernelName = csd::auto_name, typename KernelType>
  void single_task(kernel SyclKernel, KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(KernelFunc);
#else
    MNDRDesc.set(range<1>{1});
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, /*Dims*/ 0, void>(std::move(KernelFunc));
#endif
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. range argument specifies global size.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(kernel SyclKernel, range<Dims> NumWorkItems,
                    KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. range and id specify global size and offset.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(kernel SyclKernel, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. nd_range specifies global, local size and offset.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(kernel SyclKernel, nd_range<Dims> NDRange,
                    KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NDRange));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  /// This version of \c parallel_for_work_group takes two parameters
  /// representing the same kernel. The first one - \c syclKernel - is a
  /// compiled form of the second one - \c kernelFunc, which is the source form
  /// of the kernel. The same source kernel can be compiled multiple times
  /// yielding multiple kernel class objects accessible via the \c program class
  /// interface.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for_work_group(kernel SyclKernel, range<Dims> NumWorkGroups,
                               KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  /// Two-kernel version of the \c parallel_for_work_group with group and local
  /// range.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for_work_group(kernel SyclKernel, range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               KernelType KernelFunc) {
    using NameT = typename csd::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  // Explicit copy operations API
  constexpr static bool isConstOrGlobal(access::target AccessTarget) {
    return AccessTarget == access::target::global_buffer ||
           AccessTarget == access::target::constant_buffer;
  }

  constexpr static bool isImageOrImageArray(access::target AccessTarget) {
    return AccessTarget == access::target::image ||
           AccessTarget == access::target::image_array;
  }

  constexpr static bool
  isValidTargetForExplicitOp(access::target AccessTarget) {
    return isConstOrGlobal(AccessTarget) || isImageOrImageArray(AccessTarget);
  }

  // copy memory pointed by accessor to host memory pointed by shared_ptr
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void copy(accessor<T_Src, Dims, AccessMode, AccessTarget, IsPlaceholder> Src,
            shared_ptr_class<T_Dst> Dst) {
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Dst);
    T_Dst *RawDstPtr = Dst.get();
    copy(Src, RawDstPtr);
  }

  // copy memory pointer by shared_ptr to host memory pointed by accessor
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void
  copy(shared_ptr_class<T_Src> Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Src);
    T_Src *RawSrcPtr = Src.get();
    copy(RawSrcPtr, Dst);
  }

  // copy memory pointed by accessor to host memory pointed by raw pointer
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void copy(accessor<T_Src, Dims, AccessMode, AccessTarget, IsPlaceholder> Src,
            T_Dst *Dst) {
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
#ifndef __SYCL_DEVICE_ONLY__
    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Src.get_range();
      parallel_for< class __copyAcc2Ptr< T_Src, T_Dst, Dims, AccessMode,
                                         AccessTarget, IsPlaceholder>>
                                         (Range, [=](id<Dims> Index) {
        size_t LinearIndex = Index[0];
        for (int I = 1; I < Dims; ++I)
          LinearIndex += Range[I] * Index[I];
        ((T_Src *)Dst)[LinearIndex] = Src[Index];
      });

      return;
    }
#endif
    MCGType = detail::CG::COPY_ACC_TO_PTR;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = (void *)AccImpl.get();
    MDstPtr = (void *)Dst;
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  // copy memory pointed by raw pointer to host memory pointed by accessor
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void
  copy(const T_Src *Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
#ifndef __SYCL_DEVICE_ONLY__
    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Dst.get_range();
      parallel_for< class __copyPtr2Acc< T_Src, T_Dst, Dims, AccessMode,
                                         AccessTarget, IsPlaceholder>>
                                         (Range, [=](id<Dims> Index) {
        size_t LinearIndex = Index[0];
        for (int I = 1; I < Dims; ++I)
          LinearIndex += Range[I] * Index[I];

        Dst[Index] = ((T_Dst *)Src)[LinearIndex];
      });
      return;
    }
#endif
    MCGType = detail::CG::COPY_PTR_TO_ACC;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = (void *)Src;
    MDstPtr = (void *)AccImpl.get();
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  // copy memory pointed by accessor to the memory pointed by another accessor
  template <
      typename T_Src, int Dims_Src, access::mode AccessMode_Src,
      access::target AccessTarget_Src, typename T_Dst, int Dims_Dst,
      access::mode AccessMode_Dst, access::target AccessTarget_Dst,
      access::placeholder IsPlaceholder_Src = access::placeholder::false_t,
      access::placeholder IsPlaceholder_Dst = access::placeholder::false_t>
  void copy(accessor<T_Src, Dims_Src, AccessMode_Src, AccessTarget_Src,
                     IsPlaceholder_Src>
                Src,
            accessor<T_Dst, Dims_Dst, AccessMode_Dst, AccessTarget_Dst,
                     IsPlaceholder_Dst>
                Dst) {
    static_assert(isValidTargetForExplicitOp(AccessTarget_Src),
                  "Invalid source accessor target for the copy method.");
    static_assert(isValidTargetForExplicitOp(AccessTarget_Dst),
                  "Invalid destination accessor target for the copy method.");
#ifndef __SYCL_DEVICE_ONLY__
    if (MIsHost) {
      range<Dims_Src> Range = Dst.get_range();
      parallel_for< class __copyAcc2Acc< T_Src, Dims_Src, AccessMode_Src,
                                         AccessTarget_Src, T_Dst, Dims_Dst,
                                         AccessMode_Dst, AccessTarget_Dst,
                                         IsPlaceholder_Src,
                                         IsPlaceholder_Dst>>
                                         (Range, [=](id<Dims_Src> Index) {
        Dst[Index] = Src[Index];
      });

      return;
    }
#endif
    MCGType = detail::CG::COPY_ACC_TO_ACC;

    detail::AccessorBaseHost *AccBaseSrc = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImplSrc = detail::getSyclObjImpl(*AccBaseSrc);

    detail::AccessorBaseHost *AccBaseDst = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImplDst = detail::getSyclObjImpl(*AccBaseDst);

    MRequirements.push_back(AccImplSrc.get());
    MRequirements.push_back(AccImplDst.get());
    MSrcPtr = AccImplSrc.get();
    MDstPtr = AccImplDst.get();
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImplSrc));
    MAccStorage.push_back(std::move(AccImplDst));
  }

  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void
  update_host(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder> Acc) {
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the update_host method.");
    MCGType = detail::CG::UPDATE_HOST;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = (void *)AccImpl.get();
    MRequirements.push_back(AccImpl.get());
    MAccStorage.push_back(std::move(AccImpl));
  }

  // Fill memory pointed by accessor with the pattern given.
  // If the operation is submitted to queue associated with OpenCL device and
  // accessor points to one dimensional memory object then use special type for
  // filling. Otherwise fill using regular kernel.
  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void fill(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst,
            const T &Pattern) {
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the fill method.");
    if (!MIsHost && (((Dims == 1) && isConstOrGlobal(AccessTarget)) ||
                     isImageOrImageArray(AccessTarget))) {
      MCGType = detail::CG::FILL;

      detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
      detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

      MDstPtr = (void *)AccImpl.get();
      MRequirements.push_back(AccImpl.get());
      MAccStorage.push_back(std::move(AccImpl));

      MPattern.resize(sizeof(T));
      T *PatternPtr = (T *)MPattern.data();
      *PatternPtr = Pattern;
    } else {

      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Dst.get_range();
      parallel_for<class __fill<T, Dims, AccessMode, AccessTarget,
                                IsPlaceholder>>(Range, [=](id<Dims> Index) {
        Dst[Index] = Pattern;
      });
    }
  }

  // Copy memory from the source to the destination.
  void memcpy(void* Dest, const void* Src, size_t Count) {
    MSrcPtr = const_cast<void *>(Src);
    MDstPtr = Dest;
    MLength = Count;
    MCGType = detail::CG::COPY_USM;
  }

  // Fill the memory pointed to by the destination with the given bytes.
  void memset(void *Dest, int Value, size_t Count) {
    MDstPtr = Dest;
    MPattern.push_back((char)Value);
    MLength = Count;
    MCGType = detail::CG::FILL_USM;
  }

  // Prefetch the memory pointed to by the pointer.
  void prefetch(const void *Ptr, size_t Count) {
    MDstPtr = const_cast<void *>(Ptr);
    MLength = Count;
    MCGType = detail::CG::PREFETCH_USM;
  }
};
} // namespace sycl
} // namespace cl
