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
#include <CL/sycl/detail/kernel_impl.hpp>
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

#include <algorithm>
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

__SYCL_INLINE namespace cl {
namespace sycl {

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
template <typename Type> struct get_kernel_name_t<detail::auto_name, Type> {
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
  std::vector<detail::LocalAccessorImplPtr> MLocalAccStorage;
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

  void throwIfActionIsCreated() {
    if (detail::CG::NONE != MCGType)
      throw sycl::runtime_error("Attempt to set multiple actions for the "
                                "command group. Command group must consist of "
                                "a single kernel or explicit memory operation.",
                                CL_INVALID_OPERATION);
  }

  // The method extracts and prepares kernel arguments from the lambda using
  // integration header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs);

  // The method extracts and prepares kernel arguments that were set
  // via set_arg(s)
  void extractArgsAndReqs();

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource);

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
  event finalize();

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
    detail::LocalAccessorImplPtr LocalAccImpl =
        detail::getSyclObjImpl(*LocalAccBase);
    detail::LocalAccessorImplHost *Req = LocalAccImpl.get();
    MLocalAccStorage.push_back(std::move(LocalAccImpl));
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_accessor, Req,
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

  static id<1> getDelinearizedIndex(const range<1> Range, const size_t Index) {
    return {Index};
  }

  static id<2> getDelinearizedIndex(const range<2> Range, const size_t Index) {
    size_t x = Index / Range[1];
    size_t y = Index % Range[1];
    return {x, y};
  }

  static id<3> getDelinearizedIndex(const range<3> Range, const size_t Index) {
    size_t x = Index / (Range[1] * Range[2]);
    size_t y = (Index / Range[2]) % Range[1];
    size_t z = Index % Range[2];
    return {x, y, z};
  }

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
      MOSModuleHandle = detail::OSUtil::getOSModuleHandle(KI::getName());
    } else {
      // In case w/o the integration header it is necessary to process
      // accessors from the list(which are associated with this handler) as
      // arguments.
      MArgs = std::move(MAssociatedAccesors);
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
  friend class detail::stream_impl;

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
      std::is_same<detail::lambda_arg_type<KernelT>, IndexerT>::value, int>;

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
  template <typename KernelName, typename KernelType, int Dims,
            EnableIfId<KernelType, Dims> = 0>
  __attribute__((sycl_kernel)) void kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getId<Dims>());
  }

  template <typename KernelName, typename KernelType, int Dims,
            EnableIfItemWithoutOffset<KernelType, Dims> = 0>
  __attribute__((sycl_kernel)) void kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getItem<Dims, false>());
  }

  template <typename KernelName, typename KernelType, int Dims,
            EnableIfItemWithOffset<KernelType, Dims> = 0>
  __attribute__((sycl_kernel)) void kernel_parallel_for(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getItem<Dims, true>());
  }

  template <typename KernelName, typename KernelType, int Dims,
            EnableIfNDItem<KernelType, Dims> = 0>
  __attribute__((sycl_kernel)) void kernel_parallel_for(KernelType KernelFunc) {
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

  // single_task version with a kernel represented as a lambda.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
    throwIfActionIsCreated();
    MNDRDesc.set(range<1>{1});

    MArgs = std::move(MAssociatedAccesors);
    MHostKernel.reset(new detail::HostKernel<FuncT, void, 1>(std::move(Func)));
    MCGType = detail::CG::RUN_ON_HOST_INTEL;
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(kernel SyclKernel, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel SyclKernel, range<Dims> NumWorkItems,
                    KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel SyclKernel, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel SyclKernel, nd_range<Dims> NDRange,
                    KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(kernel SyclKernel, range<Dims> NumWorkGroups,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(kernel SyclKernel, range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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

  // Checks whether it is possible to copy the source shape to the destination
  // shape(the shapes are described by the accessor ranges) by using
  // copying by regions of memory and not copying element by element
  // Shapes can be 1, 2 or 3 dimensional rectangles.
  template <int Dims_Src, int Dims_Dst>
  static bool IsCopyingRectRegionAvailable(const range<Dims_Src> Src,
                                           const range<Dims_Dst> Dst) {
    if (Dims_Src > Dims_Dst)
      return false;
    for (size_t I = 0; I < Dims_Src; ++I)
      if (Src[I] > Dst[I])
        return false;
    return true;
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
    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget_Src),
                  "Invalid source accessor target for the copy method.");
    static_assert(isValidTargetForExplicitOp(AccessTarget_Dst),
                  "Invalid destination accessor target for the copy method.");
    // TODO replace to get_size() when it will provide correct values.
    assert(
        (Dst.get_range().size() * sizeof(T_Dst) >=
         Src.get_range().size() * sizeof(T_Src)) &&
        "dest must have at least as many bytes as the range accessed by src.");
    if (MIsHost ||
        !IsCopyingRectRegionAvailable(Src.get_range(), Dst.get_range())) {
      range<Dims_Src> CopyRange = Src.get_range();
      size_t Range = 1;
      for (size_t I = 0; I < Dims_Src; ++I)
        Range *= CopyRange[I];
      range<1> LinearizedRange(Range);
      parallel_for< class __copyAcc2Acc< T_Src, Dims_Src, AccessMode_Src,
                                         AccessTarget_Src, T_Dst, Dims_Dst,
                                         AccessMode_Dst, AccessTarget_Dst,
                                         IsPlaceholder_Src,
                                         IsPlaceholder_Dst>>
                                         (LinearizedRange, [=](id<1> Id) {
        size_t Index = Id[0];
        id<Dims_Src> SrcIndex = getDelinearizedIndex(Src.get_range(), Index);
        id<Dims_Dst> DstIndex = getDelinearizedIndex(Dst.get_range(), Index);
        Dst[DstIndex] = Src[SrcIndex];
      });

      return;
    }
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
    throwIfActionIsCreated();
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
    throwIfActionIsCreated();
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
  void memcpy(void *Dest, const void *Src, size_t Count) {
    throwIfActionIsCreated();
    MSrcPtr = const_cast<void *>(Src);
    MDstPtr = Dest;
    MLength = Count;
    MCGType = detail::CG::COPY_USM;
  }

  // Fill the memory pointed to by the destination with the given bytes.
  void memset(void *Dest, int Value, size_t Count) {
    throwIfActionIsCreated();
    MDstPtr = Dest;
    MPattern.push_back((char)Value);
    MLength = Count;
    MCGType = detail::CG::FILL_USM;
  }

  // Prefetch the memory pointed to by the pointer.
  void prefetch(const void *Ptr, size_t Count) {
    throwIfActionIsCreated();
    MDstPtr = const_cast<void *>(Ptr);
    MLength = Count;
    MCGType = detail::CG::PREFETCH_USM;
  }
};
} // namespace sycl
} // namespace cl
