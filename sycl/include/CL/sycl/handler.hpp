//==-------- handler.hpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/os_util.hpp>
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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration

class handler;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
namespace detail {

/// This class is the default KernelName template parameter type for kernel
/// invocation APIs such as single_task.
class auto_name {};

class kernel_impl;
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

__SYCL_EXPORT device getDeviceFromHandler(handler &);

/// These are the forward declaration for the classes that help to create
/// names for additional kernels. It is used only when there are
/// more then 1 kernels in one parallel_for() implementing SYCL reduction.
template <typename Type> class __sycl_reduction_main_2nd_kernel;
template <typename Type> class __sycl_reduction_aux_1st_kernel;
template <typename Type> class __sycl_reduction_aux_2nd_kernel;

/// Helper structs to get additional kernel name types based on given
/// \c Name and \c Type types: if \c Name is undefined (is a \c auto_name) then
/// \c Type becomes the \c Name.
template <typename Name, typename Type>
struct get_reduction_main_2nd_kernel_name_t {
  using name = __sycl_reduction_main_2nd_kernel<
      typename get_kernel_name_t<Name, Type>::name>;
};
template <typename Name, typename Type>
struct get_reduction_aux_1st_kernel_name_t {
  using name = __sycl_reduction_aux_1st_kernel<
      typename get_kernel_name_t<Name, Type>::name>;
};
template <typename Name, typename Type>
struct get_reduction_aux_2nd_kernel_name_t {
  using name = __sycl_reduction_aux_2nd_kernel<
      typename get_kernel_name_t<Name, Type>::name>;
};

device getDeviceFromHandler(handler &);

} // namespace detail

namespace intel {
namespace detail {
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPlaceholder>
class reduction_impl;
} // namespace detail
} // namespace intel

/// 4.8.3 Command group handler class
///
/// Objects of the handler class collect information about command group, such
/// as kernel, requirements to the memory, arguments for the kernel.
///
/// \code{.cpp}
/// sycl::queue::submit([](handler &CGH){
///   CGH.require(Accessor1);   // Adds a requirement to the memory object.
///   CGH.setArg(0, Accessor2); // Registers accessor given as an argument to
///                             // the kernel + adds a requirement to the memory
///                             // object.
///   CGH.setArg(1, N);         // Registers value given as an argument to the
///                             // kernel.
///   // The following registers KernelFunctor to be a kernel that will be
///   // executed in case of queue is bound to the host device, Kernel - for
///   // an OpenCL device. This function clearly indicates that command group
///   // represents kernel execution.
///   CGH.parallel_for(KernelFunctor, Kernel);
///  });
/// \endcode
///
/// The command group can represent absolutely different operations. Depending
/// on the operation we need to store different data. But, in most cases, it's
/// impossible to say what kind of operation we need to perform until the very
/// end. So, handler class contains all fields simultaneously, then during
/// "finalization" it constructs CG object, that represents specific operation,
/// passing fields that are required only.
class __SYCL_EXPORT handler {
private:
  /// Constructs SYCL handler from queue.
  ///
  /// \param Queue is a SYCL queue.
  /// \param IsHost indicates if this handler is created for SYCL host device.
  handler(shared_ptr_class<detail::queue_impl> Queue, bool IsHost)
      : MQueue(std::move(Queue)), MIsHost(IsHost) {}

  /// Stores copy of Arg passed to the MArgsStorage.
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

  /// Extracts and prepares kernel arguments from the lambda using integration
  /// header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs);

  /// Extracts and prepares kernel arguments set via set_arg(s).
  void extractArgsAndReqs();

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource);

  /// \return a string containing name of SYCL kernel.
  string_class getKernelName();

  template <typename LambdaNameT> bool lambdaAndKernelHaveEqualName() {
    // TODO It is unclear a kernel and a lambda/functor must to be equal or not
    // for parallel_for with sycl::kernel and lambda/functor together
    // Now if they are equal we extract argumets from lambda/functor for the
    // kernel. Else it is necessary use set_atg(s) for resolve the order and
    // values of arguments for the kernel.
    assert(MKernel && "MKernel is not initialized");
    const string_class LambdaName = detail::KernelInfo<LambdaNameT>::getName();
    const string_class KernelName = getKernelName();
    return LambdaName == KernelName;
  }

  /// Saves the location of user's code passed in \param CodeLoc for future
  /// usage in finalize() method.
  void saveCodeLoc(detail::code_location CodeLoc) { MCodeLoc = CodeLoc; }

  /// Stores the given \param Event to the \param Queue.
  /// Even though MQueue is a field of handler, the method addEvent() of
  /// queue_impl class cannot be called inside this handler.hpp file
  /// as queue_impl is incomplete class for handler.
  static void addEventToQueue(shared_ptr_class<detail::queue_impl> Queue,
                              cl::sycl::event Event);

  /// Constructs CG object of specific type, passes it to Scheduler and
  /// returns sycl::event object representing the command group.
  /// It's expected that the method is the latest method executed before
  /// object destruction.
  ///
  /// \return a SYCL event object representing the command group
  event finalize();

  /// Saves streams associated with this handler.
  ///
  /// Streams are then forwarded to command group and flushed in the scheduler.
  ///
  /// \param Stream is a pointer to SYCL stream.
  void addStream(shared_ptr_class<detail::stream_impl> Stream) {
    MStreamStorage.push_back(std::move(Stream));
  }

  /// Saves buffers created by handling reduction feature in handler.
  /// They are then forwarded to command group and destroyed only after
  /// the command group finishes the work on device/host.
  /// The 'MSharedPtrStorage' suits that need.
  ///
  /// @param ReduObj is a pointer to object that must be stored.
  void addReduction(shared_ptr_class<const void> ReduObj) {
    MSharedPtrStorage.push_back(std::move(ReduObj));
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

  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget>
  void dissociateWithHandler(accessor<DataT, Dims, AccessMode, AccessTarget,
                                      access::placeholder::false_t>
                                 Acc) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::Requirement *Req = AccImpl.get();

    // Remove accessor from the list of requirements, accessors storage,
    // and from the list of associated accessors.
    auto ReqIt = std::find(MRequirements.begin(), MRequirements.end(), Req);
    auto AccIt = std::find(MAccStorage.begin(), MAccStorage.end(), AccImpl);
    auto It =
        std::find_if(MAssociatedAccesors.begin(), MAssociatedAccesors.end(),
                     [Req](const detail::ArgDesc &D) { return D.MPtr == Req; });
    assert((ReqIt != MRequirements.end() && AccIt != MAccStorage.end() &&
            It != MAssociatedAccesors.end()) &&
           "Cannot dissociate accessor.");
    MRequirements.erase(ReqIt);
    MAccStorage.erase(AccIt);
    MAssociatedAccesors.erase(It);
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

  void verifyKernelInvoc(const kernel &Kernel) {
    if (is_host()) {
      throw invalid_object_error(
          "This kernel invocation method cannot be used on the host",
          PI_INVALID_DEVICE);
    }
    if (Kernel.is_host()) {
      throw invalid_object_error("Invalid kernel type, OpenCL expected",
                                 PI_INVALID_KERNEL);
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

  /// Stores lambda to the template-free object
  ///
  /// Also initializes kernel name, list of arguments and requirements using
  /// information from the integration header.
  ///
  /// \param KernelFunc is a SYCL kernel function.
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

  /// Checks whether it is possible to copy the source shape to the destination
  /// shape(the shapes are described by the accessor ranges) by using
  /// copying by regions of memory and not copying element by element
  /// Shapes can be 1, 2 or 3 dimensional rectangles.
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

  /// Handles some special cases of the copy operation from one accessor
  /// to another accessor. Returns true if the copy is handled here.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a destination SYCL accessor.
  // TODO: support atomic accessor in Src or/and Dst.
  template <typename TSrc, int DimSrc, access::mode ModeSrc,
            access::target TargetSrc, typename TDst, int DimDst,
            access::mode ModeDst, access::target TargetDst,
            access::placeholder IsPHSrc, access::placeholder IsPHDst>
  detail::enable_if_t<(DimSrc > 0) && (DimDst > 0), bool>
  copyAccToAccHelper(accessor<TSrc, DimSrc, ModeSrc, TargetSrc, IsPHSrc> Src,
                     accessor<TDst, DimDst, ModeDst, TargetDst, IsPHDst> Dst) {
    if (!MIsHost &&
        IsCopyingRectRegionAvailable(Src.get_range(), Dst.get_range()))
      return false;

    range<1> LinearizedRange(Src.get_count());
    parallel_for<class __copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc,
                                     TDst, DimDst, ModeDst, TargetDst,
                                     IsPHSrc, IsPHDst>>
                                     (LinearizedRange, [=](id<1> Id) {
      size_t Index = Id[0];
      id<DimSrc> SrcIndex = getDelinearizedIndex(Src.get_range(), Index);
      id<DimDst> DstIndex = getDelinearizedIndex(Dst.get_range(), Index);
      Dst[DstIndex] = Src[SrcIndex];
    });
    return true;
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<Dim == 0 && Mode == access::mode::atomic, T>
  readFromFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Src) const {
    atomic<T, access::address_space::global_space> AtomicSrc = Src;
    return AtomicSrc.load();
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<(Dim > 0) && Mode == access::mode::atomic, T>
  readFromFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Src) const {
    id<Dim> Id = getDelinearizedIndex(Src.get_range(), 0);
    return Src[Id].load();
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<Mode != access::mode::atomic, T>
  readFromFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Src) const {
    return *(Src.get_pointer());
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<Dim == 0 && Mode == access::mode::atomic, void>
  writeToFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Dst, T V) const {
    atomic<T, access::address_space::global_space> AtomicDst = Dst;
    AtomicDst.store(V);
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<(Dim > 0) && Mode == access::mode::atomic, void>
  writeToFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Dst, T V) const {
    id<Dim> Id = getDelinearizedIndex(Dst.get_range(), 0);
    Dst[Id].store(V);
  }

  template <typename T, int Dim, access::mode Mode, access::target Target,
            access::placeholder IsPH>
  detail::enable_if_t<Mode != access::mode::atomic, void>
  writeToFirstAccElement(accessor<T, Dim, Mode, Target, IsPH> Dst, T V) const {
    *(Dst.get_pointer()) = V;
  }

  /// Handles some special cases of the copy operation from one accessor
  /// to another accessor. Returns true if the copy is handled here.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a destination SYCL accessor.
  template <typename TSrc, int DimSrc, access::mode ModeSrc,
            access::target TargetSrc, typename TDst, int DimDst,
            access::mode ModeDst, access::target TargetDst,
            access::placeholder IsPHSrc, access::placeholder IsPHDst>
  detail::enable_if_t<DimSrc == 0 || DimDst == 0, bool>
  copyAccToAccHelper(accessor<TSrc, DimSrc, ModeSrc, TargetSrc, IsPHSrc> Src,
                     accessor<TDst, DimDst, ModeDst, TargetDst, IsPHDst> Dst) {
    if (!MIsHost)
      return false;

    single_task<class __copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc,
                                    TDst, DimDst, ModeDst, TargetDst,
                                    IsPHSrc, IsPHDst>> ([=]() {
      writeToFirstAccElement(Dst, readFromFirstAccElement(Src));
    });
    return true;
  }

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
  __attribute__((sycl_kernel)) void
  kernel_parallel_for_nd_range(KernelType KernelFunc) {
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

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  /// Requires access to the memory object associated with the placeholder
  /// accessor.
  ///
  /// The command group has a requirement to gain access to the given memory
  /// object before executing.
  ///
  /// \param Acc is a SYCL accessor describing required memory region.
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

  /// Registers event dependencies on this command group.
  ///
  /// \param Event is a valid SYCL event to wait on.
  void depends_on(event Event) {
    MEvents.push_back(std::move(detail::getSyclObjImpl(Event)));
  }

  /// Registers event dependencies on this command group.
  ///
  /// \param Events is a vector of valid SYCL events to wait on.
  void depends_on(vector_class<event> Events) {
    for (event &Event : Events) {
      MEvents.push_back(std::move(detail::getSyclObjImpl(Event)));
    }
  }

  /// Sets argument for OpenCL interoperability kernels.
  ///
  /// Registers Arg passed as argument # ArgIndex.
  ///
  /// \param ArgIndex is a positional number of argument to be set.
  /// \param Arg is an argument value to be set.
  template <typename T> void set_arg(int ArgIndex, T &&Arg) {
    setArgHelper(ArgIndex, std::move(Arg));
  }

  /// Sets arguments for OpenCL interoperability kernels.
  ///
  /// Registers pack of arguments(Args) with indexes starting from 0.
  ///
  /// \param Args are argument values to be set.
  template <typename... Ts> void set_args(Ts &&... Args) {
    setArgsHelper(0, std::move(Args)...);
  }

  /// Defines and invokes a SYCL kernel function as a function object type.
  ///
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param KernelFunc is a SYCL kernel function.
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

  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id or item for indexing in the indexing
  /// space defined by range.
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param KernelFunc is a SYCL kernel function.
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

  /// Defines and invokes a SYCL kernel on host device.
  ///
  /// \param Func is a SYCL kernel function defined by lambda function or a
  /// named function object type.
  template <typename FuncT> void run_on_host_intel(FuncT Func) {
    throwIfActionIsCreated();
    MNDRDesc.set(range<1>{1});

    MArgs = std::move(MAssociatedAccesors);
    MHostKernel.reset(new detail::HostKernel<FuncT, void, 1>(std::move(Func)));
    MCGType = detail::CG::RUN_ON_HOST_INTEL;
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offset.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id or item for indexing in the indexing
  /// space defined by range.
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param WorkItemOffset is an offset to be applied to each work item index.
  /// \param KernelFunc is a SYCL kernel function.
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

  /// Defines and invokes a SYCL kernel function for the specified nd_range.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id or item for indexing in the indexing
  /// space defined by range.
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param ExecutionRange is a ND-range defining global and local sizes as
  /// well as offset.
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_nd_range<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  /// Implements a command group function that enqueues a kernel that calls
  /// user's lambda function \param KernelFunc and does one iteration of
  /// reduction of elements in each of work-groups.
  /// This version uses tree-reduction algorithm to reduce elements in each
  /// of work-groups. At the end of each work-group the partial sum is written
  /// to a global buffer.
  ///
  /// Briefly: user's lambda, tree-reduction, CUSTOM types/ops.
  template <typename KernelName, typename KernelType, int Dims, class Reduction>
  void reduCGFunc(KernelType KernelFunc, const nd_range<Dims> &Range,
                  Reduction &Redu) {

    size_t NWorkItems = Range.get_global_range().size();
    size_t WGSize = Range.get_local_range().size();
    size_t NWorkGroups = Range.get_group_range().size();

    bool IsUnderLoaded = (NWorkGroups * WGSize - NWorkItems) != 0;
    bool IsEfficientCase = !IsUnderLoaded && ((WGSize & (WGSize - 1)) == 0);

    bool IsUpdateOfUserAcc =
        Reduction::accessor_mode == access::mode::read_write &&
        NWorkGroups == 1;

    // Use local memory to reduce elements in work-groups into 0-th element.
    // If WGSize is not power of two, then WGSize+1 elements are allocated.
    // The additional last element is used to catch elements that could
    // otherwise be lost in the tree-reduction algorithm.
    size_t NumLocalElements = WGSize + (IsEfficientCase ? 0 : 1);
    auto LocalReds = Redu.getReadWriteLocalAcc(NumLocalElements, *this);

    auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, 0, *this);
    auto ReduIdentity = Redu.getIdentity();
    if (IsEfficientCase) {
      // Efficient case: work-groups are fully loaded and work-group size
      // is power of two.
      parallel_for<KernelName>(Range, [=](nd_item<Dims> NDIt) {
        // Call user's functions. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer(ReduIdentity);
        KernelFunc(NDIt, Reducer);

        // Copy the element to local memory to prepare it for tree-reduction.
        size_t LID = NDIt.get_local_linear_id();
        LocalReds[LID] = Reducer.MValue;
        NDIt.barrier();

        // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0].
        typename Reduction::binary_operation BOp;
        size_t WGSize = NDIt.get_local_range().size();
        for (size_t CurStep = WGSize >> 1; CurStep > 0; CurStep >>= 1) {
          if (LID < CurStep)
            LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
          NDIt.barrier();
        }

        // Compute the partial sum/reduction for the work-group.
        if (LID == 0)
          Out.get_pointer().get()[NDIt.get_group_linear_id()] =
              IsUpdateOfUserAcc ? BOp(*(Out.get_pointer()), LocalReds[0])
                                : LocalReds[0];
      });
    } else {
      // Inefficient case: work-groups are not fully loaded
      // or WGSize is not power of two.
      // These two inefficient cases are handled by one kernel, which
      // can be split later into two separate kernels, if there are users who
      // really need more efficient code for them.
      using AuxName = typename detail::get_reduction_main_2nd_kernel_name_t<
          KernelName, KernelType>::name;
      parallel_for<AuxName>(Range, [=](nd_item<Dims> NDIt) {
        // Call user's functions. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer(ReduIdentity);
        KernelFunc(NDIt, Reducer);

        size_t WGSize = NDIt.get_local_range().size();
        size_t LID = NDIt.get_local_linear_id();
        size_t GID = NDIt.get_global_linear_id();
        // Copy the element to local memory to prepare it for tree-reduction.
        LocalReds[LID] = (GID < NWorkItems) ? Reducer.MValue : ReduIdentity;
        LocalReds[WGSize] = ReduIdentity;
        NDIt.barrier();

        // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
        // LocalReds[WGSize] accumulates last/odd elements when the step
        // of tree-reduction loop is not even.
        typename Reduction::binary_operation BOp;
        size_t PrevStep = WGSize;
        for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
          if (LID < CurStep)
            LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
          else if (LID == CurStep && (PrevStep & 0x1))
            LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
          NDIt.barrier();
          PrevStep = CurStep;
        }

        // Compute the partial sum/reduction for the work-group.
        if (LID == 0) {
          auto GrID = NDIt.get_group_linear_id();
          auto V = BOp(LocalReds[0], LocalReds[WGSize]);
          Out.get_pointer().get()[GrID] =
              IsUpdateOfUserAcc ? BOp(*(Out.get_pointer()), V) : V;
        }
      });
    }
  }

  /// Implements a command group function that enqueues a kernel that does one
  /// iteration of reduction of elements in each of work-groups.
  /// This version uses tree-reduction algorithm to reduce elements in each
  /// of work-groups. At the end of each work-group the partial sum is written
  /// to a global buffer.
  ///
  /// Briefly: aux kernel, tree-reduction, CUSTOM types/ops.
  template <typename KernelName, typename KernelType, int Dims, class Reduction>
  void reduAuxCGFunc(const nd_range<Dims> &Range, size_t NWorkItems,
                     size_t KernelRun, Reduction &Redu) {
    size_t WGSize = Range.get_local_range().size();
    size_t NWorkGroups = Range.get_group_range().size();

    // The last work-group may be not fully loaded with work, or the work group
    // size may be not power of those. Those two cases considered inefficient
    // as they require additional code and checks in the kernel.
    bool IsUnderLoaded = NWorkGroups * WGSize != NWorkItems;
    bool IsEfficientCase = !IsUnderLoaded && (WGSize & (WGSize - 1)) == 0;

    bool IsUpdateOfUserAcc =
        Reduction::accessor_mode == access::mode::read_write &&
        NWorkGroups == 1;

    // Use local memory to reduce elements in work-groups into 0-th element.
    // If WGSize is not power of two, then WGSize+1 elements are allocated.
    // The additional last element is used to catch elements that could
    // otherwise be lost in the tree-reduction algorithm.
    size_t NumLocalElements = WGSize + (IsEfficientCase ? 0 : 1);
    auto LocalReds = Redu.getReadWriteLocalAcc(NumLocalElements, *this);

    // Get read accessor to the buffer that was used as output
    // in the previous kernel. After that create new output buffer if needed
    // and get accessor to it (or use reduction's accessor if the kernel
    // is the last one).
    auto In = Redu.getReadAccToPreviousPartialReds(*this);
    auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, KernelRun, *this);

    if (IsEfficientCase) {
      // Efficient case: work-groups are fully loaded and work-group size
      // is power of two.
      using AuxName = typename detail::get_reduction_aux_1st_kernel_name_t<
          KernelName, KernelType>::name;
      parallel_for<AuxName>(Range, [=](nd_item<Dims> NDIt) {
        // Copy the element to local memory to prepare it for tree-reduction.
        size_t LID = NDIt.get_local_linear_id();
        size_t GID = NDIt.get_global_linear_id();
        LocalReds[LID] = In[GID];
        NDIt.barrier();

        // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
        typename Reduction::binary_operation BOp;
        size_t WGSize = NDIt.get_local_range().size();
        for (size_t CurStep = WGSize >> 1; CurStep > 0; CurStep >>= 1) {
          if (LID < CurStep)
            LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
          NDIt.barrier();
        }

        // Compute the partial sum/reduction for the work-group.
        if (LID == 0)
          Out.get_pointer().get()[NDIt.get_group_linear_id()] =
              IsUpdateOfUserAcc ? BOp(*(Out.get_pointer()), LocalReds[0])
                                : LocalReds[0];
      });
    } else {
      // Inefficient case: work-groups are not fully loaded
      // or WGSize is not power of two.
      // These two inefficient cases are handled by one kernel, which
      // can be split later into two separate kernels, if there are users
      // who really need more efficient code for them.
      using AuxName = typename detail::get_reduction_aux_2nd_kernel_name_t<
          KernelName, KernelType>::name;
      auto ReduIdentity = Redu.getIdentity();
      parallel_for<AuxName>(Range, [=](nd_item<Dims> NDIt) {
        size_t WGSize = NDIt.get_local_range().size();
        size_t LID = NDIt.get_local_linear_id();
        size_t GID = NDIt.get_global_linear_id();
        // Copy the element to local memory to prepare it for tree-reduction
        LocalReds[LID] = (GID < NWorkItems) ? In[GID] : ReduIdentity;
        LocalReds[WGSize] = ReduIdentity;
        NDIt.barrier();

        // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
        // LocalReds[WGSize] accumulates last/odd elements when the step
        // of tree-reduction loop is not even.
        typename Reduction::binary_operation BOp;
        size_t PrevStep = WGSize;
        for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
          if (LID < CurStep)
            LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
          else if (LID == CurStep && (PrevStep & 0x1))
            LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
          NDIt.barrier();
          PrevStep = CurStep;
        }

        // Compute the partial sum/reduction for the work-group.
        if (LID == 0) {
          auto GrID = NDIt.get_group_linear_id();
          auto V = BOp(LocalReds[0], LocalReds[WGSize]);
          Out.get_pointer().get()[GrID] =
              IsUpdateOfUserAcc ? BOp(*(Out.get_pointer()), V) : V;
        }
      });
    }
  }

  /// Defines and invokes a SYCL kernel function for the specified nd_range.
  /// Performs reduction operation specified in \param Redu.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id or item for indexing in the indexing
  /// space defined by range.
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// TODO: currently it calls only those versions of kernels that can handle
  /// custom types and operations. Some of types and operations may use faster
  /// implementations that use intel::reduce() and/or sycl::atomic.fetch_<op>()
  /// functions and thus provide much better performance. Those variants exist,
  /// are fully functional. They just wait for their time for code-review.
  /// TODO: Need to handle more than 1 reduction in parallel_for().
  /// TODO: Support HOST. The kernels called by this parallel_for() may use
  /// some functionality that is not yet supported on HOST such as:
  /// barrier(), and intel::reduce() that also may be used in more
  /// optimized implementations waiting for their turn of code-review.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  void parallel_for(nd_range<Dims> Range, Reduction &Redu,
                    KernelType KernelFunc) {
    size_t NWorkGroups = Range.get_group_range().size();

    // This parallel_for() is lowered to the following sequence:
    // 1) Call a kernel that a) call user's lambda function and b) performs
    //    one iteration of reduction, storing the partial reductions/sums
    //    to either a newly created global buffer or to user's reduction
    //    accessor. So, if the original 'Range' has totally
    //    N1 elements and work-group size is W, then after the first iteration
    //    there will be N2 partial sums where N2 = N1 / W.
    //    If (N2 == 1) then the partial sum is written to user's accessor.
    //    Otherwise, a new global buffer is created and partial sums are written
    //    to it.
    // 2) Call an aux kernel (if necessary, i.e. if N2 > 1) as many times as
    //    necessary to reduce all partial sums into one final sum.

    // 1. Call the kernel that includes user's lambda function.
    // If this kernel is going to be now last one, i.e. it does not write
    // to user's accessor, then detach user's accessor from this kernel
    // to make the dependencies between accessors and kernels more clean and
    // correct.
    if (NWorkGroups > 1)
      dissociateWithHandler(Redu.MAcc);

    reduCGFunc<KernelName>(KernelFunc, Range, Redu);
    auto QueueCopy = MQueue;
    MLastEvent = this->finalize();

    // 2. Run the additional aux kernel as many times as needed to reduce
    // all partial sums into one scalar.
    size_t WGSize = Range.get_local_range().size();
    size_t NWorkItems = NWorkGroups;
    size_t KernelRun = 1;
    while (NWorkItems > 1) {
      // Before creating another kernel, add the event from the previous kernel
      // to queue.
      addEventToQueue(QueueCopy, MLastEvent);

      // TODO: here the work-group size is not limited by user's needs,
      // the better strategy here is to make the work-group-size as big
      // as possible.
      WGSize = std::min(WGSize, NWorkItems);
      NWorkGroups = NWorkItems / WGSize;
      // The last group may be not fully loaded. Still register it as a group.
      if ((NWorkItems % WGSize) != 0)
        ++NWorkGroups;
      auto Range =
          nd_range<1>(range<1>(WGSize * NWorkGroups), range<1>(WGSize));

      handler AuxHandler(QueueCopy, MIsHost);
      AuxHandler.saveCodeLoc(MCodeLoc);

      // The last kernel DOES write to reduction's accessor.
      // Associate it with handler manually.
      if (NWorkGroups == 1)
        AuxHandler.associateWithHandler(Redu.MAcc);
      AuxHandler.reduAuxCGFunc<KernelName, KernelType>(Range, NWorkItems,
                                                       KernelRun, Redu);
      MLastEvent = AuxHandler.finalize();

      NWorkItems = NWorkGroups;
      ++KernelRun;
    } // end while (NWorkItems > 1)
  }

  /// Hierarchical kernel invocation method of a kernel defined as a lambda
  /// encoding the body of each work-group to launch.
  ///
  /// Lambda may contain multiple calls to parallel_for_work_item(...) methods
  /// representing the execution on each work-item. Launches NumWorkGroups
  /// work-groups of runtime-defined size.
  ///
  /// \param NumWorkGroups is a range describing the number of work-groups in
  /// each dimension.
  /// \param KernelFunc is a lambda representing kernel.
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

  /// Hierarchical kernel invocation method of a kernel defined as a lambda
  /// encoding the body of each work-group to launch.
  ///
  /// Lambda may contain multiple calls to parallel_for_work_item(...) methods
  /// representing the execution on each work-item. Launches NumWorkGroups
  /// work-groups of WorkGroupSize size.
  ///
  /// \param NumWorkGroups is a range describing the number of work-groups in
  /// each dimension.
  /// \param WorkGroupSize is a range describing the size of work-groups in
  /// each dimension.
  /// \param KernelFunc is a lambda representing kernel.
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

  /// Invokes a SYCL kernel.
  ///
  /// Executes exactly once. The kernel invocation method has no functors and
  /// cannot be called on host.
  ///
  /// \param Kernel is a SYCL kernel object.
  void single_task(kernel Kernel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kernel);
    MNDRDesc.set(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object. The kernel
  /// invocation method has no functors and cannot be called on host.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param Kenrel is a SYCL kernel function.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, kernel Kenrel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kenrel);
    MKernel = detail::getSyclObjImpl(std::move(Kenrel));
    MNDRDesc.set(std::move(NumWorkItems));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offsets.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param WorkItemOffset is an offset to be applied to each work item index.
  /// \param Kernel is a SYCL kernel function.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    kernel Kernel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kernel);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offsets.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object.
  ///
  /// \param NDRange is a ND-range defining global and local sizes as
  /// well as offset.
  /// \param Kernel is a SYCL kernel function.
  template <int Dims> void parallel_for(nd_range<Dims> NDRange, kernel Kernel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kernel);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MNDRDesc.set(std::move(NDRange));
    MCGType = detail::CG::KERNEL;
    extractArgsAndReqs();
  }

  /// Defines and invokes a SYCL kernel function.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(kernel Kernel, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(KernelFunc);
#else
    MNDRDesc.set(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, /*Dims*/ 0, void>(std::move(KernelFunc));
#endif
  }

  /// Invokes a lambda on the host. Dependencies are satisfied on the host.
  ///
  /// \param Func is a lambda that is executed on the host
  template <typename FuncT> void interop_task(FuncT Func) {

    MInteropTask.reset(new detail::InteropTask(std::move(Func)));
    MCGType = detail::CG::INTEROP_TASK_CODEPLAY;
  }

  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param NumWorkItems is a range defining indexing space.
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                    KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offsets.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param NumWorkItems is a range defining indexing space.
  /// \param WorkItemOffset is an offset to be applied to each work item index.
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offsets.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param NDRange is a ND-range defining global and local sizes as
  /// well as offset.
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(kernel Kernel, nd_range<Dims> NDRange,
                    KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_nd_range<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NDRange));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    MCGType = detail::CG::KERNEL;
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>())
      extractArgsAndReqs();
    else
      StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
#endif
  }

  /// Hierarchical kernel invocation method of a kernel.
  ///
  /// This version of \c parallel_for_work_group takes two parameters
  /// representing the same kernel. The first one - \c Kernel - is a
  /// compiled form of the second one - \c kernelFunc, which is the source form
  /// of the kernel. The same source kernel can be compiled multiple times
  /// yielding multiple kernel class objects accessible via the \c program class
  /// interface.
  ///
  /// \param Kernel is a compiled SYCL kernel.
  /// \param NumWorkGroups is a range describing the number of work-groups in
  /// each dimension.
  /// \param KernelFunc is a lambda representing kernel.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  /// Hierarchical kernel invocation method of a kernel.
  ///
  /// This version of \c parallel_for_work_group takes two parameters
  /// representing the same kernel. The first one - \c Kernel - is a
  /// compiled form of the second one - \c kernelFunc, which is the source form
  /// of the kernel. The same source kernel can be compiled multiple times
  /// yielding multiple kernel class objects accessible via the \c program class
  /// interface.
  ///
  /// \param Kernel is a compiled SYCL kernel.
  /// \param NumWorkGroups is a range describing the number of work-groups in
  /// each dimension.
  /// \param WorkGroupSize is a range describing the size of work-groups in
  /// each dimension.
  /// \param KernelFunc is a lambda representing kernel.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               KernelType KernelFunc) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    StoreLambda<NameT, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif // __SYCL_DEVICE_ONLY__
  }

  // Explicit copy operations API

  /// Copies the contents of memory object accessed by Src into the memory
  /// pointed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a smart pointer to destination memory.
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

  /// Copies the contents of memory pointed by Src into the memory object
  /// accessed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a smart pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
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

  /// Copies the contents of memory object accessed by Src into the memory
  /// pointed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a pointer to destination memory.
  // TODO: support 0-dimensional and atomic accessors.
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

  /// Copies the contents of memory pointed by Src into the memory object
  /// accessed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
  // TODO: support 0-dimensional and atomic accessors.
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

  /// Copies the contents of memory object accessed by Src to the memory
  /// object accessed by Dst.
  ///
  /// Dst must have at least as many bytes as the range accessed by Src.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a destination SYCL accessor.
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
    assert(Dst.get_size() >= Src.get_size() &&
           "The destination accessor does not fit the copied memory.");
    if (copyAccToAccHelper(Src, Dst))
      return;
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

  /// Provides guarantees that the memory object accessed via Acc is updated
  /// on the host after command group object execution is complete.
  ///
  /// \param Acc is a SYCL accessor that needs to be updated on host.
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

  /// Fills memory pointed by accessor with the pattern given.
  ///
  /// If the operation is submitted to queue associated with OpenCL device and
  /// accessor points to one dimensional memory object then use special type for
  /// filling. Otherwise fill using regular kernel.
  ///
  /// \param Dst is a destination SYCL accessor.
  /// \param Pattern is a value to be used to fill the memory.
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

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  void memcpy(void *Dest, const void *Src, size_t Count) {
    throwIfActionIsCreated();
    MSrcPtr = const_cast<void *>(Src);
    MDstPtr = Dest;
    MLength = Count;
    MCGType = detail::CG::COPY_USM;
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  ///
  /// \param Dest is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  void memset(void *Dest, int Value, size_t Count) {
    throwIfActionIsCreated();
    MDstPtr = Dest;
    MPattern.push_back((char)Value);
    MLength = Count;
    MCGType = detail::CG::FILL_USM;
  }

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  void prefetch(const void *Ptr, size_t Count) {
    throwIfActionIsCreated();
    MDstPtr = const_cast<void *>(Ptr);
    MLength = Count;
    MCGType = detail::CG::PREFETCH_USM;
  }

private:
  shared_ptr_class<detail::queue_impl> MQueue;
  /// The storage for the arguments passed.
  /// We need to store a copy of values that are passed explicitly through
  /// set_arg, require and so on, because we need them to be alive after
  /// we exit the method they are passed in.
  vector_class<vector_class<char>> MArgsStorage;
  vector_class<detail::AccessorImplPtr> MAccStorage;
  vector_class<detail::LocalAccessorImplPtr> MLocalAccStorage;
  vector_class<shared_ptr_class<detail::stream_impl>> MStreamStorage;
  vector_class<shared_ptr_class<const void>> MSharedPtrStorage;
  /// The list of arguments for the kernel.
  vector_class<detail::ArgDesc> MArgs;
  /// The list of associated accessors with this handler.
  /// These accessors were created with this handler as argument or
  /// have become required for this handler via require method.
  vector_class<detail::ArgDesc> MAssociatedAccesors;
  /// The list of requirements to the memory objects for the scheduling.
  vector_class<detail::Requirement *> MRequirements;
  /// Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;
  string_class MKernelName;
  /// Storage for a sycl::kernel object.
  shared_ptr_class<detail::kernel_impl> MKernel;
  /// Type of the command group, e.g. kernel, fill.
  detail::CG::CGTYPE MCGType = detail::CG::NONE;
  /// Pointer to the source host memory or accessor(depending on command type).
  void *MSrcPtr = nullptr;
  /// Pointer to the dest host memory or accessor(depends on command type).
  void *MDstPtr = nullptr;
  /// Length to copy or fill (for USM operations).
  size_t MLength = 0;
  /// Pattern that is used to fill memory object in case command type is fill.
  vector_class<char> MPattern;
  /// Storage for a lambda or function object.
  unique_ptr_class<detail::HostKernelBase> MHostKernel;
  detail::OSModuleHandle MOSModuleHandle;
  // Storage for a lambda or function when using InteropTasks
  std::unique_ptr<detail::InteropTask> MInteropTask;
  /// The list of events that order this operation.
  vector_class<detail::EventImplPtr> MEvents;

  bool MIsHost = false;

  detail::code_location MCodeLoc = {};
  bool MIsFinalized = false;
  event MLastEvent;

  // Make queue_impl class friend to be able to call finalize method.
  friend class detail::queue_impl;
  // Make accessor class friend to keep the list of associated accessors.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  friend class accessor;
  friend device detail::getDeviceFromHandler(handler &);

  template <typename DataT, int Dimensions, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  friend class detail::image_accessor;
  // Make stream class friend to be able to keep the list of associated streams
  friend class stream;
  friend class detail::stream_impl;
  // Make reduction_impl friend to store buffers and arrays created for it
  // in handler from reduction_impl methods.
  template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
            access::placeholder IsPlaceholder>
  friend class intel::detail::reduction_impl;
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
