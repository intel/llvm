//==-------- handler.hpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/cg_types.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/handler_proxy.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/interop_handle.hpp>
#include <CL/sycl/item.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/kernel_handler.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/sampler.hpp>
#include <CL/sycl/stl.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

// SYCL_LANGUAGE_VERSION is 4 digit year followed by 2 digit revision
#if !SYCL_LANGUAGE_VERSION || SYCL_LANGUAGE_VERSION < 202001
#define __SYCL_NONCONST_FUNCTOR__
#endif

template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __fill;

template <typename T> class __usmfill;

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

// For unit testing purposes
class MockHandler;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration

class handler;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
namespace detail {

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

template <typename F, typename SuggestedArgType>
decltype(member_ptr_helper(&F::operator())) argument_helper(int);

template <typename F, typename SuggestedArgType>
SuggestedArgType argument_helper(...);

template <typename F, typename SuggestedArgType>
using lambda_arg_type = decltype(argument_helper<F, SuggestedArgType>(0));

// Used when parallel_for range is rounded-up.
template <typename Name> class __pf_kernel_wrapper;

template <typename Type> struct get_kernel_wrapper_name_t {
  using name = __pf_kernel_wrapper<Type>;
};

__SYCL_EXPORT device getDeviceFromHandler(handler &);

#if __SYCL_ID_QUERIES_FIT_IN_INT__
template <typename T> struct NotIntMsg;

template <int Dims> struct NotIntMsg<range<Dims>> {
  constexpr static const char *Msg =
      "Provided range is out of integer limits. Pass "
      "`-fno-sycl-id-queries-fit-in-int' to disable range check.";
};

template <int Dims> struct NotIntMsg<id<Dims>> {
  constexpr static const char *Msg =
      "Provided offset is out of integer limits. Pass "
      "`-fno-sycl-id-queries-fit-in-int' to disable offset check.";
};
#endif

#if __SYCL_ID_QUERIES_FIT_IN_INT__
template <typename T, typename ValT>
typename detail::enable_if_t<std::is_same<ValT, size_t>::value ||
                             std::is_same<ValT, unsigned long long>::value>
checkValueRangeImpl(ValT V) {
  static constexpr size_t Limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (V > Limit)
    throw runtime_error(NotIntMsg<T>::Msg, PI_INVALID_VALUE);
}
#endif

template <int Dims, typename T>
typename detail::enable_if_t<std::is_same<T, range<Dims>>::value ||
                             std::is_same<T, id<Dims>>::value>
checkValueRange(const T &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  for (size_t Dim = 0; Dim < Dims; ++Dim)
    checkValueRangeImpl<T>(V[Dim]);

  {
    unsigned long long Product = 1;
    for (size_t Dim = 0; Dim < Dims; ++Dim) {
      Product *= V[Dim];
      // check value now to prevent product overflow in the end
      checkValueRangeImpl<T>(Product);
    }
  }
#else
  (void)V;
#endif
}

template <int Dims>
void checkValueRange(const range<Dims> &R, const id<Dims> &O) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  checkValueRange<Dims>(R);
  checkValueRange<Dims>(O);

  for (size_t Dim = 0; Dim < Dims; ++Dim) {
    unsigned long long Sum = R[Dim] + O[Dim];

    checkValueRangeImpl<range<Dims>>(Sum);
  }
#else
  (void)R;
  (void)O;
#endif
}

template <int Dims, typename T>
typename detail::enable_if_t<std::is_same<T, nd_range<Dims>>::value>
checkValueRange(const T &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  checkValueRange<Dims>(V.get_global_range());
  checkValueRange<Dims>(V.get_local_range());
  checkValueRange<Dims>(V.get_offset());

  checkValueRange<Dims>(V.get_global_range(), V.get_offset());
#else
  (void)V;
#endif
}

} // namespace detail

namespace ext {
namespace oneapi {
namespace detail {
template <typename T, class BinaryOperation, int Dims, bool IsUSM,
          access::placeholder IsPlaceholder>
class reduction_impl;

using cl::sycl::detail::enable_if_t;
using cl::sycl::detail::queue_impl;

template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFunc(handler &CGH, KernelType KernelFunc, const range<Dims> &Range,
                size_t MaxWGSize, uint32_t NumConcurrentWorkGroups,
                Reduction &Redu);

template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<Reduction::has_atomic_add_float64>
reduCGFuncAtomic64(handler &CGH, KernelType KernelFunc,
                   const nd_range<Dims> &Range, Reduction &Redu);

template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<Reduction::has_fast_atomics>
reduCGFunc(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
           Reduction &Redu);

template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<!Reduction::has_fast_atomics>
reduCGFunc(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
           Reduction &Redu);

template <typename KernelName, typename KernelType, class Reduction>
enable_if_t<!Reduction::has_fast_atomics, size_t>
reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
              Reduction &Redu);

template <typename KernelName, typename KernelType, int Dims,
          typename... Reductions, size_t... Is>
void reduCGFunc(handler &CGH, KernelType KernelFunc,
                const nd_range<Dims> &Range,
                std::tuple<Reductions...> &ReduTuple,
                std::index_sequence<Is...>);

template <typename KernelName, typename KernelType, typename... Reductions,
          size_t... Is>
size_t reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...>);

template <typename KernelName, class Reduction>
std::enable_if_t<!Reduction::is_usm>
reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu);

template <typename KernelName, class Reduction>
std::enable_if_t<Reduction::is_usm>
reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu);

template <typename... Reduction, size_t... Is>
std::shared_ptr<event>
reduSaveFinalResultToUserMem(std::shared_ptr<detail::queue_impl> Queue,
                             bool IsHost, std::tuple<Reduction...> &ReduTuple,
                             std::index_sequence<Is...>);

template <typename Reduction, typename... RestT>
std::enable_if_t<!Reduction::is_usm>
reduSaveFinalResultToUserMemHelper(std::vector<event> &Events,
                                   std::shared_ptr<detail::queue_impl> Queue,
                                   bool IsHost, Reduction &Redu, RestT... Rest);

__SYCL_EXPORT uint32_t
reduGetMaxNumConcurrentWorkGroups(std::shared_ptr<queue_impl> Queue);

__SYCL_EXPORT size_t reduGetMaxWGSize(std::shared_ptr<queue_impl> Queue,
                                      size_t LocalMemBytesPerWorkItem);

template <typename... ReductionT, size_t... Is>
size_t reduGetMemPerWorkItem(std::tuple<ReductionT...> &ReduTuple,
                             std::index_sequence<Is...>);

template <typename TupleT, std::size_t... Is>
std::tuple<std::tuple_element_t<Is, TupleT>...>
tuple_select_elements(TupleT Tuple, std::index_sequence<Is...>);

template <typename FirstT, typename... RestT> struct AreAllButLastReductions;

} // namespace detail
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}

/// Command group handler class.
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
///
/// \sa queue
/// \sa program
/// \sa kernel
///
/// \ingroup sycl_api
class __SYCL_EXPORT handler {
private:
  /// Constructs SYCL handler from queue.
  ///
  /// \param Queue is a SYCL queue.
  /// \param IsHost indicates if this handler is created for SYCL host device.
  handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost);

  /// Stores copy of Arg passed to the MArgsStorage.
  template <typename T, typename F = typename detail::remove_const_t<
                            typename detail::remove_reference_t<T>>>
  F *storePlainArg(T &&Arg) {
    MArgsStorage.emplace_back(sizeof(T));
    auto Storage = reinterpret_cast<F *>(MArgsStorage.back().data());
    *Storage = Arg;
    return Storage;
  }

  void setType(detail::CG::CGTYPE Type) {
    constexpr detail::CG::CG_VERSION Version = detail::CG::CG_VERSION::V1;
    MCGType = static_cast<detail::CG::CGTYPE>(
        getVersionedCGType(Type, static_cast<int>(Version)));
  }

  detail::CG::CGTYPE getType() {
    return static_cast<detail::CG::CGTYPE>(getUnversionedCGType(MCGType));
  }

  void throwIfActionIsCreated() {
    if (detail::CG::None != getType())
      throw sycl::runtime_error("Attempt to set multiple actions for the "
                                "command group. Command group must consist of "
                                "a single kernel or explicit memory operation.",
                                CL_INVALID_OPERATION);
  }

  /// Extracts and prepares kernel arguments from the lambda using integration
  /// header.
  /// TODO replace with the version below once ABI breaking changes are allowed.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs);

  /// Extracts and prepares kernel arguments from the lambda using integration
  /// header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs,
                               bool IsESIMD);

  /// Extracts and prepares kernel arguments set via set_arg(s).
  void extractArgsAndReqs();

  /// TODO replace with the version below once ABI breaking changes are allowed.
  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource);

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource, bool IsESIMD);

  /// \return a string containing name of SYCL kernel.
  std::string getKernelName();

  template <typename LambdaNameT> bool lambdaAndKernelHaveEqualName() {
    // TODO It is unclear a kernel and a lambda/functor must to be equal or not
    // for parallel_for with sycl::kernel and lambda/functor together
    // Now if they are equal we extract argumets from lambda/functor for the
    // kernel. Else it is necessary use set_atg(s) for resolve the order and
    // values of arguments for the kernel.
    assert(MKernel && "MKernel is not initialized");
    const std::string LambdaName = detail::KernelInfo<LambdaNameT>::getName();
    const std::string KernelName = getKernelName();
    return LambdaName == KernelName;
  }

  /// Saves the location of user's code passed in \param CodeLoc for future
  /// usage in finalize() method.
  void saveCodeLoc(detail::code_location CodeLoc) { MCodeLoc = CodeLoc; }

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
  void addStream(const std::shared_ptr<detail::stream_impl> &Stream) {
    MStreamStorage.push_back(Stream);
  }

  /// Saves buffers created by handling reduction feature in handler.
  /// They are then forwarded to command group and destroyed only after
  /// the command group finishes the work on device/host.
  /// The 'MSharedPtrStorage' suits that need.
  ///
  /// @param ReduObj is a pointer to object that must be stored.
  void addReduction(const std::shared_ptr<const void> &ReduObj) {
    MSharedPtrStorage.push_back(ReduObj);
  }

  ~handler() = default;

  bool is_host() { return MIsHost; }

  void associateWithHandler(detail::AccessorBaseHost *AccBase,
                            access::target AccTarget);

  // Recursively calls itself until arguments pack is fully processed.
  // The version for regular(standard layout) argument.
  template <typename T, typename... Ts>
  void setArgsHelper(int ArgIndex, T &&Arg, Ts &&... Args) {
    set_arg(ArgIndex, std::move(Arg));
    setArgsHelper(++ArgIndex, std::move(Args)...);
  }

  void setArgsHelper(int) {}

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
  typename detail::enable_if_t<AccessTarget != access::target::local, void>
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
    auto StoredArg = static_cast<void *>(storePlainArg(Arg));

    if (!std::is_same<cl_mem, T>::value && std::is_pointer<T>::value) {
      MArgs.emplace_back(detail::kernel_param_kind_t::kind_pointer, StoredArg,
                         sizeof(T), ArgIndex);
    } else {
      MArgs.emplace_back(detail::kernel_param_kind_t::kind_std_layout,
                         StoredArg, sizeof(T), ArgIndex);
    }
  }

  void setArgHelper(int ArgIndex, sampler &&Arg) {
    auto StoredArg = static_cast<void *>(storePlainArg(Arg));
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

  /// Stores lambda to the template-free object
  ///
  /// Also initializes kernel name, list of arguments and requirements using
  /// information from the integration header.
  ///
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName, typename KernelType, int Dims,
            typename LambdaArgType>
  void StoreLambda(KernelType KernelFunc) {
    if (detail::isKernelLambdaCallableWithKernelHandler<KernelType,
                                                        LambdaArgType>() &&
        MIsHost) {
      throw cl::sycl::feature_not_supported(
          "kernel_handler is not yet supported by host device.",
          PI_INVALID_OPERATION);
    }
    MHostKernel.reset(
        new detail::HostKernel<KernelType, LambdaArgType, Dims, KernelName>(
            KernelFunc));

    using KI = sycl::detail::KernelInfo<KernelName>;
    // Empty name indicates that the compilation happens without integration
    // header, so don't perform things that require it.
    if (KI::getName() != nullptr && KI::getName()[0] != '\0') {
      // TODO support ESIMD in no-integration-header case too.
      MArgs.clear();
      extractArgsAndReqsFromLambda(MHostKernel->getPtr(), KI::getNumParams(),
                                   &KI::getParamDesc(0), KI::isESIMD());
      MKernelName = KI::getName();
      MOSModuleHandle = detail::OSUtil::getOSModuleHandle(KI::getName());
    } else {
      // In case w/o the integration header it is necessary to process
      // accessors from the list(which are associated with this handler) as
      // arguments.
      MArgs = std::move(MAssociatedAccesors);
    }

    // If the kernel lambda is callable with a kernel_handler argument, manifest
    // the associated kernel handler.
    if (detail::isKernelLambdaCallableWithKernelHandler<KernelType,
                                                        LambdaArgType>()) {
      getOrInsertHandlerKernelBundle(/*Insert=*/true);
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

  // TODO: Delete these functions when ABI breaking changes are allowed.
  // Currently these functions are unused but they are static members of
  // the exported class 'handler' and has got into sycl library some time ago
  // and must stay there for a while.
  static id<1> getDelinearizedIndex(const range<1> Range, const size_t Index) {
    return detail::getDelinearizedId(Range, Index);
  }
  static id<2> getDelinearizedIndex(const range<2> Range, const size_t Index) {
    return detail::getDelinearizedId(Range, Index);
  }
  static id<3> getDelinearizedIndex(const range<3> Range, const size_t Index) {
    return detail::getDelinearizedId(Range, Index);
  }

  /// Handles some special cases of the copy operation from one accessor
  /// to another accessor. Returns true if the copy is handled here.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a destination SYCL accessor.
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

    range<1> LinearizedRange(Src.size());
    parallel_for<class __copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc,
                                     TDst, DimDst, ModeDst, TargetDst,
                                     IsPHSrc, IsPHDst>>
                                     (LinearizedRange, [=](id<1> Id) {
      size_t Index = Id[0];
      id<DimSrc> SrcId = detail::getDelinearizedId(Src.get_range(), Index);
      id<DimDst> DstId = detail::getDelinearizedId(Dst.get_range(), Index);
      Dst[DstId] = Src[SrcId];
    });
    return true;
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
      *(Dst.get_pointer()) = *(Src.get_pointer());
    });
    return true;
  }

#ifndef __SYCL_DEVICE_ONLY__
  /// Copies the content of memory object accessed by Src into the memory
  /// pointed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a pointer to destination memory.
  template <typename TSrc, typename TDst, int Dim, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPH>
  detail::enable_if_t<(Dim > 0)>
  copyAccToPtrHost(accessor<TSrc, Dim, AccMode, AccTarget, IsPH> Src,
                   TDst *Dst) {
    range<Dim> Range = Src.get_range();
    parallel_for<class __copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>
        (Range, [=](id<Dim> Index) {
      const size_t LinearIndex = detail::getLinearIndex(Index, Range);
      using TSrcNonConst = typename detail::remove_const_t<TSrc>;
      (reinterpret_cast<TSrcNonConst *>(Dst))[LinearIndex] = Src[Index];
    });
  }

  /// Copies 1 element accessed by 0-dimensional accessor Src into the memory
  /// pointed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a pointer to destination memory.
  template <typename TSrc, typename TDst, int Dim, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPH>
  detail::enable_if_t<Dim == 0>
  copyAccToPtrHost(accessor<TSrc, Dim, AccMode, AccTarget, IsPH> Src,
                   TDst *Dst) {
    single_task<class __copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>
        ([=]() {
      using TSrcNonConst = typename detail::remove_const_t<TSrc>;
      *(reinterpret_cast<TSrcNonConst *>(Dst)) = *(Src.get_pointer());
    });
  }

  /// Copies the memory pointed by Src into the memory accessed by Dst.
  ///
  /// \param Src is a pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
  template <typename TSrc, typename TDst, int Dim, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPH>
  detail::enable_if_t<(Dim > 0)>
  copyPtrToAccHost(TSrc *Src,
                   accessor<TDst, Dim, AccMode, AccTarget, IsPH> Dst) {
    range<Dim> Range = Dst.get_range();
    parallel_for<class __copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>
        (Range, [=](id<Dim> Index) {
      const size_t LinearIndex = detail::getLinearIndex(Index, Range);
      Dst[Index] = (reinterpret_cast<const TDst *>(Src))[LinearIndex];
    });
  }

  /// Copies 1 element pointed by Src to memory accessed by 0-dimensional
  /// accessor Dst.
  ///
  /// \param Src is a pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
  template <typename TSrc, typename TDst, int Dim, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPH>
  detail::enable_if_t<Dim == 0>
  copyPtrToAccHost(TSrc *Src,
                   accessor<TDst, Dim, AccMode, AccTarget, IsPH> Dst) {
    single_task<class __copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>
        ([=]() {
      *(Dst.get_pointer()) = *(reinterpret_cast<const TDst *>(Src));
    });
  }
#endif // __SYCL_DEVICE_ONLY__

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

  constexpr static bool isValidModeForSourceAccessor(access::mode AccessMode) {
    return AccessMode == access::mode::read ||
           AccessMode == access::mode::read_write;
  }

  constexpr static bool
  isValidModeForDestinationAccessor(access::mode AccessMode) {
    return AccessMode == access::mode::write ||
           AccessMode == access::mode::read_write ||
           AccessMode == access::mode::discard_write ||
           AccessMode == access::mode::discard_read_write;
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
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for_lambda_impl(range<Dims> NumWorkItems,
                                KernelType KernelFunc) {
    throwIfActionIsCreated();
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;

    // If 1D kernel argument is an integral type, convert it to sycl::item<1>
    using TransformedArgType =
        typename std::conditional<std::is_integral<LambdaArgType>::value &&
                                      Dims == 1,
                                  item<Dims>, LambdaArgType>::type;
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;

    // Range rounding can be disabled by the user.
    // Range rounding is not done on the host device.
    // Range rounding is supported only for newer SYCL standards.
#if !defined(__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__) &&                  \
    !defined(DPCPP_HOST_DEVICE_OPENMP) &&                                      \
    !defined(DPCPP_HOST_DEVICE_PERF_NATIVE) && SYCL_LANGUAGE_VERSION >= 202001
    // Range should be a multiple of this for reasonable performance.
    size_t MinFactorX = 16;
    // Range should be a multiple of this for improved performance.
    size_t GoodFactorX = 32;
    // Range should be at least this to make rounding worthwhile.
    size_t MinRangeX = 1024;

    // Check if rounding parameters have been set through environment:
    // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
    this->GetRangeRoundingSettings(MinFactorX, GoodFactorX, MinRangeX);

    // Disable the rounding-up optimizations under these conditions:
    // 1. The env var SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING is set.
    // 2. The kernel is provided via an interoperability method.
    // 3. The API "this_item" is used inside the kernel.
    // 4. The range is already a multiple of the rounding factor.
    //
    // Cases 2 and 3 could be supported with extra effort.
    // As an optimization for the common case it is an
    // implementation choice to not support those scenarios.
    // Note that "this_item" is a free function, i.e. not tied to any
    // specific id or item. When concurrent parallel_fors are executing
    // on a device it is difficult to tell which parallel_for the call is
    // being made from. One could replicate portions of the
    // call-graph to make this_item calls kernel-specific but this is
    // not considered worthwhile.

    // Get the kernel name to check condition 2.
    std::string KName = typeid(NameT *).name();
    using KI = detail::KernelInfo<KernelName>;
    bool DisableRounding =
        this->DisableRangeRounding() ||
        (KI::getName() == nullptr || KI::getName()[0] == '\0') ||
        (KI::callsThisItem());

    // Perform range rounding if rounding-up is enabled
    // and there are sufficient work-items to need rounding
    // and the user-specified range is not a multiple of a "good" value.
    if (!DisableRounding && (NumWorkItems[0] >= MinRangeX) &&
        (NumWorkItems[0] % MinFactorX != 0)) {
      // It is sufficient to round up just the first dimension.
      // Multiplying the rounded-up value of the first dimension
      // by the values of the remaining dimensions (if any)
      // will yield a rounded-up value for the total range.
      size_t NewValX =
          ((NumWorkItems[0] + GoodFactorX - 1) / GoodFactorX) * GoodFactorX;
      if (this->RangeRoundingTrace())
        std::cout << "parallel_for range adjusted from " << NumWorkItems[0]
                  << " to " << NewValX << std::endl;

      using NameWT = typename detail::get_kernel_wrapper_name_t<NameT>::name;
      auto Wrapper =
          getRangeRoundedKernelLambda<NameWT, TransformedArgType, Dims>(
              KernelFunc, NumWorkItems);

      using KName = std::conditional_t<std::is_same<KernelType, NameT>::value,
                                       decltype(Wrapper), NameWT>;

      range<Dims> AdjustedRange = NumWorkItems;
      AdjustedRange.set_range_dim0(NewValX);
      kernel_parallel_for_wrapper<KName, TransformedArgType>(Wrapper);
#ifndef __SYCL_DEVICE_ONLY__
      detail::checkValueRange<Dims>(AdjustedRange);
      MNDRDesc.set(std::move(AdjustedRange));
      StoreLambda<KName, decltype(Wrapper), Dims, TransformedArgType>(
          std::move(Wrapper));
      setType(detail::CG::Kernel);
#endif
    } else
#endif // !__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ &&                     \
       // !DPCPP_HOST_DEVICE_OPENMP && !DPCPP_HOST_DEVICE_PERF_NATIVE &&       \
       // SYCL_LANGUAGE_VERSION >= 202001
    {
      (void)NumWorkItems;
      kernel_parallel_for_wrapper<NameT, TransformedArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
      detail::checkValueRange<Dims>(NumWorkItems);
      MNDRDesc.set(std::move(NumWorkItems));
      StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
          std::move(KernelFunc));
      setType(detail::CG::Kernel);
#endif
    }
  }

  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object. The kernel
  /// invocation method has no functors and cannot be called on host.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param Kernel is a SYCL kernel function.
  template <int Dims>
  void parallel_for_impl(range<Dims> NumWorkItems, kernel Kernel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kernel);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NumWorkItems);
    MNDRDesc.set(std::move(NumWorkItems));
    setType(detail::CG::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
  }

#ifdef SYCL_LANGUAGE_VERSION
#define __SYCL_KERNEL_ATTR__ __attribute__((sycl_kernel))
#else
#define __SYCL_KERNEL_ATTR__
#endif
  // NOTE: the name of this function - "kernel_single_task" - is used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_single_task(KernelType KernelFunc) {
#else
  kernel_single_task(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc();
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of this function - "kernel_single_task" - is used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_single_task(KernelType KernelFunc, kernel_handler KH) {
#else
  kernel_single_task(const KernelType &KernelFunc, kernel_handler KH) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  // NOTE: the name of these functions - "kernel_parallel_for" - are used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for(KernelType KernelFunc) {
#else
  kernel_parallel_for(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of these functions - "kernel_parallel_for" - are used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for(KernelType KernelFunc, kernel_handler KH) {
#else
  kernel_parallel_for(const KernelType &KernelFunc, kernel_handler KH) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()), KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  // NOTE: the name of this function - "kernel_parallel_for_work_group" - is
  // used by the Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for_work_group(KernelType KernelFunc) {
#else
  kernel_parallel_for_work_group(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of this function - "kernel_parallel_for_work_group" - is
  // used by the Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType>
  __SYCL_KERNEL_ATTR__ void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for_work_group(KernelType KernelFunc, kernel_handler KH) {
#else
  kernel_parallel_for_work_group(const KernelType &KernelFunc,
                                 kernel_handler KH) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()), KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  // Wrappers for kernel_*** functions above with and without support of
  // additional kernel_handler argument.

  // NOTE: to support kernel_handler argument in kernel lambdas, only
  // kernel_***_wrapper functions must be called in this code

  // Wrappers for kernel_single_task(...)

  template <typename KernelName, typename KernelType>
  void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_single_task_wrapper(KernelType KernelFunc) {
#else
  kernel_single_task_wrapper(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
    if constexpr (detail::isKernelLambdaCallableWithKernelHandler<
                      KernelType>()) {
      kernel_handler KH;
      kernel_single_task<KernelName>(KernelFunc, KH);
    } else {
      kernel_single_task<KernelName>(KernelFunc);
    }
  }

  // Wrappers for kernel_parallel_for(...)

  template <typename KernelName, typename ElementType, typename KernelType>
  void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for_wrapper(KernelType KernelFunc) {
#else
  kernel_parallel_for_wrapper(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
    if constexpr (detail::isKernelLambdaCallableWithKernelHandler<
                      KernelType, ElementType>()) {
      kernel_handler KH;
      kernel_parallel_for<KernelName, ElementType>(KernelFunc, KH);
    } else {
      kernel_parallel_for<KernelName, ElementType>(KernelFunc);
    }
  }

  // Wrappers for kernel_parallel_for_work_group(...)

  template <typename KernelName, typename ElementType, typename KernelType>
  void
#ifdef __SYCL_NONCONST_FUNCTOR__
  kernel_parallel_for_work_group_wrapper(KernelType KernelFunc) {
#else
  kernel_parallel_for_work_group_wrapper(const KernelType &KernelFunc) {
#endif
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
    if constexpr (detail::isKernelLambdaCallableWithKernelHandler<
                      KernelType, ElementType>()) {
      kernel_handler KH;
      kernel_parallel_for_work_group<KernelName, ElementType>(KernelFunc, KH);
    } else {
      kernel_parallel_for_work_group<KernelName, ElementType>(KernelFunc);
    }
  }

  std::shared_ptr<detail::kernel_bundle_impl>
  getOrInsertHandlerKernelBundle(bool Insert) const;

  void setHandlerKernelBundle(
      const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr);

  template <typename FuncT>
  detail::enable_if_t<
      detail::check_fn_signature<detail::remove_reference_t<FuncT>,
                                 void()>::value ||
      detail::check_fn_signature<detail::remove_reference_t<FuncT>,
                                 void(interop_handle)>::value>
  host_task_impl(FuncT &&Func) {
    throwIfActionIsCreated();

    MNDRDesc.set(range<1>(1));
    MArgs = std::move(MAssociatedAccesors);

    MHostTask.reset(new detail::HostTask(std::move(Func)));

    setType(detail::CG::CodeplayHostTask);
  }

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

#if __cplusplus > 201402L
  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type Value) {

    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/true);

    detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
        KernelBundleImplPtr)
        .set_specialization_constant<SpecName>(Value);
  }

  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {

    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/true);

    return detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
               KernelBundleImplPtr)
        .get_specialization_constant<SpecName>();
  }

#endif

  void
  use_kernel_bundle(const kernel_bundle<bundle_state::executable> &ExecBundle) {
    setHandlerKernelBundle(detail::getSyclObjImpl(ExecBundle));
  }

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
#ifndef __SYCL_DEVICE_ONLY__
    associateWithHandler(&Acc, AccTarget);
#else
    (void)Acc;
#endif
  }

  /// Registers event dependencies on this command group.
  ///
  /// \param Event is a valid SYCL event to wait on.
  void depends_on(event Event) {
    MEvents.push_back(detail::getSyclObjImpl(Event));
  }

  /// Registers event dependencies on this command group.
  ///
  /// \param Events is a vector of valid SYCL events to wait on.
  void depends_on(const std::vector<event> &Events) {
    for (const event &Event : Events) {
      MEvents.push_back(detail::getSyclObjImpl(Event));
    }
  }

  template <typename T>
  using remove_cv_ref_t =
      typename detail::remove_cv_t<detail::remove_reference_t<T>>;

  template <typename U, typename T>
  using is_same_type = std::is_same<remove_cv_ref_t<U>, remove_cv_ref_t<T>>;

  template <typename T> struct ShouldEnableSetArg {
    static constexpr bool value =
        std::is_trivially_copyable<detail::remove_reference_t<T>>::value
#if SYCL_LANGUAGE_VERSION && SYCL_LANGUAGE_VERSION <= 201707
            && std::is_standard_layout<detail::remove_reference_t<T>>::value
#endif
        || is_same_type<sampler, T>::value // Sampler
        || (!is_same_type<cl_mem, T>::value &&
            std::is_pointer<remove_cv_ref_t<T>>::value) // USM
        || is_same_type<cl_mem, T>::value;              // Interop
  };

  /// Sets argument for OpenCL interoperability kernels.
  ///
  /// Registers Arg passed as argument # ArgIndex.
  ///
  /// \param ArgIndex is a positional number of argument to be set.
  /// \param Arg is an argument value to be set.
  template <typename T>
  typename detail::enable_if_t<ShouldEnableSetArg<T>::value, void>
  set_arg(int ArgIndex, T &&Arg) {
    setArgHelper(ArgIndex, std::move(Arg));
  }

  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  void
  set_arg(int ArgIndex,
          accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder> Arg) {
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
#ifdef __SYCL_NONCONST_FUNCTOR__
  void single_task(KernelType KernelFunc) {
#else
  void single_task(const KernelType &KernelFunc) {
#endif
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    kernel_single_task_wrapper<NameT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant.
    MNDRDesc.set(range<1>{1});

    StoreLambda<NameT, KernelType, /*Dims*/ 0, void>(KernelFunc);
    setType(detail::CG::Kernel);
#endif
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
#ifdef __SYCL_NONCONST_FUNCTOR__
  void parallel_for(range<1> NumWorkItems, KernelType KernelFunc) {
#else
  void parallel_for(range<1> NumWorkItems, const KernelType &KernelFunc) {
#endif
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
#ifdef __SYCL_NONCONST_FUNCTOR__
  void parallel_for(range<2> NumWorkItems, KernelType KernelFunc) {
#else
  void parallel_for(range<2> NumWorkItems, const KernelType &KernelFunc) {
#endif
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
#ifdef __SYCL_NONCONST_FUNCTOR__
  void parallel_for(range<3> NumWorkItems, KernelType KernelFunc) {
#else
  void parallel_for(range<3> NumWorkItems, const KernelType &KernelFunc) {
#endif
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  /// Defines and invokes a SYCL kernel on host device.
  ///
  /// \param Func is a SYCL kernel function defined by lambda function or a
  /// named function object type.
  template <typename FuncT> void run_on_host_intel(FuncT Func) {
    throwIfActionIsCreated();
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant
    MNDRDesc.set(range<1>{1});

    MArgs = std::move(MAssociatedAccesors);
    MHostKernel.reset(
        new detail::HostKernel<FuncT, void, 1, void>(std::move(Func)));
    setType(detail::CG::RunOnHostIntel);
  }

  template <typename FuncT>
  __SYCL2020_DEPRECATED(
      "codeplay_host_task() is deprecated, use host_task() instead")
  detail::enable_if_t<
      detail::check_fn_signature<detail::remove_reference_t<FuncT>,
                                 void()>::value ||
      detail::check_fn_signature<
          detail::remove_reference_t<FuncT>,
          void(interop_handle)>::value> codeplay_host_task(FuncT Func) {
    host_task_impl(Func);
  }

  /// Enqueues a command to the SYCL runtime to invoke \p Func once.
  template <typename FuncT>
  detail::enable_if_t<
      detail::check_fn_signature<detail::remove_reference_t<FuncT>,
                                 void()>::value ||
      detail::check_fn_signature<detail::remove_reference_t<FuncT>,
                                 void(interop_handle)>::value>
  host_task(FuncT &&Func) {
    host_task_impl(Func);
  }

// replace _KERNELFUNCPARAM(KernelFunc) with   KernelType KernelFunc
//                                     or     const KernelType &KernelFunc
#ifdef __SYCL_NONCONST_FUNCTOR__
#define _KERNELFUNCPARAM(a) KernelType a
#else
#define _KERNELFUNCPARAM(a) const KernelType &a
#endif

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
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    (void)NumWorkItems;
    (void)WorkItemOffset;
    kernel_parallel_for_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkItems, WorkItemOffset);
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
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
  void parallel_for(nd_range<Dims> ExecutionRange,
                    _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
    (void)ExecutionRange;
    kernel_parallel_for_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(ExecutionRange);
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
#endif
  }

  /// Defines and invokes a SYCL kernel function for the specified nd_range.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id for indexing in the indexing
  /// space defined by range \p Range.
  /// The parameter \p Redu contains the object creted by the reduction()
  /// function and defines the type and operation used in the corresponding
  /// argument of 'reducer' type passed to lambda/functor function.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  void parallel_for(range<Dims> Range, Reduction Redu,
                    _KERNELFUNCPARAM(KernelFunc)) {
    shared_ptr_class<detail::queue_impl> QueueCopy = MQueue;

    // Before running the kernels, check that device has enough local memory
    // to hold local arrays required for the tree-reduction algorithm.
    constexpr bool IsTreeReduction =
        !Reduction::has_fast_reduce && !Reduction::has_fast_atomics;
    size_t OneElemSize =
        IsTreeReduction ? sizeof(typename Reduction::result_type) : 0;
    uint32_t NumConcurrentWorkGroups =
#ifdef __SYCL_REDUCTION_NUM_CONCURRENT_WORKGROUPS
        __SYCL_REDUCTION_NUM_CONCURRENT_WORKGROUPS;
#else
        ext::oneapi::detail::reduGetMaxNumConcurrentWorkGroups(MQueue);
#endif
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it is safer to use queries to the kernel pre-compiled
    // for the device.
    size_t MaxWGSize =
        ext::oneapi::detail::reduGetMaxWGSize(MQueue, OneElemSize);
    ext::oneapi::detail::reduCGFunc<KernelName>(
        *this, KernelFunc, Range, MaxWGSize, NumConcurrentWorkGroups, Redu);
    if (Reduction::is_usm ||
        (Reduction::has_fast_atomics && Redu.initializeToIdentity()) ||
        (!Reduction::has_fast_atomics && Redu.hasUserDiscardWriteAccessor())) {
      this->finalize();
      handler CopyHandler(QueueCopy, MIsHost);
      CopyHandler.saveCodeLoc(MCodeLoc);
      ext::oneapi::detail::reduSaveFinalResultToUserMem<KernelName>(CopyHandler,
                                                                    Redu);
      MLastEvent = CopyHandler.finalize();
    }
  }

  /// Implements parallel_for() accepting nd_range \p Range and one reduction
  /// object. This version uses fast sycl::atomic operations to update reduction
  /// variable at the end of each work-group work.
  //
  // If the reduction variable must be initialized with the identity value
  // before the kernel run, then an additional working accessor is created,
  // initialized with the identity value and used in the kernel. That working
  // accessor is then copied to user's accessor or USM pointer after
  // the kernel run.
  // For USM pointers without initialize_to_identity properties the same scheme
  // with working accessor is used as re-using user's USM pointer in the kernel
  // would require creation of another variant of user's kernel, which does not
  // seem efficient.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  detail::enable_if_t<Reduction::has_fast_atomics>
  parallel_for(nd_range<Dims> Range, Reduction Redu,
               _KERNELFUNCPARAM(KernelFunc)) {
    std::shared_ptr<detail::queue_impl> QueueCopy = MQueue;
    ext::oneapi::detail::reduCGFunc<KernelName>(*this, KernelFunc, Range, Redu);

    if (Reduction::is_usm || Redu.initializeToIdentity()) {
      this->finalize();
      handler CopyHandler(QueueCopy, MIsHost);
      CopyHandler.saveCodeLoc(MCodeLoc);
      ext::oneapi::detail::reduSaveFinalResultToUserMem<KernelName>(CopyHandler,
                                                                    Redu);
      MLastEvent = CopyHandler.finalize();
    }
  }

  /// Implements parallel_for() accepting nd_range \p Range and one reduction
  /// object. This version is a specialization for the add operator.
  /// It performs runtime checks for device aspect "atomic64"; if found, fast
  /// sycl::atomic_ref operations are used to update the reduction at the
  /// end of each work-group work.  Otherwise the default implementation is
  /// used.
  //
  // If the reduction variable must be initialized with the identity value
  // before the kernel run, then an additional working accessor is created,
  // initialized with the identity value and used in the kernel. That working
  // accessor is then copied to user's accessor or USM pointer after
  // the kernel run.
  // For USM pointers without initialize_to_identity properties the same scheme
  // with working accessor is used as re-using user's USM pointer in the kernel
  // would require creation of another variant of user's kernel, which does not
  // seem efficient.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  detail::enable_if_t<Reduction::has_atomic_add_float64>
  parallel_for(nd_range<Dims> Range, Reduction Redu,
               _KERNELFUNCPARAM(KernelFunc)) {

    shared_ptr_class<detail::queue_impl> QueueCopy = MQueue;
    device D = detail::getDeviceFromHandler(*this);

    if (D.has(aspect::atomic64)) {

      ext::oneapi::detail::reduCGFuncAtomic64<KernelName>(*this, KernelFunc,
                                                          Range, Redu);

      if (Reduction::is_usm || Redu.initializeToIdentity()) {
        this->finalize();
        handler CopyHandler(QueueCopy, MIsHost);
        CopyHandler.saveCodeLoc(MCodeLoc);
        ext::oneapi::detail::reduSaveFinalResultToUserMem<KernelName>(
            CopyHandler, Redu);
        MLastEvent = CopyHandler.finalize();
      }
    } else {
      parallel_for_Impl<KernelName>(Range, Redu, KernelFunc);
    }
  }

  /// Defines and invokes a SYCL kernel function for the specified nd_range.
  /// Performs reduction operation specified in \p Redu.
  ///
  /// The SYCL kernel function is defined as a lambda function or a named
  /// function object type and given an id or item for indexing in the indexing
  /// space defined by \p Range.
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// TODO: Support HOST. The kernels called by this parallel_for() may use
  /// some functionality that is not yet supported on HOST such as:
  /// barrier(), and ext::oneapi::reduce() that also may be used in more
  /// optimized implementations waiting for their turn of code-review.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  detail::enable_if_t<!Reduction::has_fast_atomics &&
                      !Reduction::has_atomic_add_float64>
  parallel_for(nd_range<Dims> Range, Reduction Redu,
               _KERNELFUNCPARAM(KernelFunc)) {

    parallel_for_Impl<KernelName>(Range, Redu, KernelFunc);
  }

  template <typename KernelName, typename KernelType, int Dims,
            typename Reduction>
  detail::enable_if_t<!Reduction::has_fast_atomics>
  parallel_for_Impl(nd_range<Dims> Range, Reduction Redu,
                    KernelType KernelFunc) {
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

    // Before running the kernels, check that device has enough local memory
    // to hold local arrays that may be required for the reduction algorithm.
    // TODO: If the work-group-size is limited by the local memory, then
    // a special version of the main kernel may be created. The one that would
    // not use local accessors, which means it would not do the reduction in
    // the main kernel, but simply generate Range.get_global_range.size() number
    // of partial sums, leaving the reduction work to the additional/aux
    // kernels.
    constexpr bool HFR = Reduction::has_fast_reduce;
    size_t OneElemSize = HFR ? 0 : sizeof(typename Reduction::result_type);
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it may be safer to use queries to the kernel compiled
    // for the device.
    size_t MaxWGSize =
        ext::oneapi::detail::reduGetMaxWGSize(MQueue, OneElemSize);
    if (Range.get_local_range().size() > MaxWGSize)
      throw sycl::runtime_error("The implementation handling parallel_for with"
                                " reduction requires work group size not bigger"
                                " than " +
                                    std::to_string(MaxWGSize),
                                PI_INVALID_WORK_GROUP_SIZE);

    // 1. Call the kernel that includes user's lambda function.
    ext::oneapi::detail::reduCGFunc<KernelName>(*this, KernelFunc, Range, Redu);
    std::shared_ptr<detail::queue_impl> QueueCopy = MQueue;
    this->finalize();

    // 2. Run the additional kernel as many times as needed to reduce
    // all partial sums into one scalar.

    // TODO: Create a special slow/sequential version of the kernel that would
    // handle the reduction instead of reporting an assert below.
    if (MaxWGSize <= 1)
      throw sycl::runtime_error("The implementation handling parallel_for with "
                                "reduction requires the maximal work group "
                                "size to be greater than 1 to converge. "
                                "The maximal work group size depends on the "
                                "device and the size of the objects passed to "
                                "the reduction.",
                                PI_INVALID_WORK_GROUP_SIZE);
    size_t NWorkItems = Range.get_group_range().size();
    while (NWorkItems > 1) {
      handler AuxHandler(QueueCopy, MIsHost);
      AuxHandler.saveCodeLoc(MCodeLoc);

      NWorkItems = ext::oneapi::detail::reduAuxCGFunc<KernelName, KernelType>(
          AuxHandler, NWorkItems, MaxWGSize, Redu);
      MLastEvent = AuxHandler.finalize();
    } // end while (NWorkItems > 1)

    if (Reduction::is_usm || Redu.hasUserDiscardWriteAccessor()) {
      handler CopyHandler(QueueCopy, MIsHost);
      CopyHandler.saveCodeLoc(MCodeLoc);
      ext::oneapi::detail::reduSaveFinalResultToUserMem<KernelName>(CopyHandler,
                                                                    Redu);
      MLastEvent = CopyHandler.finalize();
    }
  }

  // This version of parallel_for may handle one or more reductions packed in
  // \p Rest argument. Note thought that the last element in \p Rest pack is
  // the kernel function.
  // TODO: this variant is currently enabled for 2+ reductions only as the
  // versions handling 1 reduction variable are more efficient right now.
  //
  // Algorithm:
  // 1) discard_write accessor (DWAcc), InitializeToIdentity = true:
  //    a) Create uninitialized buffer and read_write accessor (RWAcc).
  //    b) discard-write partial sums to RWAcc.
  //    c) Repeat the steps (a) and (b) to get one final sum.
  //    d) Copy RWAcc to DWAcc.
  // 2) read_write accessor (RWAcc), InitializeToIdentity = false:
  //    a) Create new uninitialized buffer (if #work-groups > 1) and RWAcc or
  //       re-use user's RWAcc (if #work-groups is 1).
  //    b) discard-write to RWAcc (#WG > 1), or update-write (#WG == 1).
  //    c) Repeat the steps (a) and (b) to get one final sum.
  // 3) read_write accessor (RWAcc), InitializeToIdentity = true:
  //    a) Create new uninitialized buffer (if #work-groups > 1) and RWAcc or
  //       re-use user's RWAcc (if #work-groups is 1).
  //    b) discard-write to RWAcc.
  //    c) Repeat the steps (a) and (b) to get one final sum.
  // 4) USM pointer, InitializeToIdentity = false:
  //    a) Create new uninitialized buffer (if #work-groups > 1) and RWAcc or
  //       re-use user's USM pointer (if #work-groups is 1).
  //    b) discard-write to RWAcc (#WG > 1) or
  //       update-write to USM pointer (#WG == 1).
  //    c) Repeat the steps (a) and (b) to get one final sum.
  // 5) USM pointer, InitializeToIdentity = true:
  //    a) Create new uninitialized buffer (if #work-groups > 1) and RWAcc or
  //       re-use user's USM pointer (if #work-groups is 1).
  //    b) discard-write to RWAcc (#WG > 1) or
  //       discard-write to USM pointer (#WG == 1).
  //    c) Repeat the steps (a) and (b) to get one final sum.
  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) >= 3 &&
       ext::oneapi::detail::AreAllButLastReductions<RestT...>::value)>
  parallel_for(nd_range<Dims> Range, RestT... Rest) {
    std::tuple<RestT...> ArgsTuple(Rest...);
    constexpr size_t NumArgs = sizeof...(RestT);
    auto KernelFunc = std::get<NumArgs - 1>(ArgsTuple);
    auto ReduIndices = std::make_index_sequence<NumArgs - 1>();
    auto ReduTuple =
        ext::oneapi::detail::tuple_select_elements(ArgsTuple, ReduIndices);

    size_t LocalMemPerWorkItem =
        ext::oneapi::detail::reduGetMemPerWorkItem(ReduTuple, ReduIndices);
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it is safer to use queries to the kernel compiled
    // for the device.
    size_t MaxWGSize =
        ext::oneapi::detail::reduGetMaxWGSize(MQueue, LocalMemPerWorkItem);
    if (Range.get_local_range().size() > MaxWGSize)
      throw sycl::runtime_error("The implementation handling parallel_for with"
                                " reduction requires work group size not bigger"
                                " than " +
                                    std::to_string(MaxWGSize),
                                PI_INVALID_WORK_GROUP_SIZE);

    ext::oneapi::detail::reduCGFunc<KernelName>(*this, KernelFunc, Range,
                                                ReduTuple, ReduIndices);
    std::shared_ptr<detail::queue_impl> QueueCopy = MQueue;
    this->finalize();

    size_t NWorkItems = Range.get_group_range().size();
    while (NWorkItems > 1) {
      handler AuxHandler(QueueCopy, MIsHost);
      AuxHandler.saveCodeLoc(MCodeLoc);

      NWorkItems =
          ext::oneapi::detail::reduAuxCGFunc<KernelName, decltype(KernelFunc)>(
              AuxHandler, NWorkItems, MaxWGSize, ReduTuple, ReduIndices);
      MLastEvent = AuxHandler.finalize();
    } // end while (NWorkItems > 1)

    auto CopyEvent = ext::oneapi::detail::reduSaveFinalResultToUserMem(
        QueueCopy, MIsHost, ReduTuple, ReduIndices);
    if (CopyEvent)
      MLastEvent = *CopyEvent;
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
                               _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)NumWorkGroups;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkGroups);
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
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
                               _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)NumWorkGroups;
    (void)WorkGroupSize;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    nd_range<Dims> ExecRange =
        nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize);
    detail::checkValueRange<Dims>(ExecRange);
    MNDRDesc.set(std::move(ExecRange));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
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
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant
    MNDRDesc.set(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CG::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
  }

  void parallel_for(range<1> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems, Kernel);
  }

  void parallel_for(range<2> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems, Kernel);
  }

  void parallel_for(range<3> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems, Kernel);
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
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    kernel Kernel) {
    throwIfActionIsCreated();
    verifyKernelInvoc(Kernel);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NumWorkItems, WorkItemOffset);
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    setType(detail::CG::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
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
    detail::checkValueRange<Dims>(NDRange);
    MNDRDesc.set(std::move(NDRange));
    setType(detail::CG::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
  }

  /// Defines and invokes a SYCL kernel function.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(kernel Kernel, _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    (void)Kernel;
    kernel_single_task<NameT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant
    MNDRDesc.set(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CG::Kernel);
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else
      StoreLambda<NameT, KernelType, /*Dims*/ 0, void>(std::move(KernelFunc));
#else
    detail::CheckDeviceCopyable<KernelType>();
#endif
  }

  /// Invokes a lambda on the host. Dependencies are satisfied on the host.
  ///
  /// \param Func is a lambda that is executed on the host
  template <typename FuncT> void interop_task(FuncT Func) {

    MInteropTask.reset(new detail::InteropTask(std::move(Func)));
    setType(detail::CG::CodeplayInteropTask);
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
                    _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    (void)Kernel;
    (void)NumWorkItems;
    kernel_parallel_for_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkItems);
    MNDRDesc.set(std::move(NumWorkItems));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CG::Kernel);
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else
      StoreLambda<NameT, KernelType, Dims, LambdaArgType>(
          std::move(KernelFunc));
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
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    (void)Kernel;
    (void)NumWorkItems;
    (void)WorkItemOffset;
    kernel_parallel_for_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkItems, WorkItemOffset);
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CG::Kernel);
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else
      StoreLambda<NameT, KernelType, Dims, LambdaArgType>(
          std::move(KernelFunc));
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
                    _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
    (void)Kernel;
    (void)NDRange;
    kernel_parallel_for_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NDRange);
    MNDRDesc.set(std::move(NDRange));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CG::Kernel);
    if (!MIsHost && !lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else
      StoreLambda<NameT, KernelType, Dims, LambdaArgType>(
          std::move(KernelFunc));
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
                               _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)Kernel;
    (void)NumWorkGroups;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkGroups);
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
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
                               _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)Kernel;
    (void)NumWorkGroups;
    (void)WorkGroupSize;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    nd_range<Dims> ExecRange =
        nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize);
    detail::checkValueRange<Dims>(ExecRange);
    MNDRDesc.set(std::move(ExecRange));
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
#endif // __SYCL_DEVICE_ONLY__
  }

  // Clean up KERNELFUNC macro.
#undef _KERNELFUNCPARAM

  // Explicit copy operations API

  /// Copies the content of memory object accessed by Src into the memory
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
            std::shared_ptr<T_Dst> Dst) {
    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForSourceAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Dst);
    typename std::shared_ptr<T_Dst>::element_type *RawDstPtr = Dst.get();
    copy(Src, RawDstPtr);
  }

  /// Copies the content of memory pointed by Src into the memory object
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
  copy(std::shared_ptr<T_Src> Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {
    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Src);
    typename std::shared_ptr<T_Src>::element_type *RawSrcPtr = Src.get();
    copy(RawSrcPtr, Dst);
  }

  /// Copies the content of memory object accessed by Src into the memory
  /// pointed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a source SYCL accessor.
  /// \param Dst is a pointer to destination memory.
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void copy(accessor<T_Src, Dims, AccessMode, AccessTarget, IsPlaceholder> Src,
            T_Dst *Dst) {
    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForSourceAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
#ifndef __SYCL_DEVICE_ONLY__
    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manager.
      copyAccToPtrHost(Src, Dst);
      return;
    }
#endif
    setType(detail::CG::CopyAccToPtr);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = static_cast<void *>(AccImpl.get());
    MDstPtr = static_cast<void *>(Dst);
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  /// Copies the content of memory pointed by Src into the memory object
  /// accessed by Dst.
  ///
  /// Source must have at least as many bytes as the range accessed by Dst.
  ///
  /// \param Src is a pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  void
  copy(const T_Src *Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {
    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
#ifndef __SYCL_DEVICE_ONLY__
    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manager.
      copyPtrToAccHost(Src, Dst);
      return;
    }
#endif
    setType(detail::CG::CopyPtrToAcc);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = const_cast<T_Src *>(Src);
    MDstPtr = static_cast<void *>(AccImpl.get());
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  /// Copies the content of memory object accessed by Src to the memory
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
    static_assert(isValidModeForSourceAccessor(AccessMode_Src),
                  "Invalid source accessor mode for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode_Dst),
                  "Invalid destination accessor mode for the copy method.");
    assert(Dst.get_size() >= Src.get_size() &&
           "The destination accessor does not fit the copied memory.");
    if (copyAccToAccHelper(Src, Dst))
      return;
    setType(detail::CG::CopyAccToAcc);

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
    setType(detail::CG::UpdateHost);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = static_cast<void *>(AccImpl.get());
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
            access::placeholder IsPlaceholder = access::placeholder::false_t,
            typename PropertyListT = property_list>
  void
  fill(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder, PropertyListT>
           Dst,
       const T &Pattern) {
    throwIfActionIsCreated();
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the fill method.");
    if (!MIsHost && (((Dims == 1) && isConstOrGlobal(AccessTarget)) ||
                     isImageOrImageArray(AccessTarget))) {
      setType(detail::CG::Fill);

      detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
      detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

      MDstPtr = static_cast<void *>(AccImpl.get());
      MRequirements.push_back(AccImpl.get());
      MAccStorage.push_back(std::move(AccImpl));

      MPattern.resize(sizeof(T));
      auto PatternPtr = reinterpret_cast<T *>(MPattern.data());
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

  /// Fills the specified memory with the specified pattern.
  ///
  /// \param Ptr is the pointer to the memory to fill
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Count is the number of times to fill Pattern into Ptr.
  template <typename T> void fill(void *Ptr, const T &Pattern, size_t Count) {
    throwIfActionIsCreated();
    static_assert(std::is_trivially_copyable<T>::value,
                  "Pattern must be trivially copyable");
    parallel_for<class __usmfill<T>>(range<1>(Count), [=](id<1> Index) {
      T *CastedPtr = static_cast<T *>(Ptr);
      CastedPtr[Index] = Pattern;
    });
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  void barrier() {
    throwIfActionIsCreated();
    setType(detail::CG::Barrier);
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then the barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  void barrier(const std::vector<event> &WaitList);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  void memcpy(void *Dest, const void *Src, size_t Count);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  template <typename T> void copy(const T *Src, T *Dest, size_t Count) {
    this->memcpy(Dest, Src, Count * sizeof(T));
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if \param Dest is nullptr. The behavior is undefined if \param Dest
  /// is invalid.
  ///
  /// \param Dest is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  void memset(void *Dest, int Value, size_t Count);

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  void prefetch(const void *Ptr, size_t Count);

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  void mem_advise(const void *Ptr, size_t Length, int Advice);

private:
  std::shared_ptr<detail::queue_impl> MQueue;
  /// The storage for the arguments passed.
  /// We need to store a copy of values that are passed explicitly through
  /// set_arg, require and so on, because we need them to be alive after
  /// we exit the method they are passed in.
  std::vector<std::vector<char>> MArgsStorage;
  std::vector<detail::AccessorImplPtr> MAccStorage;
  std::vector<detail::LocalAccessorImplPtr> MLocalAccStorage;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreamStorage;
  mutable std::vector<std::shared_ptr<const void>> MSharedPtrStorage;
  /// The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;
  /// The list of associated accessors with this handler.
  /// These accessors were created with this handler as argument or
  /// have become required for this handler via require method.
  std::vector<detail::ArgDesc> MAssociatedAccesors;
  /// The list of requirements to the memory objects for the scheduling.
  std::vector<detail::Requirement *> MRequirements;
  /// Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;
  std::string MKernelName;
  /// Storage for a sycl::kernel object.
  std::shared_ptr<detail::kernel_impl> MKernel;
  /// Type of the command group, e.g. kernel, fill. Can also encode version.
  /// Use getType and setType methods to access this variable unless
  /// manipulations with version are required
  detail::CG::CGTYPE MCGType = detail::CG::None;
  /// Pointer to the source host memory or accessor(depending on command type).
  void *MSrcPtr = nullptr;
  /// Pointer to the dest host memory or accessor(depends on command type).
  void *MDstPtr = nullptr;
  /// Length to copy or fill (for USM operations).
  size_t MLength = 0;
  /// Pattern that is used to fill memory object in case command type is fill.
  std::vector<char> MPattern;
  /// Storage for a lambda or function object.
  std::unique_ptr<detail::HostKernelBase> MHostKernel;
  /// Storage for lambda/function when using HostTask
  std::unique_ptr<detail::HostTask> MHostTask;
  detail::OSModuleHandle MOSModuleHandle = detail::OSUtil::ExeModuleHandle;
  // Storage for a lambda or function when using InteropTasks
  std::unique_ptr<detail::InteropTask> MInteropTask;
  /// The list of events that order this operation.
  std::vector<detail::EventImplPtr> MEvents;
  /// The list of valid SYCL events that need to complete
  /// before barrier command can be executed
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  bool MIsHost = false;

  detail::code_location MCodeLoc = {};
  bool MIsFinalized = false;
  event MLastEvent;

  // Make queue_impl class friend to be able to call finalize method.
  friend class detail::queue_impl;
  // Make accessor class friend to keep the list of associated accessors.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder,
            typename PropertyListT>
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
  template <typename T, class BinaryOperation, int Dims, bool IsUSM,
            access::placeholder IsPlaceholder>
  friend class ext::oneapi::detail::reduction_impl;

  // This method needs to call the method finalize().
  template <typename Reduction, typename... RestT>
  std::enable_if_t<!Reduction::is_usm> friend ext::oneapi::detail::
      reduSaveFinalResultToUserMemHelper(
          std::vector<event> &Events, std::shared_ptr<detail::queue_impl> Queue,
          bool IsHost, Reduction &, RestT...);

  friend void detail::associateWithHandler(handler &,
                                           detail::AccessorBaseHost *,
                                           access::target);

  friend class ::MockHandler;

  bool DisableRangeRounding();

  bool RangeRoundingTrace();

  void GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                                size_t &MinRange);

  template <typename WrapperT, typename TransformedArgType, int Dims,
            typename KernelType>
  auto getRangeRoundedKernelLambda(KernelType KernelFunc,
                                   range<Dims> NumWorkItems) {
    if constexpr (detail::isKernelLambdaCallableWithKernelHandler<
                      KernelType, TransformedArgType>()) {
      return [=](TransformedArgType Arg, kernel_handler KH) {
        if (Arg[0] >= NumWorkItems[0])
          return;
        Arg.set_allowed_range(NumWorkItems);
        KernelFunc(Arg, KH);
      };
    } else {
      return [=](TransformedArgType Arg) {
        if (Arg[0] >= NumWorkItems[0])
          return;
        Arg.set_allowed_range(NumWorkItems);
        KernelFunc(Arg);
      };
    }
  }
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
