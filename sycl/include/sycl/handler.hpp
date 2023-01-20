//==-------- handler.hpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/accessor.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/cg.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/export.hpp>
#include <sycl/detail/handler_proxy.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/id.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/item.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_handler.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/property_list.hpp>
#include <sycl/reduction_forward.hpp>
#include <sycl/sampler.hpp>
#include <sycl/stl.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

// SYCL_LANGUAGE_VERSION is 4 digit year followed by 2 digit revision
#if !SYCL_LANGUAGE_VERSION || SYCL_LANGUAGE_VERSION < 202001
#define __SYCL_NONCONST_FUNCTOR__
#endif

// replace _KERNELFUNCPARAM(KernelFunc) with   KernelType KernelFunc
//                                     or     const KernelType &KernelFunc
#ifdef __SYCL_NONCONST_FUNCTOR__
#define _KERNELFUNCPARAMTYPE KernelType
#else
#define _KERNELFUNCPARAMTYPE const KernelType &
#endif
#define _KERNELFUNCPARAM(a) _KERNELFUNCPARAMTYPE a

template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
          sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
class __fill;

template <typename T> class __usmfill;
template <typename T> class __usmfill2d;
template <typename T> class __usmmemcpy2d;

template <typename T_Src, typename T_Dst, int Dims,
          sycl::access::mode AccessMode, sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
class __copyAcc2Ptr;

template <typename T_Src, typename T_Dst, int Dims,
          sycl::access::mode AccessMode, sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
class __copyPtr2Acc;

template <typename T_Src, int Dims_Src, sycl::access::mode AccessMode_Src,
          sycl::access::target AccessTarget_Src, typename T_Dst, int Dims_Dst,
          sycl::access::mode AccessMode_Dst,
          sycl::access::target AccessTarget_Dst,
          sycl::access::placeholder IsPlaceholder_Src,
          sycl::access::placeholder IsPlaceholder_Dst>
class __copyAcc2Acc;

// For unit testing purposes
class MockHandler;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// Forward declaration

class handler;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
namespace detail {

class handler_impl;
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

// Helper for merging properties with ones defined in an optional kernel functor
// getter.
template <typename KernelType, typename PropertiesT, typename Cond = void>
struct GetMergedKernelProperties {
  using type = PropertiesT;
};
template <typename KernelType, typename PropertiesT>
struct GetMergedKernelProperties<
    KernelType, PropertiesT,
    std::enable_if_t<ext::oneapi::experimental::detail::
                         HasKernelPropertiesGetMethod<KernelType>::value>> {
  using get_method_properties =
      typename ext::oneapi::experimental::detail::HasKernelPropertiesGetMethod<
          KernelType>::properties_t;
  static_assert(
      ext::oneapi::experimental::is_property_list<get_method_properties>::value,
      "get(sycl::ext::oneapi::experimental::properties_tag) member in kernel "
      "functor class must return a valid property list.");
  using type = ext::oneapi::experimental::detail::merged_properties_t<
      PropertiesT, get_method_properties>;
};

#if __SYCL_ID_QUERIES_FIT_IN_INT__
template <typename T, typename ValT>
typename detail::enable_if_t<std::is_same<ValT, size_t>::value ||
                             std::is_same<ValT, unsigned long long>::value>
checkValueRangeImpl(ValT V) {
  static constexpr size_t Limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (V > Limit)
    throw runtime_error(NotIntMsg<T>::Msg, PI_ERROR_INVALID_VALUE);
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

template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel {
public:
  RoundedRangeKernel(range<Dims> NumWorkItems, KernelType KernelFunc)
      : NumWorkItems(NumWorkItems), KernelFunc(KernelFunc) {}

  void operator()(TransformedArgType Arg) const {
    if (Arg[0] >= NumWorkItems[0])
      return;
    Arg.set_allowed_range(NumWorkItems);
    KernelFunc(Arg);
  }

private:
  range<Dims> NumWorkItems;
  KernelType KernelFunc;
};

template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH {
public:
  RoundedRangeKernelWithKH(range<Dims> NumWorkItems, KernelType KernelFunc)
      : NumWorkItems(NumWorkItems), KernelFunc(KernelFunc) {}

  void operator()(TransformedArgType Arg, kernel_handler KH) const {
    if (Arg[0] >= NumWorkItems[0])
      return;
    Arg.set_allowed_range(NumWorkItems);
    KernelFunc(Arg, KH);
  }

private:
  range<Dims> NumWorkItems;
  KernelType KernelFunc;
};

using sycl::detail::enable_if_t;
using sycl::detail::queue_impl;

} // namespace detail

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

  /// Constructs SYCL handler from the associated queue and the submission's
  /// primary and secondary queue.
  ///
  /// \param Queue is a SYCL queue. This is equal to either PrimaryQueue or
  ///        SecondaryQueue.
  /// \param PrimaryQueue is the primary SYCL queue of the submission.
  /// \param SecondaryQueue is the secondary SYCL queue of the submission. This
  ///        is null if no secondary queue is associated with the submission.
  /// \param IsHost indicates if this handler is created for SYCL host device.
  handler(std::shared_ptr<detail::queue_impl> Queue,
          std::shared_ptr<detail::queue_impl> PrimaryQueue,
          std::shared_ptr<detail::queue_impl> SecondaryQueue, bool IsHost);

  /// Stores copy of Arg passed to the MArgsStorage.
  template <typename T, typename F = typename detail::remove_const_t<
                            typename detail::remove_reference_t<T>>>
  F *storePlainArg(T &&Arg) {
    MArgsStorage.emplace_back(sizeof(T));
    auto Storage = reinterpret_cast<F *>(MArgsStorage.back().data());
    *Storage = Arg;
    return Storage;
  }

  void setType(detail::CG::CGTYPE Type) { MCGType = Type; }

  detail::CG::CGTYPE getType() { return MCGType; }

  void throwIfActionIsCreated() {
    if (detail::CG::None != getType())
      throw sycl::runtime_error("Attempt to set multiple actions for the "
                                "command group. Command group must consist of "
                                "a single kernel or explicit memory operation.",
                                PI_ERROR_INVALID_OPERATION);
  }

  /// Extracts and prepares kernel arguments from the lambda using integration
  /// header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs,
                               bool IsESIMD);

  /// Extracts and prepares kernel arguments set via set_arg(s).
  void extractArgsAndReqs();

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
  ///
  /// @param ReduObj is a pointer to object that must be stored.
  void addReduction(const std::shared_ptr<const void> &ReduObj);

  ~handler() = default;

  // TODO: Private and unusued. Remove when ABI break is allowed.
  bool is_host() { return MIsHost; }

#ifdef __SYCL_DEVICE_ONLY__
  // In device compilation accessor isn't inherited from AccessorBaseHost, so
  // can't detect by it. Since we don't expect it to be ever called in device
  // execution, just use blind void *.
  void associateWithHandler(void *AccBase, access::target AccTarget);
#else
  void associateWithHandler(detail::AccessorBaseHost *AccBase,
                            access::target AccTarget);
#endif

  // Recursively calls itself until arguments pack is fully processed.
  // The version for regular(standard layout) argument.
  template <typename T, typename... Ts>
  void setArgsHelper(int ArgIndex, T &&Arg, Ts &&...Args) {
    set_arg(ArgIndex, std::move(Arg));
    setArgsHelper(++ArgIndex, std::move(Args)...);
  }

  void setArgsHelper(int) {}

  void setLocalAccessorArgHelper(int ArgIndex,
                                 detail::LocalAccessorBaseHost &LocalAccBase) {
    detail::LocalAccessorImplPtr LocalAccImpl =
        detail::getSyclObjImpl(LocalAccBase);
    detail::LocalAccessorImplHost *Req = LocalAccImpl.get();
    MLocalAccStorage.push_back(std::move(LocalAccImpl));
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_accessor, Req,
                       static_cast<int>(access::target::local), ArgIndex);
  }

  // setArgHelper for local accessor argument (legacy accessor interface)
  template <typename DataT, int Dims, access::mode AccessMode,
            access::placeholder IsPlaceholder>
  void setArgHelper(int ArgIndex,
                    accessor<DataT, Dims, AccessMode, access::target::local,
                             IsPlaceholder> &&Arg) {
#ifndef __SYCL_DEVICE_ONLY__
    setLocalAccessorArgHelper(ArgIndex, Arg);
#endif
  }

  // setArgHelper for local accessor argument (up to date accessor interface)
  template <typename DataT, int Dims>
  void setArgHelper(int ArgIndex, local_accessor<DataT, Dims> &&Arg) {
#ifndef __SYCL_DEVICE_ONLY__
    setLocalAccessorArgHelper(ArgIndex, Arg);
#endif
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
    detail::AccessorImplHost *Req = AccImpl.get();
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

  // TODO: Unusued. Remove when ABI break is allowed.
  void verifyKernelInvoc(const kernel &Kernel) {
    std::ignore = Kernel;
    return;
  }

  /* The kernel passed to StoreLambda can take an id, an item or an nd_item as
   * its argument. Since esimd plugin directly invokes the kernel (doesn’t use
   * piKernelSetArg), the kernel argument type must be known to the plugin.
   * However, passing kernel argument type to the plugin requires changing ABI
   * in HostKernel class. To overcome this problem, helpers below wrap the
   * “original” kernel with a functor that always takes an nd_item as argument.
   * A functor is used instead of a lambda because extractArgsAndReqsFromLambda
   * needs access to the “original” kernel and keeps references to its internal
   * data, i.e. the kernel passed as argument cannot be local in scope. The
   * functor itself is again encapsulated in a std::function since functor’s
   * type is unknown to the plugin.
   */

  // For 'id, item w/wo offset, nd_item' kernel arguments
  template <class KernelType, class NormalizedKernelType, int Dims>
  KernelType *ResetHostKernelHelper(const KernelType &KernelFunc) {
    NormalizedKernelType NormalizedKernel(KernelFunc);
    auto NormalizedKernelFunc =
        std::function<void(const sycl::nd_item<Dims> &)>(NormalizedKernel);
    auto HostKernelPtr =
        new detail::HostKernel<decltype(NormalizedKernelFunc),
                               sycl::nd_item<Dims>, Dims>(NormalizedKernelFunc);
    MHostKernel.reset(HostKernelPtr);
    return &HostKernelPtr->MKernel.template target<NormalizedKernelType>()
                ->MKernelFunc;
  }

  // For 'sycl::id<Dims>' kernel argument
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if<std::is_same<ArgT, sycl::id<Dims>>::value,
                          KernelType *>::type
  ResetHostKernel(const KernelType &KernelFunc) {
    struct NormalizedKernelType {
      KernelType MKernelFunc;
      NormalizedKernelType(const KernelType &KernelFunc)
          : MKernelFunc(KernelFunc) {}
      void operator()(const nd_item<Dims> &Arg) {
        detail::runKernelWithArg(MKernelFunc, Arg.get_global_id());
      }
    };
    return ResetHostKernelHelper<KernelType, struct NormalizedKernelType, Dims>(
        KernelFunc);
  }

  // For 'sycl::nd_item<Dims>' kernel argument
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if<std::is_same<ArgT, sycl::nd_item<Dims>>::value,
                          KernelType *>::type
  ResetHostKernel(const KernelType &KernelFunc) {
    struct NormalizedKernelType {
      KernelType MKernelFunc;
      NormalizedKernelType(const KernelType &KernelFunc)
          : MKernelFunc(KernelFunc) {}
      void operator()(const nd_item<Dims> &Arg) {
        detail::runKernelWithArg(MKernelFunc, Arg);
      }
    };
    return ResetHostKernelHelper<KernelType, struct NormalizedKernelType, Dims>(
        KernelFunc);
  }

  // For 'sycl::item<Dims, without_offset>' kernel argument
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if<std::is_same<ArgT, sycl::item<Dims, false>>::value,
                          KernelType *>::type
  ResetHostKernel(const KernelType &KernelFunc) {
    struct NormalizedKernelType {
      KernelType MKernelFunc;
      NormalizedKernelType(const KernelType &KernelFunc)
          : MKernelFunc(KernelFunc) {}
      void operator()(const nd_item<Dims> &Arg) {
        sycl::item<Dims, false> Item = detail::Builder::createItem<Dims, false>(
            Arg.get_global_range(), Arg.get_global_id());
        detail::runKernelWithArg(MKernelFunc, Item);
      }
    };
    return ResetHostKernelHelper<KernelType, struct NormalizedKernelType, Dims>(
        KernelFunc);
  }

  // For 'sycl::item<Dims, with_offset>' kernel argument
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if<std::is_same<ArgT, sycl::item<Dims, true>>::value,
                          KernelType *>::type
  ResetHostKernel(const KernelType &KernelFunc) {
    struct NormalizedKernelType {
      KernelType MKernelFunc;
      NormalizedKernelType(const KernelType &KernelFunc)
          : MKernelFunc(KernelFunc) {}
      void operator()(const nd_item<Dims> &Arg) {
        sycl::item<Dims, true> Item = detail::Builder::createItem<Dims, true>(
            Arg.get_global_range(), Arg.get_global_id(), Arg.get_offset());
        detail::runKernelWithArg(MKernelFunc, Item);
      }
    };
    return ResetHostKernelHelper<KernelType, struct NormalizedKernelType, Dims>(
        KernelFunc);
  }

  // For 'void' kernel argument (single_task)
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if_t<std::is_same<ArgT, void>::value, KernelType *>
  ResetHostKernel(const KernelType &KernelFunc) {
    struct NormalizedKernelType {
      KernelType MKernelFunc;
      NormalizedKernelType(const KernelType &KernelFunc)
          : MKernelFunc(KernelFunc) {}
      void operator()(const nd_item<Dims> &Arg) {
        (void)Arg;
        detail::runKernelWithoutArg(MKernelFunc);
      }
    };
    return ResetHostKernelHelper<KernelType, struct NormalizedKernelType, Dims>(
        KernelFunc);
  }

  // For 'sycl::group<Dims>' kernel argument
  // 'wrapper'-based approach using 'NormalizedKernelType' struct is not used
  // for 'void(sycl::group<Dims>)' since 'void(sycl::group<Dims>)' is not
  // supported in ESIMD.
  template <class KernelType, typename ArgT, int Dims>
  typename std::enable_if<std::is_same<ArgT, sycl::group<Dims>>::value,
                          KernelType *>::type
  ResetHostKernel(const KernelType &KernelFunc) {
    MHostKernel.reset(
        new detail::HostKernel<KernelType, ArgT, Dims>(KernelFunc));
    return (KernelType *)(MHostKernel->getPtr());
  }

  /// Verifies the kernel bundle to be used if any is set. This throws a
  /// sycl::exception with error code errc::kernel_not_supported if the used
  /// kernel bundle does not contain a suitable device image with the requested
  /// kernel.
  ///
  /// \param KernelName is the name of the SYCL kernel to check that the used
  ///                   kernel bundle contains.
  void verifyUsedKernelBundle(const std::string &KernelName);

  /// Stores lambda to the template-free object
  ///
  /// Also initializes kernel name, list of arguments and requirements using
  /// information from the integration header.
  ///
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName, typename KernelType, int Dims,
            typename LambdaArgType>
  void StoreLambda(KernelType KernelFunc) {
    using KI = detail::KernelInfo<KernelName>;
    constexpr bool IsCallableWithKernelHandler =
        detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                 LambdaArgType>::value;

    if (IsCallableWithKernelHandler && MIsHost) {
      throw sycl::feature_not_supported(
          "kernel_handler is not yet supported by host device.",
          PI_ERROR_INVALID_OPERATION);
    }

    KernelType *KernelPtr =
        ResetHostKernel<KernelType, LambdaArgType, Dims>(KernelFunc);

    constexpr bool KernelHasName =
        KI::getName() != nullptr && KI::getName()[0] != '\0';

    // Some host compilers may have different captures from Clang. Currently
    // there is no stable way of handling this when extracting the captures, so
    // a static assert is made to fail for incompatible kernel lambdas.
    static_assert(
        !KernelHasName || sizeof(KernelFunc) == KI::getKernelSize(),
        "Unexpected kernel lambda size. This can be caused by an "
        "external host compiler producing a lambda with an "
        "unexpected layout. This is a limitation of the compiler."
        "In many cases the difference is related to capturing constexpr "
        "variables. In such cases removing constexpr specifier aligns the "
        "captures between the host compiler and the device compiler."
        "\n"
        "In case of MSVC, passing "
        "-fsycl-host-compiler-options='/std:c++latest' "
        "might also help.");

    // Empty name indicates that the compilation happens without integration
    // header, so don't perform things that require it.
    if (KernelHasName) {
      // TODO support ESIMD in no-integration-header case too.
      MArgs.clear();
      extractArgsAndReqsFromLambda(reinterpret_cast<char *>(KernelPtr),
                                   KI::getNumParams(), &KI::getParamDesc(0),
                                   KI::isESIMD());
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
    if (IsCallableWithKernelHandler) {
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
    parallel_for<
        class __copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc, TDst, DimDst,
                            ModeDst, TargetDst, IsPHSrc, IsPHDst>>(
        LinearizedRange, [=](id<1> Id) {
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

    single_task<
        class __copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc, TDst, DimDst,
                            ModeDst, TargetDst, IsPHSrc, IsPHDst>>(
        [=]() { *(Dst.get_pointer()) = *(Src.get_pointer()); });
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
    parallel_for<
        class __copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        Range, [=](id<Dim> Index) {
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
    single_task<class __copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        [=]() {
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
    parallel_for<
        class __copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        Range, [=](id<Dim> Index) {
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
    single_task<class __copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        [=]() {
          *(Dst.get_pointer()) = *(reinterpret_cast<const TDst *>(Src));
        });
  }
#endif // __SYCL_DEVICE_ONLY__

  constexpr static bool isConstOrGlobal(access::target AccessTarget) {
    return AccessTarget == access::target::device ||
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

  template <int Dims, typename LambdaArgType> struct TransformUserItemType {
    using type = typename std::conditional<
        std::is_convertible<nd_item<Dims>, LambdaArgType>::value, nd_item<Dims>,
        typename std::conditional<
            std::is_convertible<item<Dims>, LambdaArgType>::value, item<Dims>,
            LambdaArgType>::type>::type;
  };

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
  template <typename KernelName, typename KernelType, int Dims,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void parallel_for_lambda_impl(range<Dims> NumWorkItems,
                                KernelType KernelFunc) {
    throwIfActionIsCreated();
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;

    // If 1D kernel argument is an integral type, convert it to sycl::item<1>
    // If user type is convertible from sycl::item/sycl::nd_item, use
    // sycl::item/sycl::nd_item to transport item information
    using TransformedArgType = typename std::conditional<
        std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
        typename TransformUserItemType<Dims, LambdaArgType>::type>::type;

    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;

    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());

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
    // 3. The range is already a multiple of the rounding factor.
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
        (KI::getName() == nullptr || KI::getName()[0] == '\0');

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
      kernel_parallel_for_wrapper<KName, TransformedArgType, decltype(Wrapper),
                                  PropertiesT>(Wrapper);
#ifndef __SYCL_DEVICE_ONLY__
      detail::checkValueRange<Dims>(AdjustedRange);
      MNDRDesc.set(std::move(AdjustedRange));
      StoreLambda<KName, decltype(Wrapper), Dims, TransformedArgType>(
          std::move(Wrapper));
      setType(detail::CG::Kernel);
#endif
    } else
#endif // !__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ &&
       // !DPCPP_HOST_DEVICE_OPENMP && !DPCPP_HOST_DEVICE_PERF_NATIVE &&
       // SYCL_LANGUAGE_VERSION >= 202001
    {
      (void)NumWorkItems;
      kernel_parallel_for_wrapper<NameT, TransformedArgType, KernelType,
                                  PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
      detail::checkValueRange<Dims>(NumWorkItems);
      MNDRDesc.set(std::move(NumWorkItems));
      StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
          std::move(KernelFunc));
      setType(detail::CG::Kernel);
#endif
    }
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
  /// \param Properties is the properties.
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName, typename KernelType, int Dims,
            typename PropertiesT>
  void parallel_for_impl(nd_range<Dims> ExecutionRange, PropertiesT,
                         _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
    // If user type is convertible from sycl::item/sycl::nd_item, use
    // sycl::item/sycl::nd_item to transport item information
    using TransformedArgType =
        typename TransformUserItemType<Dims, LambdaArgType>::type;
    (void)ExecutionRange;
    kernel_parallel_for_wrapper<NameT, TransformedArgType, KernelType,
                                PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(ExecutionRange);
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
        std::move(KernelFunc));
    setType(detail::CG::Kernel);
#endif
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
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NumWorkItems);
    MNDRDesc.set(std::move(NumWorkItems));
    setType(detail::CG::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
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
  template <typename KernelName, typename KernelType, int Dims,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void parallel_for_work_group_lambda_impl(range<Dims> NumWorkGroups,
                                           _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)NumWorkGroups;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType, KernelType,
                                           PropertiesT>(KernelFunc);
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
  template <typename KernelName, typename KernelType, int Dims,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void parallel_for_work_group_lambda_impl(range<Dims> NumWorkGroups,
                                           range<Dims> WorkGroupSize,
                                           _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    (void)NumWorkGroups;
    (void)WorkGroupSize;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType, KernelType,
                                           PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    nd_range<Dims> ExecRange =
        nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize);
    detail::checkValueRange<Dims>(ExecRange);
    MNDRDesc.set(std::move(ExecRange));
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
#endif // __SYCL_DEVICE_ONLY__
  }

#ifdef SYCL_LANGUAGE_VERSION
#define __SYCL_KERNEL_ATTR__ [[clang::sycl_kernel]]
#else
#define __SYCL_KERNEL_ATTR__
#endif

  // NOTE: the name of this function - "kernel_single_task" - is used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType, typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_single_task(_KERNELFUNCPARAM(KernelFunc)) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc();
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of this function - "kernel_single_task" - is used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename KernelType, typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_single_task(_KERNELFUNCPARAM(KernelFunc), kernel_handler KH) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  // NOTE: the name of these functions - "kernel_parallel_for" - are used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType,
            typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_parallel_for(_KERNELFUNCPARAM(KernelFunc)) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of these functions - "kernel_parallel_for" - are used by the
  // Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType,
            typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_parallel_for(_KERNELFUNCPARAM(KernelFunc), kernel_handler KH) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()), KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  // NOTE: the name of this function - "kernel_parallel_for_work_group" - is
  // used by the Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType,
            typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_parallel_for_work_group(_KERNELFUNCPARAM(KernelFunc)) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
#else
    (void)KernelFunc;
#endif
  }

  // NOTE: the name of this function - "kernel_parallel_for_work_group" - is
  // used by the Front End to determine kernel invocation kind.
  template <typename KernelName, typename ElementType, typename KernelType,
            typename... Props>
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
  __SYCL_KERNEL_ATTR__ void
  kernel_parallel_for_work_group(_KERNELFUNCPARAM(KernelFunc),
                                 kernel_handler KH) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()), KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }

  template <typename... Props> struct KernelPropertiesUnpackerImpl {
    // Just pass extra Props... as template parameters to the underlying
    // Caller->* member functions. Don't have reflection so try to use
    // templates as much as possible to reduce the amount of boilerplate code
    // needed. All the type checks are expected to be done at the Caller's
    // methods side.

    template <typename... TypesToForward, typename... ArgsTy>
    static void kernel_single_task_unpack(handler *h, ArgsTy... Args) {
      h->kernel_single_task<TypesToForward..., Props...>(Args...);
    }

    template <typename... TypesToForward, typename... ArgsTy>
    static void kernel_parallel_for_unpack(handler *h, ArgsTy... Args) {
      h->kernel_parallel_for<TypesToForward..., Props...>(Args...);
    }

    template <typename... TypesToForward, typename... ArgsTy>
    static void kernel_parallel_for_work_group_unpack(handler *h,
                                                      ArgsTy... Args) {
      h->kernel_parallel_for_work_group<TypesToForward..., Props...>(Args...);
    }
  };

  template <typename PropertiesT>
  struct KernelPropertiesUnpacker : public KernelPropertiesUnpackerImpl<> {
    // This should always fail outside the specialization below but must be
    // dependent to avoid failing even if not instantiated.
    static_assert(
        ext::oneapi::experimental::is_property_list<PropertiesT>::value,
        "Template type is not a property list.");
  };

  template <typename... Props>
  struct KernelPropertiesUnpacker<
      ext::oneapi::experimental::detail::properties_t<Props...>>
      : public KernelPropertiesUnpackerImpl<Props...> {};

  // Helper function to
  //
  //   * Make use of the KernelPropertiesUnpacker above
  //   * Decide if we need an extra kernel_handler parameter
  //
  // The interface uses a \p Lambda callback to propagate that information back
  // to the caller as we need the caller to communicate:
  //
  //   * Name of the method to call
  //   * Provide explicit template type parameters for the call
  //
  // Couldn't think of a better way to achieve both.
  template <typename KernelType, typename PropertiesT, bool HasKernelHandlerArg,
            typename FuncTy>
  void unpack(_KERNELFUNCPARAM(KernelFunc), FuncTy Lambda) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
    using MergedPropertiesT =
        typename detail::GetMergedKernelProperties<KernelType,
                                                   PropertiesT>::type;
    using Unpacker = KernelPropertiesUnpacker<MergedPropertiesT>;
    if constexpr (HasKernelHandlerArg) {
      kernel_handler KH;
      Lambda(Unpacker{}, this, KernelFunc, KH);
    } else {
      Lambda(Unpacker{}, this, KernelFunc);
    }
  }

  // NOTE: to support kernel_handler argument in kernel lambdas, only
  // kernel_***_wrapper functions must be called in this code

  template <typename KernelName, typename KernelType,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void kernel_single_task_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelType, PropertiesT,
           detail::KernelLambdaHasKernelHandlerArgT<KernelType>::value>(
        KernelFunc, [&](auto Unpacker, auto... args) {
          Unpacker.template kernel_single_task_unpack<KernelName, KernelType>(
              args...);
        });
  }

  template <typename KernelName, typename ElementType, typename KernelType,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void kernel_parallel_for_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelType, PropertiesT,
           detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                    ElementType>::value>(
        KernelFunc, [&](auto Unpacker, auto... args) {
          Unpacker.template kernel_parallel_for_unpack<KernelName, ElementType,
                                                       KernelType>(args...);
        });
  }

  template <typename KernelName, typename ElementType, typename KernelType,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void kernel_parallel_for_work_group_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelType, PropertiesT,
           detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                    ElementType>::value>(
        KernelFunc, [&](auto Unpacker, auto... args) {
          Unpacker.template kernel_parallel_for_work_group_unpack<
              KernelName, ElementType, KernelType>(args...);
        });
  }

  /// Defines and invokes a SYCL kernel function as a function object type.
  ///
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName, typename KernelType,
            typename PropertiesT =
                ext::oneapi::experimental::detail::empty_properties_t>
  void single_task_lambda_impl(_KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
    kernel_single_task_wrapper<NameT, KernelType, PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant.
    MNDRDesc.set(range<1>{1});

    StoreLambda<NameT, KernelType, /*Dims*/ 1, void>(KernelFunc);
    setType(detail::CG::Kernel);
#endif
  }

  void setStateExplicitKernelBundle();
  void setStateSpecConstSet();
  bool isStateExplicitKernelBundle() const;

  std::shared_ptr<detail::kernel_bundle_impl>
  getOrInsertHandlerKernelBundle(bool Insert) const;

  void setHandlerKernelBundle(kernel Kernel);

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

  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type Value) {

    setStateSpecConstSet();

    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/true);

    detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
        KernelBundleImplPtr)
        .set_specialization_constant<SpecName>(Value);
  }

  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {

    if (isStateExplicitKernelBundle())
      throw sycl::exception(make_error_code(errc::invalid),
                            "Specialization constants cannot be read after "
                            "explicitly setting the used kernel bundle");

    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/true);

    return detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
               KernelBundleImplPtr)
        .get_specialization_constant<SpecName>();
  }

  void
  use_kernel_bundle(const kernel_bundle<bundle_state::executable> &ExecBundle);

  /// Requires access to the memory object associated with the placeholder
  /// accessor. Calling this function with a non-placeholder accessor has no
  /// effect.
  ///
  /// The command group has a requirement to gain access to the given memory
  /// object before executing.
  ///
  /// \param Acc is a SYCL accessor describing required memory region.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  void require(accessor<DataT, Dims, AccMode, AccTarget, isPlaceholder> Acc) {
    if (Acc.is_placeholder())
      associateWithHandler(&Acc, AccTarget);
  }

  /// Registers event dependencies on this command group.
  ///
  /// \param Event is a valid SYCL event to wait on.
  void depends_on(event Event);

  /// Registers event dependencies on this command group.
  ///
  /// \param Events is a vector of valid SYCL events to wait on.
  void depends_on(const std::vector<event> &Events);

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

  template <typename DataT, int Dims>
  void set_arg(int ArgIndex, local_accessor<DataT, Dims> Arg) {
    setArgHelper(ArgIndex, std::move(Arg));
  }

  /// Sets arguments for OpenCL interoperability kernels.
  ///
  /// Registers pack of arguments(Args) with indexes starting from 0.
  ///
  /// \param Args are argument values to be set.
  template <typename... Ts> void set_args(Ts &&...Args) {
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
  void single_task(_KERNELFUNCPARAM(KernelFunc)) {
    single_task_lambda_impl<KernelName>(KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<1> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<2> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<3> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(NumWorkItems, std::move(KernelFunc));
  }

  /// Defines and invokes a SYCL kernel on host device.
  ///
  /// \param Func is a SYCL kernel function defined by lambda function or a
  /// named function object type.
  template <typename FuncT>
  __SYCL_DEPRECATED(
      "run_on_host_intel() is deprecated, use host_task() instead")
  void run_on_host_intel(FuncT Func) {
    throwIfActionIsCreated();
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant
    MNDRDesc.set(range<1>{1});

    MArgs = std::move(MAssociatedAccesors);
    MHostKernel.reset(new detail::HostKernel<FuncT, void, 1>(std::move(Func)));
    setType(detail::CG::RunOnHostIntel);
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
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
    parallel_for_work_group_lambda_impl<KernelName>(NumWorkGroups, KernelFunc);
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
    parallel_for_work_group_lambda_impl<KernelName>(NumWorkGroups,
                                                    WorkGroupSize, KernelFunc);
  }

  /// Invokes a SYCL kernel.
  ///
  /// Executes exactly once. The kernel invocation method has no functors and
  /// cannot be called on host.
  ///
  /// \param Kernel is a SYCL kernel object.
  void single_task(kernel Kernel) {
    throwIfActionIsCreated();
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
      StoreLambda<NameT, KernelType, /*Dims*/ 1, void>(std::move(KernelFunc));
#else
    detail::CheckDeviceCopyable<KernelType>();
#endif
  }

  /// Invokes a lambda on the host. Dependencies are satisfied on the host.
  ///
  /// \param Func is a lambda that is executed on the host
  template <typename FuncT>
  __SYCL_DEPRECATED("interop_task() is deprecated, use host_task() instead")
  void interop_task(FuncT Func) {

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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
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

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  single_task(PropertiesT, _KERNELFUNCPARAM(KernelFunc)) {
    single_task_lambda_impl<KernelName, KernelType, PropertiesT>(KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<1> NumWorkItems, PropertiesT,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 1, PropertiesT>(
        NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<2> NumWorkItems, PropertiesT,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 2, PropertiesT>(
        NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<3> NumWorkItems, PropertiesT,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 3, PropertiesT>(
        NumWorkItems, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT, int Dims>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(nd_range<Dims> Range, PropertiesT Properties,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_impl<KernelName>(Range, Properties, std::move(KernelFunc));
  }

  /// Reductions @{

  template <typename KernelName = detail::auto_name, int Dims,
            typename PropertiesT, typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<Dims> Range, PropertiesT Properties, RestT &&...Rest) {
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(range<Dims> Range, RestT &&...Rest) {
    parallel_for<KernelName>(
        Range, ext::oneapi::experimental::detail::empty_properties_t{},
        std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename PropertiesT, typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(nd_range<Dims> Range, PropertiesT Properties, RestT &&...Rest) {
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(nd_range<Dims> Range, RestT &&...Rest) {
    parallel_for<KernelName>(
        Range, ext::oneapi::experimental::detail::empty_properties_t{},
        std::forward<RestT>(Rest)...);
  }

  /// }@

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename PropertiesT>
  void parallel_for_work_group(range<Dims> NumWorkGroups, PropertiesT,
                               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_work_group_lambda_impl<KernelName, KernelType, Dims,
                                        PropertiesT>(NumWorkGroups, KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename PropertiesT>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize, PropertiesT,
                               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_work_group_lambda_impl<KernelName, KernelType, Dims,
                                        PropertiesT>(NumWorkGroups,
                                                     WorkGroupSize, KernelFunc);
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
    if (Dst.get_size() < Src.get_size())
      throw sycl::invalid_object_error(
          "The destination accessor size is too small to copy the memory into.",
          PI_ERROR_INVALID_OPERATION);

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
      parallel_for<
          class __fill<T, Dims, AccessMode, AccessTarget, IsPlaceholder>>(
          Range, [=](id<Dims> Index) { Dst[Index] = Pattern; });
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
  void ext_oneapi_barrier() {
    throwIfActionIsCreated();
    setType(detail::CG::Barrier);
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  __SYCL2020_DEPRECATED("use 'ext_oneapi_barrier' instead")
  void barrier() { ext_oneapi_barrier(); }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then the barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  void ext_oneapi_barrier(const std::vector<event> &WaitList);

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then the barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  __SYCL2020_DEPRECATED("use 'ext_oneapi_barrier' instead")
  void barrier(const std::vector<event> &WaitList);

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on this handler's
  /// device.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  void memcpy(void *Dest, const void *Src, size_t Count);

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on this handler's
  /// device.
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

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Width is the width in bytes of the 2D region to copy.
  /// \param Height is the height in number of row of the 2D region to copy.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  void ext_oneapi_memcpy2d(void *Dest, size_t DestPitch, const void *Src,
                           size_t SrcPitch, size_t Width, size_t Height) {
    throwIfActionIsCreated();
    if (Width > DestPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Destination pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_memcpy2d'");
    if (Width > SrcPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Source pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_memcpy2d'");
    // If the backends supports 2D copy we use that. Otherwise we use a fallback
    // kernel.
    if (supportsUSMMemcpy2D())
      ext_oneapi_memcpy2d_impl(Dest, DestPitch, Src, SrcPitch, Width, Height);
    else
      commonUSMCopy2DFallbackKernel<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                       Height);
  }

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Width is the width in number of elements of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  template <typename T>
  void ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                         size_t DestPitch, size_t Width, size_t Height) {
    if (Width > DestPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Destination pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_copy2d'");
    if (Width > SrcPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Source pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_copy2d'");
    // If the backends supports 2D copy we use that. Otherwise we use a fallback
    // kernel.
    if (supportsUSMMemcpy2D())
      ext_oneapi_memcpy2d_impl(Dest, DestPitch * sizeof(T), Src,
                               SrcPitch * sizeof(T), Width * sizeof(T), Height);
    else
      commonUSMCopy2DFallbackKernel<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                       Height);
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Value is the value to fill into the region in \param Dest. Value is
  /// cast as an unsigned char.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  void ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                           size_t Width, size_t Height) {
    throwIfActionIsCreated();
    if (Width > DestPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Destination pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_memset2d'");
    T CharVal = static_cast<T>(Value);
    // If the backends supports 2D fill we use that. Otherwise we use a fallback
    // kernel.
    if (supportsUSMMemset2D())
      ext_oneapi_memset2d_impl(Dest, DestPitch, Value, Width, Height);
    else
      commonUSMFill2DFallbackKernel(Dest, DestPitch, CharVal, Width, Height);
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  template <typename T>
  void ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                         size_t Width, size_t Height) {
    throwIfActionIsCreated();
    static_assert(std::is_trivially_copyable<T>::value,
                  "Pattern must be trivially copyable");
    if (Width > DestPitch)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Destination pitch must be greater than or equal "
                            "to the width specified in 'ext_oneapi_fill2d'");
    // If the backends supports 2D fill we use that. Otherwise we use a fallback
    // kernel.
    if (supportsUSMFill2D())
      ext_oneapi_fill2d_impl(Dest, DestPitch, &Pattern, sizeof(T), Width,
                             Height);
    else
      commonUSMFill2DFallbackKernel(Dest, DestPitch, Pattern, Width, Height);
  }

private:
  std::shared_ptr<detail::handler_impl> MImpl;
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
  std::vector<detail::AccessorImplHost *> MRequirements;
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
  // Make reduction friends to store buffers and arrays created for it
  // in handler from reduction methods.
  template <typename T, class BinaryOperation, int Dims, size_t Extent,
            typename RedOutVar>
  friend class detail::reduction_impl_algo;

  friend inline void detail::reduction::finalizeHandler(handler &CGH);
  template <class FunctorTy>
  friend void detail::reduction::withAuxHandler(handler &CGH, FunctorTy Func);

  template <typename KernelName, detail::reduction::strategy Strategy, int Dims,
            typename PropertiesT, typename... RestT>
  friend void detail::reduction_parallel_for(handler &CGH, range<Dims> NDRange,
                                             PropertiesT Properties,
                                             RestT... Rest);

  template <typename KernelName, detail::reduction::strategy Strategy, int Dims,
            typename PropertiesT, typename... RestT>
  friend void
  detail::reduction_parallel_for(handler &CGH, nd_range<Dims> NDRange,
                                 PropertiesT Properties, RestT... Rest);

#ifndef __SYCL_DEVICE_ONLY__
  friend void detail::associateWithHandler(handler &,
                                           detail::AccessorBaseHost *,
                                           access::target);
#endif

  friend class ::MockHandler;
  friend class detail::queue_impl;

  bool DisableRangeRounding();

  bool RangeRoundingTrace();

  void GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                                size_t &MinRange);

  template <typename WrapperT, typename TransformedArgType, int Dims,
            typename KernelType,
            std::enable_if_t<detail::KernelLambdaHasKernelHandlerArgT<
                KernelType, TransformedArgType>::value> * = nullptr>
  auto getRangeRoundedKernelLambda(KernelType KernelFunc,
                                   range<Dims> NumWorkItems) {
    return detail::RoundedRangeKernelWithKH<TransformedArgType, Dims,
                                            KernelType>(NumWorkItems,
                                                        KernelFunc);
  }

  template <typename WrapperT, typename TransformedArgType, int Dims,
            typename KernelType,
            std::enable_if_t<!detail::KernelLambdaHasKernelHandlerArgT<
                KernelType, TransformedArgType>::value> * = nullptr>
  auto getRangeRoundedKernelLambda(KernelType KernelFunc,
                                   range<Dims> NumWorkItems) {
    return detail::RoundedRangeKernel<TransformedArgType, Dims, KernelType>(
        NumWorkItems, KernelFunc);
  }

  // Checks if 2D memory operations are supported by the underlying platform.
  bool supportsUSMMemcpy2D();
  bool supportsUSMFill2D();
  bool supportsUSMMemset2D();

  // Helper function for getting a loose bound on work-items.
  id<2> computeFallbackKernelBounds(size_t Width, size_t Height);

  // Common function for launching a 2D USM memcpy kernel to avoid redefinitions
  // of the kernel from copy and memcpy.
  template <typename T>
  void commonUSMCopy2DFallbackKernel(const void *Src, size_t SrcPitch,
                                     void *Dest, size_t DestPitch, size_t Width,
                                     size_t Height) {
    // Limit number of work items to be resistant to big copies.
    id<2> Chunk = computeFallbackKernelBounds(Height, Width);
    id<2> Iterations = (Chunk + id<2>{Height, Width} - 1) / Chunk;
    parallel_for<class __usmmemcpy2d<T>>(
        range<2>{Chunk[0], Chunk[1]}, [=](id<2> Index) {
          T *CastedDest = static_cast<T *>(Dest);
          const T *CastedSrc = static_cast<const T *>(Src);
          for (uint32_t I = 0; I < Iterations[0]; ++I) {
            for (uint32_t J = 0; J < Iterations[1]; ++J) {
              id<2> adjustedIndex = Index + Chunk * id<2>{I, J};
              if (adjustedIndex[0] < Height && adjustedIndex[1] < Width) {
                CastedDest[adjustedIndex[0] * DestPitch + adjustedIndex[1]] =
                    CastedSrc[adjustedIndex[0] * SrcPitch + adjustedIndex[1]];
              }
            }
          }
        });
  }

  // Common function for launching a 2D USM fill kernel to avoid redefinitions
  // of the kernel from memset and fill.
  template <typename T>
  void commonUSMFill2DFallbackKernel(void *Dest, size_t DestPitch,
                                     const T &Pattern, size_t Width,
                                     size_t Height) {
    // Limit number of work items to be resistant to big fill operations.
    id<2> Chunk = computeFallbackKernelBounds(Height, Width);
    id<2> Iterations = (Chunk + id<2>{Height, Width} - 1) / Chunk;
    parallel_for<class __usmfill2d<T>>(
        range<2>{Chunk[0], Chunk[1]}, [=](id<2> Index) {
          T *CastedDest = static_cast<T *>(Dest);
          for (uint32_t I = 0; I < Iterations[0]; ++I) {
            for (uint32_t J = 0; J < Iterations[1]; ++J) {
              id<2> adjustedIndex = Index + Chunk * id<2>{I, J};
              if (adjustedIndex[0] < Height && adjustedIndex[1] < Width) {
                CastedDest[adjustedIndex[0] * DestPitch + adjustedIndex[1]] =
                    Pattern;
              }
            }
          }
        });
  }

  // Implementation of ext_oneapi_memcpy2d using command for native 2D memcpy.
  void ext_oneapi_memcpy2d_impl(void *Dest, size_t DestPitch, const void *Src,
                                size_t SrcPitch, size_t Width, size_t Height);

  // Untemplated version of ext_oneapi_fill2d using command for native 2D fill.
  void ext_oneapi_fill2d_impl(void *Dest, size_t DestPitch, const void *Value,
                              size_t ValueSize, size_t Width, size_t Height);

  // Implementation of ext_oneapi_memset2d using command for native 2D memset.
  void ext_oneapi_memset2d_impl(void *Dest, size_t DestPitch, int Value,
                                size_t Width, size_t Height);
};
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
