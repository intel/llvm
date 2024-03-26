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
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/impl_utils.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/reduction_forward.hpp>
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/string.hpp>
#include <sycl/detail/string_view.hpp>
#endif
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/intel/experimental/fp_control_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/kernel_execution_properties.hpp>
#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/ext/oneapi/bindless_images_memory.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <sycl/ext/oneapi/device_global/properties.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/group.hpp>
#include <sycl/id.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/item.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/kernel_handler.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/sampler.hpp>

#include <assert.h>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// TODO: refactor this header
// 41(!!!) includes of SYCL headers + 10 includes of standard headers.
// 3300+ lines of code

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

#if defined(__SYCL_UNNAMED_LAMBDA__)
// We can't use nested types (e.g. struct S defined inside main() routine) to
// name kernels. At the same time, we have to provide a unique kernel name for
// sycl::fill and the only thing we can use to introduce that uniqueness (in
// general) is the template parameter T which might be exactly that nested type.
// That means we cannot support sycl::fill(void *, T&, size_t) for such types in
// general. However, we can do better than that when unnamed lambdas are
// enabled, so do it here! See also https://github.com/intel/llvm/issues/469.
template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
          sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
using __fill = sycl::detail::auto_name;
template <typename T> using __usmfill = sycl::detail::auto_name;
template <typename T> using __usmfill2d = sycl::detail::auto_name;
template <typename T> using __usmmemcpy2d = sycl::detail::auto_name;

template <typename T_Src, typename T_Dst, int Dims,
          sycl::access::mode AccessMode, sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
using __copyAcc2Ptr = sycl::detail::auto_name;

template <typename T_Src, typename T_Dst, int Dims,
          sycl::access::mode AccessMode, sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
using __copyPtr2Acc = sycl::detail::auto_name;

template <typename T_Src, int Dims_Src, sycl::access::mode AccessMode_Src,
          sycl::access::target AccessTarget_Src, typename T_Dst, int Dims_Dst,
          sycl::access::mode AccessMode_Dst,
          sycl::access::target AccessTarget_Dst,
          sycl::access::placeholder IsPlaceholder_Src,
          sycl::access::placeholder IsPlaceholder_Dst>
using __copyAcc2Acc = sycl::detail::auto_name;
#else
// Limited fallback path for when unnamed lambdas aren't available. Cannot
// handle nested types.
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
#endif

// For unit testing purposes
class MockHandler;

namespace sycl {
inline namespace _V1 {

// Forward declaration

class handler;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

namespace ext::intel::experimental {
template <class _name, class _dataT, int32_t _min_capacity, class _propertiesT,
          class>
class pipe;
}

namespace ext::oneapi::experimental::detail {
class graph_impl;
} // namespace ext::oneapi::experimental::detail
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

// Version with two arguments to handle the case when kernel_handler is passed
// to a lambda
template <typename RetType, typename Func, typename Arg1, typename Arg2>
static Arg1 member_ptr_helper(RetType (Func::*)(Arg1, Arg2) const);

// Non-const version of the above template to match functors whose 'operator()'
// is declared w/o the 'const' qualifier.
template <typename RetType, typename Func, typename Arg1, typename Arg2>
static Arg1 member_ptr_helper(RetType (Func::*)(Arg1, Arg2));

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

// Checks if a device_global has any registered kernel usage.
__SYCL_EXPORT bool isDeviceGlobalUsedInKernel(const void *DeviceGlobalPtr);

// Extracts a pointer to the value inside a dynamic parameter
__SYCL_EXPORT void *getValueFromDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_base
        &DynamicParamBase);

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
typename std::enable_if_t<std::is_same<ValT, size_t>::value ||
                          std::is_same<ValT, unsigned long long>::value>
checkValueRangeImpl(ValT V) {
  static constexpr size_t Limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (V > Limit)
    throw sycl::exception(make_error_code(errc::nd_range), NotIntMsg<T>::Msg);
}
#endif

template <int Dims, typename T>
typename std::enable_if_t<std::is_same_v<T, range<Dims>> ||
                          std::is_same_v<T, id<Dims>>>
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
typename std::enable_if_t<std::is_same_v<T, nd_range<Dims>>>
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

template <int Dims> class RoundedRangeIDGenerator {
  id<Dims> Id;
  id<Dims> InitId;
  range<Dims> UserRange;
  range<Dims> RoundedRange;
  bool Done = false;

public:
  RoundedRangeIDGenerator(const id<Dims> &Id, const range<Dims> &UserRange,
                          const range<Dims> &RoundedRange)
      : Id(Id), InitId(Id), UserRange(UserRange), RoundedRange(RoundedRange) {
    for (int i = 0; i < Dims; ++i)
      if (Id[i] >= UserRange[i])
        Done = true;
  }

  explicit operator bool() { return !Done; }

  void updateId() {
    for (int i = 0; i < Dims; ++i) {
      Id[i] += RoundedRange[i];
      if (Id[i] < UserRange[i])
        return;
      Id[i] = InitId[i];
    }
    Done = true;
  }

  id<Dims> getId() { return Id; }

  template <typename KernelType> auto getItem() {
    if constexpr (std::is_invocable_v<KernelType, item<Dims> &> ||
                  std::is_invocable_v<KernelType, item<Dims> &, kernel_handler>)
      return detail::Builder::createItem<Dims, true>(UserRange, getId(), {});
    else {
      static_assert(std::is_invocable_v<KernelType, item<Dims, false> &> ||
                        std::is_invocable_v<KernelType, item<Dims, false> &,
                                            kernel_handler>,
                    "Kernel must be invocable with an item!");
      return detail::Builder::createItem<Dims, false>(UserRange, getId());
    }
  }
};

// TODO: The wrappers can be optimized further so that the body
// essentially looks like this:
//   for (auto z = it[2]; z < UserRange[2]; z += it.get_range(2))
//     for (auto y = it[1]; y < UserRange[1]; y += it.get_range(1))
//       for (auto x = it[0]; x < UserRange[0]; x += it.get_range(0))
//         KernelFunc({x,y,z});
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel {
public:
  range<Dims> UserRange;
  KernelType KernelFunc;
  void operator()(item<Dims> It) const {
    auto RoundedRange = It.get_range();
    for (RoundedRangeIDGenerator Gen(It.get_id(), UserRange, RoundedRange); Gen;
         Gen.updateId()) {
      auto item = Gen.template getItem<KernelType>();
      KernelFunc(item);
    }
  }
};

template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH {
public:
  range<Dims> UserRange;
  KernelType KernelFunc;
  void operator()(item<Dims> It, kernel_handler KH) const {
    auto RoundedRange = It.get_range();
    for (RoundedRangeIDGenerator Gen(It.get_id(), UserRange, RoundedRange); Gen;
         Gen.updateId()) {
      auto item = Gen.template getItem<KernelType>();
      KernelFunc(item, KH);
    }
  }
};

using std::enable_if_t;
using sycl::detail::queue_impl;

// Returns true if x*y will overflow in T;
// otherwise, returns false and stores x*y in dst.
template <typename T>
static std::enable_if_t<std::is_unsigned_v<T>, bool>
multiply_with_overflow_check(T &dst, T x, T y) {
  dst = x * y;
  return (y != 0) && (x > (std::numeric_limits<T>::max)() / y);
}

template <int Dims> bool range_size_fits_in_size_t(const range<Dims> &r) {
  size_t acc = 1;
  for (int i = 0; i < Dims; ++i) {
    bool did_overflow = multiply_with_overflow_check(acc, acc, r[i]);
    if (did_overflow)
      return false;
  }
  return true;
}
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
using string = std::string;
using string_view = std::string;
#endif

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

  /// Constructs SYCL handler from Graph.
  ///
  /// The hander will add the command-group as a node to the graph rather than
  /// enqueueing it straight away.
  ///
  /// \param Graph is a SYCL command_graph
  handler(std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph);

  /// Stores copy of Arg passed to the CGData.MArgsStorage.
  template <typename T, typename F = typename std::remove_const_t<
                            typename std::remove_reference_t<T>>>
  F *storePlainArg(T &&Arg) {
    CGData.MArgsStorage.emplace_back(sizeof(T));
    auto Storage = reinterpret_cast<F *>(CGData.MArgsStorage.back().data());
    *Storage = Arg;
    return Storage;
  }

  void setType(detail::CG::CGTYPE Type) { MCGType = Type; }

  detail::CG::CGTYPE getType() { return MCGType; }

  void throwIfActionIsCreated() {
    if (detail::CG::None != getType())
      throw sycl::exception(make_error_code(errc::runtime),
                            "Attempt to set multiple actions for the "
                            "command group. Command group must consist of "
                            "a single kernel or explicit memory operation.");
  }

  constexpr static int AccessTargetMask = 0x7ff;
  /// According to section 4.7.6.11. of the SYCL specification, a local accessor
  /// must not be used in a SYCL kernel function that is invoked via single_task
  /// or via the simple form of parallel_for that takes a range parameter.
  template <typename KernelName, typename KernelType>
  void throwOnLocalAccessorMisuse() const {
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    using KI = sycl::detail::KernelInfo<NameT>;

    auto *KernelArgs = &KI::getParamDesc(0);

    for (unsigned I = 0; I < KI::getNumParams(); ++I) {
      const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
      const access::target AccTarget =
          static_cast<access::target>(KernelArgs[I].info & AccessTargetMask);
      if ((Kind == detail::kernel_param_kind_t::kind_accessor) &&
          (AccTarget == target::local))
        throw sycl::exception(
            make_error_code(errc::kernel_argument),
            "A local accessor must not be used in a SYCL kernel function "
            "that is invoked via single_task or via the simple form of "
            "parallel_for that takes a range parameter.");
    }
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
  detail::string getKernelName();

  template <typename LambdaNameT> bool lambdaAndKernelHaveEqualName() {
    // TODO It is unclear a kernel and a lambda/functor must to be equal or not
    // for parallel_for with sycl::kernel and lambda/functor together
    // Now if they are equal we extract argumets from lambda/functor for the
    // kernel. Else it is necessary use set_atg(s) for resolve the order and
    // values of arguments for the kernel.
    assert(MKernel && "MKernel is not initialized");
    const std::string LambdaName = detail::KernelInfo<LambdaNameT>::getName();
    detail::string KernelName = getKernelName();
    return KernelName == LambdaName;
  }

  /// Saves the location of user's code passed in \p CodeLoc for future usage in
  /// finalize() method.
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

  /// Saves resources created by handling reduction feature in handler.
  /// They are then forwarded to command group and destroyed only after
  /// the command group finishes the work on device/host.
  ///
  /// @param ReduObj is a pointer to object that must be stored.
  void addReduction(const std::shared_ptr<const void> &ReduObj);

  /// Saves buffers created by handling reduction feature in handler and marks
  /// them as internal. They are then forwarded to command group and destroyed
  /// only after the command group finishes the work on device/host.
  ///
  /// @param ReduBuf is a pointer to buffer that must be stored.
  template <typename T, int Dimensions, typename AllocatorT, typename Enable>
  void
  addReduction(const std::shared_ptr<buffer<T, Dimensions, AllocatorT, Enable>>
                   &ReduBuf) {
    detail::markBufferAsInternal(getSyclObjImpl(*ReduBuf));
    addReduction(std::shared_ptr<const void>(ReduBuf));
  }

  ~handler() = default;

  // TODO: Private and unusued. Remove when ABI break is allowed.
  bool is_host() { return MIsHost; }

#ifdef __SYCL_DEVICE_ONLY__
  // In device compilation accessor isn't inherited from host base classes, so
  // can't detect by it. Since we don't expect it to be ever called in device
  // execution, just use blind void *.
  void associateWithHandler(void *AccBase, access::target AccTarget);
  void associateWithHandler(void *AccBase, image_target AccTarget);
#else
  void associateWithHandlerCommon(detail::AccessorImplPtr AccImpl,
                                  int AccTarget);
  void associateWithHandler(detail::AccessorBaseHost *AccBase,
                            access::target AccTarget);
  void associateWithHandler(detail::UnsampledImageAccessorBaseHost *AccBase,
                            image_target AccTarget);
  void associateWithHandler(detail::SampledImageAccessorBaseHost *AccBase,
                            image_target AccTarget);
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
    (void)ArgIndex;
    (void)Arg;
#ifndef __SYCL_DEVICE_ONLY__
    setLocalAccessorArgHelper(ArgIndex, Arg);
#endif
  }

  // setArgHelper for local accessor argument (up to date accessor interface)
  template <typename DataT, int Dims>
  void setArgHelper(int ArgIndex, local_accessor<DataT, Dims> &&Arg) {
    (void)ArgIndex;
    (void)Arg;
#ifndef __SYCL_DEVICE_ONLY__
    setLocalAccessorArgHelper(ArgIndex, Arg);
#endif
  }

  // setArgHelper for non local accessor argument.
  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  typename std::enable_if_t<AccessTarget != access::target::local, void>
  setArgHelper(
      int ArgIndex,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder> &&Arg) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Arg;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::AccessorImplHost *Req = AccImpl.get();
    // Add accessor to the list of requirements.
    CGData.MRequirements.push_back(Req);
    // Store copy of the accessor.
    CGData.MAccStorage.push_back(std::move(AccImpl));
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

  // setArgHelper for graph dynamic_parameters
  template <typename T>
  void
  setArgHelper(int ArgIndex,
               ext::oneapi::experimental::dynamic_parameter<T> DynamicParam) {
    // Extract and copy arg so we can move it into setArgHelper
    T ArgValue =
        *static_cast<T *>(detail::getValueFromDynamicParameter(DynamicParam));
    // Set the arg in the handler as normal
    setArgHelper(ArgIndex, std::move(ArgValue));
    // Register the dynamic parameter with the handler for later association
    // with the node being added
    registerDynamicParameter(DynamicParam, ArgIndex);
  }

  /// Registers a dynamic parameter with the handler for later association with
  /// the node being created
  /// @param DynamicParamBase
  /// @param ArgIndex
  void registerDynamicParameter(
      ext::oneapi::experimental::detail::dynamic_parameter_base
          &DynamicParamBase,
      int ArgIndex);

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
    auto HostKernelPtr = new detail::HostKernel<decltype(NormalizedKernelFunc),
                                                sycl::nd_item<Dims>, Dims>(
        std::move(NormalizedKernelFunc));
    MHostKernel.reset(HostKernelPtr);
    return &HostKernelPtr->MKernel.template target<NormalizedKernelType>()
                ->MKernelFunc;
  }

  // For 'sycl::id<Dims>' kernel argument
  template <class KernelType, typename ArgT, int Dims>
  std::enable_if_t<std::is_same_v<ArgT, sycl::id<Dims>>, KernelType *>
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
  std::enable_if_t<std::is_same_v<ArgT, sycl::nd_item<Dims>>, KernelType *>
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
  std::enable_if_t<std::is_same_v<ArgT, sycl::item<Dims, false>>, KernelType *>
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
  std::enable_if_t<std::is_same_v<ArgT, sycl::item<Dims, true>>, KernelType *>
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
  typename std::enable_if_t<std::is_same_v<ArgT, void>, KernelType *>
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
  std::enable_if_t<std::is_same_v<ArgT, sycl::group<Dims>>, KernelType *>
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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  void verifyUsedKernelBundle(const std::string &KernelName) {
    verifyUsedKernelBundleInternal(detail::string_view{KernelName});
  }
  void verifyUsedKernelBundleInternal(detail::string_view KernelName);
#else
  void verifyUsedKernelBundle(const std::string &KernelName);
#endif

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
    } else {
      // In case w/o the integration header it is necessary to process
      // accessors from the list(which are associated with this handler) as
      // arguments. We must copy the associated accessors as they are checked
      // later during finalize.
      MArgs = MAssociatedAccesors;
    }

    // If the kernel lambda is callable with a kernel_handler argument, manifest
    // the associated kernel handler.
    if (IsCallableWithKernelHandler) {
      getOrInsertHandlerKernelBundle(/*Insert=*/true);
    }
  }

  /// Process kernel properties.
  ///
  /// Stores information about kernel properties into the handler.
  template <
      typename KernelName,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void processProperties(PropertiesT Props) {
    using KI = detail::KernelInfo<KernelName>;
    static_assert(
        ext::oneapi::experimental::is_property_list<PropertiesT>::value,
        "Template type is not a property list.");
    static_assert(
        !PropertiesT::template has_property<
            sycl::ext::intel::experimental::fp_control_key>() ||
            (PropertiesT::template has_property<
                 sycl::ext::intel::experimental::fp_control_key>() &&
             KI::isESIMD()),
        "Floating point control property is supported for ESIMD kernels only.");
    if constexpr (PropertiesT::template has_property<
                      sycl::ext::intel::experimental::cache_config_key>()) {
      auto Config = Props.template get_property<
          sycl::ext::intel::experimental::cache_config_key>();
      if (Config == sycl::ext::intel::experimental::large_slm) {
        setKernelCacheConfig(PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM);
      } else if (Config == sycl::ext::intel::experimental::large_data) {
        setKernelCacheConfig(PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA);
      }
    } else {
      std::ignore = Props;
    }

    constexpr bool UsesRootSync = PropertiesT::template has_property<
        sycl::ext::oneapi::experimental::use_root_sync_key>();
    setKernelIsCooperative(UsesRootSync);
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
  std::enable_if_t<(DimSrc > 0) && (DimDst > 0), bool>
  copyAccToAccHelper(accessor<TSrc, DimSrc, ModeSrc, TargetSrc, IsPHSrc> Src,
                     accessor<TDst, DimDst, ModeDst, TargetDst, IsPHDst> Dst) {
    if (!MIsHost &&
        IsCopyingRectRegionAvailable(Src.get_range(), Dst.get_range()))
      return false;

    range<1> LinearizedRange(Src.size());
    parallel_for<__copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc, TDst, DimDst,
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
  std::enable_if_t<DimSrc == 0 || DimDst == 0, bool>
  copyAccToAccHelper(accessor<TSrc, DimSrc, ModeSrc, TargetSrc, IsPHSrc> Src,
                     accessor<TDst, DimDst, ModeDst, TargetDst, IsPHDst> Dst) {
    if (!MIsHost)
      return false;

    single_task<__copyAcc2Acc<TSrc, DimSrc, ModeSrc, TargetSrc, TDst, DimDst,
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
  std::enable_if_t<(Dim > 0)>
  copyAccToPtrHost(accessor<TSrc, Dim, AccMode, AccTarget, IsPH> Src,
                   TDst *Dst) {
    range<Dim> Range = Src.get_range();
    parallel_for<__copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        Range, [=](id<Dim> Index) {
          const size_t LinearIndex = detail::getLinearIndex(Index, Range);
          using TSrcNonConst = typename std::remove_const_t<TSrc>;
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
  std::enable_if_t<Dim == 0>
  copyAccToPtrHost(accessor<TSrc, Dim, AccMode, AccTarget, IsPH> Src,
                   TDst *Dst) {
    single_task<__copyAcc2Ptr<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
        [=]() {
          using TSrcNonConst = typename std::remove_const_t<TSrc>;
          *(reinterpret_cast<TSrcNonConst *>(Dst)) = *(Src.get_pointer());
        });
  }

  /// Copies the memory pointed by Src into the memory accessed by Dst.
  ///
  /// \param Src is a pointer to source memory.
  /// \param Dst is a destination SYCL accessor.
  template <typename TSrc, typename TDst, int Dim, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPH>
  std::enable_if_t<(Dim > 0)>
  copyPtrToAccHost(TSrc *Src,
                   accessor<TDst, Dim, AccMode, AccTarget, IsPH> Dst) {
    range<Dim> Range = Dst.get_range();
    parallel_for<__copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
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
  std::enable_if_t<Dim == 0>
  copyPtrToAccHost(TSrc *Src,
                   accessor<TDst, Dim, AccMode, AccTarget, IsPH> Dst) {
    single_task<__copyPtr2Acc<TSrc, TDst, Dim, AccMode, AccTarget, IsPH>>(
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

  // PI APIs only support select fill sizes: 1, 2, 4, 8, 16, 32, 64, 128
  constexpr static bool isBackendSupportedFillSize(size_t Size) {
    return Size == 1 || Size == 2 || Size == 4 || Size == 8 || Size == 16 ||
           Size == 32 || Size == 64 || Size == 128;
  }

  template <int Dims, typename LambdaArgType> struct TransformUserItemType {
    using type = std::conditional_t<
        std::is_convertible_v<nd_item<Dims>, LambdaArgType>, nd_item<Dims>,
        std::conditional_t<std::is_convertible_v<item<Dims>, LambdaArgType>,
                           item<Dims>, LambdaArgType>>;
  };

  std::optional<std::array<size_t, 3>> getMaxWorkGroups();
  // We need to use this version to support gcc 7.5.0. Remove when minimal
  // supported gcc version is bumped.
  std::tuple<std::array<size_t, 3>, bool> getMaxWorkGroups_v2();

  template <int Dims>
  std::tuple<range<Dims>, bool> getRoundedRange(range<Dims> UserRange) {
    range<Dims> RoundedRange = UserRange;
    // Disable the rounding-up optimizations under these conditions:
    // 1. The env var SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING is set.
    // 2. The kernel is provided via an interoperability method (this uses a
    // different code path).
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

    // Perform range rounding if rounding-up is enabled.
    if (this->DisableRangeRounding())
      return {range<Dims>{}, false};

    // Range should be a multiple of this for reasonable performance.
    size_t MinFactorX = 16;
    // Range should be a multiple of this for improved performance.
    size_t GoodFactor = 32;
    // Range should be at least this to make rounding worthwhile.
    size_t MinRangeX = 1024;

    // Check if rounding parameters have been set through environment:
    // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
    this->GetRangeRoundingSettings(MinFactorX, GoodFactor, MinRangeX);

    // In SYCL, each dimension of a global range size is specified by
    // a size_t, which can be up to 64 bits.  All backends should be
    // able to accept a kernel launch with a 32-bit global range size
    // (i.e. do not throw an error).  The OpenCL CPU backend will
    // accept every 64-bit global range, but the GPU backends will not
    // generally accept every 64-bit global range.  So, when we get a
    // non-32-bit global range, we wrap the old kernel in a new kernel
    // that has each work item peform multiple invocations the old
    // kernel in a 32-bit global range.
    id<Dims> MaxNWGs = [&] {
      auto [MaxWGs, HasMaxWGs] = getMaxWorkGroups_v2();
      if (!HasMaxWGs) {
        id<Dims> Default;
        for (int i = 0; i < Dims; ++i)
          Default[i] = (std::numeric_limits<int32_t>::max)();
        return Default;
      }

      id<Dims> IdResult;
      size_t Limit = (std::numeric_limits<int>::max)();
      for (int i = 0; i < Dims; ++i)
        IdResult[i] = (std::min)(Limit, MaxWGs[Dims - i - 1]);
      return IdResult;
    }();
    auto M = (std::numeric_limits<uint32_t>::max)();
    range<Dims> MaxRange;
    for (int i = 0; i < Dims; ++i) {
      auto DesiredSize = MaxNWGs[i] * GoodFactor;
      MaxRange[i] =
          DesiredSize <= M ? DesiredSize : (M / GoodFactor) * GoodFactor;
    }

    bool DidAdjust = false;
    auto Adjust = [&](int Dim, size_t Value) {
      if (this->RangeRoundingTrace())
        std::cout << "parallel_for range adjusted at dim " << Dim << " from "
                  << RoundedRange[Dim] << " to " << Value << std::endl;
      RoundedRange[Dim] = Value;
      DidAdjust = true;
    };

    // Perform range rounding if there are sufficient work-items to
    // need rounding and the user-specified range is not a multiple of
    // a "good" value.
    if (RoundedRange[0] % MinFactorX != 0 && RoundedRange[0] >= MinRangeX) {
      // It is sufficient to round up just the first dimension.
      // Multiplying the rounded-up value of the first dimension
      // by the values of the remaining dimensions (if any)
      // will yield a rounded-up value for the total range.
      Adjust(0, ((RoundedRange[0] + GoodFactor - 1) / GoodFactor) * GoodFactor);
    }
#ifdef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
    // If we are forcing range rounding kernels to be used, we always want the
    // rounded range kernel to be generated, even if rounding isn't needed
    DidAdjust = true;
#endif // __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__

    for (int i = 0; i < Dims; ++i)
      if (RoundedRange[i] > MaxRange[i])
        Adjust(i, MaxRange[i]);

    if (!DidAdjust)
      return {range<Dims>{}, false};
    return {RoundedRange, true};
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
  template <
      typename KernelName, typename KernelType, int Dims,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void parallel_for_lambda_impl(range<Dims> UserRange, PropertiesT Props,
                                KernelType KernelFunc) {
    throwIfActionIsCreated();
    throwOnLocalAccessorMisuse<KernelName, KernelType>();
    if (!range_size_fits_in_size_t(UserRange))
      throw sycl::exception(make_error_code(errc::runtime),
                            "The total number of work-items in "
                            "a range must fit within size_t");

    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;

    // If 1D kernel argument is an integral type, convert it to sycl::item<1>
    // If user type is convertible from sycl::item/sycl::nd_item, use
    // sycl::item/sycl::nd_item to transport item information
    using TransformedArgType = std::conditional_t<
        std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
        typename TransformUserItemType<Dims, LambdaArgType>::type>;

    static_assert(!std::is_same_v<TransformedArgType, sycl::nd_item<Dims>>,
                  "Kernel argument cannot have a sycl::nd_item type in "
                  "sycl::parallel_for with sycl::range");

#if defined(SYCL2020_CONFORMANT_APIS) ||                                       \
    defined(__INTEL_PREVIEW_BREAKING_CHANGES)
    static_assert(std::is_convertible_v<item<Dims>, LambdaArgType> ||
                      std::is_convertible_v<item<Dims, false>, LambdaArgType>,
                  "sycl::parallel_for(sycl::range) kernel must have the "
                  "first argument of sycl::item type, or of a type which is "
                  "implicitly convertible from sycl::item");

    using RefLambdaArgType = std::add_lvalue_reference_t<LambdaArgType>;
    static_assert(
        (std::is_invocable_v<KernelType, RefLambdaArgType> ||
         std::is_invocable_v<KernelType, RefLambdaArgType, kernel_handler>),
        "SYCL kernel lambda/functor has an unexpected signature, it should be "
        "invocable with sycl::item and optionally sycl::kernel_handler");
#endif

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
    auto [RoundedRange, HasRoundedRange] = getRoundedRange(UserRange);
    if (HasRoundedRange) {
      using NameWT = typename detail::get_kernel_wrapper_name_t<NameT>::name;
      auto Wrapper =
          getRangeRoundedKernelLambda<NameWT, TransformedArgType, Dims>(
              KernelFunc, UserRange);

      using KName = std::conditional_t<std::is_same<KernelType, NameT>::value,
                                       decltype(Wrapper), NameWT>;

      kernel_parallel_for_wrapper<KName, TransformedArgType, decltype(Wrapper),
                                  PropertiesT>(Wrapper);
#ifndef __SYCL_DEVICE_ONLY__
      // We are executing over the rounded range, but there are still
      // items/ids that are are constructed in ther range rounded
      // kernel use items/ids in the user range, which means that
      // __SYCL_ASSUME_INT can still be violated. So check the bounds
      // of the user range, instead of the rounded range.
      detail::checkValueRange<Dims>(UserRange);
      MNDRDesc.set(RoundedRange);
      StoreLambda<KName, decltype(Wrapper), Dims, TransformedArgType>(
          std::move(Wrapper));
      setType(detail::CG::Kernel);
      setNDRangeUsed(false);
#endif
    } else
#endif // !__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ &&
       // !DPCPP_HOST_DEVICE_OPENMP && !DPCPP_HOST_DEVICE_PERF_NATIVE &&
       // SYCL_LANGUAGE_VERSION >= 202001
    {
      (void)UserRange;
      (void)Props;
#ifndef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
      // If parallel_for range rounding is forced then only range rounded
      // kernel is generated
      kernel_parallel_for_wrapper<NameT, TransformedArgType, KernelType,
                                  PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
      processProperties<NameT, PropertiesT>(Props);
      detail::checkValueRange<Dims>(UserRange);
      MNDRDesc.set(std::move(UserRange));
      StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
          std::move(KernelFunc));
      setType(detail::CG::Kernel);
      setNDRangeUsed(false);
#endif
#else
      (void)KernelFunc;
#endif // __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
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
  void parallel_for_impl(nd_range<Dims> ExecutionRange, PropertiesT Props,
                         _KERNELFUNCPARAM(KernelFunc)) {
    throwIfActionIsCreated();
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    verifyUsedKernelBundle(detail::KernelInfo<NameT>::getName());
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
#if defined(SYCL2020_CONFORMANT_APIS) ||                                       \
    defined(__INTEL_PREVIEW_BREAKING_CHANGES)
    static_assert(
        std::is_convertible_v<sycl::nd_item<Dims>, LambdaArgType>,
        "Kernel argument of a sycl::parallel_for with sycl::nd_range "
        "must be either sycl::nd_item or be convertible from sycl::nd_item");
    using TransformedArgType = sycl::nd_item<Dims>;
#else
    // If user type is convertible from sycl::item/sycl::nd_item, use
    // sycl::item/sycl::nd_item to transport item information
    using TransformedArgType =
        typename TransformUserItemType<Dims, LambdaArgType>::type;
#endif

    (void)ExecutionRange;
    (void)Props;
    kernel_parallel_for_wrapper<NameT, TransformedArgType, KernelType,
                                PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    processProperties<NameT, PropertiesT>(Props);
    detail::checkValueRange<Dims>(ExecutionRange);
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
        std::move(KernelFunc));
    setType(detail::CG::Kernel);
    setNDRangeUsed(true);
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
    setNDRangeUsed(false);
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
  template <
      typename KernelName, typename KernelType, int Dims,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void parallel_for_work_group_lambda_impl(range<Dims> NumWorkGroups,
                                           PropertiesT Props,
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
    (void)Props;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType, KernelType,
                                           PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    processProperties<NameT, PropertiesT>(Props);
    detail::checkValueRange<Dims>(NumWorkGroups);
    MNDRDesc.setNumWorkGroups(NumWorkGroups);
    StoreLambda<NameT, KernelType, Dims, LambdaArgType>(std::move(KernelFunc));
    setType(detail::CG::Kernel);
    setNDRangeUsed(false);
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
  template <
      typename KernelName, typename KernelType, int Dims,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void parallel_for_work_group_lambda_impl(range<Dims> NumWorkGroups,
                                           range<Dims> WorkGroupSize,
                                           PropertiesT Props,
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
    (void)Props;
    kernel_parallel_for_work_group_wrapper<NameT, LambdaArgType, KernelType,
                                           PropertiesT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    processProperties<NameT, PropertiesT>(Props);
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
      "sycl-single-task",
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      nullptr,
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
      "sycl-single-task",
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      nullptr,
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
  template <typename KernelName, typename KernelType, typename PropertiesT,
            bool HasKernelHandlerArg, typename FuncTy>
  void unpack(_KERNELFUNCPARAM(KernelFunc), FuncTy Lambda) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
    using MergedPropertiesT =
        typename detail::GetMergedKernelProperties<KernelType,
                                                   PropertiesT>::type;
    using Unpacker = KernelPropertiesUnpacker<MergedPropertiesT>;
#ifndef __SYCL_DEVICE_ONLY__
    // If there are properties provided by get method then process them.
    if constexpr (ext::oneapi::experimental::detail::
                      HasKernelPropertiesGetMethod<
                          _KERNELFUNCPARAMTYPE>::value) {
      processProperties<KernelName>(
          KernelFunc.get(ext::oneapi::experimental::properties_tag{}));
    }
#endif
    if constexpr (HasKernelHandlerArg) {
      kernel_handler KH;
      Lambda(Unpacker{}, this, KernelFunc, KH);
    } else {
      Lambda(Unpacker{}, this, KernelFunc);
    }
  }

  // NOTE: to support kernel_handler argument in kernel lambdas, only
  // kernel_***_wrapper functions must be called in this code

  template <
      typename KernelName, typename KernelType,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void kernel_single_task_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelName, KernelType, PropertiesT,
           detail::KernelLambdaHasKernelHandlerArgT<KernelType>::value>(
        KernelFunc, [&](auto Unpacker, auto... args) {
          Unpacker.template kernel_single_task_unpack<KernelName, KernelType>(
              args...);
        });
  }

  template <
      typename KernelName, typename ElementType, typename KernelType,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void kernel_parallel_for_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelName, KernelType, PropertiesT,
           detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                    ElementType>::value>(
        KernelFunc, [&](auto Unpacker, auto... args) {
          Unpacker.template kernel_parallel_for_unpack<KernelName, ElementType,
                                                       KernelType>(args...);
        });
  }

  template <
      typename KernelName, typename ElementType, typename KernelType,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void kernel_parallel_for_work_group_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
    unpack<KernelName, KernelType, PropertiesT,
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
  template <
      typename KernelName, typename KernelType,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void single_task_lambda_impl(PropertiesT Props,
                               _KERNELFUNCPARAM(KernelFunc)) {
    (void)Props;
    throwIfActionIsCreated();
    throwOnLocalAccessorMisuse<KernelName, KernelType>();
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
    processProperties<NameT, PropertiesT>(Props);
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
  std::enable_if_t<detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void()>::value ||
                   detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void(interop_handle)>::value>
  host_task_impl(FuncT &&Func) {
    throwIfActionIsCreated();

    MNDRDesc.set(range<1>(1));
    // Need to copy these rather than move so that we can check associated
    // accessors during finalize
    MArgs = MAssociatedAccesors;

    MHostTask.reset(new detail::HostTask(std::move(Func)));

    setType(detail::CG::CodeplayHostTask);
  }

  /// @brief Get the command graph if any associated with this handler. It can
  /// come from either the associated queue or from being set explicitly through
  /// the appropriate constructor.
  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
  getCommandGraph() const;

  /// Sets the user facing node type of this operation, used for operations
  /// which are recorded to a graph. Since some operations may actually be a
  /// different type than the user submitted, e.g. a fill() which is performed
  /// as a kernel submission.
  /// @param Type The actual type based on what handler functions the user
  /// called.
  void setUserFacingNodeType(ext::oneapi::experimental::node_type Type);

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

  /// Requires access to the memory object associated with the placeholder
  /// accessor contained in a dynamic_parameter object. Calling this function
  /// with a non-placeholder accessor has no effect.
  ///
  /// The command group has a requirement to gain access to the given memory
  /// object before executing.
  ///
  /// \param dynamicParamAcc is dynamic_parameter containing a SYCL accessor
  /// describing required memory region.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  void require(ext::oneapi::experimental::dynamic_parameter<
               accessor<DataT, Dims, AccMode, AccTarget, isPlaceholder>>
                   dynamicParamAcc) {
    using AccT = accessor<DataT, Dims, AccMode, AccTarget, isPlaceholder>;
    AccT Acc = *static_cast<AccT *>(
        detail::getValueFromDynamicParameter(dynamicParamAcc));
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
  using remove_cv_ref_t = typename std::remove_cv_t<std::remove_reference_t<T>>;

  template <typename U, typename T>
  using is_same_type = std::is_same<remove_cv_ref_t<U>, remove_cv_ref_t<T>>;

  template <typename T> struct ShouldEnableSetArg {
    static constexpr bool value =
        std::is_trivially_copyable_v<std::remove_reference_t<T>>
#if SYCL_LANGUAGE_VERSION && SYCL_LANGUAGE_VERSION <= 201707
            && std::is_standard_layout<std::remove_reference_t<T>>::value
#endif
        || is_same_type<sampler, T>::value // Sampler
        || (!is_same_type<cl_mem, T>::value &&
            std::is_pointer_v<remove_cv_ref_t<T>>) // USM
        || is_same_type<cl_mem, T>::value;         // Interop
  };

  /// Sets argument for OpenCL interoperability kernels.
  ///
  /// Registers Arg passed as argument # ArgIndex.
  ///
  /// \param ArgIndex is a positional number of argument to be set.
  /// \param Arg is an argument value to be set.
  template <typename T>
  typename std::enable_if_t<ShouldEnableSetArg<T>::value, void>
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

  // set_arg for graph dynamic_parameters
  template <typename T>
  void set_arg(int argIndex,
               ext::oneapi::experimental::dynamic_parameter<T> &dynamicParam) {
    setArgHelper(argIndex, dynamicParam);
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
    single_task_lambda_impl<KernelName>(
        ext::oneapi::experimental::empty_properties_t{}, KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<1> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<2> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<3> NumWorkItems, _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        std::move(KernelFunc));
  }

  /// Enqueues a command to the SYCL runtime to invoke \p Func once.
  template <typename FuncT>
  std::enable_if_t<detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void()>::value ||
                   detail::check_fn_signature<std::remove_reference_t<FuncT>,
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
    using TransformedArgType = std::conditional_t<
        std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
        typename TransformUserItemType<Dims, LambdaArgType>::type>;
    (void)NumWorkItems;
    (void)WorkItemOffset;
    kernel_parallel_for_wrapper<NameT, TransformedArgType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    detail::checkValueRange<Dims>(NumWorkItems, WorkItemOffset);
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
        std::move(KernelFunc));
    setType(detail::CG::Kernel);
    setNDRangeUsed(false);
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
    parallel_for_work_group_lambda_impl<KernelName>(
        NumWorkGroups, ext::oneapi::experimental::empty_properties_t{},
        KernelFunc);
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
    parallel_for_work_group_lambda_impl<KernelName>(
        NumWorkGroups, WorkGroupSize,
        ext::oneapi::experimental::empty_properties_t{}, KernelFunc);
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
    setNDRangeUsed(false);
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
    setNDRangeUsed(true);
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
    setNDRangeUsed(false);
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
    setNDRangeUsed(false);
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
    setNDRangeUsed(true);
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
  single_task(PropertiesT Props, _KERNELFUNCPARAM(KernelFunc)) {
    single_task_lambda_impl<KernelName, KernelType, PropertiesT>(Props,
                                                                 KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<1> NumWorkItems, PropertiesT Props,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 1, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<2> NumWorkItems, PropertiesT Props,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 2, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<3> NumWorkItems, PropertiesT Props,
               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_lambda_impl<KernelName, KernelType, 3, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
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

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<1> Range, PropertiesT Properties, RestT &&...Rest) {
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<2> Range, PropertiesT Properties, RestT &&...Rest) {
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(range<3> Range, PropertiesT Properties, RestT &&...Rest) {
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(range<1> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(range<2> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(range<3> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename PropertiesT, typename... RestT>
  std::enable_if_t<
      (sizeof...(RestT) > 1) &&
      detail::AreAllButLastReductions<RestT...>::value &&
      ext::oneapi::experimental::is_property_list<PropertiesT>::value>
  parallel_for(nd_range<Dims> Range, PropertiesT Properties, RestT &&...Rest) {
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value>
  parallel_for(nd_range<Dims> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  /// }@

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename PropertiesT>
  void parallel_for_work_group(range<Dims> NumWorkGroups, PropertiesT Props,
                               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_work_group_lambda_impl<KernelName, KernelType, Dims,
                                        PropertiesT>(NumWorkGroups, Props,
                                                     KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename PropertiesT>
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize, PropertiesT Props,
                               _KERNELFUNCPARAM(KernelFunc)) {
    parallel_for_work_group_lambda_impl<KernelName, KernelType, Dims,
                                        PropertiesT>(
        NumWorkGroups, WorkGroupSize, Props, KernelFunc);
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
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForSourceAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    CGData.MSharedPtrStorage.push_back(Dst);
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
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // TODO: Add static_assert with is_device_copyable when vec is
    // device-copyable.
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    CGData.MSharedPtrStorage.push_back(Src);
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
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);

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

    CGData.MRequirements.push_back(AccImpl.get());
    MSrcPtr = static_cast<void *>(AccImpl.get());
    MDstPtr = static_cast<void *>(Dst);
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    CGData.MAccStorage.push_back(std::move(AccImpl));
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
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // TODO: Add static_assert with is_device_copyable when vec is
    // device-copyable.
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

    CGData.MRequirements.push_back(AccImpl.get());
    MSrcPtr = const_cast<T_Src *>(Src);
    MDstPtr = static_cast<void *>(AccImpl.get());
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    CGData.MAccStorage.push_back(std::move(AccImpl));
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
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);

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

    CGData.MRequirements.push_back(AccImplSrc.get());
    CGData.MRequirements.push_back(AccImplDst.get());
    MSrcPtr = AccImplSrc.get();
    MDstPtr = AccImplDst.get();
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    CGData.MAccStorage.push_back(std::move(AccImplSrc));
    CGData.MAccStorage.push_back(std::move(AccImplDst));
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
    if (Acc.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Acc);

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the update_host method.");
    setType(detail::CG::UpdateHost);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = static_cast<void *>(AccImpl.get());
    CGData.MRequirements.push_back(AccImpl.get());
    CGData.MAccStorage.push_back(std::move(AccImpl));
  }

public:
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
    assert(!MIsHost && "fill() should no longer be callable on a host device.");

    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);

    throwIfActionIsCreated();
    setUserFacingNodeType(ext::oneapi::experimental::node_type::memfill);
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the fill method.");
    // CG::Fill will result in piEnqueuFillBuffer/Image which requires that mem
    // data is contiguous. Thus we check range and offset when dim > 1
    // Images don't allow ranged accessors and are fine.
    if constexpr (isBackendSupportedFillSize(sizeof(T)) &&
                  ((Dims <= 1) || isImageOrImageArray(AccessTarget))) {
      StageFillCG(Dst, Pattern);
    } else if constexpr (Dims == 0) {
      // Special case for zero-dim accessors.
      parallel_for<__fill<T, Dims, AccessMode, AccessTarget, IsPlaceholder>>(
          range<1>(1), [=](id<1>) { Dst = Pattern; });
    } else {
      // Dim > 1
      bool OffsetUsable = (Dst.get_offset() == sycl::id<Dims>{});
      detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
      bool RangesUsable =
          (AccBase->getAccessRange() == AccBase->getMemoryRange());
      if (OffsetUsable && RangesUsable &&
          isBackendSupportedFillSize(sizeof(T))) {
        StageFillCG(Dst, Pattern);
      } else {
        range<Dims> Range = Dst.get_range();
        parallel_for<__fill<T, Dims, AccessMode, AccessTarget, IsPlaceholder>>(
            Range, [=](id<Dims> Index) { Dst[Index] = Pattern; });
      }
    }
  }

  /// Fills the specified memory with the specified pattern.
  ///
  /// \param Ptr is the pointer to the memory to fill
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// device copyable.
  /// \param Count is the number of times to fill Pattern into Ptr.
  template <typename T> void fill(void *Ptr, const T &Pattern, size_t Count) {
    throwIfActionIsCreated();
    setUserFacingNodeType(ext::oneapi::experimental::node_type::memfill);
    static_assert(is_device_copyable<T>::value,
                  "Pattern must be device copyable");
    parallel_for<__usmfill<T>>(range<1>(Count), [=](id<1> Index) {
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
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then the barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  void ext_oneapi_barrier(const std::vector<event> &WaitList);

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on this handler's
  /// device.
  /// No operations is done if \p Count is zero. An exception is thrown if
  /// either \p Dest or \p Src is nullptr. The behavior is undefined if any of
  /// the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  void memcpy(void *Dest, const void *Src, size_t Count);

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on this handler's
  /// device.
  /// No operations is done if \p Count is zero. An exception is thrown if
  /// either \p Dest or \p Src is nullptr. The behavior is undefined if any of
  /// the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  template <typename T> void copy(const T *Src, T *Dest, size_t Count) {
    this->memcpy(Dest, Src, Count * sizeof(T));
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \p Count is zero. An exception is thrown if \p
  /// Dest is nullptr. The behavior is undefined if \p Dest is invalid.
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
  /// No operations is done if \p Width or \p Height is zero. An exception is
  /// thrown if either \p Dest or \p Src is nullptr or if \p Width is strictly
  /// greater than either \p DestPitch or \p SrcPitch. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \p Dest.
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \p Src.
  /// \param Width is the width in bytes of the 2D region to copy.
  /// \param Height is the height in number of row of the 2D region to copy.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  void ext_oneapi_memcpy2d(void *Dest, size_t DestPitch, const void *Src,
                           size_t SrcPitch, size_t Width, size_t Height);

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \p Width or \p Height is zero. An exception is
  /// thrown if either \p Dest or \p Src is nullptr or if \p Width is strictly
  /// greater than either \p DestPitch or \p SrcPitch. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \p Src.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \p Dest.
  /// \param Width is the width in number of elements of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  template <typename T>
  void ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                         size_t DestPitch, size_t Width, size_t Height);

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \p Width or \p Height is zero. An exception is
  /// thrown if either \p Dest or \p Src is nullptr or if \p Width is strictly
  /// greater than \p DestPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \p Dest.
  /// \param Value is the value to fill into the region in \p Dest. Value is
  /// cast as an unsigned char.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  void ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                           size_t Width, size_t Height);

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \p Width or \p Height is zero. An exception is
  /// thrown if either \p Dest or \p Src is nullptr or if \p Width is strictly
  /// greater than \p DestPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \p Dest.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// device copyable.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  template <typename T>
  void ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                         size_t Width, size_t Height);

  /// Copies data from a USM memory region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \p Dest, as specified through \p NumBytes and \p DestOffset.
  ///
  /// \param Dest is the destination device_glboal.
  /// \param Src is a USM pointer to the source memory.
  /// \param NumBytes is a number of bytes to copy.
  /// \param DestOffset is the offset into \p Dest to copy to.
  template <typename T, typename PropertyListT>
  void memcpy(ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
              const void *Src, size_t NumBytes = sizeof(T),
              size_t DestOffset = 0) {
    throwIfGraphAssociated<
        ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
            sycl_ext_oneapi_device_global>();
    if (sizeof(T) < DestOffset + NumBytes)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Copy to device_global is out of bounds.");

    constexpr bool IsDeviceImageScoped = PropertyListT::template has_property<
        ext::oneapi::experimental::device_image_scope_key>();

    if (!detail::isDeviceGlobalUsedInKernel(&Dest)) {
      // If the corresponding device_global isn't used in any kernels, we fall
      // back to doing the memory operation on host-only.
      memcpyToHostOnlyDeviceGlobal(&Dest, Src, sizeof(T), IsDeviceImageScoped,
                                   NumBytes, DestOffset);
      return;
    }

    memcpyToDeviceGlobal(&Dest, Src, IsDeviceImageScoped, NumBytes, DestOffset);
  }

  /// Copies data from a device_global to USM memory.
  /// Throws an exception if the copy operation intends to read outside the
  /// memory range \p Src, as specified through \p NumBytes and \p SrcOffset.
  ///
  /// \param Dest is a USM pointer to copy to.
  /// \param Src is the source device_global.
  /// \param NumBytes is a number of bytes to copy.
  /// \param SrcOffset is the offset into \p Src to copy from.
  template <typename T, typename PropertyListT>
  void
  memcpy(void *Dest,
         const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
         size_t NumBytes = sizeof(T), size_t SrcOffset = 0) {
    throwIfGraphAssociated<
        ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
            sycl_ext_oneapi_device_global>();
    if (sizeof(T) < SrcOffset + NumBytes)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Copy from device_global is out of bounds.");

    constexpr bool IsDeviceImageScoped = PropertyListT::template has_property<
        ext::oneapi::experimental::device_image_scope_key>();

    if (!detail::isDeviceGlobalUsedInKernel(&Src)) {
      // If the corresponding device_global isn't used in any kernels, we fall
      // back to doing the memory operation on host-only.
      memcpyFromHostOnlyDeviceGlobal(Dest, &Src, IsDeviceImageScoped, NumBytes,
                                     SrcOffset);
      return;
    }

    memcpyFromDeviceGlobal(Dest, &Src, IsDeviceImageScoped, NumBytes,
                           SrcOffset);
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a USM memory
  /// region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \p Dest, as specified through \p Count and \p StartIndex.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is the destination device_glboal.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in \p Dest to copy to.
  template <typename T, typename PropertyListT>
  void copy(const std::remove_all_extents_t<T> *Src,
            ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
            size_t Count = sizeof(T) / sizeof(std::remove_all_extents_t<T>),
            size_t StartIndex = 0) {
    this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                 StartIndex * sizeof(std::remove_all_extents_t<T>));
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a
  /// device_global to a USM memory region.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \p Src, as specified through \p Count and \p StartIndex.
  ///
  /// \param Src is the source device_global.
  /// \param Dest is a USM pointer to copy to.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in \p Src to copy
  ///        from.
  template <typename T, typename PropertyListT>
  void
  copy(const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
       std::remove_all_extents_t<T> *Dest,
       size_t Count = sizeof(T) / sizeof(std::remove_all_extents_t<T>),
       size_t StartIndex = 0) {
    this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                 StartIndex * sizeof(std::remove_all_extents_t<T>));
  }
  /// Executes a command_graph.
  ///
  /// \param Graph Executable command_graph to run
  void ext_oneapi_graph(ext::oneapi::experimental::command_graph<
                        ext::oneapi::experimental::graph_state::executable>
                            Graph);

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle. An exception is
  /// thrown if either \p Src is nullptr or \p Dest is incomplete. The behavior
  /// is undefined if \p Desc is inconsistent with the allocated memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestImgDesc is the image descriptor
  void ext_oneapi_copy(
      void *Src, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc);

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region. Pixel size is determined
  /// by \p DestImgDesc
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the origin where the x, y, and z
  ///                  components are measured in bytes, rows, and slices
  ///                  respectively
  /// \param SrcExtent is the extent of the source memory to copy, measured in
  ///                  pixels (pixel size determined by \p DestImgDesc )
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels as determined by \p DestImgDesc
  void ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent);

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer.
  /// An exception is thrown if either \p Src is incomplete or \p Dest is
  /// nullptr. The behavior is undefined if \p Desc is inconsistent with the
  /// allocated memory region.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param SrcImgDesc is the source image descriptor
  void ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc);

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p Extent are
  /// used to determine the sub-region.  Pixel size is determined
  /// by \p SrcImgDesc
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the origin of source measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DestExtent is the extent of the dest memory to copy, measured in
  ///                  pixels (pixel size determined by \p DestImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  void
  ext_oneapi_copy(ext::oneapi::experimental::image_mem_handle Src,
                  sycl::range<3> SrcOffset,
                  const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
                  void *Dest, sycl::range<3> DestOffset,
                  sycl::range<3> DestExtent, sycl::range<3> CopyExtent);

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. An exception is thrown if either \p Src is nullptr, \p
  /// Dest is nullptr, or \p Pitch is inconsistent with hardware requirements.
  /// The behavior is undefined if \p Desc is inconsistent with the allocated
  /// memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DeviceImgDesc is the image descriptor (format, order, dimensions).
  /// \param DeviceRowPitch is the pitch of the rows on the device.
  void ext_oneapi_copy(
      void *Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch);

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. Allows for a sub-region copy, where \p SrcOffset ,
  /// \p DestOffset , and \p Extent are used to determine the sub-region.
  /// Pixel size is determined by \p DestImgDesc
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DeviceImgDesc is the device image descriptor
  /// \param DeviceRowPitch is the row pitch on the device
  /// \param HostExtent is the extent of the dest memory to copy, measured in
  ///                  pixels (pixel size determined by \p DeviceImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p DeviceImgDesc )
  void ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, sycl::range<3> HostExtent,
      sycl::range<3> CopyExtent);

  /// Instruct the queue with a non-blocking wait on an external semaphore.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  void ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle
          SemaphoreHandle);

  /// Instruct the queue to signal the external semaphore once all previous
  /// commands have completed execution.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  void ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle
          SemaphoreHandle);

private:
  std::shared_ptr<detail::handler_impl> MImpl;
  std::shared_ptr<detail::queue_impl> MQueue;

  /// The storage for the arguments passed.
  /// We need to store a copy of values that are passed explicitly through
  /// set_arg, require and so on, because we need them to be alive after
  /// we exit the method they are passed in.
  mutable detail::CG::StorageInitHelper CGData;
  std::vector<detail::LocalAccessorImplPtr> MLocalAccStorage;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreamStorage;
  /// The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;
  /// The list of associated accessors with this handler.
  /// These accessors were created with this handler as argument or
  /// have become required for this handler via require method.
  std::vector<detail::ArgDesc> MAssociatedAccesors;
  /// Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;
  detail::string MKernelName;
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
  /// The list of valid SYCL events that need to complete
  /// before barrier command can be executed
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  /// The graph that is associated with this handler.
  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph;
  /// If we are submitting a graph using ext_oneapi_graph this will be the graph
  /// to be executed.
  std::shared_ptr<ext::oneapi::experimental::detail::exec_graph_impl>
      MExecGraph;
  /// Storage for a node created from a subgraph submission.
  std::shared_ptr<ext::oneapi::experimental::detail::node_impl> MSubgraphNode;
  /// Storage for the CG created when handling graph nodes added explicitly.
  std::unique_ptr<detail::CG> MGraphNodeCG;

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
            bool ExplicitIdentity, typename RedOutVar>
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
  friend void detail::associateWithHandler(
      handler &, detail::UnsampledImageAccessorBaseHost *, image_target);
  friend void detail::associateWithHandler(
      handler &, detail::SampledImageAccessorBaseHost *, image_target);
#endif

  friend class ::MockHandler;
  friend class detail::queue_impl;

  // Make pipe class friend to be able to call ext_intel_read/write_host_pipe
  // method.
  template <class _name, class _dataT, int32_t _min_capacity,
            class _propertiesT, class>
  friend class ext::intel::experimental::pipe;

  /// Read from a host pipe given a host address and
  /// \param Name name of the host pipe to be passed into lower level runtime
  /// \param Ptr host pointer of host pipe as identified by address of its const
  ///        expr m_Storage member
  /// \param Size the size of data getting read back / to.
  /// \param Block if read operation is blocking, default to false.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  void ext_intel_read_host_pipe(const std::string &Name, void *Ptr, size_t Size,
                                bool Block = false) {
    ext_intel_read_host_pipe(detail::string_view(Name), Ptr, Size, Block);
  }
  void ext_intel_read_host_pipe(detail::string_view Name, void *Ptr,
                                size_t Size, bool Block = false);
#else
  void ext_intel_read_host_pipe(const std::string &Name, void *Ptr, size_t Size,
                                bool Block = false);
#endif

  /// Write to host pipes given a host address and
  /// \param Name name of the host pipe to be passed into lower level runtime
  /// \param Ptr host pointer of host pipe as identified by address of its const
  /// expr m_Storage member
  /// \param Size the size of data getting read back / to.
  /// \param Block if write opeartion is blocking, default to false.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  void ext_intel_write_host_pipe(const std::string &Name, void *Ptr,
                                 size_t Size, bool Block = false) {
    ext_intel_write_host_pipe(detail::string_view(Name), Ptr, Size, Block);
  }
  void ext_intel_write_host_pipe(detail::string_view Name, void *Ptr,
                                 size_t Size, bool Block = false);
#else
  void ext_intel_write_host_pipe(const std::string &Name, void *Ptr,
                                 size_t Size, bool Block = false);
#endif
  friend class ext::oneapi::experimental::detail::graph_impl;
  friend class ext::oneapi::experimental::detail::dynamic_parameter_impl;

  bool DisableRangeRounding();

  bool RangeRoundingTrace();

  void GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                                size_t &MinRange);

  template <typename WrapperT, typename TransformedArgType, int Dims,
            typename KernelType,
            std::enable_if_t<detail::KernelLambdaHasKernelHandlerArgT<
                KernelType, TransformedArgType>::value> * = nullptr>
  auto getRangeRoundedKernelLambda(KernelType KernelFunc,
                                   range<Dims> UserRange) {
    return detail::RoundedRangeKernelWithKH<TransformedArgType, Dims,
                                            KernelType>{UserRange, KernelFunc};
  }

  template <typename WrapperT, typename TransformedArgType, int Dims,
            typename KernelType,
            std::enable_if_t<!detail::KernelLambdaHasKernelHandlerArgT<
                KernelType, TransformedArgType>::value> * = nullptr>
  auto getRangeRoundedKernelLambda(KernelType KernelFunc,
                                   range<Dims> UserRange) {
    return detail::RoundedRangeKernel<TransformedArgType, Dims, KernelType>{
        UserRange, KernelFunc};
  }

  const std::shared_ptr<detail::context_impl> &getContextImplPtr() const;

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
    // Otherwise the data is accessible on the device so we do the operation
    // there instead.
    // Limit number of work items to be resistant to big copies.
    id<2> Chunk = computeFallbackKernelBounds(Height, Width);
    id<2> Iterations = (Chunk + id<2>{Height, Width} - 1) / Chunk;
    parallel_for<__usmmemcpy2d<T>>(
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

  // Common function for launching a 2D USM memcpy host-task to avoid
  // redefinitions of the kernel from copy and memcpy.
  template <typename T>
  void commonUSMCopy2DFallbackHostTask(const void *Src, size_t SrcPitch,
                                       void *Dest, size_t DestPitch,
                                       size_t Width, size_t Height) {
    // If both pointers are host USM or unknown (assumed non-USM) we use a
    // host-task to satisfy dependencies.
    host_task([=] {
      const T *CastedSrc = static_cast<const T *>(Src);
      T *CastedDest = static_cast<T *>(Dest);
      for (size_t I = 0; I < Height; ++I) {
        const T *SrcItBegin = CastedSrc + SrcPitch * I;
        T *DestItBegin = CastedDest + DestPitch * I;
        std::copy(SrcItBegin, SrcItBegin + Width, DestItBegin);
      }
    });
  }

  // StageFillCG()  Supporting function to fill()
  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t,
            typename PropertyListT = property_list>
  void StageFillCG(
      accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder, PropertyListT>
          Dst,
      const T &Pattern) {
    setType(detail::CG::Fill);
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = static_cast<void *>(AccImpl.get());
    CGData.MRequirements.push_back(AccImpl.get());
    CGData.MAccStorage.push_back(std::move(AccImpl));

    MPattern.resize(sizeof(T));
    auto PatternPtr = reinterpret_cast<T *>(MPattern.data());
    *PatternPtr = Pattern;
  }

  // Common function for launching a 2D USM fill kernel to avoid redefinitions
  // of the kernel from memset and fill.
  template <typename T>
  void commonUSMFill2DFallbackKernel(void *Dest, size_t DestPitch,
                                     const T &Pattern, size_t Width,
                                     size_t Height) {
    // Otherwise the data is accessible on the device so we do the operation
    // there instead.
    // Limit number of work items to be resistant to big fill operations.
    id<2> Chunk = computeFallbackKernelBounds(Height, Width);
    id<2> Iterations = (Chunk + id<2>{Height, Width} - 1) / Chunk;
    parallel_for<__usmfill2d<T>>(
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

  // Common function for launching a 2D USM fill kernel or host_task to avoid
  // redefinitions of the kernel from memset and fill.
  template <typename T>
  void commonUSMFill2DFallbackHostTask(void *Dest, size_t DestPitch,
                                       const T &Pattern, size_t Width,
                                       size_t Height) {
    // If the pointer is host USM or unknown (assumed non-USM) we use a
    // host-task to satisfy dependencies.
    host_task([=] {
      T *CastedDest = static_cast<T *>(Dest);
      for (size_t I = 0; I < Height; ++I) {
        T *ItBegin = CastedDest + DestPitch * I;
        std::fill(ItBegin, ItBegin + Width, Pattern);
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

  // Implementation of memcpy to device_global.
  void memcpyToDeviceGlobal(const void *DeviceGlobalPtr, const void *Src,
                            bool IsDeviceImageScoped, size_t NumBytes,
                            size_t Offset);

  // Implementation of memcpy from device_global.
  void memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                              bool IsDeviceImageScoped, size_t NumBytes,
                              size_t Offset);

  // Implementation of memcpy to an unregistered device_global.
  void memcpyToHostOnlyDeviceGlobal(const void *DeviceGlobalPtr,
                                    const void *Src, size_t DeviceGlobalTSize,
                                    bool IsDeviceImageScoped, size_t NumBytes,
                                    size_t Offset);

  // Implementation of memcpy from an unregistered device_global.
  void memcpyFromHostOnlyDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                      bool IsDeviceImageScoped, size_t NumBytes,
                                      size_t Offset);

  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t,
            typename PropertyListT = property_list>
  void checkIfPlaceholderIsBoundToHandler(
      accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder, PropertyListT>
          Acc) {
    auto *AccBase = reinterpret_cast<detail::AccessorBaseHost *>(&Acc);
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::AccessorImplHost *Req = AccImpl.get();
    if (std::find_if(MAssociatedAccesors.begin(), MAssociatedAccesors.end(),
                     [&](const detail::ArgDesc &AD) {
                       return AD.MType ==
                                  detail::kernel_param_kind_t::kind_accessor &&
                              AD.MPtr == Req &&
                              AD.MSize == static_cast<int>(AccessTarget);
                     }) == MAssociatedAccesors.end())
      throw sycl::exception(make_error_code(errc::kernel_argument),
                            "placeholder accessor must be bound by calling "
                            "handler::require() before it can be used.");
  }

  // Set value of the gpu cache configuration for the kernel.
  void setKernelCacheConfig(sycl::detail::pi::PiKernelCacheConfig);
  // Set value of the kernel is cooperative flag
  void setKernelIsCooperative(bool);

  template <
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures FeatureT>
  void throwIfGraphAssociated() const {

    if (getCommandGraph()) {
      std::string FeatureString =
          ext::oneapi::experimental::detail::UnsupportedFeatureToString(
              FeatureT);
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "The " + FeatureString +
                                " feature is not yet available "
                                "for use with the SYCL Graph extension.");
    }
  }

  // Set that an ND Range was used during a call to parallel_for
  void setNDRangeUsed(bool Value);
};
} // namespace _V1
} // namespace sycl
