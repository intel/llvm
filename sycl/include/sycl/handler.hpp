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
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/id_queries_fit_in_int.hpp>
#include <sycl/detail/impl_utils.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/kernel_launch_helper.hpp>
#include <sycl/detail/kernel_name_based_cache.hpp>
#include <sycl/detail/kernel_name_str_t.hpp>
#include <sycl/detail/reduction_forward.hpp>
#include <sycl/detail/string.hpp>
#include <sycl/detail/string_view.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/ext/oneapi/bindless_images_mem_handle.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <sycl/ext/oneapi/device_global/properties.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/experimental/use_root_sync_prop.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/group.hpp>
#include <sycl/id.hpp>
#include <sycl/item.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle_enums.hpp>
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

template <bundle_state State> class kernel_bundle;
class handler;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

namespace ext::intel::experimental {
template <class _name, class _dataT, int32_t _min_capacity, class _propertiesT,
          class>
class pipe;
}

namespace ext ::oneapi ::experimental {
template <typename, typename> class work_group_memory;
template <typename, typename> class dynamic_work_group_memory;
struct image_descriptor;
__SYCL_EXPORT void async_free(sycl::handler &h, void *ptr);
__SYCL_EXPORT void *async_malloc(sycl::handler &h, sycl::usm::alloc kind,
                                 size_t size);
__SYCL_EXPORT void *async_malloc_from_pool(sycl::handler &h, size_t size,
                                           const memory_pool &pool);
} // namespace ext::oneapi::experimental

namespace ext::oneapi::experimental::detail {
class dynamic_parameter_base;
class dynamic_work_group_memory_base;
class dynamic_local_accessor_base;
class graph_impl;
class dynamic_parameter_impl;
} // namespace ext::oneapi::experimental::detail
namespace detail {

class type_erased_cgfo_ty {
  // From SYCL 2020,  command group function object:
  // A type which is callable with operator() that takes a reference to a
  // command group handler, that defines a command group which can be submitted
  // by a queue. The function object can be a named type, lambda function or
  // std::function.
  template <typename T> struct invoker {
    static void call(const void *object, handler &cgh) {
      (*const_cast<T *>(static_cast<const T *>(object)))(cgh);
    }
  };
  const void *object;
  using invoker_ty = void (*)(const void *, handler &);
  const invoker_ty invoker_f;

public:
  template <class T>
  type_erased_cgfo_ty(T &&f)
      // NOTE: Even if `f` is a pointer to a function, `&f` is a pointer to a
      // pointer to a function and as such can be casted to `void *` (pointer to
      // a function cannot be casted).
      : object(static_cast<const void *>(&f)),
        invoker_f(&invoker<std::remove_reference_t<T>>::call) {}
  ~type_erased_cgfo_ty() = default;

  type_erased_cgfo_ty(const type_erased_cgfo_ty &) = delete;
  type_erased_cgfo_ty(type_erased_cgfo_ty &&) = delete;
  type_erased_cgfo_ty &operator=(const type_erased_cgfo_ty &) = delete;
  type_erased_cgfo_ty &operator=(type_erased_cgfo_ty &&) = delete;

  void operator()(handler &cgh) const { invoker_f(object, cgh); }
};

class kernel_bundle_impl;
class work_group_memory_impl;
class handler_impl;
class kernel_impl;
class queue_impl;
class stream_impl;
class event_impl;
class context_impl;
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor;
class HandlerAccess;
class HostTask;

using EventImplPtr = std::shared_ptr<event_impl>;

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
device_impl &getDeviceImplFromHandler(handler &);

// Checks if a device_global has any registered kernel usage.
__SYCL_EXPORT bool isDeviceGlobalUsedInKernel(const void *DeviceGlobalPtr);

// Extracts a pointer to the value inside a dynamic parameter
__SYCL_EXPORT void *getValueFromDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_base
        &DynamicParamBase);

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

  // Copy the properties_tag getter from the original kernel to propagate
  // property(s)
  template <
      typename T = KernelType,
      typename = std::enable_if_t<ext::oneapi::experimental::detail::
                                      HasKernelPropertiesGetMethod<T>::value>>
  auto get(ext::oneapi::experimental::properties_tag) const {
    return KernelFunc.get(ext::oneapi::experimental::properties_tag{});
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

  // Copy the properties_tag getter from the original kernel to propagate
  // property(s)
  template <
      typename T = KernelType,
      typename = std::enable_if_t<ext::oneapi::experimental::detail::
                                      HasKernelPropertiesGetMethod<T>::value>>
  auto get(ext::oneapi::experimental::properties_tag) const {
    return KernelFunc.get(ext::oneapi::experimental::properties_tag{});
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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Constructs SYCL handler from the pre-constructed stack-allocated
  /// `handler_impl` (not enforced, but meaningless to do a heap allocation
  /// outside handler instance).
  ///
  /// \param HandlerImpl is a pre-constructed handler_impl.
  //
  // Can't provide this overload outside preview because `handler` lacks
  // required data members.
  handler(detail::handler_impl &HandlerImpl);
#else
  /// Constructs SYCL handler from queue.
  ///
  /// \param Queue is a SYCL queue.
  /// \param CallerNeedsEvent indicates if the event resulting from this handler
  ///        is needed by the caller.
  handler(std::shared_ptr<detail::queue_impl> Queue, bool CallerNeedsEvent);
  /// Constructs SYCL handler from the associated queue and the submission's
  /// primary and secondary queue.
  ///
  /// \param Queue is a SYCL queue. This is equal to either PrimaryQueue or
  ///        SecondaryQueue.
  /// \param PrimaryQueue is the primary SYCL queue of the submission.
  /// \param SecondaryQueue is the secondary SYCL queue of the submission. This
  ///        is null if no secondary queue is associated with the submission.
  /// \param CallerNeedsEvent indicates if the event resulting from this handler
  ///        is needed by the caller.
  handler(std::shared_ptr<detail::queue_impl> Queue,
          std::shared_ptr<detail::queue_impl> PrimaryQueue,
          std::shared_ptr<detail::queue_impl> SecondaryQueue,
          bool CallerNeedsEvent);
  __SYCL_DLL_LOCAL handler(std::shared_ptr<detail::queue_impl> Queue,
                           detail::queue_impl *SecondaryQueue,
                           bool CallerNeedsEvent);

  /// Constructs SYCL handler from Graph.
  ///
  /// The handler will add the command-group as a node to the graph rather than
  /// enqueueing it straight away.
  ///
  /// \param Graph is a SYCL command_graph
  handler(std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph);
#endif
  handler(std::unique_ptr<detail::handler_impl> &&HandlerImpl);

  ~handler();

  void *storeRawArg(const void *Ptr, size_t Size);

  void *
  storeRawArg(const sycl::ext::oneapi::experimental::raw_kernel_arg &RKA) {
    return storeRawArg(RKA.MArgData, RKA.MArgSize);
  }

  /// Stores copy of Arg passed to the argument storage.
  template <typename T> void *storePlainArg(T &&Arg) {
    return storeRawArg(&Arg, sizeof(T));
  }

  void setType(detail::CGType Type);

  detail::CGType getType() const;

  void throwIfActionIsCreated() {
    if (detail::CGType::None != getType())
      throw sycl::exception(make_error_code(errc::runtime),
                            "Attempt to set multiple actions for the "
                            "command group. Command group must consist of "
                            "a single kernel or explicit memory operation.");
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // TODO: Those functions are not used anymore, remove it in the next
  // ABI-breaking window.
  void extractArgsAndReqsFromLambda(
      char *LambdaPtr,
      const std::vector<detail::kernel_param_desc_t> &ParamDescs, bool IsESIMD);
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs,
                               bool IsESIMD);
#endif
  /// Extracts and prepares kernel arguments from the lambda using information
  /// from the built-ins or integration header.
  void extractArgsAndReqsFromLambda(
      char *LambdaPtr, detail::kernel_param_desc_t (*ParamDescGetter)(int),
      size_t NumKernelParams, bool IsESIMD);

  /// Extracts and prepares kernel arguments set via set_arg(s).
  void extractArgsAndReqs();

#if defined(__INTEL_PREVIEW_BREAKING_CHANGES)
  // TODO: processArg need not to be public
  __SYCL_DLL_LOCAL
#endif
  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource, bool IsESIMD);

  /// \return a string containing name of SYCL kernel.
  detail::ABINeutralKernelNameStrT getKernelName();

  template <typename LambdaNameT> bool lambdaAndKernelHaveEqualName() {
    // TODO It is unclear a kernel and a lambda/functor must to be equal or not
    // for parallel_for with sycl::kernel and lambda/functor together
    // Now if they are equal we extract arguments from lambda/functor for the
    // kernel. Else it is necessary use set_atg(s) for resolve the order and
    // values of arguments for the kernel.
    assert(MKernel && "MKernel is not initialized");
    constexpr std::string_view LambdaName =
        detail::getKernelName<LambdaNameT>();
    detail::ABINeutralKernelNameStrT KernelName = getKernelName();
    return KernelName == LambdaName;
  }

  /// Saves the location of user's code passed in \p CodeLoc for future usage in
  /// finalize() method.
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void saveCodeLoc(detail::code_location CodeLoc);
#endif
  void saveCodeLoc(detail::code_location CodeLoc, bool IsTopCodeLoc);
  void copyCodeLoc(const handler &other);

  /// Constructs CG object of specific type, passes it to Scheduler and
  /// returns sycl::event object representing the command group.
  /// It's expected that the method is the latest method executed before
  /// object destruction.
  ///
  /// \return a SYCL event object representing the command group
  ///
  /// Note: in preview mode, handler.finalize() is expected to return
  /// nullptr if the event is not needed (discarded).
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  detail::EventImplPtr finalize();
#else
  event finalize();
#endif

  /// Constructs CG object of specific type, passes it to Scheduler and
  /// returns sycl::event object representing the command group.
  /// It's expected that the method is the latest method executed before
  /// object destruction.
  /// \param CallerNeedsEvent Specifies if the caller needs an event
  /// representing the work related to this handler.
  ///
  /// \return a SYCL event object representing the command group
  event finalize(bool CallerNeedsEvent);

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
    set_arg(ArgIndex, std::forward<T>(Arg));
    setArgsHelper(++ArgIndex, std::forward<Ts>(Args)...);
  }

  void setArgsHelper(int) {}

  void setLocalAccessorArgHelper(int ArgIndex,
                                 detail::LocalAccessorBaseHost &LocalAccBase) {
    detail::LocalAccessorImplPtr LocalAccImpl =
        detail::getSyclObjImpl(LocalAccBase);
    detail::LocalAccessorImplHost *Req = LocalAccImpl.get();
    MLocalAccStorage.push_back(std::move(LocalAccImpl));
    addArg(detail::kernel_param_kind_t::kind_accessor, Req,
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

  void setArgHelper(int ArgIndex, detail::work_group_memory_impl &Arg);

  // setArgHelper for non local accessor argument.
  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  typename std::enable_if_t<AccessTarget != access::target::local, void>
  setArgHelper(
      int ArgIndex,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder> &&Arg) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Arg;
    const detail::AccessorImplPtr &AccImpl = detail::getSyclObjImpl(*AccBase);
    detail::AccessorImplHost *Req = AccImpl.get();
    // Add accessor to the list of arguments.
    addArg(detail::kernel_param_kind_t::kind_accessor, Req,
           static_cast<int>(AccessTarget), ArgIndex);
  }

  template <typename T> void setArgHelper(int ArgIndex, T &&Arg) {
    void *StoredArg = storePlainArg(Arg);

    if (!std::is_same<cl_mem, T>::value && std::is_pointer<T>::value) {
      addArg(detail::kernel_param_kind_t::kind_pointer, StoredArg, sizeof(T),
             ArgIndex);
    } else {
      addArg(detail::kernel_param_kind_t::kind_std_layout, StoredArg, sizeof(T),
             ArgIndex);
    }
  }

  void setArgHelper(int ArgIndex, sampler &&Arg) {
    void *StoredArg = storePlainArg(Arg);
    addArg(detail::kernel_param_kind_t::kind_sampler, StoredArg,
           sizeof(sampler), ArgIndex);
  }

  void setArgHelper(int ArgIndex, stream &&Str);

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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    registerDynamicParameter(detail::getSyclObjImpl(DynamicParam).get(),
                             ArgIndex);
#else
    registerDynamicParameter(DynamicParam, ArgIndex);
#endif
  }

  template <typename DataT, typename PropertyListT>
  void setArgHelper(
      int ArgIndex,
      ext::oneapi::experimental::dynamic_work_group_memory<DataT, PropertyListT>
          &DynWorkGroupMem) {
    (void)ArgIndex;
    (void)DynWorkGroupMem;

#ifndef __SYCL_DEVICE_ONLY__
    ext::oneapi::experimental::detail::dynamic_work_group_memory_base
        &DynWorkGroupBase = DynWorkGroupMem;

    ext::oneapi::experimental::detail::dynamic_parameter_impl *DynParamImpl =
        detail::getSyclObjImpl(DynWorkGroupBase).get();

    addArg(detail::kernel_param_kind_t::kind_dynamic_work_group_memory,
           DynParamImpl, 0, ArgIndex);
    registerDynamicParameter(DynParamImpl, ArgIndex);
#endif
  }

  template <typename DataT, int Dimensions>
  void setArgHelper(
      int ArgIndex,
      ext::oneapi::experimental::dynamic_local_accessor<DataT, Dimensions>
          &DynLocalAccessor) {
    (void)ArgIndex;
    (void)DynLocalAccessor;
#ifndef __SYCL_DEVICE_ONLY__
    ext::oneapi::experimental::detail::dynamic_local_accessor_base
        &DynLocalAccessorBase = DynLocalAccessor;

    ext::oneapi::experimental::detail::dynamic_parameter_impl *DynParamImpl =
        detail::getSyclObjImpl(DynLocalAccessorBase).get();

    addArg(detail::kernel_param_kind_t::kind_dynamic_accessor, DynParamImpl, 0,
           ArgIndex);
    registerDynamicParameter(DynParamImpl, ArgIndex);
#endif
  }

  // setArgHelper for the raw_kernel_arg extension type.
  void setArgHelper(int ArgIndex,
                    sycl::ext::oneapi::experimental::raw_kernel_arg &&Arg) {
    auto StoredArg = storeRawArg(Arg);
    addArg(detail::kernel_param_kind_t::kind_std_layout, StoredArg,
           Arg.MArgSize, ArgIndex);
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // TODO: Remove in the next ABI-breaking window.
  void registerDynamicParameter(
      ext::oneapi::experimental::detail::dynamic_parameter_base
          &DynamicParamBase,
      int ArgIndex);
#endif

  /// Registers a dynamic parameter with the handler for later association with
  /// the node being created.
  /// @param DynamicParamImpl The dynamic parameter impl object.
  /// @param ArgIndex The index of the kernel argument that this dynamic
  /// parameter represents.
  void registerDynamicParameter(
      ext::oneapi::experimental::detail::dynamic_parameter_impl
          *DynamicParamImpl,
      int ArgIndex);

  /// Verifies the kernel bundle to be used if any is set. This throws a
  /// sycl::exception with error code errc::kernel_not_supported if the used
  /// kernel bundle does not contain a suitable device image with the requested
  /// kernel.
  ///
  /// \param KernelName is the name of the SYCL kernel to check that the used
  ///                   kernel bundle contains.
  void verifyUsedKernelBundleInternal(detail::string_view KernelName);

  // TODO: Legacy symbol, remove when ABI breaking is allowed.
  void verifyUsedKernelBundle(const std::string &KernelName) {
    verifyUsedKernelBundleInternal(detail::string_view{KernelName});
  }

  /// Stores lambda to the template-free object
  ///
  /// Also initializes the kernel name and prepares for arguments to
  /// be extracted from the lambda in handler::finalize().
  ///
  /// \param KernelFunc is a SYCL kernel function
  /// \param ParamDescs is the vector of kernel parameter descriptors.
  template <typename KernelName, typename KernelType, int Dims,
            typename LambdaArgType, typename KernelTypeUniversalRef>
  void StoreLambda(KernelTypeUniversalRef &&KernelFunc) {
    constexpr bool IsCallableWithKernelHandler =
        detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                 LambdaArgType>::value;

    // Not using `std::make_unique` to avoid unnecessary instantiations of
    // `std::unique_ptr<HostKernel<...>>`. Only
    // `std::unique_ptr<HostKernelBase>` is necessary.
    MHostKernel.reset(new detail::HostKernel<KernelType, LambdaArgType, Dims>(
        std::forward<KernelTypeUniversalRef>(KernelFunc)));

    // Instantiating the kernel on the host improves debugging.
    // Passing this pointer to another translation unit prevents optimization.
#ifndef NDEBUG
    instantiateKernelOnHost(
        detail::GetInstantiateKernelOnHostPtr<KernelType, LambdaArgType,
                                              Dims>());
#endif

    constexpr bool KernelHasName =
        detail::getKernelName<KernelName>() != nullptr &&
        detail::getKernelName<KernelName>()[0] != '\0';

    // Some host compilers may have different captures from Clang. Currently
    // there is no stable way of handling this when extracting the captures, so
    // a static assert is made to fail for incompatible kernel lambdas.

    // TODO remove the ifdef once the kernel size builtin is supported.
#ifdef __INTEL_SYCL_USE_INTEGRATION_HEADERS
    static_assert(
        !KernelHasName ||
            sizeof(KernelType) == detail::getKernelSize<KernelName>(),
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
#endif
    // Empty name indicates that the compilation happens without integration
    // header, so don't perform things that require it.
    if constexpr (KernelHasName) {
      // TODO support ESIMD in no-integration-header case too.

      // Force hasSpecialCaptures to be evaluated at compile-time.
      constexpr bool HasSpecialCapt = detail::hasSpecialCaptures<KernelName>();
      setKernelInfo((void *)MHostKernel->getPtr(),
                    detail::getKernelNumParams<KernelName>(),
                    &(detail::getKernelParamDesc<KernelName>),
                    detail::isKernelESIMD<KernelName>(), HasSpecialCapt);

      constexpr std::string_view KernelNameStr =
          detail::getKernelName<KernelName>();
      MKernelName = KernelNameStr;
    } else {
      // In case w/o the integration header it is necessary to process
      // accessors from the list(which are associated with this handler) as
      // arguments. We must copy the associated accessors as they are checked
      // later during finalize.
      setArgsToAssociatedAccessors();
    }
    setKernelNameBasedCachePtr(detail::getKernelNameBasedCache<KernelName>());

    // If the kernel lambda is callable with a kernel_handler argument, manifest
    // the associated kernel handler.
    if constexpr (IsCallableWithKernelHandler) {
      getOrInsertHandlerKernelBundlePtr(/*Insert=*/true);
    }
  }

  void verifyDeviceHasProgressGuarantee(
      sycl::ext::oneapi::experimental::forward_progress_guarantee guarantee,
      sycl::ext::oneapi::experimental::execution_scope threadScope,
      sycl::ext::oneapi::experimental::execution_scope coordinationScope);

  template <typename Properties>
  void checkAndSetClusterRange(const Properties &Props) {
    namespace syclex = sycl::ext::oneapi::experimental;
    constexpr std::size_t ClusterDim =
        syclex::detail::getClusterDim<Properties>();
    if constexpr (ClusterDim > 0) {
      auto ClusterSize = Props
                             .template get_property<
                                 syclex::cuda::cluster_size_key<ClusterDim>>()
                             .get_cluster_size();
      setKernelClusterLaunch(ClusterSize);
    }
  }

  /// Process runtime kernel properties.
  ///
  /// Stores information about kernel properties into the handler.
  template <typename PropertiesT>
  void processLaunchProperties(PropertiesT Props) {
    if constexpr (PropertiesT::template has_property<
                      sycl::ext::intel::experimental::cache_config_key>()) {
      auto Config = Props.template get_property<
          sycl::ext::intel::experimental::cache_config_key>();
      if (Config == sycl::ext::intel::experimental::large_slm) {
        setKernelCacheConfig(StableKernelCacheConfig::LargeSLM);
      } else if (Config == sycl::ext::intel::experimental::large_data) {
        setKernelCacheConfig(StableKernelCacheConfig::LargeData);
      }
    } else {
      std::ignore = Props;
    }

    constexpr bool UsesRootSync = PropertiesT::template has_property<
        sycl::ext::oneapi::experimental::use_root_sync_key>();
    if (UsesRootSync) {
      setKernelIsCooperative(UsesRootSync);
    }
    if constexpr (PropertiesT::template has_property<
                      sycl::ext::oneapi::experimental::
                          work_group_progress_key>()) {
      auto prop = Props.template get_property<
          sycl::ext::oneapi::experimental::work_group_progress_key>();
      verifyDeviceHasProgressGuarantee(
          prop.guarantee,
          sycl::ext::oneapi::experimental::execution_scope::work_group,
          prop.coordinationScope);
    }
    if constexpr (PropertiesT::template has_property<
                      sycl::ext::oneapi::experimental::
                          sub_group_progress_key>()) {
      auto prop = Props.template get_property<
          sycl::ext::oneapi::experimental::sub_group_progress_key>();
      verifyDeviceHasProgressGuarantee(
          prop.guarantee,
          sycl::ext::oneapi::experimental::execution_scope::sub_group,
          prop.coordinationScope);
    }
    if constexpr (PropertiesT::template has_property<
                      sycl::ext::oneapi::experimental::
                          work_item_progress_key>()) {
      auto prop = Props.template get_property<
          sycl::ext::oneapi::experimental::work_item_progress_key>();
      verifyDeviceHasProgressGuarantee(
          prop.guarantee,
          sycl::ext::oneapi::experimental::execution_scope::work_item,
          prop.coordinationScope);
    }

    if constexpr (PropertiesT::template has_property<
                      sycl::ext::oneapi::experimental::
                          work_group_scratch_size>()) {
      auto WorkGroupMemSize = Props.template get_property<
          sycl::ext::oneapi::experimental::work_group_scratch_size>();
      setKernelWorkGroupMem(WorkGroupMemSize.size);
    }

    checkAndSetClusterRange(Props);
  }

  /// Process kernel properties.
  ///
  /// Stores information about kernel properties into the handler.
  ///
  /// Note: it is important that this function *does not* depend on kernel
  /// name or kernel type, because then it will be instantiated for every
  /// kernel, even though body of those instantiated functions could be almost
  /// the same, thus unnecessary increasing compilation time.
  template <
      bool IsESIMDKernel,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  void processProperties(PropertiesT Props) {
    static_assert(
        ext::oneapi::experimental::is_property_list<PropertiesT>::value,
        "Template type is not a property list.");
    static_assert(
        !PropertiesT::template has_property<
            sycl::ext::intel::experimental::fp_control_key>() ||
            (PropertiesT::template has_property<
                 sycl::ext::intel::experimental::fp_control_key>() &&
             IsESIMDKernel),
        "Floating point control property is supported for ESIMD kernels only.");
    static_assert(
        !PropertiesT::template has_property<
            sycl::ext::oneapi::experimental::indirectly_callable_key>(),
        "indirectly_callable property cannot be applied to SYCL kernels");

    processLaunchProperties(Props);
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
    if (IsCopyingRectRegionAvailable(Src.get_range(), Dst.get_range()))
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
  copyAccToAccHelper(accessor<TSrc, DimSrc, ModeSrc, TargetSrc, IsPHSrc>,
                     accessor<TDst, DimDst, ModeDst, TargetDst, IsPHDst>) {
    return false;
  }

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

  // UR APIs only support select fill sizes: 1, 2, 4, 8, 16, 32, 64, 128
  constexpr static bool isBackendSupportedFillSize(size_t Size) {
    return Size == 1 || Size == 2 || Size == 4 || Size == 8 || Size == 16 ||
           Size == 32 || Size == 64 || Size == 128;
  }

  bool eventNeeded() const;

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

#ifdef __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
    size_t GoodExpFactor = 1;
    switch (Dims) {
    case 1:
      GoodExpFactor = 32; // Make global range multiple of {32}
      break;
    case 2:
      GoodExpFactor = 16; // Make global range multiple of {16, 16}
      break;
    case 3:
      GoodExpFactor = 8; // Make global range multiple of {8, 8, 8}
      break;
    }

    // Check if rounding parameters have been set through environment:
    // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
    this->GetRangeRoundingSettings(MinFactorX, GoodExpFactor, MinRangeX);

    for (auto i = 0; i < Dims; ++i)
      if (UserRange[i] % GoodExpFactor) {
        Adjust(i, ((UserRange[i] / GoodExpFactor) + 1) * GoodExpFactor);
      }
#else
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
#endif // __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
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
                                const KernelType &KernelFunc) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfActionIsCreated();
    throwOnKernelParameterMisuse<KernelName, KernelType>();
    if (!range_size_fits_in_size_t(UserRange))
      throw sycl::exception(make_error_code(errc::runtime),
                            "The total number of work-items in "
                            "a range must fit within size_t");
#endif

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

    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;

    // Range rounding can be disabled by the user.
    // Range rounding is supported only for newer SYCL standards.
#if !defined(__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__) &&                  \
    SYCL_LANGUAGE_VERSION >= 202012L
    auto [RoundedRange, HasRoundedRange] = getRoundedRange(UserRange);
    if (HasRoundedRange) {
      using NameWT = typename detail::get_kernel_wrapper_name_t<NameT>::name;
      auto Wrapper =
          getRangeRoundedKernelLambda<NameWT, TransformedArgType, Dims>(
              KernelFunc, UserRange);

      using KName = std::conditional_t<std::is_same<KernelType, NameT>::value,
                                       decltype(Wrapper), NameWT>;

      detail::KernelWrapper<detail::WrapAs::parallel_for, KName,
                            decltype(Wrapper), TransformedArgType,
                            PropertiesT>::wrap(Wrapper);

      detail::KernelLaunchPropertyWrapper::parseProperties<KName>(this,
                                                                  Wrapper);
#ifndef __SYCL_DEVICE_ONLY__
      constexpr detail::string_view Name{detail::getKernelName<NameT>()};
      verifyUsedKernelBundleInternal(Name);
      // We are executing over the rounded range, but there are still
      // items/ids that are are constructed in ther range rounded
      // kernel use items/ids in the user range, which means that
      // __SYCL_ASSUME_INT can still be violated. So check the bounds
      // of the user range, instead of the rounded range.
      detail::checkValueRange<Dims>(UserRange);
      setNDRangeDescriptor(RoundedRange);
      StoreLambda<KName, decltype(Wrapper), Dims, TransformedArgType>(
          std::move(Wrapper));
      setType(detail::CGType::Kernel);
#endif
    } else
#endif // !__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ &&
       // SYCL_LANGUAGE_VERSION >= 202012L
    {
      (void)UserRange;
      (void)Props;
#ifndef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
      // If parallel_for range rounding is forced then only range rounded
      // kernel is generated
      detail::KernelWrapper<detail::WrapAs::parallel_for, NameT, KernelType,
                            TransformedArgType, PropertiesT>::wrap(KernelFunc);
      detail::KernelLaunchPropertyWrapper::parseProperties<NameT>(this,
                                                                  KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
      constexpr detail::string_view Name{detail::getKernelName<NameT>()};

      verifyUsedKernelBundleInternal(Name);
      processProperties<detail::isKernelESIMD<NameT>(), PropertiesT>(Props);
      detail::checkValueRange<Dims>(UserRange);
      setNDRangeDescriptor(std::move(UserRange));
      StoreLambda<NameT, KernelType, Dims, TransformedArgType>(
          std::move(KernelFunc));
      setType(detail::CGType::Kernel);
#endif
#else
      (void)KernelFunc;
#endif // __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
    }
  }

  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object. The kernel
  /// invocation method has no functors and cannot be called on host.
  ///
  /// \param NumWorkItems is a range defining indexing space.
  /// \param Kernel is a SYCL kernel function.
  /// \param Properties is the properties.
  template <int Dims, typename PropertiesT>
  void parallel_for_impl([[maybe_unused]] range<Dims> NumWorkItems,
                         [[maybe_unused]] PropertiesT Props,
                         [[maybe_unused]] kernel Kernel) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfActionIsCreated();
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NumWorkItems);
    setNDRangeDescriptor(std::move(NumWorkItems));
    processLaunchProperties<PropertiesT>(Props);
    setType(detail::CGType::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
#endif
  }

  /// Defines and invokes a SYCL kernel function for the specified range and
  /// offsets.
  ///
  /// The SYCL kernel function is defined as SYCL kernel object.
  ///
  /// \param NDRange is a ND-range defining global and local sizes as
  /// well as offset.
  /// \param Properties is the properties.
  /// \param Kernel is a SYCL kernel function.
  template <int Dims, typename PropertiesT>
  void parallel_for_impl([[maybe_unused]] nd_range<Dims> NDRange,
                         [[maybe_unused]] PropertiesT Props,
                         [[maybe_unused]] kernel Kernel) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfActionIsCreated();
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NDRange);
    setNDRangeDescriptor(std::move(NDRange));
    processLaunchProperties(Props);
    setType(detail::CGType::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
#endif
  }

  template <
      detail::WrapAs WrapAsVal, typename KernelName,
      typename ElementType = void, int Dims = 1, bool SetNumWorkGroups = false,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t,
      typename KernelType, typename... RangeParams>
  void wrap_kernel(const KernelType &KernelFunc, const PropertiesT &Props,
                   [[maybe_unused]] RangeParams &&...params) {
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    (void)Props;
    detail::KernelWrapper<WrapAsVal, NameT, KernelType, ElementType,
                          PropertiesT>::wrap(KernelFunc);
    detail::KernelLaunchPropertyWrapper::parseProperties<NameT>(this,
                                                                KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    if constexpr (WrapAsVal == detail::WrapAs::single_task) {
      throwOnKernelParameterMisuse<KernelName, KernelType>();
    }
    throwIfActionIsCreated();
    constexpr detail::string_view Name{detail::getKernelName<NameT>()};
    verifyUsedKernelBundleInternal(Name);
    setType(detail::CGType::Kernel);

    detail::checkValueRange<Dims>(params...);
    if constexpr (SetNumWorkGroups) {
      setNDRangeDescriptor(std::move(params)...,
                           /*SetNumWorkGroups=*/true);
    } else {
      setNDRangeDescriptor(std::move(params)...);
    }

    StoreLambda<NameT, KernelType, Dims, ElementType>(std::move(KernelFunc));
    processProperties<detail::isKernelESIMD<NameT>(), PropertiesT>(Props);
#endif
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Implementation for something that had to be removed long ago but now stuck
  // until next major release...
  template <
      detail::WrapAs WrapAsVal, typename KernelName,
      typename ElementType = void, int Dims = 1, bool SetNumWorkGroups = false,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t,
      typename KernelType, typename... RangeParams>
  void wrap_kernel_legacy(const KernelType &KernelFunc, kernel &Kernel,
                          const PropertiesT &Props,
                          [[maybe_unused]] RangeParams &&...params) {
    // TODO: Properties may change the kernel function, so in order to avoid
    //       conflicts they should be included in the name.
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    (void)Props;
    (void)Kernel;
    detail::KernelWrapper<WrapAsVal, NameT, KernelType, ElementType,
                          PropertiesT>::wrap(KernelFunc);
    detail::KernelLaunchPropertyWrapper::parseProperties<NameT>(this,
                                                                KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    if constexpr (WrapAsVal == detail::WrapAs::single_task) {
      throwOnKernelParameterMisuse<KernelName, KernelType>();
    }
    throwIfActionIsCreated();
    // Ignore any set kernel bundles and use the one associated with the
    // kernel.
    setHandlerKernelBundle(Kernel);
    constexpr detail::string_view Name{detail::getKernelName<NameT>()};
    verifyUsedKernelBundleInternal(Name);
    setType(detail::CGType::Kernel);

    detail::checkValueRange<Dims>(params...);
    if constexpr (SetNumWorkGroups) {
      setNDRangeDescriptor(std::move(params)...,
                           /*SetNumWorkGroups=*/true);
    } else {
      setNDRangeDescriptor(std::move(params)...);
    }

    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    if (!lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else {
      StoreLambda<NameT, KernelType, Dims, ElementType>(std::move(KernelFunc));
    }
    processProperties<detail::isKernelESIMD<NameT>(), PropertiesT>(Props);
#endif
  }
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  // NOTE: to support kernel_handler argument in kernel lambdas, only
  // detail::KernelWrapper<...>::wrap() must be called in this code.

  void setStateExplicitKernelBundle();
  void setStateSpecConstSet();
  bool isStateExplicitKernelBundle() const;

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  std::shared_ptr<detail::kernel_bundle_impl>
  getOrInsertHandlerKernelBundle(bool Insert) const;
#endif

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // Rename to just getOrInsertHandlerKernelBundle
#endif
  detail::kernel_bundle_impl *
  getOrInsertHandlerKernelBundlePtr(bool Insert) const;

  void setHandlerKernelBundle(kernel Kernel);

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void setHandlerKernelBundle(
      const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr);
#endif

  template <typename SharedPtrT>
  void setHandlerKernelBundle(SharedPtrT &&NewKernelBundleImpPtr);

  void SetHostTask(std::function<void()> &&Func);
  void SetHostTask(std::function<void(interop_handle)> &&Func);

  template <typename FuncT>
  std::enable_if_t<detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void()>::value ||
                   detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void(interop_handle)>::value>
  host_task_impl(FuncT &&Func) {
    throwIfActionIsCreated();

    // Need to copy these rather than move so that we can check associated
    // accessors during finalize
    setArgsToAssociatedAccessors();

    SetHostTask(std::move(Func));
  }

  template <typename FuncT>
  std::enable_if_t<detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void(interop_handle)>::value>
  ext_codeplay_enqueue_native_command_impl(FuncT &&Func) {
    throwIfActionIsCreated();

    // Need to copy these rather than move so that we can check associated
    // accessors during finalize
    setArgsToAssociatedAccessors();

    SetHostTask(std::move(Func));
    setType(detail::CGType::EnqueueNativeCommand);
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

  kernel_bundle<bundle_state::input> getKernelBundle() const;

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  // Out-of-class definition within kernel_bundle.hpp
  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type Value);

  // Out-of-class definition within kernel_bundle.hpp
  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const;

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
        || is_same_type<cl_mem, T>::value          // Interop
        || is_same_type<stream, T>::value;         // Stream
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

  template <typename DataT, typename PropertyListT =
                                ext::oneapi::experimental::empty_properties_t>
  void set_arg(
      int ArgIndex,
      ext::oneapi::experimental::work_group_memory<DataT, PropertyListT> &Arg) {
    // slice the base class object out of Arg
    detail::work_group_memory_impl &ArgImpl = Arg;
    setArgHelper(ArgIndex, ArgImpl);
  }

  // set_arg for graph dynamic_parameters
  template <typename T>
  void set_arg(int argIndex,
               ext::oneapi::experimental::dynamic_parameter<T> &dynamicParam) {
    setArgHelper(argIndex, dynamicParam);
  }

  // set_arg for graph dynamic_work_group_memory
  template <typename DataT, typename PropertyListT =
                                ext::oneapi::experimental::empty_properties_t>
  void set_arg(
      int argIndex,
      ext::oneapi::experimental::dynamic_work_group_memory<DataT, PropertyListT>
          &DynWorkGroupMem) {
    setArgHelper<DataT, PropertyListT>(argIndex, DynWorkGroupMem);
  }

  // set_arg for graph dynamic_local_accessor
  template <typename DataT, int Dimensions>
  void
  set_arg(int argIndex,
          ext::oneapi::experimental::dynamic_local_accessor<DataT, Dimensions>
              &DynLocalAccessor) {
    setArgHelper<DataT, Dimensions>(argIndex, DynLocalAccessor);
  }

  // set_arg for the raw_kernel_arg extension type.
  void set_arg(int argIndex, ext::oneapi::experimental::raw_kernel_arg &&Arg) {
    setArgHelper(argIndex, std::move(Arg));
  }

  /// Sets arguments for OpenCL interoperability kernels.
  ///
  /// Registers pack of arguments(Args) with indexes starting from 0.
  ///
  /// \param Args are argument values to be set.
  template <typename... Ts> void set_args(Ts &&...Args) {
    setArgsHelper(0, std::forward<Ts>(Args)...);
  }
  /// Defines and invokes a SYCL kernel function as a function object type.
  ///
  /// If it is a named function object and the function object type is
  /// globally visible, there is no need for the developer to provide
  /// a kernel name for it.
  ///
  /// \param KernelFunc is a SYCL kernel function.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::single_task, KernelName>(
        KernelFunc, {} /*Props*/, range<1>{1});
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<1> NumWorkItems, const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<2> NumWorkItems, const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        KernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  void parallel_for(range<3> NumWorkItems, const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName>(
        NumWorkItems, ext::oneapi::experimental::empty_properties_t{},
        KernelFunc);
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

  /// Enqueues a command to the SYCL runtime to invoke \p Func immediately.
  template <typename FuncT>
  std::enable_if_t<detail::check_fn_signature<std::remove_reference_t<FuncT>,
                                              void(interop_handle)>::value>
  ext_codeplay_enqueue_native_command([[maybe_unused]] FuncT &&Func) {
#ifndef __SYCL_DEVICE_ONLY__
    ext_codeplay_enqueue_native_command_impl(Func);
#endif
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
                    const KernelType &KernelFunc) {
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    using TransformedArgType = std::conditional_t<
        std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
        typename TransformUserItemType<Dims, LambdaArgType>::type>;
    wrap_kernel<detail::WrapAs::parallel_for, KernelName, TransformedArgType,
                Dims>(KernelFunc, {} /*Props*/, NumWorkItems, WorkItemOffset);
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
                               const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::parallel_for_work_group, KernelName,
                detail::lambda_arg_type<KernelType, group<Dims>>, Dims,
                /*SetNumWorkGroups=*/true>(KernelFunc, {} /*Props*/,
                                           NumWorkGroups);
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
                               const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::parallel_for_work_group, KernelName,
                detail::lambda_arg_type<KernelType, group<Dims>>, Dims>(
        KernelFunc, {} /*Props*/,
        nd_range<Dims>{NumWorkGroups * WorkGroupSize, WorkGroupSize});
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
    setNDRangeDescriptor(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CGType::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
  }

  void parallel_for(range<1> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems,
                      ext::oneapi::experimental::empty_properties_t{}, Kernel);
  }

  void parallel_for(range<2> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems,
                      ext::oneapi::experimental::empty_properties_t{}, Kernel);
  }

  void parallel_for(range<3> NumWorkItems, kernel Kernel) {
    parallel_for_impl(NumWorkItems,
                      ext::oneapi::experimental::empty_properties_t{}, Kernel);
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
  void parallel_for([[maybe_unused]] range<Dims> NumWorkItems,
                    [[maybe_unused]] id<Dims> WorkItemOffset,
                    [[maybe_unused]] kernel Kernel) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfActionIsCreated();
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    detail::checkValueRange<Dims>(NumWorkItems, WorkItemOffset);
    setNDRangeDescriptor(std::move(NumWorkItems), std::move(WorkItemOffset));
    setType(detail::CGType::Kernel);
    extractArgsAndReqs();
    MKernelName = getKernelName();
#endif
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
    parallel_for_impl(NDRange, ext::oneapi::experimental::empty_properties_t{},
                      Kernel);
  }

  /// Defines and invokes a SYCL kernel function.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(kernel Kernel, const KernelType &KernelFunc) {
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    (void)Kernel;
    detail::KernelWrapperHelperFuncs::kernel_single_task<NameT>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
    throwIfActionIsCreated();
    constexpr detail::string_view Name{detail::getKernelName<NameT>()};
    verifyUsedKernelBundleInternal(Name);
    // No need to check if range is out of INT_MAX limits as it's compile-time
    // known constant
    setNDRangeDescriptor(range<1>{1});
    MKernel = detail::getSyclObjImpl(std::move(Kernel));
    setType(detail::CGType::Kernel);
    if (!lambdaAndKernelHaveEqualName<NameT>()) {
      extractArgsAndReqs();
      MKernelName = getKernelName();
    } else
      StoreLambda<NameT, KernelType, /*Dims*/ 1, void>(std::move(KernelFunc));
#else
    detail::CheckDeviceCopyable<KernelType>();
#endif
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Defines and invokes a SYCL kernel function for the specified range.
  ///
  /// \param Kernel is a SYCL kernel that is executed on a SYCL device
  /// (except for the host device).
  /// \param NumWorkItems is a range defining indexing space.
  /// \param KernelFunc is a lambda that is used if device, queue is bound to,
  /// is a host device.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  __SYCL_DEPRECATED("This overload isn't part of SYCL2020 and will be removed.")
  void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                    const KernelType &KernelFunc) {
    // Ignore any set kernel bundles and use the one associated with the kernel
    setHandlerKernelBundle(Kernel);
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    wrap_kernel_legacy<detail::WrapAs::parallel_for, KernelName, LambdaArgType,
                       Dims>(KernelFunc, Kernel, {} /*Props*/, NumWorkItems);
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
  __SYCL_DEPRECATED("This overload isn't part of SYCL2020 and will be removed.")
  void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, const KernelType &KernelFunc) {
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    wrap_kernel_legacy<detail::WrapAs::parallel_for, KernelName, LambdaArgType,
                       Dims>(KernelFunc, Kernel, {} /*Props*/, NumWorkItems,
                             WorkItemOffset);
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
  __SYCL_DEPRECATED("This overload isn't part of SYCL2020 and will be removed.")
  void parallel_for(kernel Kernel, nd_range<Dims> NDRange,
                    const KernelType &KernelFunc) {
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
    wrap_kernel_legacy<detail::WrapAs::parallel_for, KernelName, LambdaArgType,
                       Dims>(KernelFunc, Kernel, {} /*Props*/, NDRange);
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
  __SYCL_DEPRECATED("This overload isn't part of SYCL2020 and will be removed.")
  void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                               const KernelType &KernelFunc) {
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    wrap_kernel_legacy<detail::WrapAs::parallel_for_work_group, KernelName,
                       LambdaArgType, Dims,
                       /*SetNumWorkGroups*/ true>(KernelFunc, Kernel,
                                                  {} /*Props*/, NumWorkGroups);
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
  __SYCL_DEPRECATED("This overload isn't part of SYCL2020 and will be removed.")
  void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize,
                               const KernelType &KernelFunc) {
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, group<Dims>>;
    nd_range<Dims> ExecRange =
        nd_range<Dims>(NumWorkGroups * WorkGroupSize, WorkGroupSize);
    wrap_kernel_legacy<detail::WrapAs::parallel_for_work_group, KernelName,
                       LambdaArgType, Dims>(KernelFunc, Kernel, {} /*Props*/,
                                            ExecRange);
  }
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<ext::oneapi::experimental::is_property_list<
      PropertiesT>::value> single_task(PropertiesT Props,
                                       const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::single_task, KernelName>(KernelFunc, Props,
                                                         range<1>{1});
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<ext::oneapi::experimental::is_property_list<
      PropertiesT>::value> parallel_for(range<1> NumWorkItems,
                                        PropertiesT Props,
                                        const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName, KernelType, 1, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<ext::oneapi::experimental::is_property_list<
      PropertiesT>::value> parallel_for(range<2> NumWorkItems,
                                        PropertiesT Props,
                                        const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName, KernelType, 2, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<ext::oneapi::experimental::is_property_list<
      PropertiesT>::value> parallel_for(range<3> NumWorkItems,
                                        PropertiesT Props,
                                        const KernelType &KernelFunc) {
    parallel_for_lambda_impl<KernelName, KernelType, 3, PropertiesT>(
        NumWorkItems, Props, std::move(KernelFunc));
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT, int Dims>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<ext::oneapi::experimental::is_property_list<
      PropertiesT>::value> parallel_for(nd_range<Dims> Range,
                                        PropertiesT Properties,
                                        const KernelType &KernelFunc) {
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, nd_item<Dims>>;
    static_assert(
        std::is_convertible_v<sycl::nd_item<Dims>, LambdaArgType>,
        "Kernel argument of a sycl::parallel_for with sycl::nd_range "
        "must be either sycl::nd_item or be convertible from sycl::nd_item");
    using TransformedArgType = sycl::nd_item<Dims>;

    wrap_kernel<detail::WrapAs::parallel_for, KernelName, TransformedArgType,
                Dims>(KernelFunc, Properties, Range);
  }

  /// Reductions @{

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<(sizeof...(RestT) > 1) &&
                   detail::AreAllButLastReductions<RestT...>::value &&
                   ext::oneapi::experimental::is_property_list<
                       PropertiesT>::value> parallel_for(range<1> Range,
                                                         PropertiesT Properties,
                                                         RestT &&...Rest) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
#endif
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<(sizeof...(RestT) > 1) &&
                   detail::AreAllButLastReductions<RestT...>::value &&
                   ext::oneapi::experimental::is_property_list<
                       PropertiesT>::value> parallel_for(range<2> Range,
                                                         PropertiesT Properties,
                                                         RestT &&...Rest) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
#endif
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename PropertiesT,
            typename... RestT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<(sizeof...(RestT) > 1) &&
                   detail::AreAllButLastReductions<RestT...>::value &&
                   ext::oneapi::experimental::is_property_list<
                       PropertiesT>::value> parallel_for(range<3> Range,
                                                         PropertiesT Properties,
                                                         RestT &&...Rest) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
#endif
    detail::reduction_parallel_for<KernelName>(*this, Range, Properties,
                                               std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value &&
                   (sizeof...(RestT) > 1)>
  parallel_for(range<1> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value &&
                   (sizeof...(RestT) > 1)>
  parallel_for(range<2> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value &&
                   (sizeof...(RestT) > 1)>
  parallel_for(range<3> Range, RestT &&...Rest) {
    parallel_for<KernelName>(Range,
                             ext::oneapi::experimental::empty_properties_t{},
                             std::forward<RestT>(Rest)...);
  }

  template <typename KernelName = detail::auto_name, int Dims,
            typename PropertiesT, typename... RestT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  std::enable_if_t<(sizeof...(RestT) > 1) &&
                   detail::AreAllButLastReductions<RestT...>::value &&
                   ext::oneapi::experimental::is_property_list<
                       PropertiesT>::value> parallel_for(nd_range<Dims> Range,
                                                         PropertiesT Properties,
                                                         RestT &&...Rest) {
#ifndef __SYCL_DEVICE_ONLY__
    throwIfGraphAssociated<ext::oneapi::experimental::detail::
                               UnsupportedGraphFeatures::sycl_reductions>();
#endif
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
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  void parallel_for_work_group(range<Dims> NumWorkGroups, PropertiesT Props,
                               const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::parallel_for_work_group, KernelName,
                detail::lambda_arg_type<KernelType, group<Dims>>, Dims,
                /*SetNumWorkGroups=*/true>(KernelFunc, Props, NumWorkGroups);
  }

  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename PropertiesT>
  __SYCL_DEPRECATED("To specify properties, use a launch configuration object "
                    "of type launch_config or a kernel functor with a "
                    "get(sycl::ext::oneapi::experimental::properties_tag) "
                    "member function instead.")
  void parallel_for_work_group(range<Dims> NumWorkGroups,
                               range<Dims> WorkGroupSize, PropertiesT Props,
                               const KernelType &KernelFunc) {
    wrap_kernel<detail::WrapAs::parallel_for_work_group, KernelName,
                detail::lambda_arg_type<KernelType, group<Dims>>, Dims>(
        KernelFunc, Props,
        nd_range<Dims>{NumWorkGroups * WorkGroupSize, WorkGroupSize});
  }

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
#ifndef __SYCL_DEVICE_ONLY__
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);
#endif

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForSourceAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    addLifetimeSharedPtrStorage(Dst);
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);
#endif

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // TODO: Add static_assert with is_device_copyable when vec is
    // device-copyable.
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    addLifetimeSharedPtrStorage(Src);
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);
#endif

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForSourceAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    setType(detail::CGType::CopyAccToPtr);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MSrcPtr = static_cast<void *>(AccImpl.get());
    MDstPtr = static_cast<void *>(Dst);
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);
#endif

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the copy method.");
    static_assert(isValidModeForDestinationAccessor(AccessMode),
                  "Invalid accessor mode for the copy method.");
    // TODO: Add static_assert with is_device_copyable when vec is
    // device-copyable.

    setType(detail::CGType::CopyPtrToAcc);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MSrcPtr = const_cast<T_Src *>(Src);
    MDstPtr = static_cast<void *>(AccImpl.get());
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Src.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Src);
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);
#endif

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
      throw sycl::exception(make_error_code(errc::invalid),
                            "The destination accessor size is too small to "
                            "copy the memory into.");

    if (copyAccToAccHelper(Src, Dst))
      return;
    setType(detail::CGType::CopyAccToAcc);

    detail::AccessorBaseHost *AccBaseSrc = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImplSrc = detail::getSyclObjImpl(*AccBaseSrc);

    detail::AccessorBaseHost *AccBaseDst = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImplDst = detail::getSyclObjImpl(*AccBaseDst);

    MSrcPtr = AccImplSrc.get();
    MDstPtr = AccImplDst.get();
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Acc.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Acc);
#endif

    throwIfActionIsCreated();
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the update_host method.");
    setType(detail::CGType::UpdateHost);

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = static_cast<void *>(AccImpl.get());
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
#ifndef __SYCL_DEVICE_ONLY__
    if (Dst.is_placeholder())
      checkIfPlaceholderIsBoundToHandler(Dst);
#endif

    throwIfActionIsCreated();
    setUserFacingNodeType(ext::oneapi::experimental::node_type::memfill);
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    static_assert(isValidTargetForExplicitOp(AccessTarget),
                  "Invalid accessor target for the fill method.");
    // CG::Fill will result in urEnqueueMemBufferFill which requires that mem
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
    if (getDeviceBackend() == backend::ext_oneapi_level_zero) {
      parallel_for<__usmfill<T>>(range<1>(Count), [=](id<1> Index) {
        T *CastedPtr = static_cast<T *>(Ptr);
        CastedPtr[Index] = Pattern;
      });
    } else {
      this->fill_impl(Ptr, &Pattern, sizeof(T), Count);
    }
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  void ext_oneapi_barrier() {
    throwIfActionIsCreated();
    setType(detail::CGType::Barrier);
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
  void memcpy([[maybe_unused]] ext::oneapi::experimental::device_global<
                  T, PropertyListT> &Dest,
              [[maybe_unused]] const void *Src,
              [[maybe_unused]] size_t NumBytes = sizeof(T),
              [[maybe_unused]] size_t DestOffset = 0) {
#ifndef __SYCL_DEVICE_ONLY__
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
#endif
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
  void memcpy([[maybe_unused]] void *Dest,
              [[maybe_unused]] const ext::oneapi::experimental::device_global<
                  T, PropertyListT> &Src,
              [[maybe_unused]] size_t NumBytes = sizeof(T),
              [[maybe_unused]] size_t SrcOffset = 0) {
#ifndef __SYCL_DEVICE_ONLY__
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
#endif
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

  /// Copies data from host to device, where \p Src is a USM pointer and \p Dest
  /// is an opaque image memory handle. An exception is thrown if either \p Src
  /// is nullptr or \p Dest is incomplete. The behavior is undefined if
  /// \p DestImgDesc is inconsistent with the allocated allocated memory
  /// regions.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestImgDesc is the image descriptor.
  void ext_oneapi_copy(
      const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc);

  /// Copies data from host to device, where \p Src is a USM pointer and \p Dest
  /// is an opaque image memory handle. Allows for a sub-region copy, where
  /// \p SrcOffset , \p DestOffset , and \p CopyExtent are used to determine the
  /// sub-region. Pixel size is determined by \p DestImgDesc . An exception is
  /// thrown if either \p Src is nullptr or \p Dest is incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the origin where the x, y, and z
  ///                  components are measured in bytes, rows, and slices
  ///                  respectively.
  /// \param SrcExtent is the size of the source memory, measured in
  ///                  pixels. (Pixel size determined by \p DestImgDesc .)
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels. (Pixel size determined by \p DestImgDesc .)
  /// \param DestImgDesc is the destination image descriptor.
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///                   measured in pixels. (Pixel size determined by
  ///                   \p SrcImgDesc .)
  void ext_oneapi_copy(
      const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent);

  /// Copies data from device to host, where \p Src is an opaque image memory
  /// handle and \p Dest is a USM pointer. An exception is thrown if either
  /// \p Src is incomplete or \p Dest is nullptr. The behavior is undefined if
  /// \p SrcImgDesc is inconsistent with the allocated memory regions.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param SrcImgDesc is the source image descriptor.
  void ext_oneapi_copy(
      const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc);

  /// Copies data from device to host, where \p Src is an opaque image memory
  /// handle and \p Dest is a USM pointer. Allows for a sub-region copy, where
  /// \p SrcOffset , \p DestOffset , and \p CopyExtent are used to determine the
  /// sub-region. Pixel size is determined by \p SrcImgDesc . An exception is
  /// thrown if either \p Src is nullptr or \p Dest is incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the source origin measured in pixels.
  ///                  (Pixel size determined by \p SrcImgDesc .)
  /// \param SrcImgDesc is the source image descriptor.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin where the
  ///                   x, y, and z components are measured in bytes, rows,
  ///                   and slices respectively.
  /// \param DestExtent is the size of the destination memory, measured in
  ///                   pixels. (Pixel size determined by \p SrcImgDesc .)
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///                   measured in pixels. (Pixel size determined by
  ///                   \p SrcImgDesc .)
  void
  ext_oneapi_copy(const ext::oneapi::experimental::image_mem_handle Src,
                  sycl::range<3> SrcOffset,
                  const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
                  void *Dest, sycl::range<3> DestOffset,
                  sycl::range<3> DestExtent, sycl::range<3> CopyExtent);

  /// Copies data from host to device or device to host, where \p Src and
  /// \p Dest are USM pointers. An exception is thrown if either \p Src is
  /// nullptr or \p Dest is nullptr. The behavior is undefined if
  /// \p DeviceImgDesc is inconsistent with the allocated memory regions or
  /// \p DeviceRowPitch is inconsistent with hardware requirements.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DeviceImgDesc is the device image descriptor.
  /// \param DeviceRowPitch is the pitch of the rows of the memory on the
  ///                       device.
  void ext_oneapi_copy(
      const void *Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch);

  /// Copies data from host to device or device to host, where \p Src and
  /// \p Dest are USM pointers. Allows for a sub-region copy, where
  /// \p SrcOffset , \p DestOffset , and \p CopyExtent are used to determine the
  /// sub-region. Pixel size is determined by \p DeviceImgDesc . An exception is
  /// thrown if either \p Src is nullptr or \p Dest is nullptr. The behavior is
  /// undefined if \p DeviceRowPitch is inconsistent with hardware requirements
  /// or \p HostExtent is inconsistent with its respective memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an destination offset from the origin where the
  ///                   x, y, and z components are measured in bytes, rows,
  ///                   and slices respectively.
  /// \param DeviceImgDesc is the device image descriptor.
  /// \param DeviceRowPitch is the pitch of the rows of the image on the device.
  /// \param HostExtent is the size of the host memory, measured in pixels.
  ///                   (Pixel size determined by \p DeviceImgDesc .)
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///                   measured in pixels. (Pixel size determined by
  ///                   \p DeviceImgDesc .)
  void ext_oneapi_copy(
      const void *Src, sycl::range<3> SrcOffset, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, sycl::range<3> HostExtent,
      sycl::range<3> CopyExtent);

  /// Copies data from device to device, where \p Src and \p Dest are opaque
  /// image memory handles. An exception is thrown if either \p Src or \p Dest
  /// are incomplete. The behavior is undefined if \p SrcImgDesc or
  /// \p DestImgDesc are inconsistent with their respective allocated memory
  /// regions.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcImgDesc is the source image descriptor.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestImgDesc is the destination image descriptor.
  void ext_oneapi_copy(
      const ext::oneapi::experimental::image_mem_handle Src,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc);

  /// Copies data from device to device, where \p Src and \p Dest are opaque
  /// image memory handles. Allows for a sub-region copy, where \p SrcOffset ,
  /// \p DestOffset and \p CopyExtent are used to determine the sub-region.
  /// Pixel size is determined by \p SrcImgDesc . An exception is thrown if
  /// either \p Src or \p Dest is incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the source origin measured in pixels.
  ///                  (Pixel size determined by \p SrcImgDesc .)
  /// \param SrcImgDesc is the source image descriptor.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels. (Pixel size determined by \p SrcImgDesc .)
  /// \param DestImgDesc is the destination image descriptor.
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///                   measured in pixels. (Pixel size determined by
  ///                   \p SrcImgDesc .)
  void ext_oneapi_copy(
      const ext::oneapi::experimental::image_mem_handle Src,
      sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent);

  /// Copies data from device to device, where \p Src is an opaque image memory
  /// handle and \p Dest is a USM pointer. An exception is thrown if either
  /// \p Src is incomplete or \p Dest is nullptr. The behavior is undefined if
  /// \p SrcImgDesc or \p DestImgDesc are inconsistent with their respective
  /// allocated memory regions or \p DestRowPitch is inconsistent with hardware
  /// requirements.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcImgDesc is the source image descriptor.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestImgDesc is the destination image descriptor.
  /// \param DestRowPitch is the pitch of the rows of the destination memory.
  void ext_oneapi_copy(
      const ext::oneapi::experimental::image_mem_handle Src,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      size_t DestRowPitch);

  /// Copies data from device to device, where \p Src is an opaque image memory
  /// handle and \p Dest is a USM pointer. Allows for a sub-region copy, where
  /// \p SrcOffset, \p DestOffset and \p CopyExtent are used to determine the
  /// sub-region. Pixel size is determined by \p SrcImgDesc . An exception is
  /// thrown if either \p Src is incomplete or \p Dest is nullptr. The behavior
  /// is undefined if \p DestRowPitch is inconsistent with hardware
  /// requirements.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the source origin measured in Pixels
  ///                   (Pixel size determined by \p SrcImgDesc .)
  /// \param SrcImgDesc is the source image descriptor
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels. (Pixel size determined by \p SrcImgDesc .)
  /// \param DestImgDesc is the destination image descriptor.
  /// \param DestRowPitch is the pitch of the rows of the destination memory.
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels. (Pixel size determined by
  ///               \p SrcImgDesc .)
  void ext_oneapi_copy(
      const ext::oneapi::experimental::image_mem_handle Src,
      sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      size_t DestRowPitch, sycl::range<3> CopyExtent);

  /// Copies data from device to device memory, where \p Src is USM pointer and
  /// \p Dest is an opaque image memory handle. An exception is thrown if either
  /// \p Src is nullptr or \p Dest is incomplete. The behavior is undefined if
  /// \p SrcImgDesc or \p DestImgDesc are inconsistent with their respective
  /// allocated memory regions or \p SrcRowPitch is inconsistent with hardware
  /// requirements.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcImgDesc is the source image descriptor.
  /// \param SrcRowPitch is the pitch of the rows of the source memory.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestImgDesc is the destination image descriptor.
  void ext_oneapi_copy(
      const void *Src,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      size_t SrcRowPitch, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc);

  /// Copies data from device to device memory, where \p Src is USM pointer and
  /// \p Dest is an opaque image memory handle. Allows for a sub-region
  /// copy, where \p SrcOffset, \p DestOffset and \p CopyExtent are used to
  /// determine the sub-region. Pixel size is determined by \p SrcImgDesc . An
  /// exception is thrown if either \p Src is nullptr or \p Dest is incomplete.
  /// The behavior is undefined if \p SrcRowPitch is inconsistent with hardware
  /// requirements.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the source origin measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param SrcRowPitch is the pitch of the rows of the destination memory.
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p SrcImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  void ext_oneapi_copy(
      const void *Src, sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      size_t SrcRowPitch, ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent);

  /// Copies data from DtoD or HtoH memory, where \p Src and \p Dest are USM
  /// pointers. An exception is thrown if either \p Src or \p Dest are nullptr.
  /// The behavior is undefined if \p SrcImgDesc or \p DestImgDesc are
  /// inconsistent with their respective allocated memory regions or
  /// \p SrcRowPitch or \p DestRowPitch are inconsistent with hardware
  /// requirements.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcImgDesc is the source image descriptor.
  /// \param SrcRowPitch is the pitch of the rows of the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestImgDesc is the destination image descriptor.
  /// \param SrcRowPitch is the pitch of the rows of the destination memory.
  void ext_oneapi_copy(
      const void *Src,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      size_t SrcRowPitch, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      size_t DestRowPitch);

  /// Copies data from DtoD or HtoH memory, where \p Src and \p Dest are USM
  /// pointers. Allows for a sub-region copy, where \p SrcOffset, \p DestOffset
  /// and \p CopyExtent are used to determine the sub-region. Pixel size is
  /// determined by \p SrcImgDesc . An exception is thrown if either \p Src or
  /// \p Dest are nullptr. The behavior is undefined if \p SrcRowPitch or
  /// \p DestRowPitch are inconsistent with hardware requirements.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the source origin measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param SrcRowPitch is the pitch of the rows of the destination memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p SrcImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param DestRowPitch is the pitch of the rows of the destination memory.
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  void ext_oneapi_copy(
      const void *Src, sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      size_t SrcRowPitch, void *Dest, sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      size_t DestRowPitch, sycl::range<3> CopyExtent);

  /// Submit a non-blocking device-side wait on an external
  //  semaphore to the queue.
  /// An exception is thrown if \p extSemaphore is incomplete, or if the
  /// type of semaphore requires an explicit value to wait upon.
  ///
  /// \param extSemaphore is an opaque external semaphore object
  void ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::external_semaphore extSemaphore);

  /// Submit a non-blocking device-side wait on an external
  //  semaphore to the queue.
  /// An exception is thrown if \p extSemaphore is incomplete, or if the
  /// type of semaphore does not support waiting on an explicitly passed value.
  ///
  /// \param extSemaphore is an opaque external semaphore object
  /// \param WaitValue is the value that this semaphore will wait upon, until it
  ///                  allows any further commands to execute on the queue.
  void ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::external_semaphore extSemaphore,
      uint64_t WaitValue);

  /// Instruct the queue to signal the external semaphore once all previous
  /// commands submitted to the queue have completed execution.
  /// An exception is thrown if \p extSemaphore is incomplete, or if the
  /// type of semaphore requires an explicit value to signal.
  ///
  /// \param extSemaphore is an opaque external semaphore object
  void ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::external_semaphore extSemaphore);

  /// Instruct the queue to set the state of the external semaphore to
  /// \p SignalValue once all previous commands submitted to the queue have
  /// completed execution.
  /// An exception is thrown if \p extSemaphore is incomplete, or if the
  /// type of semaphore does not support signalling an explicitly passed value.
  ///
  /// \param extSemaphore is an opaque external semaphore object.
  /// \param SignalValue is the value that this semaphore signal, once all
  ///                    prior opeartions on the queue complete.
  void ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::external_semaphore extSemaphore,
      uint64_t SignalValue);

private:
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  std::unique_ptr<detail::handler_impl> implOwner;
  detail::handler_impl *impl;
#else
  std::shared_ptr<detail::handler_impl> impl;

  // Use impl->get_queue*() instead:
  std::shared_ptr<detail::queue_impl> MQueueDoNotUse;
#endif
  std::vector<detail::LocalAccessorImplPtr> MLocalAccStorage;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreamStorage;
  detail::ABINeutralKernelNameStrT MKernelName;
  /// Storage for a sycl::kernel object.
  std::shared_ptr<detail::kernel_impl> MKernel;
  /// Pointer to the source host memory or accessor(depending on command type).
  void *MSrcPtr = nullptr;
  /// Pointer to the dest host memory or accessor(depends on command type).
  void *MDstPtr = nullptr;
  /// Length to copy or fill (for USM operations).
  size_t MLength = 0;
  /// Pattern that is used to fill memory object in case command type is fill.
  std::vector<unsigned char> MPattern;
  /// Storage for a lambda or function object.
  std::unique_ptr<detail::HostKernelBase> MHostKernel;

  detail::code_location MCodeLoc = {};
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Was used for the previous reduction implementation (via `withAuxHandler`).
  bool MIsFinalizedDoNotUse = false;
  event MLastEventDoNotUse;
#endif

  // Make queue_impl class friend to be able to call finalize method.
  friend class detail::queue_impl;
  // Make accessor class friend to keep the list of associated accessors.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder,
            typename PropertyListT>
  friend class accessor;
  friend device detail::getDeviceFromHandler(handler &);
  friend detail::device_impl &detail::getDeviceImplFromHandler(handler &);

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

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  /// Read from a host pipe given a host address and
  /// \param Name name of the host pipe to be passed into lower level runtime
  /// \param Ptr host pointer of host pipe as identified by address of its const
  ///        expr m_Storage member
  /// \param Size the size of data getting read back / to.
  /// \param Block if read operation is blocking, default to false.
  void ext_intel_read_host_pipe(const std::string &Name, void *Ptr, size_t Size,
                                bool Block = false) {
    ext_intel_read_host_pipe(detail::string_view(Name), Ptr, Size, Block);
  }
  void ext_intel_read_host_pipe(detail::string_view Name, void *Ptr,
                                size_t Size, bool Block = false);

  /// Write to host pipes given a host address and
  /// \param Name name of the host pipe to be passed into lower level runtime
  /// \param Ptr host pointer of host pipe as identified by address of its const
  /// expr m_Storage member
  /// \param Size the size of data getting read back / to.
  /// \param Block if write opeartion is blocking, default to false.
  void ext_intel_write_host_pipe(const std::string &Name, void *Ptr,
                                 size_t Size, bool Block = false) {
    ext_intel_write_host_pipe(detail::string_view(Name), Ptr, Size, Block);
  }
  void ext_intel_write_host_pipe(detail::string_view Name, void *Ptr,
                                 size_t Size, bool Block = false);
  friend class ext::oneapi::experimental::detail::graph_impl;
  friend class ext::oneapi::experimental::detail::dynamic_parameter_impl;
  friend class ext::oneapi::experimental::detail::dynamic_command_group_impl;

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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  const std::shared_ptr<detail::context_impl> &getContextImplPtr() const;
#endif
  detail::context_impl &getContextImpl() const;

  // Checks if 2D memory operations are supported by the underlying platform.
  bool supportsUSMMemcpy2D();
  bool supportsUSMFill2D();
  bool supportsUSMMemset2D();

  // Helper function for getting a loose bound on work-items.
  id<2> computeFallbackKernelBounds(size_t Width, size_t Height);

  // Function to get information about the backend for which the code is
  // compiled for
  backend getDeviceBackend() const;

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
    setType(detail::CGType::Fill);
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = static_cast<void *>(AccImpl.get());

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

  // Implementation of USM fill using command for native fill.
  void fill_impl(void *Dest, const void *Value, size_t ValueSize, size_t Count);

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

  // Changing values in this will break ABI/API.
  enum class StableKernelCacheConfig : int32_t {
    Default = 0,
    LargeSLM = 1,
    LargeData = 2
  };

  // Set value of the gpu cache configuration for the kernel.
  void setKernelCacheConfig(StableKernelCacheConfig);
  // Set value of the kernel is cooperative flag
  void setKernelIsCooperative(bool);

  // Set using cuda thread block cluster launch flag and set the launch bounds.
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void setKernelClusterLaunch(sycl::range<3> ClusterSize, int Dims);
#endif
  void setKernelClusterLaunch(sycl::range<3> ClusterSize);
  void setKernelClusterLaunch(sycl::range<2> ClusterSize);
  void setKernelClusterLaunch(sycl::range<1> ClusterSize);

  // Set the request work group memory size (work_group_static ext).
  void setKernelWorkGroupMem(size_t Size);

  // Various checks that are only meaningful for host compilation, because they
  // result in runtime errors (i.e. exceptions being thrown). To save time
  // during device compilations (by reducing amount of templates we have to
  // instantiate), those are only available during host compilation pass.
#ifndef __SYCL_DEVICE_ONLY__
  constexpr static int AccessTargetMask = 0x7ff;
  /// According to section 4.7.6.11. of the SYCL specification, a local accessor
  /// must not be used in a SYCL kernel function that is invoked via single_task
  /// or via the simple form of parallel_for that takes a range parameter.
  //
  // Exception handling generates lots of code, outline it out of template
  // method to improve compilation times.
  void throwOnKernelParameterMisuseHelper(
      int N, detail::kernel_param_desc_t (*f)(int)) const {
    for (int I = 0; I < N; ++I) {
      detail::kernel_param_desc_t ParamDesc = (*f)(I);
      const detail::kernel_param_kind_t &Kind = ParamDesc.kind;
      const access::target AccTarget =
          static_cast<access::target>(ParamDesc.info & AccessTargetMask);
      if ((Kind == detail::kernel_param_kind_t::kind_accessor) &&
          (AccTarget == target::local))
        throw sycl::exception(
            make_error_code(errc::kernel_argument),
            "A local accessor must not be used in a SYCL kernel function "
            "that is invoked via single_task or via the simple form of "
            "parallel_for that takes a range parameter.");
      if (Kind == detail::kernel_param_kind_t::kind_work_group_memory ||
          Kind == detail::kernel_param_kind_t::kind_dynamic_work_group_memory)
        throw sycl::exception(
            make_error_code(errc::kernel_argument),
            "A work group memory object must not be used in a SYCL kernel "
            "function that is invoked via single_task or via the simple form "
            "of parallel_for that takes a range parameter.");
    }
  }
  template <typename KernelName, typename KernelType>
  void throwOnKernelParameterMisuse() const {
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    throwOnKernelParameterMisuseHelper(detail::getKernelNumParams<NameT>(),
                                       &detail::getKernelParamDesc<NameT>);
  }

  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t,
            typename PropertyListT = property_list>
  void checkIfPlaceholderIsBoundToHandler(
      accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder, PropertyListT>
          Acc) {
    auto *AccBase = reinterpret_cast<detail::AccessorBaseHost *>(&Acc);
    detail::AccessorImplHost *Req = detail::getSyclObjImpl(*AccBase).get();
    if (HasAssociatedAccessor(Req, AccessTarget))
      throw sycl::exception(make_error_code(errc::kernel_argument),
                            "placeholder accessor must be bound by calling "
                            "handler::require() before it can be used.");
  }

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
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Set that an ND Range was used during a call to parallel_for
  void setNDRangeUsed(bool Value);
#endif

  inline void internalProfilingTagImpl() {
    throwIfActionIsCreated();
    setType(detail::CGType::ProfilingTag);
  }

  void addAccessorReq(detail::AccessorImplPtr Accessor);

  void addLifetimeSharedPtrStorage(std::shared_ptr<const void> SPtr);

  void addArg(detail::kernel_param_kind_t ArgKind, void *Req, int AccessTarget,
              int ArgIndex);
  void clearArgs();
  void setArgsToAssociatedAccessors();

  bool HasAssociatedAccessor(detail::AccessorImplHost *Req,
                             access::target AccessTarget) const;

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void setNDRangeDescriptorPadded(sycl::range<3> N, bool SetNumWorkGroups,
                                  int Dims);
  void setNDRangeDescriptorPadded(sycl::range<3> NumWorkItems,
                                  sycl::id<3> Offset, int Dims);
  void setNDRangeDescriptorPadded(sycl::range<3> NumWorkItems,
                                  sycl::range<3> LocalSize, sycl::id<3> Offset,
                                  int Dims);
#endif

  template <int Dims>
  void setNDRangeDescriptor(sycl::range<Dims> N,
                            bool SetNumWorkGroups = false) {
    return setNDRangeDescriptor(N, SetNumWorkGroups);
  }
  template <int Dims>
  void setNDRangeDescriptor(sycl::range<Dims> NumWorkItems,
                            sycl::id<Dims> Offset) {
    return setNDRangeDescriptor(NumWorkItems, Offset);
  }
  template <int Dims>
  void setNDRangeDescriptor(sycl::nd_range<Dims> ExecutionRange) {
    return setNDRangeDescriptor(ExecutionRange.get_global_range(),
                                ExecutionRange.get_local_range(),
                                ExecutionRange.get_offset());
  }

  void setNDRangeDescriptor(sycl::range<3> N, bool SetNumWorkGroups);
  void setNDRangeDescriptor(sycl::range<3> NumWorkItems, sycl::id<3> Offset);
  void setNDRangeDescriptor(sycl::range<3> NumWorkItems,
                            sycl::range<3> LocalSize, sycl::id<3> Offset);

  void setNDRangeDescriptor(sycl::range<2> N, bool SetNumWorkGroups);
  void setNDRangeDescriptor(sycl::range<2> NumWorkItems, sycl::id<2> Offset);
  void setNDRangeDescriptor(sycl::range<2> NumWorkItems,
                            sycl::range<2> LocalSize, sycl::id<2> Offset);

  void setNDRangeDescriptor(sycl::range<1> N, bool SetNumWorkGroups);
  void setNDRangeDescriptor(sycl::range<1> NumWorkItems, sycl::id<1> Offset);
  void setNDRangeDescriptor(sycl::range<1> NumWorkItems,
                            sycl::range<1> LocalSize, sycl::id<1> Offset);

  void setKernelInfo(void *KernelFuncPtr, int KernelNumArgs,
                     detail::kernel_param_desc_t (*KernelParamDescGetter)(int),
                     bool KernelIsESIMD, bool KernelHasSpecialCaptures);

  void instantiateKernelOnHost(void *InstantiateKernelOnHostPtr);

  friend class detail::HandlerAccess;
  friend struct detail::KernelLaunchPropertyWrapper;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  __SYCL_DLL_LOCAL detail::handler_impl *get_impl() { return impl; }
#else
  __SYCL_DLL_LOCAL detail::handler_impl *get_impl() { return impl.get(); }
#endif
  // Friend free-functions for asynchronous allocation and freeing.
  __SYCL_EXPORT friend void
  ext::oneapi::experimental::async_free(sycl::handler &h, void *ptr);

  __SYCL_EXPORT friend void *
  ext::oneapi::experimental::async_malloc(sycl::handler &h,
                                          sycl::usm::alloc kind, size_t size);

  __SYCL_EXPORT friend void *ext::oneapi::experimental::async_malloc_from_pool(
      sycl::handler &h, size_t size,
      const ext::oneapi::experimental::memory_pool &pool);

  void setKernelNameBasedCachePtr(
      detail::KernelNameBasedCacheT *KernelNameBasedCachePtr);

  queue getQueue();

protected:
  /// Registers event dependencies in this command group.
  void depends_on(const detail::EventImplPtr &Event);
  /// Registers event dependencies in this command group.
  void depends_on(const std::vector<detail::EventImplPtr> &Events);
};

namespace detail {
class HandlerAccess {
public:
  static void internalProfilingTagImpl(handler &Handler) {
    Handler.internalProfilingTagImpl();
  }

  template <typename RangeT, typename PropertiesT>
  static void parallelForImpl(handler &Handler, RangeT Range, PropertiesT Props,
                              kernel Kernel) {
    Handler.parallel_for_impl(Range, Props, Kernel);
  }

  static void swap(handler &LHS, handler &RHS) {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    std::swap(LHS.implOwner, RHS.implOwner);
#endif
    std::swap(LHS.impl, RHS.impl);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    std::swap(LHS.MQueueDoNotUse, RHS.MQueueDoNotUse);
#endif
    std::swap(LHS.MLocalAccStorage, RHS.MLocalAccStorage);
    std::swap(LHS.MStreamStorage, RHS.MStreamStorage);
    std::swap(LHS.MKernelName, RHS.MKernelName);
    std::swap(LHS.MKernel, RHS.MKernel);
    std::swap(LHS.MSrcPtr, RHS.MSrcPtr);
    std::swap(LHS.MDstPtr, RHS.MDstPtr);
    std::swap(LHS.MLength, RHS.MLength);
    std::swap(LHS.MPattern, RHS.MPattern);
    std::swap(LHS.MHostKernel, RHS.MHostKernel);
    std::swap(LHS.MCodeLoc, RHS.MCodeLoc);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    std::swap(LHS.MIsFinalizedDoNotUse, RHS.MIsFinalizedDoNotUse);
    std::swap(LHS.MLastEventDoNotUse, RHS.MLastEventDoNotUse);
#endif
  }

  // pre/postProcess are used only for reductions right now, but the
  // abstractions they provide aren't reduction-specific. The main problem they
  // solve is
  //
  //   # User code
  //   q.submit([&](handler &cgh) {
  //     set_dependencies(cgh);
  //     enqueue_whatever(cgh);
  //   });  // single submission
  //
  // that needs to be implemented as multiple enqueues involving
  // pre-/post-processing internally. SYCL prohibits recursive submits from
  // inside control group function object (lambda above) so we need some
  // internal interface to implement that.
  __SYCL_EXPORT static void preProcess(handler &CGH, type_erased_cgfo_ty F);
  __SYCL_EXPORT static void postProcess(handler &CGH, type_erased_cgfo_ty F);

  template <class FunctorTy>
  static void preProcess(handler &CGH, FunctorTy &Func) {
    preProcess(CGH, type_erased_cgfo_ty{Func});
  }
  template <class FunctorTy>
  static void postProcess(handler &CGH, FunctorTy &Func) {
    postProcess(CGH, type_erased_cgfo_ty{Func});
  }
};
} // namespace detail

} // namespace _V1
} // namespace sycl
