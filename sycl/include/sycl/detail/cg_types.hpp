//==---- cg_types.hpp - Auxiliary types required by command group class ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/array.hpp>               // for array
#include <sycl/detail/common.hpp>              // for InitializedVal, NDLoop
#include <sycl/detail/helpers.hpp>             // for Builder
#include <sycl/detail/host_profiling_info.hpp> // for HostProfilingInfo
#include <sycl/detail/item_base.hpp>           // for id
#include <sycl/detail/kernel_desc.hpp>         // for kernel_param_kind_t
#include <sycl/exception.hpp>
#include <sycl/group.hpp>                      // for group
#include <sycl/h_item.hpp>                     // for h_item
#include <sycl/id.hpp>                         // for id
#include <sycl/item.hpp>                       // for item
#include <sycl/kernel_handler.hpp>             // for kernel_handler
#include <sycl/nd_item.hpp>                    // for nd_item
#include <sycl/nd_range.hpp>                   // for nd_range
#include <sycl/range.hpp>                      // for range, operator*
#include <ur_api.h> // for UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE

#include <functional>  // for function
#include <stddef.h>    // for size_t
#include <type_traits> // for enable_if_t, false_type
#include <utility>     // for declval

namespace sycl {
inline namespace _V1 {
class interop_handle;
class handler;
namespace detail {
class HostTask;

/// Type of the command group.
/// NOTE: Changing the values of any of these enumerators is an API-break.
enum class CGType : unsigned int {
  None = 0,
  Kernel = 1,
  CopyAccToPtr = 2,
  CopyPtrToAcc = 3,
  CopyAccToAcc = 4,
  Barrier = 5,
  BarrierWaitlist = 6,
  Fill = 7,
  UpdateHost = 8,
  CopyUSM = 10,
  FillUSM = 11,
  PrefetchUSM = 12,
  CodeplayHostTask = 14,
  AdviseUSM = 15,
  Copy2DUSM = 16,
  Fill2DUSM = 17,
  Memset2DUSM = 18,
  CopyToDeviceGlobal = 19,
  CopyFromDeviceGlobal = 20,
  ReadWriteHostPipe = 21,
  ExecCommandBuffer = 22,
  CopyImage = 23,
  SemaphoreWait = 24,
  SemaphoreSignal = 25,
  ProfilingTag = 26,
  EnqueueNativeCommand = 27,
};

template <typename, typename T> struct check_fn_signature {
  static_assert(std::integral_constant<T, false>::value,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct check_fn_signature<F, RetT(Args...)> {
private:
  template <typename T>
  static constexpr auto check(T *) -> typename std::is_same<
      decltype(std::declval<T>().operator()(std::declval<Args>()...)),
      RetT>::type;

  template <typename> static constexpr std::false_type check(...);

  using type = decltype(check<F>(0));

public:
  static constexpr bool value = type::value;
};

template <typename F, typename... Args>
static constexpr bool check_kernel_lambda_takes_args() {
  return check_fn_signature<std::remove_reference_t<F>, void(Args...)>::value;
}

// isKernelLambdaCallableWithKernelHandlerImpl checks if LambdaArgType is void
// (e.g., in single_task), and based on that, calls
// check_kernel_lambda_takes_args with proper set of arguments. Also this type
// trait workarounds compilation error which happens only with msvc.

template <
    typename KernelType, typename LambdaArgType,
    typename std::enable_if_t<std::is_same_v<LambdaArgType, void>> * = nullptr>
constexpr bool isKernelLambdaCallableWithKernelHandlerImpl() {
  return check_kernel_lambda_takes_args<KernelType, kernel_handler>();
}

template <
    typename KernelType, typename LambdaArgType,
    typename std::enable_if_t<!std::is_same_v<LambdaArgType, void>> * = nullptr>
constexpr bool isKernelLambdaCallableWithKernelHandlerImpl() {
  return check_kernel_lambda_takes_args<KernelType, LambdaArgType,
                                        kernel_handler>();
}

// Type trait to find out if kernal lambda has kernel_handler argument
template <typename KernelType, typename LambdaArgType = void>
struct KernelLambdaHasKernelHandlerArgT {
  constexpr static bool value =
      isKernelLambdaCallableWithKernelHandlerImpl<KernelType, LambdaArgType>();
};

// Helpers for running kernel lambda on the host device

template <typename KernelType>
typename std::enable_if_t<KernelLambdaHasKernelHandlerArgT<KernelType>::value>
runKernelWithoutArg(KernelType KernelName) {
  kernel_handler KH;
  KernelName(KH);
}

template <typename KernelType>
typename std::enable_if_t<!KernelLambdaHasKernelHandlerArgT<KernelType>::value>
runKernelWithoutArg(KernelType KernelName) {
  KernelName();
}

template <typename ArgType, typename KernelType>
typename std::enable_if_t<
    KernelLambdaHasKernelHandlerArgT<KernelType, ArgType>::value>
runKernelWithArg(KernelType KernelName, ArgType Arg) {
  kernel_handler KH;
  KernelName(Arg, KH);
}

template <typename ArgType, typename KernelType>
typename std::enable_if_t<
    !KernelLambdaHasKernelHandlerArgT<KernelType, ArgType>::value>
runKernelWithArg(KernelType KernelName, ArgType Arg) {
  KernelName(Arg);
}

// The pure virtual class aimed to store lambda/functors of any type.
class HostKernelBase {
public:
  // Return pointer to the lambda object.
  // Used to extract captured variables.
  virtual char *getPtr() = 0;
  virtual ~HostKernelBase() = default;
  // NOTE: InstatiateKernelOnHost() should not be called.
  virtual void InstantiateKernelOnHost() = 0;
};

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims>
class HostKernel : public HostKernelBase {
  using IDBuilder = sycl::detail::Builder;
  KernelType MKernel;
  // Allowing accessing MKernel from 'ResetHostKernelHelper' method of
  // 'sycl::handler'
  friend class sycl::handler;

public:
  HostKernel(KernelType Kernel) : MKernel(Kernel) {}

  char *getPtr() override { return reinterpret_cast<char *>(&MKernel); }

  ~HostKernel() = default;

  // This function is needed for host-side compilation to keep kernels
  // instantitated. This is important for debuggers to be able to associate
  // kernel code instructions with source code lines.
  // NOTE: InstatiateKernelOnHost() should not be called.
  void InstantiateKernelOnHost() override {
    if constexpr (std::is_same_v<KernelArgType, void>) {
      runKernelWithoutArg(MKernel);
    } else if constexpr (std::is_same_v<KernelArgType, sycl::id<Dims>>) {
      sycl::id ID = InitializedVal<Dims, id>::template get<0>();
      runKernelWithArg<const KernelArgType &>(MKernel, ID);
    } else if constexpr (std::is_same_v<KernelArgType, item<Dims, true>> ||
                         std::is_same_v<KernelArgType, item<Dims, false>>) {
      constexpr bool HasOffset =
          std::is_same_v<KernelArgType, item<Dims, true>>;
      KernelArgType Item = IDBuilder::createItem<Dims, HasOffset>(
          InitializedVal<Dims, range>::template get<1>(),
          InitializedVal<Dims, id>::template get<0>());
      runKernelWithArg<KernelArgType>(MKernel, Item);
    } else if constexpr (std::is_same_v<KernelArgType, nd_item<Dims>>) {
      sycl::range<Dims> Range = InitializedVal<Dims, range>::template get<1>();
      sycl::id<Dims> ID = InitializedVal<Dims, id>::template get<0>();
      sycl::group<Dims> Group =
          IDBuilder::createGroup<Dims>(Range, Range, Range, ID);
      sycl::item<Dims, true> GlobalItem =
          IDBuilder::createItem<Dims, true>(Range, ID, ID);
      sycl::item<Dims, false> LocalItem =
          IDBuilder::createItem<Dims, false>(Range, ID);
      KernelArgType NDItem =
          IDBuilder::createNDItem<Dims>(GlobalItem, LocalItem, Group);
      runKernelWithArg<const KernelArgType>(MKernel, NDItem);
    } else if constexpr (std::is_same_v<KernelArgType, sycl::group<Dims>>) {
      sycl::range<Dims> Range = InitializedVal<Dims, range>::template get<1>();
      sycl::id<Dims> ID = InitializedVal<Dims, id>::template get<0>();
      KernelArgType Group =
          IDBuilder::createGroup<Dims>(Range, Range, Range, ID);
      runKernelWithArg<KernelArgType>(MKernel, Group);
    } else {
      // Assume that anything else can be default-constructed. If not, this
      // should fail to compile and the implementor should implement a generic
      // case for the new argument type.
      runKernelWithArg<KernelArgType>(MKernel, KernelArgType{});
    }
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
