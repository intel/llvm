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
};

} // namespace detail
} // namespace _V1
} // namespace sycl
