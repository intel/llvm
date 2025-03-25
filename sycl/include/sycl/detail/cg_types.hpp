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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif

class HostKernel
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    : public HostKernelBase
#endif
{
  // SYCL kernels must be device-copyable, so simply storing bytes is enough for
  // RT purposes. However, accessor/stream don't seem to be
  // `std::trivially_copyable`, so we still do placement new/manual destructor
  // invocation as some e2e tests would fail otherwise.
  std::unique_ptr<char[]> KernelBytes;
  void (*KernelDeleter)(void *) = nullptr;
  // NOTE: This is *NOT* for debugger only. Host-side optimizations affect
  // device code linking, see
  // `test-e2e/SeparateCompile/sycl-external-within-staticlib.cpp`.
  void (*InstantiateOnHostHelper)(void *) = nullptr;

  template <class KernelType, class KernelArgType, int Dims>
  struct InstantiateKernelOnHostHelper {
    static void foo(void *ptr) {
      auto &MKernel = *static_cast<KernelType *>(ptr);
      using IDBuilder = sycl::detail::Builder;
      if constexpr (std::is_same_v<KernelArgType, void>) {
        runKernelWithoutArg(MKernel);
      } else if constexpr (std::is_same_v<KernelArgType, sycl::id<Dims>>) {
        sycl::id ID = InitializedVal<Dims, id>::template get<0>();
        runKernelWithArg<const KernelArgType &>(MKernel, ID);
      } else if constexpr (std::is_same_v<KernelArgType, item<Dims, true>> ||
                           std::is_same_v<KernelArgType, item<Dims, false>>) {
        constexpr bool HasOffset =
            std::is_same_v<KernelArgType, item<Dims, true>>;
        if constexpr (!HasOffset) {
          KernelArgType Item = IDBuilder::createItem<Dims, HasOffset>(
              InitializedVal<Dims, range>::template get<1>(),
              InitializedVal<Dims, id>::template get<0>());
          runKernelWithArg<KernelArgType>(MKernel, Item);
        } else {
          KernelArgType Item = IDBuilder::createItem<Dims, HasOffset>(
              InitializedVal<Dims, range>::template get<1>(),
              InitializedVal<Dims, id>::template get<0>(),
              InitializedVal<Dims, id>::template get<0>());
          runKernelWithArg<KernelArgType>(MKernel, Item);
        }
      } else if constexpr (std::is_same_v<KernelArgType, nd_item<Dims>>) {
        sycl::range<Dims> Range =
            InitializedVal<Dims, range>::template get<1>();
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
        sycl::range<Dims> Range =
            InitializedVal<Dims, range>::template get<1>();
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

  template <typename KernelType> struct Deleter {
    static void execute(void *p) {
      static_cast<KernelType *>(p)->~KernelType();
    }
  };

public:
  HostKernel() = default;
  HostKernel(HostKernel &&Other)
      : KernelBytes(std::move(Other.KernelBytes)),
        KernelDeleter(Other.KernelDeleter) {
    Other.KernelDeleter = nullptr;
  }
  HostKernel &operator=(HostKernel &&Other) {
    if (KernelDeleter)
      KernelDeleter(KernelBytes.get());
    KernelBytes = std::move(Other.KernelBytes);
    KernelDeleter = Other.KernelDeleter;
    Other.KernelDeleter = nullptr;
    return *this;
  }

  // Can't specify explicit template parameters when invoking a ctor, so has to
  // be a static member function.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  template <class KernelType, class KernelArgType, int Dims>
  static HostKernel create(KernelType Kernel) {
    HostKernel Tmp;
    Tmp.KernelBytes.reset(
        new (std::align_val_t(alignof(KernelType))) char[sizeof(Kernel)]);
    // Note, `device_copyable` isn't the same as `std::is_trivially_copyable`,
    // so `memcpy` wouldn't be enough.
    new (Tmp.KernelBytes.get()) KernelType(Kernel);
    Tmp.KernelDeleter = &Deleter<KernelType>::execute;
    Tmp.InstantiateOnHostHelper = &InstantiateKernelOnHostHelper<KernelType, KernelArgType, Dims>::foo;
    return Tmp;
  }
#else
  template <class KernelType, class KernelArgType, int Dims>
  static std::unique_ptr<HostKernelBase> create(KernelType Kernel) {
    auto Unique = std::make_unique<HostKernel>();
    Unique->KernelBytes.reset(
        new (std::align_val_t(alignof(KernelType))) char[sizeof(Kernel)]);
    // Note, `device_copyable` isn't the same as `std::is_trivially_copyable`,
    // so `memcpy` wouldn't be enough.
    new (Unique->KernelBytes.get()) KernelType(Kernel);
    Unique->KernelDeleter = &Deleter<KernelType>::execute;
    Unique->InstantiateOnHostHelper = &InstantiateKernelOnHostHelper<KernelType, KernelArgType, Dims>::foo;
    return Unique;
  }
#endif

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  char *getPtr() { return KernelBytes.get(); }
  ~HostKernel() {
    if (KernelDeleter)
      KernelDeleter(KernelBytes.get());
  }
#else
  // Non-preview needs `override`s.
  char *getPtr() override { return KernelBytes.get(); }
  ~HostKernel() override {
    if (KernelDeleter)
      KernelDeleter(KernelBytes.get());
  }

  void InstantiateKernelOnHost() override {}
#endif
};

} // namespace detail
} // namespace _V1
} // namespace sycl
