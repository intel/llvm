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
  AsyncAlloc = 28,
  AsyncFree = 29,
};

// The structure represents NDRange - global, local sizes, global offset and
// number of dimensions.

// TODO: A lot of tests rely on particular values to be set for dimensions that
// are not used. To clarify, for example, if a 2D kernel is invoked, in
// NDRDescT, the value of index 2 in GlobalSize must be set to either 1 or 0
// depending on which constructor is used for no clear reason.
// Instead, only sensible defaults should be used and tests should be updated
// to reflect this.
class NDRDescT {

public:
  NDRDescT() = default;
  NDRDescT(const NDRDescT &Desc) = default;
  NDRDescT(NDRDescT &&Desc) = default;

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> N, bool SetNumWorkGroups) : Dims{size_t(Dims_)} {
    if (SetNumWorkGroups) {
      for (size_t I = 0; I < Dims_; ++I) {
        NumWorkGroups[I] = N[I];
      }
    } else {
      for (size_t I = 0; I < Dims_; ++I) {
        GlobalSize[I] = N[I];
      }

      for (int I = Dims_; I < 3; ++I) {
        GlobalSize[I] = 1;
      }
    }
  }

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> NumWorkItems, sycl::range<Dims_> LocalSizes,
           sycl::id<Dims_> Offset)
      : Dims{size_t(Dims_)} {
    for (size_t I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = LocalSizes[I];
      GlobalOffset[I] = Offset[I];
    }

    for (int I = Dims_; I < 3; ++I) {
      LocalSize[I] = LocalSizes[0] ? 1 : 0;
    }

    for (int I = Dims_; I < 3; ++I) {
      GlobalSize[I] = 1;
    }
  }

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> NumWorkItems, sycl::id<Dims_> Offset)
      : Dims{size_t(Dims_)} {
    for (size_t I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      GlobalOffset[I] = Offset[I];
    }
  }

  template <int Dims_>
  NDRDescT(sycl::nd_range<Dims_> ExecutionRange)
      : NDRDescT(ExecutionRange.get_global_range(),
                 ExecutionRange.get_local_range(),
                 ExecutionRange.get_offset()) {}

  template <int Dims_>
  NDRDescT(sycl::range<Dims_> Range)
      : NDRDescT(Range, /*SetNumWorkGroups=*/false) {}

  template <int Dims_> void setClusterDimensions(sycl::range<Dims_> N) {
    if (this->Dims != size_t(Dims_)) {
      throw std::runtime_error(
          "Dimensionality of cluster, global and local ranges must be same");
    }

    for (int I = 0; I < Dims_; ++I)
      ClusterDimensions[I] = N[I];
  }

  NDRDescT &operator=(const NDRDescT &Desc) = default;
  NDRDescT &operator=(NDRDescT &&Desc) = default;

  std::array<size_t, 3> GlobalSize{0, 0, 0};
  std::array<size_t, 3> LocalSize{0, 0, 0};
  std::array<size_t, 3> GlobalOffset{0, 0, 0};
  /// Number of workgroups, used to record the number of workgroups from the
  /// simplest form of parallel_for_work_group. If set, all other fields must be
  /// zero
  std::array<size_t, 3> NumWorkGroups{0, 0, 0};
  std::array<size_t, 3> ClusterDimensions{1, 1, 1};
  size_t Dims = 0;
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

template <typename KernelType, bool HasKernelHandlerArg>
void runKernelWithoutArg(KernelType KernelName,
                         const std::bool_constant<HasKernelHandlerArg> &) {
  if constexpr (HasKernelHandlerArg) {
    kernel_handler KH;
    KernelName(KH);
  } else {
    KernelName();
  }
}
template <typename ArgType, typename KernelType, bool HasKernelHandlerArg>
void runKernelWithArg(KernelType KernelName, ArgType Arg,
                      const std::bool_constant<HasKernelHandlerArg> &) {
  if constexpr (HasKernelHandlerArg) {
    kernel_handler KH;
    KernelName(Arg, KH);
  } else {
    KernelName(Arg);
  }
}

// The pure virtual class aimed to store lambda/functors of any type.
class HostKernelBase {
public:
  // Return pointer to the lambda object.
  // Used to extract captured variables.
  virtual char *getPtr() = 0;
  virtual ~HostKernelBase() = default;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // NOTE: InstatiateKernelOnHost() should not be called.
  virtual void InstantiateKernelOnHost() = 0;
#endif
};

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims>
class HostKernel : public HostKernelBase {
  KernelType MKernel;

public:
  HostKernel(const KernelType &Kernel) : MKernel(Kernel) {}
  HostKernel(KernelType &&Kernel) : MKernel(std::move(Kernel)) {}

  char *getPtr() override { return reinterpret_cast<char *>(&MKernel); }

  ~HostKernel() = default;

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // This function is needed for host-side compilation to keep kernels
  // instantitated. This is important for debuggers to be able to associate
  // kernel code instructions with source code lines.
  // NOTE: InstatiateKernelOnHost() should not be called.
  void InstantiateKernelOnHost() override {
    using IDBuilder = sycl::detail::Builder;
    constexpr bool HasKernelHandlerArg =
        KernelLambdaHasKernelHandlerArgT<KernelType, KernelArgType>::value;
    if constexpr (std::is_same_v<KernelArgType, void>) {
      runKernelWithoutArg(MKernel, std::bool_constant<HasKernelHandlerArg>());
    } else if constexpr (std::is_same_v<KernelArgType, sycl::id<Dims>>) {
      sycl::id ID = InitializedVal<Dims, id>::template get<0>();
      runKernelWithArg<const KernelArgType &>(
          MKernel, ID, std::bool_constant<HasKernelHandlerArg>());
    } else if constexpr (std::is_same_v<KernelArgType, item<Dims, true>> ||
                         std::is_same_v<KernelArgType, item<Dims, false>>) {
      constexpr bool HasOffset =
          std::is_same_v<KernelArgType, item<Dims, true>>;
      if constexpr (!HasOffset) {
        KernelArgType Item = IDBuilder::createItem<Dims, HasOffset>(
            InitializedVal<Dims, range>::template get<1>(),
            InitializedVal<Dims, id>::template get<0>());
        runKernelWithArg<KernelArgType>(
            MKernel, Item, std::bool_constant<HasKernelHandlerArg>());
      } else {
        KernelArgType Item = IDBuilder::createItem<Dims, HasOffset>(
            InitializedVal<Dims, range>::template get<1>(),
            InitializedVal<Dims, id>::template get<0>(),
            InitializedVal<Dims, id>::template get<0>());
        runKernelWithArg<KernelArgType>(
            MKernel, Item, std::bool_constant<HasKernelHandlerArg>());
      }
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
      runKernelWithArg<const KernelArgType>(
          MKernel, NDItem, std::bool_constant<HasKernelHandlerArg>());
    } else if constexpr (std::is_same_v<KernelArgType, sycl::group<Dims>>) {
      sycl::range<Dims> Range = InitializedVal<Dims, range>::template get<1>();
      sycl::id<Dims> ID = InitializedVal<Dims, id>::template get<0>();
      KernelArgType Group =
          IDBuilder::createGroup<Dims>(Range, Range, Range, ID);
      runKernelWithArg<KernelArgType>(
          MKernel, Group, std::bool_constant<HasKernelHandlerArg>());
    } else {
      // Assume that anything else can be default-constructed. If not, this
      // should fail to compile and the implementor should implement a generic
      // case for the new argument type.
      runKernelWithArg<KernelArgType>(
          MKernel, KernelArgType{}, std::bool_constant<HasKernelHandlerArg>());
    }
  }
#endif
};

// This function is needed for host-side compilation to keep kernels
// instantitated. This is important for debuggers to be able to associate
// kernel code instructions with source code lines.
template <class KernelType, class KernelArgType, int Dims>
constexpr void *GetInstantiateKernelOnHostPtr() {
  if constexpr (std::is_same_v<KernelArgType, void>) {
    constexpr bool HasKernelHandlerArg =
        KernelLambdaHasKernelHandlerArgT<KernelType>::value;
    return reinterpret_cast<void *>(
        &runKernelWithoutArg<KernelType, HasKernelHandlerArg>);
  } else {
    constexpr bool HasKernelHandlerArg =
        KernelLambdaHasKernelHandlerArgT<KernelType, KernelArgType>::value;
    return reinterpret_cast<void *>(
        &runKernelWithArg<KernelArgType, KernelType, HasKernelHandlerArg>);
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
