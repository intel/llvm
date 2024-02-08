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
#include <sycl/detail/pi.h>                    // for PI_ERROR_INVALID_WORK...
#include <sycl/exception.hpp>                  // for nd_range_error
#include <sycl/group.hpp>                      // for group
#include <sycl/h_item.hpp>                     // for h_item
#include <sycl/id.hpp>                         // for id
#include <sycl/interop_handle.hpp>             // for interop_handle
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
namespace detail {

// The structure represents kernel argument.
class ArgDesc {
public:
  ArgDesc(sycl::detail::kernel_param_kind_t Type, void *Ptr, int Size,
          int Index)
      : MType(Type), MPtr(Ptr), MSize(Size), MIndex(Index) {}

  sycl::detail::kernel_param_kind_t MType;
  void *MPtr;
  int MSize;
  int MIndex;
};

// The structure represents NDRange - global, local sizes, global offset and
// number of dimensions.
class NDRDescT {
  // The method initializes all sizes for dimensions greater than the passed one
  // to the default values, so they will not affect execution.
  void setNDRangeLeftover(int Dims_) {
    for (int I = Dims_; I < 3; ++I) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = 0;
    }
  }

public:
  NDRDescT()
      : GlobalSize{0, 0, 0}, LocalSize{0, 0, 0}, NumWorkGroups{0, 0, 0},
        Dims{0} {}

  template <int Dims_> void set(sycl::range<Dims_> NumWorkItems) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  // Initializes this ND range descriptor with given range of work items and
  // offset.
  template <int Dims_>
  void set(sycl::range<Dims_> NumWorkItems, sycl::id<Dims_> Offset) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = 0;
      GlobalOffset[I] = Offset[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  template <int Dims_> void set(sycl::nd_range<Dims_> ExecutionRange) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = ExecutionRange.get_global_range()[I];
      LocalSize[I] = ExecutionRange.get_local_range()[I];
      GlobalOffset[I] = ExecutionRange.get_offset()[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  void set(int Dims_, sycl::nd_range<3> ExecutionRange) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = ExecutionRange.get_global_range()[I];
      LocalSize[I] = ExecutionRange.get_local_range()[I];
      GlobalOffset[I] = ExecutionRange.get_offset()[I];
      NumWorkGroups[I] = 0;
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  template <int Dims_> void setNumWorkGroups(sycl::range<Dims_> N) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = 0;
      // '0' is a mark to adjust before kernel launch when there is enough info:
      LocalSize[I] = 0;
      GlobalOffset[I] = 0;
      NumWorkGroups[I] = N[I];
    }
    setNDRangeLeftover(Dims_);
    Dims = Dims_;
  }

  sycl::range<3> GlobalSize;
  sycl::range<3> LocalSize;
  sycl::id<3> GlobalOffset;
  /// Number of workgroups, used to record the number of workgroups from the
  /// simplest form of parallel_for_work_group. If set, all other fields must be
  /// zero
  sycl::range<3> NumWorkGroups;
  size_t Dims;
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
  // The method executes lambda stored using NDRange passed.
  virtual void call(const NDRDescT &NDRDesc, HostProfilingInfo *HPI) = 0;
  // Return pointer to the lambda object.
  // Used to extract captured variables.
  virtual char *getPtr() = 0;
  virtual ~HostKernelBase() = default;
};

class HostTask {
  std::function<void()> MHostTask;
  std::function<void(interop_handle)> MInteropTask;

public:
  HostTask() : MHostTask([]() {}) {}
  HostTask(std::function<void()> &&Func) : MHostTask(Func) {}
  HostTask(std::function<void(interop_handle)> &&Func) : MInteropTask(Func) {}

  bool isInteropTask() const { return !!MInteropTask; }

  void call(HostProfilingInfo *HPI) {
    if (HPI)
      HPI->start();
    MHostTask();
    if (HPI)
      HPI->end();
  }

  void call(HostProfilingInfo *HPI, interop_handle handle) {
    if (HPI)
      HPI->start();
    MInteropTask(handle);
    if (HPI)
      HPI->end();
  }
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
  void call(const NDRDescT &NDRDesc, HostProfilingInfo *HPI) override {
    // adjust ND range for serial host:
    NDRDescT AdjustedRange = NDRDesc;

    if (NDRDesc.GlobalSize[0] == 0 && NDRDesc.NumWorkGroups[0] != 0) {
      // This is a special case - NDRange information is not complete, only the
      // desired number of work groups is set by the user. Choose work group
      // size (LocalSize), calculate the missing NDRange characteristics
      // needed to invoke the kernel and adjust the NDRange descriptor
      // accordingly. For some devices the work group size selection requires
      // access to the device's properties, hence such late "adjustment".
      range<3> WGsize{1, 1, 1}; // no better alternative for serial host?
      AdjustedRange.set(NDRDesc.Dims,
                        nd_range<3>(NDRDesc.NumWorkGroups * WGsize, WGsize));
    }
    // If local size for host is not set explicitly, let's adjust it to 1,
    // so nd_range_error for zero local size is not thrown.
    if (AdjustedRange.LocalSize[0] == 0)
      for (size_t I = 0; I < AdjustedRange.Dims; ++I)
        AdjustedRange.LocalSize[I] = 1;
    if (HPI)
      HPI->start();
    runOnHost(AdjustedRange);
    if (HPI)
      HPI->end();
  }

  char *getPtr() override { return reinterpret_cast<char *>(&MKernel); }

  template <class ArgT = KernelArgType>
  typename std::enable_if_t<std::is_same_v<ArgT, void>>
  runOnHost(const NDRDescT &) {
    runKernelWithoutArg(MKernel);
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if_t<std::is_same_v<ArgT, sycl::id<Dims>>>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    sycl::id<Dims> Offset;
    sycl::range<Dims> Stride(
        InitializedVal<Dims, range>::template get<1>()); // initialized to 1
    sycl::range<Dims> UpperBound(
        InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      Range[I] = NDRDesc.GlobalSize[I];
      Offset[I] = NDRDesc.GlobalOffset[I];
      UpperBound[I] = Range[I] + Offset[I];
    }

    detail::NDLoop<Dims>::iterate(
        /*LowerBound=*/Offset, Stride, UpperBound,
        [&](const sycl::id<Dims> &ID) {
          sycl::item<Dims, /*Offset=*/true> Item =
              IDBuilder::createItem<Dims, true>(Range, ID, Offset);

          runKernelWithArg<const sycl::id<Dims> &>(MKernel, ID);
        });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if_t<std::is_same_v<ArgT, item<Dims, /*Offset=*/false>>>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::id<Dims> ID;
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    detail::NDLoop<Dims>::iterate(Range, [&](const sycl::id<Dims> ID) {
      sycl::item<Dims, /*Offset=*/false> Item =
          IDBuilder::createItem<Dims, false>(Range, ID);
      sycl::item<Dims, /*Offset=*/true> ItemWithOffset = Item;

      runKernelWithArg<sycl::item<Dims, /*Offset=*/false>>(MKernel, Item);
    });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if_t<std::is_same_v<ArgT, item<Dims, /*Offset=*/true>>>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    sycl::id<Dims> Offset;
    sycl::range<Dims> Stride(
        InitializedVal<Dims, range>::template get<1>()); // initialized to 1
    sycl::range<Dims> UpperBound(
        InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      Range[I] = NDRDesc.GlobalSize[I];
      Offset[I] = NDRDesc.GlobalOffset[I];
      UpperBound[I] = Range[I] + Offset[I];
    }

    detail::NDLoop<Dims>::iterate(
        /*LowerBound=*/Offset, Stride, UpperBound,
        [&](const sycl::id<Dims> &ID) {
          sycl::item<Dims, /*Offset=*/true> Item =
              IDBuilder::createItem<Dims, true>(Range, ID, Offset);

          runKernelWithArg<sycl::item<Dims, /*Offset=*/true>>(MKernel, Item);
        });
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if_t<std::is_same_v<ArgT, nd_item<Dims>>>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> GroupSize(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_ERROR_INVALID_WORK_GROUP_SIZE);
      GroupSize[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(InitializedVal<Dims, range>::template get<0>());
    sycl::range<Dims> GlobalSize(
        InitializedVal<Dims, range>::template get<0>());
    sycl::id<Dims> GlobalOffset;
    for (int I = 0; I < Dims; ++I) {
      GlobalOffset[I] = NDRDesc.GlobalOffset[I];
      LocalSize[I] = NDRDesc.LocalSize[I];
      GlobalSize[I] = NDRDesc.GlobalSize[I];
    }

    detail::NDLoop<Dims>::iterate(GroupSize, [&](const id<Dims> &GroupID) {
      sycl::group<Dims> Group = IDBuilder::createGroup<Dims>(
          GlobalSize, LocalSize, GroupSize, GroupID);

      detail::NDLoop<Dims>::iterate(LocalSize, [&](const id<Dims> &LocalID) {
        id<Dims> GlobalID = GroupID * LocalSize + LocalID + GlobalOffset;
        const sycl::item<Dims, /*Offset=*/true> GlobalItem =
            IDBuilder::createItem<Dims, true>(GlobalSize, GlobalID,
                                              GlobalOffset);
        const sycl::item<Dims, /*Offset=*/false> LocalItem =
            IDBuilder::createItem<Dims, false>(LocalSize, LocalID);
        const sycl::nd_item<Dims> NDItem =
            IDBuilder::createNDItem<Dims>(GlobalItem, LocalItem, Group);

        runKernelWithArg<const sycl::nd_item<Dims>>(MKernel, NDItem);
      });
    });
  }

  template <typename ArgT = KernelArgType>
  std::enable_if_t<std::is_same_v<ArgT, sycl::group<Dims>>>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> NGroups(InitializedVal<Dims, range>::template get<0>());

    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_ERROR_INVALID_WORK_GROUP_SIZE);
      NGroups[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> LocalSize(InitializedVal<Dims, range>::template get<0>());
    sycl::range<Dims> GlobalSize(
        InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      LocalSize[I] = NDRDesc.LocalSize[I];
      GlobalSize[I] = NDRDesc.GlobalSize[I];
    }
    detail::NDLoop<Dims>::iterate(NGroups, [&](const id<Dims> &GroupID) {
      sycl::group<Dims> Group =
          IDBuilder::createGroup<Dims>(GlobalSize, LocalSize, NGroups, GroupID);
      runKernelWithArg<sycl::group<Dims>>(MKernel, Group);
    });
  }

  ~HostKernel() = default;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
