//==---- cg_types.hpp - Auxiliary types required by command group class ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/interop_handle.hpp>
#include <CL/sycl/interop_handler.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/kernel_handler.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// The structure represents kernel argument.
class ArgDesc {
public:
  ArgDesc(cl::sycl::detail::kernel_param_kind_t Type, void *Ptr, int Size,
          int Index)
      : MType(Type), MPtr(Ptr), MSize(Size), MIndex(Index) {}

  cl::sycl::detail::kernel_param_kind_t MType;
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
      : GlobalSize{0, 0, 0}, LocalSize{0, 0, 0}, NumWorkGroups{0, 0, 0} {}

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

template <typename KernelType, typename LambdaArgType,
          typename std::enable_if_t<std::is_same<LambdaArgType, void>::value>
              * = nullptr>
constexpr bool isKernelLambdaCallableWithKernelHandlerImpl() {
  return check_kernel_lambda_takes_args<KernelType, kernel_handler>();
}

template <typename KernelType, typename LambdaArgType,
          typename std::enable_if_t<!std::is_same<LambdaArgType, void>::value>
              * = nullptr>
constexpr bool isKernelLambdaCallableWithKernelHandlerImpl() {
  return check_kernel_lambda_takes_args<KernelType, LambdaArgType,
                                        kernel_handler>();
}

// Type traits to find out if kernal lambda has kernel_handler argument

template <typename KernelType>
constexpr bool isKernelLambdaCallableWithKernelHandler() {
  return check_kernel_lambda_takes_args<KernelType, kernel_handler>();
}

template <typename KernelType, typename LambdaArgType>
constexpr bool isKernelLambdaCallableWithKernelHandler() {
  return isKernelLambdaCallableWithKernelHandlerImpl<KernelType,
                                                     LambdaArgType>();
}

// Helpers for running kernel lambda on the host device

template <typename KernelType> void runKernelWithoutArg(KernelType KernelName) {
  if constexpr (isKernelLambdaCallableWithKernelHandler<KernelType>()) {
    kernel_handler KH;
    KernelName(KH);
  } else {
    KernelName();
  }
}

template <typename ArgType, typename KernelType>
constexpr void runKernelWithArg(KernelType KernelName, ArgType Arg) {
  if constexpr (isKernelLambdaCallableWithKernelHandler<KernelType,
                                                        ArgType>()) {
    kernel_handler KH;
    KernelName(Arg, KH);
  } else {
    KernelName(Arg);
  }
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

class InteropTask {
  std::function<void(cl::sycl::interop_handler)> MFunc;

public:
  InteropTask(function_class<void(cl::sycl::interop_handler)> Func)
      : MFunc(Func) {}
  void call(cl::sycl::interop_handler &h) { MFunc(h); }
};

class HostTask {
  std::function<void()> MHostTask;
  std::function<void(interop_handle)> MInteropTask;

public:
  HostTask() : MHostTask([]() {}) {}
  HostTask(std::function<void()> &&Func) : MHostTask(Func) {}
  HostTask(std::function<void(interop_handle)> &&Func) : MInteropTask(Func) {}

  bool isInteropTask() const { return !!MInteropTask; }

  void call() { MHostTask(); }
  void call(interop_handle handle) { MInteropTask(handle); }
};

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims, typename KernelName>
class HostKernel : public HostKernelBase {
  using IDBuilder = sycl::detail::Builder;
  KernelType MKernel;

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
  typename detail::enable_if_t<std::is_same<ArgT, void>::value>
  runOnHost(const NDRDescT &) {
    runKernelWithoutArg(MKernel);
  }

  template <class ArgT = KernelArgType>
  typename detail::enable_if_t<std::is_same<ArgT, sycl::id<Dims>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    using KI = detail::KernelInfo<KernelName>;
    constexpr bool StoreLocation = KI::callsAnyThisFreeFunction();

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

          if (StoreLocation) {
            store_id(&ID);
            store_item(&Item);
          }
          runKernelWithArg<const sycl::id<Dims> &>(MKernel, ID);
        });
  }

  template <class ArgT = KernelArgType>
  typename detail::enable_if_t<
      std::is_same<ArgT, item<Dims, /*Offset=*/false>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    using KI = detail::KernelInfo<KernelName>;
    constexpr bool StoreLocation = KI::callsAnyThisFreeFunction();

    sycl::id<Dims> ID;
    sycl::range<Dims> Range(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    detail::NDLoop<Dims>::iterate(Range, [&](const sycl::id<Dims> ID) {
      sycl::item<Dims, /*Offset=*/false> Item =
          IDBuilder::createItem<Dims, false>(Range, ID);
      sycl::item<Dims, /*Offset=*/true> ItemWithOffset = Item;

      if (StoreLocation) {
        store_id(&ID);
        store_item(&ItemWithOffset);
      }
      runKernelWithArg<sycl::item<Dims, /*Offset=*/false>>(MKernel, Item);
    });
  }

  template <class ArgT = KernelArgType>
  typename detail::enable_if_t<
      std::is_same<ArgT, item<Dims, /*Offset=*/true>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    using KI = detail::KernelInfo<KernelName>;
    constexpr bool StoreLocation = KI::callsAnyThisFreeFunction();

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

          if (StoreLocation) {
            store_id(&ID);
            store_item(&Item);
          }
          runKernelWithArg<sycl::item<Dims, /*Offset=*/true>>(MKernel, Item);
        });
  }

  template <class ArgT = KernelArgType>
  typename detail::enable_if_t<std::is_same<ArgT, nd_item<Dims>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    using KI = detail::KernelInfo<KernelName>;
    constexpr bool StoreLocation = KI::callsAnyThisFreeFunction();

    sycl::range<Dims> GroupSize(InitializedVal<Dims, range>::template get<0>());
    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_INVALID_WORK_GROUP_SIZE);
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

        if (StoreLocation) {
          store_id(&GlobalID);
          store_item(&GlobalItem);
          store_nd_item(&NDItem);
          auto g = NDItem.get_group();
          store_group(&g);
        }
        runKernelWithArg<const sycl::nd_item<Dims>>(MKernel, NDItem);
      });
    });
  }

  template <typename ArgT = KernelArgType>
  enable_if_t<std::is_same<ArgT, cl::sycl::group<Dims>>::value>
  runOnHost(const NDRDescT &NDRDesc) {
    sycl::range<Dims> NGroups(InitializedVal<Dims, range>::template get<0>());

    for (int I = 0; I < Dims; ++I) {
      if (NDRDesc.LocalSize[I] == 0 ||
          NDRDesc.GlobalSize[I] % NDRDesc.LocalSize[I] != 0)
        throw sycl::nd_range_error("Invalid local size for global size",
                                   PI_INVALID_WORK_GROUP_SIZE);
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
