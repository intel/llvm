//==-------- kernel_launch_helper.hpp --- SYCL kernel launch utilities ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/compile_time_kernel_info.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/is_device_copyable.hpp>
#include <sycl/ext/intel/experimental/fp_control_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/kernel_execution_properties.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/use_root_sync_prop.hpp>
#include <sycl/ext/oneapi/experimental/virtual_functions.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/kernel_handler.hpp>

#include <assert.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

enum class WrapAs { single_task, parallel_for, parallel_for_work_group };

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

struct KernelWrapperHelperFuncs {

#ifdef SYCL_LANGUAGE_VERSION
#ifndef __INTEL_SYCL_USE_INTEGRATION_HEADERS
#define __SYCL_KERNEL_ATTR__ [[clang::sycl_kernel_entry_point(KernelName)]]
#else
#define __SYCL_KERNEL_ATTR__ [[clang::sycl_kernel]]
#endif // __INTEL_SYCL_USE_INTEGRATION_HEADERS
#else
#define __SYCL_KERNEL_ATTR__
#endif // SYCL_LANGUAGE_VERSION

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

  __SYCL_KERNEL_ATTR__ static void
  kernel_single_task(const KernelType &KernelFunc) {
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
  __SYCL_KERNEL_ATTR__ static void
  kernel_single_task(const KernelType &KernelFunc, kernel_handler KH) {
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
  __SYCL_KERNEL_ATTR__ static void
  kernel_parallel_for(const KernelType &KernelFunc) {
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
  __SYCL_KERNEL_ATTR__ static void
  kernel_parallel_for(const KernelType &KernelFunc, kernel_handler KH) {
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
  __SYCL_KERNEL_ATTR__ static void
  kernel_parallel_for_work_group(const KernelType &KernelFunc) {
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
  __SYCL_KERNEL_ATTR__ static void
  kernel_parallel_for_work_group(const KernelType &KernelFunc,
                                 kernel_handler KH) {
#ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()), KH);
#else
    (void)KernelFunc;
    (void)KH;
#endif
  }
}; // KernelWrapperSingletonFunc

// The KernelWrapper below has two purposes.
//
// First, from SYCL 2020, Table 129 (Member functions of the `handler ` class)
//   > The callable ... can optionally take a `kernel_handler` ... in
//   > which case the SYCL runtime will construct an instance of
//   > `kernel_handler` and pass it to the callable.
//
// Note: "..." due to slight wording variability between
// single_task/parallel_for (e.g. only parameter vs last). This helper class
// calls `kernel_*` entry points (both hardcoded names known to FE and special
// device-specific entry point attributes) with proper arguments (with/without
// `kernel_handler` argument, depending on the signature of the SYCL kernel
// function).
//
// Second, it performs a few checks and some properties processing (including
// the one provided via `sycl_ext_oneapi_kernel_properties` extension by
// embedding them into the kernel's type).

template <WrapAs WrapAsVal, typename KernelName, typename KernelType,
          typename ElementType,
          typename PropertiesT = ext::oneapi::experimental::empty_properties_t,
          typename MergedPropertiesT = typename detail::
              GetMergedKernelProperties<KernelType, PropertiesT>::type>
struct KernelWrapper;
template <WrapAs WrapAsVal, typename KernelName, typename KernelType,
          typename ElementType, typename PropertiesT, typename... MergedProps>
struct KernelWrapper<
    WrapAsVal, KernelName, KernelType, ElementType, PropertiesT,
    ext::oneapi::experimental::detail::properties_t<MergedProps...>>
    : public KernelWrapperHelperFuncs {

  static void wrap([[maybe_unused]] const KernelType &KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    detail::CheckDeviceCopyable<KernelType>();
#endif
    // Note: the static_assert below need to be run on both the host and the
    // device ends to avoid test issues, so don't put it into the #ifdef
    // __SYCL_DEVICE_ONLY__ directive above print out diagnostic message if
    // the kernel functor has a get(properties_tag) member, but it's not const
    static_assert(
        (ext::oneapi::experimental::detail::HasKernelPropertiesGetMethod<
            const KernelType &>::value) ||
            !(ext::oneapi::experimental::detail::HasKernelPropertiesGetMethod<
                KernelType>::value),
        "get(sycl::ext::oneapi::experimental::properties_tag) member in "
        "kernel functor class must be declared as a const member function");
    auto L = [&](auto &&...args) {
      if constexpr (WrapAsVal == WrapAs::single_task) {
        kernel_single_task<KernelName, KernelType, MergedProps...>(
            std::forward<decltype(args)>(args)...);
      } else if constexpr (WrapAsVal == WrapAs::parallel_for) {
        kernel_parallel_for<KernelName, ElementType, KernelType,
                            MergedProps...>(
            std::forward<decltype(args)>(args)...);
      } else if constexpr (WrapAsVal == WrapAs::parallel_for_work_group) {
        kernel_parallel_for_work_group<KernelName, ElementType, KernelType,
                                       MergedProps...>(
            std::forward<decltype(args)>(args)...);
      } else {
        // Always false, but template-dependent. Can't compare `WrapAsVal`
        // with itself because of `-Wtautological-compare` warning.
        static_assert(!std::is_same_v<KernelName, KernelName>,
                      "Unexpected WrapAsVal");
      }
    };
    if constexpr (detail::KernelLambdaHasKernelHandlerArgT<
                      KernelType, ElementType>::value) {
      kernel_handler KH;
      L(KernelFunc, KH);
    } else {
      L(KernelFunc);
    }
  }
}; // KernelWrapper struct

// This class encapsulates everything related to parsing kernel launch
// properties.
class KernelLaunchPropertyWrapper {
public:
  // This struct is used to store kernel launch properties.
  // std::optional is used to indicate that the property is not set.
  // In some code paths, kernel launch properties are set multiple times
  // for the same kernel, that is why using std::optional to avoid overriding
  // previously set properties.
  // This struct is used to pass kernel launch properties across the ABI
  // boundary.
  struct KernelLaunchPropertiesT {
    struct ScopeForwardProgressProperty {
      std::optional<sycl::ext::oneapi::experimental::forward_progress_guarantee>
          Guarantee;
      std::optional<sycl::ext::oneapi::experimental::execution_scope> ExecScope;
      std::optional<sycl::ext::oneapi::experimental::execution_scope>
          CoordinationScope;
    };

    std::optional<ur_kernel_cache_config_t> MCacheConfig = std::nullopt;
    std::optional<bool> MIsCooperative = std::nullopt;
    std::optional<uint32_t> MWorkGroupMemorySize = std::nullopt;
    std::optional<bool> MUsesClusterLaunch = std::nullopt;
    size_t MClusterDims = 0;
    std::array<size_t, 3> MClusterSize = {0, 0, 0};

    // Forward progress guarantee properties for work_item, sub_group and
    // work_group scopes. We need to store them for validation later.
    std::array<ScopeForwardProgressProperty, 3> MForwardProgressProperties;
  };

  /// Process runtime kernel properties.
  template <typename PropertiesT>
  static KernelLaunchPropertiesT
  processKernelLaunchProperties(PropertiesT Props) {
    using namespace sycl::ext::oneapi::experimental;
    using namespace sycl::ext::oneapi::experimental::detail;
    KernelLaunchPropertiesT retval;

    // Process Kernel cache configuration property.
    {
      if constexpr (PropertiesT::template has_property<
                        sycl::ext::intel::experimental::cache_config_key>()) {
        auto Config = Props.template get_property<
            sycl::ext::intel::experimental::cache_config_key>();
        if (Config == sycl::ext::intel::experimental::large_slm) {
          retval.MCacheConfig = UR_KERNEL_CACHE_CONFIG_LARGE_SLM;
        } else if (Config == sycl::ext::intel::experimental::large_data) {
          retval.MCacheConfig = UR_KERNEL_CACHE_CONFIG_LARGE_DATA;
        }
      } else {
        std::ignore = Props;
      }
    }

    // Process Kernel cooperative property.
    {
      if constexpr (PropertiesT::template has_property<use_root_sync_key>())
        retval.MIsCooperative = true;
    }

    // Process device progress properties.
    {
      using forward_progress =
          sycl::ext::oneapi::experimental::forward_progress_guarantee;
      if constexpr (PropertiesT::template has_property<
                        work_group_progress_key>()) {
        auto prop = Props.template get_property<work_group_progress_key>();
        retval.MForwardProgressProperties[0].Guarantee = prop.guarantee;
        retval.MForwardProgressProperties[0].ExecScope =
            execution_scope::work_group;
        retval.MForwardProgressProperties[0].CoordinationScope =
            prop.coordinationScope;

        // If we are here, the device supports the guarantee required but there
        // is a caveat in that if the guarantee required is a concurrent
        // guarantee, then we most likely also need to enable cooperative launch
        // of the kernel. That is, although the device supports the required
        // guarantee, some setup work is needed to truly make the device provide
        // that guarantee at runtime. Otherwise, we will get the default
        // guarantee which is weaker than concurrent. Same reasoning applies for
        // sub_group but not for work_item.
        // TODO: Further design work is probably needed to reflect this behavior
        // in Unified Runtime.
        if constexpr (prop.guarantee == forward_progress::concurrent)
          retval.MIsCooperative = true;
      }
      if constexpr (PropertiesT::template has_property<
                        sub_group_progress_key>()) {
        auto prop = Props.template get_property<sub_group_progress_key>();
        retval.MForwardProgressProperties[1].Guarantee = prop.guarantee;
        retval.MForwardProgressProperties[1].ExecScope =
            execution_scope::sub_group;
        retval.MForwardProgressProperties[1].CoordinationScope =
            prop.coordinationScope;

        // Same reasoning as above for work_group applies here.
        if constexpr (prop.guarantee == forward_progress::concurrent)
          retval.MIsCooperative = true;
      }
      if constexpr (PropertiesT::template has_property<
                        work_item_progress_key>()) {
        auto prop = Props.template get_property<work_item_progress_key>();
        retval.MForwardProgressProperties[2].Guarantee = prop.guarantee;
        retval.MForwardProgressProperties[2].ExecScope =
            execution_scope::work_item;
        retval.MForwardProgressProperties[2].CoordinationScope =
            prop.coordinationScope;
      }
    }

    // Process work group scratch memory property.
    {
      if constexpr (PropertiesT::template has_property<
                        work_group_scratch_size>()) {
        auto WorkGroupMemSize =
            Props.template get_property<work_group_scratch_size>();
        retval.MWorkGroupMemorySize = WorkGroupMemSize.size;
      }
    }

    // Parse cluster properties.
    {
      constexpr std::size_t ClusterDim = getClusterDim<PropertiesT>();
      if constexpr (ClusterDim > 0) {

        auto ClusterSize =
            Props.template get_property<cuda::cluster_size_key<ClusterDim>>()
                .get_cluster_size();
        retval.MUsesClusterLaunch = true;
        retval.MClusterDims = ClusterDim;
        if (ClusterDim == 1) {
          retval.MClusterSize[0] = ClusterSize[0];
        } else if (ClusterDim == 2) {
          retval.MClusterSize[0] = ClusterSize[0];
          retval.MClusterSize[1] = ClusterSize[1];
        } else if (ClusterDim == 3) {
          retval.MClusterSize[0] = ClusterSize[0];
          retval.MClusterSize[1] = ClusterSize[1];
          retval.MClusterSize[2] = ClusterSize[2];
        } else {
          assert(ClusterDim <= 3 &&
                 "Only 1D, 2D, and 3D cluster launch is supported.");
        }
      }
    }

    return retval;
  }

  /// Process kernel properties.
  /// Note: it is important that this function *does not* depend on kernel
  /// name or kernel type, because then it will be instantiated for every
  /// kernel, even though body of those instantiated functions could be almost
  /// the same, thus unnecessary increasing compilation time.
  template <
      bool IsESIMDKernel,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  static KernelLaunchPropertiesT processKernelProperties(PropertiesT Props) {
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

    return processKernelLaunchProperties(Props);
  }

  // Returns KernelLaunchPropertiesT or std::nullopt based on whether the
  // kernel functor has a get method that returns properties.
  template <typename KernelName, bool isESIMD, typename KernelType>
  static std::optional<KernelLaunchPropertiesT>
  parseProperties([[maybe_unused]] const KernelType &KernelFunc) {
#ifndef __SYCL_DEVICE_ONLY__
    // If there are properties provided by get method then process them.
    if constexpr (ext::oneapi::experimental::detail::
                      HasKernelPropertiesGetMethod<const KernelType &>::value) {

      return processKernelProperties<isESIMD>(
          KernelFunc.get(ext::oneapi::experimental::properties_tag{}));
    }
#endif
    // If there are no properties provided by get method then return empty
    // optional.
    return std::nullopt;
  }
}; // KernelLaunchPropertyWrapper struct

} // namespace detail
} // namespace _V1
} // namespace sycl
