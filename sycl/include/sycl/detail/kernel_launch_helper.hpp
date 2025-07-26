//==-------- kernel_launch_helper.hpp --- SYCL kernel launch utilities ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/helpers.hpp>
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

// TODO: Remove this.
namespace syclex = sycl::ext::oneapi::experimental;

// This class is inherited by handler_impl and sycl::handler is using the static
// methods.
class KernelLaunchPropertyWrapper {
public:
  // Changing values in this will break ABI/API.
  enum class StableKernelCacheConfig : int32_t {
    Default = 0,
    LargeSLM = 1,
    LargeData = 2
  };

  ur_kernel_cache_config_t MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;
  bool MKernelIsCooperative = false;
  uint32_t MKernelWorkGroupMemorySize = 0;

  // Kernel Cluster launch
  bool MKernelUsesClusterLaunch = false;
  size_t MKernelClusterDims = 0;
  std::array<size_t, 3> MKernelClusterSize = {0, 0, 0};

  struct KernelLaunchPropertiesT {
    std::optional<ur_kernel_cache_config_t> MKernelCacheConfig;
    std::optional<bool> MKernelIsCooperative;
    std::optional<uint32_t> MKernelWorkGroupMemorySize;
    std::optional<bool> MKernelUsesClusterLaunch;
    size_t MKernelClusterDims;
    std::array<size_t, 3> MKernelClusterSize;

    KernelLaunchPropertiesT() {
      MKernelCacheConfig = std::nullopt;
      MKernelIsCooperative = std::nullopt;
      MKernelWorkGroupMemorySize = std::nullopt;
      MKernelUsesClusterLaunch = std::nullopt;
      MKernelClusterDims = 0;
      MKernelClusterSize.fill(0);
    }
  };

  template <syclex::detail::UnsupportedGraphFeatures FeatureT>
  static void throwIfGraphAssociated(bool HasGraph) {
    if (!HasGraph)
      return;

    std::string FeatureString =
        syclex::detail::UnsupportedFeatureToString(FeatureT);
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "The " + FeatureString +
                              " feature is not yet available "
                              "for use with the SYCL Graph extension.");
  }

private:
  static void verifyDeviceHasProgressGuarantee(
      bool supported, syclex::forward_progress_guarantee guarantee,
      syclex::execution_scope threadScope, KernelLaunchPropertiesT &prop) {

    using execution_scope = syclex::execution_scope;
    using forward_progress = syclex::forward_progress_guarantee;

    if (threadScope == execution_scope::work_group) {
      if (!supported) {
        throw sycl::exception(
            sycl::errc::feature_not_supported,
            "Required progress guarantee for work groups is not "
            "supported by this device.");
      }
      // If we are here, the device supports the guarantee required but there is
      // a caveat in that if the guarantee required is a concurrent guarantee,
      // then we most likely also need to enable cooperative launch of the
      // kernel. That is, although the device supports the required guarantee,
      // some setup work is needed to truly make the device provide that
      // guarantee at runtime. Otherwise, we will get the default guarantee
      // which is weaker than concurrent. Same reasoning applies for sub_group
      // but not for work_item.
      // TODO: Further design work is probably needed to reflect this behavior
      // in Unified Runtime.
      if (guarantee == forward_progress::concurrent)
        prop.MKernelIsCooperative = true;
    } else if (threadScope == execution_scope::sub_group) {
      if (!supported) {
        throw sycl::exception(
            sycl::errc::feature_not_supported,
            "Required progress guarantee for sub groups is not "
            "supported by this device.");
      }
      // Same reasoning as above.
      if (guarantee == forward_progress::concurrent)
        prop.MKernelIsCooperative = true;
    } else { // threadScope is execution_scope::work_item otherwise undefined
             // behavior
      if (!supported) {
        throw sycl::exception(
            sycl::errc::feature_not_supported,
            "Required progress guarantee for work items is not "
            "supported by this device.");
      }
    }
  }

public:
  /// Process runtime kernel properties.
  ///
  /// Stores information about kernel properties into the handler.
  template <typename PropertiesT, typename PropertyProcessorPtrT>
  static KernelLaunchPropertiesT
  processLaunchProperties(PropertiesT Props, PropertyProcessorPtrT p) {

    KernelLaunchPropertiesT retval;
    bool HasGraph = p->isKernelAssociatedWithGraph();

    // Process Kernel cache configuration property.
    {
      auto ConvertToURCacheConfig = [](StableKernelCacheConfig cc) {
        switch (cc) {
        case StableKernelCacheConfig::Default:
          return UR_KERNEL_CACHE_CONFIG_DEFAULT;
        case StableKernelCacheConfig::LargeSLM:
          return UR_KERNEL_CACHE_CONFIG_LARGE_SLM;
        case StableKernelCacheConfig::LargeData:
          return UR_KERNEL_CACHE_CONFIG_LARGE_DATA;
        }
      };

      if constexpr (PropertiesT::template has_property<
                        sycl::ext::intel::experimental::cache_config_key>()) {
        auto Config = Props.template get_property<
            sycl::ext::intel::experimental::cache_config_key>();
        if (Config == sycl::ext::intel::experimental::large_slm) {
          retval.MKernelCacheConfig =
              ConvertToURCacheConfig(StableKernelCacheConfig::LargeSLM);
        } else if (Config == sycl::ext::intel::experimental::large_data) {
          retval.MKernelCacheConfig =
              ConvertToURCacheConfig(StableKernelCacheConfig::LargeData);
        }
      } else {
        std::ignore = Props;
      }
    }

    // Process Kernel cooperative property.
    {
      constexpr bool UsesRootSync = PropertiesT::template has_property<
          sycl::ext::oneapi::experimental::use_root_sync_key>();
      if (UsesRootSync)
        retval.MKernelIsCooperative = UsesRootSync;
    }

    // Process device progress properties.
    {
      if constexpr (PropertiesT::template has_property<
                        sycl::ext::oneapi::experimental::
                            work_group_progress_key>()) {
        auto prop = Props.template get_property<
            sycl::ext::oneapi::experimental::work_group_progress_key>();
        bool supported = p->deviceSupportForwardProgress(
            prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::work_group,
            prop.coordinationScope);
        verifyDeviceHasProgressGuarantee(
            supported, prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::work_group,
            retval);
      }
      if constexpr (PropertiesT::template has_property<
                        sycl::ext::oneapi::experimental::
                            sub_group_progress_key>()) {
        auto prop = Props.template get_property<
            sycl::ext::oneapi::experimental::sub_group_progress_key>();
        bool supported = p->deviceSupportForwardProgress(
            prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::sub_group,
            prop.coordinationScope);
        verifyDeviceHasProgressGuarantee(
            supported, prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::sub_group,
            retval);
      }
      if constexpr (PropertiesT::template has_property<
                        sycl::ext::oneapi::experimental::
                            work_item_progress_key>()) {
        auto prop = Props.template get_property<
            sycl::ext::oneapi::experimental::work_item_progress_key>();
        bool supported = p->deviceSupportForwardProgress(
            prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::work_item,
            prop.coordinationScope);
        verifyDeviceHasProgressGuarantee(
            supported, prop.guarantee,
            sycl::ext::oneapi::experimental::execution_scope::work_item,
            retval);
      }
    }

    // Process work group scratch memory property.
    {
      if constexpr (PropertiesT::template has_property<
                        sycl::ext::oneapi::experimental::
                            work_group_scratch_size>()) {
        throwIfGraphAssociated<syclex::detail::UnsupportedGraphFeatures::
                                   sycl_ext_oneapi_work_group_scratch_memory>(
            HasGraph);
        auto WorkGroupMemSize = Props.template get_property<
            sycl::ext::oneapi::experimental::work_group_scratch_size>();
        retval.MKernelWorkGroupMemorySize = WorkGroupMemSize.size;
      }
    }

    // Parse cluster properties.
    {
      constexpr std::size_t ClusterDim =
          syclex::detail::getClusterDim<PropertiesT>();
      if constexpr (ClusterDim > 0) {
        throwIfGraphAssociated<
            syclex::detail::UnsupportedGraphFeatures::
                sycl_ext_oneapi_experimental_cuda_cluster_launch>(HasGraph);
        auto ClusterSize = Props
                               .template get_property<
                                   syclex::cuda::cluster_size_key<ClusterDim>>()
                               .get_cluster_size();
        retval.MKernelUsesClusterLaunch = true;
        retval.MKernelClusterDims = ClusterDim;
        if (ClusterDim == 1) {
          retval.MKernelClusterSize[0] = ClusterSize[0];
        } else if (ClusterDim == 2) {
          retval.MKernelClusterSize[0] = ClusterSize[0];
          retval.MKernelClusterSize[1] = ClusterSize[1];
        } else if (ClusterDim == 3) {
          retval.MKernelClusterSize[0] = ClusterSize[0];
          retval.MKernelClusterSize[1] = ClusterSize[1];
          retval.MKernelClusterSize[2] = ClusterSize[2];
        } else {
          assert(ClusterDim > 3 &&
                 "Only 1D, 2D, and 3D cluster launch is supported.");
        }
      }
    }

    return retval;
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
      bool IsESIMDKernel, typename PropertyProcessorPtrT,
      typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
  static KernelLaunchPropertiesT processProperties(PropertiesT Props,
                                                   PropertyProcessorPtrT p) {
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

    return processLaunchProperties(Props, p);
  }

  template <typename KernelName, typename KernelType,
            typename PropertyProcessorPtrT>
  static std::optional<KernelLaunchPropertiesT>
  parseProperties([[maybe_unused]] const KernelType &KernelFunc,
                  [[maybe_unused]] PropertyProcessorPtrT p) {
#ifndef __SYCL_DEVICE_ONLY__
    // If there are properties provided by get method then process them.
    if constexpr (ext::oneapi::experimental::detail::
                      HasKernelPropertiesGetMethod<const KernelType &>::value) {

      return processProperties<detail::isKernelESIMD<KernelName>()>(
          KernelFunc.get(ext::oneapi::experimental::properties_tag{}), p);
    }
#endif
    // If there are no properties provided by get method then return empty
    // optional.
    return std::nullopt;
  }

  void parseAndSetKernelLaunchProperties(KernelLaunchPropertiesT &props) {
    if (props.MKernelCacheConfig)
      MKernelCacheConfig = *props.MKernelCacheConfig;

    if (props.MKernelIsCooperative)
      MKernelIsCooperative = *props.MKernelIsCooperative;

    if (props.MKernelWorkGroupMemorySize)
      MKernelWorkGroupMemorySize = *props.MKernelWorkGroupMemorySize;

    if (props.MKernelUsesClusterLaunch) {
      MKernelUsesClusterLaunch = *props.MKernelUsesClusterLaunch;
      MKernelClusterDims = props.MKernelClusterDims;
      MKernelClusterSize = props.MKernelClusterSize;
    }
  }
}; // KernelLaunchPropertyWrapper struct

} // namespace detail
} // namespace _V1
} // namespace sycl
