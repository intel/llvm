//==-------- device_info.hpp - SYCL device info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_util.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/common_info.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/platform.hpp>

#include <chrono>
#include <thread>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

inline std::vector<info::fp_config> read_fp_bitfield(pi_device_fp_config bits) {
  std::vector<info::fp_config> result;
  if (bits & PI_FP_DENORM)
    result.push_back(info::fp_config::denorm);
  if (bits & PI_FP_INF_NAN)
    result.push_back(info::fp_config::inf_nan);
  if (bits & PI_FP_ROUND_TO_NEAREST)
    result.push_back(info::fp_config::round_to_nearest);
  if (bits & PI_FP_ROUND_TO_ZERO)
    result.push_back(info::fp_config::round_to_zero);
  if (bits & PI_FP_ROUND_TO_INF)
    result.push_back(info::fp_config::round_to_inf);
  if (bits & PI_FP_FMA)
    result.push_back(info::fp_config::fma);
  if (bits & PI_FP_SOFT_FLOAT)
    result.push_back(info::fp_config::soft_float);
  if (bits & PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    result.push_back(info::fp_config::correctly_rounded_divide_sqrt);
  return result;
}

inline std::vector<info::partition_affinity_domain>
read_domain_bitfield(pi_device_affinity_domain bits) {
  std::vector<info::partition_affinity_domain> result;
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_NUMA)
    result.push_back(info::partition_affinity_domain::numa);
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
    result.push_back(info::partition_affinity_domain::L4_cache);
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
    result.push_back(info::partition_affinity_domain::L3_cache);
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
    result.push_back(info::partition_affinity_domain::L2_cache);
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
    result.push_back(info::partition_affinity_domain::L1_cache);
  if (bits & PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
    result.push_back(info::partition_affinity_domain::next_partitionable);
  return result;
}

inline std::vector<info::execution_capability>
read_execution_bitfield(pi_device_exec_capabilities bits) {
  std::vector<info::execution_capability> result;
  if (bits & PI_EXEC_KERNEL)
    result.push_back(info::execution_capability::exec_kernel);
  if (bits & PI_EXEC_NATIVE_KERNEL)
    result.push_back(info::execution_capability::exec_native_kernel);
  return result;
}

inline std::string
affinityDomainToString(info::partition_affinity_domain AffinityDomain) {
  switch (AffinityDomain) {
#define __SYCL_AFFINITY_DOMAIN_STRING_CASE(DOMAIN)                             \
  case DOMAIN:                                                                 \
    return #DOMAIN;

    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::numa)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L4_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L3_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L2_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L1_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::next_partitionable)
#undef __SYCL_AFFINITY_DOMAIN_STRING_CASE
  default:
    assert(false && "Missing case for affinity domain.");
    return "unknown";
  }
}

// Mapping expected SYCL return types to those returned by PI calls
template <typename T> struct sycl_to_pi {
  using type = T;
};
template <> struct sycl_to_pi<bool> {
  using type = pi_bool;
};
template <> struct sycl_to_pi<device> {
  using type = RT::PiDevice;
};
template <> struct sycl_to_pi<platform> {
  using type = RT::PiPlatform;
};

// Mapping fp_config device info types to the values used to check fp support
template <typename Param> struct check_fp_support {};

template <> struct check_fp_support<info::device::half_fp_config> {
  using type = info::device::native_vector_width_half;
};

template <> struct check_fp_support<info::device::double_fp_config> {
  using type = info::device::native_vector_width_double;
};

// Structs for emulating function template partial specialization
// Default template for the general case
// TODO: get rid of remaining uses of OpenCL directly
//
template <typename ReturnT, typename Param> struct get_device_info_impl {
  static ReturnT get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<ReturnT>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, PiInfoCode<Param>::value,
                                            sizeof(result), &result, nullptr);
    return ReturnT(result);
  }
};

// Specialization for platform
template <typename Param> struct get_device_info_impl<platform, Param> {
  static platform get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<platform>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, PiInfoCode<Param>::value,
                                            sizeof(result), &result, nullptr);
    // TODO: Change PiDevice to device_impl.
    // Use the Plugin from the device_impl class after plugin details
    // are added to the class.
    return createSyclObjFromImpl<platform>(
        platform_impl::getOrMakePlatformImpl(result, Plugin));
  }
};

// Helper function to allow using the specialization of get_device_info_impl
// for string return type in other specializations.
inline std::string get_device_info_string(RT::PiDevice dev,
                                          RT::PiDeviceInfo InfoCode,
                                          const plugin &Plugin) {
  size_t resultSize = 0;
  Plugin.call<PiApiKind::piDeviceGetInfo>(dev, InfoCode, 0, nullptr,
                                          &resultSize);
  if (resultSize == 0) {
    return std::string();
  }
  std::unique_ptr<char[]> result(new char[resultSize]);
  Plugin.call<PiApiKind::piDeviceGetInfo>(dev, InfoCode, resultSize,
                                          result.get(), nullptr);

  return std::string(result.get());
}

// Specialization for string return type, variable return size
template <typename Param> struct get_device_info_impl<std::string, Param> {
  static std::string get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_string(dev, PiInfoCode<Param>::value, Plugin);
  }
};

// Specialization for parent device
template <typename ReturnT>
struct get_device_info_impl<ReturnT, info::device::parent_device> {
  static ReturnT get(RT::PiDevice dev, const plugin &Plugin);
};

// Specialization for fp_config types, checks the corresponding fp type support
template <typename Param>
struct get_device_info_impl<std::vector<info::fp_config>, Param> {
  static std::vector<info::fp_config> get(RT::PiDevice dev,
                                          const plugin &Plugin) {
    // Check if fp type is supported
    if (!get_device_info_impl<
            typename check_fp_support<Param>::type::return_type,
            typename check_fp_support<Param>::type>::get(dev, Plugin)) {
      return {};
    }
    cl_device_fp_config result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, PiInfoCode<Param>::value,
                                            sizeof(result), &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for OpenCL version, splits the string returned by OpenCL
template <> struct get_device_info_impl<std::string, info::device::version> {
  static std::string get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result = get_device_info_string(
        dev, PiInfoCode<info::device::version>::value, Plugin);

    // Extract OpenCL version from the returned string.
    // For example, for the string "OpenCL 2.1 (Build 0)"
    // return '2.1'.
    auto dotPos = result.find('.');
    if (dotPos == std::string::npos)
      return result;

    auto leftPos = result.rfind(' ', dotPos);
    if (leftPos == std::string::npos)
      leftPos = 0;
    else
      leftPos++;

    auto rightPos = result.find(' ', dotPos);
    return result.substr(leftPos, rightPos - leftPos);
  }
};

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info_impl<std::vector<info::fp_config>,
                            info::device::single_fp_config> {
  static std::vector<info::fp_config> get(RT::PiDevice dev,
                                          const plugin &Plugin) {
    pi_device_fp_config result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::single_fp_config>::value, sizeof(result),
        &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling, OpenCL returns a bitfield
template <> struct get_device_info_impl<bool, info::device::queue_profiling> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    cl_command_queue_properties result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::queue_profiling>::value, sizeof(result),
        &result, nullptr);
    return (result & CL_QUEUE_PROFILING_ENABLE);
  }
};

// Specialization for atomic_memory_order_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_order>,
                            info::device::atomic_memory_order_capabilities> {
  static std::vector<memory_order> get(RT::PiDevice dev, const plugin &Plugin) {
    pi_memory_order_capabilities result;
    Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::atomic_memory_order_capabilities>::value,
        sizeof(pi_memory_order_capabilities), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_memory_scope_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_scope>,
                            info::device::atomic_memory_scope_capabilities> {
  static std::vector<memory_scope> get(RT::PiDevice dev, const plugin &Plugin) {
    pi_memory_scope_capabilities result;
    Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::atomic_memory_scope_capabilities>::value,
        sizeof(pi_memory_scope_capabilities), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for bf16 math functions
template <>
struct get_device_info_impl<bool,
                            info::device::ext_oneapi_bfloat16_math_functions> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    bool result = false;

    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev,
        PiInfoCode<info::device::ext_oneapi_bfloat16_math_functions>::value,
        sizeof(result), &result, nullptr);
    if (Err != PI_SUCCESS) {
      return false;
    }
    return result;
  }
};

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info_impl<std::vector<info::execution_capability>,
                            info::device::execution_capabilities> {
  static std::vector<info::execution_capability> get(RT::PiDevice dev,
                                                     const plugin &Plugin) {
    pi_device_exec_capabilities result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::execution_capabilities>::value,
        sizeof(result), &result, nullptr);
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernel identifiers
template <>
struct get_device_info_impl<std::vector<kernel_id>,
                            info::device::built_in_kernel_ids> {
  static std::vector<kernel_id> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result = get_device_info_string(
        dev, PiInfoCode<info::device::built_in_kernels>::value, Plugin);
    auto names = split_string(result, ';');

    std::vector<kernel_id> ids;
    ids.reserve(names.size());
    for (const auto &name : names) {
      ids.push_back(ProgramManager::getInstance().getBuiltInKernelID(name));
    }
    return ids;
  }
};

// Specialization for built in kernels, splits the string returned by OpenCL
template <>
struct get_device_info_impl<std::vector<std::string>,
                            info::device::built_in_kernels> {
  static std::vector<std::string> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result = get_device_info_string(
        dev, PiInfoCode<info::device::built_in_kernels>::value, Plugin);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info_impl<std::vector<std::string>,
                            info::device::extensions> {
  static std::vector<std::string> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result =
        get_device_info_impl<std::string, info::device::extensions>::get(
            dev, Plugin);
    return split_string(result, ' ');
  }
};

static bool is_sycl_partition_property(info::partition_property PP) {
  switch (PP) {
  case info::partition_property::no_partition:
  case info::partition_property::partition_equally:
  case info::partition_property::partition_by_counts:
  case info::partition_property::partition_by_affinity_domain:
  case info::partition_property::ext_intel_partition_by_cslice:
    return true;
  }
  return false;
}

// Specialization for partition properties, variable OpenCL return size
template <>
struct get_device_info_impl<std::vector<info::partition_property>,
                            info::device::partition_properties> {
  static std::vector<info::partition_property> get(RT::PiDevice dev,
                                                   const plugin &Plugin) {
    auto info_partition = PiInfoCode<info::device::partition_properties>::value;

    size_t resultSize;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, info_partition, 0, nullptr,
                                            &resultSize);

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);
    if (arrayLength == 0) {
      return {};
    }
    std::unique_ptr<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, info_partition, resultSize,
                                            arrayResult.get(), nullptr);

    std::vector<info::partition_property> result;
    for (size_t i = 0; i < arrayLength; ++i) {
      // OpenCL extensions may have partition_properties that
      // are not yet defined for SYCL (eg. CL_DEVICE_PARTITION_BY_NAMES_INTEL)
      info::partition_property pp(
          static_cast<info::partition_property>(arrayResult[i]));
      if (is_sycl_partition_property(pp))
        result.push_back(pp);
    }
    return result;
  }
};

// Specialization for partition affinity domains, OpenCL returns a bitfield
template <>
struct get_device_info_impl<std::vector<info::partition_affinity_domain>,
                            info::device::partition_affinity_domains> {
  static std::vector<info::partition_affinity_domain>
  get(RT::PiDevice dev, const plugin &Plugin) {
    pi_device_affinity_domain result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::partition_affinity_domains>::value,
        sizeof(result), &result, nullptr);
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, OpenCL can return other
// partition properties instead
template <>
struct get_device_info_impl<info::partition_affinity_domain,
                            info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain get(RT::PiDevice dev,
                                             const plugin &Plugin) {
    size_t resultSize;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::partition_type_affinity_domain>::value, 0,
        nullptr, &resultSize);
    if (resultSize != 1) {
      return info::partition_affinity_domain::not_applicable;
    }
    cl_device_partition_property result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::partition_type_affinity_domain>::value,
        sizeof(result), &result, nullptr);
    if (result == PI_DEVICE_AFFINITY_DOMAIN_NUMA ||
        result == PI_DEVICE_AFFINITY_DOMAIN_L4_CACHE ||
        result == PI_DEVICE_AFFINITY_DOMAIN_L3_CACHE ||
        result == PI_DEVICE_AFFINITY_DOMAIN_L2_CACHE ||
        result == PI_DEVICE_AFFINITY_DOMAIN_L1_CACHE) {
      return info::partition_affinity_domain(result);
    }

    return info::partition_affinity_domain::not_applicable;
  }
};

// Specialization for partition type
template <>
struct get_device_info_impl<info::partition_property,
                            info::device::partition_type_property> {
  static info::partition_property get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, PI_DEVICE_INFO_PARTITION_TYPE,
                                            0, nullptr, &resultSize);
    if (!resultSize)
      return info::partition_property::no_partition;

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);

    std::unique_ptr<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, PI_DEVICE_INFO_PARTITION_TYPE,
                                            resultSize, arrayResult.get(),
                                            nullptr);
    if (!arrayResult[0])
      return info::partition_property::no_partition;
    return info::partition_property(arrayResult[0]);
  }
};
// Specialization for supported subgroup sizes
template <>
struct get_device_info_impl<std::vector<size_t>,
                            info::device::sub_group_sizes> {
  static std::vector<size_t> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::sub_group_sizes>::value, 0, nullptr,
        &resultSize);

    std::vector<size_t> result(resultSize / sizeof(size_t));
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::sub_group_sizes>::value, resultSize,
        result.data(), nullptr);
    return result;
  }
};

// Specialization for kernel to kernel pipes.
// Here we step away from OpenCL, since there is no appropriate cl_device_info
// enum for global pipes feature.
template <>
struct get_device_info_impl<bool, info::device::kernel_kernel_pipe_support> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    // We claim, that all Intel FPGA devices support kernel to kernel pipe
    // feature (at least at the scope of SYCL_INTEL_data_flow_pipes extension).
    platform plt = get_device_info_impl<platform, info::device::platform>::get(
        dev, Plugin);
    std::string platform_name = plt.get_info<info::platform::name>();
    if (platform_name == "Intel(R) FPGA Emulation Platform for OpenCL(TM)" ||
        platform_name == "Intel(R) FPGA SDK for OpenCL(TM)")
      return true;

    // TODO: a better way is to query for supported SPIR-V capabilities when
    // it's started to be possible. Also, if a device's backend supports
    // SPIR-V 1.1 (where Pipe Storage feature was defined), than it supports
    // the feature as well.
    return false;
  }
};

template <int Dimensions> id<Dimensions> construct_id(size_t *values) = delete;
// Due to the flipping of work group dimensions before kernel launch, the values
// should also be reversed.
template <> inline id<1> construct_id<1>(size_t *values) { return {values[0]}; }
template <> inline id<2> construct_id<2>(size_t *values) {
  return {values[1], values[0]};
}
template <> inline id<3> construct_id<3>(size_t *values) {
  return {values[2], values[1], values[0]};
}

// Specialization for max_work_item_sizes.
template <int Dimensions>
struct get_device_info_impl<id<Dimensions>,
                            info::device::max_work_item_sizes<Dimensions>> {
  static id<Dimensions> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::max_work_item_sizes<Dimensions>>::value,
        sizeof(result), &result, nullptr);
    return construct_id<Dimensions>(result);
  }
};

template <>
struct get_device_info_impl<
    size_t, ext::oneapi::experimental::info::device::max_global_work_groups> {
  static size_t get(RT::PiDevice dev, const plugin &Plugin) {
    (void)dev; // Silence unused warning
    (void)Plugin;
    return static_cast<size_t>((std::numeric_limits<int>::max)());
  }
};
template <>
struct get_device_info_impl<
    id<1>, ext::oneapi::experimental::info::device::max_work_groups<1>> {
  static id<1> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        PiInfoCode<
            ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
        sizeof(result), &result, nullptr);
    return id<1>(std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info_impl<
    id<2>, ext::oneapi::experimental::info::device::max_work_groups<2>> {
  static id<2> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        PiInfoCode<
            ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
        sizeof(result), &result, nullptr);
    return id<2>(std::min(Limit, result[1]), std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info_impl<
    id<3>, ext::oneapi::experimental::info::device::max_work_groups<3>> {
  static id<3> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        PiInfoCode<
            ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
        sizeof(result), &result, nullptr);
    return id<3>(std::min(Limit, result[2]), std::min(Limit, result[1]),
                 std::min(Limit, result[0]));
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_global_work_groups>
template <>
struct get_device_info_impl<size_t,
                            info::device::ext_oneapi_max_global_work_groups> {
  static size_t get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_impl<size_t,
                                ext::oneapi::experimental::info::device::
                                    max_global_work_groups>::get(dev, Plugin);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_1d>
template <>
struct get_device_info_impl<id<1>,
                            info::device::ext_oneapi_max_work_groups_1d> {
  static id<1> get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_impl<id<1>, ext::oneapi::experimental::info::device::
                                           max_work_groups<1>>::get(dev,
                                                                    Plugin);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_2d>
template <>
struct get_device_info_impl<id<2>,
                            info::device::ext_oneapi_max_work_groups_2d> {
  static id<2> get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_impl<id<2>, ext::oneapi::experimental::info::device::
                                           max_work_groups<2>>::get(dev,
                                                                    Plugin);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_3d>
template <>
struct get_device_info_impl<id<3>,
                            info::device::ext_oneapi_max_work_groups_3d> {
  static id<3> get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_impl<id<3>, ext::oneapi::experimental::info::device::
                                           max_work_groups<3>>::get(dev,
                                                                    Plugin);
  }
};

// Specialization for parent device
template <> struct get_device_info_impl<device, info::device::parent_device> {
  static device get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<device>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::parent_device>::value, sizeof(result),
        &result, nullptr);
    if (result == nullptr)
      throw invalid_object_error(
          "No parent for device because it is not a subdevice",
          PI_ERROR_INVALID_DEVICE);

    // Get the platform of this device
    std::shared_ptr<detail::platform_impl> Platform =
        platform_impl::getPlatformFromPiDevice(dev, Plugin);
    return createSyclObjFromImpl<device>(
        Platform->getOrMakeDeviceImpl(result, Platform));
  }
};

// USM

// Specialization for device usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_device_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::usm_device_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for host usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_host_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::usm_host_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for shared usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_shared_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::usm_shared_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for restricted usm query
template <>
struct get_device_info_impl<bool,
                            info::device::usm_restricted_shared_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::usm_restricted_shared_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);
    // Check that we don't support any cross device sharing
    return (Err != PI_SUCCESS)
               ? false
               : !(caps & (PI_USM_ACCESS | PI_USM_CONCURRENT_ACCESS));
  }
};

// Specialization for system usm query
template <>
struct get_device_info_impl<bool, info::device::usm_system_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::usm_system_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for memory channel query
template <>
struct get_device_info_impl<bool, info::device::ext_intel_mem_channel> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_mem_properties caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, PiInfoCode<info::device::ext_intel_mem_channel>::value,
        sizeof(pi_mem_properties), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_MEM_PROPERTIES_CHANNEL);
  }
};

template <typename Param>
typename Param::return_type get_device_info(RT::PiDevice dev,
                                            const plugin &Plugin) {
  static_assert(is_device_info_desc<Param>::value,
                "Invalid device information descriptor");
  return get_device_info_impl<typename Param::return_type, Param>::get(dev,
                                                                       Plugin);
}

// SYCL host device information

// Default template is disabled, all possible instantiations are
// specified explicitly.
template <typename Param>
inline typename Param::return_type get_device_info_host() = delete;

template <>
inline info::device_type get_device_info_host<info::device::device_type>() {
  return info::device_type::host;
}

template <> inline uint32_t get_device_info_host<info::device::vendor_id>() {
  return 0x8086;
}

template <>
inline uint32_t get_device_info_host<info::device::max_compute_units>() {
  return std::thread::hardware_concurrency();
}

template <>
inline uint32_t get_device_info_host<info::device::max_work_item_dimensions>() {
  return 3;
}

template <>
inline id<1> get_device_info_host<info::device::max_work_item_sizes<1>>() {
  // current value is the required minimum
  return {1};
}

template <>
inline id<2> get_device_info_host<info::device::max_work_item_sizes<2>>() {
  // current value is the required minimum
  return {1, 1};
}

template <>
inline id<3> get_device_info_host<info::device::max_work_item_sizes<3>>() {
  // current value is the required minimum
  return {1, 1, 1};
}

template <>
inline constexpr size_t get_device_info_host<
    ext::oneapi::experimental::info::device::max_global_work_groups>() {
  // See handler.hpp for the maximum value :
  return static_cast<size_t>((std::numeric_limits<int>::max)());
}

template <>
inline id<1> get_device_info_host<
    ext::oneapi::experimental::info::device::max_work_groups<1>>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit = get_device_info_host<
      ext::oneapi::experimental::info::device::max_global_work_groups>();
  return {Limit};
}

template <>
inline id<2> get_device_info_host<
    ext::oneapi::experimental::info::device::max_work_groups<2>>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit = get_device_info_host<
      ext::oneapi::experimental::info::device::max_global_work_groups>();
  return {Limit, Limit};
}

template <>
inline id<3> get_device_info_host<
    ext::oneapi::experimental::info::device::max_work_groups<3>>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit = get_device_info_host<
      ext::oneapi::experimental::info::device::max_global_work_groups>();
  return {Limit, Limit, Limit};
}

// TODO:remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_global_work_groups>
template <>
inline constexpr size_t
get_device_info_host<info::device::ext_oneapi_max_global_work_groups>() {
  return get_device_info_host<
      ext::oneapi::experimental::info::device::max_global_work_groups>();
}

// TODO:remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_1d>
template <>
inline id<1>
get_device_info_host<info::device::ext_oneapi_max_work_groups_1d>() {

  return get_device_info_host<
      ext::oneapi::experimental::info::device::max_work_groups<1>>();
}

// TODO:remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_2d>
template <>
inline id<2>
get_device_info_host<info::device::ext_oneapi_max_work_groups_2d>() {
  return get_device_info_host<
      ext::oneapi::experimental::info::device::max_work_groups<2>>();
}

// TODO:remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_3d>
template <>
inline id<3>
get_device_info_host<info::device::ext_oneapi_max_work_groups_3d>() {
  return get_device_info_host<
      ext::oneapi::experimental::info::device::max_work_groups<3>>();
}

template <>
inline size_t get_device_info_host<info::device::max_work_group_size>() {
  // current value is the required minimum
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_char>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_short>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_int>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_long>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_float>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_double>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline uint32_t
get_device_info_host<info::device::preferred_vector_width_half>() {
  // TODO update when appropriate
  return 0;
}

template <>
inline uint32_t get_device_info_host<info::device::native_vector_width_char>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Char);
}

template <>
inline uint32_t
get_device_info_host<info::device::native_vector_width_short>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Short);
}

template <>
inline uint32_t get_device_info_host<info::device::native_vector_width_int>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Int);
}

template <>
inline uint32_t get_device_info_host<info::device::native_vector_width_long>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Long);
}

template <>
inline uint32_t
get_device_info_host<info::device::native_vector_width_float>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Float);
}

template <>
inline uint32_t
get_device_info_host<info::device::native_vector_width_double>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Double);
}

template <>
inline uint32_t get_device_info_host<info::device::native_vector_width_half>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Half);
}

template <>
inline uint32_t get_device_info_host<info::device::max_clock_frequency>() {
  return PlatformUtil::getMaxClockFrequency();
}

template <> inline uint32_t get_device_info_host<info::device::address_bits>() {
  return sizeof(void *) * 8;
}

template <>
inline uint64_t get_device_info_host<info::device::global_mem_size>() {
  return static_cast<uint64_t>(OSUtil::getOSMemSize());
}

template <>
inline uint64_t get_device_info_host<info::device::max_mem_alloc_size>() {
  // current value is the required minimum
  const uint64_t a = get_device_info_host<info::device::global_mem_size>() / 4;
  const uint64_t b = 128ul * 1024 * 1024;
  return (a > b) ? a : b;
}

template <> inline bool get_device_info_host<info::device::image_support>() {
  return true;
}

template <> inline bool get_device_info_host<info::device::atomic64>() {
  return false;
}

template <>
inline std::vector<memory_order>
get_device_info_host<info::device::atomic_memory_order_capabilities>() {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline std::vector<memory_scope>
get_device_info_host<info::device::atomic_memory_scope_capabilities>() {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

template <>
inline bool
get_device_info_host<info::device::ext_oneapi_bfloat16_math_functions>() {
  return false;
}

template <>
inline uint32_t get_device_info_host<info::device::max_read_image_args>() {
  // current value is the required minimum
  return 128;
}

template <>
inline uint32_t get_device_info_host<info::device::max_write_image_args>() {
  // current value is the required minimum
  return 8;
}

template <>
inline size_t get_device_info_host<info::device::image2d_max_width>() {
  // SYCL guarantees at least 8192. Some devices already known to provide more
  // than that (i.e. it is 16384 for opencl:gpu), which may create issues during
  // image object allocation on host.
  // Using any fixed number (i.e. 16384) brings the risk of having similar
  // issues on newer devices in future. Thus it does not make sense limiting
  // the returned value on host. Practially speaking the returned value on host
  // depends only on memory required for the image, which also depends on
  // the image channel_type and the image height. Both are not known in this
  // query, thus it becomes user's responsibility to choose proper image
  // parameters depending on similar query to (non-host device) and amount
  // of available/allocatable memory.
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline size_t get_device_info_host<info::device::image2d_max_height>() {
  // SYCL guarantees at least 8192. Some devices already known to provide more
  // than that (i.e. it is 16384 for opencl:gpu), which may create issues during
  // image object allocation on host.
  // Using any fixed number (i.e. 16384) brings the risk of having similar
  // issues on newer devices in future. Thus it does not make sense limiting
  // the returned value on host. Practially speaking the returned value on host
  // depends only on memory required for the image, which also depends on
  // the image channel_type and the image width. Both are not known in this
  // query, thus it becomes user's responsibility to choose proper image
  // parameters depending on similar query to (non-host device) and amount
  // of available/allocatable memory.
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline size_t get_device_info_host<info::device::image3d_max_width>() {
  // SYCL guarantees at least 8192. Some devices already known to provide more
  // than that (i.e. it is 16384 for opencl:gpu), which may create issues during
  // image object allocation on host.
  // Using any fixed number (i.e. 16384) brings the risk of having similar
  // issues on newer devices in future. Thus it does not make sense limiting
  // the returned value on host. Practially speaking the returned value on host
  // depends only on memory required for the image, which also depends on
  // the image channel_type and the image height/depth. Both are not known
  // in this query, thus it becomes user's responsibility to choose proper image
  // parameters depending on similar query to (non-host device) and amount
  // of available/allocatable memory.
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline size_t get_device_info_host<info::device::image3d_max_height>() {
  // SYCL guarantees at least 8192. Some devices already known to provide more
  // than that (i.e. it is 16384 for opencl:gpu), which may create issues during
  // image object allocation on host.
  // Using any fixed number (i.e. 16384) brings the risk of having similar
  // issues on newer devices in future. Thus it does not make sense limiting
  // the returned value on host. Practially speaking the returned value on host
  // depends only on memory required for the image, which also depends on
  // the image channel_type and the image width/depth. Both are not known
  // in this query, thus it becomes user's responsibility to choose proper image
  // parameters depending on similar query to (non-host device) and amount
  // of available/allocatable memory.
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline size_t get_device_info_host<info::device::image3d_max_depth>() {
  // SYCL guarantees at least 8192. Some devices already known to provide more
  // than that (i.e. it is 16384 for opencl:gpu), which may create issues during
  // image object allocation on host.
  // Using any fixed number (i.e. 16384) brings the risk of having similar
  // issues on newer devices in future. Thus it does not make sense limiting
  // the returned value on host. Practially speaking the returned value on host
  // depends only on memory required for the image, which also depends on
  // the image channel_type and the image height/width, which are not known
  // in this query, thus it becomes user's responsibility to choose proper image
  // parameters depending on similar query to (non-host device) and amount
  // of available/allocatable memory.
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline size_t get_device_info_host<info::device::image_max_buffer_size>() {
  // Not supported in SYCL
  return 0;
}

template <>
inline size_t get_device_info_host<info::device::image_max_array_size>() {
  // current value is the required minimum
  return 2048;
}

template <> inline uint32_t get_device_info_host<info::device::max_samplers>() {
  // current value is the required minimum
  return 16;
}

template <>
inline size_t get_device_info_host<info::device::max_parameter_size>() {
  // current value is the required minimum
  return 1024;
}

template <>
inline uint32_t get_device_info_host<info::device::mem_base_addr_align>() {
  return 1024;
}

template <>
inline std::vector<info::fp_config>
get_device_info_host<info::device::half_fp_config>() {
  // current value is the required minimum
  return {};
}

template <>
inline std::vector<info::fp_config>
get_device_info_host<info::device::single_fp_config>() {
  // current value is the required minimum
  return {info::fp_config::round_to_nearest, info::fp_config::inf_nan};
}

template <>
inline std::vector<info::fp_config>
get_device_info_host<info::device::double_fp_config>() {
  // current value is the required minimum
  return {info::fp_config::fma,           info::fp_config::round_to_nearest,
          info::fp_config::round_to_zero, info::fp_config::round_to_inf,
          info::fp_config::inf_nan,       info::fp_config::denorm};
}

template <>
inline info::global_mem_cache_type
get_device_info_host<info::device::global_mem_cache_type>() {
  return info::global_mem_cache_type::read_write;
}

template <>
inline uint32_t
get_device_info_host<info::device::global_mem_cache_line_size>() {
  return PlatformUtil::getMemCacheLineSize();
}

template <>
inline uint64_t get_device_info_host<info::device::global_mem_cache_size>() {
  return PlatformUtil::getMemCacheSize();
}

template <>
inline uint64_t get_device_info_host<info::device::max_constant_buffer_size>() {
  // current value is the required minimum
  return 64 * 1024;
}

template <>
inline uint32_t get_device_info_host<info::device::max_constant_args>() {
  // current value is the required minimum
  return 8;
}

template <>
inline info::local_mem_type
get_device_info_host<info::device::local_mem_type>() {
  return info::local_mem_type::global;
}

template <>
inline uint64_t get_device_info_host<info::device::local_mem_size>() {
  // current value is the required minimum
  return 32 * 1024;
}

template <>
inline bool get_device_info_host<info::device::error_correction_support>() {
  return false;
}

template <>
inline bool get_device_info_host<info::device::host_unified_memory>() {
  return true;
}

template <>
inline size_t get_device_info_host<info::device::profiling_timer_resolution>() {
  typedef std::ratio_divide<std::chrono::high_resolution_clock::period,
                            std::nano>
      ns_period;
  return ns_period::num / ns_period::den;
}

template <> inline bool get_device_info_host<info::device::is_endian_little>() {
  union {
    uint16_t a;
    uint8_t b[2];
  } u = {0x0100};

  return u.b[1];
}

template <> inline bool get_device_info_host<info::device::is_available>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::is_compiler_available>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::is_linker_available>() {
  return true;
}

template <>
inline std::vector<info::execution_capability>
get_device_info_host<info::device::execution_capabilities>() {
  return {info::execution_capability::exec_kernel};
}

template <> inline bool get_device_info_host<info::device::queue_profiling>() {
  return true;
}

template <>
inline std::vector<kernel_id>
get_device_info_host<info::device::built_in_kernel_ids>() {
  return {};
}

template <>
inline std::vector<std::string>
get_device_info_host<info::device::built_in_kernels>() {
  return {};
}

template <> inline platform get_device_info_host<info::device::platform>() {
  return createSyclObjFromImpl<platform>(platform_impl::getHostPlatformImpl());
}

template <> inline std::string get_device_info_host<info::device::name>() {
  return "SYCL host device";
}

template <> inline std::string get_device_info_host<info::device::vendor>() {
  return "";
}

template <>
inline std::string get_device_info_host<info::device::driver_version>() {
  return "1.2";
}

template <> inline std::string get_device_info_host<info::device::profile>() {
  return "FULL PROFILE";
}

template <> inline std::string get_device_info_host<info::device::version>() {
  return "1.2";
}

template <>
inline std::string get_device_info_host<info::device::opencl_c_version>() {
  return "not applicable";
}

template <>
inline std::vector<std::string>
get_device_info_host<info::device::extensions>() {
  // TODO update when appropriate
  return {};
}

template <>
inline size_t get_device_info_host<info::device::printf_buffer_size>() {
  // current value is the required minimum
  return 1024 * 1024;
}

template <>
inline bool get_device_info_host<info::device::preferred_interop_user_sync>() {
  return false;
}

template <> inline device get_device_info_host<info::device::parent_device>() {
  throw invalid_object_error(
      "Partitioning to subdevices of the host device is not implemented",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t
get_device_info_host<info::device::partition_max_sub_devices>() {
  // TODO update once subdevice creation is enabled
  return 1;
}

template <>
inline std::vector<info::partition_property>
get_device_info_host<info::device::partition_properties>() {
  // TODO update once subdevice creation is enabled
  return {};
}

template <>
inline std::vector<info::partition_affinity_domain>
get_device_info_host<info::device::partition_affinity_domains>() {
  // TODO update once subdevice creation is enabled
  return {};
}

template <>
inline info::partition_property
get_device_info_host<info::device::partition_type_property>() {
  return info::partition_property::no_partition;
}

template <>
inline info::partition_affinity_domain
get_device_info_host<info::device::partition_type_affinity_domain>() {
  // TODO update once subdevice creation is enabled
  return info::partition_affinity_domain::not_applicable;
}

template <>
inline uint32_t get_device_info_host<info::device::reference_count>() {
  // TODO update once subdevice creation is enabled
  return 1;
}

template <>
inline uint32_t get_device_info_host<info::device::max_num_sub_groups>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline std::vector<size_t>
get_device_info_host<info::device::sub_group_sizes>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline bool
get_device_info_host<info::device::sub_group_independent_forward_progress>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline bool get_device_info_host<info::device::kernel_kernel_pipe_support>() {
  return false;
}

template <>
inline std::string get_device_info_host<info::device::backend_version>() {
  throw runtime_error(
      "Backend version feature is not supported on HOST device.",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline bool get_device_info_host<info::device::usm_device_allocations>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::usm_host_allocations>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::usm_shared_allocations>() {
  return true;
}

template <>
inline bool
get_device_info_host<info::device::usm_restricted_shared_allocations>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::usm_system_allocations>() {
  return true;
}

template <>
inline bool get_device_info_host<info::device::ext_intel_mem_channel>() {
  return false;
}

// Specializations for intel extensions for Level Zero low-level
// detail device descriptors (not support on host).
template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::device_id>() {
  throw runtime_error(
      "Obtaining the device ID is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline std::string
get_device_info_host<ext::intel::info::device::pci_address>() {
  throw runtime_error(
      "Obtaining the PCI address is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t get_device_info_host<ext::intel::info::device::gpu_eu_count>() {
  throw runtime_error("Obtaining the EU count is not supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::gpu_eu_simd_width>() {
  throw runtime_error(
      "Obtaining the EU SIMD width is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t get_device_info_host<ext::intel::info::device::gpu_slices>() {
  throw runtime_error(
      "Obtaining the number of slices is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::gpu_subslices_per_slice>() {
  throw runtime_error("Obtaining the number of subslices per slice is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::gpu_eu_count_per_subslice>() {
  throw runtime_error(
      "Obtaining the EU count per subslice is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::gpu_hw_threads_per_eu>() {
  throw runtime_error(
      "Obtaining the HW threads count per EU is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline uint64_t
get_device_info_host<ext::intel::info::device::max_mem_bandwidth>() {
  throw runtime_error(
      "Obtaining the maximum memory bandwidth is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
template <>
inline detail::uuid_type
get_device_info_host<ext::intel::info::device::uuid>() {
  throw runtime_error(
      "Obtaining the device uuid is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_pci_address>()
template <>
inline std::string get_device_info_host<info::device::ext_intel_pci_address>() {
  throw runtime_error(
      "Obtaining the PCI address is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_eu_count>()
template <>
inline uint32_t get_device_info_host<info::device::ext_intel_gpu_eu_count>() {
  throw runtime_error("Obtaining the EU count is not supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_eu_simd_width>()
template <>
inline uint32_t
get_device_info_host<info::device::ext_intel_gpu_eu_simd_width>() {
  throw runtime_error(
      "Obtaining the EU SIMD width is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_slices>()
template <>
inline uint32_t get_device_info_host<info::device::ext_intel_gpu_slices>() {
  throw runtime_error(
      "Obtaining the number of slices is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_subslices_per_slice>()
template <>
inline uint32_t
get_device_info_host<info::device::ext_intel_gpu_subslices_per_slice>() {
  throw runtime_error("Obtaining the number of subslices per slice is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_eu_count_per_subslices>()
template <>
inline uint32_t
get_device_info_host<info::device::ext_intel_gpu_eu_count_per_subslice>() {
  throw runtime_error(
      "Obtaining the EU count per subslice is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_gpu_hw_threads_per_eu>()
template <>
inline uint32_t
get_device_info_host<info::device::ext_intel_gpu_hw_threads_per_eu>() {
  throw runtime_error(
      "Obtaining the HW threads count per EU is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_max_mem_bandwidth>()
template <>
inline uint64_t
get_device_info_host<info::device::ext_intel_max_mem_bandwidth>() {
  throw runtime_error(
      "Obtaining the maximum memory bandwidth is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}
// TODO:Move to namespace ext::intel::info::device
template <> inline bool get_device_info_host<info::device::ext_oneapi_srgb>() {
  return false;
}

// TODO: Remove with deprecated feature
// device::get_info<info::device::ext_intel_device_info_uuid>()
template <>
inline detail::uuid_type
get_device_info_host<info::device::ext_intel_device_info_uuid>() {
  throw runtime_error(
      "Obtaining the device uuid is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint64_t get_device_info_host<ext::intel::info::device::free_memory>() {
  throw runtime_error(
      "Obtaining the device free memory is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::memory_clock_rate>() {
  throw runtime_error(
      "Obtaining the device memory clock rate is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t
get_device_info_host<ext::intel::info::device::memory_bus_width>() {
  throw runtime_error(
      "Obtaining the device memory bus width is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

template <>
inline int32_t
get_device_info_host<ext::intel::info::device::max_compute_queue_indices>() {
  throw runtime_error(
      "Obtaining max compute queue indices is not supported on HOST device",
      PI_ERROR_INVALID_DEVICE);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
