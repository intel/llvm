//==-------- device_info.hpp - SYCL device info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/memory_enums.hpp>
#include <CL/sycl/platform.hpp>
#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_util.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <chrono>
#include <thread>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

inline std::vector<info::fp_config> read_fp_bitfield(cl_device_fp_config bits) {
  std::vector<info::fp_config> result;
  if (bits & CL_FP_DENORM)
    result.push_back(info::fp_config::denorm);
  if (bits & CL_FP_INF_NAN)
    result.push_back(info::fp_config::inf_nan);
  if (bits & CL_FP_ROUND_TO_NEAREST)
    result.push_back(info::fp_config::round_to_nearest);
  if (bits & CL_FP_ROUND_TO_ZERO)
    result.push_back(info::fp_config::round_to_zero);
  if (bits & CL_FP_ROUND_TO_INF)
    result.push_back(info::fp_config::round_to_inf);
  if (bits & CL_FP_FMA)
    result.push_back(info::fp_config::fma);
  if (bits & CL_FP_SOFT_FLOAT)
    result.push_back(info::fp_config::soft_float);
  if (bits & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    result.push_back(info::fp_config::correctly_rounded_divide_sqrt);
  return result;
}

inline std::vector<info::partition_affinity_domain>
read_domain_bitfield(cl_device_affinity_domain bits) {
  std::vector<info::partition_affinity_domain> result;
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_NUMA)
    result.push_back(info::partition_affinity_domain::numa);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
    result.push_back(info::partition_affinity_domain::L4_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
    result.push_back(info::partition_affinity_domain::L3_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
    result.push_back(info::partition_affinity_domain::L2_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
    result.push_back(info::partition_affinity_domain::L1_cache);
  if (bits & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
    result.push_back(info::partition_affinity_domain::next_partitionable);
  return result;
}

inline std::vector<info::execution_capability>
read_execution_bitfield(cl_device_exec_capabilities bits) {
  std::vector<info::execution_capability> result;
  if (bits & CL_EXEC_KERNEL)
    result.push_back(info::execution_capability::exec_kernel);
  if (bits & CL_EXEC_NATIVE_KERNEL)
    result.push_back(info::execution_capability::exec_native_kernel);
  return result;
}

// Mapping expected SYCL return types to those returned by PI calls
template <typename T> struct sycl_to_pi { using type = T; };
template <> struct sycl_to_pi<bool> { using type = pi_bool; };
template <> struct sycl_to_pi<device> { using type = RT::PiDevice; };
template <> struct sycl_to_pi<platform> { using type = RT::PiPlatform; };

// Mapping fp_config device info types to the values used to check fp support
template <info::device param> struct check_fp_support {};

template <> struct check_fp_support<info::device::half_fp_config> {
  static const info::device value = info::device::native_vector_width_half;
};

template <> struct check_fp_support<info::device::double_fp_config> {
  static const info::device value = info::device::native_vector_width_double;
};

// Structs for emulating function template partial specialization
// Default template for the general case
// TODO: get rid of remaining uses of OpenCL directly
//
template <typename T, info::device param> struct get_device_info {
  static T get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<T>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            sizeof(result), &result, nullptr);
    return T(result);
  }
};

// Specialization for platform
template <info::device param> struct get_device_info<platform, param> {
  static platform get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<platform>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            sizeof(result), &result, nullptr);
    // TODO: Change PiDevice to device_impl.
    // Use the Plugin from the device_impl class after plugin details
    // are added to the class.
    return createSyclObjFromImpl<platform>(
        platform_impl::getOrMakePlatformImpl(result, Plugin));
  }
};

// Helper struct to allow using the specialization of get_device_info
// for string return type in other specializations.
template <info::device param> struct get_device_info_string {
  static std::string get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(param), 0, nullptr, &resultSize);
    if (resultSize == 0) {
      return std::string();
    }
    std::unique_ptr<char[]> result(new char[resultSize]);
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            resultSize, result.get(), nullptr);

    return std::string(result.get());
  }
};

// Specialization for string return type, variable return size
template <info::device param> struct get_device_info<std::string, param> {
  static std::string get(RT::PiDevice dev, const plugin &Plugin) {
    return get_device_info_string<param>::get(dev, Plugin);
  }
};

// Specialization for parent device
template <typename T> struct get_device_info<T, info::device::parent_device> {
  static T get(RT::PiDevice dev, const plugin &Plugin);
};

// Specialization for id return type
template <info::device param> struct get_device_info<id<3>, param> {
  static id<3> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            sizeof(result), &result, nullptr);
    return id<3>(result[0], result[1], result[2]);
  }
};

// Specialization for fp_config types, checks the corresponding fp type support
template <info::device param>
struct get_device_info<std::vector<info::fp_config>, param> {
  static std::vector<info::fp_config> get(RT::PiDevice dev,
                                          const plugin &Plugin) {
    // Check if fp type is supported
    if (!get_device_info<
            typename info::param_traits<
                info::device, check_fp_support<param>::value>::return_type,
            check_fp_support<param>::value>::get(dev, Plugin)) {
      return {};
    }
    cl_device_fp_config result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            sizeof(result), &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for OpenCL version, splits the string returned by OpenCL
template <> struct get_device_info<std::string, info::device::version> {
  static std::string get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result =
        get_device_info_string<info::device::version>::get(dev, Plugin);

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
struct get_device_info<std::vector<info::fp_config>,
                       info::device::single_fp_config> {
  static std::vector<info::fp_config> get(RT::PiDevice dev,
                                          const plugin &Plugin) {
    cl_device_fp_config result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::single_fp_config),
        sizeof(result), &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling, OpenCL returns a bitfield
template <> struct get_device_info<bool, info::device::queue_profiling> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    cl_command_queue_properties result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::queue_profiling),
        sizeof(result), &result, nullptr);
    return (result & CL_QUEUE_PROFILING_ENABLE);
  }
};

// Specialization for atomic64 that is necessary because
// PI_DEVICE_INFO_ATOMIC_64 is currently only implemented for the cuda backend.
template <> struct get_device_info<bool, info::device::atomic64> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {

    bool result = false;

    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::atomic64), sizeof(result),
        &result, nullptr);
    if (Err != PI_SUCCESS) {
      return false;
    }
    return result;
  }
};

// Specialization for atomic_memory_order_capabilities, PI returns a bitfield
template <>
struct get_device_info<std::vector<memory_order>,
                       info::device::atomic_memory_order_capabilities> {
  static std::vector<memory_order> get(RT::PiDevice dev, const plugin &Plugin) {
    pi_memory_order_capabilities result;
    Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(
            info::device::atomic_memory_order_capabilities),
        sizeof(pi_memory_order_capabilities), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_memory_scope_capabilities, PI returns a bitfield
template <>
struct get_device_info<std::vector<memory_scope>,
                       info::device::atomic_memory_scope_capabilities> {
  static std::vector<memory_scope> get(RT::PiDevice dev, const plugin &Plugin) {
    pi_memory_scope_capabilities result;
    Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(
            info::device::atomic_memory_scope_capabilities),
        sizeof(pi_memory_scope_capabilities), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info<std::vector<info::execution_capability>,
                       info::device::execution_capabilities> {
  static std::vector<info::execution_capability> get(RT::PiDevice dev,
                                                     const plugin &Plugin) {
    cl_device_exec_capabilities result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::execution_capabilities),
        sizeof(result), &result, nullptr);
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernel identifiers
template <>
struct get_device_info<std::vector<kernel_id>,
                       info::device::built_in_kernel_ids> {
  static std::vector<kernel_id> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result =
        get_device_info<std::string, info::device::built_in_kernels>::get(
            dev, Plugin);
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
struct get_device_info<std::vector<std::string>,
                       info::device::built_in_kernels> {
  static std::vector<std::string> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result =
        get_device_info<std::string, info::device::built_in_kernels>::get(
            dev, Plugin);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info<std::vector<std::string>, info::device::extensions> {
  static std::vector<std::string> get(RT::PiDevice dev, const plugin &Plugin) {
    std::string result =
        get_device_info<std::string, info::device::extensions>::get(dev,
                                                                    Plugin);
    return split_string(result, ' ');
  }
};

static bool is_sycl_partition_property(info::partition_property PP) {
  switch (PP) {
  case info::partition_property::no_partition:
  case info::partition_property::partition_equally:
  case info::partition_property::partition_by_counts:
  case info::partition_property::partition_by_affinity_domain:
    return true;
  }
  return false;
}

// Specialization for partition properties, variable OpenCL return size
template <>
struct get_device_info<std::vector<info::partition_property>,
                       info::device::partition_properties> {
  static std::vector<info::partition_property> get(RT::PiDevice dev,
                                                   const plugin &Plugin) {
    auto info_partition =
        pi::cast<RT::PiDeviceInfo>(info::device::partition_properties);

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
struct get_device_info<std::vector<info::partition_affinity_domain>,
                       info::device::partition_affinity_domains> {
  static std::vector<info::partition_affinity_domain>
  get(RT::PiDevice dev, const plugin &Plugin) {
    cl_device_affinity_domain result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(info::device::partition_affinity_domains),
        sizeof(result), &result, nullptr);
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, OpenCL can return other
// partition properties instead
template <>
struct get_device_info<info::partition_affinity_domain,
                       info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain get(RT::PiDevice dev,
                                             const plugin &Plugin) {
    size_t resultSize;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(
            info::device::partition_type_affinity_domain),
        0, nullptr, &resultSize);
    if (resultSize != 1) {
      return info::partition_affinity_domain::not_applicable;
    }
    cl_device_partition_property result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(
            info::device::partition_type_affinity_domain),
        sizeof(result), &result, nullptr);
    if (result == CL_DEVICE_AFFINITY_DOMAIN_NUMA ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ||
        result == CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE) {
      return info::partition_affinity_domain(result);
    }

    return info::partition_affinity_domain::not_applicable;
  }
};

// Specialization for partition type
template <>
struct get_device_info<info::partition_property,
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
struct get_device_info<std::vector<size_t>, info::device::sub_group_sizes> {
  static std::vector<size_t> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::sub_group_sizes), 0,
        nullptr, &resultSize);

    std::vector<size_t> result(resultSize / sizeof(size_t));
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::sub_group_sizes),
        resultSize, result.data(), nullptr);
    return result;
  }
};

// Specialization for kernel to kernel pipes.
// Here we step away from OpenCL, since there is no appropriate cl_device_info
// enum for global pipes feature.
template <>
struct get_device_info<bool, info::device::kernel_kernel_pipe_support> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    // We claim, that all Intel FPGA devices support kernel to kernel pipe
    // feature (at least at the scope of SYCL_INTEL_data_flow_pipes extension).
    platform plt =
        get_device_info<platform, info::device::platform>::get(dev, Plugin);
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

// Specialization for max_work_item_sizes.
// Due to the flipping of work group dimensions before kernel launch, the max
// sizes should also be reversed.
template <> struct get_device_info<id<3>, info::device::max_work_item_sizes> {
  static id<3> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::max_work_item_sizes),
        sizeof(result), &result, nullptr);
    return id<3>(result[2], result[1], result[0]);
  }
};

template <>
struct get_device_info<size_t,
                       info::device::ext_oneapi_max_global_work_groups> {
  static size_t get(RT::PiDevice dev, const plugin &Plugin) {
    (void)dev; // Silence unused warning
    (void)Plugin;
    return static_cast<size_t>((std::numeric_limits<int>::max)());
  }
};

template <>
struct get_device_info<id<1>, info::device::ext_oneapi_max_work_groups_1d> {
  static id<1> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit = get_device_info<
        size_t, info::device::ext_oneapi_max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(info::device::ext_oneapi_max_work_groups_3d),
        sizeof(result), &result, nullptr);
    return id<1>(std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info<id<2>, info::device::ext_oneapi_max_work_groups_2d> {
  static id<2> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit = get_device_info<
        size_t, info::device::ext_oneapi_max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(info::device::ext_oneapi_max_work_groups_3d),
        sizeof(result), &result, nullptr);
    return id<2>(std::min(Limit, result[1]), std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info<id<3>, info::device::ext_oneapi_max_work_groups_3d> {
  static id<3> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t result[3];
    size_t Limit = get_device_info<
        size_t, info::device::ext_oneapi_max_global_work_groups>::get(dev,
                                                                      Plugin);
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(info::device::ext_oneapi_max_work_groups_3d),
        sizeof(result), &result, nullptr);
    return id<3>(std::min(Limit, result[2]), std::min(Limit, result[1]),
                 std::min(Limit, result[0]));
  }
};

// Specialization for parent device
template <> struct get_device_info<device, info::device::parent_device> {
  static device get(RT::PiDevice dev, const plugin &Plugin) {
    typename sycl_to_pi<device>::type result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::parent_device),
        sizeof(result), &result, nullptr);
    if (result == nullptr)
      throw invalid_object_error(
          "No parent for device because it is not a subdevice",
          PI_INVALID_DEVICE);

    // Get the platform of this device
    std::shared_ptr<detail::platform_impl> Platform =
        platform_impl::getPlatformFromPiDevice(dev, Plugin);
    return createSyclObjFromImpl<device>(
        Platform->getOrMakeDeviceImpl(result, Platform));
  }
};

// SYCL host device information

// Default template is disabled, all possible instantiations are
// specified explicitly.
template <info::device param>
inline typename info::param_traits<info::device, param>::return_type
get_device_info_host() = delete;

template <>
inline info::device_type get_device_info_host<info::device::device_type>() {
  return info::device_type::host;
}

template <> inline cl_uint get_device_info_host<info::device::vendor_id>() {
  return 0x8086;
}

template <>
inline cl_uint get_device_info_host<info::device::max_compute_units>() {
  return std::thread::hardware_concurrency();
}

template <>
inline cl_uint get_device_info_host<info::device::max_work_item_dimensions>() {
  return 3;
}

template <>
inline id<3> get_device_info_host<info::device::max_work_item_sizes>() {
  // current value is the required minimum
  return {1, 1, 1};
}

template <>
inline constexpr size_t
get_device_info_host<info::device::ext_oneapi_max_global_work_groups>() {
  // See handler.hpp for the maximum value :
  return static_cast<size_t>((std::numeric_limits<int>::max)());
}

template <>
inline id<1>
get_device_info_host<info::device::ext_oneapi_max_work_groups_1d>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit =
      get_device_info_host<info::device::ext_oneapi_max_global_work_groups>();
  return {Limit};
}

template <>
inline id<2>
get_device_info_host<info::device::ext_oneapi_max_work_groups_2d>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit =
      get_device_info_host<info::device::ext_oneapi_max_global_work_groups>();
  return {Limit, Limit};
}

template <>
inline id<3>
get_device_info_host<info::device::ext_oneapi_max_work_groups_3d>() {
  // See handler.hpp for the maximum value :
  static constexpr size_t Limit =
      get_device_info_host<info::device::ext_oneapi_max_global_work_groups>();
  return {Limit, Limit, Limit};
}

template <>
inline size_t get_device_info_host<info::device::max_work_group_size>() {
  // current value is the required minimum
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_char>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_short>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_int>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_long>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_float>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_double>() {
  // TODO update when appropriate
  return 1;
}

template <>
inline cl_uint
get_device_info_host<info::device::preferred_vector_width_half>() {
  // TODO update when appropriate
  return 0;
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_char>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Char);
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_short>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Short);
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_int>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Int);
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_long>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Long);
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_float>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Float);
}

template <>
inline cl_uint
get_device_info_host<info::device::native_vector_width_double>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Double);
}

template <>
inline cl_uint get_device_info_host<info::device::native_vector_width_half>() {
  return PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex::Half);
}

template <>
inline cl_uint get_device_info_host<info::device::max_clock_frequency>() {
  return PlatformUtil::getMaxClockFrequency();
}

template <> inline cl_uint get_device_info_host<info::device::address_bits>() {
  return sizeof(void *) * 8;
}

template <>
inline cl_ulong get_device_info_host<info::device::global_mem_size>() {
  return static_cast<cl_ulong>(OSUtil::getOSMemSize());
}

template <>
inline cl_ulong get_device_info_host<info::device::max_mem_alloc_size>() {
  // current value is the required minimum
  const cl_ulong a = get_device_info_host<info::device::global_mem_size>() / 4;
  const cl_ulong b = 128ul * 1024 * 1024;
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
inline cl_uint get_device_info_host<info::device::max_read_image_args>() {
  // current value is the required minimum
  return 128;
}

template <>
inline cl_uint get_device_info_host<info::device::max_write_image_args>() {
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

template <> inline cl_uint get_device_info_host<info::device::max_samplers>() {
  // current value is the required minimum
  return 16;
}

template <>
inline size_t get_device_info_host<info::device::max_parameter_size>() {
  // current value is the required minimum
  return 1024;
}

template <>
inline cl_uint get_device_info_host<info::device::mem_base_addr_align>() {
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
inline cl_uint
get_device_info_host<info::device::global_mem_cache_line_size>() {
  return PlatformUtil::getMemCacheLineSize();
}

template <>
inline cl_ulong get_device_info_host<info::device::global_mem_cache_size>() {
  return PlatformUtil::getMemCacheSize();
}

template <>
inline cl_ulong get_device_info_host<info::device::max_constant_buffer_size>() {
  // current value is the required minimum
  return 64 * 1024;
}

template <>
inline cl_uint get_device_info_host<info::device::max_constant_args>() {
  // current value is the required minimum
  return 8;
}

template <>
inline info::local_mem_type
get_device_info_host<info::device::local_mem_type>() {
  return info::local_mem_type::global;
}

template <>
inline cl_ulong get_device_info_host<info::device::local_mem_size>() {
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
  return platform();
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
  // TODO: implement host device partitioning
  throw runtime_error(
      "Partitioning to subdevices of the host device is not implemented yet",
      PI_INVALID_DEVICE);
}

template <>
inline cl_uint get_device_info_host<info::device::partition_max_sub_devices>() {
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
inline cl_uint get_device_info_host<info::device::reference_count>() {
  // TODO update once subdevice creation is enabled
  return 1;
}

template <>
inline cl_uint get_device_info_host<info::device::max_num_sub_groups>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_INVALID_DEVICE);
}

template <>
inline std::vector<size_t>
get_device_info_host<info::device::sub_group_sizes>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_INVALID_DEVICE);
}

template <>
inline bool
get_device_info_host<info::device::sub_group_independent_forward_progress>() {
  // TODO update once subgroups are enabled
  throw runtime_error("Sub-group feature is not supported on HOST device.",
                      PI_INVALID_DEVICE);
}

template <>
inline bool get_device_info_host<info::device::kernel_kernel_pipe_support>() {
  return false;
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

cl_uint get_native_vector_width(size_t idx);

// USM

// Specialization for device usm query.
template <> struct get_device_info<bool, info::device::usm_device_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::usm_device_allocations),
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for host usm query.
template <> struct get_device_info<bool, info::device::usm_host_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::usm_host_allocations),
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for shared usm query.
template <> struct get_device_info<bool, info::device::usm_shared_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::usm_shared_allocations),
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for restricted usm query
template <>
struct get_device_info<bool, info::device::usm_restricted_shared_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev,
        pi::cast<RT::PiDeviceInfo>(
            info::device::usm_restricted_shared_allocations),
        sizeof(pi_usm_capabilities), &caps, nullptr);
    // Check that we don't support any cross device sharing
    return (Err != PI_SUCCESS)
               ? false
               : !(caps & (PI_USM_ACCESS | PI_USM_CONCURRENT_ACCESS));
  }
};

// Specialization for system usm query
template <> struct get_device_info<bool, info::device::usm_system_allocations> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::usm_system_allocations),
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for memory channel query
template <> struct get_device_info<bool, info::device::ext_intel_mem_channel> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_mem_properties caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::ext_intel_mem_channel),
        sizeof(pi_mem_properties), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_MEM_PROPERTIES_CHANNEL);
  }
};

// Specializations for intel extensions for Level Zero low-level
// detail device descriptors (not support on host).
template <>
inline std::string get_device_info_host<info::device::ext_intel_pci_address>() {
  throw runtime_error(
      "Obtaining the PCI address is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <>
inline cl_uint get_device_info_host<info::device::ext_intel_gpu_eu_count>() {
  throw runtime_error("Obtaining the EU count is not supported on HOST device",
                      PI_INVALID_DEVICE);
}
template <>
inline cl_uint
get_device_info_host<info::device::ext_intel_gpu_eu_simd_width>() {
  throw runtime_error(
      "Obtaining the EU SIMD width is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <>
inline cl_uint get_device_info_host<info::device::ext_intel_gpu_slices>() {
  throw runtime_error(
      "Obtaining the number of slices is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <>
inline cl_uint
get_device_info_host<info::device::ext_intel_gpu_subslices_per_slice>() {
  throw runtime_error("Obtaining the number of subslices per slice is not "
                      "supported on HOST device",
                      PI_INVALID_DEVICE);
}
template <>
inline cl_uint
get_device_info_host<info::device::ext_intel_gpu_eu_count_per_subslice>() {
  throw runtime_error(
      "Obtaining the EU count per subslice is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <>
inline cl_uint
get_device_info_host<info::device::ext_intel_gpu_hw_threads_per_eu>() {
  throw runtime_error(
      "Obtaining the HW threads count per EU is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <>
inline cl_ulong
get_device_info_host<info::device::ext_intel_max_mem_bandwidth>() {
  throw runtime_error(
      "Obtaining the maximum memory bandwidth is not supported on HOST device",
      PI_INVALID_DEVICE);
}
template <> inline bool get_device_info_host<info::device::ext_oneapi_srgb>() {
  return false;
}

template <>
inline detail::uuid_type
get_device_info_host<info::device::ext_intel_device_info_uuid>() {
  throw runtime_error(
      "Obtaining the device uuid is not supported on HOST device",
      PI_INVALID_DEVICE);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
