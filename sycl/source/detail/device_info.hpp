//==-------- device_info.hpp - SYCL device info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/device_impl.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_util.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/platform.hpp>

#include <chrono>
#include <sstream>
#include <thread>

#include "split_string.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail {

inline std::vector<info::fp_config>
read_fp_bitfield(ur_device_fp_capability_flags_t bits) {
  std::vector<info::fp_config> result;
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_DENORM)
    result.push_back(info::fp_config::denorm);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN)
    result.push_back(info::fp_config::inf_nan);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST)
    result.push_back(info::fp_config::round_to_nearest);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO)
    result.push_back(info::fp_config::round_to_zero);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF)
    result.push_back(info::fp_config::round_to_inf);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_FMA)
    result.push_back(info::fp_config::fma);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT)
    result.push_back(info::fp_config::soft_float);
  if (bits & UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    result.push_back(info::fp_config::correctly_rounded_divide_sqrt);
  return result;
}

inline std::vector<info::partition_affinity_domain>
read_domain_bitfield(ur_device_affinity_domain_flags_t bits) {
  std::vector<info::partition_affinity_domain> result;
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA)
    result.push_back(info::partition_affinity_domain::numa);
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE)
    result.push_back(info::partition_affinity_domain::L4_cache);
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE)
    result.push_back(info::partition_affinity_domain::L3_cache);
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE)
    result.push_back(info::partition_affinity_domain::L2_cache);
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE)
    result.push_back(info::partition_affinity_domain::L1_cache);
  if (bits & UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE)
    result.push_back(info::partition_affinity_domain::next_partitionable);
  return result;
}

inline std::vector<info::execution_capability>
read_execution_bitfield(ur_device_exec_capability_flag_t bits) {
  std::vector<info::execution_capability> result;
  if (bits & UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL)
    result.push_back(info::execution_capability::exec_kernel);
  if (bits & UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL)
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

// Mapping expected SYCL return types to those returned by UR calls
template <typename T> struct sycl_to_ur {
  using type = T;
};
template <> struct sycl_to_ur<bool> {
  using type = ur_bool_t;
};
template <> struct sycl_to_ur<device> {
  using type = ur_device_handle_t;
};
template <> struct sycl_to_ur<platform> {
  using type = ur_platform_handle_t;
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

template <typename ReturnT, typename Param> struct get_device_info_impl {
  static ReturnT get(const DeviceImplPtr &Dev) {
    typename sycl_to_ur<ReturnT>::type result;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<Param>::value, sizeof(result), &result,
                           nullptr);
    return ReturnT(result);
  }
};

// Specialization for platform
template <typename Param> struct get_device_info_impl<platform, Param> {
  static platform get(const DeviceImplPtr &Dev) {
    typename sycl_to_ur<platform>::type result;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<Param>::value, sizeof(result), &result,
                           nullptr);
    // TODO: Change UrDevice to device_impl.
    // Use the Plugin from the device_impl class after plugin details
    // are added to the class.
    return createSyclObjFromImpl<platform>(
        platform_impl::getOrMakePlatformImpl(result, Dev->getPlugin()));
  }
};

// Helper function to allow using the specialization of get_device_info_impl
// for string return type in other specializations.
inline std::string
device_impl::get_device_info_string(ur_device_info_t InfoCode) const {
  size_t resultSize = 0;
  getPlugin()->call(urDeviceGetInfo, getHandleRef(), InfoCode, 0, nullptr,
                    &resultSize);
  if (resultSize == 0) {
    return std::string();
  }
  std::unique_ptr<char[]> result(new char[resultSize]);
  getPlugin()->call(urDeviceGetInfo, getHandleRef(), InfoCode, resultSize,
                    result.get(), nullptr);

  return std::string(result.get());
}

// Specialization for string return type, variable return size
template <typename Param> struct get_device_info_impl<std::string, Param> {
  static std::string get(const DeviceImplPtr &Dev) {
    return Dev->get_device_info_string(UrInfoCode<Param>::value);
  }
};

// Specialization for parent device
template <typename ReturnT>
struct get_device_info_impl<ReturnT, info::device::parent_device> {
  static ReturnT get(const DeviceImplPtr &Dev);
};

// Specialization for fp_config types, checks the corresponding fp type support
template <typename Param>
struct get_device_info_impl<std::vector<info::fp_config>, Param> {
  static std::vector<info::fp_config> get(const DeviceImplPtr &Dev) {
    // Check if fp type is supported
    if (!get_device_info_impl<
            typename check_fp_support<Param>::type::return_type,
            typename check_fp_support<Param>::type>::get(Dev)) {
      return {};
    }
    ur_device_fp_capability_flags_t result;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<Param>::value, sizeof(result), &result,
                           nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for device version
template <> struct get_device_info_impl<std::string, info::device::version> {
  static std::string get(const DeviceImplPtr &Dev) {
    return Dev->get_device_info_string(
        UrInfoCode<info::device::version>::value);
  }
};

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info_impl<std::vector<info::fp_config>,
                            info::device::single_fp_config> {
  static std::vector<info::fp_config> get(const DeviceImplPtr &Dev) {
    ur_device_fp_capability_flags_t result;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<info::device::single_fp_config>::value,
                           sizeof(result), &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling. In addition to ur_queue level profiling,
// urDeviceGetGlobalTimestamps is not supported, command_submit, command_start,
// command_end will be calculated. See MFallbackProfiling
template <> struct get_device_info_impl<bool, info::device::queue_profiling> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_queue_flags_t Properties;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<info::device::queue_profiling>::value,
                           sizeof(Properties), &Properties, nullptr);
    return Properties & UR_QUEUE_FLAG_PROFILING_ENABLE;
  }
};

// Specialization for atomic_memory_order_capabilities, UR returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_order>,
                            info::device::atomic_memory_order_capabilities> {
  static std::vector<memory_order> get(const DeviceImplPtr &Dev) {
    ur_memory_order_capability_flag_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::atomic_memory_order_capabilities>::value,
        sizeof(result), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_fence_order_capabilities, UR returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_order>,
                            info::device::atomic_fence_order_capabilities> {
  static std::vector<memory_order> get(const DeviceImplPtr &Dev) {
    ur_memory_order_capability_flag_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::atomic_fence_order_capabilities>::value,
        sizeof(result), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_memory_scope_capabilities, UR returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_scope>,
                            info::device::atomic_memory_scope_capabilities> {
  static std::vector<memory_scope> get(const DeviceImplPtr &Dev) {
    size_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::atomic_memory_scope_capabilities>::value,
        sizeof(result), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for atomic_fence_scope_capabilities, UR returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_scope>,
                            info::device::atomic_fence_scope_capabilities> {
  static std::vector<memory_scope> get(const DeviceImplPtr &Dev) {
    size_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::atomic_fence_scope_capabilities>::value,
        sizeof(result), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for cuda cluster group
template <>
struct get_device_info_impl<bool, info::device::ext_oneapi_cuda_cluster_group> {
  static bool get(const DeviceImplPtr &Dev) {
    bool result = false;
    if (Dev->getBackend() == backend::ext_oneapi_cuda) {
      auto Err = Dev->getPlugin()->call_nocheck(
          urDeviceGetInfo, Dev->getHandleRef(),
          UrInfoCode<info::device::ext_oneapi_cuda_cluster_group>::value,
          sizeof(result), &result, nullptr);
      if (Err != UR_RESULT_SUCCESS) {
        return false;
      }
    }
    return result;
  }
};

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info_impl<std::vector<info::execution_capability>,
                            info::device::execution_capabilities> {
  static std::vector<info::execution_capability> get(const DeviceImplPtr &Dev) {
    ur_device_exec_capability_flag_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::execution_capabilities>::value, sizeof(result),
        &result, nullptr);
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernel identifiers
template <>
struct get_device_info_impl<std::vector<kernel_id>,
                            info::device::built_in_kernel_ids> {
  static std::vector<kernel_id> get(const DeviceImplPtr &Dev) {
    std::string result = Dev->get_device_info_string(
        UrInfoCode<info::device::built_in_kernels>::value);
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
  static std::vector<std::string> get(const DeviceImplPtr &Dev) {
    std::string result = Dev->get_device_info_string(
        UrInfoCode<info::device::built_in_kernels>::value);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info_impl<std::vector<std::string>,
                            info::device::extensions> {
  static std::vector<std::string> get(const DeviceImplPtr &Dev) {
    std::string result =
        get_device_info_impl<std::string, info::device::extensions>::get(Dev);
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
  static std::vector<info::partition_property> get(const DeviceImplPtr &Dev) {
    auto info_partition = UrInfoCode<info::device::partition_properties>::value;
    const auto &Plugin = Dev->getPlugin();

    size_t resultSize;
    Plugin->call(urDeviceGetInfo, Dev->getHandleRef(), info_partition, 0,
                 nullptr, &resultSize);

    size_t arrayLength = resultSize / sizeof(ur_device_partition_t);
    if (arrayLength == 0) {
      return {};
    }
    std::unique_ptr<ur_device_partition_t[]> arrayResult(
        new ur_device_partition_t[arrayLength]);
    Plugin->call(urDeviceGetInfo, Dev->getHandleRef(), info_partition,
                 resultSize, arrayResult.get(), nullptr);

    std::vector<info::partition_property> result;
    for (size_t i = 0; i < arrayLength; ++i) {
      // OpenCL extensions may have partition_properties that
      // are not yet defined for SYCL (eg. CL_DEVICE_PARTITION_BY_NAMES_INTEL)
      info::partition_property pp(
          info::ConvertPartitionProperty(arrayResult[i]));
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
  get(const DeviceImplPtr &Dev) {
    ur_device_affinity_domain_flags_t result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::partition_affinity_domains>::value,
        sizeof(result), &result, nullptr);
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, UR can return other
// partition properties instead
template <>
struct get_device_info_impl<info::partition_affinity_domain,
                            info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain get(const DeviceImplPtr &Dev) {
    std::vector<ur_device_partition_property_t> PartitionProperties;
    size_t PropertiesSize = 0;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::partition_type_affinity_domain>::value, 0,
        nullptr, &PropertiesSize);
    if (PropertiesSize == 0)
      return info::partition_affinity_domain::not_applicable;

    PartitionProperties.resize(PropertiesSize /
                               sizeof(ur_device_partition_property_t));

    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::partition_type_affinity_domain>::value,
        PropertiesSize, PartitionProperties.data(), nullptr);

    for (const auto &PartitionProp : PartitionProperties) {
      if (PartitionProp.type != UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)
        continue;
      return info::ConvertAffinityDomain(PartitionProp.value.affinity_domain);
    }

    return info::partition_affinity_domain::not_applicable;
  }
};

// Specialization for partition type
template <>
struct get_device_info_impl<info::partition_property,
                            info::device::partition_type_property> {
  static info::partition_property get(const DeviceImplPtr &Dev) {
    std::vector<ur_device_partition_property_t> PartitionProperties;
    size_t PropertiesSize = 0;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::partition_type_affinity_domain>::value, 0,
        nullptr, &PropertiesSize);
    if (PropertiesSize == 0)
      return info::partition_property::no_partition;

    PartitionProperties.resize(PropertiesSize /
                               sizeof(ur_device_partition_property_t));

    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::partition_type_affinity_domain>::value,
        PropertiesSize, PartitionProperties.data(), nullptr);
    // The old UR implementation also just checked the first element, is that
    // correct?
    return info::ConvertPartitionProperty(PartitionProperties[0].type);
  }
};

// Specialization for supported subgroup sizes
template <>
struct get_device_info_impl<std::vector<size_t>,
                            info::device::sub_group_sizes> {
  static std::vector<size_t> get(const DeviceImplPtr &Dev) {
    size_t resultSize = 0;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<info::device::sub_group_sizes>::value, 0,
                           nullptr, &resultSize);

    std::vector<uint32_t> result32(resultSize / sizeof(uint32_t));
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<info::device::sub_group_sizes>::value,
                           resultSize, result32.data(), nullptr);

    std::vector<size_t> result;
    result.reserve(result32.size());
    for (uint32_t value : result32) {
      result.push_back(value);
    }
    return result;
  }
};

// Specialization for kernel to kernel pipes.
// Here we step away from OpenCL, since there is no appropriate cl_device_info
// enum for global pipes feature.
template <>
struct get_device_info_impl<bool, info::device::kernel_kernel_pipe_support> {
  static bool get(const DeviceImplPtr &Dev) {
    // We claim, that all Intel FPGA devices support kernel to kernel pipe
    // feature (at least at the scope of SYCL_INTEL_data_flow_pipes extension).
    platform plt =
        get_device_info_impl<platform, info::device::platform>::get(Dev);
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

template <int Dimensions>
range<Dimensions> construct_range(size_t *values) = delete;
// Due to the flipping of work group dimensions before kernel launch, the values
// should also be reversed.
template <> inline range<1> construct_range<1>(size_t *values) {
  return {values[0]};
}
template <> inline range<2> construct_range<2>(size_t *values) {
  return {values[1], values[0]};
}
template <> inline range<3> construct_range<3>(size_t *values) {
  return {values[2], values[1], values[0]};
}

// Specialization for max_work_item_sizes.
template <int Dimensions>
struct get_device_info_impl<range<Dimensions>,
                            info::device::max_work_item_sizes<Dimensions>> {
  static range<Dimensions> get(const DeviceImplPtr &Dev) {
    size_t result[3];
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::max_work_item_sizes<Dimensions>>::value,
        sizeof(result), &result, nullptr);
    return construct_range<Dimensions>(result);
  }
};

using oneapi_exp_arch = sycl::ext::oneapi::experimental::architecture;

// Only for NVIDIA and AMD GPU architectures
constexpr std::pair<const char *, oneapi_exp_arch> NvidiaAmdGPUArchitectures[] =
    {
        {"5.0", oneapi_exp_arch::nvidia_gpu_sm_50},
        {"5.2", oneapi_exp_arch::nvidia_gpu_sm_52},
        {"5.3", oneapi_exp_arch::nvidia_gpu_sm_53},
        {"6.0", oneapi_exp_arch::nvidia_gpu_sm_60},
        {"6.1", oneapi_exp_arch::nvidia_gpu_sm_61},
        {"6.2", oneapi_exp_arch::nvidia_gpu_sm_62},
        {"7.0", oneapi_exp_arch::nvidia_gpu_sm_70},
        {"7.2", oneapi_exp_arch::nvidia_gpu_sm_72},
        {"7.5", oneapi_exp_arch::nvidia_gpu_sm_75},
        {"8.0", oneapi_exp_arch::nvidia_gpu_sm_80},
        {"8.6", oneapi_exp_arch::nvidia_gpu_sm_86},
        {"8.7", oneapi_exp_arch::nvidia_gpu_sm_87},
        {"8.9", oneapi_exp_arch::nvidia_gpu_sm_89},
        {"9.0", oneapi_exp_arch::nvidia_gpu_sm_90},
        {"gfx701", oneapi_exp_arch::amd_gpu_gfx701},
        {"gfx702", oneapi_exp_arch::amd_gpu_gfx702},
        {"gfx801", oneapi_exp_arch::amd_gpu_gfx801},
        {"gfx802", oneapi_exp_arch::amd_gpu_gfx802},
        {"gfx803", oneapi_exp_arch::amd_gpu_gfx803},
        {"gfx805", oneapi_exp_arch::amd_gpu_gfx805},
        {"gfx810", oneapi_exp_arch::amd_gpu_gfx810},
        {"gfx900", oneapi_exp_arch::amd_gpu_gfx900},
        {"gfx902", oneapi_exp_arch::amd_gpu_gfx902},
        {"gfx904", oneapi_exp_arch::amd_gpu_gfx904},
        {"gfx906", oneapi_exp_arch::amd_gpu_gfx906},
        {"gfx908", oneapi_exp_arch::amd_gpu_gfx908},
        {"gfx909", oneapi_exp_arch::amd_gpu_gfx909},
        {"gfx90a", oneapi_exp_arch::amd_gpu_gfx90a},
        {"gfx90c", oneapi_exp_arch::amd_gpu_gfx90c},
        {"gfx940", oneapi_exp_arch::amd_gpu_gfx940},
        {"gfx941", oneapi_exp_arch::amd_gpu_gfx941},
        {"gfx942", oneapi_exp_arch::amd_gpu_gfx942},
        {"gfx1010", oneapi_exp_arch::amd_gpu_gfx1010},
        {"gfx1011", oneapi_exp_arch::amd_gpu_gfx1011},
        {"gfx1012", oneapi_exp_arch::amd_gpu_gfx1012},
        {"gfx1013", oneapi_exp_arch::amd_gpu_gfx1013},
        {"gfx1030", oneapi_exp_arch::amd_gpu_gfx1030},
        {"gfx1031", oneapi_exp_arch::amd_gpu_gfx1031},
        {"gfx1032", oneapi_exp_arch::amd_gpu_gfx1032},
        {"gfx1033", oneapi_exp_arch::amd_gpu_gfx1033},
        {"gfx1034", oneapi_exp_arch::amd_gpu_gfx1034},
        {"gfx1035", oneapi_exp_arch::amd_gpu_gfx1035},
        {"gfx1036", oneapi_exp_arch::amd_gpu_gfx1036},
        {"gfx1100", oneapi_exp_arch::amd_gpu_gfx1100},
        {"gfx1101", oneapi_exp_arch::amd_gpu_gfx1101},
        {"gfx1102", oneapi_exp_arch::amd_gpu_gfx1102},
        {"gfx1103", oneapi_exp_arch::amd_gpu_gfx1103},
        {"gfx1150", oneapi_exp_arch::amd_gpu_gfx1150},
        {"gfx1151", oneapi_exp_arch::amd_gpu_gfx1151},
        {"gfx1200", oneapi_exp_arch::amd_gpu_gfx1200},
        {"gfx1201", oneapi_exp_arch::amd_gpu_gfx1201},
};

// Only for Intel GPU architectures
constexpr std::pair<const int, oneapi_exp_arch> IntelGPUArchitectures[] = {
    {0x02000000, oneapi_exp_arch::intel_gpu_bdw},
    {0x02400009, oneapi_exp_arch::intel_gpu_skl},
    {0x02404009, oneapi_exp_arch::intel_gpu_kbl},
    {0x02408009, oneapi_exp_arch::intel_gpu_cfl},
    {0x0240c000, oneapi_exp_arch::intel_gpu_apl},
    {0x02410000, oneapi_exp_arch::intel_gpu_glk},
    {0x02414000, oneapi_exp_arch::intel_gpu_whl},
    {0x02418000, oneapi_exp_arch::intel_gpu_aml},
    {0x0241c000, oneapi_exp_arch::intel_gpu_cml},
    {0x02c00000, oneapi_exp_arch::intel_gpu_icllp},
    {0x02c08000, oneapi_exp_arch::intel_gpu_ehl},
    {0x03000000, oneapi_exp_arch::intel_gpu_tgllp},
    {0x03004000, oneapi_exp_arch::intel_gpu_rkl},
    {0x03008000, oneapi_exp_arch::intel_gpu_adl_s},
    {0x0300c000, oneapi_exp_arch::intel_gpu_adl_p},
    {0x03010000, oneapi_exp_arch::intel_gpu_adl_n},
    {0x03028000, oneapi_exp_arch::intel_gpu_dg1},
    {0x030dc000, oneapi_exp_arch::intel_gpu_acm_g10}, // A0
    {0x030dc001, oneapi_exp_arch::intel_gpu_acm_g10}, // A1
    {0x030dc004, oneapi_exp_arch::intel_gpu_acm_g10}, // B0
    {0x030dc008, oneapi_exp_arch::intel_gpu_acm_g10}, // C0
    {0x030e0000, oneapi_exp_arch::intel_gpu_acm_g11}, // A0
    {0x030e0004, oneapi_exp_arch::intel_gpu_acm_g11}, // B0
    {0x030e0005, oneapi_exp_arch::intel_gpu_acm_g11}, // B1
    {0x030e4000, oneapi_exp_arch::intel_gpu_acm_g12}, // A0
    {0x030f0000, oneapi_exp_arch::intel_gpu_pvc},     // XL-A0
    {0x030f0001, oneapi_exp_arch::intel_gpu_pvc},     // XL-AOP
    {0x030f0003, oneapi_exp_arch::intel_gpu_pvc},     // XT-A0
    {0x030f0005, oneapi_exp_arch::intel_gpu_pvc},     // XT-B0
    {0x030f0006, oneapi_exp_arch::intel_gpu_pvc},     // XT-B1
    {0x030f0007, oneapi_exp_arch::intel_gpu_pvc},     // XT-C0
    {0x030f4007, oneapi_exp_arch::intel_gpu_pvc_vg},  // C0
    {0x03118000, oneapi_exp_arch::intel_gpu_mtl_u},   // A0
    {0x03118004, oneapi_exp_arch::intel_gpu_mtl_u},   // B0
    {0x0311c000, oneapi_exp_arch::intel_gpu_mtl_h},   // A0
    {0x0311c004, oneapi_exp_arch::intel_gpu_mtl_h},   // B0
    {0x03128000, oneapi_exp_arch::intel_gpu_arl_h},   // A0
    {0x03128004, oneapi_exp_arch::intel_gpu_arl_h},   // B0
    {0x05004000, oneapi_exp_arch::intel_gpu_bmg_g21}, // A0
    {0x05004001, oneapi_exp_arch::intel_gpu_bmg_g21}, // A1
    {0x05004004, oneapi_exp_arch::intel_gpu_bmg_g21}, // B0
    {0x05010000, oneapi_exp_arch::intel_gpu_lnl_m},   // A0
    {0x05010001, oneapi_exp_arch::intel_gpu_lnl_m},   // A1
    {0x05010004, oneapi_exp_arch::intel_gpu_lnl_m},   // B0
};

// Only for Intel CPU architectures
constexpr std::pair<const int, oneapi_exp_arch> IntelCPUArchitectures[] = {
    {8, oneapi_exp_arch::intel_cpu_spr},
    {9, oneapi_exp_arch::intel_cpu_gnr},
};

template <>
struct get_device_info_impl<
    ext::oneapi::experimental::architecture,
    ext::oneapi::experimental::info::device::architecture> {
  static ext::oneapi::experimental::architecture get(const DeviceImplPtr &Dev) {
    backend CurrentBackend = Dev->getBackend();
    auto LookupIPVersion = [&](auto &ArchList)
        -> std::optional<ext::oneapi::experimental::architecture> {
      uint32_t DeviceIp;
      ur_result_t Err = Dev->getPlugin()->call_nocheck(
          urDeviceGetInfo, Dev->getHandleRef(),
          UrInfoCode<
              ext::oneapi::experimental::info::device::architecture>::value,
          sizeof(DeviceIp), &DeviceIp, nullptr);
      if (Err == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
        // Not all devices support this device info query
        return std::nullopt;
      }
      Dev->getPlugin()->checkUrResult(Err);

      for (const auto &Item : ArchList) {
        if (Item.first == static_cast<int>(DeviceIp))
          return Item.second;
      }
      return std::nullopt;
    };

    if (Dev->is_gpu() && (backend::ext_oneapi_level_zero == CurrentBackend ||
                          backend::opencl == CurrentBackend)) {
      return LookupIPVersion(IntelGPUArchitectures)
          .value_or(ext::oneapi::experimental::architecture::unknown);
    } else if (Dev->is_gpu() && (backend::ext_oneapi_cuda == CurrentBackend ||
                                 backend::ext_oneapi_hip == CurrentBackend)) {
      auto MapArchIDToArchName = [](const char *arch) {
        for (const auto &Item : NvidiaAmdGPUArchitectures) {
          if (std::string_view(Item.first) == arch)
            return Item.second;
        }
        return ext::oneapi::experimental::architecture::unknown;
      };
      size_t ResultSize = 0;
      Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                             UrInfoCode<info::device::version>::value, 0,
                             nullptr, &ResultSize);
      std::unique_ptr<char[]> DeviceArch(new char[ResultSize]);
      Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                             UrInfoCode<info::device::version>::value,
                             ResultSize, DeviceArch.get(), nullptr);
      std::string DeviceArchCopy(DeviceArch.get());
      std::string DeviceArchSubstr =
          DeviceArchCopy.substr(0, DeviceArchCopy.find(":"));
      return MapArchIDToArchName(DeviceArchSubstr.data());
    } else if (Dev->is_cpu() && backend::opencl == CurrentBackend) {
      return LookupIPVersion(IntelCPUArchitectures)
          .value_or(ext::oneapi::experimental::architecture::x86_64);
    } // else is not needed
    // TODO: add support of other architectures by extending with else if
    return ext::oneapi::experimental::architecture::unknown;
  }
};

template <>
struct get_device_info_impl<
    std::vector<ext::oneapi::experimental::matrix::combination>,
    ext::oneapi::experimental::info::device::matrix_combinations> {
  static std::vector<ext::oneapi::experimental::matrix::combination>
  get(const DeviceImplPtr &Dev) {
    using namespace ext::oneapi::experimental::matrix;
    using namespace ext::oneapi::experimental;
    backend CurrentBackend = Dev->getBackend();
    auto get_current_architecture = [&Dev]() -> std::optional<architecture> {
      // this helper lambda ignores all runtime-related exceptions from
      // quering the device architecture. For instance, if device architecture
      // on user's machine is not supported by
      // sycl_ext_oneapi_device_architecture, the runtime exception is omitted,
      // and std::nullopt is returned.
      try {
        return get_device_info_impl<
            architecture,
            ext::oneapi::experimental::info::device::architecture>::get(Dev);
      } catch (sycl::exception &e) {
        if (e.code() != errc::runtime)
          std::rethrow_exception(std::make_exception_ptr(e));
      }
      return std::nullopt;
    };
    std::optional<architecture> DeviceArchOpt = get_current_architecture();
    if (!DeviceArchOpt.has_value())
      return {};
    architecture DeviceArch = DeviceArchOpt.value();
    if (architecture::intel_cpu_spr == DeviceArch)
      return {
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::intel_cpu_gnr == DeviceArch)
      return {
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {16, 16, 32, 0, 0, 0, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::intel_gpu_pvc == DeviceArch)
      return {
          {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {8, 0, 0, 0, 16, 8, matrix_type::tf32, matrix_type::tf32,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if ((architecture::intel_gpu_dg2_g10 == DeviceArch) ||
             (architecture::intel_gpu_dg2_g11 == DeviceArch) ||
             (architecture::intel_gpu_dg2_g12 == DeviceArch))
      return {
          {8, 0, 0, 0, 8, 32, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {8, 0, 0, 0, 8, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 32, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::amd_gpu_gfx90a == DeviceArch)
      return {
          {0, 0, 0, 32, 32, 8, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 32, 8, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 32, 8, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 4, matrix_type::fp64, matrix_type::fp64,
           matrix_type::fp64, matrix_type::fp64},
      };
    else if (backend::ext_oneapi_cuda == CurrentBackend) {
      // TODO: Tho following can be simplified when comparison of architectures
      // using < and > will be implemented
      using oneapi_exp_arch = sycl::ext::oneapi::experimental::architecture;
      constexpr std::pair<float, oneapi_exp_arch> NvidiaArchNumbs[] = {
          {5.0, oneapi_exp_arch::nvidia_gpu_sm_50},
          {5.2, oneapi_exp_arch::nvidia_gpu_sm_52},
          {5.3, oneapi_exp_arch::nvidia_gpu_sm_53},
          {6.0, oneapi_exp_arch::nvidia_gpu_sm_60},
          {6.1, oneapi_exp_arch::nvidia_gpu_sm_61},
          {6.2, oneapi_exp_arch::nvidia_gpu_sm_62},
          {7.0, oneapi_exp_arch::nvidia_gpu_sm_70},
          {7.2, oneapi_exp_arch::nvidia_gpu_sm_72},
          {7.5, oneapi_exp_arch::nvidia_gpu_sm_75},
          {8.0, oneapi_exp_arch::nvidia_gpu_sm_80},
          {8.6, oneapi_exp_arch::nvidia_gpu_sm_86},
          {8.7, oneapi_exp_arch::nvidia_gpu_sm_87},
          {8.9, oneapi_exp_arch::nvidia_gpu_sm_89},
          {9.0, oneapi_exp_arch::nvidia_gpu_sm_90},
      };
      auto GetArchNum = [&](const architecture &arch) {
        for (const auto &Item : NvidiaArchNumbs)
          if (Item.second == arch)
            return Item.first;
        return 0.f;
      };
      float ComputeCapability = GetArchNum(DeviceArch);
      std::vector<combination> sm_70_combinations = {
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32}};
      std::vector<combination> sm_72_combinations = {
          {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 8, 32, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 8, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 16, 16, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 8, 32, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 8, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32}};
      std::vector<combination> sm_80_combinations = {
          {0, 0, 0, 16, 16, 8, matrix_type::tf32, matrix_type::tf32,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 8, 4, matrix_type::fp64, matrix_type::fp64,
           matrix_type::fp64, matrix_type::fp64}};
      if (ComputeCapability >= 8.0) {
        sm_80_combinations.insert(sm_80_combinations.end(),
                                  sm_72_combinations.begin(),
                                  sm_72_combinations.end());
        sm_80_combinations.insert(sm_80_combinations.end(),
                                  sm_70_combinations.begin(),
                                  sm_70_combinations.end());
        return sm_80_combinations;
      } else if (ComputeCapability >= 7.2) {
        sm_72_combinations.insert(sm_72_combinations.end(),
                                  sm_70_combinations.begin(),
                                  sm_70_combinations.end());
        return sm_72_combinations;
      } else if (ComputeCapability >= 7.0)
        return sm_70_combinations;
    }
    return {};
  }
};

template <>
struct get_device_info_impl<
    size_t, ext::oneapi::experimental::info::device::max_global_work_groups> {
  static size_t get(const DeviceImplPtr) {
    return static_cast<size_t>((std::numeric_limits<int>::max)());
  }
};
template <>
struct get_device_info_impl<
    id<1>, ext::oneapi::experimental::info::device::max_work_groups<1>> {
  static id<1> get(const DeviceImplPtr &Dev) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(Dev);
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
            ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
        sizeof(result), &result, nullptr);
    return id<1>(std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info_impl<
    id<2>, ext::oneapi::experimental::info::device::max_work_groups<2>> {
  static id<2> get(const DeviceImplPtr &Dev) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(Dev);
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
            ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
        sizeof(result), &result, nullptr);
    return id<2>(std::min(Limit, result[1]), std::min(Limit, result[0]));
  }
};

template <>
struct get_device_info_impl<
    id<3>, ext::oneapi::experimental::info::device::max_work_groups<3>> {
  static id<3> get(const DeviceImplPtr &Dev) {
    size_t result[3];
    size_t Limit =
        get_device_info_impl<size_t, ext::oneapi::experimental::info::device::
                                         max_global_work_groups>::get(Dev);
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
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
  static size_t get(const DeviceImplPtr &Dev) {
    return get_device_info_impl<size_t,
                                ext::oneapi::experimental::info::device::
                                    max_global_work_groups>::get(Dev);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_1d>
template <>
struct get_device_info_impl<id<1>,
                            info::device::ext_oneapi_max_work_groups_1d> {
  static id<1> get(const DeviceImplPtr &Dev) {
    return get_device_info_impl<
        id<1>,
        ext::oneapi::experimental::info::device::max_work_groups<1>>::get(Dev);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_2d>
template <>
struct get_device_info_impl<id<2>,
                            info::device::ext_oneapi_max_work_groups_2d> {
  static id<2> get(const DeviceImplPtr &Dev) {
    return get_device_info_impl<
        id<2>,
        ext::oneapi::experimental::info::device::max_work_groups<2>>::get(Dev);
  }
};

// TODO:Remove with deprecated feature
// device::get_info<info::device::ext_oneapi_max_work_groups_3d>
template <>
struct get_device_info_impl<id<3>,
                            info::device::ext_oneapi_max_work_groups_3d> {
  static id<3> get(const DeviceImplPtr &Dev) {
    return get_device_info_impl<
        id<3>,
        ext::oneapi::experimental::info::device::max_work_groups<3>>::get(Dev);
  }
};

// Specialization for parent device
template <> struct get_device_info_impl<device, info::device::parent_device> {
  static device get(const DeviceImplPtr &Dev) {
    typename sycl_to_ur<device>::type result;
    Dev->getPlugin()->call(urDeviceGetInfo, Dev->getHandleRef(),
                           UrInfoCode<info::device::parent_device>::value,
                           sizeof(result), &result, nullptr);
    if (result == nullptr)
      throw exception(make_error_code(errc::invalid),
                      "No parent for device because it is not a subdevice");

    const auto &Platform = Dev->getPlatformImpl();
    return createSyclObjFromImpl<device>(
        Platform->getOrMakeDeviceImpl(result, Platform));
  }
};

// Specialization for image_support
template <> struct get_device_info_impl<bool, info::device::image_support> {
  static bool get(const DeviceImplPtr &) {
    // No devices currently support SYCL 2020 images.
    return false;
  }
};

// USM

// Specialization for device usm query.

template <>
struct get_device_info_impl<bool, info::device::usm_device_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_device_usm_access_capability_flags_t caps;
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::usm_device_allocations>::value,
        sizeof(ur_device_usm_access_capability_flags_t), &caps, nullptr);

    return (Err != UR_RESULT_SUCCESS)
               ? false
               : (caps & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  }
};

// Specialization for host usm query.

template <>
struct get_device_info_impl<bool, info::device::usm_host_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_device_usm_access_capability_flags_t caps;
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::usm_host_allocations>::value,
        sizeof(ur_device_usm_access_capability_flags_t), &caps, nullptr);

    return (Err != UR_RESULT_SUCCESS)
               ? false
               : (caps & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  }
};

// Specialization for shared usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_shared_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_device_usm_access_capability_flags_t caps;
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::usm_shared_allocations>::value,
        sizeof(ur_device_usm_access_capability_flags_t), &caps, nullptr);
    return (Err != UR_RESULT_SUCCESS)
               ? false
               : (caps & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  }
};

// Specialization for restricted usm query
template <>
struct get_device_info_impl<bool,
                            info::device::usm_restricted_shared_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_device_usm_access_capability_flags_t caps;
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::usm_restricted_shared_allocations>::value,
        sizeof(ur_device_usm_access_capability_flags_t), &caps, nullptr);
    // Check that we don't support any cross device sharing
    return (Err != UR_RESULT_SUCCESS)
               ? false
               : !(caps &
                   (UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
                    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS));
  }
};

// Specialization for system usm query
template <>
struct get_device_info_impl<bool, info::device::usm_system_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    ur_device_usm_access_capability_flags_t caps;
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<info::device::usm_system_allocations>::value,
        sizeof(ur_device_usm_access_capability_flags_t), &caps, nullptr);
    return (Err != UR_RESULT_SUCCESS)
               ? false
               : (caps & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  }
};

// Specialization for kernel fusion support
template <>
struct get_device_info_impl<
    bool, ext::codeplay::experimental::info::device::supports_fusion> {
  static bool get(const DeviceImplPtr &Dev) {
#if SYCL_EXT_CODEPLAY_KERNEL_FUSION
    // If the JIT library can't be loaded or entry points in the JIT library
    // can't be resolved, fusion is not available.
    if (!jit_compiler::get_instance().isAvailable()) {
      return false;
    }
    // Currently fusion is only supported for SPIR-V based backends,
    // CUDA and HIP.
    if (Dev->getBackend() == backend::opencl) {
      // Exclude all non-CPU or non-GPU devices on OpenCL, in particular
      // accelerators.
      return Dev->is_cpu() || Dev->is_gpu();
    }

    return (Dev->getBackend() == backend::ext_oneapi_level_zero) ||
           (Dev->getBackend() == backend::ext_oneapi_cuda) ||
           (Dev->getBackend() == backend::ext_oneapi_hip);
#else  // SYCL_EXT_CODEPLAY_KERNEL_FUSION
    (void)Dev;
    return false;
#endif // SYCL_EXT_CODEPLAY_KERNEL_FUSION
  }
};

// Specialization for max registers per work-group
template <>
struct get_device_info_impl<
    uint32_t,
    ext::codeplay::experimental::info::device::max_registers_per_work_group> {
  static uint32_t get(const DeviceImplPtr &Dev) {
    uint32_t maxRegsPerWG;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<ext::codeplay::experimental::info::device::
                       max_registers_per_work_group>::value,
        sizeof(maxRegsPerWG), &maxRegsPerWG, nullptr);
    return maxRegsPerWG;
  }
};

// Specialization for composite devices extension.
template <>
struct get_device_info_impl<
    std::vector<sycl::device>,
    ext::oneapi::experimental::info::device::component_devices> {
  static std::vector<sycl::device> get(const DeviceImplPtr &Dev) {
    size_t ResultSize = 0;
    // First call to get DevCount.
    ur_result_t Err = Dev->getPlugin()->call_nocheck(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
            ext::oneapi::experimental::info::device::component_devices>::value,
        0, nullptr, &ResultSize);

    // If the feature is unsupported or if the result was empty, return an empty
    // list of devices.
    if (Err == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION ||
        (Err == UR_RESULT_SUCCESS && ResultSize == 0))
      return {};

    // Otherwise, if there was an error from UR it is unexpected and we should
    // handle it accordingly.
    Dev->getPlugin()->checkUrResult(Err);

    size_t DevCount = ResultSize / sizeof(ur_device_handle_t);

    // Second call to get the list.
    std::vector<ur_device_handle_t> Devs(DevCount);
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
            ext::oneapi::experimental::info::device::component_devices>::value,
        ResultSize, Devs.data(), nullptr);
    std::vector<sycl::device> Result;
    const auto &Platform = Dev->getPlatformImpl();
    for (const auto &d : Devs)
      Result.push_back(createSyclObjFromImpl<device>(
          Platform->getOrMakeDeviceImpl(d, Platform)));

    return Result;
  }
};

template <>
struct get_device_info_impl<
    sycl::device, ext::oneapi::experimental::info::device::composite_device> {
  static sycl::device get(const DeviceImplPtr &Dev) {
    if (!Dev->has(sycl::aspect::ext_oneapi_is_component))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Only devices with aspect::ext_oneapi_is_component "
                            "can call this function.");

    typename sycl_to_ur<device>::type Result;
    Dev->getPlugin()->call(
        urDeviceGetInfo, Dev->getHandleRef(),
        UrInfoCode<
            ext::oneapi::experimental::info::device::composite_device>::value,
        sizeof(Result), &Result, nullptr);

    if (Result) {
      const auto &Platform = Dev->getPlatformImpl();
      return createSyclObjFromImpl<device>(
          Platform->getOrMakeDeviceImpl(Result, Platform));
    }
    throw sycl::exception(make_error_code(errc::invalid),
                          "A component with aspect::ext_oneapi_is_component "
                          "must have a composite device.");
  }
};

template <typename Param>
typename Param::return_type get_device_info(const DeviceImplPtr &Dev) {
  static_assert(is_device_info_desc<Param>::value,
                "Invalid device information descriptor");
  return get_device_info_impl<typename Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::device_id::return_type
get_device_info<ext::intel::info::device::device_id>(const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_device_id))
    throw exception(make_error_code(errc::feature_not_supported),
                    "The device does not have the ext_intel_device_id aspect");
  using Param = ext::intel::info::device::device_id;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::uuid::return_type
get_device_info<ext::intel::info::device::uuid>(const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_device_info_uuid))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_device_info_uuid aspect");
  using Param = ext::intel::info::device::uuid;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::pci_address::return_type
get_device_info<ext::intel::info::device::pci_address>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_pci_address))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_pci_address aspect");
  using Param = ext::intel::info::device::pci_address;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_eu_simd_width::return_type
get_device_info<ext::intel::info::device::gpu_eu_simd_width>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_eu_simd_width))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_gpu_eu_simd_width aspect");
  using Param = ext::intel::info::device::gpu_eu_simd_width;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_eu_count::return_type
get_device_info<ext::intel::info::device::gpu_eu_count>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_eu_count))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_gpu_eu_count aspect");
  using Param = ext::intel::info::device::gpu_eu_count;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_slices::return_type
get_device_info<ext::intel::info::device::gpu_slices>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_slices))
    throw exception(make_error_code(errc::feature_not_supported),
                    "The device does not have the ext_intel_gpu_slices aspect");
  using Param = ext::intel::info::device::gpu_slices;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_subslices_per_slice::return_type
get_device_info<ext::intel::info::device::gpu_subslices_per_slice>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_subslices_per_slice))
    throw exception(make_error_code(errc::feature_not_supported),
                    "The device does not have the "
                    "ext_intel_gpu_subslices_per_slice aspect");
  using Param = ext::intel::info::device::gpu_subslices_per_slice;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_eu_count_per_subslice::return_type
get_device_info<ext::intel::info::device::gpu_eu_count_per_subslice>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_eu_count_per_subslice))
    throw exception(make_error_code(errc::feature_not_supported),
                    "The device does not have the "
                    "ext_intel_gpu_eu_count_per_subslice aspect");
  using Param = ext::intel::info::device::gpu_eu_count_per_subslice;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::gpu_hw_threads_per_eu::return_type
get_device_info<ext::intel::info::device::gpu_hw_threads_per_eu>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_gpu_hw_threads_per_eu))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_gpu_hw_threads_per_eu aspect");
  using Param = ext::intel::info::device::gpu_hw_threads_per_eu;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::max_mem_bandwidth::return_type
get_device_info<ext::intel::info::device::max_mem_bandwidth>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_max_mem_bandwidth))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_max_mem_bandwidth aspect");
  using Param = ext::intel::info::device::max_mem_bandwidth;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::free_memory::return_type
get_device_info<ext::intel::info::device::free_memory>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_free_memory))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_free_memory aspect");
  using Param = ext::intel::info::device::free_memory;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::memory_clock_rate::return_type
get_device_info<ext::intel::info::device::memory_clock_rate>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_memory_clock_rate))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_memory_clock_rate aspect");
  using Param = ext::intel::info::device::memory_clock_rate;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

template <>
inline ext::intel::info::device::memory_bus_width::return_type
get_device_info<ext::intel::info::device::memory_bus_width>(
    const DeviceImplPtr &Dev) {
  if (!Dev->has(aspect::ext_intel_memory_bus_width))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "The device does not have the ext_intel_memory_bus_width aspect");
  using Param = ext::intel::info::device::memory_bus_width;
  return get_device_info_impl<Param::return_type, Param>::get(Dev);
}

// Returns the list of all progress guarantees that can be requested for
// work_groups from the coordination level of root_group when using the device
// given by Dev. First it calls getProgressGuarantee to get the strongest
// guarantee available and then calls getProgressGuaranteesUpTo to get a list of
// all guarantees that are either equal to the strongest guarantee or weaker
// than it. The next 5 definitions follow the same model but for different
// scopes.
template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::work_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {
    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::work_group,
                                  execution_scope::root_group));
  }
};
template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {
    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::sub_group,
                                  execution_scope::root_group));
  }
};

template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {

    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::sub_group,
                                  execution_scope::work_group));
  }
};

template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {

    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::work_item,
                                  execution_scope::root_group));
  }
};
template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {

    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::work_item,
                                  execution_scope::work_group));
  }
};

template <typename ReturnT>
struct get_device_info_impl<
    ReturnT,
    ext::oneapi::experimental::info::device::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::sub_group>> {
  static ReturnT get(const DeviceImplPtr &Dev) {

    using execution_scope = ext::oneapi::experimental::execution_scope;
    return device_impl::getProgressGuaranteesUpTo<ReturnT>(
        Dev->getProgressGuarantee(execution_scope::work_item,
                                  execution_scope::sub_group));
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
