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
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/platform.hpp>

#include <chrono>
#include <thread>

namespace sycl {
inline namespace _V1 {
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
  using type = sycl::detail::pi::PiDevice;
};
template <> struct sycl_to_pi<platform> {
  using type = sycl::detail::pi::PiPlatform;
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
  static ReturnT get(const DeviceImplPtr &Dev) {
    typename sycl_to_pi<ReturnT>::type result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<Param>::value, sizeof(result), &result,
        nullptr);
    return ReturnT(result);
  }
};

// Specialization for platform
template <typename Param> struct get_device_info_impl<platform, Param> {
  static platform get(const DeviceImplPtr &Dev) {
    typename sycl_to_pi<platform>::type result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<Param>::value, sizeof(result), &result,
        nullptr);
    // TODO: Change PiDevice to device_impl.
    // Use the Plugin from the device_impl class after plugin details
    // are added to the class.
    return createSyclObjFromImpl<platform>(
        platform_impl::getOrMakePlatformImpl(result, Dev->getPlugin()));
  }
};

// Helper function to allow using the specialization of get_device_info_impl
// for string return type in other specializations.
inline std::string device_impl::get_device_info_string(
    sycl::detail::pi::PiDeviceInfo InfoCode) const {
  size_t resultSize = 0;
  getPlugin()->call<PiApiKind::piDeviceGetInfo>(getHandleRef(), InfoCode, 0,
                                                nullptr, &resultSize);
  if (resultSize == 0) {
    return std::string();
  }
  std::unique_ptr<char[]> result(new char[resultSize]);
  getPlugin()->call<PiApiKind::piDeviceGetInfo>(
      getHandleRef(), InfoCode, resultSize, result.get(), nullptr);

  return std::string(result.get());
}

// Specialization for string return type, variable return size
template <typename Param> struct get_device_info_impl<std::string, Param> {
  static std::string get(const DeviceImplPtr &Dev) {
    return Dev->get_device_info_string(PiInfoCode<Param>::value);
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
    cl_device_fp_config result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<Param>::value, sizeof(result), &result,
        nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for device version
template <> struct get_device_info_impl<std::string, info::device::version> {
  static std::string get(const DeviceImplPtr &Dev) {
    return Dev->get_device_info_string(
        PiInfoCode<info::device::version>::value);
  }
};

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info_impl<std::vector<info::fp_config>,
                            info::device::single_fp_config> {
  static std::vector<info::fp_config> get(const DeviceImplPtr &Dev) {
    pi_device_fp_config result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<info::device::single_fp_config>::value,
        sizeof(result), &result, nullptr);
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling. In addition to pi_queue level profiling,
// piGetDeviceAndHostTimer is not supported, command_submit, command_start,
// command_end will be calculated. See MFallbackProfiling
template <> struct get_device_info_impl<bool, info::device::queue_profiling> {
  static bool get(const DeviceImplPtr &Dev) {
    pi_queue_properties Properties;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<info::device::queue_profiling>::value,
        sizeof(Properties), &Properties, nullptr);
    return Properties & PI_QUEUE_FLAG_PROFILING_ENABLE;
  }
};

// Specialization for atomic_memory_order_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_order>,
                            info::device::atomic_memory_order_capabilities> {
  static std::vector<memory_order> get(const DeviceImplPtr &Dev) {
    pi_memory_order_capabilities result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::atomic_memory_order_capabilities>::value,
        sizeof(pi_memory_order_capabilities), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_fence_order_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_order>,
                            info::device::atomic_fence_order_capabilities> {
  static std::vector<memory_order> get(const DeviceImplPtr &Dev) {
    pi_memory_order_capabilities result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::atomic_fence_order_capabilities>::value,
        sizeof(pi_memory_order_capabilities), &result, nullptr);
    return readMemoryOrderBitfield(result);
  }
};

// Specialization for atomic_memory_scope_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_scope>,
                            info::device::atomic_memory_scope_capabilities> {
  static std::vector<memory_scope> get(const DeviceImplPtr &Dev) {
    pi_memory_scope_capabilities result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::atomic_memory_scope_capabilities>::value,
        sizeof(pi_memory_scope_capabilities), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for atomic_fence_scope_capabilities, PI returns a bitfield
template <>
struct get_device_info_impl<std::vector<memory_scope>,
                            info::device::atomic_fence_scope_capabilities> {
  static std::vector<memory_scope> get(const DeviceImplPtr &Dev) {
    pi_memory_scope_capabilities result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::atomic_fence_scope_capabilities>::value,
        sizeof(pi_memory_scope_capabilities), &result, nullptr);
    return readMemoryScopeBitfield(result);
  }
};

// Specialization for bf16 math functions
template <>
struct get_device_info_impl<bool,
                            info::device::ext_oneapi_bfloat16_math_functions> {
  static bool get(const DeviceImplPtr &Dev) {
    bool result = false;

    sycl::detail::pi::PiResult Err =
        Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
            Dev->getHandleRef(),
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
  static std::vector<info::execution_capability> get(const DeviceImplPtr &Dev) {
    pi_device_exec_capabilities result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::execution_capabilities>::value, sizeof(result),
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
        PiInfoCode<info::device::built_in_kernels>::value);
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
        PiInfoCode<info::device::built_in_kernels>::value);
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
    auto info_partition = PiInfoCode<info::device::partition_properties>::value;
    const auto &Plugin = Dev->getPlugin();

    size_t resultSize;
    Plugin->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), info_partition, 0, nullptr, &resultSize);

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);
    if (arrayLength == 0) {
      return {};
    }
    std::unique_ptr<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    Plugin->call<PiApiKind::piDeviceGetInfo>(Dev->getHandleRef(),
                                             info_partition, resultSize,
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
  get(const DeviceImplPtr &Dev) {
    pi_device_affinity_domain result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::partition_affinity_domains>::value,
        sizeof(result), &result, nullptr);
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, OpenCL can return other
// partition properties instead
template <>
struct get_device_info_impl<info::partition_affinity_domain,
                            info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain get(const DeviceImplPtr &Dev) {
    size_t resultSize;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::partition_type_affinity_domain>::value, 0,
        nullptr, &resultSize);
    if (resultSize != 1) {
      return info::partition_affinity_domain::not_applicable;
    }
    cl_device_partition_property result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::partition_type_affinity_domain>::value,
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
  static info::partition_property get(const DeviceImplPtr &Dev) {
    size_t resultSize;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PI_DEVICE_INFO_PARTITION_TYPE, 0, nullptr,
        &resultSize);
    if (!resultSize)
      return info::partition_property::no_partition;

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);

    std::unique_ptr<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PI_DEVICE_INFO_PARTITION_TYPE, resultSize,
        arrayResult.get(), nullptr);
    if (!arrayResult[0])
      return info::partition_property::no_partition;
    return info::partition_property(arrayResult[0]);
  }
};
// Specialization for supported subgroup sizes
template <>
struct get_device_info_impl<std::vector<size_t>,
                            info::device::sub_group_sizes> {
  static std::vector<size_t> get(const DeviceImplPtr &Dev) {
    size_t resultSize = 0;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<info::device::sub_group_sizes>::value,
        0, nullptr, &resultSize);

    std::vector<size_t> result(resultSize / sizeof(size_t));
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<info::device::sub_group_sizes>::value,
        resultSize, result.data(), nullptr);
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
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::max_work_item_sizes<Dimensions>>::value,
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
        {"gfx90a", oneapi_exp_arch::amd_gpu_gfx90a},
        {"gfx1010", oneapi_exp_arch::amd_gpu_gfx1010},
        {"gfx1011", oneapi_exp_arch::amd_gpu_gfx1011},
        {"gfx1012", oneapi_exp_arch::amd_gpu_gfx1012},
        {"gfx1013", oneapi_exp_arch::amd_gpu_gfx1013},
        {"gfx1030", oneapi_exp_arch::amd_gpu_gfx1030},
        {"gfx1031", oneapi_exp_arch::amd_gpu_gfx1031},
        {"gfx1032", oneapi_exp_arch::amd_gpu_gfx1032},
        {"gfx1034", oneapi_exp_arch::amd_gpu_gfx1034},
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
    {0x030dc008, oneapi_exp_arch::intel_gpu_acm_g10},
    {0x030e0005, oneapi_exp_arch::intel_gpu_acm_g11},
    {0x030e4000, oneapi_exp_arch::intel_gpu_acm_g12},
    {0x030f0007, oneapi_exp_arch::intel_gpu_pvc},
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
    if (Dev->is_gpu() && (backend::ext_oneapi_level_zero == CurrentBackend ||
                          backend::opencl == CurrentBackend)) {
      auto MapArchIDToArchName = [](const int arch) {
        for (const auto &Item : IntelGPUArchitectures) {
          if (Item.first == arch)
            return Item.second;
        }
        throw sycl::exception(
            make_error_code(errc::runtime),
            "The current device architecture is not supported by "
            "sycl_ext_oneapi_device_architecture.");
      };
      uint32_t DeviceIp;
      Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
          Dev->getHandleRef(),
          PiInfoCode<
              ext::oneapi::experimental::info::device::architecture>::value,
          sizeof(DeviceIp), &DeviceIp, nullptr);
      return MapArchIDToArchName(DeviceIp);
    } else if (Dev->is_gpu() && (backend::ext_oneapi_cuda == CurrentBackend ||
                                 backend::ext_oneapi_hip == CurrentBackend)) {
      auto MapArchIDToArchName = [](const char *arch) {
        for (const auto &Item : NvidiaAmdGPUArchitectures) {
          if (std::string_view(Item.first) == arch)
            return Item.second;
        }
        throw sycl::exception(
            make_error_code(errc::runtime),
            "The current device architecture is not supported by "
            "sycl_ext_oneapi_device_architecture.");
      };
      size_t ResultSize = 0;
      Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
          Dev->getHandleRef(), PiInfoCode<info::device::version>::value, 0,
          nullptr, &ResultSize);
      std::unique_ptr<char[]> DeviceArch(new char[ResultSize]);
      Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
          Dev->getHandleRef(), PiInfoCode<info::device::version>::value,
          ResultSize, DeviceArch.get(), nullptr);
      std::string DeviceArchCopy(DeviceArch.get());
      std::string DeviceArchSubstr =
          DeviceArchCopy.substr(0, DeviceArchCopy.find(":"));
      return MapArchIDToArchName(DeviceArchSubstr.data());
    } else if (Dev->is_cpu() && backend::opencl == CurrentBackend) {
      auto MapArchIDToArchName = [](const int arch) {
        for (const auto &Item : IntelCPUArchitectures) {
          if (Item.first == arch)
            return Item.second;
        }
        return sycl::ext::oneapi::experimental::architecture::x86_64;
      };
      uint32_t DeviceIp;
      Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
          Dev->getHandleRef(),
          PiInfoCode<
              ext::oneapi::experimental::info::device::architecture>::value,
          sizeof(DeviceIp), &DeviceIp, nullptr);
      return MapArchIDToArchName(DeviceIp);
    } // else is not needed
    // TODO: add support of other architectures by extending with else if
    // Generating a user-friendly error message
    std::string DeviceStr;
    if (Dev->is_gpu())
      DeviceStr = "GPU";
    else if (Dev->is_cpu())
      DeviceStr = "CPU";
    else if (Dev->is_accelerator())
      DeviceStr = "accelerator";
    // else if not needed
    std::stringstream ErrorMessage;
    ErrorMessage
        << "sycl_ext_oneapi_device_architecture feature is not supported on "
        << DeviceStr << " device with sycl::backend::" << CurrentBackend
        << " backend.";
    throw sycl::exception(make_error_code(errc::runtime), ErrorMessage.str());
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
    architecture DeviceArch = get_device_info_impl<
        ext::oneapi::experimental::architecture,
        ext::oneapi::experimental::info::device::architecture>::get(Dev);
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
        throw sycl::exception(
            make_error_code(errc::runtime),
            "The current device architecture is not supported by "
            "sycl_ext_oneapi_matrix.");
      };
      float ComputeCapability = GetArchNum(DeviceArch);
      constexpr std::vector<combination> sm_70_combinations = {
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
      constexpr std::vector<combination> sm_72_combinations = {
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
      constexpr std::vector<combination> sm_80_combinations = {
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
        std::move(sm_70_combinations.begin(), sm_70_combinations.end(),
                  std::back_inserter(sm_80_combinations));
        std::move(sm_72_combinations.begin(), sm_72_combinations.end(),
                  std::back_inserter(sm_80_combinations));
        return sm_80_combinations;
      } else if (ComputeCapability >= 7.2) {
        std::move(sm_70_combinations.begin(), sm_70_combinations.end(),
                  std::back_inserter(sm_72_combinations));
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
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<
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
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<
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
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
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
    typename sycl_to_pi<device>::type result;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PiInfoCode<info::device::parent_device>::value,
        sizeof(result), &result, nullptr);
    if (result == nullptr)
      throw invalid_object_error(
          "No parent for device because it is not a subdevice",
          PI_ERROR_INVALID_DEVICE);

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
    pi_usm_capabilities caps;
    pi_result Err = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::usm_device_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for host usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_host_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    pi_usm_capabilities caps;
    pi_result Err = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::usm_host_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);

    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for shared usm query.
template <>
struct get_device_info_impl<bool, info::device::usm_shared_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    pi_usm_capabilities caps;
    pi_result Err = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::usm_shared_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for restricted usm query
template <>
struct get_device_info_impl<bool,
                            info::device::usm_restricted_shared_allocations> {
  static bool get(const DeviceImplPtr &Dev) {
    pi_usm_capabilities caps;
    pi_result Err = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::usm_restricted_shared_allocations>::value,
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
  static bool get(const DeviceImplPtr &Dev) {
    pi_usm_capabilities caps;
    pi_result Err = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<info::device::usm_system_allocations>::value,
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

// Specialization for kernel fusion support
template <>
struct get_device_info_impl<
    bool, ext::codeplay::experimental::info::device::supports_fusion> {
  static bool get(const DeviceImplPtr &Dev) {
#if SYCL_EXT_CODEPLAY_KERNEL_FUSION
    // Currently fusion is only supported for SPIR-V based backends,
    // CUDA and HIP.
    return (Dev->getBackend() == backend::ext_oneapi_level_zero) ||
           (Dev->getBackend() == backend::opencl) ||
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
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(),
        PiInfoCode<ext::codeplay::experimental::info::device::
                       max_registers_per_work_group>::value,
        sizeof(maxRegsPerWG), &maxRegsPerWG, nullptr);
    return maxRegsPerWG;
  }
};

// Specialization for graph extension support
template <>
struct get_device_info_impl<
    ext::oneapi::experimental::graph_support_level,
    ext::oneapi::experimental::info::device::graph_support> {
  static ext::oneapi::experimental::graph_support_level
  get(const DeviceImplPtr &Dev) {
    size_t ResultSize = 0;
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PI_DEVICE_INFO_EXTENSIONS, 0, nullptr,
        &ResultSize);
    if (ResultSize == 0)
      return ext::oneapi::experimental::graph_support_level::unsupported;

    std::unique_ptr<char[]> Result(new char[ResultSize]);
    Dev->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
        Dev->getHandleRef(), PI_DEVICE_INFO_EXTENSIONS, ResultSize,
        Result.get(), nullptr);

    std::string_view ExtensionsString(Result.get());
    bool CmdBufferSupport =
        ExtensionsString.find("ur_exp_command_buffer") != std::string::npos;
    return CmdBufferSupport
               ? ext::oneapi::experimental::graph_support_level::native
               : ext::oneapi::experimental::graph_support_level::unsupported;
  }
};

template <typename Param>
typename Param::return_type get_device_info(const DeviceImplPtr &Dev) {
  static_assert(is_device_info_desc<Param>::value,
                "Invalid device information descriptor");
  if (std::is_same<Param,
                   sycl::_V1::ext::intel::info::device::free_memory>::value) {
    if (!Dev->has(aspect::ext_intel_free_memory))
      throw invalid_object_error(
          "The device does not have the ext_intel_free_memory aspect",
          PI_ERROR_INVALID_DEVICE);
  }
  return get_device_info_impl<typename Param::return_type, Param>::get(Dev);
}

// SYCL host device information

// Default template is disabled, all possible instantiations are
// specified explicitly.
template <typename Param>
inline typename Param::return_type get_device_info_host() = delete;

template <>
inline std::vector<sycl::aspect> get_device_info_host<info::device::aspects>() {
  return std::vector<sycl::aspect>();
}

template <>
inline ext::oneapi::experimental::architecture
get_device_info_host<ext::oneapi::experimental::info::device::architecture>() {
  return ext::oneapi::experimental::architecture::x86_64;
}

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
inline range<1> get_device_info_host<info::device::max_work_item_sizes<1>>() {
  // current value is the required minimum
  return {1};
}

template <>
inline range<2> get_device_info_host<info::device::max_work_item_sizes<2>>() {
  // current value is the required minimum
  return {1, 1};
}

template <>
inline range<3> get_device_info_host<info::device::max_work_item_sizes<3>>() {
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
inline std::vector<memory_order>
get_device_info_host<info::device::atomic_fence_order_capabilities>() {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel};
}

template <>
inline std::vector<memory_scope>
get_device_info_host<info::device::atomic_memory_scope_capabilities>() {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

template <>
inline std::vector<memory_scope>
get_device_info_host<info::device::atomic_fence_scope_capabilities>() {
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
inline uint32_t get_device_info_host<ext::intel::info::device::device_id>() {
  throw runtime_error("Obtaining the device ID is not supported on HOST device",
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

template <>
inline bool get_device_info_host<
    ext::codeplay::experimental::info::device::supports_fusion>() {
  // No support for fusion on the host device.
  return false;
}

template <>
inline uint32_t get_device_info_host<
    ext::codeplay::experimental::info::device::max_registers_per_work_group>() {
  throw runtime_error("Obtaining the maximum number of available registers per "
                      "work-group is not supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline ext::oneapi::experimental::graph_support_level
get_device_info_host<ext::oneapi::experimental::info::device::graph_support>() {
  // No support for graphs on the host device.
  return ext::oneapi::experimental::graph_support_level::unsupported;
}

template <>
inline uint32_t get_device_info_host<
    ext::oneapi::experimental::info::device::image_row_pitch_align>() {
  throw runtime_error("Obtaining image pitch alignment is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t get_device_info_host<
    ext::oneapi::experimental::info::device::max_image_linear_row_pitch>() {
  throw runtime_error("Obtaining max image linear pitch is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline std::vector<ext::oneapi::experimental::matrix::combination>
get_device_info_host<
    ext::oneapi::experimental::info::device::matrix_combinations>() {
  throw runtime_error("Obtaining matrix combinations is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t get_device_info_host<
    ext::oneapi::experimental::info::device::max_image_linear_width>() {
  throw runtime_error("Obtaining max image linear width is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline uint32_t get_device_info_host<
    ext::oneapi::experimental::info::device::max_image_linear_height>() {
  throw runtime_error("Obtaining max image linear height is not "
                      "supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

template <>
inline float get_device_info_host<
    ext::oneapi::experimental::info::device::mipmap_max_anisotropy>() {
  throw runtime_error("Bindless image mipaps are not supported on HOST device",
                      PI_ERROR_INVALID_DEVICE);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
