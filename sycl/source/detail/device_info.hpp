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
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <detail/plugin.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

vector_class<info::fp_config> read_fp_bitfield(cl_device_fp_config bits);

vector_class<info::partition_affinity_domain>
read_domain_bitfield(cl_device_affinity_domain bits);

vector_class<info::execution_capability>
read_execution_bitfield(cl_device_exec_capabilities bits);

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
        std::make_shared<platform_impl>(result, Plugin));
  }
};

// Specialization for string return type, variable return size
template <info::device param> struct get_device_info<string_class, param> {
  static string_class get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(param), 0, nullptr, &resultSize);
    if (resultSize == 0) {
      return string_class();
    }
    unique_ptr_class<char[]> result(new char[resultSize]);
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev,
                                            pi::cast<RT::PiDeviceInfo>(param),
                                            resultSize, result.get(), nullptr);

    return string_class(result.get());
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
struct get_device_info<vector_class<info::fp_config>, param> {
  static vector_class<info::fp_config> get(RT::PiDevice dev,
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

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info<vector_class<info::fp_config>,
                       info::device::single_fp_config> {
  static vector_class<info::fp_config> get(RT::PiDevice dev,
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

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info<vector_class<info::execution_capability>,
                       info::device::execution_capabilities> {
  static vector_class<info::execution_capability> get(RT::PiDevice dev,
                                                      const plugin &Plugin) {
    cl_device_exec_capabilities result;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::execution_capabilities),
        sizeof(result), &result, nullptr);
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernels, splits the string returned by OpenCL
template <>
struct get_device_info<vector_class<string_class>,
                       info::device::built_in_kernels> {
  static vector_class<string_class> get(RT::PiDevice dev,
                                        const plugin &Plugin) {
    string_class result =
        get_device_info<string_class, info::device::built_in_kernels>::get(
            dev, Plugin);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info<vector_class<string_class>, info::device::extensions> {
  static vector_class<string_class> get(RT::PiDevice dev,
                                        const plugin &Plugin) {
    string_class result =
        get_device_info<string_class, info::device::extensions>::get(dev,
                                                                     Plugin);
    return split_string(result, ' ');
  }
};

// Specialization for partition properties, variable OpenCL return size
template <>
struct get_device_info<vector_class<info::partition_property>,
                       info::device::partition_properties> {
  static vector_class<info::partition_property> get(RT::PiDevice dev,
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
    unique_ptr_class<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    Plugin.call<PiApiKind::piDeviceGetInfo>(dev, info_partition, resultSize,
                                            arrayResult.get(), nullptr);

    vector_class<info::partition_property> result;
    for (size_t i = 0; i < arrayLength - 1; ++i) {
      result.push_back(info::partition_property(arrayResult[i]));
    }
    return result;
  }
};

// Specialization for partition affinity domains, OpenCL returns a bitfield
template <>
struct get_device_info<vector_class<info::partition_affinity_domain>,
                       info::device::partition_affinity_domains> {
  static vector_class<info::partition_affinity_domain>
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

    unique_ptr_class<cl_device_partition_property[]> arrayResult(
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
struct get_device_info<vector_class<size_t>, info::device::sub_group_sizes> {
  static vector_class<size_t> get(RT::PiDevice dev, const plugin &Plugin) {
    size_t resultSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::sub_group_sizes), 0,
        nullptr, &resultSize);

    vector_class<size_t> result(resultSize / sizeof(size_t));
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
    string_class platform_name = plt.get_info<info::platform::name>();
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

// SYCL host device information

// Default template is disabled, all possible instantiations are
// specified explicitly.
template <info::device param>
typename info::param_traits<info::device, param>::return_type
get_device_info_host() = delete;

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template <> ret_type get_device_info_host<info::param_type::param>();

#include <CL/sycl/info/device_traits.def>

#undef PARAM_TRAITS_SPEC

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
template <> struct get_device_info<bool, info::device::usm_system_allocator> {
  static bool get(RT::PiDevice dev, const plugin &Plugin) {
    pi_usm_capabilities caps;
    pi_result Err = Plugin.call_nocheck<PiApiKind::piDeviceGetInfo>(
        dev, pi::cast<RT::PiDeviceInfo>(info::device::usm_system_allocator),
        sizeof(pi_usm_capabilities), &caps, nullptr);
    return (Err != PI_SUCCESS) ? false : (caps & PI_USM_ACCESS);
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
