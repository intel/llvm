//==-------- device_info.hpp - SYCL device info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<info::fp_config> read_fp_bitfield(cl_device_fp_config bits);

vector_class<info::partition_affinity_domain>
read_domain_bitfield(cl_device_affinity_domain bits);

vector_class<info::execution_capability>
read_execution_bitfield(cl_device_exec_capabilities bits);

// Mapping expected SYCL return types to those returned by PI calls
template <typename T> struct sycl_to_pi { using type = T; };
template <> struct sycl_to_pi<bool>     { using type = pi_bool; };
template <> struct sycl_to_pi<device>   { using type = RT::PiDevice; };
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
  static T _(RT::PiDevice dev) {
    typename sycl_to_pi<T>::type result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param), sizeof(result), &result, NULL));
    return T(result);
  }
};

// Specialization for platform
template <info::device param>
struct get_device_info<platform, param> {
  static platform _(RT::PiDevice dev) {
    typename sycl_to_pi<platform>::type result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param), sizeof(result), &result, NULL));
    return createSyclObjFromImpl<platform>(
        std::make_shared<platform_impl_pi>(result));
  }
};

// Specialization for string return type, variable return size
template <info::device param> struct get_device_info<string_class, param> {
  static string_class _(RT::PiDevice dev) {
    size_t resultSize;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param), 0, NULL, &resultSize));
    if (resultSize == 0) {
      return string_class();
    }
    unique_ptr_class<char[]> result(new char[resultSize]);
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param),
      resultSize, result.get(), NULL));

    return string_class(result.get());
  }
};

// Specialization for parent device
template <typename T>
struct get_device_info<T, info::device::parent_device> {
  static T _(RT::PiDevice dev);
};

// Specialization for id return type
template <info::device param> struct get_device_info<id<3>, param> {
  static id<3> _(RT::PiDevice dev) {
    size_t result[3];
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param), sizeof(result), &result, NULL));
    return id<3>(result[0], result[1], result[2]);
  }
};

// Specialization for fp_config types, checks the corresponding fp type support
template <info::device param>
struct get_device_info<vector_class<info::fp_config>, param> {
  static vector_class<info::fp_config> _(RT::PiDevice dev) {
    // Check if fp type is supported
    if (!get_device_info<
            typename info::param_traits<
                info::device, check_fp_support<param>::value>::return_type,
            check_fp_support<param>::value>::_(dev)) {
      return {};
    }
    cl_device_fp_config result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(param), sizeof(result), &result, NULL));
    return read_fp_bitfield(result);
  }
};

// Specialization for single_fp_config, no type support check required
template <>
struct get_device_info<vector_class<info::fp_config>,
                          info::device::single_fp_config> {
  static vector_class<info::fp_config> _(RT::PiDevice dev) {
    cl_device_fp_config result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::single_fp_config),
      sizeof(result), &result, NULL));
    return read_fp_bitfield(result);
  }
};

// Specialization for queue_profiling, OpenCL returns a bitfield
template <> struct get_device_info<bool, info::device::queue_profiling> {
  static bool _(RT::PiDevice dev) {
    cl_command_queue_properties result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::queue_profiling),
      sizeof(result), &result, NULL));
    return (result & CL_QUEUE_PROFILING_ENABLE);
  }
};

// Specialization for exec_capabilities, OpenCL returns a bitfield
template <>
struct get_device_info<vector_class<info::execution_capability>,
                       info::device::execution_capabilities> {
  static vector_class<info::execution_capability> _(RT::PiDevice dev) {
    cl_device_exec_capabilities result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::execution_capabilities),
      sizeof(result), &result, NULL));
    return read_execution_bitfield(result);
  }
};

// Specialization for built in kernels, splits the string returned by OpenCL
template <>
struct get_device_info<vector_class<string_class>,
                       info::device::built_in_kernels> {
  static vector_class<string_class> _(RT::PiDevice dev) {
    string_class result =
        get_device_info<string_class, info::device::built_in_kernels>::_(dev);
    return split_string(result, ';');
  }
};

// Specialization for extensions, splits the string returned by OpenCL
template <>
struct get_device_info<vector_class<string_class>,
                       info::device::extensions> {
  static vector_class<string_class> _(RT::PiDevice dev) {
    string_class result =
        get_device_info<string_class, info::device::extensions>::_(dev);
    return split_string(result, ' ');
  }
};

// Specialization for partition properties, variable OpenCL return size
template <>
struct get_device_info<vector_class<info::partition_property>,
                       info::device::partition_properties> {
  static vector_class<info::partition_property> _(RT::PiDevice dev) {
    auto info_partition =
      pi::cast<RT::PiDeviceInfo>(info::device::partition_properties);

    size_t resultSize;
    PI_CALL(RT::piDeviceGetInfo(dev, info_partition, 0, NULL, &resultSize));

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);
    if (arrayLength == 0) {
      return {};
    }
    unique_ptr_class<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    PI_CALL(RT::piDeviceGetInfo(
      dev, info_partition, resultSize, arrayResult.get(), NULL));

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
  static vector_class<info::partition_affinity_domain> _(RT::PiDevice dev) {
    cl_device_affinity_domain result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::partition_affinity_domains),
      sizeof(result), &result, NULL));
    return read_domain_bitfield(result);
  }
};

// Specialization for partition type affinity domain, OpenCL can return other
// partition properties instead
template <>
struct get_device_info<info::partition_affinity_domain,
                       info::device::partition_type_affinity_domain> {
  static info::partition_affinity_domain _(RT::PiDevice dev) {
    size_t resultSize;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(
             info::device::partition_type_affinity_domain),
      0, NULL, &resultSize));
    if (resultSize != 1) {
      return info::partition_affinity_domain::not_applicable;
    }
    cl_device_partition_property result;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(
             info::device::partition_type_affinity_domain),
      sizeof(result), &result, NULL));
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
  static info::partition_property _(RT::PiDevice dev) {
    size_t resultSize;
    PI_CALL(RT::piDeviceGetInfo(
      dev, PI_DEVICE_INFO_PARTITION_TYPE, 0, NULL, &resultSize));
    if (!resultSize)
      return info::partition_property::no_partition;

    size_t arrayLength = resultSize / sizeof(cl_device_partition_property);

    unique_ptr_class<cl_device_partition_property[]> arrayResult(
        new cl_device_partition_property[arrayLength]);
    PI_CALL(RT::piDeviceGetInfo(
      dev, PI_DEVICE_INFO_PARTITION_TYPE, resultSize, arrayResult.get(), 0));
    if (!arrayResult[0])
      return info::partition_property::no_partition;
    return info::partition_property(arrayResult[0]);
  }
};
// Specialization for supported subgroup sizes
template <>
struct get_device_info<vector_class<size_t>,
                       info::device::sub_group_sizes> {
  static vector_class<size_t> _(RT::PiDevice dev) {
    size_t resultSize = 0;
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::sub_group_sizes),
      0, nullptr, &resultSize));

    vector_class<size_t> result(resultSize/sizeof(size_t));
    PI_CALL(RT::piDeviceGetInfo(
      dev, pi::cast<RT::PiDeviceInfo>(info::device::sub_group_sizes),
      resultSize, result.data(), nullptr));
    return result;
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

} // namespace detail
} // namespace sycl
} // namespace cl
