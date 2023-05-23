//===--------- device.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "device.hpp"
#include "common.hpp"

#include <cassert>
#include <sycl/detail/cl.h>

cl_int cl_adapter::getDeviceVersion(cl_device_id dev, OCLV::OpenCLVersion &version) {

  size_t devVerSize = 0;
  CL_RETURN_ON_FAILURE(
      clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, nullptr, &devVerSize));

  std::string devVer(devVerSize, '\0');
  CL_RETURN_ON_FAILURE(clGetDeviceInfo(dev, CL_DEVICE_VERSION, devVerSize,
                                       devVer.data(), nullptr));

  version = OCLV::OpenCLVersion(devVer);
  if (!version.isValid()) {
    return CL_INVALID_DEVICE;
  }

  return CL_SUCCESS;
}

cl_int cl_adapter::checkDeviceExtensions(cl_device_id dev,
                                 const std::vector<std::string> &exts,
                                 bool &supported) {
  size_t extSize = 0;
  CL_RETURN_ON_FAILURE(
      clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &extSize));

  std::string extStr(extSize, '\0');

  CL_RETURN_ON_FAILURE(clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, extSize,
                                       extStr.data(), nullptr));

  supported = true;
  for (const std::string &ext : exts) {
    if (!(supported = (extStr.find(ext) != std::string::npos))) {
      break;
    }
  }

  return CL_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {

  cl_device_type type;
  switch (DeviceType) {
  case UR_DEVICE_TYPE_ALL:
    type = CL_DEVICE_TYPE_ALL;
    break;
  case UR_DEVICE_TYPE_GPU:
    type = CL_DEVICE_TYPE_GPU;
    break;
  case UR_DEVICE_TYPE_CPU:
    type = CL_DEVICE_TYPE_CPU;
    break;
  case UR_DEVICE_TYPE_FPGA:
  case UR_DEVICE_TYPE_MCA:
  case UR_DEVICE_TYPE_VPU:
    type = CL_DEVICE_TYPE_ACCELERATOR;
    break;
  case UR_DEVICE_TYPE_DEFAULT:
    type = UR_DEVICE_TYPE_DEFAULT;
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  cl_int result = clGetDeviceIDs(cl_adapter::cast<cl_platform_id>(hPlatform),
                                 type, cl_adapter::cast<cl_uint>(NumEntries),
                                 cl_adapter::cast<cl_device_id *>(phDevices),
                                 cl_adapter::cast<cl_uint *>(pNumDevices));

  // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
  if (result == CL_DEVICE_NOT_FOUND) {
    result = CL_SUCCESS;
    if (pNumDevices) {
      *pNumDevices = 0;
    }
  }

  return map_cl_error_to_ur(result);
}

ur_device_fp_capability_flags_t
map_ur_cl_device_fp_config_to_ur(cl_device_fp_config cl_value) {

  ur_device_fp_capability_flags_t ur_value = 0;
  if (cl_value & CL_FP_DENORM) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  }
  if (cl_value & CL_FP_INF_NAN) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  }
  if (cl_value & CL_FP_ROUND_TO_NEAREST) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  }
  if (cl_value & CL_FP_ROUND_TO_ZERO) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  }
  if (cl_value & CL_FP_ROUND_TO_INF) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  }
  if (cl_value & CL_FP_FMA) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
  }
  if (cl_value & CL_FP_SOFT_FLOAT) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT;
  }
  if (cl_value & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
    ur_value |= UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  return ur_value;
}

cl_int map_ur_device_info_to_cl(ur_device_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_DEVICE_INFO_TYPE:
    cl_propName = CL_DEVICE_TYPE;
    break;
  case UR_DEVICE_INFO_PARENT_DEVICE:
    cl_propName = CL_DEVICE_PARENT_DEVICE;
    break;
  case UR_DEVICE_INFO_PLATFORM:
    cl_propName = CL_DEVICE_PLATFORM;
    break;
  case UR_DEVICE_INFO_VENDOR_ID:
    cl_propName = CL_DEVICE_VENDOR_ID;
    break;
  case UR_DEVICE_INFO_EXTENSIONS:
    cl_propName = CL_DEVICE_EXTENSIONS;
    break;
  case UR_DEVICE_INFO_NAME:
    cl_propName = CL_DEVICE_NAME;
    break;
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
    cl_propName = CL_DEVICE_COMPILER_AVAILABLE;
    break;
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
    cl_propName = CL_DEVICE_LINKER_AVAILABLE;
    break;
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS:
    cl_propName = CL_DEVICE_MAX_COMPUTE_UNITS;
    break;
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    cl_propName = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
    break;
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    cl_propName = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    break;
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES:
    cl_propName = CL_DEVICE_MAX_WORK_ITEM_SIZES;
    break;
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    cl_propName = CL_DEVICE_MAX_CLOCK_FREQUENCY;
    break;
  case UR_DEVICE_INFO_ADDRESS_BITS:
    cl_propName = CL_DEVICE_ADDRESS_BITS;
    break;
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    cl_propName = CL_DEVICE_MAX_MEM_ALLOC_SIZE;
    break;
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE:
    cl_propName = CL_DEVICE_GLOBAL_MEM_SIZE;
    break;
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE:
    cl_propName = CL_DEVICE_LOCAL_MEM_SIZE;
    break;
  case UR_DEVICE_INFO_IMAGE_SUPPORTED:
    cl_propName = CL_DEVICE_IMAGE_SUPPORT;
    break;
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    cl_propName = CL_DEVICE_HOST_UNIFIED_MEMORY;
    break;
  case UR_DEVICE_INFO_AVAILABLE:
    cl_propName = CL_DEVICE_AVAILABLE;
    break;
  case UR_DEVICE_INFO_VENDOR:
    cl_propName = CL_DEVICE_VENDOR;
    break;
  case UR_DEVICE_INFO_DRIVER_VERSION:
    cl_propName = CL_DRIVER_VERSION;
    break;
  case UR_DEVICE_INFO_VERSION:
    cl_propName = CL_DEVICE_VERSION;
    break;
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
    cl_propName = CL_DEVICE_PARTITION_MAX_SUB_DEVICES;
    break;
  case UR_DEVICE_INFO_REFERENCE_COUNT:
    cl_propName = CL_DEVICE_REFERENCE_COUNT;
    break;
  case UR_DEVICE_INFO_PARTITION_PROPERTIES:
    cl_propName = CL_DEVICE_PARTITION_PROPERTIES;
    break;
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    cl_propName = CL_DEVICE_PARTITION_AFFINITY_DOMAIN;
    break;
  case UR_DEVICE_INFO_PARTITION_TYPE:
    cl_propName = CL_DEVICE_PARTITION_TYPE;
    break;
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION:
    cl_propName = CL_DEVICE_OPENCL_C_VERSION;
    break;
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    cl_propName = CL_DEVICE_PREFERRED_INTEROP_USER_SYNC;
    break;
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    cl_propName = CL_DEVICE_PRINTF_BUFFER_SIZE;
    break;
  case UR_DEVICE_INFO_PROFILE:
    cl_propName = CL_DEVICE_PROFILE;
    break;
  case UR_DEVICE_INFO_BUILT_IN_KERNELS:
    cl_propName = CL_DEVICE_BUILT_IN_KERNELS;
    break;
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
    cl_propName = CL_DEVICE_QUEUE_PROPERTIES;
    break;
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES:
    cl_propName = CL_DEVICE_QUEUE_ON_HOST_PROPERTIES;
    break;
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES:
    cl_propName = CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES;
    break;
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES:
    cl_propName = CL_DEVICE_EXECUTION_CAPABILITIES;
    break;
  case UR_DEVICE_INFO_ENDIAN_LITTLE:
    cl_propName = CL_DEVICE_ENDIAN_LITTLE;
    break;
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    cl_propName = CL_DEVICE_ERROR_CORRECTION_SUPPORT;
    break;
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    cl_propName = CL_DEVICE_PROFILING_TIMER_RESOLUTION;
    break;
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE:
    cl_propName = CL_DEVICE_LOCAL_MEM_TYPE;
    break;
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS:
    cl_propName = CL_DEVICE_MAX_CONSTANT_ARGS;
    break;
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    cl_propName = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
    break;
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    cl_propName = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE;
    break;
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    cl_propName = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
    break;
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    cl_propName = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
    break;
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE:
    cl_propName = CL_DEVICE_MAX_PARAMETER_SIZE;
    break;
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    cl_propName = CL_DEVICE_MEM_BASE_ADDR_ALIGN;
    break;
  case UR_DEVICE_INFO_MAX_SAMPLERS:
    cl_propName = CL_DEVICE_MAX_SAMPLERS;
    break;
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    cl_propName = CL_DEVICE_MAX_READ_IMAGE_ARGS;
    break;
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    cl_propName = CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
    break;
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS:
    cl_propName = CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS;
    break;
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG:
    cl_propName = CL_DEVICE_SINGLE_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_HALF_FP_CONFIG:
    cl_propName = CL_DEVICE_HALF_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG:
    cl_propName = CL_DEVICE_DOUBLE_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    cl_propName = CL_DEVICE_IMAGE2D_MAX_WIDTH;
    break;
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    cl_propName = CL_DEVICE_IMAGE2D_MAX_HEIGHT;
    break;
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    cl_propName = CL_DEVICE_IMAGE3D_MAX_WIDTH;
    break;
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    cl_propName = CL_DEVICE_IMAGE3D_MAX_HEIGHT;
    break;
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    cl_propName = CL_DEVICE_IMAGE3D_MAX_DEPTH;
    break;
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    cl_propName = CL_DEVICE_IMAGE_MAX_BUFFER_SIZE;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
    break;
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    cl_propName = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    cl_propName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF;
    break;
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS:
    cl_propName = CL_DEVICE_MAX_NUM_SUB_GROUPS;
    break;
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
    cl_propName = CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS;
    break;
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    cl_propName = CL_DEVICE_SUB_GROUP_SIZES_INTEL;
    break;
  case UR_DEVICE_INFO_IL_VERSION:
    cl_propName = CL_DEVICE_IL_VERSION;
    break;
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    cl_propName = CL_DEVICE_IMAGE_MAX_ARRAY_SIZE;
    break;
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
    cl_propName = CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL;
    break;
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
    cl_propName = CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL;
    break;
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
    cl_propName = CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL;
    break;
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
    cl_propName = CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL;
    break;
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
    cl_propName = CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {

  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  const cl_device_info cl_propName = map_ur_device_info_to_cl(propName);

  /* TODO UR: Casting to uint32_t to silence warnings due to some values not
   * being part of the enum. Can be removed once all UR_EXT enums are promoted
   * to UR */
  switch (static_cast<uint32_t>(propName)) {
  case UR_DEVICE_INFO_TYPE: {
    cl_device_type cl_type;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        sizeof(cl_device_type), &cl_type, nullptr));

    /* TODO UR: If the device is an Accelerator (FPGA, VPU, etc.), there is not
     * enough information in the OpenCL runtime to know exactly which type it
     * is. Assuming FPGA for now */
    /* TODO UR: In OpenCL, a device can have multiple types (e.g. CPU and GPU).
     * We are potentially losing information by returning only one type */
    ur_device_type_t ur_device_type = UR_DEVICE_TYPE_DEFAULT;
    if (cl_type & CL_DEVICE_TYPE_CPU) {
      ur_device_type = UR_DEVICE_TYPE_CPU;
    } else if (cl_type & CL_DEVICE_TYPE_GPU) {
      ur_device_type = UR_DEVICE_TYPE_GPU;
    } else if (cl_type & CL_DEVICE_TYPE_ACCELERATOR) {
      ur_device_type = UR_DEVICE_TYPE_FPGA;
    }

    return ReturnValue(ur_device_type);
  }
  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION: {
    OCLV::OpenCLVersion version;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), version));

    const std::string results = std::to_string(version.getMajor()) + "." +
                                std::to_string(version.getMinor());
    return ReturnValue(results.c_str(), results.size() + 1);
  }
  case UR_DEVICE_INFO_PARTITION_PROPERTIES:
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    size_t cl_size;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName, 0,
                        nullptr, &cl_size));
    const size_t n_properties = cl_size / sizeof(cl_device_partition_property);

    /* Special case for UR_DEVICE_INFO_PARTITION_TYPE because OpenCL
     * implementation returns a size of 0 if the device is not a sub-device.
     * But UR implementation expects a size of 1 element with a value of 0. */
    if (propName == UR_DEVICE_INFO_PARTITION_TYPE && cl_size == 0) {
      return ReturnValue(static_cast<ur_device_partition_property_t>(0));
    }

    auto cl_value =
        reinterpret_cast<cl_device_partition_property *>(alloca(cl_size));
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        cl_size, cl_value, nullptr));

    std::vector<ur_device_partition_property_t> ur_value{};
    for (size_t i = 0; i < n_properties; ++i) {
      if (cl_value[i] != CL_DEVICE_PARTITION_BY_NAMES_INTEL) {
        ur_value.push_back(
            static_cast<ur_device_partition_property_t>(cl_value[i]));
      }
    }

    return ReturnValue(ur_value.data(), ur_value.size());
  }
  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    /* Returns the maximum sizes of a work group for each dimension one could
     * use to submit a kernel. There is no such query defined in OpenCL. So
     * we'll return the maximum value. */
    static constexpr uint32_t max_work_item_dimensions = 3u;
    static constexpr size_t Max = (std::numeric_limits<size_t>::max)();

    struct {
      size_t sizes[max_work_item_dimensions];
    } return_sizes;

    return_sizes.sizes[0] = Max;
    return_sizes.sizes[1] = Max;
    return_sizes.sizes[2] = Max;
    return ReturnValue(return_sizes);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(static_cast<uint32_t>(1u));
  }
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    /* Corresponding OpenCL query is only available starting with OpenCL 2.1
     * and we have to emulate it on older OpenCL runtimes. */
    OCLV::OpenCLVersion devVer;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), devVer));

    if (devVer >= OCLV::V2_1) {
      cl_uint cl_value;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          cl_adapter::cast<cl_device_id>(hDevice), CL_DEVICE_MAX_NUM_SUB_GROUPS,
          sizeof(cl_uint), &cl_value, nullptr));

      if (cl_value == 0u) {
        /* OpenCL returns 0 if sub-groups are not supported, but SYCL 2020
         * spec says that minimum possible value is 1. */
        return ReturnValue(1u);
      } else {
        return ReturnValue(static_cast<uint32_t>(cl_value));
      }
    } else {
      /* Otherwise, we can't query anything, because even cl_khr_subgroups
       * does not provide similar query. Therefore, simply return minimum
       * possible value 1 here. */
      return ReturnValue(1u);
    }
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG:
  case UR_DEVICE_INFO_HALF_FP_CONFIG:
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    /* CL type: cl_device_fp_config
     * UR type: ur_device_fp_capability_flags_t */
    if (propName == UR_DEVICE_INFO_HALF_FP_CONFIG) {
      bool supported;
      CL_RETURN_ON_FAILURE(cl_adapter::checkDeviceExtensions(
          cl_adapter::cast<cl_device_id>(hDevice), {"cl_khr_fp16"}, supported));

      if (!supported) {
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
      }
    }

    cl_device_fp_config cl_value;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        sizeof(cl_device_fp_config), &cl_value, nullptr));

    return ReturnValue(map_ur_cl_device_fp_config_to_ur(cl_value));
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    /* This query is missing before OpenCL 3.0. Check version and handle
     * appropriately */
    OCLV::OpenCLVersion devVer;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), devVer));

    /* Minimum required capability to be returned. For OpenCL 1.2, this is all
     * that is required */
    ur_memory_order_capability_flags_t ur_capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED;

    if (devVer >= OCLV::V3_0) {
      /* For OpenCL >=3.0, the query should be implemented */
      cl_device_atomic_capabilities cl_capabilities;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          cl_adapter::cast<cl_device_id>(hDevice),
          CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &cl_capabilities, nullptr));

      /* Mask operation to only consider atomic_memory_order* capabilities */
      const cl_int mask = CL_DEVICE_ATOMIC_ORDER_RELAXED |
                          CL_DEVICE_ATOMIC_ORDER_ACQ_REL |
                          CL_DEVICE_ATOMIC_ORDER_SEQ_CST;
      cl_capabilities &= mask;

      /* The memory order capabilities are hierarchical, if one is implied, all
       * preceding capabilities are implied as well. Especially in the case of
       * ACQ_REL. */
      if (cl_capabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        ur_capabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
      if (cl_capabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) {
        ur_capabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                           UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                           UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE;
      }
    } else if (devVer >= OCLV::V2_0) {
      /* For OpenCL 2.x, return all capabilities.
       * (https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_memory_consistency_model)
       */
      ur_capabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                         UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
                         UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                         UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
    }
    /* cl_device_atomic_capabilities is uint64_t and
     * ur_memory_order_capability_flags_t is uint32_t */
    return ReturnValue(
        static_cast<ur_memory_order_capability_flags_t>(ur_capabilities));
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2. Because scopes are hierarchical, wider scopes support
     * all narrower scopes. At a minimum, each device must support WORK_ITEM,
     * SUB_GROUP and WORK_GROUP.
     * (https://github.com/KhronosGroup/SYCL-Docs/pull/382) */
    ur_memory_scope_capability_flags_t ur_capabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;

    OCLV::OpenCLVersion devVer;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), devVer));

    cl_device_atomic_capabilities cl_capabilities;
    if (devVer >= OCLV::V3_0) {
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          cl_adapter::cast<cl_device_id>(hDevice),
          CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &cl_capabilities, nullptr));

      assert((cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");

      /* Because scopes are hierarchical, wider scopes support all narrower
       * scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
       * WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382). We
       * already initialized to these minimum mandated capabilities. Just check
       * wider scopes. */
      if (cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
      }

      if (cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    } else {
      /* This info is only available in OpenCL version >= 3.0. Just return
       * minimum mandated capabilities for older versions. OpenCL 1.x minimum
       * mandated capabilities are WORK_GROUP, we already initialized using it.
       */
      if (devVer >= OCLV::V2_0) {
        /* OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
         * ALL_DEVICES */
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    }

    /* cl_device_atomic_capabilities is uint64_t and
     * ur_memory_scope_capability_flags_t is uint32_t */
    return ReturnValue(
        static_cast<ur_memory_scope_capability_flags_t>(ur_capabilities));
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2 */
    ur_memory_order_capability_flags_t ur_capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;

    OCLV::OpenCLVersion devVer;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), devVer));

    cl_device_atomic_capabilities cl_capabilities;
    if (devVer >= OCLV::V3_0) {
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          cl_adapter::cast<cl_device_id>(hDevice),
          CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &cl_capabilities, nullptr));

      assert((cl_capabilities & CL_DEVICE_ATOMIC_ORDER_RELAXED) &&
             "Violates minimum mandated guarantee");
      assert((cl_capabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) &&
             "Violates minimum mandated guarantee");

      /* We already initialized to minimum mandated capabilities. Just check
       * stronger orders. */
      if (cl_capabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        ur_capabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    } else {
      /* This info is only available in OpenCL version >= 3.0. Just return
       * minimum mandated capabilities for older versions. OpenCL 1.x minimum
       * mandated capabilities are RELAXED | ACQ_REL, we already initialized
       * using these. */
      if (devVer >= OCLV::V2_0) {
        /* OpenCL 2.x minimum mandated capabilities are RELAXED | ACQ_REL |
         * SEQ_CST */
        ur_capabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    }

    /* cl_device_atomic_capabilities is uint64_t and
     * ur_memory_order_capability_flags_t is uint32_t */
    return ReturnValue(
        static_cast<ur_memory_order_capability_flags_t>(ur_capabilities));
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2. Because scopes are hierarchical, wider scopes support
     * all narrower scopes. At a minimum, each device must support WORK_ITEM,
     * SUB_GROUP and WORK_GROUP.
     * (https://github.com/KhronosGroup/SYCL-Docs/pull/382) */
    ur_memory_scope_capability_flags_t ur_capabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;

    OCLV::OpenCLVersion devVer;
    CL_RETURN_ON_FAILURE(
        cl_adapter::getDeviceVersion(cl_adapter::cast<cl_device_id>(hDevice), devVer));

    cl_device_atomic_capabilities cl_capabilities;
    if (devVer >= OCLV::V3_0) {
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          cl_adapter::cast<cl_device_id>(hDevice),
          CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &cl_capabilities, nullptr));

      assert((cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");

      /* Because scopes are hierarchical, wider scopes support all narrower
       * scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
       * WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382). We
       * already initialized to these minimum mandated capabilities. Just check
       * wider scopes. */
      if (cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
      }

      if (cl_capabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    } else {
      /* This info is only available in OpenCL version >= 3.0. Just return
       * minimum mandated capabilities for older versions. OpenCL 1.x minimum
       * mandated capabilities are WORK_GROUP, we already initialized using it.
       */
      if (devVer >= OCLV::V2_0) {
        /* OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
         * ALL_DEVICES */
        ur_capabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    }

    /* cl_device_atomic_capabilities is uint64_t and
     * ur_memory_scope_capability_flags_t is uint32_t */
    return ReturnValue(
        static_cast<ur_memory_scope_capability_flags_t>(ur_capabilities));
  }

  case UR_DEVICE_INFO_IMAGE_SRGB: {
    return ReturnValue(true);
  }

  case UR_DEVICE_INFO_BFLOAT16: {
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_ATOMIC_64: {
    bool supported = false;
    CL_RETURN_ON_FAILURE(cl_adapter::checkDeviceExtensions(
        cl_adapter::cast<cl_device_id>(hDevice),
        {"cl_khr_int64_base_atomics", "cl_khr_int64_extended_atomics"},
        supported));

    return ReturnValue(supported);
  }
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE: {

    cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), CL_DEVICE_TYPE,
                        sizeof(cl_device_type), &devType, nullptr));

    return ReturnValue(devType == CL_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT: {
    bool supported = false;
    CL_RETURN_ON_FAILURE(
        cl_adapter::checkDeviceExtensions(cl_adapter::cast<cl_device_id>(hDevice),
                              {"cl_intel_mem_channel_property"}, supported));

    return ReturnValue(supported);
  }
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES:
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE:
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES:
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    /* CL type: cl_bitfield
     * UR type: ur_flags_t (uint32_t) */

    cl_bitfield cl_value;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        sizeof(cl_bitfield), &cl_value, nullptr));

    /* We can just static_cast the output because OpenCL and UR bitfields
     * map 1 to 1 for these properties. cl_bitfield is uint64_t and ur_flags_t
     * types are uint32_t */
    return ReturnValue(static_cast<uint32_t>(cl_value));
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORTED:
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
  case UR_DEVICE_INFO_ENDIAN_LITTLE:
  case UR_DEVICE_INFO_AVAILABLE:
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
  case UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS:
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    /* CL type: cl_bool
     * UR type: ur_bool_t */

    cl_bool cl_value;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        sizeof(cl_bool), &cl_value, nullptr));

    /* cl_bool is uint32_t and ur_bool_t is bool */
    return ReturnValue(static_cast<ur_bool_t>(cl_value));
  }
  case UR_DEVICE_INFO_VENDOR_ID:
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS:
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
  case UR_DEVICE_INFO_ADDRESS_BITS:
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS:
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
  case UR_DEVICE_INFO_MAX_SAMPLERS:
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS:
  case UR_DEVICE_INFO_REFERENCE_COUNT:
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE:
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE:
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE:
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE:
  case UR_DEVICE_INFO_PLATFORM:
  case UR_DEVICE_INFO_PARENT_DEVICE:
  case UR_DEVICE_INFO_IL_VERSION:
  case UR_DEVICE_INFO_NAME:
  case UR_DEVICE_INFO_VENDOR:
  case UR_DEVICE_INFO_DRIVER_VERSION:
  case UR_DEVICE_INFO_PROFILE:
  case UR_DEVICE_INFO_VERSION:
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION:
  case UR_DEVICE_INFO_EXTENSIONS:
  case UR_DEVICE_INFO_BUILT_IN_KERNELS:
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES:
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    /* We can just use the OpenCL outputs because the sizes of OpenCL types
     * are the same as UR.
     * | CL                 | UR                     | Size |
     * | char[]             | char[]                 | 8    |
     * | cl_uint            | uint32_t               | 4    |
     * | cl_ulong           | uint64_t               | 8    |
     * | size_t             | size_t                 | 8    |
     * | cl_platform_id     | ur_platform_handle_t   | 8    |
     * | ur_device_handle_t | cl_device_id           | 8    |
     */

    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), cl_propName,
                        propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  /* TODO: Check regularly to see if support is enabled in OpenCL. Intel GPU
   * EU device-specific information extensions. Some of the queries are
   * enabled by cl_intel_device_attribute_query extension, but it's not yet in
   * the Registry. */
  case UR_DEVICE_INFO_PCI_ADDRESS:
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_GPU_EU_SLICES:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH:
  /* TODO: Check if device UUID extension is enabled in OpenCL. For details
   * about Intel UUID extension, see
   * sycl/doc/extensions/supported/sycl_ext_intel_device_info.md */
  case UR_DEVICE_INFO_UUID:
  /* This enums have no equivalent in OpenCL */
  case UR_DEVICE_INFO_DEVICE_ID:
  case UR_DEVICE_INFO_GLOBAL_MEM_FREE:
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH:
  case UR_DEVICE_INFO_ASYNC_BARRIER:
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urDevicePartition(
    ur_device_handle_t hDevice,
    const ur_device_partition_property_t *pProperties, uint32_t NumDevices,
    ur_device_handle_t *phSubDevices, uint32_t *pNumDevicesRet) {

  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pProperties, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_uint cl_num_devices_ret;
  CL_RETURN_ON_FAILURE(clCreateSubDevices(
      cl_adapter::cast<cl_device_id>(hDevice),
      cl_adapter::cast<const cl_device_partition_property *>(pProperties), 0,
      nullptr, &cl_num_devices_ret));

  if (pNumDevicesRet) {
    *pNumDevicesRet = cl_num_devices_ret;
  }

  /*If NumDevices is less than the number of sub-devices available, then the
   * function shall only retrieve that number of sub-devices. */
  if (phSubDevices) {
    std::vector<cl_device_id> cl_sub_devices(cl_num_devices_ret);
    CL_RETURN_ON_FAILURE(clCreateSubDevices(
        cl_adapter::cast<cl_device_id>(hDevice),
        cl_adapter::cast<const cl_device_partition_property *>(pProperties),
        cl_num_devices_ret, cl_sub_devices.data(), nullptr));

    std::memcpy(phSubDevices, cl_sub_devices.data(),
                sizeof(cl_device_id) * NumDevices);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t hDevice) {

  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  cl_int result = clRetainDevice(cl_adapter::cast<cl_device_id>(hDevice));

  return map_cl_error_to_ur(result);
}

UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(ur_device_handle_t hDevice) {

  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  cl_int result = clReleaseDevice(cl_adapter::cast<cl_device_id>(hDevice));

  return map_cl_error_to_ur(result);
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeDevice) {

  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeDevice, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeDevice = reinterpret_cast<ur_native_handle_t>(hDevice);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_platform_handle_t,
    const ur_device_native_properties_t *, ur_device_handle_t *phDevice) {

  UR_ASSERT(hNativeDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  *phDevice = reinterpret_cast<ur_device_handle_t>(hNativeDevice);
  return UR_RESULT_SUCCESS;
}
