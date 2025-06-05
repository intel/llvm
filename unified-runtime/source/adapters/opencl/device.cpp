//===--------- device.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "device.hpp"
#include "adapter.hpp"
#include "common.hpp"
#include "platform.hpp"

#include <array>
#include <cassert>

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {

  cl_device_type Type;
  switch (DeviceType) {
  case UR_DEVICE_TYPE_ALL:
    Type = CL_DEVICE_TYPE_ALL;
    break;
  case UR_DEVICE_TYPE_GPU:
    Type = CL_DEVICE_TYPE_GPU;
    break;
  case UR_DEVICE_TYPE_CPU:
    Type = CL_DEVICE_TYPE_CPU;
    break;
  case UR_DEVICE_TYPE_FPGA:
  case UR_DEVICE_TYPE_MCA:
  case UR_DEVICE_TYPE_VPU:
    Type = CL_DEVICE_TYPE_ACCELERATOR;
    break;
  case UR_DEVICE_TYPE_DEFAULT:
    Type = CL_DEVICE_TYPE_DEFAULT;
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  try {
    uint32_t AllDevicesNum = hPlatform->Devices.size();
    uint32_t DeviceNumIter = 0;
    for (uint32_t i = 0; i < AllDevicesNum; i++) {
      cl_device_type DevTy = hPlatform->Devices[i]->Type;
      if (DevTy == Type || Type == CL_DEVICE_TYPE_ALL) {
        if (phDevices) {
          phDevices[DeviceNumIter] = hPlatform->Devices[i].get();
        }
        DeviceNumIter++;
      }
    }
    if (pNumDevices) {
      *pNumDevices = DeviceNumIter;
    }

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

static ur_device_fp_capability_flags_t
mapCLDeviceFpConfigToUR(cl_device_fp_config CLValue) {

  ur_device_fp_capability_flags_t URValue = 0;
  if (CLValue & CL_FP_DENORM) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  }
  if (CLValue & CL_FP_INF_NAN) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  }
  if (CLValue & CL_FP_ROUND_TO_NEAREST) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  }
  if (CLValue & CL_FP_ROUND_TO_ZERO) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  }
  if (CLValue & CL_FP_ROUND_TO_INF) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  }
  if (CLValue & CL_FP_FMA) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
  }
  if (CLValue & CL_FP_SOFT_FLOAT) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT;
  }
  if (CLValue & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
    URValue |= UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  return URValue;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  /* We can convert between OpenCL and UR outputs because the sizes
   * of OpenCL types are the same as UR.
   * | CL                 | UR                     | Size |
   * | char[]             | char[]                 | 8    |
   * | cl_uint            | uint32_t               | 4    |
   * | cl_ulong           | uint64_t               | 8    |
   * | size_t             | size_t                 | 8    |
   * | cl_platform_id     | ur_platform_handle_t   | 8    |
   * | cl_device_id       | ur_device_handle_t     | 8    |
   *
   * These other types are equivalent:
   * | cl_device_fp_config | ur_device_fp_capability_flags_t |
   * | cl_bitfield / enum | ur_flags_t |
   * | cl_bool | ur_bool_t |
   * | cl_device_atomic_capabilities | ur_memory_order_capability_flags_t |
   */

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  /* TODO UR: Casting to uint32_t to silence warnings due to some values not
   * being part of the enum. Can be removed once all UR_EXT enums are promoted
   * to UR */
  switch (static_cast<uint32_t>(propName)) {
  case UR_DEVICE_INFO_TYPE: {
    cl_device_type CLType = hDevice->Type;

    /* TODO UR: If the device is an Accelerator (FPGA, VPU, etc.), there is not
     * enough information in the OpenCL runtime to know exactly which type it
     * is. Assuming FPGA for now */
    /* TODO UR: In OpenCL, a device can have multiple types (e.g. CPU and GPU).
     * We are potentially losing information by returning only one type */
    ur_device_type_t URDeviceType = UR_DEVICE_TYPE_DEFAULT;
    if (CLType & CL_DEVICE_TYPE_CPU) {
      URDeviceType = UR_DEVICE_TYPE_CPU;
    } else if (CLType & CL_DEVICE_TYPE_GPU) {
      URDeviceType = UR_DEVICE_TYPE_GPU;
    } else if (CLType & CL_DEVICE_TYPE_ACCELERATOR) {
      URDeviceType = UR_DEVICE_TYPE_FPGA;
    }

    return ReturnValue(URDeviceType);
  }
  case UR_DEVICE_INFO_DEVICE_ID: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));

    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }

    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_ID_INTEL,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }

  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION: {
    oclv::OpenCLVersion Version;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(Version));

    const std::string Results = std::to_string(Version.getMajor()) + "." +
                                std::to_string(Version.getMinor());
    return ReturnValue(Results.c_str(), Results.size() + 1);
  }
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    const cl_device_info info_name = CL_DEVICE_PARTITION_PROPERTIES;
    size_t CLSize;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, info_name, 0, nullptr, &CLSize));
    const size_t NProperties = CLSize / sizeof(cl_device_partition_property);

    std::vector<cl_device_partition_property> CLValue(NProperties);
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, info_name, CLSize,
                                         CLValue.data(), nullptr));

    /* The OpenCL implementation returns a value of 0 if no properties are
     * supported. UR will return a size of 0 for now.
     */
    if (pPropSizeRet && CLValue[0] == 0) {
      *pPropSizeRet = 0;
      return UR_RESULT_SUCCESS;
    }

    std::vector<ur_device_partition_t> URValue{};
    for (size_t i = 0; i < NProperties; ++i) {
      if (CLValue[i] != CL_DEVICE_PARTITION_BY_NAMES_INTEL && CLValue[i] != 0) {
        URValue.push_back(static_cast<ur_device_partition_t>(CLValue[i]));
      }
    }
    return ReturnValue(URValue.data(), URValue.size());
  }
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    const cl_device_info info_name = CL_DEVICE_PARTITION_TYPE;
    size_t CLSize;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, info_name, 0, nullptr, &CLSize));
    const size_t NProperties = CLSize / sizeof(cl_device_partition_property);

    /* The OpenCL implementation returns either a size of 0 or a value of 0 if
     * the device is not a sub-device. UR will return a size of 0 for now.
     * TODO Ideally, this could become an error once PI is removed from SYCL RT
     */
    if (pPropSizeRet && (CLSize == 0 || NProperties == 1)) {
      *pPropSizeRet = 0;
      return UR_RESULT_SUCCESS;
    }

    auto CLValue =
        reinterpret_cast<cl_device_partition_property *>(alloca(CLSize));
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, info_name, CLSize,
                                         CLValue, nullptr));

    std::vector<ur_device_partition_property_t> URValue(NProperties - 1);

    /* OpenCL will always return exactly one partition type followed by one or
     * more values. */
    for (uint32_t i = 0; i < URValue.size(); ++i) {
      URValue[i].type = static_cast<ur_device_partition_t>(CLValue[0]);
      switch (URValue[i].type) {
      case UR_DEVICE_PARTITION_EQUALLY: {
        URValue[i].value.equally = static_cast<uint32_t>(CLValue[i + 1]);
        break;
      }
      case UR_DEVICE_PARTITION_BY_COUNTS: {
        URValue[i].value.count = static_cast<uint32_t>(CLValue[i + 1]);
        break;
      }
      case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: {
        URValue[i].value.affinity_domain =
            static_cast<uint32_t>(CLValue[i + 1]);
        break;
      }
      default: {
        return UR_RESULT_ERROR_UNKNOWN;
      }
      }
    }

    return ReturnValue(URValue.data(), URValue.size());
  }
  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    /* Returns the maximum sizes of a work group for each dimension one could
     * use to submit a kernel. There is no such query defined in OpenCL. So
     * we'll return the maximum value. */
    static constexpr uint32_t MaxWorkItemDimensions = 3u;
    static constexpr size_t Max = (std::numeric_limits<size_t>::max)();

    struct {
      size_t sizes[MaxWorkItemDimensions];
    } ReturnSizes;

    ReturnSizes.sizes[0] = Max;
    ReturnSizes.sizes[1] = Max;
    ReturnSizes.sizes[2] = Max;
    return ReturnValue(ReturnSizes);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(static_cast<uint32_t>(1u));
  }
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    /* Corresponding OpenCL query is only available starting with OpenCL 2.1
     * and we have to emulate it on older OpenCL runtimes. */
    oclv::OpenCLVersion DevVer;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

    if (DevVer >= oclv::V2_1) {
      cl_uint CLValue;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                           CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                           sizeof(cl_uint), &CLValue, nullptr));

      if (CLValue == 0u) {
        /* OpenCL returns 0 if sub-groups are not supported, but SYCL 2020
         * spec says that minimum possible value is 1. */
        return ReturnValue(1u);
      } else {
        return ReturnValue(static_cast<uint32_t>(CLValue));
      }
    } else {
      /* Otherwise, we can't query anything, because even cl_khr_subgroups
       * does not provide similar query. Therefore, simply return minimum
       * possible value 1 here. */
      return ReturnValue(1u);
    }
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    cl_device_fp_config CLValue;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_SINGLE_FP_CONFIG,
                        sizeof(cl_device_fp_config), &CLValue, nullptr));

    return ReturnValue(mapCLDeviceFpConfigToUR(CLValue));
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(
        hDevice->checkDeviceExtensions({"cl_khr_fp16"}, Supported));

    if (!Supported) {
      // If we don't support the extension then our capabilities are 0.
      ur_device_fp_capability_flags_t halfCapabilities = 0;
      return ReturnValue(halfCapabilities);
    }

    cl_device_fp_config CLValue;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_HALF_FP_CONFIG,
                        sizeof(cl_device_fp_config), &CLValue, nullptr));

    return ReturnValue(mapCLDeviceFpConfigToUR(CLValue));
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    cl_device_fp_config CLValue;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_DOUBLE_FP_CONFIG,
                        sizeof(cl_device_fp_config), &CLValue, nullptr));

    return ReturnValue(mapCLDeviceFpConfigToUR(CLValue));
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    /* This query is missing before OpenCL 3.0. Check version and handle
     * appropriately */
    oclv::OpenCLVersion DevVer;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

    /* Minimum required capability to be returned. For OpenCL 1.2, this is all
     * that is required */
    ur_memory_order_capability_flags_t URCapabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED;

    if (DevVer >= oclv::V3_0) {
      /* For OpenCL >=3.0, the query should be implemented */
      cl_device_atomic_capabilities CLCapabilities;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &CLCapabilities, nullptr));

      /* Mask operation to only consider atomic_memory_order* capabilities */
      const cl_int Mask = CL_DEVICE_ATOMIC_ORDER_RELAXED |
                          CL_DEVICE_ATOMIC_ORDER_ACQ_REL |
                          CL_DEVICE_ATOMIC_ORDER_SEQ_CST;
      CLCapabilities &= Mask;

      /* The memory order capabilities are hierarchical, if one is implied, all
       * preceding capabilities are implied as well. Especially in the case of
       * ACQ_REL. */
      if (CLCapabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        URCapabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
      if (CLCapabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) {
        URCapabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                          UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                          UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE;
      }
    } else if (DevVer >= oclv::V2_0) {
      /* For OpenCL 2.x, return all capabilities.
       * (https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_memory_consistency_model)
       */
      URCapabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
                        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                        UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
    }
    return ReturnValue(
        static_cast<ur_memory_order_capability_flags_t>(URCapabilities));
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2. Because scopes are hierarchical, wider scopes support
     * all narrower scopes. At a minimum, each device must support WORK_ITEM,
     * SUB_GROUP and WORK_GROUP.
     * (https://github.com/KhronosGroup/SYCL-Docs/pull/382) */
    ur_memory_scope_capability_flags_t URCapabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;

    oclv::OpenCLVersion DevVer;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

    cl_device_atomic_capabilities CLCapabilities;
    if (DevVer >= oclv::V3_0) {
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &CLCapabilities, nullptr));

      assert((CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");

      /* Because scopes are hierarchical, wider scopes support all narrower
       * scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
       * WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382). We
       * already initialized to these minimum mandated capabilities. Just check
       * wider scopes. */
      if (CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
        URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
      }

      if (CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
        URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    } else {
      /* This info is only available in OpenCL version >= 3.0. Just return
       * minimum mandated capabilities for older versions. OpenCL 1.x minimum
       * mandated capabilities are WORK_GROUP, we already initialized using it.
       */
      if (DevVer >= oclv::V2_0) {
        /* OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
         * ALL_DEVICES */
        URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                          UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    }

    return ReturnValue(
        static_cast<ur_memory_scope_capability_flags_t>(URCapabilities));
  }

  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2 */
    ur_memory_order_capability_flags_t URCapabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;

    oclv::OpenCLVersion DevVer;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

    cl_device_atomic_capabilities CLCapabilities;
    if (DevVer >= oclv::V3_0) {
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &CLCapabilities, nullptr));

      assert((CLCapabilities & CL_DEVICE_ATOMIC_ORDER_RELAXED) &&
             "Violates minimum mandated guarantee");
      assert((CLCapabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) &&
             "Violates minimum mandated guarantee");

      /* We already initialized to minimum mandated capabilities. Just check
       * stronger orders. */
      if (CLCapabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        URCapabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    } else {
      /* This info is only available in OpenCL version >= 3.0. Just return
       * minimum mandated capabilities for older versions. OpenCL 1.x minimum
       * mandated capabilities are RELAXED | ACQ_REL, we already initialized
       * using these. */
      if (DevVer >= oclv::V2_0) {
        /* OpenCL 2.x minimum mandated capabilities are RELAXED | ACQ_REL |
         * SEQ_CST */
        URCapabilities |= UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    }

    return ReturnValue(
        static_cast<ur_memory_order_capability_flags_t>(URCapabilities));
  }

  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    /* Initialize result to minimum mandated capabilities according to
     * SYCL2020 4.6.3.2. Because scopes are hierarchical, wider scopes support
     * all narrower scopes. At a minimum, each device must support WORK_ITEM,
     * SUB_GROUP and WORK_GROUP.
     * (https://github.com/KhronosGroup/SYCL-Docs/pull/382) */
    ur_memory_scope_capability_flags_t URCapabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;

    oclv::OpenCLVersion DevVer;
    UR_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

    auto convertCapabilities =
        [](cl_device_atomic_capabilities CLCapabilities) {
          ur_memory_scope_capability_flags_t URCapabilities = 0;
          /* Because scopes are hierarchical, wider scopes support all narrower
           * scopes. At a minimum, each device must support WORK_ITEM,
           * SUB_GROUP and WORK_GROUP.
           * (https://github.com/KhronosGroup/SYCL-Docs/pull/382). We already
           * initialized to these minimum mandated capabilities. Just check
           * wider scopes. */
          if (CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
            URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
          }

          if (CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
            URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
          }
          return URCapabilities;
        };

    if (DevVer >= oclv::V3_0) {
      cl_device_atomic_capabilities CLCapabilities;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &CLCapabilities, nullptr));
      assert((CLCapabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");
      URCapabilities |= convertCapabilities(CLCapabilities);
    } else if (DevVer >= oclv::V2_0) {
      /* OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
         ALL_DEVICES */
      URCapabilities |= UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;

    } else {
      // FIXME: Special case for Intel FPGA driver which is currently an
      // OpenCL 1.2 device but is more capable than the default. This is a
      // temporary work around until the Intel FPGA driver is updated to
      // OpenCL 3.0. If the query is successful, then use the result but do
      // not return an error if the query is unsuccessful as this is expected
      // of an OpenCL 1.2 driver.
      cl_device_atomic_capabilities CLCapabilities;
      if (CL_SUCCESS == clGetDeviceInfo(hDevice->CLDevice,
                                        CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                                        sizeof(cl_device_atomic_capabilities),
                                        &CLCapabilities, nullptr)) {
        URCapabilities |= convertCapabilities(CLCapabilities);
      }
    }

    return ReturnValue(
        static_cast<ur_memory_scope_capability_flags_t>(URCapabilities));
  }

  case UR_DEVICE_INFO_IMAGE_SRGB: {
    return ReturnValue(true);
  }

  case UR_DEVICE_INFO_ATOMIC_64: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_khr_int64_base_atomics", "cl_khr_int64_extended_atomics"},
        Supported));

    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE: {

    cl_device_type DevType = CL_DEVICE_TYPE_DEFAULT;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                         sizeof(cl_device_type), &DevType,
                                         nullptr));

    return ReturnValue(DevType == CL_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_mem_channel_property"}, Supported));

    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_ESIMD_SUPPORT: {
    bool Supported = false;
    cl_device_type DevType = CL_DEVICE_TYPE_DEFAULT;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                         sizeof(cl_device_type), &DevType,
                                         nullptr));

    cl_uint VendorID = 0;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_VENDOR_ID,
                                         sizeof(VendorID), &VendorID, nullptr));

    /* ESIMD is only supported by Intel GPUs. */
    Supported = DevType == CL_DEVICE_TYPE_GPU && VendorID == 0x8086;

    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT: {
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_NUM_COMPUTE_UNITS: {

    bool ExtensionSupported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, ExtensionSupported));

    cl_device_type CLType;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                         sizeof(cl_device_type), &CLType,
                                         nullptr));

    cl_uint NumComputeUnits;
    if (ExtensionSupported && (CLType & CL_DEVICE_TYPE_GPU)) {
      cl_uint SliceCount = 0;
      cl_uint SubSlicePerSliceCount = 0;
      CL_RETURN_ON_FAILURE(
          clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_NUM_SLICES_INTEL,
                          sizeof(cl_uint), &SliceCount, nullptr));
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL,
          sizeof(cl_uint), &SubSlicePerSliceCount, nullptr));
      NumComputeUnits = SliceCount * SubSlicePerSliceCount;
    } else {
      CL_RETURN_ON_FAILURE(
          clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cl_uint), &NumComputeUnits, nullptr));
    }

    return ReturnValue(static_cast<uint32_t>(NumComputeUnits));
  }
  case UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP: {
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP: {
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_program_scope_host_pipe"}, Supported));
    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_GLOBAL_VARIABLE_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_global_variable_access"}, Supported));
    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_QUEUE_PROPERTIES: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_QUEUE_PROPERTIES,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_LOCAL_MEM_TYPE,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_EXECUTION_CAPABILITIES,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    cl_bitfield CLValue = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
                        sizeof(cl_bitfield), &CLValue, nullptr));

    return ReturnValue(static_cast<uint32_t>(CLValue));
  }
  case UR_DEVICE_INFO_USM_HOST_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_unified_shared_memory"}, Supported));
    if (Supported) {
      cl_bitfield CLValue = 0;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
          sizeof(cl_bitfield), &CLValue, nullptr));
      return ReturnValue(static_cast<uint32_t>(CLValue));
    } else {
      return ReturnValue(0);
    }
  }
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_unified_shared_memory"}, Supported));
    if (Supported) {
      cl_bitfield CLValue = 0;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
          sizeof(cl_bitfield), &CLValue, nullptr));
      return ReturnValue(static_cast<uint32_t>(CLValue));
    } else {
      return ReturnValue(0);
    }
  }
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_unified_shared_memory"}, Supported));
    if (Supported) {
      cl_bitfield CLValue = 0;
      CL_RETURN_ON_FAILURE(
          clGetDeviceInfo(hDevice->CLDevice,
                          CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
                          sizeof(cl_bitfield), &CLValue, nullptr));
      return ReturnValue(static_cast<uint32_t>(CLValue));
    } else {
      return ReturnValue(0);
    }
  }
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_unified_shared_memory"}, Supported));
    if (Supported) {
      cl_bitfield CLValue = 0;
      CL_RETURN_ON_FAILURE(
          clGetDeviceInfo(hDevice->CLDevice,
                          CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
                          sizeof(cl_bitfield), &CLValue, nullptr));
      return ReturnValue(static_cast<uint32_t>(CLValue));
    } else {
      return ReturnValue(0);
    }
  }
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_unified_shared_memory"}, Supported));
    if (Supported) {
      cl_bitfield CLValue = 0;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
          sizeof(cl_bitfield), &CLValue, nullptr));
      return ReturnValue(static_cast<uint32_t>(CLValue));
    } else {
      return ReturnValue(0);
    }
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORT: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE_SUPPORT,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_HOST_UNIFIED_MEMORY,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_ENDIAN_LITTLE: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_ENDIAN_LITTLE,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_AVAILABLE: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_AVAILABLE,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_COMPILER_AVAILABLE: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_COMPILER_AVAILABLE,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_LINKER_AVAILABLE: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_LINKER_AVAILABLE,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    cl_bool CLValue;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
                                         sizeof(cl_bool), &CLValue, nullptr));

    return ReturnValue(static_cast<ur_bool_t>(CLValue));
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    oclv::OpenCLVersion DevVer;
    CL_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));
    /* Independent forward progress query is only supported as of OpenCL 2.1
     * if version is older we return a default false. */
    if (DevVer >= oclv::V2_1) {
      cl_bool CLValue;
      CL_RETURN_ON_FAILURE(clGetDeviceInfo(
          hDevice->CLDevice, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
          sizeof(cl_bool), &CLValue, nullptr));

      return ReturnValue(static_cast<ur_bool_t>(CLValue));
    } else {
      return ReturnValue(false);
    }
  }
  case UR_DEVICE_INFO_VENDOR_ID: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_VENDOR_ID,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_COMPUTE_UNITS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(
        hDevice->CLDevice, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, propSize,
        pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_ADDRESS_BITS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_READ_IMAGE_ARGS,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_SAMPLERS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_SAMPLERS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_CONSTANT_ARGS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT: {
    return ReturnValue(hDevice->getReferenceCount());
  }
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_MEM_ALLOC_SIZE, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_GLOBAL_MEM_SIZE, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_LOCAL_MEM_SIZE, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE2D_MAX_WIDTH, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE2D_MAX_HEIGHT, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE3D_MAX_WIDTH, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE3D_MAX_HEIGHT, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE3D_MAX_DEPTH, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_PARAMETER_SIZE, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PRINTF_BUFFER_SIZE, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PLATFORM: {
    return ReturnValue(hDevice->Platform);
  }
  case UR_DEVICE_INFO_PARENT_DEVICE: {
    return ReturnValue(hDevice->ParentDevice);
  }
  case UR_DEVICE_INFO_IL_VERSION: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IL_VERSION, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NAME: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_NAME,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_VENDOR: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_VENDOR,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_DRIVER_VERSION: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DRIVER_VERSION,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PROFILE: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_PROFILE,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_VERSION: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_VERSION,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_OPENCL_C_VERSION, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_BUILT_IN_KERNELS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_BUILT_IN_KERNELS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PCI_ADDRESS: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(
        hDevice->checkDeviceExtensions({"cl_khr_pci_bus_info"}, Supported));

    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }

    cl_device_pci_bus_info_khr PciInfo = {};
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_PCI_BUS_INFO_KHR,
                                         sizeof(PciInfo), &PciInfo, nullptr));

    constexpr size_t AddressBufferSize = 13;
    char AddressBuffer[AddressBufferSize];
    std::snprintf(AddressBuffer, AddressBufferSize, "%04x:%02x:%02x.%01x",
                  PciInfo.pci_domain, PciInfo.pci_bus, PciInfo.pci_device,
                  PciInfo.pci_function);
    return ReturnValue(AddressBuffer);
  }
  case UR_DEVICE_INFO_GPU_EU_COUNT: {
    /* The EU count can be queried using CL_DEVICE_MAX_COMPUTE_UNITS for Intel
     * GPUs. */

    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }

    cl_device_type CLType;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                         sizeof(cl_device_type), &CLType,
                                         nullptr));
    if (!(CLType & CL_DEVICE_TYPE_GPU)) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }

    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_MAX_COMPUTE_UNITS, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GPU_EU_SLICES: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NUM_SLICES_INTEL, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(
        hDevice->CLDevice, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, propSize,
        pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_NUM_THREADS_PER_EU_INTEL,
                                         propSize, pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_IP_VERSION: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_device_attribute_query"}, Supported));
    if (!Supported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_IP_VERSION_INTEL, propSize,
                                         pPropValue, pPropSizeRet));

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    const cl_device_info info_name = CL_DEVICE_SUB_GROUP_SIZES_INTEL;
    bool isExtensionSupported = false;
    if (hDevice->checkDeviceExtensions({"cl_intel_required_subgroup_size"},
                                       isExtensionSupported) !=
            UR_RESULT_SUCCESS ||
        !isExtensionSupported) {
      std::vector<uint32_t> aThreadIsItsOwnSubGroup({1});
      return ReturnValue(aThreadIsItsOwnSubGroup.data(),
                         aThreadIsItsOwnSubGroup.size());
    }

    // Have to convert size_t to uint32_t
    size_t SubGroupSizesSize = 0;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, info_name, 0,
                                         nullptr, &SubGroupSizesSize));
    std::vector<size_t> SubGroupSizes(SubGroupSizesSize / sizeof(size_t));
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, info_name,
                                         SubGroupSizesSize,
                                         SubGroupSizes.data(), nullptr));
    return ReturnValue.template operator()<uint32_t>(SubGroupSizes.data(),
                                                     SubGroupSizes.size());
  }

  case UR_DEVICE_INFO_UUID: {
    // Use the cl_khr_device_uuid extension, if available.
    bool isKhrDeviceUuidSupported = false;
    if (hDevice->checkDeviceExtensions({"cl_khr_device_uuid"},
                                       isKhrDeviceUuidSupported) !=
            UR_RESULT_SUCCESS ||
        !isKhrDeviceUuidSupported) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    static_assert(CL_UUID_SIZE_KHR == 16);
    std::array<uint8_t, CL_UUID_SIZE_KHR> UUID{};
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_UUID_KHR,
                                         UUID.size(), UUID.data(), nullptr));
    return ReturnValue(UUID);
  }
  case UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP: {
    bool Is2DBlockIOSupported = false;
    if (hDevice->checkDeviceExtensions({"cl_intel_subgroup_2d_block_io"},
                                       Is2DBlockIOSupported) !=
            UR_RESULT_SUCCESS ||
        !Is2DBlockIOSupported) {
      return ReturnValue(
          static_cast<ur_exp_device_2d_block_array_capability_flags_t>(0));
    }
    return ReturnValue(UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_LOAD |
                       UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_STORE);
  }
  case UR_DEVICE_INFO_BFLOAT16_CONVERSIONS_NATIVE: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_bfloat16_conversions"}, Supported));
    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP: {
    cl_device_id Dev = hDevice->CLDevice;
    size_t ExtSize = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(Dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ExtSize));

    std::string ExtStr(ExtSize, '\0');
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(Dev, CL_DEVICE_EXTENSIONS, ExtSize,
                                         ExtStr.data(), nullptr));

    // cl_khr_command_buffer is required for UR command-buffer support
    cl_device_command_buffer_capabilities_khr Caps = 0;
    if (ExtStr.find("cl_khr_command_buffer") != std::string::npos) {
      // A UR command-buffer user needs to be able to enqueue another
      // submission of the same UR command-buffer without having to manually
      // check if the first submission has completed.
      CL_RETURN_ON_FAILURE(
          clGetDeviceInfo(Dev, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                          sizeof(Caps), &Caps, nullptr));
    }

    return ReturnValue(
        0 != (Caps & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR));
  }
  case UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP: {
    cl_device_id Dev = hDevice->CLDevice;
    ur_device_command_buffer_update_capability_flags_t UpdateCapabilities = 0;
    CL_RETURN_ON_FAILURE(
        getDeviceCommandBufferUpdateCapabilities(Dev, UpdateCapabilities));
    return ReturnValue(UpdateCapabilities);
  }
  case UR_DEVICE_INFO_PROGRAM_SET_SPECIALIZATION_CONSTANTS: {
    return ReturnValue(
        ur::cl::getAdapter()->clSetProgramSpecializationConstant != nullptr);
  }
  case UR_DEVICE_INFO_USE_NATIVE_ASSERT: {
    bool Supported = false;
    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions(
        {"cl_intel_devicelib_assert"}, Supported));
    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_EXTENSIONS: {
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice,
                                         CL_DEVICE_EXTENSIONS, propSize,
                                         pPropValue, pPropSizeRet));
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_MULTI_DEVICE_COMPILE_SUPPORT_EXP:
    return ReturnValue(true);
  case UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES:
    return ReturnValue(0);
  // TODO: We can't query to check if these are supported, they will need to be
  // manually updated if support is ever implemented.
  case UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS:
  case UR_DEVICE_INFO_ASYNC_BARRIER:
  case UR_DEVICE_INFO_USM_POOL_SUPPORT: // end of TODO
  case UR_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP:
  case UR_DEVICE_INFO_COMMAND_BUFFER_SUBGRAPH_SUPPORT_EXP:
  case UR_DEVICE_INFO_LOW_POWER_EVENTS_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP:
  case UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP:
  case UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP:
  case UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP:
  case UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP:
  case UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP:
  case UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP:
  case UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_SUPPORT_EXP:
  case UR_DEVICE_INFO_BINDLESS_IMAGES_GATHER_SUPPORT_EXP:
  case UR_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP:
  case UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP:
  /* TODO: Check regularly to see if support is enabled in OpenCL. Intel GPU
   * EU device-specific information extensions. Some of the queries are
   * enabled by cl_intel_device_attribute_query extension, but it's not yet in
   * the Registry. */
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH:
  /* These enums have no equivalent in OpenCL */
  case UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP:
  case UR_DEVICE_INFO_GLOBAL_MEM_FREE:
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH:
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
  case UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS:
  case UR_DEVICE_INFO_FAN_SPEED:
  case UR_DEVICE_INFO_MIN_POWER_LIMIT:
  case UR_DEVICE_INFO_MAX_POWER_LIMIT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urDevicePartition(
    ur_device_handle_t hDevice,
    const ur_device_partition_properties_t *pProperties, uint32_t NumDevices,
    ur_device_handle_t *phSubDevices, uint32_t *pNumDevicesRet) {

  std::vector<cl_device_partition_property> CLProperties(
      pProperties->PropCount + 2);

  /* The type must be the same for all properties since OpenCL doesn't support
   * property lists with multiple types */
  CLProperties[0] =
      static_cast<cl_device_partition_property>(pProperties->pProperties->type);

  for (uint32_t i = 0; i < pProperties->PropCount; ++i) {
    cl_device_partition_property CLProperty;
    switch (pProperties->pProperties->type) {
    case UR_DEVICE_PARTITION_EQUALLY: {
      CLProperty = static_cast<cl_device_partition_property>(
          pProperties->pProperties[i].value.equally);
      break;
    }
    case UR_DEVICE_PARTITION_BY_COUNTS: {
      CLProperty = static_cast<cl_device_partition_property>(
          pProperties->pProperties[i].value.count);
      break;
    }
    case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: {
      CLProperty = static_cast<cl_device_partition_property>(
          pProperties->pProperties[i].value.affinity_domain);
      break;
    }
    default: {
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    }
    CLProperties[i + 1] = CLProperty;
  }

  /* Terminate the list with 0 */
  CLProperties[CLProperties.size() - 1] = 0;

  cl_uint CLNumDevicesRet;
  CL_RETURN_ON_FAILURE(clCreateSubDevices(
      hDevice->CLDevice, CLProperties.data(), 0, nullptr, &CLNumDevicesRet));

  if (pNumDevicesRet) {
    *pNumDevicesRet = CLNumDevicesRet;
  }

  /*If NumDevices is less than the number of sub-devices available, then the
   * function shall only retrieve that number of sub-devices. */
  if (phSubDevices) {
    std::vector<cl_device_id> CLSubDevices(CLNumDevicesRet);
    CL_RETURN_ON_FAILURE(
        clCreateSubDevices(hDevice->CLDevice, CLProperties.data(),
                           CLNumDevicesRet, CLSubDevices.data(), nullptr));
    for (uint32_t i = 0; i < std::min(CLNumDevicesRet, NumDevices); i++) {
      try {
        auto URSubDevice = std::make_unique<ur_device_handle_t_>(
            CLSubDevices[i], hDevice->Platform, hDevice);
        phSubDevices[i] = URSubDevice.release();
      } catch (std::bad_alloc &) {
        // Delete all the successfully created subdevices before the failed one.
        for (uint32_t j = 0; j < i; j++) {
          delete phSubDevices[j];
        }
        return UR_RESULT_ERROR_OUT_OF_RESOURCES;
      } catch (...) {
        // Delete all the successfully created subdevices before the failed one.
        for (uint32_t j = 0; j < i; j++) {
          delete phSubDevices[j];
        }
        return UR_RESULT_ERROR_UNKNOWN;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

// Root devices ref count are unchanged through out the program lifetime.
UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t hDevice) {
  if (hDevice->ParentDevice) {
    hDevice->incrementReferenceCount();
  }

  return UR_RESULT_SUCCESS;
}

// Root devices ref count are unchanged through out the program lifetime.
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(ur_device_handle_t hDevice) {
  if (hDevice->ParentDevice) {
    if (hDevice->decrementReferenceCount() == 0) {
      delete hDevice;
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeDevice) {

  *phNativeDevice = reinterpret_cast<ur_native_handle_t>(hDevice->CLDevice);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_adapter_handle_t,
    const ur_device_native_properties_t *pProperties,
    ur_device_handle_t *phDevice) {

  auto SetDeviceProps = [&]() {
    (*phDevice)->IsNativeHandleOwned =
        pProperties ? pProperties->isNativeHandleOwned : false;
  };

  cl_device_id NativeHandle = reinterpret_cast<cl_device_id>(hNativeDevice);

  uint32_t NumPlatforms = 0;
  UR_RETURN_ON_FAILURE(urPlatformGet(nullptr, 0, nullptr, &NumPlatforms));
  std::vector<ur_platform_handle_t> Platforms(NumPlatforms);
  UR_RETURN_ON_FAILURE(
      urPlatformGet(nullptr, NumPlatforms, Platforms.data(), nullptr));

  for (uint32_t i = 0; i < NumPlatforms; i++) {
    uint32_t NumDevices = 0;
    UR_RETURN_ON_FAILURE(
        urDeviceGet(Platforms[i], UR_DEVICE_TYPE_ALL, 0, nullptr, &NumDevices));
    std::vector<ur_device_handle_t> Devices(NumDevices);
    UR_RETURN_ON_FAILURE(urDeviceGet(Platforms[i], UR_DEVICE_TYPE_ALL,
                                     NumDevices, Devices.data(), nullptr));

    for (auto &Device : Devices) {
      if (Device->CLDevice == NativeHandle) {
        *phDevice = Device;
        SetDeviceProps();
        return UR_RESULT_SUCCESS;
      }
    }
  }

  // Handle sub-devices by storing/querying a map stored in the platform
  cl_device_id Parent = nullptr;
  CL_RETURN_ON_FAILURE(clGetDeviceInfo(NativeHandle, CL_DEVICE_PARENT_DEVICE,
                                       sizeof(Parent), &Parent, nullptr));
  if (Parent != nullptr) {
    ur_device_handle_t ParentUrHandle;
    // This will either create a new device handle, or return an existing one
    UR_RETURN_ON_FAILURE(urDeviceCreateWithNativeHandle(
        reinterpret_cast<ur_native_handle_t>(Parent), nullptr, nullptr,
        &ParentUrHandle));

    ur_platform_handle_t PlatformHandle = ParentUrHandle->Platform;
    assert(PlatformHandle);

    {
      std::lock_guard lock{PlatformHandle->SubDevicesLock};

      if (PlatformHandle->SubDevices.count(NativeHandle)) {
        *phDevice = PlatformHandle->SubDevices[NativeHandle];
      } else {
        *phDevice = std::make_unique<ur_device_handle_t_>(
                        NativeHandle, PlatformHandle, ParentUrHandle)
                        .release();
        PlatformHandle->SubDevices[NativeHandle] = *phDevice;
      }
    }

    SetDeviceProps();
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_INVALID_DEVICE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, uint64_t *pDeviceTimestamp,
    uint64_t *pHostTimestamp) {
  oclv::OpenCLVersion DevVer, PlatVer;
  cl_device_id DeviceId = hDevice->CLDevice;

  // TODO: Cache OpenCL version for each device and platform
  auto RetErr = hDevice->getDeviceVersion(DevVer);
  CL_RETURN_ON_FAILURE(RetErr);

  RetErr = hDevice->Platform->getPlatformVersion(PlatVer);

  if (PlatVer < oclv::V2_1 || DevVer < oclv::V2_1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  if (pDeviceTimestamp) {
    uint64_t Dummy;
    clGetDeviceAndHostTimer(DeviceId, pDeviceTimestamp,
                            pHostTimestamp == nullptr ? &Dummy
                                                      : pHostTimestamp);

  } else if (pHostTimestamp) {
    clGetHostTimer(DeviceId, pHostTimestamp);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {
  // TODO: this is a bare-bones implementation for choosing a device image
  // that would be compatible with the targeted device. An AOT-compiled
  // image is preferred over SPIR-V for known devices (i.e. Intel devices)
  // The implementation makes no effort to differentiate between multiple images
  // for the given device, and simply picks the first one compatible
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_ERROR_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  // Choose the binary target for the provided device
  const char *ImageTarget = nullptr;
  // Get the type of the device
  cl_device_type DeviceType;
  constexpr uint32_t InvalidInd = std::numeric_limits<uint32_t>::max();
  cl_int RetErr = clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                  sizeof(cl_device_type), &DeviceType, nullptr);
  if (RetErr != CL_SUCCESS) {
    *pSelectedBinary = InvalidInd;
    CL_RETURN_ON_FAILURE(RetErr);
  }

  switch (DeviceType) {
    // TODO: Factor out vendor specifics into a separate source
    // E.g. sycl/source/detail/vendor/intel/detail/pi_opencl.cpp?

    // We'll attempt to find an image that was AOT-compiled
    // from a SPIR-V image into an image specific for:

  case CL_DEVICE_TYPE_CPU: // OpenCL 64-bit CPU
    ImageTarget = UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64;
    break;
  case CL_DEVICE_TYPE_GPU: // OpenCL 64-bit GEN GPU
    ImageTarget = UR_DEVICE_BINARY_TARGET_SPIRV64_GEN;
    break;
  case CL_DEVICE_TYPE_ACCELERATOR: // OpenCL 64-bit FPGA
    ImageTarget = UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA;
    break;
  default:
    // Otherwise, we'll attempt to find and JIT-compile
    // a device-independent SPIR-V image
    ImageTarget = UR_DEVICE_BINARY_TARGET_SPIRV64;
    break;
  }

  // Find the appropriate device image, fallback to spirv if not found
  uint32_t Fallback = InvalidInd;
  for (uint32_t i = 0; i < NumBinaries; ++i) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec, ImageTarget) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
    if (strcmp(pBinaries[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      Fallback = i;
  }
  // Points to a spirv image, if such indeed was found
  if ((*pSelectedBinary = Fallback) != InvalidInd)
    return UR_RESULT_SUCCESS;
  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}
