//===--------- kernel.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "kernel.hpp"
#include "memory.hpp"

#include "../device.hpp"
#include "../platform.hpp"
#include "../program.hpp"
#include "../ur_interface_loader.hpp"

ur_single_device_kernel_t::ur_single_device_kernel_t(ur_device_handle_t hDevice,
                                                     ze_kernel_handle_t hKernel,
                                                     bool ownZeHandle)
    : hDevice(hDevice), hKernel(hKernel, ownZeHandle) {
  zeKernelProperties.Compute =
      [hKernel = hKernel](ze_kernel_properties_t &properties) {
        ZE_CALL_NOCHECK(zeKernelGetProperties, (hKernel, &properties));
      };
}

ur_result_t ur_single_device_kernel_t::release() {
  hKernel.reset();
  return UR_RESULT_SUCCESS;
}

ur_kernel_handle_t_::ur_kernel_handle_t_(ur_program_handle_t hProgram,
                                         const char *kernelName)
    : hProgram(hProgram),
      deviceKernels(hProgram->Context->getPlatform()->getNumDevices()) {
  ur::level_zero::urProgramRetain(hProgram);

  for (auto [zeDevice, zeModule] : hProgram->ZeModuleMap) {
    ZeStruct<ze_kernel_desc_t> zeKernelDesc;
    zeKernelDesc.pKernelName = kernelName;

    ze_kernel_handle_t zeKernel;
    ZE2UR_CALL_THROWS(zeKernelCreate, (zeModule, &zeKernelDesc, &zeKernel));

    auto urDevice = std::find_if(hProgram->Context->getDevices().begin(),
                                 hProgram->Context->getDevices().end(),
                                 [zeDevice = zeDevice](const auto &urDevice) {
                                   return urDevice->ZeDevice == zeDevice;
                                 });
    assert(urDevice != hProgram->Context->getDevices().end());
    auto deviceId = (*urDevice)->Id.value();

    deviceKernels[deviceId].emplace(*urDevice, zeKernel, true);
  }
  completeInitialization();
}

ur_kernel_handle_t_::ur_kernel_handle_t_(
    ur_native_handle_t hNativeKernel, ur_program_handle_t hProgram,
    const ur_kernel_native_properties_t *pProperties)
    : hProgram(hProgram), deviceKernels(1) {
  ze_kernel_handle_t zeKernel = ur_cast<ze_kernel_handle_t>(hNativeKernel);

  if (!zeKernel) {
    throw UR_RESULT_ERROR_INVALID_KERNEL;
  }

  deviceKernels.back().emplace(nullptr, zeKernel,
                               pProperties->isNativeHandleOwned);
  completeInitialization();
}

ur_result_t ur_kernel_handle_t_::release() {
  // manually release kernels to allow errors to be propagated
  for (auto &singleDeviceKernelOpt : deviceKernels) {
    if (singleDeviceKernelOpt.has_value()) {
      singleDeviceKernelOpt.value().hKernel.reset();
    }
  }

  UR_CALL_THROWS(ur::level_zero::urProgramRelease(hProgram));

  return UR_RESULT_SUCCESS;
}

void ur_kernel_handle_t_::completeInitialization() {
  // Cache kernel name. Should be the same for all devices
  assert(deviceKernels.size() > 0);
  auto nonEmptyKernel =
      std::find_if(deviceKernels.begin(), deviceKernels.end(),
                   [](const auto &kernel) { return kernel.has_value(); });

  zeKernelName.Compute = [kernel =
                              &nonEmptyKernel->value()](std::string &name) {
    size_t size = 0;
    ZE_CALL_NOCHECK(zeKernelGetName, (kernel->hKernel.get(), &size, nullptr));
    name.resize(size);
    ZE_CALL_NOCHECK(zeKernelGetName,
                    (kernel->hKernel.get(), &size, name.data()));
  };
}

ze_kernel_handle_t
ur_kernel_handle_t_::getZeHandle(ur_device_handle_t hDevice) {
  // root-device's kernel can be submitted to a sub-device's queue
  if (hDevice->isSubDevice()) {
    hDevice = hDevice->RootDevice;
  }

  if (deviceKernels.size() == 1) {
    assert(deviceKernels[0].has_value());
    assert(deviceKernels[0].value().hKernel.get());

    auto &kernel = deviceKernels[0].value();

    // hDevice is nullptr for native handle
    if ((kernel.hDevice != nullptr && kernel.hDevice != hDevice)) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }

    return kernel.hKernel.get();
  }

  if (!deviceKernels[hDevice->Id.value()].has_value()) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  assert(deviceKernels[hDevice->Id.value()].value().hKernel.get());

  return deviceKernels[hDevice->Id.value()].value().hKernel.get();
}

const std::string &ur_kernel_handle_t_::getName() const {
  return *zeKernelName.operator->();
}

const ze_kernel_properties_t &
ur_kernel_handle_t_::getProperties(ur_device_handle_t hDevice) const {
  if (!deviceKernels[hDevice->Id.value()].has_value()) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  assert(deviceKernels[hDevice->Id.value()].value().hKernel.get());

  return *deviceKernels[hDevice->Id.value()]
              .value()
              .zeKernelProperties.
              operator->();
}

ur_result_t ur_kernel_handle_t_::setArgValue(
    uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties,
    const void *pArgValue) {
  std::ignore = pProperties;

  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  if (argSize == sizeof(void *) && pArgValue &&
      *(void **)(const_cast<void *>(pArgValue)) == nullptr) {
    pArgValue = nullptr;
  }

  std::scoped_lock<ur_shared_mutex> guard(Mutex);

  for (auto &singleDeviceKernel : deviceKernels) {
    if (!singleDeviceKernel.has_value()) {
      continue;
    }

    ZE2UR_CALL(zeKernelSetArgumentValue,
               (singleDeviceKernel.value().hKernel.get(), argIndex, argSize,
                pArgValue));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_kernel_handle_t_::setArgPointer(
    uint32_t argIndex, const ur_kernel_arg_pointer_properties_t *pProperties,
    const void *pArgValue) {
  std::ignore = pProperties;

  // KernelSetArgValue is expecting a pointer to the argument
  return setArgValue(argIndex, sizeof(const void *), nullptr, &pArgValue);
}

ur_program_handle_t ur_kernel_handle_t_::getProgramHandle() const {
  return hProgram;
}

ur_result_t ur_kernel_handle_t_::setExecInfo(ur_kernel_exec_info_t propName,
                                             const void *pPropValue) {
  std::scoped_lock<ur_shared_mutex> Guard(Mutex);

  for (auto &kernel : deviceKernels) {
    if (!kernel.has_value())
      continue;
    if (propName == UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS &&
        *(static_cast<const ur_bool_t *>(pPropValue)) == true) {
      // The whole point for users really was to not need to know anything
      // about the types of allocations kernel uses. So in DPC++ we always
      // just set all 3 modes for each kernel.
      ze_kernel_indirect_access_flags_t indirectFlags =
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
      ZE2UR_CALL(zeKernelSetIndirectAccess,
                 (kernel->hKernel.get(), indirectFlags));
    } else if (propName == UR_KERNEL_EXEC_INFO_CACHE_CONFIG) {
      ze_cache_config_flag_t zeCacheConfig{};
      auto cacheConfig =
          *(static_cast<const ur_kernel_cache_config_t *>(pPropValue));
      if (cacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_SLM)
        zeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_SLM;
      else if (cacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_DATA)
        zeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_DATA;
      else if (cacheConfig == UR_KERNEL_CACHE_CONFIG_DEFAULT)
        zeCacheConfig = static_cast<ze_cache_config_flag_t>(0);
      else
        // Unexpected cache configuration value.
        return UR_RESULT_ERROR_INVALID_VALUE;
      ZE2UR_CALL(zeKernelSetCacheConfig,
                 (kernel->hKernel.get(), zeCacheConfig););
    } else {
      logger::error("urKernelSetExecInfo: unsupported ParamName");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  return UR_RESULT_SUCCESS;
}

std::vector<ur_device_handle_t> ur_kernel_handle_t_::getDevices() const {
  std::vector<ur_device_handle_t> devices;
  for (size_t i = 0; i < deviceKernels.size(); ++i) {
    if (deviceKernels[i].has_value()) {
      devices.push_back(deviceKernels[i].value().hDevice);
    }
  }
  return devices;
}

namespace ur::level_zero {
ur_result_t urKernelCreate(ur_program_handle_t hProgram,
                           const char *pKernelName,
                           ur_kernel_handle_t *phKernel) {
  *phKernel = new ur_kernel_handle_t_(hProgram, pKernelName);
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
  hKernel->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
  if (!hKernel->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  hKernel->release();
  delete hKernel;

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgValue");
  return hKernel->setArgValue(argIndex, argSize, pProperties, pArgValue);
}

ur_result_t urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgPointer");
  return hKernel->setArgPointer(argIndex, pProperties, pArgValue);
}

ur_result_t
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *pProperties,
                     ur_mem_handle_t hArgValue) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgMemObj");

  // TODO: support properties
  std::ignore = pProperties;

  auto kernelDevices = hKernel->getDevices();
  if (kernelDevices.size() == 1) {
    auto zePtr = hArgValue->getPtr(kernelDevices.front());
    return hKernel->setArgPointer(argIndex, nullptr, zePtr);
  } else {
    // TODO: if devices do not have p2p capabilities, we need to have allocation
    // on each device. Do this the same way as in legacy (keep a pending Args
    // vector and do actual allocation on kernel submission) or allocate the
    // memory immediately (only for small allocations?).

    // Get memory that is accessible by the first device.
    // If kernel is submitted to a different device the memory
    // will be accessed trough the link or migrated in enqueueKernelLaunch.
    auto zePtr = hArgValue->getPtr(kernelDevices.front());
    return hKernel->setArgPointer(argIndex, nullptr, zePtr);
  }
}

ur_result_t
urKernelSetArgLocal(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize,
                    const ur_kernel_arg_local_properties_t *pProperties) {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgLocal");

  std::ignore = pProperties;

  return hKernel->setArgValue(argIndex, argSize, nullptr, nullptr);
}

ur_result_t urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t
        *pProperties, ///< [in][optional] pointer to execution info properties
    const void *pPropValue ///< [in][range(0, propSize)] pointer to memory
                           ///< location holding the property value.
) {
  std::ignore = propSize;
  std::ignore = pProperties;

  return hKernel->setExecInfo(propName, pPropValue);
}

ur_result_t urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_group_info_t
        paramName, ///< [in] name of the work Group property to query
    size_t
        paramValueSize, ///< [in] size of the Kernel Work Group property value
    void *pParamValue,  ///< [in,out][optional][range(0, propSize)] value of the
                        ///< Kernel Work Group property.
    size_t *pParamValueSizeRet ///< [out][optional] pointer to the actual size
                               ///< in bytes of data being queried by propName.
) {
  UrReturnHelper returnValue(paramValueSize, pParamValue, pParamValueSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);
  switch (paramName) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    // TODO: To revisit after level_zero/issues/262 is resolved
    struct {
      size_t Arr[3];
    } GlobalWorkSize = {{(hDevice->ZeDeviceComputeProperties->maxGroupSizeX *
                          hDevice->ZeDeviceComputeProperties->maxGroupCountX),
                         (hDevice->ZeDeviceComputeProperties->maxGroupSizeY *
                          hDevice->ZeDeviceComputeProperties->maxGroupCountY),
                         (hDevice->ZeDeviceComputeProperties->maxGroupSizeZ *
                          hDevice->ZeDeviceComputeProperties->maxGroupCountZ)}};
    return returnValue(GlobalWorkSize);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    ZeStruct<ze_kernel_max_group_size_properties_ext_t> workGroupProperties;
    workGroupProperties.maxGroupSize = 0;

    ZeStruct<ze_kernel_properties_t> kernelProperties;
    kernelProperties.pNext = &workGroupProperties;

    auto zeDevice = hKernel->getZeHandle(hDevice);
    if (zeDevice) {
      auto zeResult =
          ZE_CALL_NOCHECK(zeKernelGetProperties, (zeDevice, &kernelProperties));
      if (zeResult == ZE_RESULT_SUCCESS &&
          workGroupProperties.maxGroupSize != 0) {
        return returnValue(workGroupProperties.maxGroupSize);
      }
      return returnValue(
          uint64_t{hDevice->ZeDeviceComputeProperties->maxTotalGroupSize});
    }
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    auto props = hKernel->getProperties(hDevice);
    struct {
      size_t Arr[3];
    } WgSize = {{props.requiredGroupSizeX, props.requiredGroupSizeY,
                 props.requiredGroupSizeZ}};
    return returnValue(WgSize);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    auto props = hKernel->getProperties(hDevice);
    return returnValue(uint32_t{props.localMemSize});
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    return returnValue(
        size_t{hDevice->ZeDeviceProperties->physicalEUSimdWidth});
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    auto props = hKernel->getProperties(hDevice);
    return returnValue(uint32_t{props.privateMemSize});
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE:
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE:
    // No corresponding enumeration in Level Zero
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default: {
    logger::error(
        "Unknown ParamName in urKernelGetGroupInfo: ParamName={}(0x{})",
        paramName, logger::toHex(paramName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t
        propName,     ///< [in] name of the SubGroup property to query
    size_t propSize,  ///< [in] size of the Kernel SubGroup property value
    void *pPropValue, ///< [in,out][range(0, propSize)][optional] value of the
                      ///< Kernel SubGroup property.
    size_t *pPropSizeRet ///< [out][optional] pointer to the actual size in
                         ///< bytes of data being queried by propName.
) {
  std::ignore = hDevice;

  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  auto props = hKernel->getProperties(hDevice);

  std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);
  if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE) {
    returnValue(uint32_t{props.maxSubgroupSize});
  } else if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS) {
    returnValue(uint32_t{props.maxNumSubgroups});
  } else if (propName == UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS) {
    returnValue(uint32_t{props.requiredNumSubGroups});
  } else if (propName == UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL) {
    returnValue(uint32_t{props.requiredSubgroupSize});
  } else {
    die("urKernelGetSubGroupInfo: parameter not implemented");
    return {};
  }
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
