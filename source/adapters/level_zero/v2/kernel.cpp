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
#include "../helpers/kernel_helpers.hpp"
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

  for (auto &Dev : hProgram->AssociatedDevices) {
    auto zeDevice = Dev->ZeDevice;
    // Program may be associated with all devices from the context but built
    // only for subset of devices.
    if (hProgram->getState(zeDevice) != ur_program_handle_t_::state::Exe)
      continue;

    auto zeModule = hProgram->getZeModuleHandle(zeDevice);
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
    ur_context_handle_t context,
    const ur_kernel_native_properties_t *pProperties)
    : hProgram(hProgram),
      deviceKernels(context ? context->getPlatform()->getNumDevices() : 0) {
  ur::level_zero::urProgramRetain(hProgram);

  auto ownZeHandle = pProperties ? pProperties->isNativeHandleOwned : false;

  ze_kernel_handle_t zeKernel = ur_cast<ze_kernel_handle_t>(hNativeKernel);

  if (!zeKernel) {
    throw UR_RESULT_ERROR_INVALID_KERNEL;
  }

  for (auto &Dev : context->getDevices()) {
    deviceKernels[*Dev->Id].emplace(Dev, zeKernel, ownZeHandle);

    // owned only by the first entry
    ownZeHandle = false;
  }
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
  nonEmptyKernel =
      &std::find_if(deviceKernels.begin(), deviceKernels.end(),
                    [](const auto &kernel) { return kernel.has_value(); })
           ->value();

  zeCommonProperties.Compute = [kernel = nonEmptyKernel](
                                   common_properties_t &props) {
    size_t size = 0;
    ZE_CALL_NOCHECK(zeKernelGetName, (kernel->hKernel.get(), &size, nullptr));
    props.name.resize(size);
    ZE_CALL_NOCHECK(zeKernelGetName,
                    (kernel->hKernel.get(), &size, props.name.data()));
    props.numKernelArgs = kernel->zeKernelProperties->numKernelArgs;
  };
}

size_t ur_kernel_handle_t_::deviceIndex(ur_device_handle_t hDevice) const {
  if (!hDevice) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  // root-device's kernel can be submitted to a sub-device's queue
  if (hDevice->isSubDevice()) {
    hDevice = hDevice->RootDevice;
  }

  if (!deviceKernels[hDevice->Id.value()].has_value()) {
    throw UR_RESULT_ERROR_INVALID_DEVICE;
  }

  assert(deviceKernels[hDevice->Id.value()].value().hDevice == hDevice);
  assert(deviceKernels[hDevice->Id.value()].value().hKernel.get());

  return hDevice->Id.value();
}

ze_kernel_handle_t ur_kernel_handle_t_::getNativeZeHandle() const {
  for (const auto &singleDeviceKernel : deviceKernels) {
    if (singleDeviceKernel.has_value()) {
      return singleDeviceKernel.value().hKernel.get();
    }
  }
  return nullptr;
}

ze_kernel_handle_t
ur_kernel_handle_t_::getZeHandle(ur_device_handle_t hDevice) {
  auto &deviceKernel = deviceKernels[deviceIndex(hDevice)].value();
  return deviceKernel.hKernel.get();
}

ur_kernel_handle_t_::common_properties_t
ur_kernel_handle_t_::getCommonProperties() const {
  return zeCommonProperties.get();
}

const ze_kernel_properties_t &
ur_kernel_handle_t_::getProperties(ur_device_handle_t hDevice) const {
  auto &deviceKernel = deviceKernels[deviceIndex(hDevice)].value();
  return deviceKernel.zeKernelProperties.get();
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

  if (argIndex > zeCommonProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }

  for (auto &singleDeviceKernel : deviceKernels) {
    if (!singleDeviceKernel.has_value()) {
      continue;
    }

    auto zeResult = ZE_CALL_NOCHECK(zeKernelSetArgumentValue,
                                    (singleDeviceKernel.value().hKernel.get(),
                                     argIndex, argSize, pArgValue));

    if (zeResult == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
      return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE;
    } else if (zeResult != ZE_RESULT_SUCCESS) {
      return ze2urResult(zeResult);
    }
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

// Perform any required allocations and set the kernel arguments.
ur_result_t ur_kernel_handle_t_::prepareForSubmission(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const size_t *pGlobalWorkOffset, uint32_t workDim, uint32_t groupSizeX,
    uint32_t groupSizeY, uint32_t groupSizeZ,
    std::function<void(void *, void *, size_t)> migrate) {
  auto hZeKernel = getZeHandle(hDevice);

  if (pGlobalWorkOffset != NULL) {
    UR_CALL(
        setKernelGlobalOffset(hContext, hZeKernel, workDim, pGlobalWorkOffset));
  }

  ZE2UR_CALL(zeKernelSetGroupSize,
             (hZeKernel, groupSizeX, groupSizeY, groupSizeZ));

  for (auto &pending : pending_allocations) {
    auto zePtr = pending.hMem->getDevicePtr(hDevice, pending.mode, 0,
                                            pending.hMem->getSize(), migrate);
    UR_CALL(setArgPointer(pending.argIndex, nullptr, zePtr));
  }
  pending_allocations.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_kernel_handle_t_::addPendingMemoryAllocation(
    pending_memory_allocation_t allocation) {
  if (allocation.argIndex > zeCommonProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }

  pending_allocations.push_back(allocation);

  return UR_RESULT_SUCCESS;
}

std::vector<char> ur_kernel_handle_t_::getSourceAttributes() const {
  uint32_t size;
  ZE2UR_CALL_THROWS(zeKernelGetSourceAttributes,
                    (nonEmptyKernel->hKernel.get(), &size, nullptr));
  std::vector<char> attributes(size);
  char *dataPtr = attributes.data();
  ZE2UR_CALL_THROWS(zeKernelGetSourceAttributes,
                    (nonEmptyKernel->hKernel.get(), &size, &dataPtr));
  return attributes;
}

namespace ur::level_zero {
ur_result_t urKernelCreate(ur_program_handle_t hProgram,
                           const char *pKernelName,
                           ur_kernel_handle_t *phKernel) try {
  *phKernel = new ur_kernel_handle_t_(hProgram, pKernelName);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelGetNativeHandle(ur_kernel_handle_t hKernel,
                                    ur_native_handle_t *phNativeKernel) try {
  // Return the handle of the kernel for the first device
  *phNativeKernel =
      reinterpret_cast<ur_native_handle_t>(hKernel->getNativeZeHandle());
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urKernelCreateWithNativeHandle(ur_native_handle_t hNativeKernel,
                               ur_context_handle_t hContext,
                               ur_program_handle_t hProgram,
                               const ur_kernel_native_properties_t *pProperties,
                               ur_kernel_handle_t *phKernel) try {
  if (!hProgram) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  *phKernel =
      new ur_kernel_handle_t_(hNativeKernel, hProgram, hContext, pProperties);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
    ) try {
  hKernel->RefCount.increment();
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
    ) try {
  if (!hKernel->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  hKernel->release();
  delete hKernel;

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
    ) try {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgValue");

  std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);
  return hKernel->setArgValue(argIndex, argSize, pProperties, pArgValue);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] argument properties
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
    ) try {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgPointer");

  std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);
  return hKernel->setArgPointer(argIndex, pProperties, pArgValue);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

static ur_mem_handle_t_::device_access_mode_t memAccessFromKernelProperties(
    const ur_kernel_arg_mem_obj_properties_t *pProperties) {
  if (pProperties) {
    switch (pProperties->memoryAccess) {
    case UR_MEM_FLAG_READ_WRITE:
      return ur_mem_handle_t_::device_access_mode_t::read_write;
    case UR_MEM_FLAG_WRITE_ONLY:
      return ur_mem_handle_t_::device_access_mode_t::write_only;
    case UR_MEM_FLAG_READ_ONLY:
      return ur_mem_handle_t_::device_access_mode_t::read_only;
    default:
      return ur_mem_handle_t_::device_access_mode_t::read_write;
    }
  }
  return ur_mem_handle_t_::device_access_mode_t::read_write;
}

ur_result_t
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *pProperties,
                     ur_mem_handle_t hArgValue) try {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgMemObj");

  std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);

  UR_CALL(hKernel->addPendingMemoryAllocation(
      {hArgValue, memAccessFromKernelProperties(pProperties), argIndex}));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urKernelSetArgLocal(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize,
                    const ur_kernel_arg_local_properties_t *pProperties) try {
  TRACK_SCOPE_LATENCY("ur_kernel_handle_t_::setArgLocal");

  std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);

  std::ignore = pProperties;

  return hKernel->setArgValue(argIndex, argSize, nullptr, nullptr);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t
        *pProperties, ///< [in][optional] pointer to execution info properties
    const void *pPropValue ///< [in][range(0, propSize)] pointer to memory
                           ///< location holding the property value.
    ) try {
  std::ignore = propSize;
  std::ignore = pProperties;

  std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);

  return hKernel->setExecInfo(propName, pPropValue);
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ) try {
  UrReturnHelper returnValue(paramValueSize, pParamValue, pParamValueSizeRet);

  // No locking needed here, we only read const members
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
    auto zeResult =
        ZE_CALL_NOCHECK(zeKernelGetProperties, (zeDevice, &kernelProperties));
    if (zeResult == ZE_RESULT_SUCCESS &&
        workGroupProperties.maxGroupSize != 0) {
      return returnValue(workGroupProperties.maxGroupSize);
    }
    return returnValue(
        uint64_t{hDevice->ZeDeviceComputeProperties->maxTotalGroupSize});
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
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ) try {
  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  auto props = hKernel->getProperties(hDevice);

  // No locking needed here, we only read const members
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
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urKernelGetInfo(ur_kernel_handle_t hKernel,
                            ur_kernel_info_t paramName, size_t propSize,
                            void *pKernelInfo, size_t *pPropSizeRet) try {

  UrReturnHelper ReturnValue(propSize, pKernelInfo, pPropSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);
  switch (paramName) {
  case UR_KERNEL_INFO_CONTEXT:
    return ReturnValue(
        ur_context_handle_t{hKernel->getProgramHandle()->Context});
  case UR_KERNEL_INFO_PROGRAM:
    return ReturnValue(ur_program_handle_t{hKernel->getProgramHandle()});
  case UR_KERNEL_INFO_FUNCTION_NAME: {
    auto kernelName = hKernel->getCommonProperties().name;
    return ReturnValue(static_cast<const char *>(kernelName.c_str()));
  }
  case UR_KERNEL_INFO_NUM_REGS:
  case UR_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(uint32_t{hKernel->getCommonProperties().numKernelArgs});
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hKernel->RefCount.load()});
  case UR_KERNEL_INFO_ATTRIBUTES: {
    auto attributes = hKernel->getSourceAttributes();
    return ReturnValue(static_cast<const char *>(attributes.data()));
  }
  default:
    logger::error(
        "Unsupported ParamName in urKernelGetInfo: ParamName={}(0x{})",
        paramName, logger::toHex(paramName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
