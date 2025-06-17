//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "queue.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "platform.hpp"

cl_command_queue_info mapURQueueInfoToCL(const ur_queue_info_t PropName) {

  switch (PropName) {
  case UR_QUEUE_INFO_CONTEXT:
    return CL_QUEUE_CONTEXT;
  case UR_QUEUE_INFO_DEVICE:
    return CL_QUEUE_DEVICE;
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return CL_QUEUE_DEVICE_DEFAULT;
  case UR_QUEUE_INFO_FLAGS:
    return CL_QUEUE_PROPERTIES;
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return CL_QUEUE_REFERENCE_COUNT;
  case UR_QUEUE_INFO_SIZE:
    return CL_QUEUE_SIZE;
  default:
    return -1;
  }
}

cl_command_queue_properties
convertURQueuePropertiesToCL(const ur_queue_properties_t *URQueueProperties) {
  cl_command_queue_properties CLCommandQueueProperties = 0;

  if (URQueueProperties->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    CLCommandQueueProperties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    CLCommandQueueProperties |= CL_QUEUE_PROFILING_ENABLE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE) {
    CLCommandQueueProperties |= CL_QUEUE_ON_DEVICE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE_DEFAULT) {
    CLCommandQueueProperties |= CL_QUEUE_ON_DEVICE_DEFAULT;
  }

  return CLCommandQueueProperties;
}

ur_queue_flags_t
mapCLQueuePropsToUR(const cl_command_queue_properties &Properties) {
  ur_queue_flags_t Flags = 0;
  if (Properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    Flags |= UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (Properties & CL_QUEUE_PROFILING_ENABLE) {
    Flags |= UR_QUEUE_FLAG_PROFILING_ENABLE;
  }
  if (Properties & CL_QUEUE_ON_DEVICE) {
    Flags |= UR_QUEUE_FLAG_ON_DEVICE;
  }
  if (Properties & CL_QUEUE_ON_DEVICE_DEFAULT) {
    Flags |= UR_QUEUE_FLAG_ON_DEVICE_DEFAULT;
  }
  return Flags;
}

ur_result_t ur_queue_handle_t_::makeWithNative(native_type NativeQueue,
                                               ur_context_handle_t Context,
                                               ur_device_handle_t Device,
                                               ur_queue_handle_t &Queue) {
  try {
    cl_context CLContext;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        NativeQueue, CL_QUEUE_CONTEXT, sizeof(CLContext), &CLContext, nullptr));
    cl_device_id CLDevice;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        NativeQueue, CL_QUEUE_DEVICE, sizeof(CLDevice), &CLDevice, nullptr));
    if (Context->CLContext != CLContext) {
      return UR_RESULT_ERROR_INVALID_CONTEXT;
    }
    if (Device) {
      if (Device->CLDevice != CLDevice) {
        return UR_RESULT_ERROR_INVALID_DEVICE;
      }
    } else {
      ur_native_handle_t hNativeHandle =
          reinterpret_cast<ur_native_handle_t>(CLDevice);
      UR_RETURN_ON_FAILURE(urDeviceCreateWithNativeHandle(
          hNativeHandle, nullptr, nullptr, &Device));
    }
    auto URQueue = std::make_unique<ur_queue_handle_t_>(NativeQueue, Context,
                                                        Device, false);
    Queue = URQueue.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {

  ur_platform_handle_t CurPlatform = hDevice->Platform;

  cl_command_queue_properties CLProperties =
      pProperties ? convertURQueuePropertiesToCL(pProperties) : 0;

  // Properties supported by OpenCL backend.
  const cl_command_queue_properties SupportByOpenCL =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE |
      CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;

  oclv::OpenCLVersion Version;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(CurPlatform->getPlatformVersion(Version),
                                    phQueue);

  cl_int RetErr = CL_INVALID_OPERATION;

  bool InOrder = !(CLProperties & SupportByOpenCL &
                   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  if (Version < oclv::V2_0) {
    cl_command_queue Queue =
        clCreateCommandQueue(hContext->CLContext, hDevice->CLDevice,
                             CLProperties & SupportByOpenCL, &RetErr);
    CL_RETURN_ON_FAILURE(RetErr);
    try {
      auto URQueue = std::make_unique<ur_queue_handle_t_>(Queue, hContext,
                                                          hDevice, InOrder);
      *phQueue = URQueue.release();
    } catch (std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }

    return UR_RESULT_SUCCESS;
  }

  /* TODO: Add support for CL_QUEUE_PRIORITY_KHR */
  cl_queue_properties CreationFlagProperties[] = {
      CL_QUEUE_PROPERTIES, CLProperties & SupportByOpenCL, 0};
  cl_command_queue Queue = clCreateCommandQueueWithProperties(
      hContext->CLContext, hDevice->CLDevice, CreationFlagProperties, &RetErr);
  CL_RETURN_ON_FAILURE(RetErr);
  try {
    auto URQueue =
        std::make_unique<ur_queue_handle_t_>(Queue, hContext, hDevice, InOrder);
    *phQueue = URQueue.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  cl_command_queue_info CLCommandQueueInfo = mapURQueueInfoToCL(propName);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  if (propName == UR_QUEUE_INFO_EMPTY) {
    if (!hQueue->LastEvent) {
      // OpenCL doesn't provide API to check the status of the queue.
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    } else {
      ur_event_status_t Status;
      UR_RETURN_ON_FAILURE(urEventGetInfo(
          hQueue->LastEvent, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
          sizeof(ur_event_status_t), (void *)&Status, nullptr));
      if (Status == UR_EVENT_STATUS_COMPLETE) {
        return ReturnValue(true);
      }
      return ReturnValue(false);
    }
  }
  switch (propName) {
  case UR_QUEUE_INFO_CONTEXT: {
    return ReturnValue(hQueue->Context);
  }
  case UR_QUEUE_INFO_DEVICE: {
    return ReturnValue(hQueue->Device);
  }
  case UR_QUEUE_INFO_DEVICE_DEFAULT: {
    size_t CheckPropSize = 0;
    ur_queue_handle_t_::native_type NewDefault = 0;
    cl_int RetErr = clGetCommandQueueInfo(
        hQueue->CLQueue, CL_QUEUE_DEVICE_DEFAULT, sizeof(CheckPropSize),
        &NewDefault, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    CL_RETURN_ON_FAILURE(RetErr);

    // If we have an existing default device, release it
    if (hQueue->DeviceDefault.has_value()) {
      urQueueRelease(*hQueue->DeviceDefault);
      hQueue->DeviceDefault.reset();
    }

    // Then either return this queue (if it is the device default) or create a
    // new handle to hold onto
    if (hQueue->CLQueue == NewDefault) {
      return ReturnValue(hQueue);
    } else {
      ur_queue_handle_t NewHandle;
      UR_RETURN_ON_FAILURE(ur_queue_handle_t_::makeWithNative(
          NewDefault, hQueue->Context, hQueue->Device, NewHandle));
      hQueue->DeviceDefault = NewHandle;
      return ReturnValue(NewHandle);
    }
    break;
  }

  // Unfortunately the size of cl_bitfield (unsigned long) doesn't line up with
  // our enums (forced to be sizeof(uint32_t)) so this needs special handling.
  case UR_QUEUE_INFO_FLAGS: {
    cl_command_queue_properties QueueProperties = 0;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        hQueue->CLQueue, CLCommandQueueInfo, sizeof(QueueProperties),
        &QueueProperties, nullptr));

    return ReturnValue(mapCLQueuePropsToUR(QueueProperties));
  }
  case UR_QUEUE_INFO_REFERENCE_COUNT: {
    return ReturnValue(hQueue->getReferenceCount());
  }
  default: {
    size_t CheckPropSize = 0;
    cl_int RetErr = clGetCommandQueueInfo(hQueue->CLQueue, CLCommandQueueInfo,
                                          propSize, pPropValue, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    CL_RETURN_ON_FAILURE(RetErr);
    if (pPropSizeRet) {
      *pPropSizeRet = CheckPropSize;
    }
  }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(ur_queue_handle_t hQueue, ur_queue_native_desc_t *,
                       ur_native_handle_t *phNativeQueue) {
  return getNativeHandle(hQueue->CLQueue, phNativeQueue);
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {

  cl_command_queue NativeHandle =
      reinterpret_cast<cl_command_queue>(hNativeQueue);

  UR_RETURN_ON_FAILURE(ur_queue_handle_t_::makeWithNative(
      NativeHandle, hContext, hDevice, *phQueue));

  (*phQueue)->IsNativeHandleOwned =
      pProperties ? pProperties->isNativeHandleOwned : false;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  cl_int RetErr = clFinish(hQueue->CLQueue);
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  cl_int RetErr = clFinish(hQueue->CLQueue);
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  hQueue->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  if (hQueue->decrementReferenceCount() == 0) {
    delete hQueue;
  }
  return UR_RESULT_SUCCESS;
}
