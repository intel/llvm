//===--------- context.cpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "adapter.hpp"

#include <mutex>
#include <set>
#include <unordered_map>

namespace ur::opencl {

ur_result_t
ur_context_handle_t_::makeWithNative(native_type Ctx, uint32_t DevCount,
                                     const ur_device_handle_t *phDevices,
                                     ur_context_handle_t &Context) {
  try {
    uint32_t CLDeviceCount;
    CL_RETURN_ON_FAILURE(clGetContextInfo(Ctx, CL_CONTEXT_NUM_DEVICES,
                                          sizeof(CLDeviceCount), &CLDeviceCount,
                                          nullptr));
    std::vector<cl_device_id> CLDevices(CLDeviceCount);
    CL_RETURN_ON_FAILURE(clGetContextInfo(
        Ctx, CL_CONTEXT_DEVICES, sizeof(CLDevices), CLDevices.data(), nullptr));
    std::vector<ur::opencl::ur_device_handle_t_ *> URDevices;
    if (DevCount) {
      if (DevCount != CLDeviceCount) {
        return UR_RESULT_ERROR_INVALID_CONTEXT;
      }
      for (uint32_t i = 0; i < DevCount; i++) {
        if (cast(phDevices[i])->CLDevice != CLDevices[i]) {
          return UR_RESULT_ERROR_INVALID_CONTEXT;
        }
        URDevices.push_back(cast(phDevices[i]));
      }
    } else {
      DevCount = CLDeviceCount;
      for (uint32_t i = 0; i < CLDeviceCount; i++) {
        ur_device_handle_t UrDevice = nullptr;
        ur_native_handle_t hNativeHandle =
            reinterpret_cast<ur_native_handle_t>(CLDevices[i]);
        UR_RETURN_ON_FAILURE(ur::opencl::urDeviceCreateWithNativeHandle(
            hNativeHandle, nullptr, nullptr, &UrDevice));
        URDevices.push_back(cast(UrDevice));
      }
    }

    auto URContext =
        std::make_unique<ur_context_handle_t_>(Ctx, DevCount, URDevices.data());
    Context = cast(URContext.release());
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urContextCreate(uint32_t DeviceCount,
                            const ur_device_handle_t *phDevices,
                            const ur_context_properties_t *,
                            ur_context_handle_t *phContext) {

  cl_int Ret;
  std::vector<cl_device_id> CLDevices(DeviceCount);
  std::vector<ur::opencl::ur_device_handle_t_ *> URDevices(DeviceCount);
  for (size_t i = 0; i < DeviceCount; i++) {
    URDevices[i] = cast(phDevices[i]);
    CLDevices[i] = URDevices[i]->CLDevice;
  }

  try {
    cl_context Ctx = clCreateContext(nullptr, static_cast<cl_uint>(DeviceCount),
                                     CLDevices.data(), nullptr, nullptr,
                                     static_cast<cl_int *>(&Ret));
    CL_RETURN_ON_FAILURE(Ret);
    auto URContext = std::make_unique<ur_context_handle_t_>(Ctx, DeviceCount,
                                                            URDevices.data());
    *phContext = cast(URContext.release());
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return mapCLErrorToUR(Ret);
}

ur_result_t urContextGetInfo(ur_context_handle_t hContext,
                             ur_context_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet) {

  auto Context = cast(hContext);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (static_cast<uint32_t>(propName)) {
  /* 2D USM memops are not supported. */
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT: {
    return ReturnValue(false);
  }
  case UR_CONTEXT_INFO_NUM_DEVICES: {
    return ReturnValue(Context->DeviceCount);
  }
  case UR_CONTEXT_INFO_DEVICES: {
    return ReturnValue(&Context->Devices[0], Context->DeviceCount);
  }
  case UR_CONTEXT_INFO_REFERENCE_COUNT: {
    return ReturnValue(Context->RefCount.getCount());
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

ur_result_t urContextRelease(ur_context_handle_t hContext) {
  auto Context = cast(hContext);
  if (Context->RefCount.release()) {
    delete Context;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urContextRetain(ur_context_handle_t hContext) {
  auto Context = cast(hContext);
  Context->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urContextGetNativeHandle(ur_context_handle_t hContext,
                                     ur_native_handle_t *phNativeContext) {

  auto Context = cast(hContext);
  *phNativeContext = reinterpret_cast<ur_native_handle_t>(Context->CLContext);
  return UR_RESULT_SUCCESS;
}

ur_result_t urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, ur_adapter_handle_t, uint32_t numDevices,
    const ur_device_handle_t *phDevices,
    const ur_context_native_properties_t *pProperties,
    ur_context_handle_t *phContext) {

  cl_context NativeHandle = reinterpret_cast<cl_context>(hNativeContext);
  UR_RETURN_ON_FAILURE(ur_context_handle_t_::makeWithNative(
      NativeHandle, numDevices, phDevices, *phContext));
  cast(*phContext)->IsNativeHandleOwned =
      pProperties ? pProperties->isNativeHandleOwned : false;
  return UR_RESULT_SUCCESS;
}

ur_result_t
urContextSetExtendedDeleter(ur_context_handle_t hContext,
                            ur_context_extended_deleter_t pfnDeleter,
                            void *pUserData) {
  if (!cast(ur::cl::getAdapter())->clSetContextDestructorCallbackFn) {
    UR_LOG_L(cast(ur::cl::getAdapter())->log, WARN,
             "clSetContextDestructorCallback not found, consider upgrading the "
             "OpenCL-ICD-Loader to the latest version.");
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  static std::unordered_map<ur_context_handle_t,
                            std::set<ur_context_extended_deleter_t>>
      ContextCallbackMap;
  static std::mutex ContextCallbackMutex;

  {
    std::lock_guard<std::mutex> Lock(ContextCallbackMutex);
    // Callbacks can only be registered once and we need to avoid double
    // allocating.
    if (ContextCallbackMap.count(hContext) &&
        ContextCallbackMap[hContext].count(pfnDeleter)) {
      return UR_RESULT_SUCCESS;
    }

    ContextCallbackMap[hContext].insert(pfnDeleter);
  }

  struct ContextCallback {
    void execute() {
      pfnDeleter(pUserData);
      {
        std::lock_guard<std::mutex> Lock(*CallbackMutex);
        (*CallbackMap)[hContext].erase(pfnDeleter);
        if ((*CallbackMap)[hContext].empty()) {
          CallbackMap->erase(hContext);
        }
      }
      delete this;
    }
    ur_context_handle_t hContext;
    ur_context_extended_deleter_t pfnDeleter;
    void *pUserData;
    std::unordered_map<ur_context_handle_t,
                       std::set<ur_context_extended_deleter_t>> *CallbackMap;
    std::mutex *CallbackMutex;
  };
  auto Callback =
      new ContextCallback({hContext, pfnDeleter, pUserData, &ContextCallbackMap,
                           &ContextCallbackMutex});
  auto ClCallback = [](cl_context, void *pUserData) {
    auto *C = static_cast<ContextCallback *>(pUserData);
    C->execute();
  };
  CL_RETURN_ON_FAILURE(
      cast(ur::cl::getAdapter())
          ->clSetContextDestructorCallbackFn(cast(hContext)->CLContext,
                                             ClCallback, Callback));

  return UR_RESULT_SUCCESS;
}

} // namespace ur::opencl
