/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_ddi.cpp
 *
 */

#include "tsan_ddi.hpp"
#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "tsan_interceptor.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

namespace {

ur_result_t setupContext(ur_context_handle_t Context, uint32_t numDevices,
                         const ur_device_handle_t *phDevices) {
  std::shared_ptr<ContextInfo> CI;
  UR_CALL(getTsanInterceptor()->insertContext(Context, CI));
  for (uint32_t i = 0; i < numDevices; i++) {
    std::shared_ptr<DeviceInfo> DI;
    UR_CALL(getTsanInterceptor()->insertDevice(phDevices[i], DI));
    DI->Type = GetDeviceType(Context, DI->Handle);
    if (DI->Type == DeviceType::UNKNOWN) {
      getContext()->logger.error("Unsupport device");
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }
    if (!DI->Shadow)
      UR_CALL(DI->allocShadowMemory());
    CI->DeviceList.emplace_back(DI->Handle);
  }
  return UR_RESULT_SUCCESS;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL urContextCreate(
    /// [in] the number of devices given in phDevices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] array of handle of devices.
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to context creation properties.
    const ur_context_properties_t *pProperties,
    /// [out] pointer to handle of context object created
    ur_context_handle_t *phContext) {
  getContext()->logger.debug("==== urContextCreate");

  UR_CALL(getContext()->urDdiTable.Context.pfnCreate(numDevices, phDevices,
                                                     pProperties, phContext));

  UR_CALL(setupContext(*phContext, numDevices, phDevices));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the getContext()->
    ur_native_handle_t hNativeContext, ur_adapter_handle_t hAdapter,
    /// [in] number of devices associated with the context
    uint32_t numDevices,
    /// [in][range(0, numDevices)] list of devices associated with the
    /// context
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to native context properties struct
    const ur_context_native_properties_t *pProperties,
    /// [out] pointer to the handle of the context object created.
    ur_context_handle_t *phContext) {
  getContext()->logger.debug("==== urContextCreateWithNativeHandle");

  UR_CALL(getContext()->urDdiTable.Context.pfnCreateWithNativeHandle(
      hNativeContext, hAdapter, numDevices, phDevices, pProperties, phContext));

  UR_CALL(setupContext(*phContext, numDevices, phDevices));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
ur_result_t urContextRetain(

    /// [in] handle of the context to get a reference of.
    ur_context_handle_t hContext) {
  getContext()->logger.debug("==== urContextRetain");

  UR_CALL(getContext()->urDdiTable.Context.pfnRetain(hContext));

  auto ContextInfo = getTsanInterceptor()->getContextInfo(hContext);
  if (!ContextInfo) {
    getContext()->logger.error("Invalid context");
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }
  ContextInfo->RefCount++;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
ur_result_t urContextRelease(
    /// [in] handle of the context to release.
    ur_context_handle_t hContext) {
  getContext()->logger.debug("==== urContextRelease");

  UR_CALL(getContext()->urDdiTable.Context.pfnRelease(hContext));

  auto ContextInfo = getTsanInterceptor()->getContextInfo(hContext);
  if (!ContextInfo) {
    getContext()->logger.error("Invalid context");
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  if (--ContextInfo->RefCount == 0) {
    UR_CALL(getTsanInterceptor()->eraseContext(hContext));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
ur_result_t urProgramBuild(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the program object
    ur_program_handle_t hProgram,
    /// [in] string of build options
    const char *pOptions) {
  getContext()->logger.debug("==== urProgramBuild");

  UR_CALL(
      getContext()->urDdiTable.Program.pfnBuild(hContext, hProgram, pOptions));

  UR_CALL(getTsanInterceptor()->registerProgram(hProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
ur_result_t urProgramBuildExp(
    /// [in] Handle of the program to build.
    ur_program_handle_t hProgram,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  getContext()->logger.debug("==== urProgramBuildExp");

  UR_CALL(getContext()->urDdiTable.ProgramExp.pfnBuildExp(hProgram, numDevices,
                                                          phDevices, pOptions));
  UR_CALL(getTsanInterceptor()->registerProgram(hProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
ur_result_t urProgramLink(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] number of program handles in `phPrograms`.
    uint32_t count,
    /// [in][range(0, count)] pointer to array of program handles.
    const ur_program_handle_t *phPrograms,
    /// [in][optional] pointer to linker options null-terminated string.
    const char *pOptions,
    /// [out] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  getContext()->logger.debug("==== urProgramLink");

  UR_CALL(getContext()->urDdiTable.Program.pfnLink(hContext, count, phPrograms,
                                                   pOptions, phProgram));

  UR_CALL(getTsanInterceptor()->registerProgram(*phProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLinkExp
ur_result_t urProgramLinkExp(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in] number of program handles in `phPrograms`.
    uint32_t count,
    /// [in][range(0, count)] pointer to array of program handles.
    const ur_program_handle_t *phPrograms,
    /// [in][optional] pointer to linker options null-terminated string.
    const char *pOptions,
    /// [out] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  getContext()->logger.debug("==== urProgramLinkExp");

  UR_CALL(getContext()->urDdiTable.ProgramExp.pfnLinkExp(
      hContext, numDevices, phDevices, count, phPrograms, pOptions, phProgram));

  UR_CALL(getTsanInterceptor()->registerProgram(*phProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
__urdlllocal ur_result_t UR_APICALL urUSMDeviceAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM device memory object
    void **ppMem) {
  getContext()->logger.debug("==== urUSMDeviceAlloc");

  return getTsanInterceptor()->allocateMemory(
      hContext, hDevice, pUSMDesc, pool, size, AllocType::DEVICE_USM, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMHostAlloc
__urdlllocal ur_result_t UR_APICALL urUSMHostAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM host memory object
    void **ppMem) {
  getContext()->logger.debug("==== urUSMHostAlloc");

  return getTsanInterceptor()->allocateMemory(hContext, nullptr, pUSMDesc, pool,
                                              size, AllocType::HOST_USM, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMSharedAlloc
__urdlllocal ur_result_t UR_APICALL urUSMSharedAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] Pointer to USM memory allocation descriptor.
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM shared memory object
    void **ppMem) {
  getContext()->logger.debug("==== urUSMSharedAlloc");

  return getTsanInterceptor()->allocateMemory(
      hContext, hDevice, pUSMDesc, pool, size, AllocType::SHARED_USM, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
ur_result_t urEnqueueKernelLaunch(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *pGlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that will
    /// execute the kernel function. If nullptr, the runtime implementation will
    /// choose the work-group size.
    const size_t *pLocalWorkSize,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *phEvent) {
  getContext()->logger.debug("==== urEnqueueKernelLaunch");

  LaunchInfo LaunchInfo(GetContext(hQueue), GetDevice(hQueue));

  UR_CALL(getTsanInterceptor()->preLaunchKernel(hKernel, hQueue, LaunchInfo));

  UR_CALL(getContext()->urDdiTable.Enqueue.pfnKernelLaunch(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent));

  UR_CALL(getTsanInterceptor()->postLaunchKernel(hKernel, hQueue, LaunchInfo));

  return UR_RESULT_SUCCESS;
}

ur_result_t urCheckVersion(ur_api_version_t version) {
  if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
          UR_MINOR_VERSION(version)) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetContextProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_context_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
          UR_MINOR_VERSION(version)) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreate = ur_sanitizer_layer::tsan::urContextCreate;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_sanitizer_layer::tsan::urContextCreateWithNativeHandle;
  pDdiTable->pfnRetain = ur_sanitizer_layer::tsan::urContextRetain;
  pDdiTable->pfnRelease = ur_sanitizer_layer::tsan::urContextRelease;

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_program_dditable_t *pDdiTable) {
  pDdiTable->pfnBuild = ur_sanitizer_layer::tsan::urProgramBuild;
  pDdiTable->pfnLink = ur_sanitizer_layer::tsan::urProgramLink;

  return UR_RESULT_SUCCESS;
}

/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramExpProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_program_exp_dditable_t *pDdiTable) {
  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnBuildExp = ur_sanitizer_layer::tsan::urProgramBuildExp;
  pDdiTable->pfnLinkExp = ur_sanitizer_layer::tsan::urProgramLinkExp;

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetUSMProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
          UR_MINOR_VERSION(version)) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnDeviceAlloc = ur_sanitizer_layer::tsan::urUSMDeviceAlloc;
  pDdiTable->pfnHostAlloc = ur_sanitizer_layer::tsan::urUSMHostAlloc;
  pDdiTable->pfnSharedAlloc = ur_sanitizer_layer::tsan::urUSMSharedAlloc;

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
          UR_MINOR_VERSION(version)) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnKernelLaunch = ur_sanitizer_layer::tsan::urEnqueueKernelLaunch;

  return result;
}

} // namespace tsan

ur_result_t initTsanDDITable(ur_dditable_t *dditable) {
  ur_result_t result = UR_RESULT_SUCCESS;

  getContext()->logger.always("==== DeviceSanitizer: TSAN");

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urCheckVersion(UR_API_VERSION_CURRENT);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetContextProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->Context);
  }

  if (UR_RESULT_SUCCESS == result) {
    result =
        ur_sanitizer_layer::tsan::urGetProgramProcAddrTable(&dditable->Program);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetProgramExpProcAddrTable(
        &dditable->ProgramExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetUSMProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->USM);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetEnqueueProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->Enqueue);
  }

  if (result != UR_RESULT_SUCCESS) {
    getContext()->logger.error("Initialize TSAN DDI table failed: {}", result);
  }

  return result;
}

} // namespace ur_sanitizer_layer
