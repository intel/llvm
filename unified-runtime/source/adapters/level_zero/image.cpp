//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023-2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "helpers/image_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "memory.hpp"
#include "sampler.hpp"
#include "ur_interface_loader.hpp"


namespace {

bool Is3ChannelOrder(ur_image_channel_order_t ChannelOrder) {
  switch (ChannelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_RGB:
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    return true;
  default:
    return false;
  }
}

} // namespace

namespace ur::level_zero {

ur_result_t
urBindlessImagesImageFreeExp(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  UR_CALL(ur::level_zero::urMemRelease(
      reinterpret_cast<ur_mem_handle_t>(hImageMem)));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, const void *pSrc, void *pDst,
    const ur_image_desc_t *pSrcImageDesc, const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::scoped_lock<ur_shared_mutex> Lock(hQueue->Mutex);

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst && pSrc && pSrcImageFormat && pSrcImageDesc && pDstImageDesc &&
                pCopyRegion,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrcImageDesc->type == pDstImageDesc->type,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(!(UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags),
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(!(pSrcImageDesc && UR_MEM_TYPE_IMAGE1D_ARRAY < pSrcImageDesc->type),
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  bool UseCopyEngine = hQueue->useCopyEngine(/*PreferCopyEngine*/ true);
  // Due to the limitation of the copy engine, disable usage of Copy Engine
  // Given 3 channel image
  if (Is3ChannelOrder(
          ur_cast<ur_image_channel_order_t>(pSrcImageFormat->channelOrder)) ||
      Is3ChannelOrder(
          ur_cast<ur_image_channel_order_t>(pDstImageFormat->channelOrder))) {
    UseCopyEngine = false;
  }

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  bool Blocking = false;
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event, UR_COMMAND_MEM_IMAGE_COPY,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;
  
  auto res = handleImageCopyFlags(pSrc, pDst, pSrcImageDesc, pDstImageDesc, 
                        pSrcImageFormat, pDstImageFormat, pCopyRegion,
                        imageCopyFlags, ZeCommandList, ZeEvent, WaitList.Length,
                        WaitList.ZeEventList);

  if (res == UR_RESULT_SUCCESS)
    UR_CALL(hQueue->executeCommandList(CommandList, Blocking, OkToBatch));

  return res;
}

ur_result_t urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ur_exp_external_mem_handle_t *phExternalMem) {

  UR_ASSERT(hContext && hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pExternalMemDesc && phExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  struct ur_ze_external_memory_data *externalMemoryData =
      new struct ur_ze_external_memory_data;

  void *pNext = const_cast<void *>(pExternalMemDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
      ze_external_memory_import_fd_t *importFd =
          new ze_external_memory_import_fd_t;
      importFd->stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
      importFd->pNext = nullptr;
      auto FileDescriptor =
          static_cast<const ur_exp_file_descriptor_t *>(pNext);
      importFd->fd = FileDescriptor->fd;
      importFd->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
      externalMemoryData->importExtensionDesc = importFd;
      externalMemoryData->type = UR_ZE_EXTERNAL_OPAQUE_FD;
    } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
      ze_external_memory_import_win32_handle_t *importWin32 =
          new ze_external_memory_import_win32_handle_t;
      importWin32->stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
      importWin32->pNext = nullptr;
      auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);

      switch (memHandleType) {
      case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT:
        importWin32->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
        break;
      case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE:
        importWin32->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE;
        break;
      case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD:
      default:
        delete importWin32;
        delete externalMemoryData;
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      importWin32->handle = Win32Handle->handle;
      externalMemoryData->importExtensionDesc = importWin32;
      externalMemoryData->type = UR_ZE_EXTERNAL_WIN32;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }
  externalMemoryData->size = size;

  *phExternalMem =
      reinterpret_cast<ur_exp_external_mem_handle_t>(externalMemoryData);
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem) {

  UR_ASSERT(hContext && hDevice && hExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pImageFormat && pImageDesc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  struct ur_ze_external_memory_data *externalMemoryData =
      reinterpret_cast<ur_ze_external_memory_data *>(hExternalMem);

  ze_image_bindless_exp_desc_t ZeImageBindlessDesc = {};
  ZeImageBindlessDesc.stype = ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC;

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pImageFormat, pImageDesc, ZeImageDesc));

  ZeImageBindlessDesc.pNext = externalMemoryData->importExtensionDesc;
  ZeImageBindlessDesc.flags = ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS;
  ZeImageDesc.pNext = &ZeImageBindlessDesc;

  ze_image_handle_t ZeImage;
  ZE2UR_CALL(zeImageCreate,
             (hContext->ZeContext, hDevice->ZeDevice, &ZeImageDesc, &ZeImage));
  ZE2UR_CALL(zeContextMakeImageResident,
             (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  UR_CALL(
      createUrMemFromZeImage(hContext, ZeImage, true, ZeImageDesc, phImageMem));
  externalMemoryData->urMemoryHandle =
      reinterpret_cast<ur_mem_handle_t>(*phImageMem);
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **phRetMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = size;
  std::ignore = offset;
  std::ignore = hExternalMem;
  std::ignore = phRetMem;
  logger::error("[UR][L0] {} function not implemented!",
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_handle_t hExternalMem) {

  UR_ASSERT(hContext && hDevice && hExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  struct ur_ze_external_memory_data *externalMemoryData =
      reinterpret_cast<ur_ze_external_memory_data *>(hExternalMem);

  UR_CALL(ur::level_zero::urMemRelease(externalMemoryData->urMemoryHandle));

  switch (externalMemoryData->type) {
  case UR_ZE_EXTERNAL_OPAQUE_FD:
    delete (reinterpret_cast<ze_external_memory_import_fd_t *>(
        externalMemoryData->importExtensionDesc));
    break;
  case UR_ZE_EXTERNAL_WIN32:
    delete (reinterpret_cast<ze_external_memory_import_win32_handle_t *>(
        externalMemoryData->importExtensionDesc));
    break;
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  delete (externalMemoryData);

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ur_exp_external_semaphore_handle_t *phExternalSemaphoreHandle) {

  auto UrPlatform = hContext->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  ze_intel_external_semaphore_exp_desc_t SemDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_EXP_DESC, nullptr,
      ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD};
  ze_intel_external_semaphore_exp_handle_t ExtSemaphoreHandle;
  ze_intel_external_semaphore_desc_fd_exp_desc_t FDExpDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_FD_EXP_DESC, nullptr, 0};
  _ze_intel_external_semaphore_win32_exp_desc_t Win32ExpDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WIN32_EXP_DESC, nullptr,
      nullptr, nullptr};
  void *pNext = const_cast<void *>(pExternalSemaphoreDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
      auto FileDescriptor =
          static_cast<const ur_exp_file_descriptor_t *>(pNext);
      FDExpDesc.fd = FileDescriptor->fd;
      SemDesc.pNext = &FDExpDesc;
      switch (semHandleType) {
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD;
        break;
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_TIMELINE_FD:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_TIMELINE_SEMAPHORE_FD;
        break;
      default:
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
      SemDesc.pNext = &Win32ExpDesc;
      auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);
      switch (semHandleType) {
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_WIN32;
        break;
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_D3D12_FENCE;
        break;
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_TIMELINE_WIN32_NT:
        SemDesc.flags =
            ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_TIMELINE_SEMAPHORE_WIN32;
        break;
      default:
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      Win32ExpDesc.handle = Win32Handle->handle;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }

  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt.zexImportExternalSemaphoreExp,
             (hDevice->ZeDevice, &SemDesc, &ExtSemaphoreHandle));
  *phExternalSemaphoreHandle =
      (ur_exp_external_semaphore_handle_t)ExtSemaphoreHandle;

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  std::ignore = hDevice;
  auto UrPlatform = hContext->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  ZE2UR_CALL(
      UrPlatform->ZeExternalSemaphoreExt.zexDeviceReleaseExternalSemaphoreExp,
      ((ze_intel_external_semaphore_exp_handle_t)hExternalSemaphore));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event,
                                       UR_COMMAND_EXTERNAL_SEMAPHORE_WAIT_EXP,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ze_intel_external_semaphore_wait_params_exp_t WaitParams = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXP, nullptr, 0};
  WaitParams.value = hasValue ? waitValue : 0;
  const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
      reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);
  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendWaitExternalSemaphoresExp,
             (ZeCommandList, 1, &hExtSemaphore, &WaitParams, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = hasValue;
  std::ignore = signalValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event,
                                       UR_COMMAND_EXTERNAL_SEMAPHORE_SIGNAL_EXP,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ze_intel_external_semaphore_signal_params_exp_t SignalParams = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXP, nullptr, 0};
  SignalParams.value = hasValue ? signalValue : 0;
  const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
      reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);

  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendSignalExternalSemaphoresExp,
             (ZeCommandList, 1, &hExtSemaphore, &SignalParams, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
