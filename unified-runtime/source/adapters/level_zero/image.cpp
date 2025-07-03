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
#include "image_common.hpp"
#include "logger/ur_logger.hpp"
#include "memory.hpp"
#include "sampler.hpp"
#include "ur_api.h"
#include "ur_interface_loader.hpp"

#include "loader/ze_loader.h"

namespace ur::level_zero {

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
  if (is3ChannelOrder(
          ur_cast<ur_image_channel_order_t>(pSrcImageFormat->channelOrder)) ||
      is3ChannelOrder(
          ur_cast<ur_image_channel_order_t>(pDstImageFormat->channelOrder))) {
    UseCopyEngine = false;
  }

  ur_ze_event_list_t TmpWaitList;
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

  auto res = bindlessImagesHandleCopyFlags(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, ZeCommandList, ZeEvent,
      WaitList.Length, WaitList.ZeEventList);

  if (res == UR_RESULT_SUCCESS)
    UR_CALL(hQueue->executeCommandList(CommandList, Blocking, OkToBatch));

  return res;
}

ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    UR_LOG_LEGACY(ERR,
                  logger::LegacyMessage("[UR][L0] {} function not supported!"),
                  "{} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  ur_ze_event_list_t TmpWaitList;
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

  if (UrPlatform->ZeExternalSemaphoreExt.LoaderExtension) {
    ze_external_semaphore_wait_params_ext_t WaitParams = {
        ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXT, nullptr, 0};
    WaitParams.value = hasValue ? waitValue : 0;
    ze_external_semaphore_ext_handle_t hExtSemaphore =
        reinterpret_cast<ze_external_semaphore_ext_handle_t>(hSemaphore);
    ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                   .zexCommandListAppendWaitExternalSemaphoresExp,
               (ZeCommandList, 1, &hExtSemaphore, &WaitParams, ZeEvent,
                WaitList.Length, WaitList.ZeEventList));
  } else {
    ze_command_list_handle_t translatedCommandList;
    ZE2UR_CALL(zelLoaderTranslateHandle,
               (ZEL_HANDLE_COMMAND_LIST, ZeCommandList,
                (void **)&translatedCommandList));
    ze_event_handle_t translatedEvent = ZeEvent;
    if (ZeEvent) {
      ZE2UR_CALL(zelLoaderTranslateHandle,
                 (ZEL_HANDLE_EVENT, ZeEvent, (void **)&translatedEvent));
    }
    std::vector<ze_event_handle_t> EventHandles(WaitList.Length + 1, nullptr);
    if (WaitList.Length > 0) {
      for (size_t i = 0; i < WaitList.Length; i++) {
        ze_event_handle_t ZeEvent = WaitList.ZeEventList[i];
        ZE2UR_CALL(zelLoaderTranslateHandle,
                   (ZEL_HANDLE_EVENT, ZeEvent, (void **)&EventHandles[i + 1]));
      }
    }
    ze_intel_external_semaphore_wait_params_exp_t WaitParams = {
        ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXP, nullptr, 0};
    WaitParams.value = hasValue ? waitValue : 0;
    const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
        reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);
    ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                   .zexExpCommandListAppendWaitExternalSemaphoresExp,
               (translatedCommandList, 1, &hExtSemaphore, &WaitParams,
                translatedEvent, WaitList.Length, EventHandles.data()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    UR_LOG_LEGACY(ERR,
                  logger::LegacyMessage("[UR][L0] {} function not supported!"),
                  "{} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  ur_ze_event_list_t TmpWaitList;
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

  if (UrPlatform->ZeExternalSemaphoreExt.LoaderExtension) {
    ze_external_semaphore_signal_params_ext_t SignalParams = {
        ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXT, nullptr, 0};
    SignalParams.value = hasValue ? signalValue : 0;
    ze_external_semaphore_ext_handle_t hExtSemaphore =
        reinterpret_cast<ze_external_semaphore_ext_handle_t>(hSemaphore);

    ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                   .zexCommandListAppendSignalExternalSemaphoresExp,
               (ZeCommandList, 1, &hExtSemaphore, &SignalParams, ZeEvent,
                WaitList.Length, WaitList.ZeEventList));
  } else {
    ze_intel_external_semaphore_signal_params_exp_t SignalParams = {
        ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXP, nullptr,
        0};
    SignalParams.value = hasValue ? signalValue : 0;
    const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
        reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);

    ze_command_list_handle_t translatedCommandList;
    ZE2UR_CALL(zelLoaderTranslateHandle,
               (ZEL_HANDLE_COMMAND_LIST, ZeCommandList,
                (void **)&translatedCommandList));
    ze_event_handle_t translatedEvent = ZeEvent;
    if (ZeEvent) {
      ZE2UR_CALL(zelLoaderTranslateHandle,
                 (ZEL_HANDLE_EVENT, ZeEvent, (void **)&translatedEvent));
    }
    std::vector<ze_event_handle_t> EventHandles(WaitList.Length + 1, nullptr);
    if (WaitList.Length > 0) {
      for (size_t i = 0; i < WaitList.Length; i++) {
        ze_event_handle_t ZeEvent = WaitList.ZeEventList[i];
        ZE2UR_CALL(zelLoaderTranslateHandle,
                   (ZEL_HANDLE_EVENT, ZeEvent, (void **)&EventHandles[i + 1]));
      }
    }
    ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                   .zexExpCommandListAppendSignalExternalSemaphoresExp,
               (translatedCommandList, 1, &hExtSemaphore, &SignalParams,
                translatedEvent, WaitList.Length, EventHandles.data()));
  }

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
