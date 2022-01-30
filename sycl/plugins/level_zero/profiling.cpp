//===-------------- profiling.cpp - L0 Kernel Profiling --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <level_zero/layers/zel_tracing_api.h>

void enterEventDestroy(ze_event_destroy_params_t *params, ze_result_t result,
                       void *global_data, void **instance_data) {
  if (*(params->phEvent) != nullptr) {
  }
}

void OnEnterEventHostReset(ze_event_host_reset_params_t *params,
                           ze_result_t result, void *global_data,
                           void **instance_data) {
  if (*(params->phEvent) != nullptr) {
  }
}

static void OnEnterEventPoolCreate(ze_event_pool_create_params_t *params,
                                   ze_result_t result, void *global_data,
                                   void **instance_data) {
  const ze_event_pool_desc_t *desc = *(params->pdesc);
  if (desc == nullptr) {
    return;
  }
  if (desc->flags & ZE_EVENT_POOL_FLAG_IPC) {
    return;
  }

  ze_event_pool_desc_t *profiling_desc = new ze_event_pool_desc_t;
  PTI_ASSERT(profiling_desc != nullptr);
  profiling_desc->stype = desc->stype;
  // PTI_ASSERT(profiling_desc->stype == ZE_STRUCTURE_TYPE_EVENT_POOL_DESC);
  profiling_desc->pNext = desc->pNext;
  profiling_desc->flags = desc->flags;
  profiling_desc->flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  profiling_desc->flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  profiling_desc->count = desc->count;

  *(params->pdesc) = profiling_desc;
  *instance_data = profiling_desc;
}

void enableL0Profiling() {
#ifdef 1
  zel_tracer_desc_t TracerDesc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, nullptr,
                                  nullptr};

  zel_tracer_handle_t Tracer = nullptr;

  ze_result_t Status = zelTracerCreate(&TracerDesc, &Tracer);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "[WARNING] Failed to create Level Zero tracer\n";
    return;
  }

  zet_core_callbacks_t PrologueCallbacks{};
  zet_core_callbacks_t EpilogueCallbacks{};

  PrologueCallbacks.Event.pfnDestroyCb = enterEventDestroy;
  PrologueCallbacks.Event.pfnHostResetCb = enterEventHostReset;

  PrologueCallbacks.EventPool.pfnCreateCb = enterEventPoolCreateCb;
  EpilogueCallbacks.EventPool.pfnCreateCb = exitEventPoolCreateCb;

  PrologueCallbacks.CommandList.pfnAppendLaunchKernelCb =
      enterCommandListAppendLaunchKernel;
  EpilogueCallbacks.CommandList.pfnAppendLaunchKernelCb =
      exitCommandListAppendLaunchKernel;

  EpilogueCallbacks.CommandList.pfnCreateCb = exitCommandListCreate;
  EpilogueCallbacks.CommandList.pfnCreateImmediateCb =
      exitCommandListCreateImmediate;
  EpilogueCallbacks.CommandList.pfnDestroyCb = exitCommandListDestroy;
  EpilogueCallbacks.CommandList.pfnResetCb = exitCommandListReset;

  EpilogueCallbacks.CommandQueue.pfnExecuteCommandListsCb =
      exitCommandQueueExecuteCommandLists;
  EpilogueCallbacks.CommandQueue.pfnSynchronizeCb = exitCommandQueueSynchronize;
  EpilogueCallbacks.CommandQueue.pfnDestroyCb = exitCommandQueueDestroy;

  Status = zelTracerSetPrologues(Tracer, &PrologueCallbacks);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to initialize prologue callbacks\n";
    std::terminate();
  }
  Status = zelTracerSetEpilogues(Tracer, &EpilogueCallbacks);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to initialize prologue callbacks\n";
    std::terminate();
  }
  Status = zelTracerSetEnabled(Tracer, true);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to enable L0 tracer\n";
    std::terminate();
  }
#endif
}
