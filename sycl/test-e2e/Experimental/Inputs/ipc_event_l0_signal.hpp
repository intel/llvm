// Test-only helper that signals a SYCL IPC event through Level Zero interop by
// appending a barrier on an immediate command list.

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ipc_event_test {

// Create an in-order immediate command list on the device's compute engine.
inline ze_command_list_handle_t
createImmediateComputeCmdList(const sycl::context &Ctx,
                              const sycl::device &Dev) {
  // Tests link the L0 loader statically; the driver must be initialised
  // explicitly before any L0 entry points are called.
  zeInit(ZE_INIT_FLAG_GPU_ONLY);

  ze_context_handle_t ZeCtx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Ctx);
  ze_device_handle_t ZeDev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Dev);

  uint32_t NumGroups = 0;
  if (zeDeviceGetCommandQueueGroupProperties(ZeDev, &NumGroups, nullptr) !=
      ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ipc_event_test: zeDeviceGetCommandQueueGroup* "
                         "failed\n");
    std::exit(2);
  }

  std::vector<ze_command_queue_group_properties_t> Props(
      NumGroups, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
  zeDeviceGetCommandQueueGroupProperties(ZeDev, &NumGroups, Props.data());

  uint32_t ComputeOrdinal = 0;
  for (uint32_t I = 0; I < NumGroups; ++I) {
    if (Props[I].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ComputeOrdinal = I;
      break;
    }
  }

  ze_command_queue_desc_t QDesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  QDesc.ordinal = ComputeOrdinal;
  QDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  QDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;

  ze_command_list_handle_t CmdList = nullptr;
  if (zeCommandListCreateImmediate(ZeCtx, ZeDev, &QDesc, &CmdList) !=
      ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ipc_event_test: zeCommandListCreateImmediate "
                         "failed\n");
    std::exit(2);
  }
  return CmdList;
}

// Signal SyclEvent by appending a barrier and then host-synchronize before
// returning (the signal has completed once this returns).
inline void signalEventViaLevelZero(sycl::event &SyclEvent,
                                    const sycl::context &Ctx,
                                    const sycl::device &Dev) {
  ze_event_handle_t ZeEvt =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(SyclEvent);
  ze_command_list_handle_t CmdList = createImmediateComputeCmdList(Ctx, Dev);

  if (zeCommandListAppendBarrier(CmdList, ZeEvt, 0, nullptr) !=
      ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ipc_event_test: zeCommandListAppendBarrier "
                         "failed\n");
    std::exit(2);
  }
  if (zeEventHostSynchronize(ZeEvt, UINT64_MAX) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ipc_event_test: zeEventHostSynchronize failed\n");
    std::exit(2);
  }
  zeCommandListDestroy(CmdList);
}

// Submit a barrier that signals SyclEvent only after DepEvent completes, then
// return WITHOUT host-synchronizing: SyclEvent may still be unsignaled when
// this returns. The signal is thus ordered strictly after DepEvent, so the
// only synchronization that guarantees DepEvent's effects to a remote waiter
// is a wait on SyclEvent itself.
//
// The returned command list must be kept alive until the signal has completed
// (e.g. after a later wait on SyclEvent) and then destroyed by the caller via
// zeCommandListDestroy.
inline ze_command_list_handle_t
submitSignalDependentOnEvent(sycl::event &SyclEvent, sycl::event &DepEvent,
                             const sycl::context &Ctx,
                             const sycl::device &Dev) {
  ze_event_handle_t ZeEvt =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(SyclEvent);
  ze_event_handle_t ZeDep =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(DepEvent);
  ze_command_list_handle_t CmdList = createImmediateComputeCmdList(Ctx, Dev);

  if (zeCommandListAppendBarrier(CmdList, ZeEvt, 1, &ZeDep) !=
      ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ipc_event_test: zeCommandListAppendBarrier (with "
                         "dependency) failed\n");
    std::exit(2);
  }
  return CmdList;
}

} // namespace ipc_event_test
