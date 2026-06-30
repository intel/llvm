// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>

#include <level_zero/ze_api.h>

namespace uur {
namespace event {

// Signals an IPC-shareable producer event using the native Level Zero API:
// append a barrier that signals the event on a fresh immediate command list,
// then host-synchronize so the event is already in the signaled state when
// the consumer side waits on the imported handle.
inline void signalEventViaLevelZero(ur_event_handle_t event,
                                    ur_context_handle_t context,
                                    ur_device_handle_t device) {
  ze_event_handle_t zeEvent = nullptr;
  ze_context_handle_t zeContext = nullptr;
  ze_device_handle_t zeDevice = nullptr;
  ASSERT_SUCCESS(urEventGetNativeHandle(
      event, reinterpret_cast<ur_native_handle_t *>(&zeEvent)));
  ASSERT_SUCCESS(urContextGetNativeHandle(
      context, reinterpret_cast<ur_native_handle_t *>(&zeContext)));
  ASSERT_SUCCESS(urDeviceGetNativeHandle(
      device, reinterpret_cast<ur_native_handle_t *>(&zeDevice)));
  ASSERT_NE(zeEvent, nullptr);

  uint32_t numGroups = 0;
  ASSERT_EQ(
      zeDeviceGetCommandQueueGroupProperties(zeDevice, &numGroups, nullptr),
      ZE_RESULT_SUCCESS);
  ze_command_queue_group_properties_t propsInit{};
  propsInit.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  std::vector<ze_command_queue_group_properties_t> props(numGroups, propsInit);
  ASSERT_EQ(zeDeviceGetCommandQueueGroupProperties(zeDevice, &numGroups,
                                                   props.data()),
            ZE_RESULT_SUCCESS);

  uint32_t computeOrdinal = 0;
  for (uint32_t i = 0; i < numGroups; ++i) {
    if (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      computeOrdinal = i;
      break;
    }
  }

  ze_command_queue_desc_t qDesc{};
  qDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qDesc.ordinal = computeOrdinal;
  qDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;

  ze_command_list_handle_t cmdList = nullptr;
  ASSERT_EQ(zeCommandListCreateImmediate(zeContext, zeDevice, &qDesc, &cmdList),
            ZE_RESULT_SUCCESS);
  ASSERT_EQ(zeCommandListAppendBarrier(cmdList, zeEvent, 0, nullptr),
            ZE_RESULT_SUCCESS);
  ASSERT_EQ(zeEventHostSynchronize(zeEvent, UINT64_MAX), ZE_RESULT_SUCCESS);
  zeCommandListDestroy(cmdList);
}

/// Fixture for the inter-process event sharing experimental APIs
/// (urIPC{Get,Put,Open}EventHandleExp).
///
/// SetUp:
///   - skips on backends other than Level Zero (the producer event is signaled
///     via the native Level Zero API),
///   - skips on devices that don't advertise
///     UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP,
///   - creates an IPC-shareable event via urEventCreateExp with
///     UR_EXP_EVENT_FLAG_IPC_EXP set, and signals it.
/// Derives from urQueueTest (no profiling) since IPC and per-event profiling
/// are mutually exclusive.
struct urIPCEventTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

    ur_backend_t backend = UR_BACKEND_UNKNOWN;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    if (backend != UR_BACKEND_LEVEL_ZERO) {
      GTEST_SKIP() << "IPC event tests are only supported on Level Zero.";
    }

    ur_bool_t ipcEventSupport = false;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP,
                                   sizeof(ipcEventSupport), &ipcEventSupport,
                                   nullptr));
    if (!ipcEventSupport) {
      GTEST_SKIP() << "IPC event feature is not supported on this device.";
    }

    // Initializing the Level Zero driver is required when the test is linked
    // statically with the Level Zero loader, otherwise the driver will not be
    // initialized.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr,
                             UR_EXP_EVENT_FLAG_IPC_EXP};
    ASSERT_SUCCESS(urEventCreateExp(context, device, &desc, &event));
    ASSERT_NE(event, nullptr);

    UUR_RETURN_ON_FATAL_FAILURE(
        signalEventViaLevelZero(event, context, device));
  }

  void TearDown() override {
    if (event) {
      EXPECT_SUCCESS(urEventRelease(event));
      event = nullptr;
    }
    uur::urQueueTest::TearDown();
  }

  ur_event_handle_t event = nullptr;
};

} // namespace event
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED
