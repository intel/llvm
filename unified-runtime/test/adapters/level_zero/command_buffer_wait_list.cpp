// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v1 ZE_ENABLE_TRACING_LAYER=1 UR_L0_CMD_BUFFER_USE_IMMEDIATE_APPEND_PATH=0 ./command_buffer_wait_list-test
// REQUIRES: v1

#include "ze_tracer_common.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

// Checks that enqueueing a command-buffer with a wait event does not make the
// dependency barrier wait on the events of all previous submissions too. The
// growing barrier made each submission slower and eventually overflowed the
// driver's command stream.

namespace {

bool TraceBarriers = false;
uint32_t MaxBarrierWaitEvents = 0;

void ZE_APICALL onAppendBarrier(ze_command_list_append_barrier_params_t *params,
                                ze_result_t, void *, void **) {
  if (TraceBarriers)
    MaxBarrierWaitEvents =
        std::max(MaxBarrierWaitEvents, *params->pnumWaitEvents);
}

struct urCommandBufferWaitListTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    ur_bool_t CommandBufferSupport = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP,
        sizeof(CommandBufferSupport), &CommandBufferSupport, nullptr));
    if (!CommandBufferSupport) {
      GTEST_SKIP() << "EXP command-buffer feature is not supported.";
    }

    ASSERT_SUCCESS(
        urQueueCreate(context, device, &queue_properties, &OtherQueue));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    AllocationSize, &DevicePtr));

    ur_exp_command_buffer_desc_t Desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                      nullptr, false, false, false};
    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &Desc, &CmdBuf));
    uint32_t Pattern = 42;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        CmdBuf, DevicePtr, &Pattern, sizeof(Pattern), AllocationSize, 0,
        nullptr, 0, nullptr, nullptr, nullptr, nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(CmdBuf));

    zel_core_callbacks_t Callbacks{};
    Callbacks.CommandList.pfnAppendBarrierCb = onAppendBarrier;
    Tracer = enableTracing(std::move(Callbacks), {});
    ASSERT_NE(Tracer, nullptr);
  }

  void TearDown() override {
    if (Tracer) {
      EXPECT_EQ(zelTracerSetEnabled(Tracer.get(), false), ZE_RESULT_SUCCESS);
      Tracer.reset();
    }
    if (CmdBuf) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(CmdBuf));
    }
    if (DevicePtr) {
      EXPECT_SUCCESS(urUSMFree(context, DevicePtr));
    }
    if (OtherQueue) {
      EXPECT_SUCCESS(urQueueRelease(OtherQueue));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
  }

  static constexpr size_t AllocationSize = 1024;
  std::shared_ptr<_zel_tracer_handle_t> Tracer;
  ur_queue_handle_t OtherQueue = nullptr;
  ur_exp_command_buffer_handle_t CmdBuf = nullptr;
  void *DevicePtr = nullptr;
};

} // namespace

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferWaitListTest);

TEST_P(urCommandBufferWaitListTest, DependencyBarrierDoesNotGrow) {
  constexpr size_t NumSubmissions = 16;
  std::vector<uint32_t> WaitEvents;

  for (size_t I = 0; I < NumSubmissions; ++I) {
    ur_event_handle_t Dependency = nullptr;
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(OtherQueue, 0, nullptr, &Dependency));

    MaxBarrierWaitEvents = 0;
    TraceBarriers = true;
    ASSERT_SUCCESS(
        urEnqueueCommandBufferExp(queue, CmdBuf, 1, &Dependency, nullptr));
    TraceBarriers = false;
    ASSERT_SUCCESS(urEventRelease(Dependency));
    WaitEvents.push_back(MaxBarrierWaitEvents);
  }

  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_SUCCESS(urQueueFinish(OtherQueue));

  // From the second submission on, the in-order queue's last command is waited
  // for too, so the count is only expected to be stable after the first one.
  ASSERT_GE(WaitEvents[1], 1u);
  for (size_t I = 2; I < NumSubmissions; ++I) {
    EXPECT_EQ(WaitEvents[I], WaitEvents[1]) << "at submission " << I;
  }
}
