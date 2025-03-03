// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <ze_api.h>

#include "../../../conformance/enqueue/helpers.h"
#include "../ze_helpers.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"

struct urEnqueueKernelLaunchTest : uur::urKernelExecutionTest {
  void SetUp() override {
    // Initialize Level Zero driver is required if this test is linked
    // statically with Level Zero loader, the driver will not be init otherwise.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  uint32_t val = 42;
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchTest);

TEST_P(urEnqueueKernelLaunchTest, DeferredKernelRelease) {
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);

  auto zeEvent = createZeEvent(context, device);

  ur_event_handle_t event;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeEvent.get()), context, nullptr,
      &event));

  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 1, &event, nullptr));
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, nullptr));
  ASSERT_SUCCESS(urKernelRelease(kernel));

  // Kernel should still be alive since kernel launch is pending
  ur_context_handle_t contextFromKernel;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                 sizeof(ur_context_handle_t),
                                 &contextFromKernel, nullptr));

  ASSERT_EQ(context, contextFromKernel);

  ze_event_handle_t ze_event = nullptr;
  ASSERT_SUCCESS(urEventGetNativeHandle(
      event, reinterpret_cast<ur_native_handle_t *>(&ze_event)));
  ASSERT_EQ(zeEventHostSignal(ze_event), ZE_RESULT_SUCCESS);

  ASSERT_SUCCESS(urQueueFinish(queue));

  kernel = nullptr;

  ASSERT_SUCCESS(urEventRelease(event));
}

struct urMultiQueueLaunchKernelDeferFreeTest
    : uur::urMultiQueueMultiDeviceTest<2> {
  std::string KernelName;

  static constexpr char ProgramName[] = "foo";
  static constexpr size_t ArraySize = 100;
  static constexpr uint32_t InitialValue = 1;

  ur_program_handle_t program = nullptr;
  ur_kernel_handle_t kernel = nullptr;

  void SetUp() override {
    if (devices.size() < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices";
    }

    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiQueueMultiDeviceTest<2>::SetUp());

    KernelName =
        uur::KernelsEnvironment::instance->GetEntryPointNames(ProgramName)[0];

    std::shared_ptr<std::vector<char>> il_binary;
    std::vector<ur_program_metadata_t> metadatas{};

    uur::KernelsEnvironment::instance->LoadSource(ProgramName, platform,
                                                  il_binary);

    const ur_program_properties_t properties = {
        UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
        static_cast<uint32_t>(metadatas.size()),
        metadatas.empty() ? nullptr : metadatas.data()};

    ASSERT_SUCCESS(urProgramCreateWithIL(
        context, il_binary->data(), il_binary->size(), &properties, &program));

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urProgramBuild(context, program, nullptr));
    ASSERT_SUCCESS(urKernelCreate(program, KernelName.data(), &kernel));
  }

  void TearDown() override {
    // kernel will be release in the actual test

    urProgramRelease(program);
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiQueueMultiDeviceTest<2>::TearDown());
  }
};

UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMultiQueueLaunchKernelDeferFreeTest);

TEST_P(urMultiQueueLaunchKernelDeferFreeTest, Success) {
  auto zeEvent1 = createZeEvent(context, devices[0]);
  auto zeEvent2 = createZeEvent(context, devices[1]);

  ur_event_handle_t event1;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeEvent1.get()), context, nullptr,
      &event1));
  ur_event_handle_t event2;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeEvent2.get()), context, nullptr,
      &event2));

  size_t global_offset = 0;
  size_t global_size = 1;

  ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], 1, &event1, nullptr));
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernel, 1, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       nullptr));

  ASSERT_SUCCESS(urEnqueueEventsWait(queues[1], 1, &event2, nullptr));
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernel, 1, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       nullptr));

  ASSERT_SUCCESS(urKernelRelease(kernel));

  // Kernel should still be alive since both kernels are pending
  ur_context_handle_t contextFromKernel;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                 sizeof(ur_context_handle_t),
                                 &contextFromKernel, nullptr));
  ASSERT_EQ(context, contextFromKernel);

  ASSERT_EQ(zeEventHostSignal(zeEvent2.get()), ZE_RESULT_SUCCESS);
  ASSERT_SUCCESS(urQueueFinish(queues[1]));

  // Kernel should still be alive since kernel launch is pending
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                 sizeof(ur_context_handle_t),
                                 &contextFromKernel, nullptr));
  ASSERT_EQ(context, contextFromKernel);

  ASSERT_EQ(zeEventHostSignal(zeEvent1.get()), ZE_RESULT_SUCCESS);
  ASSERT_SUCCESS(urQueueFinish(queues[0]));

  ASSERT_SUCCESS(urEventRelease(event1));
  ASSERT_SUCCESS(urEventRelease(event2));
}
