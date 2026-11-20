// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./deferred_kernel-test
// RUN: %with-v2 env UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1 ./deferred_kernel-test --gtest_filter=*SynchronizeAtKernelLimit*
// REQUIRES: v2
// UNSUPPORTED: system-windows

#include <chrono>
#include <future>
#include <vector>
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
UUR_INSTANTIATE_DEVICE_TEST_SUITE_MULTI_QUEUE(urEnqueueKernelLaunchTest);

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
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queue, kernel, n_dimensions, &global_offset, &global_size, nullptr,
      GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
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

struct urKernelReclamationTest : uur::urKernelExecutionTest {
  void SetUp() override {
    ASSERT_EQ(zeInit(ZE_INIT_FLAG_GPU_ONLY), ZE_RESULT_SUCCESS);
    program_name = "foo";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }
};
INSTANTIATE_TEST_SUITE_P(
    , urKernelReclamationTest,
    testing::Combine(
        testing::ValuesIn(uur::DevicesEnvironment::instance->devices),
        testing::Values(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE,
                        static_cast<ur_queue_flag_t>(
                            UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE |
                            UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE))),
    uur::devicePrinter);

TEST_P(urKernelReclamationTest, SynchronizeAtKernelLimit) {
  size_t global_size = 1;
  size_t global_offset = 0;
  const size_t numCommandLists =
      std::get<1>(GetParam()) & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE ? 4
                                                                            : 1;
  auto zeGate = createZeEvent(context, device);
  uur::raii::Event gate;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeGate.get()), context, nullptr,
      gate.ptr()));
  ur_queue_properties_t properties = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                      nullptr,
                                      UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE};
  uur::raii::Queue otherQueue;
  ASSERT_SUCCESS(urQueueCreate(context, device, &properties, otherQueue.ptr()));

  // Fill each immediate list with 10,000 distinct kernels without blocking GPU
  // progress on a host signal.
  std::vector<uur::raii::Kernel> kernels(10000);
  for (auto &submittedKernel : kernels) {
    if (&submittedKernel == &kernels.back()) {
      // Reusing a kernel must not count towards the distinct-kernel limit.
      for (size_t i = 0; i < numCommandLists; ++i) {
        ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
            queue, kernels.front(), 1, &global_offset, &global_size, nullptr,
            GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
      }
      uint32_t references = 0;
      ASSERT_SUCCESS(urKernelGetInfo(kernels[1], UR_KERNEL_INFO_REFERENCE_COUNT,
                                     sizeof(references), &references, nullptr));
      EXPECT_EQ(references, numCommandLists + 1);
    }
    ASSERT_SUCCESS(
        urKernelCreate(program, kernel_name.c_str(), submittedKernel.ptr()));
    for (size_t i = 0; i < numCommandLists; ++i) {
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, submittedKernel, 1, &global_offset, &global_size, nullptr,
          GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
    }
  }

  uint32_t references = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, numCommandLists + 1);

  // The next launch on each list synchronizes and clears its retained kernels,
  // including the previous reference to the kernel being reused.
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue, 1, gate.ptr(), nullptr));
  auto launch = std::async(std::launch::async, [&] {
    return urEnqueueKernelLaunchWithArgsExp(
        queue, kernels.front(), 1, &global_offset, &global_size, nullptr,
        GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr);
  });
  EXPECT_EQ(launch.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  // Another queue must be able to enqueue the same kernel before the host
  // signals the gate. Signal even on timeout so a regression cannot hang.
  auto otherLaunch = std::async(std::launch::async, [&] {
    return urEnqueueKernelLaunchWithArgsExp(
        otherQueue, kernels.front(), 1, &global_offset, &global_size, nullptr,
        GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr);
  });
  EXPECT_EQ(otherLaunch.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_EQ(zeEventHostSignal(zeGate.get()), ZE_RESULT_SUCCESS);
  EXPECT_SUCCESS(otherLaunch.get());
  ASSERT_SUCCESS(launch.get());
  ASSERT_SUCCESS(urQueueFinish(otherQueue));
  for (size_t i = 1; i < numCommandLists; ++i) {
    ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
        queue, kernels.front(), 1, &global_offset, &global_size, nullptr,
        GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
  }
  EXPECT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, numCommandLists + 1);
  EXPECT_SUCCESS(urKernelGetInfo(kernels[1], UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 1u);
  EXPECT_SUCCESS(urKernelGetInfo(kernels.back(), UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 1u);

  for (size_t i = 0; i < numCommandLists; ++i) {
    ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
        queue, kernels.front(), 1, &global_offset, &global_size, nullptr,
        GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
  }
  EXPECT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, numCommandLists + 1);
  ASSERT_SUCCESS(urQueueFinish(queue));
  EXPECT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 1u);
}

TEST_P(urKernelReclamationTest, SynchronizeAtKernelLimitPreservesArguments) {
  constexpr size_t maxPendingKernels = 10000;
  const size_t numCommandLists =
      std::get<1>(GetParam()) & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE ? 4
                                                                            : 1;
  size_t global_size = 1;
  size_t global_offset = 0;

  std::shared_ptr<std::vector<char>> fillBinary;
  UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
      "fill", platform, fillBinary));
  uur::raii::Program fillProgram;
  UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->CreateProgram(
      platform, context, device, *fillBinary, nullptr, fillProgram.ptr()));
  ASSERT_SUCCESS(urProgramBuild(context, fillProgram, nullptr));
  const auto fillNames =
      uur::KernelsEnvironment::instance->GetEntryPointNames("fill");
  ASSERT_FALSE(fillNames.empty());
  uur::raii::Kernel sharedKernel;
  ASSERT_SUCCESS(urKernelCreate(fillProgram, fillNames.front().c_str(),
                                sharedKernel.ptr()));

  ur_queue_properties_t properties = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                      nullptr,
                                      UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE};
  uur::raii::Queue otherQueue;
  ASSERT_SUCCESS(urQueueCreate(context, device, &properties, otherQueue.ptr()));
  ur_queue_handle_t otherQueueHandle = otherQueue;
  ur_kernel_handle_t sharedKernelHandle = sharedKernel;
  uur::KernelLaunchHelper firstArgs(platform, context, sharedKernelHandle,
                                    queue);
  uur::KernelLaunchHelper secondArgs(platform, context, sharedKernelHandle,
                                     otherQueueHandle);
  uur::raii::Mem firstBuffer;
  uur::raii::Mem secondBuffer;
  constexpr uint32_t firstValue = 42;
  constexpr uint32_t secondValue = 43;
  UUR_RETURN_ON_FATAL_FAILURE(
      firstArgs.AddBuffer1DArg(sizeof(firstValue), firstBuffer.ptr(), nullptr));
  firstArgs.AddPodArg(firstValue);
  UUR_RETURN_ON_FATAL_FAILURE(secondArgs.AddBuffer1DArg(
      sizeof(secondValue), secondBuffer.ptr(), nullptr));
  secondArgs.AddPodArg(secondValue);

  std::vector<uur::raii::Kernel> kernels(maxPendingKernels);
  for (auto &submittedKernel : kernels) {
    ASSERT_SUCCESS(
        urKernelCreate(program, kernel_name.c_str(), submittedKernel.ptr()));
    for (size_t i = 0; i < numCommandLists; ++i) {
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, submittedKernel, 1, &global_offset, &global_size, nullptr,
          GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
    }
  }

  auto zeGate = createZeEvent(context, device);
  uur::raii::Event gate;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeGate.get()), context, nullptr,
      gate.ptr()));
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue, 1, gate.ptr(), nullptr));
  auto firstLaunch = std::async(std::launch::async, [&] {
    return urEnqueueKernelLaunchWithArgsExp(
        queue, sharedKernel, 1, &global_offset, &global_size, nullptr,
        firstArgs.GetNumArgs(), firstArgs.GetArgs(), nullptr, 0, nullptr,
        nullptr);
  });
  EXPECT_EQ(firstLaunch.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  auto secondLaunch = std::async(std::launch::async, [&] {
    return urEnqueueKernelLaunchWithArgsExp(
        otherQueue, sharedKernel, 1, &global_offset, &global_size, nullptr,
        secondArgs.GetNumArgs(), secondArgs.GetArgs(), nullptr, 0, nullptr,
        nullptr);
  });
  // Always release the gate so a locking regression fails without hanging.
  EXPECT_EQ(secondLaunch.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_EQ(zeEventHostSignal(zeGate.get()), ZE_RESULT_SUCCESS);
  EXPECT_SUCCESS(secondLaunch.get());
  EXPECT_SUCCESS(firstLaunch.get());
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_SUCCESS(urQueueFinish(otherQueue));
  uint32_t firstResult = 0;
  uint32_t secondResult = 0;
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, firstBuffer, true, 0,
                                        sizeof(firstResult), &firstResult, 0,
                                        nullptr, nullptr));
  ASSERT_SUCCESS(urEnqueueMemBufferRead(otherQueue, secondBuffer, true, 0,
                                        sizeof(secondResult), &secondResult, 0,
                                        nullptr, nullptr));
  EXPECT_EQ(firstResult, firstValue);
  EXPECT_EQ(secondResult, secondValue);
}

TEST_P(urKernelReclamationTest, CapturedKernelsAreNotSynchronizedAtLimit) {
  constexpr size_t maxPendingKernels = 10000;
  ur_bool_t supported = false;
  ASSERT_SUCCESS(urDeviceGetInfo(
      device, UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP,
      sizeof(supported), &supported, nullptr));
  if (!supported)
    GTEST_SKIP() << "Graph capture is not supported.";

  const size_t numCommandLists =
      (getQueueMode() & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? 4 : 1;
  size_t global_size = 1;
  std::vector<uur::raii::Kernel> pending(maxPendingKernels);
  for (auto &submittedKernel : pending) {
    ASSERT_SUCCESS(
        urKernelCreate(program, kernel_name.c_str(), submittedKernel.ptr()));
    for (size_t i = 0; i < numCommandLists; ++i) {
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, submittedKernel, 1, nullptr, &global_size, nullptr,
          GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
    }
  }

  std::vector<uur::raii::Kernel> kernels(maxPendingKernels + 1);
  for (auto &capturedKernel : kernels) {
    ASSERT_SUCCESS(
        urKernelCreate(program, kernel_name.c_str(), capturedKernel.ptr()));
  }

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    for (auto &capturedKernel : kernels) {
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, capturedKernel, 1, nullptr, &global_size, nullptr,
          GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
    }
  }
  uint32_t references = 0;
  ASSERT_SUCCESS(urKernelGetInfo(pending.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, numCommandLists + 1);
  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));

  // Ordinary submissions can retire pending kernels, but not captured ones.
  for (size_t i = 0; i < numCommandLists; ++i) {
    ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
        queue, kernel, 1, nullptr, &global_size, nullptr, GetNumArgs(),
        GetArgs(), nullptr, 0, nullptr, nullptr));
  }
  ASSERT_SUCCESS(urKernelGetInfo(pending.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 1u);
  ASSERT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 2u);

  ur_exp_executable_graph_handle_t executable = nullptr;
  ASSERT_SUCCESS(urGraphInstantiateGraphExp(graph, &executable));
  ASSERT_SUCCESS(urEnqueueGraphExp(queue, executable, 0, nullptr, nullptr));
  uur::raii::Event completed;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, completed.ptr()));
  ASSERT_SUCCESS(urEventWait(1, completed.ptr()));
  ASSERT_SUCCESS(urKernelGetInfo(kernels.front(),
                                 UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 2u);
  ASSERT_SUCCESS(urGraphExecutableGraphDestroyExp(executable));
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
  ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(urKernelReclamationTest, ImplicitlyCapturedKernelSurvivesLimitCleanup) {
  ur_bool_t supported = false;
  ASSERT_SUCCESS(urDeviceGetInfo(
      device, UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP,
      sizeof(supported), &supported, nullptr));
  if (!supported)
    GTEST_SKIP() << "Graph capture is not supported.";

  ur_queue_properties_t properties = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                      nullptr, getQueueMode()};
  uur::raii::Queue primaryQueue;
  ASSERT_SUCCESS(
      urQueueCreate(context, device, &properties, primaryQueue.ptr()));
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(primaryQueue));
  uur::raii::Event fork;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(primaryQueue, 0, nullptr, fork.ptr()));

  size_t global_size = 1;
  uur::raii::Event joined;
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queue, kernel, 1, nullptr, &global_size, nullptr, GetNumArgs(), GetArgs(),
      nullptr, 1, fork.ptr(), joined.ptr()));
  bool capturing = false;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue, &capturing));
  ASSERT_TRUE(capturing);
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(primaryQueue, 1, joined.ptr(), nullptr));
  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(primaryQueue, &graph));

  const size_t numCommandLists =
      (getQueueMode() & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? 4 : 1;
  std::vector<uur::raii::Kernel> kernels(10000);
  for (auto &submittedKernel : kernels) {
    ASSERT_SUCCESS(
        urKernelCreate(program, kernel_name.c_str(), submittedKernel.ptr()));
    for (size_t i = 0; i < numCommandLists; ++i) {
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, submittedKernel, 1, nullptr, &global_size, nullptr,
          GetNumArgs(), GetArgs(), nullptr, 0, nullptr, nullptr));
    }
  }
  for (size_t i = 0; i < numCommandLists; ++i) {
    ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
        queue, kernels.front(), 1, nullptr, &global_size, nullptr, GetNumArgs(),
        GetArgs(), nullptr, 0, nullptr, nullptr));
  }

  uint32_t references = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 2u);
  ASSERT_SUCCESS(urKernelGetInfo(kernels.back(), UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(references), &references, nullptr));
  EXPECT_EQ(references, 1u);

  ur_exp_executable_graph_handle_t executable = nullptr;
  ASSERT_SUCCESS(urGraphInstantiateGraphExp(graph, &executable));
  ASSERT_SUCCESS(
      urEnqueueGraphExp(primaryQueue, executable, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(primaryQueue));
  ASSERT_SUCCESS(urGraphExecutableGraphDestroyExp(executable));
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
  ASSERT_SUCCESS(urQueueFinish(queue));
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
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queues[0], kernel, 1, &global_offset, &global_size, nullptr, 0, nullptr,
      nullptr, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urEnqueueEventsWait(queues[1], 1, &event2, nullptr));
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queues[1], kernel, 1, &global_offset, &global_size, nullptr, 0, nullptr,
      nullptr, 0, nullptr, nullptr));

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
