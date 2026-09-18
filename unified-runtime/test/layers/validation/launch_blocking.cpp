// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// COM: This test doesn't have any filecheck rules
// RUN: %use-mock launch_blocking-test

#include "fixtures.hpp"

namespace {
int QueueFinishCount = 0;

ur_result_t countQueueFinish(void *) {
  ++QueueFinishCount;
  return UR_RESULT_SUCCESS;
}

bool Capturing = false;

ur_result_t reportCapturing(void *pParams) {
  const auto &Params =
      *static_cast<ur_queue_is_graph_capture_enabled_exp_params_t *>(pParams);
  **Params.ppResult = Capturing;
  return UR_RESULT_SUCCESS;
}

struct launchBlockingTest : ::testing::Test {
  void initWithLayer(const char *Layer) {
    QueueFinishCount = 0;
    Capturing = false;
    mock::getCallbacks().set_replace_callback("urQueueFinish",
                                              &countQueueFinish);
    mock::getCallbacks().set_replace_callback("urQueueIsGraphCaptureEnabledExp",
                                              &reportCapturing);

    ASSERT_EQ(urLoaderConfigCreate(&LoaderConfig), UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderConfigEnableLayer(LoaderConfig, Layer),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderInit(0, LoaderConfig), UR_RESULT_SUCCESS);

    uint32_t Count = 0;
    ASSERT_EQ(urAdapterGet(0, nullptr, &Count), UR_RESULT_SUCCESS);
    ASSERT_GT(Count, 0u);
    Adapters.resize(Count);
    ASSERT_EQ(urAdapterGet(Count, Adapters.data(), nullptr), UR_RESULT_SUCCESS);
    ASSERT_EQ(urPlatformGet(Adapters[0], 1, &Platform, nullptr),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urDeviceGet(Platform, UR_DEVICE_TYPE_ALL, 1, &Device, nullptr),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urContextCreate(1, &Device, nullptr, &Context),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urQueueCreate(Context, Device, nullptr, &Queue),
              UR_RESULT_SUCCESS);
    QueueFinishCount = 0;
  }

  void TearDown() override {
    if (Queue)
      urQueueRelease(Queue);
    if (Context)
      urContextRelease(Context);
    if (Device)
      urDeviceRelease(Device);
    for (auto Adapter : Adapters)
      urAdapterRelease(Adapter);
    if (LoaderConfig)
      urLoaderConfigRelease(LoaderConfig);
    mock::getCallbacks().resetCallbacks();
    urLoaderTearDown();
  }

  // Any command that is not a marker; USMfill, for instance.
  ur_result_t enqueueWork() {
    uint8_t Pattern = 0;
    return urEnqueueUSMFill(Queue, &Storage, sizeof(Pattern), &Pattern,
                            sizeof(Storage), 0, nullptr, nullptr);
  }

  ur_loader_config_handle_t LoaderConfig = nullptr;
  std::vector<ur_adapter_handle_t> Adapters;
  ur_platform_handle_t Platform = nullptr;
  ur_device_handle_t Device = nullptr;
  ur_context_handle_t Context = nullptr;
  ur_queue_handle_t Queue = nullptr;
  uint64_t Storage = 0;
};

TEST_F(launchBlockingTest, DrainsAfterCommand) {
  initWithLayer("UR_LAYER_LAUNCH_BLOCKING");
  ASSERT_EQ(enqueueWork(), UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 1);
}

TEST_F(launchBlockingTest, DoesNotDrainWhenDisabled) {
  initWithLayer("UR_LAYER_PARAMETER_VALIDATION");
  ASSERT_EQ(enqueueWork(), UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 0);
}

// Blocking is a debugging aid, not a validation check.
TEST_F(launchBlockingTest, FullValidationDoesNotEnableIt) {
  initWithLayer("UR_LAYER_FULL_VALIDATION");
  ASSERT_EQ(enqueueWork(), UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 0);
}

// A marker enqueues no work of its own, and its wait list may hold an event the
// application signals later.
TEST_F(launchBlockingTest, DoesNotDrainAfterBarrier) {
  initWithLayer("UR_LAYER_LAUNCH_BLOCKING");
  ASSERT_EQ(urEnqueueEventsWaitWithBarrier(Queue, 0, nullptr, nullptr),
            UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 0);
}

// Commands are recorded rather than run, so there is nothing to wait for.
TEST_F(launchBlockingTest, DoesNotDrainWhileCapturingAGraph) {
  initWithLayer("UR_LAYER_LAUNCH_BLOCKING");
  Capturing = true;
  ASSERT_EQ(enqueueWork(), UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 0);
}

// A fault the drain reports becomes the command's result, which is the point of
// blocking: it is reported at the command that caused it.
TEST_F(launchBlockingTest, ReportsAFailedDrain) {
  initWithLayer("UR_LAYER_LAUNCH_BLOCKING");
  mock::getCallbacks().set_replace_callback("urQueueFinish", [](void *) {
    ++QueueFinishCount;
    return UR_RESULT_ERROR_DEVICE_LOST;
  });
  EXPECT_EQ(enqueueWork(), UR_RESULT_ERROR_DEVICE_LOST);
  EXPECT_EQ(QueueFinishCount, 1);
}

// A failed enqueue submitted nothing to wait for.
TEST_F(launchBlockingTest, DoesNotDrainWhenTheCommandFails) {
  initWithLayer("UR_LAYER_LAUNCH_BLOCKING");
  mock::getCallbacks().set_replace_callback(
      "urEnqueueUSMFill", [](void *) { return UR_RESULT_ERROR_INVALID_VALUE; });
  ASSERT_NE(enqueueWork(), UR_RESULT_SUCCESS);
  EXPECT_EQ(QueueFinishCount, 0);
}
} // namespace
