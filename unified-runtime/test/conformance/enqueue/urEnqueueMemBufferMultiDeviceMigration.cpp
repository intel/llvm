// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Multi-device buffer tests that stress host-staged migration when a discrete
// buffer is accessed from different devices/queues (for example when device
// peer access is not available). Corresponds to L0 v2 discrete-buffer
// getDevicePtr migration ordering.
//
// The tests cover two migration paths inside getDevicePtr:
// - Async path (cmdList != nullptr): triggered by urEnqueueMem* operations.
// - Sync fallback (cmdList == nullptr): triggered by urMemGetNativeHandle.

#include <uur/fixtures.h>
#include <vector>

struct urEnqueueMemBufferMultiDeviceMigrationTest
    : uur::urMultiDeviceMemBufferQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiDeviceMemBufferQueueTest::SetUp());

    if (devices.size() < 2) {
      GTEST_SKIP() << "Test requires at least 2 devices";
    }

    // Check that the USM P2P extension is supported on both devices.
    for (size_t i = 0; i < 2; i++) {
      ur_bool_t usm_p2p_support = false;
      ASSERT_SUCCESS(
          urDeviceGetInfo(devices[i], UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP,
                          sizeof(usm_p2p_support), &usm_p2p_support, nullptr));
      if (!usm_p2p_support) {
        GTEST_SKIP() << "EXP usm p2p feature is not supported on device " << i;
      }
    }

    // This test exercises the host-mediated migration fallback, which is only
    // triggered when P2P peer access is not currently enabled between the two
    // devices. Best-effort disable peer access so that the fallback path is
    // exercised even on hardware that supports P2P; ignore
    // UR_RESULT_ERROR_INVALID_OPERATION (peer access was already disabled or
    // was never enabled).
    ur_result_t disableRes =
        urUsmP2PDisablePeerAccessExp(devices[0], devices[1]);
    if (disableRes != UR_RESULT_SUCCESS &&
        disableRes != UR_RESULT_ERROR_INVALID_OPERATION) {
      GTEST_SKIP() << "Could not disable P2P peer access: " << disableRes;
    }
  }
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urEnqueueMemBufferMultiDeviceMigrationTest);

TEST_P(urEnqueueMemBufferMultiDeviceMigrationTest,
       AsyncFillThenReadOnSecondQueueWithWait) {
  const uint32_t pattern = 0xA5A5A501;
  ur_event_handle_t fillEv = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &pattern,
                                        sizeof(pattern), 0, size, 0, nullptr,
                                        &fillEv));
  ASSERT_NE(fillEv, nullptr);

  std::vector<uint32_t> output(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, true, 0, size,
                                        output.data(), 1, &fillEv, nullptr));

  ASSERT_SUCCESS(urEventRelease(fillEv));

  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(pattern, output[i]) << "Mismatch at index " << i;
  }
}

TEST_P(urEnqueueMemBufferMultiDeviceMigrationTest,
       PingPongFillBetweenTwoDeviceQueues) {
  const uint32_t pattern1 = 0xC001D00u;
  ur_event_handle_t evFill1 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &pattern1,
                                        sizeof(pattern1), 0, size, 0, nullptr,
                                        &evFill1));
  ASSERT_NE(evFill1, nullptr);

  std::vector<uint32_t> stage1(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, true, 0, size,
                                        stage1.data(), 1, &evFill1, nullptr));
  ASSERT_SUCCESS(urEventRelease(evFill1));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(pattern1, stage1[i]);
  }

  const uint32_t pattern2 = 0xD00DAD00u;
  ur_event_handle_t evFill2 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[1], buffer, &pattern2,
                                        sizeof(pattern2), 0, size, 0, nullptr,
                                        &evFill2));
  ASSERT_NE(evFill2, nullptr);

  std::vector<uint32_t> stage2(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], buffer, true, 0, size,
                                        stage2.data(), 1, &evFill2, nullptr));
  ASSERT_SUCCESS(urEventRelease(evFill2));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(pattern2, stage2[i]);
  }
}

TEST_P(urEnqueueMemBufferMultiDeviceMigrationTest,
       ChainedAsyncOpsAcrossQueuesWithEvents) {
  const uint32_t patternA = 0x11111111u;
  ur_event_handle_t evFill = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &patternA,
                                        sizeof(patternA), 0, size, 0, nullptr,
                                        &evFill));
  ASSERT_NE(evFill, nullptr);

  std::vector<uint32_t> verifyA(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, true, 0, size,
                                        verifyA.data(), 1, &evFill, nullptr));
  ASSERT_SUCCESS(urEventRelease(evFill));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(patternA, verifyA[i]);
  }

  const uint32_t patternB = 0x22222222u;
  std::vector<uint32_t> hostB(count, patternB);
  ur_event_handle_t evWrite = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[1], buffer, true, 0, size,
                                         hostB.data(), 0, nullptr, &evWrite));
  ASSERT_NE(evWrite, nullptr);

  std::vector<uint32_t> verifyB(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], buffer, true, 0, size,
                                        verifyB.data(), 1, &evWrite, nullptr));
  ASSERT_SUCCESS(urEventRelease(evWrite));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(patternB, verifyB[i]);
  }
}

// Exercise the synchronous fallback migration path in getDevicePtr
// (cmdList == nullptr), which is triggered by urMemGetNativeHandle.
// Fill the buffer on device 0, then request its native pointer on device 1 to
// force a synchronous host-staged migration, then verify the data on device 1.
TEST_P(urEnqueueMemBufferMultiDeviceMigrationTest,
       SyncFallbackMigrationViaNativeHandle) {
  const uint32_t pattern = 0xDEADBEEFu;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &pattern,
                                        sizeof(pattern), 0, size, 0, nullptr,
                                        nullptr));
  ASSERT_SUCCESS(urQueueFinish(queues[0]));

  // urMemGetNativeHandle calls getDevicePtr with cmdList == nullptr,
  // triggering the synchronous device->host->device migration path.
  ur_native_handle_t nativePtr = 0;
  ASSERT_SUCCESS(urMemGetNativeHandle(buffer, devices[1], &nativePtr));
  ASSERT_NE(nativePtr, (ur_native_handle_t)0);

  std::vector<uint32_t> output(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, true, 0, size,
                                        output.data(), 0, nullptr, nullptr));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(pattern, output[i]) << "Mismatch at index " << i;
  }
}
