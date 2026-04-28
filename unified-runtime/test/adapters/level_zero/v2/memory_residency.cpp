// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ZES_ENABLE_SYSMAN=1 ./memory_residency-test
// REQUIRES: v2

#include "uur/fixtures.h"
#include "uur/utils.h"

#include <algorithm>
#include <vector>

using urMemoryResidencyTest = uur::urMultiDeviceContextTestTemplate<1>;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryResidencyTest);

TEST_P(urMemoryResidencyTest, allocatingDeviceMemoryWillResultInOOM) {
  static constexpr size_t allocSize = 1024 * 1024;

  if (!uur::isPVC(devices[0])) {
    GTEST_SKIP() << "Test requires a PVC device";
  }

  uint64_t initialMemFree = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFree, nullptr));

  if (initialMemFree < allocSize) {
    GTEST_SKIP() << "Not enough device memory available";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  uint64_t currentMemFree = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &currentMemFree, nullptr));

  // amount of free memory should decrease after making a memory allocation
  // resident
  ASSERT_LE(currentMemFree, initialMemFree);

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}

struct urMemoryMultiResidencyTest : uur::urMultiDeviceContextTestTemplate<2> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiDeviceContextTestTemplate<2>::SetUp());

    for (std::size_t i = 0; i < 2; i++) {
      ur_bool_t usm_p2p_support = false;
      ASSERT_SUCCESS(
          urDeviceGetInfo(devices[i], UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP,
                          sizeof(usm_p2p_support), &usm_p2p_support, nullptr));
      if (!usm_p2p_support) {
        GTEST_SKIP() << "EXP usm p2p feature is not supported.";
      }
    }
  }

  void TearDown() override {
    // Disable peer access if a test enabled it but did not clean up (e.g. due
    // to an assertion failure), so subsequent tests start from a clean state.
    if (peerAccessEnabled) {
      EXPECT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiDeviceContextTestTemplate<2>::TearDown());
  }

  // Returns true when hardware P2P connectivity exists in the direction used
  // by the tests: devices[1] accessing allocations on devices[0].
  bool hasHardwareP2PSupport() {
    int supported = 0;
    if (urUsmP2PPeerAccessGetInfoExp(
            devices[1], devices[0], UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORT,
            sizeof(int), &supported, nullptr) != UR_RESULT_SUCCESS) {
      return false;
    }
    return supported != 0;
  }

  // Whether peer access from devices[1] to devices[0] has been enabled by
  // this test and must be disabled in TearDown.
  bool peerAccessEnabled = false;
};

UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryMultiResidencyTest);

// Verify that allocating USM memory on devices[0] does NOT make it resident on
// devices[1] when peer access has not been enabled. Memory is physically only
// on the source device (devices[0]); peer device memory is unaffected.
TEST_P(urMemoryMultiResidencyTest, allocationInitiallyAbsentOnPeer) {
  if (!uur::isPVC(devices[0]) || !uur::isPVC(devices[1])) {
    GTEST_SKIP() << "Test requires PVC devices";
  }
  if (!hasHardwareP2PSupport()) {
    GTEST_SKIP() << "No hardware P2P connection between devices";
  }

  static constexpr size_t allocSize = 1024 * 1024;
  uint64_t initialMemFreeSource = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreeSource,
                                 nullptr));
  uint64_t initialMemFreePeer = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreePeer,
                                 nullptr));
  if (initialMemFreeSource < allocSize) {
    GTEST_SKIP() << "Not enough source device memory available";
  }
  if (initialMemFreePeer < allocSize) {
    GTEST_SKIP()
        << "Not enough peer device memory available for reliable check";
  }

  // Allocate on devices[0] WITHOUT enabling P2P.
  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  // Save return codes so ptr is freed before any ASSERT terminates the test.
  uint64_t currentMemFreeSource = 0;
  ur_result_t res1 =
      urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreeSource, nullptr);
  uint64_t currentMemFreePeer = 0;
  ur_result_t res2 =
      urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreePeer, nullptr);

  ASSERT_SUCCESS(urUSMFree(context, ptr));

  ASSERT_SUCCESS(res1);
  ASSERT_SUCCESS(res2);
  // Allocation is physically on devices[0]: its free memory must decrease.
  ASSERT_LE(currentMemFreeSource, initialMemFreeSource - allocSize);
  // Without P2P, the allocation must not be resident on the peer:
  // free memory on devices[1] must not have decreased by a full allocSize.
  ASSERT_GT(currentMemFreePeer, initialMemFreePeer - allocSize);
}

// Verify that enabling peer access succeeds and that a second enable attempt
// returns UR_RESULT_ERROR_INVALID_OPERATION (access already enabled). Confirms
// that source-device free memory decreases by at least allocSize, showing the
// allocation succeeded on devices[0] with P2P enabled. Also verifies end-to-end
// P2P data transfer: the allocation on devices[0] is filled with a known
// pattern and then read by devices[1]'s command engine; the result is checked
// for correctness to confirm the feature works in the correct direction.
// Note: peer-device free memory is not checked because
// UR_DEVICE_INFO_GLOBAL_MEM_FREE does not reliably reflect
// zeContextMakeMemoryResident behaviour for device USM allocations.
TEST_P(urMemoryMultiResidencyTest,
       enablePeerAccessStateMachineAndSourceAllocation) {
  if (!uur::isPVC(devices[0]) || !uur::isPVC(devices[1])) {
    GTEST_SKIP() << "Test requires PVC devices";
  }
  if (!hasHardwareP2PSupport()) {
    GTEST_SKIP() << "No hardware P2P connection between devices";
  }

  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  // A second enable must be rejected because access is already enabled.
  ASSERT_EQ(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]),
            UR_RESULT_ERROR_INVALID_OPERATION);

  static constexpr size_t allocSize = 1024 * 1024;
  uint64_t initialMemFreeSource = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreeSource,
                                 nullptr));
  if (initialMemFreeSource < allocSize) {
    GTEST_SKIP() << "Not enough source device memory available";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  // Fill ptr on devices[0] with a known pattern using devices[0]'s queue.
  static constexpr uint8_t fillPattern = 0xAB;
  ur_queue_handle_t srcQueue = nullptr;
  ur_result_t fillRes1 = urQueueCreate(context, devices[0], nullptr, &srcQueue);
  ur_result_t fillRes2 =
      (fillRes1 == UR_RESULT_SUCCESS)
          ? urEnqueueUSMFill(srcQueue, ptr, sizeof(fillPattern), &fillPattern,
                             allocSize, 0, nullptr, nullptr)
          : fillRes1;
  ur_result_t fillRes3 =
      (fillRes2 == UR_RESULT_SUCCESS) ? urQueueFinish(srcQueue) : fillRes2;
  if (srcQueue) {
    urQueueRelease(srcQueue);
  }

  // Verify end-to-end P2P access: copy ptr (on devices[0]) to dstPtr (on
  // devices[1]) using devices[1]'s queue, then read back for data verification.
  void *dstPtr = nullptr;
  ur_queue_handle_t peerQueue = nullptr;
  std::vector<uint8_t> hostData(allocSize);
  ur_result_t p2pRes1 = urUSMDeviceAlloc(context, devices[1], nullptr, nullptr,
                                         allocSize, &dstPtr);
  ur_result_t p2pRes2 =
      (p2pRes1 == UR_RESULT_SUCCESS)
          ? urQueueCreate(context, devices[1], nullptr, &peerQueue)
          : p2pRes1;
  // devices[1]'s engine reads ptr from devices[0] via P2P.
  ur_result_t p2pRes3 = (p2pRes2 == UR_RESULT_SUCCESS)
                            ? urEnqueueUSMMemcpy(peerQueue, true, dstPtr, ptr,
                                                 allocSize, 0, nullptr, nullptr)
                            : p2pRes2;
  // Read result back to host for verification.
  ur_result_t p2pRes4 =
      (p2pRes3 == UR_RESULT_SUCCESS)
          ? urEnqueueUSMMemcpy(peerQueue, true, hostData.data(), dstPtr,
                               allocSize, 0, nullptr, nullptr)
          : p2pRes3;

  // Save return code so ptr is freed before any ASSERT terminates the test.
  uint64_t currentMemFreeSource = 0;
  ur_result_t res =
      urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreeSource, nullptr);

  if (peerQueue) {
    urQueueRelease(peerQueue);
  }
  if (dstPtr) {
    urUSMFree(context, dstPtr);
  }
  ASSERT_SUCCESS(urUSMFree(context, ptr));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

  ASSERT_SUCCESS(res);
  // Allocation is physically on devices[0]: its free memory must decrease.
  ASSERT_LE(currentMemFreeSource, initialMemFreeSource - allocSize);

  ASSERT_SUCCESS(fillRes1);
  ASSERT_SUCCESS(fillRes2);
  ASSERT_SUCCESS(fillRes3);

  ASSERT_SUCCESS(p2pRes1);
  ASSERT_SUCCESS(p2pRes2);
  ASSERT_SUCCESS(p2pRes3);
  ASSERT_SUCCESS(p2pRes4);
  // All bytes transferred via P2P must match the fill pattern.
  EXPECT_TRUE(std::all_of(hostData.begin(), hostData.end(),
                          [](uint8_t b) { return b == fillPattern; }));
}

// Verify that disabling peer access succeeds and that a second disable attempt
// returns UR_RESULT_ERROR_INVALID_OPERATION (access already disabled). Confirms
// that source-device free memory still shows the allocation after peer access is
// disabled, proving the allocation on devices[0] remains valid.
// Note: peer-device eviction is not verified via free memory because
// UR_DEVICE_INFO_GLOBAL_MEM_FREE does not reliably reflect
// zeContextEvictMemory behaviour for device USM allocations.
TEST_P(urMemoryMultiResidencyTest,
       disablePeerAccessStateMachineAndSourceAllocationPersists) {
  if (!uur::isPVC(devices[0]) || !uur::isPVC(devices[1])) {
    GTEST_SKIP() << "Test requires PVC devices";
  }
  if (!hasHardwareP2PSupport()) {
    GTEST_SKIP() << "No hardware P2P connection between devices";
  }

  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  static constexpr size_t allocSize = 1024 * 1024;
  uint64_t initialMemFreeSource = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreeSource,
                                 nullptr));
  if (initialMemFreeSource < allocSize) {
    GTEST_SKIP() << "Not enough source device memory available";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  // Disable P2P; the runtime evicts the allocation from devices[1] (not
  // verified here, see the function-level comment above). Save return codes so
  // ptr is freed before any ASSERT terminates the test.
  ur_result_t res1 = urUsmP2PDisablePeerAccessExp(devices[1], devices[0]);
  if (res1 == UR_RESULT_SUCCESS) {
    peerAccessEnabled = false;
  }

  // A second disable must be rejected because access is already disabled.
  ur_result_t res2 = urUsmP2PDisablePeerAccessExp(devices[1], devices[0]);

  uint64_t currentMemFreeSource = 0;
  ur_result_t res3 =
      urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreeSource, nullptr);

  ASSERT_SUCCESS(urUSMFree(context, ptr));

  ASSERT_SUCCESS(res1);
  ASSERT_EQ(res2, UR_RESULT_ERROR_INVALID_OPERATION);
  ASSERT_SUCCESS(res3);
  // Allocation is physically on devices[0]: its free memory must decrease.
  ASSERT_LE(currentMemFreeSource, initialMemFreeSource - allocSize);
}
