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

// Allocation size must exceed the disjoint pool MaxPoolableSize (4 MB for
// device USM) so that allocations bypass the pool's free-list cache and go
// directly to the UMF memory provider, which is where residency is
// established.  Using pool-sized allocations leads to cross-test interference:
// a freed allocation stays in the pool cache with stale residency from a
// previous test, causing the next test's free-memory measurement to be wrong.
static constexpr size_t kAllocSize = 5 * 1024 * 1024;

using urMemoryResidencyTest = uur::urMultiDeviceContextTestTemplate<1>;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryResidencyTest);

TEST_P(urMemoryResidencyTest, allocatingDeviceMemoryWillResultInOOM) {
  constexpr size_t allocSize = kAllocSize;

  if (!uur::isPVC(devices[0])) {
    GTEST_SKIP() << "Test requires a PVC device";
  }

  uint64_t initialMemFree = 0;
  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &initialMemFree, nullptr),
      UR_DEVICE_INFO_GLOBAL_MEM_FREE);

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

    if (!uur::isPVC(devices[0]) || !uur::isPVC(devices[1])) {
      GTEST_SKIP() << "Test requires PVC devices";
    }

    if (!hasHardwareP2PSupport()) {
      GTEST_SKIP() << "No hardware P2P connection between devices";
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

  // Allocate allocSize bytes on devices[0], submit a fill command, wait for
  // completion, and release the queue.  The allocation is NOT freed; the caller
  // is responsible for calling urUSMFree(*outPtr).
  void allocAndFillOnDevice0(size_t allocSize, uint8_t fillPattern,
                             void **outPtr) {
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[0], nullptr, nullptr,
                                    allocSize, outPtr));
    ur_queue_handle_t queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, devices[0], nullptr, &queue));
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, *outPtr, sizeof(fillPattern),
                                    &fillPattern, allocSize, 0, nullptr,
                                    nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    urQueueRelease(queue);
  }

  // Whether peer access from devices[1] to devices[0] has been enabled by
  // this test and must be disabled in TearDown.
  bool peerAccessEnabled = false;
};

UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryMultiResidencyTest);

// Verify that allocating USM memory on devices[0] does NOT make it resident on
// devices[1] when peer access has not been enabled.  Only the peer device's
// free memory is checked: it must not decrease by allocSize.  The source
// device free memory is intentionally not checked because deferred frees from
// earlier tests complete asynchronously and make the source baseline
// unreliable; that property is already covered by
// allocatingDeviceMemoryWillResultInOOM.
TEST_P(urMemoryMultiResidencyTest, allocationInitiallyAbsentOnPeer) {
  constexpr size_t allocSize = kAllocSize;

  uint64_t initialMemFreePeer = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreePeer,
                                 nullptr));
  if (initialMemFreePeer < allocSize) {
    GTEST_SKIP()
        << "Not enough peer device memory available for reliable check";
  }

  // Allocate on devices[0] WITHOUT enabling P2P.
  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  uint64_t currentMemFreePeer = 0;
  ur_result_t res =
      urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreePeer, nullptr);

  ASSERT_SUCCESS(urUSMFree(context, ptr));

  ASSERT_SUCCESS(res);
  // Without P2P, the allocation must not be resident on the peer:
  // free memory on devices[1] must not have decreased by a full allocSize.
  ASSERT_GT(currentMemFreePeer, initialMemFreePeer - allocSize);
}

TEST_P(urMemoryMultiResidencyTest, allocAfterEnablingPeerAccess) {
  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  constexpr size_t allocSize = kAllocSize;
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
  ASSERT_SUCCESS(urUSMFree(context, ptr));

  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;
}

TEST_P(urMemoryMultiResidencyTest, allocBeforeEnablingPeerAccess) {
  constexpr size_t allocSize = kAllocSize;
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

  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}

// Verify that enabling peer access succeeds and that a second enable attempt
// returns UR_RESULT_ERROR_INVALID_OPERATION (access already enabled). Also
// verifies end-to-end P2P data transfer: the allocation on devices[0] is filled
// with a known pattern and then read by devices[1]'s command engine; the result
// is checked for correctness to confirm the feature works in the correct
// direction.
TEST_P(urMemoryMultiResidencyTest,
       enablePeerAccessStateMachineAndSourceAllocation) {
  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  // A second enable must be rejected because access is already enabled.
  ASSERT_EQ(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]),
            UR_RESULT_ERROR_INVALID_OPERATION);

  constexpr size_t allocSize = kAllocSize;

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

  if (peerQueue) {
    urQueueRelease(peerQueue);
  }
  if (dstPtr) {
    urUSMFree(context, dstPtr);
  }
  ASSERT_SUCCESS(urUSMFree(context, ptr));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

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
// returns UR_RESULT_ERROR_INVALID_OPERATION (access already disabled).
// Source-device free memory is not checked because deferred frees from earlier
// tests complete asynchronously and make the baseline unreliable; that property
// is already covered by allocatingDeviceMemoryWillResultInOOM.
TEST_P(urMemoryMultiResidencyTest, disablePeerAccessStateMachine) {
  // Enable devices[1] to access allocations on devices[0], so that new
  // allocations on devices[0] are made resident on devices[1] too.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  // Disable P2P; the runtime evicts the allocation from devices[1].
  ur_result_t res1 = urUsmP2PDisablePeerAccessExp(devices[1], devices[0]);
  if (res1 == UR_RESULT_SUCCESS) {
    peerAccessEnabled = false;
  }

  // A second disable must be rejected because access is already disabled.
  ur_result_t res2 = urUsmP2PDisablePeerAccessExp(devices[1], devices[0]);

  ASSERT_SUCCESS(res1);
  ASSERT_EQ(res2, UR_RESULT_ERROR_INVALID_OPERATION);
}

// Verify that USM memory allocated on devices[0] and filled with a known
// pattern can be correctly read by devices[1] when P2P access is enabled.
// P2P is enabled before the allocation here, but enabling it after the
// allocation is equally valid: urUsmP2PEnablePeerAccessExp calls
// changeResidentDevice on all contexts, which retroactively makes existing
// allocations resident on the peer device.
TEST_P(urMemoryMultiResidencyTest, p2pReadSucceedsWithPeerAccessEnabled) {
  constexpr size_t allocSize = kAllocSize;
  static constexpr uint8_t fillPattern = 0xAB;

  // Enable P2P: devices[1] can now access allocations on devices[0].
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  // Allocate on devices[0] and fill with a known pattern.
  void *srcPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[0], nullptr, nullptr,
                                  allocSize, &srcPtr));
  ur_queue_handle_t srcQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[0], nullptr, &srcQueue));
  ASSERT_SUCCESS(urEnqueueUSMFill(srcQueue, srcPtr, sizeof(fillPattern),
                                  &fillPattern, allocSize, 0, nullptr,
                                  nullptr));
  ASSERT_SUCCESS(urQueueFinish(srcQueue));
  urQueueRelease(srcQueue);

  // Copy srcPtr (on devices[0]) to dstPtr (on devices[1]) using devices[1]'s
  // queue (P2P read), then copy dstPtr back to the host for data verification.
  void *dstPtr = nullptr;
  ur_queue_handle_t peerQueue = nullptr;
  std::vector<uint8_t> hostData(allocSize, 0);
  ur_result_t res1 = urUSMDeviceAlloc(context, devices[1], nullptr, nullptr,
                                      allocSize, &dstPtr);
  ur_result_t res2 =
      (res1 == UR_RESULT_SUCCESS)
          ? urQueueCreate(context, devices[1], nullptr, &peerQueue)
          : res1;
  ur_result_t res3 = (res2 == UR_RESULT_SUCCESS)
                         ? urEnqueueUSMMemcpy(peerQueue, true, dstPtr, srcPtr,
                                              allocSize, 0, nullptr, nullptr)
                         : res2;
  ur_result_t res4 =
      (res3 == UR_RESULT_SUCCESS)
          ? urEnqueueUSMMemcpy(peerQueue, true, hostData.data(), dstPtr,
                               allocSize, 0, nullptr, nullptr)
          : res3;

  if (peerQueue) {
    urQueueRelease(peerQueue);
  }
  if (dstPtr) {
    urUSMFree(context, dstPtr);
  }
  ASSERT_SUCCESS(urUSMFree(context, srcPtr));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

  ASSERT_SUCCESS(res1);
  ASSERT_SUCCESS(res2);
  ASSERT_SUCCESS(res3);
  ASSERT_SUCCESS(res4);
  // All bytes transferred via P2P must match the fill pattern.
  EXPECT_TRUE(std::all_of(hostData.begin(), hostData.end(),
                          [](uint8_t b) { return b == fillPattern; }));
}

// Verify that a USM copy from devices[1]'s queue succeeds after P2P is
// enabled, and that the data transferred matches the fill pattern written on
// devices[0].
TEST_P(urMemoryMultiResidencyTest, p2pReadSucceedsAfterEnablingAccess) {
  constexpr size_t allocSize = kAllocSize;
  static constexpr uint8_t fillPattern = 0xCD;

  uint64_t initialMemFreeSource = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreeSource,
                                 nullptr));
  if (initialMemFreeSource < allocSize) {
    GTEST_SKIP() << "Not enough source device memory available";
  }

  uint64_t initialMemFreePeer = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreePeer,
                                 nullptr));
  if (initialMemFreePeer < allocSize) {
    GTEST_SKIP() << "Not enough peer device memory available";
  }

  void *srcPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[0], nullptr, nullptr,
                                  allocSize, &srcPtr));

  void *dstPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[1], nullptr, nullptr,
                                  allocSize, &dstPtr));

  ur_queue_handle_t srcQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[0], nullptr, &srcQueue));
  ASSERT_SUCCESS(urEnqueueUSMFill(srcQueue, srcPtr, sizeof(fillPattern),
                                  &fillPattern, allocSize, 0, nullptr,
                                  nullptr));
  ASSERT_SUCCESS(urQueueFinish(srcQueue));
  urQueueRelease(srcQueue);

  ur_queue_handle_t peerQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[1], nullptr, &peerQueue));

  // Enable P2P: devices[1] can now access allocations on devices[0].
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  // Retry the copy — must succeed now.
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(peerQueue, true, dstPtr, srcPtr, allocSize,
                                    0, nullptr, nullptr));

  // Read result back to host for verification.
  std::vector<uint8_t> hostData(allocSize);
  ur_queue_handle_t hostQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[1], nullptr, &hostQueue));
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(hostQueue, true, hostData.data(), dstPtr,
                                    allocSize, 0, nullptr, nullptr));
  urQueueRelease(hostQueue);

  urQueueRelease(peerQueue);
  urUSMFree(context, dstPtr);
  ASSERT_SUCCESS(urUSMFree(context, srcPtr));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

  EXPECT_TRUE(std::all_of(hostData.begin(), hostData.end(),
                          [](uint8_t b) { return b == fillPattern; }));
}

// Verify that a USM copy from devices[1]'s queue succeeds even after peer
// access has been revoked.  P2P access controls memory residency, not hardware
// data transfer: Level Zero can still move data between devices via the
// interconnect regardless of residency state.  A successful copy is first
// performed with P2P enabled to confirm the setup is correct; then P2P is
// disabled and the same copy is expected to succeed.
TEST_P(urMemoryMultiResidencyTest, p2pCopySucceedsAfterRevokingAccess) {
  constexpr size_t allocSize = kAllocSize;

  uint64_t initialMemFreeSource = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreeSource,
                                 nullptr));
  if (initialMemFreeSource < allocSize) {
    GTEST_SKIP() << "Not enough source device memory available";
  }

  uint64_t initialMemFreePeer = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreePeer,
                                 nullptr));
  if (initialMemFreePeer < allocSize) {
    GTEST_SKIP() << "Not enough peer device memory available";
  }

  void *srcPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[0], nullptr, nullptr,
                                  allocSize, &srcPtr));

  void *dstPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[1], nullptr, nullptr,
                                  allocSize, &dstPtr));

  // Enable P2P and confirm a copy from devices[0] to devices[1] succeeds.
  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = true;

  ur_queue_handle_t peerQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[1], nullptr, &peerQueue));

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(peerQueue, true, dstPtr, srcPtr, allocSize,
                                    0, nullptr, nullptr));

  // Revoke P2P access.
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[1], devices[0]));
  peerAccessEnabled = false;

  // Copy must still succeed: P2P controls residency, not hardware transfer.
  ur_result_t copyResult = urEnqueueUSMMemcpy(peerQueue, true, dstPtr, srcPtr,
                                              allocSize, 0, nullptr, nullptr);

  urQueueRelease(peerQueue);
  urUSMFree(context, dstPtr);
  ASSERT_SUCCESS(urUSMFree(context, srcPtr));

  ASSERT_SUCCESS(copyResult);
}

// Verify that a USM allocation on devices[0] is NOT made resident on
// devices[1] when P2P access has not been enabled.  The feature under test
// restricts residency, not hardware access: Level Zero hardware can still
// transfer data cross-device via the interconnect regardless of residency
// state, so the copy result is not checked here.  The observable guarantee
// is that devices[1] free memory must not decrease by a full allocSize,
// proving the allocation was never pinned on the peer device.
//
// The peer free-memory check is placed immediately after the allocation —
// before the fill — to keep the measurement window as short as possible.
// Residency decisions are made at allocation time; the subsequent fill on
// devices[0] does not affect devices[1] residency and is excluded from the
// window to avoid false failures due to concurrent GPU activity on shared
// CI machines.
TEST_P(urMemoryMultiResidencyTest, allocationNotResidentOnPeerWithoutP2P) {
  constexpr size_t allocSize = kAllocSize;
  static constexpr uint8_t fillPattern = 0xAB;

  // Warm-up pass: perform one full alloc+fill+free cycle before taking the
  // baseline.  The first submission to a queue can trigger one-time driver
  // initialization (command-list pool creation, kernel loading, event-pool
  // allocation, etc.) that transiently affects free-memory counters on both
  // devices.  Completing this initialization up-front ensures the baseline
  // measurement is taken in a stable state and avoids sporadic failures caused
  // by first-time overhead being counted against the test's budget.
  void *warmupPtr = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      allocAndFillOnDevice0(allocSize, fillPattern, &warmupPtr));
  ASSERT_SUCCESS(urUSMFree(context, warmupPtr));

  uint64_t initialMemFreePeer = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(uint64_t), &initialMemFreePeer,
                                 nullptr));
  if (initialMemFreePeer < allocSize) {
    GTEST_SKIP()
        << "Not enough peer device memory available for reliable check";
  }

  // Allocate on devices[0] WITHOUT enabling P2P — must not consume
  // devices[1] memory.
  void *srcPtr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, devices[0], nullptr, nullptr,
                                  allocSize, &srcPtr));

  // Measure peer free memory immediately after allocation, before any GPU
  // work, to minimise the observation window for concurrent external
  // allocations on devices[1].
  uint64_t currentMemFreePeer = 0;
  ur_result_t memRes =
      urDeviceGetInfo(devices[1], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                      sizeof(uint64_t), &currentMemFreePeer, nullptr);

  ur_queue_handle_t srcQueue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, devices[0], nullptr, &srcQueue));
  ASSERT_SUCCESS(urEnqueueUSMFill(srcQueue, srcPtr, sizeof(fillPattern),
                                  &fillPattern, allocSize, 0, nullptr,
                                  nullptr));
  ASSERT_SUCCESS(urQueueFinish(srcQueue));
  urQueueRelease(srcQueue);

  ASSERT_SUCCESS(urUSMFree(context, srcPtr));
  ASSERT_SUCCESS(memRes);
  // Without P2P the allocation must not be resident on devices[1]: free
  // memory on devices[1] must not have decreased by a full allocSize.
  ASSERT_GT(currentMemFreePeer, initialMemFreePeer - allocSize);
}
