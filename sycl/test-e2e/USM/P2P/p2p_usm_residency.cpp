//==-- p2p_usm_residency.cpp - P2P USM residency test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verify that the Level Zero v2 adapter correctly makes USM device memory
// resident on peer devices when P2P access is enabled.
//
// Phase 1: Allocates memory on dev0, fills it with a known pattern, enables
// P2P access from dev1 to dev0, then uses dev1's queue to copy the data to
// the host and verifies all values match the fill pattern.
//
// Phase 2 (opposite direction): Allocates memory on dev1, fills it with a
// different pattern, enables P2P access from dev0 to dev1, then uses dev0's
// queue to copy the data to the host and verifies correctness.
//
// Phase 3: Allocates memory on dev0, enables then disables P2P access from
// dev1, and verifies that a subsequent device-to-device memcpy via dev1's
// queue still succeeds (P2P controls residency, not hardware data transfer).
//
// Phase 4: Verify that a device-to-device memcpy succeeds after enabling P2P
// and that the transferred data matches the fill pattern.
//
// REQUIRES: level_zero && two-or-more-gpu-devices
// UNSUPPORTED: level_zero_v1_adapter
// UNSUPPORTED-INTENDED: Test is specific to the Level Zero v2 adapter.
//
// RUN: %{build} -o %t.out
// RUN: env UR_LOADER_USE_LEVEL_ZERO_V2=1 %{run} %t.out

#include <iostream>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

// Allocate N ints on srcQueue's device, fill with fillVal, enable P2P so that
// dstDev can access srcDev's allocations, copy to host via dstQueue, verify
// all values, then clean up.  Returns false on failure.
static bool testP2PRead(context &ctx, queue &srcQueue, device &srcDev,
                        queue &dstQueue, device &dstDev, size_t N, int fillVal,
                        const char *label) {
  int *src = sycl::malloc_device<int>(N, srcQueue);
  if (!src) {
    std::cout << label << ": device alloc failed. Skipping.\n";
    return true; // not a test failure
  }
  srcQueue.fill(src, fillVal, N).wait();

  // Enable P2P: dstDev may now access allocations on srcDev.  Under the
  // Level Zero v2 adapter this also makes the srcDev allocation resident
  // on dstDev.
  std::cout << "Enabling P2P: dstDev may now access allocations on srcDev.\n";
  dstDev.ext_oneapi_enable_peer_access(srcDev);

  std::vector<int> result(N, 0);
  dstQueue.memcpy(result.data(), src, N * sizeof(int)).wait();

  std::cout
      << "Disabling P2P: dstDev may no longer access allocations on srcDev.\n";
  dstDev.ext_oneapi_disable_peer_access(srcDev);
  sycl::free(src, ctx);

  for (size_t i = 0; i < N; ++i) {
    if (result[i] != fillVal) {
      std::cout << label << ": FAIL at index " << i << ": got " << result[i]
                << ", expected " << fillVal << "\n";
      return false;
    }
  }
  std::cout << label << ": OK\n";
  return true;
}

// Allocate N ints on srcQueue's device, fill with fillVal, enable P2P, then
// disable P2P, and verify that a device-to-device memcpy from dstQueue still
// succeeds.  P2P access controls memory residency, not hardware data transfer:
// Level Zero can still move data between devices via the interconnect
// regardless of residency state.
static bool testP2PCopySucceedsAfterDisable(context &ctx, queue &srcQueue,
                                            device &srcDev, queue &dstQueue,
                                            device &dstDev, size_t N,
                                            int fillVal, const char *label) {
  int *src = sycl::malloc_device<int>(N, srcQueue);
  if (!src) {
    std::cout << label << ": device alloc failed (src). Skipping.\n";
    return true;
  }

  int *dst = sycl::malloc_device<int>(N, dstQueue);
  if (!dst) {
    std::cout << label << ": device alloc failed (dst). Skipping.\n";
    sycl::free(src, ctx);
    return true;
  }

  srcQueue.fill(src, fillVal, N).wait();

  // Enable then disable P2P.
  std::cout << "Enabling P2P (temporarily).\n";
  dstDev.ext_oneapi_enable_peer_access(srcDev);
  std::cout << "Disabling P2P.\n";
  dstDev.ext_oneapi_disable_peer_access(srcDev);

  // Attempt a device-to-device memcpy after P2P has been revoked — must still
  // succeed because the adapter no longer blocks cross-device copies.
  bool copyOk = true;
  try {
    dstQueue.memcpy(dst, src, N * sizeof(int)).wait();
  } catch (sycl::exception &e) {
    std::cout << label << ": memcpy threw exception: " << e.what() << "\n";
    copyOk = false;
  }

  sycl::free(dst, ctx);
  sycl::free(src, ctx);

  if (!copyOk) {
    std::cout << label
              << ": FAIL — device-to-device memcpy failed after P2P disable\n";
    return false;
  }
  std::cout << label << ": OK (memcpy succeeded after P2P disable)\n";
  return true;
}

// Verify that a device-to-device memcpy from dstQueue succeeds after P2P is
// enabled and that the transferred data matches the fill pattern.
static bool testP2PReadSucceedsAfterEnable(context &ctx, queue &srcQueue,
                                           device &srcDev, queue &dstQueue,
                                           device &dstDev, size_t N,
                                           int fillVal, const char *label) {
  int *src = sycl::malloc_device<int>(N, srcQueue);
  if (!src) {
    std::cout << label << ": device alloc failed (src). Skipping.\n";
    return true;
  }

  int *dst = sycl::malloc_device<int>(N, dstQueue);
  if (!dst) {
    std::cout << label << ": device alloc failed (dst). Skipping.\n";
    sycl::free(src, ctx);
    return true;
  }

  srcQueue.fill(src, fillVal, N).wait();

  // Enable P2P: dstDev may now access allocations on srcDev.
  std::cout << label << ": enabling P2P.\n";
  dstDev.ext_oneapi_enable_peer_access(srcDev);

  bool copyOk = true;
  std::vector<int> result(N, 0);
  try {
    dstQueue.memcpy(dst, src, N * sizeof(int)).wait();
    // Read back to host for verification.
    dstQueue.memcpy(result.data(), dst, N * sizeof(int)).wait();
  } catch (sycl::exception &e) {
    std::cout << label << ": memcpy (P2P enabled) threw: " << e.what() << "\n";
    copyOk = false;
  }

  std::cout << label << ": disabling P2P.\n";
  dstDev.ext_oneapi_disable_peer_access(srcDev);
  sycl::free(dst, ctx);
  sycl::free(src, ctx);

  if (!copyOk)
    return false;

  for (size_t i = 0; i < N; ++i) {
    if (result[i] != fillVal) {
      std::cout << label << ": FAIL at index " << i << ": got " << result[i]
                << ", expected " << fillVal << "\n";
      return false;
    }
  }
  std::cout << label << ": OK\n";
  return true;
}

int main() {
  // Find a platform with at least two GPU devices.
  std::vector<device> gpus;
  for (auto &plat : platform::get_platforms()) {
    gpus = plat.get_devices(info::device_type::gpu);
    if (gpus.size() >= 2)
      break;
  }

  if (gpus.size() < 2) {
    std::cout << "Test requires at least two GPU devices on the same platform. "
                 "Skipping.\n";
    return 0;
  }

  device &dev0 = gpus[0];
  device &dev1 = gpus[1];

  std::cout << "Device 0: " << dev0.get_info<info::device::name>() << "\n";
  std::cout << "Device 1: " << dev1.get_info<info::device::name>() << "\n";

  // Both devices share a single context for cross-device USM.
  context ctx({dev0, dev1});
  queue q0(ctx, dev0);
  queue q1(ctx, dev1);

  // Allocation size must exceed the disjoint pool's MaxPoolableSize (4 MB for
  // device memory) so that the allocation goes directly to the memory provider
  // where residency is established.
  constexpr size_t N = 2 * 1024 * 1024; // 2M ints = 8 MB

  // Phase 1: dev1 reads dev0's memory (P2P: dev1 -> dev0).
  std::cout << "Phase 1: dev1 reads dev0's memory (P2P: dev1 -> dev0).\n";
  if (!dev1.ext_oneapi_can_access_peer(
          dev0, ext::oneapi::peer_access::access_supported)) {
    std::cout << "No hardware P2P support (dev1->dev0). Skipping.\n";
    return 0;
  }
  if (!testP2PRead(ctx, q0, dev0, q1, dev1, N, 0x42,
                   "Phase 1 (dev1 reads dev0)"))
    return 1;

  // Phase 2 (opposite): dev0 reads dev1's memory (P2P: dev0 -> dev1).
  std::cout
      << "Phase 2 (opposite): dev0 reads dev1's memory (P2P: dev0 -> dev1).\n";
  if (!dev0.ext_oneapi_can_access_peer(
          dev1, ext::oneapi::peer_access::access_supported)) {
    std::cout << "No hardware P2P support (dev0->dev1). Skipping phase 2.\n";
    std::cout << "PASS\n";
    return 0;
  }
  if (!testP2PRead(ctx, q1, dev1, q0, dev0, N, 0x55,
                   "Phase 2 (dev0 reads dev1)"))
    return 1;

  // Phase 3: verify that memcpy still succeeds after P2P is disabled.
  std::cout << "Phase 3: verify memcpy still succeeds after P2P is disabled.\n";
  if (!testP2PCopySucceedsAfterDisable(
          ctx, q0, dev0, q1, dev1, N, 0x77,
          "Phase 3 (dev1 reads dev0 after disable)"))
    return 1;

  // Phase 4: verify that memcpy succeeds after enabling P2P.
  std::cout << "Phase 4: verify memcpy succeeds after enabling P2P.\n";
  if (!testP2PReadSucceedsAfterEnable(ctx, q0, dev0, q1, dev1, N, 0xAA,
                                      "Phase 4 (dev1 reads dev0 after enable)"))
    return 1;

  std::cout << "PASS\n";
  return 0;
}
