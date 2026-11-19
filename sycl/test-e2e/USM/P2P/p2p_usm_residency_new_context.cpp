// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: level_zero, gpu, level_zero_v2_adapter
// UNSUPPORTED: run-mode && !two-or-more-gpu-devices
// UNSUPPORTED-INTENDED: Directional P2P residency requires two GPU devices.

// RUN: %{build} -o %t.out
// RUN: env UR_LOADER_USE_LEVEL_ZERO_V2=1 SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 UMF_LOG="level:debug;flush:debug;output:stderr;pid:yes" %{run} %t.out > %t.log 2>&1
// RUN: %{run-aux} FileCheck %s < %t.log || FileCheck %s --check-prefix=CHECK-SKIP < %t.log
// RUN: env UR_LOADER_USE_LEVEL_ZERO_V2=1 SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 UMF_LOG="level:debug;flush:debug;output:stderr;pid:yes" %{run} %t.out reverse > %t.reverse.log 2>&1
// RUN: %{run-aux} FileCheck %s < %t.reverse.log || FileCheck %s --check-prefix=CHECK-SKIP < %t.reverse.log

// Enable only command -> peer before creating the context and its USM pools.
// Existing pools are updated directly by enable_peer_access, which bypasses
// the peers[] lookup whose direction this test checks. Inspect UMF residency
// operations rather than memcpy success or fluctuating free-memory counters.

#include <cassert>
#include <iostream>
#include <utility>

#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>

int main(int argc, char **) {
  auto Devices = sycl::platform(sycl::gpu_selector_v)
                     .get_devices(sycl::info::device_type::gpu);
  if (Devices.size() < 2) {
    std::cout << "SKIP: requires two GPU devices\n";
    return 0;
  }

  auto Peer = Devices[0];
  auto Command = Devices[1];
  if (argc > 1)
    std::swap(Peer, Command);
  if (!Command.ext_oneapi_can_access_peer(Peer) ||
      !Peer.ext_oneapi_can_access_peer(Command)) {
    std::cout << "SKIP: requires hardware P2P support in both directions\n";
    return 0;
  }
  // CHECK-SKIP: {{^}}SKIP: requires {{(two GPU devices|hardware P2P support in both directions)$}}

  Command.ext_oneapi_enable_peer_access(Peer);
  sycl::context Context({Peer, Command});

  // Bypass the disjoint pool's 4 MB cache so each allocation reaches UMF.
  constexpr size_t Size = 8 * 1024 * 1024;
  std::cout << "Allocating on peer" << std::endl;
  void *PeerPtr = sycl::malloc_device(Size, Peer, Context);
  assert(PeerPtr);

  // The peer's allocation must be resident on its owner AND the command
  // device. Both log entries must refer to the same allocation.
  // CHECK: Allocating on peer
  // CHECK: allocation [[PEER_PTR:(0[xX])?[0-9a-fA-F]+]] of size: 8388608 made resident on device [[PEER_DEVICE:(0[xX])?[0-9a-fA-F]+]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: allocation [[PEER_PTR]] of size: 8388608 made resident on device [[COMMAND_DEVICE:(0[xX])?[0-9a-fA-F]+]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident

  std::cout << "Allocating on command" << std::endl;
  void *CommandPtr = sycl::malloc_device(Size, Command, Context);
  assert(CommandPtr);

  // The reverse direction was never enabled: this allocation must be
  // resident only on its owner, not on the peer.
  // CHECK: Allocating on command
  // CHECK: allocation {{(0[xX])?[0-9a-fA-F]+}} of size: 8388608 made resident on device [[COMMAND_DEVICE]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: Allocations checked
  std::cout << "Allocations checked" << std::endl;

  sycl::free(CommandPtr, Context);
  sycl::free(PeerPtr, Context);
  Command.ext_oneapi_disable_peer_access(Peer);
}
