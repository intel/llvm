// REQUIRES: level_zero, level_zero_v2_adapter, aspect-ext_oneapi_async_memory_alloc
// UNSUPPORTED: run-mode && !two-or-more-gpu-devices
// UNSUPPORTED-INTENDED: Peer residency requires two GPU devices.

// RUN: %{build} -o %t.out
// RUN: env UR_LOADER_USE_LEVEL_ZERO_V2=1 SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 UMF_LOG="level:debug;flush:debug;output:stderr;pid:yes" %{run} %t.out > %t.log 2>&1
// RUN: %{run-aux} FileCheck %s < %t.log || FileCheck %s --check-prefix=CHECK-SKIP < %t.log
// RUN: env UR_LOADER_USE_LEVEL_ZERO_V2=1 UMF_LOG="level:debug;flush:debug;output:stderr;pid:yes" %{run} %t.out > %t.default.log 2>&1
// RUN: %{run-aux} FileCheck %s < %t.default.log || FileCheck %s --check-prefix=CHECK-SKIP < %t.default.log

// An existing context's default asynchronous pool must track peer enable and
// disable. Inspect provider residency rather than hardware copy success.

#include <cassert>
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/platform.hpp>
#include <sycl/properties/queue_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  auto Devices =
      sycl::device{}.get_platform().get_devices(sycl::info::device_type::gpu);
  if (Devices.size() < 2) {
    std::cout << "SKIP: requires two GPU devices\n";
    return 0;
  }
  auto Peer = Devices[0];
  auto Command = Devices[1];
  if (!Peer.has(sycl::aspect::ext_oneapi_async_memory_alloc) ||
      !Command.ext_oneapi_can_access_peer(Peer)) {
    std::cout << "SKIP: requires asynchronous allocation and hardware P2P\n";
    return 0;
  }
  // CHECK-SKIP: {{^}}SKIP: requires {{(two GPU devices|asynchronous allocation and hardware P2P)$}}

  sycl::context Context({Peer, Command});
  sycl::queue Queue(Context, Peer, sycl::property::queue::in_order{});
  // Bypass the disjoint pool's 4 MB cache and keep allocations alive so each
  // phase reaches the provider instead of reusing memory with old residency.
  constexpr size_t Size = 8 * 1024 * 1024;
  std::cout << "Before enable" << std::endl;
  void *Before = syclexp::async_malloc(Queue, sycl::usm::alloc::device, Size);
  assert(Before);
  Queue.wait_and_throw();
  // CHECK: Before enable
  // CHECK: allocation {{(0[xX])?[0-9a-fA-F]+}} of size: 8388608 made resident on device [[PEER_DEVICE:(0[xX])?[0-9a-fA-F]+]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: Enabling peer access

  std::cout << "Enabling peer access" << std::endl;
  Command.ext_oneapi_enable_peer_access(Peer);
  std::cout << "After enable" << std::endl;
  void *Enabled = syclexp::async_malloc(Queue, sycl::usm::alloc::device, Size);
  assert(Enabled);
  Queue.wait_and_throw();
  // CHECK: After enable
  // CHECK: allocation [[ENABLED_PTR:(0[xX])?[0-9a-fA-F]+]] of size: 8388608 made resident on device [[PEER_DEVICE]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: allocation [[ENABLED_PTR]] of size: 8388608 made resident on device {{(0[xX])?[0-9a-fA-F]+}}
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: Disabling peer access

  std::cout << "Disabling peer access" << std::endl;
  Command.ext_oneapi_disable_peer_access(Peer);
  std::cout << "After disable" << std::endl;
  void *Disabled = syclexp::async_malloc(Queue, sycl::usm::alloc::device, Size);
  assert(Disabled);
  Queue.wait_and_throw();
  // CHECK: After disable
  // CHECK: allocation {{(0[xX])?[0-9a-fA-F]+}} of size: 8388608 made resident on device [[PEER_DEVICE]]
  // CHECK-NOT: allocation {{.*}} of size: 8388608 made resident
  // CHECK: Allocations checked
  std::cout << "Allocations checked" << std::endl;

  syclexp::async_free(Queue, Disabled);
  syclexp::async_free(Queue, Enabled);
  syclexp::async_free(Queue, Before);
  Queue.wait_and_throw();
}
