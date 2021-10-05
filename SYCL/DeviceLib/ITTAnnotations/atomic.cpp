// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fsycl-instrument-device-code %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-instrument-device-code %s -o %t.cpu.out	\
// RUN: -fsycl-targets=spir64_x86_64-unknown-unknown
// RUN: %CPU_RUN_PLACEHOLDER %t.cpu.out

#include "CL/sycl.hpp"

using namespace sycl;

int main() {
  queue q{};

  int source = 42;
  int target = 0;
  {
    buffer<int> source_buf(&source, 1);
    buffer<int> target_buf(&target, 1);

    // Ensure that a simple kernel gets run when instrumented with
    // ITT start/finish annotations and ITT atomic start/finish annotations.
    q.submit([&](handler &cgh) {
      auto source_acc =
          source_buf.template get_access<access::mode::read_write>(cgh);
      auto target_acc =
          target_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class simple_atomic_kernel>([=]() {
        auto source_atomic =
            ext::oneapi::atomic_ref<int, memory_order::relaxed,
                                    memory_scope::device,
                                    access::address_space::global_space>(
                source_acc[0]);
        // Store source value into target
        target_acc[0] = source_atomic.load();
        // Nullify source
        source_atomic.store(0);
      });
    });
  }

  return 0;
}
