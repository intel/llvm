// RUN: %{build} -fsycl-instrument-device-code -o %t.out
// RUN: %{run} %t.out

#include "sycl/atomic_ref.hpp"
#include "sycl/detail/core.hpp"

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
            atomic_ref<int, memory_order::relaxed, memory_scope::device,
                       access::address_space::global_space>(source_acc[0]);
        // Store source value into target
        target_acc[0] = source_atomic.load();
        // Nullify source
        source_atomic.store(0);
      });
    });
  }

  return 0;
}
