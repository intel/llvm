// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Out-of-order queue signal behavior

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q; // out-of-order by default
  auto event = syclex::make_event(q.get_context());

  std::vector<int> data(N, 0);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class OutOfOrderOp1>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = 5; });
  });

  // Signal creates barrier in out-of-order queue
  syclex::enqueue_signal_event(q, event);
  event.wait();

  auto host_acc = buf.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 5);
  }

  return 0;
}
