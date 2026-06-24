// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Event reassociation with pending dependency

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  auto event = syclex::make_event(q.get_context());

  std::vector<int> data1(N, 1);
  std::vector<int> data2(N, 0);
  sycl::buffer<int> buf1(data1.data(), sycl::range<1>(N));
  sycl::buffer<int> buf2(data2.data(), sycl::range<1>(N));

  // First operation
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf1.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class ReassocOp1>(sycl::range<1>(N),
                                       [=](sycl::id<1> idx) { acc[idx] = 10; });
  });
  syclex::enqueue_signal_event(q, event);

  // Second operation depends on event
  auto evt2 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(event);
    auto acc = buf2.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class ReassocOp2>(sycl::range<1>(N),
                                       [=](sycl::id<1> idx) { acc[idx] = 20; });
  });

  // Reassociate event with new operation (legal before evt2 completes)
  syclex::enqueue_signal_event(q, event);

  evt2.wait();
  event.wait();

  auto host_acc1 = buf1.get_host_access();
  auto host_acc2 = buf2.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc1[i] == 10);
    assert(host_acc2[i] == 20);
  }

  return 0;
}
