// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Event reuse across multiple operations

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  auto event = syclex::make_event(q.get_context());

  std::vector<int> data(N, 0);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  // First operation
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class Kernel1>(sycl::range<1>(N),
                                    [=](sycl::id<1> idx) { acc[idx] = 1; });
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Reuse event for second operation
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class Kernel2>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] + 10; });
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Reuse event for third operation
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class Kernel3>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] * 2; });
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Verify final result: (1 + 10) * 2 = 22
  auto host_acc = buf.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 22);
  }

  return 0;
}
