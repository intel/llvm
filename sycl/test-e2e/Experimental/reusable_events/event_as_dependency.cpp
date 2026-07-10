// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Event as dependency via handler::depends_on

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  auto event = syclex::make_event(q.get_context());

  std::vector<int> data(N, 5);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  // First kernel
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class DepKernel1>(sycl::range<1>(N),
                                       [=](sycl::id<1> idx) { acc[idx] = 10; });
  });
  syclex::enqueue_signal_event(q, event);

  // Second kernel depends on event
  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(event);
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class DepKernel2>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] + 7; });
  });

  q.wait();

  // Verify: 10 + 7 = 17
  auto host_acc = buf.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 17);
  }

  return 0;
}
