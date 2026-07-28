// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: In-order queue signal behavior

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  auto event = syclex::make_event(q.get_context());

  std::vector<int> data(N, 0);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  // Submit multiple operations
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class InOrderOp1>(sycl::range<1>(N),
                                       [=](sycl::id<1> idx) { acc[idx] = 1; });
  });

  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class InOrderOp2>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] + 2; });
  });

  // Signal after all operations
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // All operations should be complete
  auto host_acc = buf.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 3);
  }

  return 0;
}
