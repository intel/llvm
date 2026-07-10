// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Queue-returned events compatibility

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q;
  std::vector<int> data(N, 1);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  auto event = q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class QueueEvent>(sycl::range<1>(N),
                                       [=](sycl::id<1> idx) { acc[idx] = 42; });
  });

  // Can use with extension functions
  syclex::enqueue_wait_event(q, event);
  q.wait();

  return 0;
}
