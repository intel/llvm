// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: enqueue_wait_events with multiple events

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::device dev;
  sycl::context ctxt = dev.get_platform().khr_get_default_context();
  sycl::queue q1{ctxt, dev, sycl::property::queue::in_order{}};
  sycl::queue q2{ctxt, dev, sycl::property::queue::in_order{}};
  sycl::queue q3{ctxt, dev, sycl::property::queue::in_order{}};

  std::vector<int> data1(N, 1);
  std::vector<int> data2(N, 2);
  std::vector<int> result(N, 0);

  sycl::buffer<int> buf1(data1.data(), sycl::range<1>(N));
  sycl::buffer<int> buf2(data2.data(), sycl::range<1>(N));
  sycl::buffer<int> buf_result(result.data(), sycl::range<1>(N));

  auto event1 = syclex::make_event(ctxt);
  auto event2 = syclex::make_event(ctxt);

  // Launch on q1
  q1.submit([&](sycl::handler &cgh) {
    auto acc = buf1.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class MultiQ1>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] * 3; });
  });
  syclex::enqueue_signal_event(q1, event1);

  // Launch on q2
  q2.submit([&](sycl::handler &cgh) {
    auto acc = buf2.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class MultiQ2>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc[idx] = acc[idx] * 5; });
  });
  syclex::enqueue_signal_event(q2, event2);

  // Wait for both events on q3
  std::vector<sycl::event> events{event1, event2};
  syclex::enqueue_wait_events(q3, events);

  // Combine results on q3
  q3.submit([&](sycl::handler &cgh) {
    auto acc1 = buf1.get_access<sycl::access::mode::read>(cgh);
    auto acc2 = buf2.get_access<sycl::access::mode::read>(cgh);
    auto acc_result = buf_result.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class MultiQ3>(sycl::range<1>(N), [=](sycl::id<1> idx) {
      acc_result[idx] = acc1[idx] + acc2[idx];
    });
  });

  q3.wait();

  // Verify: (1 * 3) + (2 * 5) = 3 + 10 = 13
  auto host_acc = buf_result.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 13);
  }

  return 0;
}
