// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Cross-queue dependency with enqueue_signal_event and enqueue_wait_event

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

  std::vector<int> data1(N, 1);
  std::vector<int> data2(N, 2);
  std::vector<int> result(N, 0);

  sycl::buffer<int> buf1(data1.data(), sycl::range<1>(N));
  sycl::buffer<int> buf2(data2.data(), sycl::range<1>(N));
  sycl::buffer<int> buf_result(result.data(), sycl::range<1>(N));

  auto event = syclex::make_event(ctxt);

  // Launch kernel on q1
  q1.submit([&](sycl::handler &cgh) {
    auto acc1 = buf1.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class KernelQ1>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { acc1[idx] = acc1[idx] * 2; });
  });

  // Signal event when q1 work completes
  syclex::enqueue_signal_event(q1, event);

  // Wait on q2 for q1 to complete
  syclex::enqueue_wait_event(q2, event);

  // Launch kernel on q2
  q2.submit([&](sycl::handler &cgh) {
    auto acc1 = buf1.get_access<sycl::access::mode::read>(cgh);
    auto acc2 = buf2.get_access<sycl::access::mode::read>(cgh);
    auto acc_result = buf_result.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class KernelQ2>(sycl::range<1>(N), [=](sycl::id<1> idx) {
      acc_result[idx] = acc1[idx] + acc2[idx];
    });
  });

  // Reassociate event with q2 completion
  syclex::enqueue_signal_event(q2, event);

  event.wait();

  // Verify results: (1 * 2) + 2 = 4
  auto host_acc = buf_result.get_host_access();
  for (size_t i = 0; i < N; ++i) {
    assert(host_acc[i] == 4);
  }

  return 0;
}
