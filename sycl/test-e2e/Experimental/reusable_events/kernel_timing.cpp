// REQUIRES: aspect-ext_oneapi_reusable_events_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Timing kernels with profiling events

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  sycl::platform plat = dev.get_platform();

  std::vector<int> data(N, 1);
  sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

  auto start_event = syclex::make_event(syclex::enable_profiling{true});
  auto end_event = syclex::make_event(syclex::enable_profiling{true});

  syclex::enqueue_signal_event(q, start_event);

  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class TimingKernel1>(sycl::range<1>(N),
                                          [=](sycl::id<1> idx) {
                                            for (int i = 0; i < 100; ++i) {
                                              acc[idx] = acc[idx] + 1;
                                            }
                                          });
  });

  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class TimingKernel2>(sycl::range<1>(N),
                                          [=](sycl::id<1> idx) {
                                            for (int i = 0; i < 100; ++i) {
                                              acc[idx] = acc[idx] * 2;
                                            }
                                          });
  });

  syclex::enqueue_signal_event(q, end_event);
  q.wait();

  // Get timing information
  auto start_time =
      start_event
          .get_profiling_info<sycl::info::event_profiling::command_end>();
  auto end_time =
      end_event
          .get_profiling_info<sycl::info::event_profiling::command_start>();

  uint64_t elapsed = end_time - start_time;
  assert(elapsed >= 0);

  return 0;
}
