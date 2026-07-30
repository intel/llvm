// Only L0V2 supports urEnqueueHostTaskExp.
// REQUIRES: level_zero_v2_adapter

// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-INTENDED: UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP is not
// supported on win&gen12.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Event reuse across kernel and native host task operations.
// Uses malloc_device for GPU kernels and malloc_host for the host task.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  auto event = syclex::make_event(q.get_context());

  int *dev_data = sycl::malloc_device<int>(N, q);
  int *host_data = sycl::malloc_host<int>(N, q);

  // First operation: set all device values to 1
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel1>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { dev_data[idx] = 1; });
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Copy device -> host so the host task can access and modify the data
  q.memcpy(host_data, dev_data, N * sizeof(int));
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Reuse event for second operation: native host task adds 10
  syclex::host_task(q, [=] {
    for (size_t i = 0; i < N; ++i)
      host_data[i] += 10;
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Copy host -> device so the final kernel can operate on the updated data
  q.memcpy(dev_data, host_data, N * sizeof(int));
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Reuse event for third operation: multiply by 2 on device
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel2>(
        sycl::range<1>(N), [=](sycl::id<1> idx) { dev_data[idx] *= 2; });
  });
  syclex::enqueue_signal_event(q, event);
  event.wait();

  // Copy result back to host for verification
  q.memcpy(host_data, dev_data, N * sizeof(int));
  q.wait();

  // Verify final result: (1 + 10) * 2 = 22
  for (size_t i = 0; i < N; ++i)
    assert(host_data[i] == 22);

  sycl::free(dev_data, q);
  sycl::free(host_data, q);
  return 0;
}
