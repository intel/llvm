// RUN: %{build} -o %t.out
// There is an issue with reported device time for the L0 backend, works only on
// pvc for now. No such problems for other backends.
// RUN: %if (!level_zero || arch-intel_gpu_pvc) %{ %{run} %t.out %}

// Check that submission time is calculated properly.

// Test fails on hip flakily, disable temprorarily.
// UNSUPPORTED: hip

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

int main(void) {
  constexpr size_t n = 16;
  sycl::queue q({sycl::property::queue::enable_profiling{}});
  int *data = sycl::malloc_host<int>(n, q);
  int *dest = sycl::malloc_host<int>(n, q);

  for (int i = 0; i < 5; i++) {
    auto event = q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class KernelTime>(
          sycl::range<1>(n), [=](sycl::id<1> idx) { data[idx] = idx; });
    });

    event.wait();
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    // Print for debugging
    std::cout << "Kernel Event - Submit: " << submit_time
              << ", Start: " << start_time << ", End: " << end_time
              << std::endl;

    assert(submit_time != 0 && "Submit time should not be zero");
    assert((submit_time <= start_time) && (start_time <= end_time));
  }

  // All shortcut memory operations use queue_impl::submitMemOpHelper.
  // This test covers memcpy as a representative, extend if other operations
  // diverge.
  uint64_t memcpy_submit_time = 0;
  uint64_t memcpy_start_time = 0;
  uint64_t memcpy_end_time = 0;
  for (int i = 0; i < 5; i++) {
    auto memcpy_event = q.memcpy(dest, data, sizeof(int) * n);
    memcpy_event.wait();

    auto memcpy_submit_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto memcpy_start_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto memcpy_end_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    // Print for debugging
    std::cout << "Memcpy Event - Submit: " << memcpy_submit_time
              << ", Start: " << memcpy_start_time
              << ", End: " << memcpy_end_time << std::endl;

    assert(memcpy_submit_time != 0 && "Submit time should not be zero");
    assert((memcpy_submit_time <= memcpy_start_time) &&
           (memcpy_start_time <= memcpy_end_time));
  }

  sycl::free(data, q);
  sycl::free(dest, q);

  // Check that host_task profiling timestamps share the same base as device
  // tasks.
  auto host_task_event =
      q.submit([&](sycl::handler &cgh) { cgh.host_task([=]() {}); });
  q.wait();
  auto device_task_event = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class DeviceTask>(sycl::range<1>(1), [=](sycl::id<1>) {});
  });

  const uint64_t host_submitted = host_task_event.template get_profiling_info<
      sycl::info::event_profiling::command_submit>();
  const uint64_t host_start = host_task_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const uint64_t host_end = host_task_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  std::cout << "Submit time: " << host_submitted
            << ", Start time: " << host_start << ", End time: " << host_end
            << std::endl;

  const uint64_t device_submitted =
      device_task_event.template get_profiling_info<
          sycl::info::event_profiling::command_submit>();
  const uint64_t device_start = device_task_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const uint64_t device_end = device_task_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  std::cout << "Device Submit time: " << device_submitted
            << ", Device Start time: " << device_start
            << ", Device End time: " << device_end << std::endl;
  assert(host_submitted >= memcpy_submit_time &&
         host_submitted <= device_submitted &&
         "Host and device submit expected to share the same base");
  assert(host_start >= memcpy_start_time && host_start <= device_start &&
         "Host and device start expected to share the same base");
  assert(host_end >= memcpy_end_time && host_end <= device_end &&
         "Host and device end expected to share the same base");
  return 0;
}
