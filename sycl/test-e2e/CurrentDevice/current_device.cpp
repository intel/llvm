// REQUIRES: any-device-is-cpu, any-device-is-gpu

// RUN: %{build} -o %t.out %threads_lib
// RUN: %{run-unfiltered-devices} %t.out

// E2E tests for sycl_ext_oneapi_current_device

#include <sycl/ext/oneapi/experimental/current_device.hpp>

#include <thread>

void check_get_eq(sycl::device dev) {
  auto device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  if (device != dev)
    assert(false && "check_get_eq failed.");
}

void check_get_ne(sycl::device dev) {
  auto device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  if (device == dev)
    assert(false && "check_get_ne failed.");
}

void check_set_get_eq(sycl::device dev) {
  sycl::ext::oneapi::experimental::this_thread::set_current_device(dev);
  auto device =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  if (device != dev)
    assert(false && "check_set_get_eq failed.");
}

int main() {
  // Test 1
  std::thread t1(check_get_eq, sycl::device{sycl::default_selector_v});
  std::thread t2(check_get_eq, sycl::device{sycl::default_selector_v});

  t1.join();
  t2.join();

  // Test 2
  // As GPU device is required, it is always has higher score than CPU device,
  // so test must not fail.
  t1 = std::thread(check_get_ne, sycl::device{sycl::cpu_selector_v});
  t2 = std::thread(check_get_ne, sycl::device{sycl::cpu_selector_v});

  t1.join();
  t2.join();

  // Test 3
  t1 = std::thread(check_set_get_eq, sycl::device{sycl::cpu_selector_v});
  t2 = std::thread(check_set_get_eq, sycl::device{sycl::gpu_selector_v});

  t1.join();
  t2.join();

  // Test 4
  auto device_1 =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  if (device_1 != sycl::device{sycl::default_selector_v})
    assert(false && "get_current_device check failed.");

  // Test 5
  sycl::ext::oneapi::experimental::this_thread::set_current_device(
      sycl::device{sycl::cpu_selector_v});
  auto device_2 =
      sycl::ext::oneapi::experimental::this_thread::get_current_device();
  if (device_2 != sycl::device{sycl::cpu_selector_v})
    assert(false && "set/get_current_device check failed.");

  return 0;
}
