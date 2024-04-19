// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that the queue constructor throws a sycl::exception if the
// device selected by the provided selector is not in the specified context.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <optional>

std::optional<sycl::device>
FindOtherDevice(const sycl::device &ExcludedDevice) {
  for (sycl::device Device : sycl::device::get_devices())
    if (Device != ExcludedDevice)
      return Device;
  return std::nullopt;
}

int main() {
  const sycl::device SelectedDevice(sycl::default_selector_v);
  std::optional<sycl::device> OtherDevice = FindOtherDevice(SelectedDevice);

  if (!OtherDevice.has_value()) {
    std::cout << "No other device found. Skipping test." << std::endl;
    return 0;
  }

  sycl::context Ctx(*OtherDevice);
  try {
    sycl::queue q(Ctx, sycl::default_selector_v);
    std::cout << "Queue constructor did not throw." << std::endl;
    return 1;
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid &&
           "Error code should be sycl::errc::invalid.");
  }
  return 0;
}
