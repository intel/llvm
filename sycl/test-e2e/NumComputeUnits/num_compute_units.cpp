// This test checks whether the number of compute units in the device descriptor
// returns a valid value.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {

  sycl::queue Queue;
  sycl::device Device = Queue.get_device();

  size_t NumberComputeUnits =
      Device.get_info<sycl::ext::oneapi::info::device::num_compute_units>();

  assert(NumberComputeUnits >= 1 &&
         "The minimum value for number of compute units in the device is 1");

  return 0;
}