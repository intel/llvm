// REQUIRES: opencl

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that the Unified Runtime "UR_RESULT_ERROR_UNSUPPORTED_FEATURE" error
// code is passed up to the SYCL runtime and is handled appropriately, when an
// entry-point is not implemented in a given adapter.
// IMPORTANT: This test should be updated if the feature used for testing later
// receives support - use another unsupported feature instead
// Currently using "piextMemImageAllocate"

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  sycl::device Device;
  sycl::queue Queue(Device);
  sycl::context Context = Queue.get_context();

  sycl::ext::oneapi::experimental::image_descriptor Descriptor(
      {0}, sycl::image_channel_order::rgba, sycl::image_channel_type::fp32);

  bool Success = false;

  try {
    sycl::ext::oneapi::experimental::image_mem imgMem0(Descriptor, Device,
                                                       Context);
  } catch (sycl::exception &e) {
    if (e.code() == sycl::errc::feature_not_supported) {
      Success = true;
    }
  }

  // We want this test to succeed by "failing" and specifically catching a
  // "sycl::errc::feature_not_supported" exception.
  assert(Success);

  return 0;
}
