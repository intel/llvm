// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/half_type.hpp>

int main() {

  sycl::device dev;

  try {
    // Extension: get pitch alignment information from device -- device info
    // Make sure our pitch alignment queries work properly
    // These can be different depending on the device so we cannot test that the
    // values are correct
    // But we should at least see that the query itself works

    size_t pitchAlign = 0;
    size_t maxPitch = 0;
    size_t maxWidth = 0;
    size_t maxHeight = 0;

    pitchAlign = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_row_pitch_align>();
    std::cout << "Image row pitch alignment = " << pitchAlign << std::endl;
    maxPitch = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                max_image_linear_row_pitch>();
    std::cout << "Max image linear row pitch = " << maxPitch << std::endl;
    maxWidth = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                max_image_linear_width>();
    std::cout << "Max image linear width = " << maxWidth << std::endl;
    maxHeight = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                 max_image_linear_height>();
    std::cout << "Max image linear height = " << maxHeight << std::endl;

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }
  std::cout << "Test passed!" << std::endl;

  return 0;
}
