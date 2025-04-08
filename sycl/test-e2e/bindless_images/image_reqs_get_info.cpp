// REQUIRES: aspect-ext_oneapi_bindless_images

// UNSUPPORTED: level_zero
// UNSUPPORTED-INTENDED: The feature is not implemented in the Level Zero stack.
// https://github.com/intel/llvm/issues/17663

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

int main() {

  sycl::device dev;

  bool validated = true;

  try {
    // Extension: get pitch alignment information from device -- device info
    // Make sure our pitch alignment queries work properly
    // These can be different depending on the device so we cannot test that the
    // values are correct
    // But we should at least see that the query itself works
    auto pitchAlign = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_row_pitch_align>();
    auto maxPitch = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                     max_image_linear_row_pitch>();
    auto maxWidth = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                     max_image_linear_width>();
    auto maxheight = dev.get_info<sycl::ext::oneapi::experimental::info::
                                      device::max_image_linear_height>();

#ifdef VERBOSE_PRINT
    std::cout << "image_row_pitch_align: " << pitchAlign
              << "\nmax_image_linear_row_pitch: " << maxPitch
              << "\nmax_image_linear_width: " << maxWidth
              << "\nmax_image_linear_height: " << maxheight << "\n";
#endif

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  if (validated) {
    std::cout << "Test Passed!\n";
    return 0;
  }

  std::cout << "Test Failed!" << std::endl;
  return 3;
}
