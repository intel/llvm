// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: ls %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseExternalAllocatorForSshAndDsh=1 %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
#define VERBOSE_PRINT

class image_addition;

int main() {
  std::cout << "LINE: " << __LINE__ << "\n";
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  std::cout << "LINE: " << __LINE__ << "\n";

  // declare image data
  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<float> expected(width);
  std::vector<sycl::float4> dataIn1(width);
  std::vector<sycl::float4> dataIn2(width);
  float exp = 512;
  for (int i = 0; i < width; i++) {
    expected[i] = exp;
    dataIn1[i] = sycl::float4(i, i, i, i);
    dataIn2[i] = sycl::float4(width - i, width - i, width - i, width - i);
  }
  std::cout << "LINE: " << __LINE__ << "\n";

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, 4, sycl::image_channel_type::fp32);
    std::cout << "LINE: " << __LINE__ << "\n";
    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, dev, ctxt);
    std::cout << "LINE: " << __LINE__ << "\n";

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }
  std::cout << "LINE: " << __LINE__ << "\n";

  return 0;
}
