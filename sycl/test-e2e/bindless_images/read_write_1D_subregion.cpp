// REQUIRES: aspect-ext_oneapi_bindless_images

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Undetermined issue in 'create_image' in this test.

// Test flakily failing on Windows+ BMG
// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17439

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);

  // declare image data
  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<float> expected(width);
  std::vector<float> dataIn1(width);
  std::vector<float> dataIn2(width);
  for (int i = 0; i < width; i++) {
    expected[i] = i * 3;
    dataIn1[i] = i;
    dataIn2[i] = i * 2;
  }

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, 1, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem00(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem2(desc, q);

    // We're able to use move semantics
    auto imgMem0 = std::move(imgMem00);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle3 =
        sycl::ext::oneapi::experimental::create_image(imgMem2, desc, q);

    // Extension: copy over data to device (2 subregions)
    sycl::range copySrcOffset = {0, 0, 0};
    sycl::range copyExtent1 = {width / 2, 1, 1};
    sycl::range srcExtent = {width, 0, 0};

    q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, srcExtent,
                      imgMem0.get_handle(), {0, 0, 0}, desc, copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, 0, 0}, srcExtent,
                      imgMem0.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent1);

    q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, srcExtent,
                      imgMem1.get_handle(), {0, 0, 0}, desc, copyExtent1);
    q.ext_oneapi_copy(dataIn2.data(), {width / 2, 0, 0}, srcExtent,
                      imgMem1.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent1);

    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            float sum = 0;
            // Extension: fetch image data from handle
            float px1 = sycl::ext::oneapi::experimental::fetch_image<float>(
                imgHandle1, int(dim0));
            float px2 = sycl::ext::oneapi::experimental::fetch_image<float>(
                imgHandle2, int(dim0));

            sum = px1 + px2;
            sycl::ext::oneapi::experimental::write_image<float>(imgHandle3,
                                                                int(dim0), sum);
          });
    });

    q.wait_and_throw();

    // Extension: copy data from device to host (two sub-regions)
    sycl::range copyExtent2 = {width / 2, 1, 1};
    sycl::range destExtent = {width, 0, 0};
    q.ext_oneapi_copy(imgMem2.get_handle(), {0, 0, 0}, desc, out.data(),
                      {0, 0, 0}, destExtent, copyExtent2);
    q.ext_oneapi_copy(imgMem2.get_handle(), {width / 2, 0, 0}, desc, out.data(),
                      {width / 2, 0, 0}, destExtent, copyExtent2);
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle3, q);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < width; i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
#else
      break;
#endif
    }
  }
  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cout << "Test failed!" << std::endl;
  return 3;
}
