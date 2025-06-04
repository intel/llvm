// REQUIRES: aspect-ext_oneapi_bindless_images

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Undetermined issue in 'create_image' in this test.

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// This tests that sycl::buffer works with image fetches
// Currently this fails when
// https://github.com/intel/llvm/commit/f9c8c01d38f8fbea81db99ab90b7d0f2bdcc8b4d
// is cherry-picked. See https://github.com/intel/llvm/issues/16503

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 512;
  std::vector<sycl::float4> out(width);
  std::vector<float> expected(width);
  std::vector<sycl::float4> dataIn1(width);
  std::vector<sycl::float4> dataIn2(width);
  float exp = 512;
  for (int i = 0; i < width; i++) {
    expected[i] = exp;
    dataIn1[i] = sycl::float4(i, i, i, i);
    dataIn2[i] = sycl::float4(width - i, width - i, width - i, width - i);
  }

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, 4, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    // Input images memory
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, dev, ctxt);

    // Output image memory
    sycl::ext::oneapi::experimental::image_mem imgMem2(desc, dev, ctxt);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), imgMem0.get_handle(), desc);
    q.ext_oneapi_copy(dataIn2.data(), imgMem1.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgIn1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, dev, ctxt);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgIn2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, dev, ctxt);

    sycl::ext::oneapi::experimental::unsampled_image_handle imgOut =
        sycl::ext::oneapi::experimental::create_image(imgMem2, desc, dev, ctxt);
    sycl::range<1> r(1);
    sycl::buffer<sycl::ext::oneapi::experimental::unsampled_image_handle, 1>
        imgHandlesBuf{&imgIn1, r};
    sycl::buffer<sycl::float4, 1> buf(out.data(), sycl::range<1>{width});
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor imgHandleAcc{imgHandlesBuf, cgh, sycl::read_only};
      sycl::accessor outAcc{buf, cgh, sycl::write_only};
      cgh.parallel_for<image_addition>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            float sum = 0;
            // Extension: fetch image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgHandleAcc[0], int(dim0));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgIn2, int(dim0));

            sum = px1[0] + px2[0];
            // Extension: write to image with handle
            outAcc[dim0][0] = sum;
          });
    });

    q.wait_and_throw();
    // Extension: copy data from device to host
    q.ext_oneapi_copy(imgMem2.get_handle(), out.data(), desc);
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgIn1, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgIn2, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgOut, dev, ctxt);
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
    if (out[i][0] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i][0] << std::endl;
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
