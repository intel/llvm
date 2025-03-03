// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  std::cerr << "[SYCL] Init device" << std::endl;
  sycl::device dev;
  std::cerr << "[SYCL] Init queue" << std::endl;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  std::cerr << "[SYCL] Init local data" << std::endl;
  // declare image data
  size_t height = 32;
  size_t width = 32;
  size_t N = height * width;
  std::vector<sycl::float4> out(N);
  std::vector<float> expected(N);
  std::cerr << "[SYCL] Init SYCL data" << std::endl;
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = j * 3;
      dataIn1[i + (width * j)] = {j, j, j, j};
      dataIn2[i + (width * j)] = {j * 2, j * 2, j * 2, j * 2};
    }
  }

  std::cerr << "[SYCL] Test begin" << std::endl;
  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 4, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    // Input images memory
    std::cerr << "[SYCL] Image mem 0" << std::endl;
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);
    std::cerr << "[SYCL] Image mem 1" << std::endl;
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, dev, ctxt);

    // Output image memory
    std::cerr << "[SYCL] Image mem 2 output" << std::endl;
    sycl::ext::oneapi::experimental::image_mem imgMem2(desc, dev, ctxt);

    // Extension: copy over data to device
    std::cerr << "[SYCL] Copying data to device(1)" << std::endl;
    std::cerr << "[SYCL] Src: " << dataIn1.data() << " Handle: " << reinterpret_cast<void *>(imgMem0.get_handle().raw_handle) << std::endl;
    q.ext_oneapi_copy(dataIn1.data(), imgMem0.get_handle(), desc);
    std::cerr << "[SYCL] Copying data to device(2)" << std::endl;
    std::cerr << "[SYCL] Src: " << dataIn2.data() << " Handle: " << reinterpret_cast<void *>(imgMem1.get_handle().raw_handle) << std::endl;
    q.ext_oneapi_copy(dataIn2.data(), imgMem1.get_handle(), desc);
    std::cerr << "[SYCL] Wait" << std::endl;
    q.wait_and_throw();

    std::cout << __FUNCTION__ << ":dataIn1[ 50]:" << dataIn1[ 50][0] << std::endl;
    std::cout << __FUNCTION__ << ":dataIn1[150]:" << dataIn1[150][0] << std::endl;
    std::cout << __FUNCTION__ << ":dataIn1[250]:" << dataIn1[250][0] << std::endl;

    // Extension: create the image and return the handle
    std::cerr << "[SYCL] Create image 1" << std::endl;
    sycl::ext::oneapi::experimental::unsampled_image_handle imgIn1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, dev, ctxt);
    std::cerr << "[SYCL] Create image 2" << std::endl;
    sycl::ext::oneapi::experimental::unsampled_image_handle imgIn2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, dev, ctxt);
    std::cerr << "[SYCL] Create image out" << std::endl;
    sycl::ext::oneapi::experimental::unsampled_image_handle imgOut =
        sycl::ext::oneapi::experimental::create_image(imgMem2, desc, dev, ctxt);

    std::cerr << "[SYCL] Submit kernel" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      sycl::stream os(1024, 128, cgh);
      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            // Extension: fetch image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgIn1, sycl::int2(dim0, dim1));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgIn2, sycl::int2(dim0, dim1));
            os << "px1: " << px1 << " px2: " << px2 << sycl::endl;
            sum = px1[0] + px2[0];

            // Extension: write to image with handle
            sycl::ext::oneapi::experimental::write_image<sycl::float4>(
                imgOut, sycl::int2(dim0, dim1), sycl::float4(sum));
          });
    });
    std::cerr << "[SYCL] Wait 2" << std::endl;
    q.wait_and_throw();

    // Extension: copy data from device to host (handler variant)
    std::cerr << "[SYCL] Submit copy" << std::endl;
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(imgMem2.get_handle(), out.data(), desc);
    });

    std::cerr << "[SYCL] Wait 3" << std::endl;
    q.wait_and_throw();
    std::cout << __FUNCTION__ << ":out[ 50]:" << out[ 50][0] << std::endl;
    std::cout << __FUNCTION__ << ":out[150]:" << out[150][0] << std::endl;
    std::cout << __FUNCTION__ << ":out[250]:" << out[250][0] << std::endl;

    // Extension: cleanup
    std::cerr << "[SYCL] cleanup 1" << std::endl;
    sycl::ext::oneapi::experimental::destroy_image_handle(imgIn1, dev, ctxt);
    std::cerr << "[SYCL] cleanup 2" << std::endl;
    sycl::ext::oneapi::experimental::destroy_image_handle(imgIn2, dev, ctxt);
    std::cerr << "[SYCL] cleanup out" << std::endl;
    sycl::ext::oneapi::experimental::destroy_image_handle(imgOut, dev, ctxt);
    std::cerr << "[SYCL] cleanup done" << std::endl;

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }
  std::cerr << "[SYCL] validate" << std::endl;
  // collect and validate output
  bool validated = true;
  for (int i = 0; i < N; i++) {
    bool mismatch = false;
    if (out[i][0] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
    std::cerr << "[SYCL] Result mismatch! Expected: " << expected[i] 
                << ", Actual: " << out[i][0] << "at i = " << i << std::endl;
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i][0] << std::endl;
#else
      break;
#endif
    }
  }
  if (validated) {
    std::cerr << "[SYCL] Test passed!" << std::endl;
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "[SYCL] Test failed!" << std::endl;
  std::cout << "Test failed!" << std::endl;
  return 3;
}
