// REQUIRES: cuda
// REQUIRES: aspect-fp16

// RUN: %{build} -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  size_t width = 5;
  size_t height = 6;
  size_t N = width * height;
  std::vector<sycl::half> out(N);
  std::vector<sycl::half> expected(N);
  std::vector<sycl::half4> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = i + (width * j);
      dataIn[i + (width * j)] = {i + (width * j), 0, 0, 0};
    }
  }

  try {
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    unsigned int elementSizebytes = sizeof(sycl::half) * 4;
    size_t widthInBytes = width * elementSizebytes;
    size_t pitch = 0;

    // Extension: returns the device pointer to USM allocated pitched memory
    auto imgMem = sycl::ext::oneapi::experimental::pitched_alloc_device(
        &pitch, widthInBytes, height, elementSizebytes, q);

    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 4, sycl::image_channel_type::fp16);

    if (imgMem == nullptr) {
      std::cout << "Error allocating images!" << std::endl;
      return 1;
    }

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn.data(), imgMem, desc, pitch);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem, pitch, samp, desc,
                                                      dev, ctxt);

    sycl::buffer<sycl::half, 2> buf((sycl::half *)out.data(),
                                    sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;

            // Extension: sample image data from handle
            sycl::half4 px1 =
                sycl::ext::oneapi::experimental::sample_image<sycl::half4>(
                    imgHandle, sycl::float2(fdim0, fdim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle, dev, ctxt);
    sycl::free(imgMem, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < N; i++) {
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
