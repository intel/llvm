// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/sycl.hpp>

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
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = (i + (width * j)) * 3;
      dataIn1[i + (width * j)] = {(i + (width * j)), 0, 0, 0};
      dataIn2[i + (width * j)] = {(i + (width * j)) * 2, 0, 0, 0};
    }
  }

  try {
    sycl::ext::oneapi::experimental::bindless_image_sampler tempSamp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Default constructible sampler
    sycl::ext::oneapi::experimental::bindless_image_sampler samp;

    // Sampler follows by-value semantics
    sycl::ext::oneapi::experimental::bindless_image_sampler tempSamp2(tempSamp);
    samp = tempSamp2;

    // Extension: image descriptor -- can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32);
    size_t pitch = 0;

    // Extension: returns the device pointer to USM allocated pitched memory
    auto imgMemUSM0 =
        sycl::ext::oneapi::experimental::pitched_alloc_device(&pitch, desc, q);

    if (imgMemUSM0 == nullptr) {
      std::cout << "Error allocating images!" << std::endl;
      return 1;
    }

    // Extension: allocate memory on device
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);

    // Extension: copy over data to device for USM image (handler variant)
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(dataIn1.data(), imgMemUSM0, desc, pitch);
    });

    // Extension: copy over data to device for non-USM image
    q.ext_oneapi_copy(dataIn2.data(), imgMem0.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the images and return the handles
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(imgMemUSM0, pitch, samp,
                                                      desc, dev, ctxt);
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, samp, desc, dev,
                                                      ctxt);

    sycl::buffer<float, 2> buf((float *)out.data(),
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
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::sample_image<sycl::float4>(
                    imgHandle1, sycl::float2(fdim0, fdim1));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::sample_image<sycl::float4>(
                    imgHandle2, sycl::float2(fdim0, fdim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1[0] + px2[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, dev,
                                                          ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, dev,
                                                          ctxt);
    sycl::free(imgMemUSM0, ctxt);
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
