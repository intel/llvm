// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

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
  size_t width = 4;
  size_t height = 6;
  size_t depth = 8;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[i + width * (j + height * k)] = i + width * (j + height * k);
        dataIn[i + width * (j + height * k)] = {i + width * (j + height * k), 0,
                                                0, 0};
      }
    }
  }

  try {
    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, 4, sycl::image_channel_type::fp32);

    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: allocate memory on device
    sycl::ext::oneapi::experimental::image_mem imgMem(desc, dev, ctxt);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem, samp, desc, dev,
                                                      ctxt);

    sycl::buffer<float, 3> buf((float *)out.data(),
                               sycl::range<3>{depth, height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<3>{depth, height, width});

      cgh.parallel_for<image_addition>(
          sycl::nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](sycl::nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;
            float fdim2 = float(dim2 + 0.5f) / (float)depth;

            // Extension: sample image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::sample_image<sycl::float4>(
                    imgHandle, sycl::float3(fdim0, fdim1, fdim2));

            outAcc[sycl::id<3>{dim2, dim1, dim0}] = px1[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle, dev, ctxt);
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
