// REQUIRES: cuda,aspect-ext_oneapi_cubemap
// REQUIRES: aspect-ext_oneapi_cubemap_seamless_filtering

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../user_types/user_types_common.hpp"
#include <iostream>

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

class sample_cubemap;

int main() {

  namespace syclexp = sycl::ext::oneapi::experimental;

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Declare image data.
  // Width and height must be equal.
  size_t width = 8;
  size_t height = 8;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn(N * 6);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < 6; k++) {
        dataIn[i + width * (j + height * k)] = {i + width * (j + height * k), 0,
                                                0, 0};
      }
    }
  }

  int j = 0;
  for (int i = N - 1; i >= 0; i--) {
    expected[j] = static_cast<float>(i);
    j++;
  }

  // Extension: image descriptor - Cubemap.
  syclexp::image_descriptor desc({width, height}, 4,
                                 sycl::image_channel_type::fp32,
                                 syclexp::image_type::cubemap, 1, 6);

  syclexp::bindless_image_sampler samp(
      sycl::addressing_mode::clamp_to_edge,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::nearest, syclexp::cubemap_filtering_mode::seamless);

  try {
    // Extension: allocate memory on device and create the handle.
    syclexp::image_mem imgMem(desc, dev, ctxt);

    // Extension: create the image and return the handle.
    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(imgMem, samp, desc, dev, ctxt);

    // Extension: copy over data to device (handler variant).
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    });
    q.wait_and_throw();

    sycl::buffer<float, 2> buf((float *)out.data(),
                               sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      // Emanating vector scans one face.
      cgh.parallel_for<sample_cubemap>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Direction Vector
            // x -- largest magnitude
            // y -- shifted between [-0.99, 0.99] + offset
            // z -- shifted between [-0.99, 0.99] + offset
            //
            // [-0.99, 0.99] -- maintains x as largest magnitude
            //
            // 4 elems == [-1, -0.5, 0, 0.5] -- need offset to bring uniformity
            // +0.25 = [-0.75, -0.25, 0.25, 0.75]
            float fdim0 = 1.f;
            float fdim1 = (((float(dim0) / (float)width) * 1.98) - 0.99) +
                          (1.f / (float)width);
            float fdim2 = (((float(dim1) / (float)height) * 1.98) - 0.99) +
                          (1.f / (float)height);

            my_float4 myPixel{};

            // Extension: sample cubemap data from handle.
            myPixel = syclexp::sample_cubemap<my_float4, sycl::float4>(
                imgHandle, sycl::float3(fdim0, fdim1, fdim2));

            outAcc[sycl::id<2>{dim0, dim1}] = myPixel.x;
          });
    });
    q.wait_and_throw();

    // Extension: cleanup.
    syclexp::destroy_image_handle(imgHandle, dev, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // Collect and validate output.
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
