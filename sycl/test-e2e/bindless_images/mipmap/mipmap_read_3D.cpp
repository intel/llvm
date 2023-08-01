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
  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[i + width * (j + height * k)] = i + width * (j + height * k);
        dataIn1[i + width * (j + height * k)] = {i + width * (j + height * k),
                                                 0, 0, 0};
      }
    }
  }
  for (int i = 0; i < (N / 8); i++) {
    dataIn2[i] = i;
  }

  try {

    // Extension: image descriptor -- number of levels
    unsigned int numLevels = 2;
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32,
        sycl::ext::oneapi::experimental::image_type::mipmap, numLevels);

    // Extension: define a sampler object -- extended mipmap attributes
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest, sycl::filtering_mode::nearest, 0.0f,
        (float)numLevels, 8.0f);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mipMem(desc, dev, ctxt);

    // Extension: copy data to device levels 0 and 1
    q.ext_oneapi_copy(dataIn1.data(), mipMem.get_mip_level_mem_handle(0),
                      desc.get_mip_level_desc(0));
    q.ext_oneapi_copy(dataIn2.data(), mipMem.get_mip_level_mem_handle(1),
                      desc.get_mip_level_desc(1));
    q.wait();

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(mipMem, samp, desc, dev,
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
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;
            float fdim2 = float(dim2 + 0.5) / (float)depth;

            // Extension: read mipmap with anisotropic filtering with zero
            // viewing gradients
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                    mipHandle, sycl::float4(fdim0, fdim1, fdim2, (float)0),
                    sycl::float4(0.0f, 0.0f, 0.0f, 0.0f),
                    sycl::float4(0.0f, 0.0f, 0.0f, 0.0f));

            outAcc[sycl::id<3>{dim2, dim1, dim0}] = px1[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(mipHandle, dev, ctxt);

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
