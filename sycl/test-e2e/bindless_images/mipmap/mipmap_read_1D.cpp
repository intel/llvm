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
  constexpr size_t N = 15;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N / 2);
  std::vector<sycl::float4> copyOut(N / 2);

  for (int i = 0; i < N; i++) {
    // Populate input data (to-be mipmap image layers)
    dataIn1[i] = sycl::float4(i, i, i, i);
    if (i < (N / 2)) {
      dataIn2[i] = sycl::float4(i + 10, i + 10, i + 10, i + 10);
      copyOut[i] = sycl::float4{0, 0, 0, 0};
    }

    // Calculate expected output data
    float norm_coord = ((i + 0.5f) / (float)N);
    int x = norm_coord * (N >> 1);
    expected[i] = dataIn1[i][0] + dataIn2[x][0];
  }

  try {

    size_t width = N;
    unsigned int numLevels = 2;

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32,
        sycl::ext::oneapi::experimental::image_type::mipmap, numLevels);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mipMem(desc, dev, ctxt);

    // Extension: retrieve level 0
    sycl::ext::oneapi::experimental::image_mem_handle imgMem1 =
        mipMem.get_mip_level_mem_handle(0);

    // Extension: copy over data to device at level 0
    q.ext_oneapi_copy(dataIn1.data(), imgMem1, desc);

    // Extension: copy data to device at level 1
    q.ext_oneapi_copy(dataIn2.data(), mipMem.get_mip_level_mem_handle(1),
                      desc.get_mip_level_desc(1));
    q.wait_and_throw();

    // Extension: define a sampler object -- extended mipmap attributes
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::mirrored_repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest, sycl::filtering_mode::nearest, 0.0f,
        (float)numLevels, 8.0f);

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(mipMem, samp, desc, dev,
                                                      ctxt);

    sycl::buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](sycl::id<1> id) {
        float sum = 0;
        float x = float(id[0] + 0.5) / (float)N;
        // Extension: read mipmap level 0 with anisotropic filtering and level 1
        // with LOD
        sycl::float4 px1 =
            sycl::ext::oneapi::experimental::read_image<sycl::float4>(mipHandle,
                                                                      x, 0.0f);
        sycl::float4 px2 =
            sycl::ext::oneapi::experimental::read_image<sycl::float4>(mipHandle,
                                                                      x, 1.0f);

        sum = px1[0] + px2[0];
        outAcc[id] = sum;
      });
    });

    q.wait_and_throw();

    // Extension: copy data from device
    q.ext_oneapi_copy(mipMem.get_mip_level_mem_handle(1), copyOut.data(),
                      desc.get_mip_level_desc(1));
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
