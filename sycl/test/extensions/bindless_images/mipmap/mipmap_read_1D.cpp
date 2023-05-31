// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
class image_addition;

int main() {

  device dev;
  queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t N = 16;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N / 2);
  std::vector<float4> copyOut(N / 2);
  int j = 0;
  for (int i = 0; i < N; i++) {
    expected[i] = i + (j + 10);
    if (i % 2)
      j++;
    dataIn1[i] = float4(i, i, i, i);
    if (i < (N / 2)) {
      dataIn2[i] = float4(i + 10, i + 10, i + 10, i + 10);
      copyOut[i] = float4{0, 0, 0, 0};
    }
  }

  try {

    size_t width = N;
    unsigned int num_levels = 2;

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, image_channel_order::rgba, image_channel_type::fp32,
        sycl::ext::oneapi::experimental::image_type::mipmap, num_levels);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mip_mem(ctxt, desc);

    // Extension: retrieve level 0
    sycl::ext::oneapi::experimental::image_mem_handle img_mem1 =
        mip_mem.get_mip_level(0);

    // Extension: copy over data to device at level 0
    q.ext_oneapi_copy(dataIn1.data(), img_mem1, desc);

    // Extension: copy data to device at level 1 -- copy func handles desc
    // sizing
    unsigned int level = 1;
    q.ext_oneapi_copy(dataIn2.data(), mip_mem, desc, level);
    q.wait_and_throw();

    // Extension: define a sampler object -- extended mipmap attributes
    sampler samp(coordinate_normalization_mode::normalized,
                 addressing_mode::mirrored_repeat, filtering_mode::nearest,
                 mipmap_filtering_mode::nearest, 0.0f, (float)num_levels, 8.0f);

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(ctxt, mip_mem, samp,
                                                      desc);

    buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        float sum = 0;
        float x = float(id[0] + 0.5) / (float)N;
        // Extension: read mipmap level 0 with anisotropic filtering and level 1
        // with LOD
        float4 px1 = sycl::ext::oneapi::experimental::read_image<float4>(
            mipHandle, x, 0.0f, 0.0f);
        float4 px2 = sycl::ext::oneapi::experimental::read_image<float4>(
            mipHandle, x, 1.0f);

        sum = px1[0] + px2[0];
        outAcc[id] = sum;
      });
    });

    q.wait_and_throw();
    // Extension: copy data from device -- copy func handles sizing
    q.ext_oneapi_copy(mip_mem, copyOut.data(), desc, level);
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, mipHandle);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
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
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
    }
  }
  if (validated) {
    std::cout << "Correct output!" << std::endl;
  }

  return 0;
}
