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
  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[k + depth * (j + height * i)] = k + depth * (j + height * i);
        dataIn1[k + depth * (j + height * i)] = {k + depth * (j + height * i),
                                                 0, 0, 0};
      }
    }
  }
  for (int i = 0; i < (N / 8); i++) {
    dataIn2[i] = i;
  }

  try {

    // Extension: image descriptor -- number of levels
    unsigned int num_levels = 2;
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, image_channel_order::rgba,
        image_channel_type::fp32,
        sycl::ext::oneapi::experimental::image_type::mipmap, num_levels);

    // Extension: define a sampler object -- extended mipmap attributes
    sampler samp(coordinate_normalization_mode::normalized,
                 addressing_mode::clamp, filtering_mode::nearest,
                 mipmap_filtering_mode::nearest, 0.0f, (float)num_levels, 8.0f);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mip_mem(ctxt, desc);

    // Extension: copy data to device at all levels -- copy func handles desc
    // sizing
    q.ext_oneapi_copy(dataIn1.data(), mip_mem, desc, 0);
    q.ext_oneapi_copy(dataIn2.data(), mip_mem, desc, 1);
    q.wait();

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(ctxt, mip_mem, samp,
                                                      desc);

    buffer<float, 3> buf((float *)out.data(), range<3>{depth, height, width});
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(
          cgh, range<3>{depth, height, width});

      cgh.parallel_for<image_addition>(
          nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;
            float fdim2 = float(dim2 + 0.5) / (float)depth;

            // Extension: read mipmap with anisotropic filtering with zero
            // viewing gradients
            float4 px1 = sycl::ext::oneapi::experimental::read_image<float4>(
                mipHandle, float4(fdim0, fdim1, fdim2, (float)0),
                float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f));

            outAcc[id<3>{dim2, dim1, dim0}] = px1[0];
          });
    });

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
