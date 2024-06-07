// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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
  size_t width = 2;
  size_t height = 2;
  size_t depth = 2;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        dataIn[i + width * (j + height * k)] = i + width * (j + height * k);
      }
    }
  }

  // Apply the address modes one-by-one as a transform
  // i.e. a' = x(y(z(a)))
  // a = {0,1,2,3,4,5,6,7}
  expected = {6, 7, 4, 5, 6, 7, 4, 5};

  try {

    namespace syclexp = sycl::ext::oneapi::experimental;

    // Extension: image descriptor
    syclexp::image_descriptor desc({width, height, depth}, 1,
                                   sycl::image_channel_type::fp32);

    sycl::addressing_mode addrModes[3];
    addrModes[0] = sycl::addressing_mode::repeat; // repeat in x-dim
    addrModes[1] =
        sycl::addressing_mode::mirrored_repeat; // mirror repeat in y-dim
    addrModes[2] =
        sycl::addressing_mode::clamp_to_edge; // clamp to edge in z-dim
    syclexp::bindless_image_sampler samp(
        addrModes, sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: allocate memory on device
    syclexp::image_mem_handle memHandle =
        syclexp::alloc_image_mem(desc, dev, ctxt);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn.data(), memHandle, desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(memHandle, samp, desc, dev, ctxt);

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

            // Address outside dimension domain
            float fdim0 = float(dim0 + width + 0.5f) / (float)width;
            float fdim1 = float(dim1 + height + 0.5f) / (float)height;
            float fdim2 = float(dim2 + depth + 0.5f) / (float)depth;

            // Extension: sample image data from handle
            float px1 = syclexp::sample_image<float>(
                imgHandle, sycl::float3(fdim0, fdim1, fdim2));

            outAcc[sycl::id<3>{dim2, dim1, dim0}] = px1;
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    syclexp::destroy_image_handle(imgHandle, dev, ctxt);
    syclexp::free_image_mem(memHandle, syclexp::image_type::standard, dev,
                            ctxt);
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
