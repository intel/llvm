// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

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
  std::vector<sycl::half4> dataIn1(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = j + (height * i);
      dataIn1[j + (height * i)] = {j + (height * i), 0, 0, 0};
    }
  }

  try {
    sycl::sampler samp1(sycl::coordinate_normalization_mode::normalized,
                        sycl::addressing_mode::repeat,
                        sycl::filtering_mode::linear);

    unsigned int element_size_bytes = sizeof(sycl::half) * 4;
    size_t width_in_bytes = width * element_size_bytes;
    size_t pitch = 0;
    
    // Extension: returns the device pointer to USM allocated pitched memory
    auto img_mem_0 = sycl::ext::oneapi::experimental::pitched_alloc_device(
        &pitch, width_in_bytes, height, element_size_bytes, q);

    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp16);

    if (img_mem_0 == nullptr) {
      std::cout << "Error allocating images!" << std::endl;
      return 1;
    }

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), img_mem_0, desc, pitch);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(ctxt, img_mem_0, pitch,
                                                      samp1, desc);

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
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;

            // Extension: read image data from handle
            sycl::half4 px1 =
                sycl::ext::oneapi::experimental::read_image<sycl::half4>(
                    imgHandle1, sycl::float2(fdim0, fdim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle1);
    sycl::free(img_mem_0, ctxt);
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
