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
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = (j + (height * i)) * 3;
      dataIn1[j + (height * i)] = {j + (height * i), 0, 0, 0};
      dataIn2[j + (height * i)] = {(j + (height * i)) * 2, 0, 0, 0};
    }
  }

  try {
    sycl::sampler samp1(sycl::coordinate_normalization_mode::normalized,
                        sycl::addressing_mode::repeat, sycl::filtering_mode::linear);

    // Extension: image descriptor -- can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32);
    size_t pitch = 0;

    // Extension: returns the device pointer to USM allocated pitched memory
    auto img_mem_usm_0 =
        sycl::ext::oneapi::experimental::pitched_alloc_device(&pitch, desc, q);

    if (img_mem_usm_0 == nullptr) {
      std::cout << "Error allocating images!" << std::endl;
      return 1;
    }

    // Extension: allocate memory on device
    sycl::ext::oneapi::experimental::image_mem img_mem_0(ctxt, desc);

    // Extension: copy over data to device for USM image (handler variant)
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(dataIn1.data(), img_mem_usm_0, desc, pitch);
    });

    // Extension: copy over data to device for non-USM image
    q.ext_oneapi_copy(dataIn2.data(), img_mem_0, desc);
    q.wait_and_throw();

    // Extension: create the images and return the handles
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(ctxt, img_mem_usm_0,
                                                      pitch, samp1, desc);
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(ctxt, img_mem_0, samp1,
                                                      desc);

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
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;

            // Extension: read image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                    imgHandle1, sycl::float2(fdim0, fdim1));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                    imgHandle2, sycl::float2(fdim0, fdim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1[0] + px2[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle1);
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle2);
    sycl::free(img_mem_usm_0, ctxt);
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
