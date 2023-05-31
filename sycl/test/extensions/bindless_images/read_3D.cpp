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
  size_t width = 1024;
  size_t height = 1024;
  size_t depth = 16;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[k + depth * (j + height * i)] = j * 3;
        dataIn1[k + depth * (j + height * i)] = {j, j, j, j};
        dataIn2[k + depth * (j + height * i)] = {j * 2, j * 2, j * 2, j * 2};
      }
    }
  }

  try {

    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem img_mem_0(ctxt, desc);
    sycl::ext::oneapi::experimental::image_mem img_mem_1(ctxt, desc);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), img_mem_0, desc);
    q.ext_oneapi_copy(dataIn2.data(), img_mem_1, desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(ctxt, img_mem_0, desc);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(ctxt, img_mem_1, desc);

    sycl::buffer<float, 3> buf((float *)out.data(),
                               sycl::range<3>{depth, height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<3>{depth, height, width});

      cgh.parallel_for<image_addition>(
          sycl::nd_range<3>{{width, height, depth}, {16, 16, 4}},
          [=](sycl::nd_item<3> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);
            size_t dim2 = it.get_global_id(2);
            float sum = 0;
            // Extension: read image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                    imgHandle1, sycl::int4(dim0, dim1, dim2, 0));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                    imgHandle2, sycl::int4(dim0, dim1, dim2, 0));

            sum = px1[0] + px2[0];
            outAcc[sycl::id<3>{dim2, dim1, dim0}] = sum;
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle1);
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle2);
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
