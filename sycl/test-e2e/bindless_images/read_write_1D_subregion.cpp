// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<float> expected(width);
  std::vector<float> dataIn1(width);
  std::vector<float> dataIn2(width);
  for (int i = 0; i < width; i++) {
    expected[i] = i * 3;
    dataIn1[i] = i;
    dataIn2[i] = i * 2;
  }

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::r, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem img_mem_00(desc, q);
    sycl::ext::oneapi::experimental::image_mem img_mem_1(desc, q);
    sycl::ext::oneapi::experimental::image_mem img_mem_2(desc, q);

    // We're able to use move semantics
    auto img_mem_0 = std::move(img_mem_00);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(img_mem_0, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(img_mem_1, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle3 =
        sycl::ext::oneapi::experimental::create_image(img_mem_2, desc, q);

    // Extension: copy over data to device (2 subregions)
    sycl::range copySrcOffset = {0, 0, 0};
    sycl::range copyExtent = {width / 2, 1, 1};
    sycl::range srcExtent = {width, 0, 0};

    q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, srcExtent,
                      img_mem_0.get_handle(), {0, 0, 0}, desc, copyExtent);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, 0, 0}, srcExtent,
                      img_mem_0.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent);

    q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, srcExtent,
                      img_mem_1.get_handle(), {0, 0, 0}, desc, copyExtent);
    q.ext_oneapi_copy(dataIn2.data(), {width / 2, 0, 0}, srcExtent,
                      img_mem_1.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent);

    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(width, [=](sycl::id<1> id) {
        float sum = 0;
        // Extension: read image data from handle
        float px1 = sycl::ext::oneapi::experimental::read_image<float>(
            imgHandle1, int(id[0]));
        float px2 = sycl::ext::oneapi::experimental::read_image<float>(
            imgHandle2, int(id[0]));

        sum = px1 + px2;
        sycl::ext::oneapi::experimental::write_image<float>(imgHandle3,
                                                            int(id[0]), sum);
      });
    });

    q.wait_and_throw();

    // Extension: copy data from device to host (two sub-regions)
    sycl::range copy_extent_2 = {width / 2, 1, 1};
    sycl::range dest_extent_0 = {width, 0, 0};
    q.ext_oneapi_copy(img_mem_2.get_handle(), {0, 0, 0}, desc, out.data(),
                      {0, 0, 0}, dest_extent_0, copy_extent_2);
    q.ext_oneapi_copy(img_mem_2.get_handle(), {width / 2, 0, 0}, desc,
                      out.data(), {width / 2, 0, 0}, dest_extent_0,
                      copy_extent_2);
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, q);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < width; i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i][0] << std::endl;
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
  return 1;
}
