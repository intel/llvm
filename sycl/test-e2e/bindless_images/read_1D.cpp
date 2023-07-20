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
  std::vector<sycl::float4> dataIn1(width);
  std::vector<sycl::float4> dataIn2(width);
  float exp = 512;
  for (int i = 0; i < width; i++) {
    expected[i] = exp;
    dataIn1[i] = sycl::float4(i, i, i, i);
    dataIn2[i] = sycl::float4(width - i, width - i, width - i, width - i);
  }

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem img_mem_0(desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem img_mem_1(desc, dev, ctxt);

    // std::hash specialization to ensure `image_mem` follows common reference
    // semantics
    assert(std::hash<sycl::ext::oneapi::experimental::image_mem>{}(img_mem_0) !=
           std::hash<sycl::ext::oneapi::experimental::image_mem>{}(img_mem_1));

    // We're able to use move semantics
    // Move construct
    sycl::ext::oneapi::experimental::image_mem img_mem_0_move_construct(
        std::move(img_mem_0));
    // Move assign
    sycl::ext::oneapi::experimental::image_mem img_mem_0_move_assign;
    img_mem_0_move_assign = std::move(img_mem_0_move_construct);

    // We're able to use copy semantics
    // Copy construct
    sycl::ext::oneapi::experimental::image_mem img_mem_1_copy_construct(
        img_mem_1);
    // Copy assign
    sycl::ext::oneapi::experimental::image_mem img_mem_1_copy_assign;
    img_mem_1_copy_assign = img_mem_1_copy_construct;

    // Equality operators to ensure `image_mem` follows common reference
    // semantics
    assert(img_mem_0_move_assign != img_mem_1_copy_assign);
    assert(img_mem_1 == img_mem_1_copy_assign);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(img_mem_0_move_assign,
                                                      desc, dev, ctxt);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(img_mem_1_copy_assign,
                                                      desc, dev, ctxt);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), img_mem_0_move_assign.get_handle(), desc);
    q.ext_oneapi_copy(dataIn2.data(), img_mem_1_copy_assign.get_handle(), desc);

    q.wait_and_throw();

    sycl::buffer<float, 1> buf((float *)out.data(), width);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<image_addition>(width, [=](sycl::id<1> id) {
        float sum = 0;
        // Extension: read image data from handle
        sycl::float4 px1 =
            sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                imgHandle1, int(id[0]));
        sycl::float4 px2 =
            sycl::ext::oneapi::experimental::read_image<sycl::float4>(
                imgHandle2, int(id[0]));

        sum = px1[0] + px2[0];
        outAcc[id] = sum;
      });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, dev,
                                                          ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, dev,
                                                          ctxt);
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
  return 1;
}
