// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

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
        {width}, 4, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, dev, ctxt);

    // std::hash specialization to ensure `image_mem` follows common reference
    // semantics
    assert(std::hash<sycl::ext::oneapi::experimental::image_mem>{}(imgMem0) !=
           std::hash<sycl::ext::oneapi::experimental::image_mem>{}(imgMem1));

    // We're able to use move semantics
    // Move construct
    sycl::ext::oneapi::experimental::image_mem imgMem0MoveConstruct(
        std::move(imgMem0));
    // Move assign
    sycl::ext::oneapi::experimental::image_mem imgMem0MoveAssign;
    imgMem0MoveAssign = std::move(imgMem0MoveConstruct);

    // We're able to use copy semantics
    // Copy construct
    sycl::ext::oneapi::experimental::image_mem imgMem1CopyConstruct(imgMem1);
    // Copy assign
    sycl::ext::oneapi::experimental::image_mem imgMem1CopyAssign;
    imgMem1CopyAssign = imgMem1CopyConstruct;

    // Equality operators to ensure `image_mem` follows common reference
    // semantics
    assert(imgMem0MoveAssign != imgMem1CopyAssign);
    assert(imgMem1 == imgMem1CopyAssign);

    // We can default construct image handles
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1;

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle tmpHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem0MoveAssign, desc,
                                                      dev, ctxt);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1CopyAssign, desc,
                                                      dev, ctxt);

    // Default constructed image handles are not valid until we assign a valid
    // raw handle to the struct
    imgHandle1.raw_handle = tmpHandle.raw_handle;

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), imgMem0MoveAssign.get_handle(), desc);
    q.ext_oneapi_copy(dataIn2.data(), imgMem1CopyAssign.get_handle(), desc);

    q.wait_and_throw();

    sycl::buffer<float, 1> buf((float *)out.data(), width);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<image_addition>(width, [=](sycl::id<1> id) {
        float sum = 0;
        // Extension: fetch image data from handle
        sycl::float4 px1 =
            sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                imgHandle1, int(id[0]));
        sycl::float4 px2 =
            sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
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
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
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
  return 3;
}
