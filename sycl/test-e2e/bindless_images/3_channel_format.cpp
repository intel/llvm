// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_kernel;

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<float> expected(width);
  std::vector<sycl::float3> dataIn(width);
  float exp = 512;
  for (int i = 0; i < width; i++) {
    expected[i] = exp;
    dataIn[i] = sycl::float3(exp, width, i);
  }

  try {
    // Main point of this test is to check creating an image
    // with a 3-channel format
    syclexp::image_descriptor desc({width}, 3, sycl::image_channel_type::fp32);

    syclexp::image_mem imgMem(desc, dev, ctxt);

    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    // Some backends don't support 3-channel formats
    // We still try to create the image,
    // but we expect it to fail with UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
    syclexp::unsampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem, desc, dev, ctxt);

    sycl::buffer<float> buf(out.data(), width);

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor outAcc{buf, cgh};

      cgh.parallel_for<image_kernel>(width, [=](sycl::id<1> id) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        // This shouldn't be hit anyway since CUDA doesn't support
        // 3-channel formats, but we need to ensure the kernel can compile
        using pixel_t = sycl::float4;
#else
        using pixel_t = sycl::float3;
#endif
        auto pixel = syclexp::fetch_image<pixel_t>(imgHandle, int(id[0]));
        outAcc[id] = pixel[0];
      });
    });
    q.wait_and_throw();

  } catch (const sycl::exception &ex) {
    const std::string_view errMsg(ex.what());
    if (ctxt.get_backend() == sycl::backend::ext_oneapi_cuda) {
      if (errMsg.find("UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT") !=
          std::string::npos) {
        std::cout << "CUDA doesn't support 3-channel formats, test passed."
                  << std::endl;
        return 0;
      }
    }
    std::cerr << "Unexpected SYCL exception: " << errMsg << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

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
