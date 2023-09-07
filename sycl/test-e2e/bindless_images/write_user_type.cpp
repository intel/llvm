// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

struct my_float4 {
  float x, y, z, w;
};

int main() {
  // Define and populate array for the output data
  sycl::float4 dest[256];
  for (int i = 0; i < 256; i++) {
    dest[i] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  try {
    sycl::queue myQueue;

    namespace syclexp = sycl::ext::oneapi::experimental;

    syclexp::image_descriptor descOut(sycl::range<2>{16, 16},
                                      sycl::image_channel_order::rgba,
                                      sycl::image_channel_type::fp32);

    syclexp::image_mem imgMemoryOut(descOut, myQueue);

    syclexp::unsampled_image_handle imgOut =
        syclexp::create_image(imgMemoryOut, descOut, myQueue);

    myQueue.ext_oneapi_copy(dest, imgMemoryOut.get_handle(), descOut);
    myQueue.wait_and_throw();

    myQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range{16, 16}, [=](sycl::id<2> id) {
        sycl::int2 coords = sycl::int2(id[0], id[1]);

        my_float4 my_pixel = {1.0f, 1.0f, 1.0f, 1.0f};
        syclexp::write_image(imgOut, coords, my_pixel);
      });
    });
    myQueue.wait_and_throw();

    myQueue.ext_oneapi_copy(imgMemoryOut.get_handle(), dest, descOut);
    myQueue.wait_and_throw();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  float expected = 1.0f;
  bool validated = true;
  bool mismatch = false;
  for (int i = 0; i < 256; ++i) {
    int j = 0;
    for (; j < 4; ++j) {
      if (dest[i][j] != expected) {
        validated = false;
        mismatch = true;
        break;
      }
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected << ", Actual[" << i
                << "][" << j << "]: " << dest[i][j] << std::endl;
#else
      break;
#endif
    }
  }

  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed!" << std::endl;
    return 3;
  }
}
