// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the different styles of reading when templating with
// image_channel_order works correctly.

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 20;
  std::vector<sycl::vec<float, 4>> out(width);
  std::vector<sycl::vec<float, 4>> expected(width);
  std::vector<sycl::vec<float, 4>> dataIn1(width);
  float exp = 20;
  for (int i = 0; i < width; i++) {
    expected[i] = sycl::vec<float, 4>(i, i + 1, i + 2, i + 3);
    dataIn1[i] = sycl::vec<float, 4>(i, i + 1, i + 2, i + 3);
  }

  try {
    // Extension: image descriptor - can use the same for both images.
    syclexp::image_descriptor desc({width}, 4, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle.
    syclexp::image_mem imgMem0(desc, dev, ctxt);

    // We can default construct image handles.
    syclexp::unsampled_image_handle imgHandle1 =
        syclexp::create_image(imgMem0, desc, dev, ctxt);

    // Extension: copy over data to device.
    q.ext_oneapi_copy(dataIn1.data(), imgMem0.get_handle(), desc);

    q.wait_and_throw();

    sycl::buffer<sycl::vec<float, 4>, 1> buf((sycl::vec<float, 4> *)out.data(),
                                             width);

    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<image_addition>(width, [=](sycl::id<1> id) {
        // Read from an image using the four different templating styles.

        // Only DataT template argument. Returns data in RGBA channel order.
        sycl::vec<float, 4> px1 =
            syclexp::fetch_image<sycl::vec<float, 4>>(imgHandle1, int(id[0]));

        // DataT and channel_order template arguments. Returns data in ABGR
        // channel order.
        sycl::vec<float, 4> px2 =
            syclexp::fetch_image<sycl::vec<float, 4>,
                                 sycl::image_channel_order::abgr>(imgHandle1,
                                                                  int(id[0]));

        // DataT and HintT template arguments. Returns data in RGBA channel
        // order.
        sycl::vec<float, 4> px3 =
            syclexp::fetch_image<sycl::vec<float, 4>, sycl::vec<float, 4>>(
                imgHandle1, int(id[0]));

        // DataT, HintT and channel_order template arguments. Returns data in
        // ABGR channel order.
        sycl::vec<float, 4> px4 =
            syclexp::fetch_image<sycl::vec<float, 4>, sycl::vec<float, 4>,
                                 sycl::image_channel_order::abgr>(imgHandle1,
                                                                  int(id[0]));

        // Manually shuffle the data back into RGBA channel order.
        outAcc[id[0]][0] = px1[0];
        outAcc[id[0]][1] = px2[2];
        outAcc[id[0]][2] = px3[2];
        outAcc[id[0]][3] = px4[0];
      });
    });

    q.wait_and_throw();

    // Extension: cleanup.
    syclexp::destroy_image_handle(imgHandle1, dev, ctxt);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // Collect and validate output.
  bool validated = true;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < 4; ++j) {
      bool mismatch = false;
      if (out[i][j] != expected[i][j]) {
        mismatch = true;
        validated = false;
      }

      if (mismatch) {
#ifdef VERBOSE_PRINT
        std::cout << "Result mismatch! Expected: " << expected[i][j]
                  << ", Actual: " << out[i][j] << std::endl;
#else
        break;
#endif
      }
    }
  }
  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cout << "Test failed!" << std::endl;
  return 3;
}
