// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: gpu-intel-pvc
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop_task for image.
// Level-Zero
#include <iostream>
#include <level_zero/ze_api.h>
// SYCL
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t SIZE = 16;

int main() {
  queue queue{};

  try {
    image<2> image(image_channel_order::rgba, image_channel_type::fp32,
                   {SIZE, SIZE});

    ze_context_handle_t ze_context =
        get_native<backend::ext_oneapi_level_zero>(queue.get_context());

    queue
        .submit([&](handler &cgh) {
          auto image_acc = image.get_access<float4, access::mode::write>(cgh);
          cgh.interop_task([=](const interop_handler &ih) {
            ze_image_handle_t ze_image =
                ih.get_mem<backend::ext_oneapi_level_zero>(image_acc);
            assert(ze_image != nullptr);
          });
        })
        .wait();
  } catch (exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  } catch (const char *msg) {
    std::cout << "Exception caught: " << msg << std::endl;
    return 1;
  }

  return 0;
}
