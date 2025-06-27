// Test to isolate sycl::vec bug due to use of std::array in
// the constructor.
// REQUIRES: windows && debug_sycl_library

// RUN: %clangxx -O0 %fsycl -D_DEBUG -shared %s -nostdlib -Xclang --dependent-lib=msvcrtd -fms-runtime-lib=dll_dbg

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

auto Reproducer(sycl::queue q, sampled_image_handle imgHanlde) {
  return q.submit([&](sycl::handler &cg) {
    cg.parallel_for(
        sycl::nd_range<3>({1, 1, 1}, {1, 1, 1}), [=](sycl::nd_item<3> item) {
          [[maybe_unused]] auto val =
              sample_image<unsigned char>(imgHanlde, sycl::float2(1, 2));
        });
  });
}
