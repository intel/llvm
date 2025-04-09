// Test to isolate sycl::vec bug due to use of std::array in
// the constructor.
// REQUIRES: windows

// RUN: not %clangxx -O0 -fsycl -D_DEBUG %s &> %t.compile.log
// RUN: FileCheck %s -input-file=%t.compile.log

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

int main(int argc, char *argv[]) {
  sycl::queue q;

  bindless_image_sampler samp(sycl::addressing_mode::repeat,
                              sycl::coordinate_normalization_mode::normalized,
                              sycl::filtering_mode::linear);

  image_descriptor desc(sycl::range<2>{16, 16}, 2,
                        sycl::image_channel_type::fp32);
  image_mem imgMemoryIn2(desc, q);
  auto sampledImgIn = create_image(imgMemoryIn2, samp, desc, q);

  // CHECK: UnsupportedVarArgFunction: Variadic functions other than 'printf' are not supported in SPIR-V.
  // CHECK: clang{{.*}} error: llvm-spirv command failed with exit code 23{{.*}}
  q.submit([&](sycl::handler &cg) {
     cg.parallel_for(
         sycl::nd_range<3>(sycl::range(256, 256, 256), sycl::range(32, 32, 32)),
         [=](sycl::nd_item<3> item) {
           [[maybe_unused]] auto val =
               sample_image<unsigned char>(sampledImgIn, sycl::float2(1, 2));
         });
   }).wait_and_throw();

  return 0;
}
