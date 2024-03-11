// RUN: %clangxx -S -emit-llvm -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o - | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>

// CHECK: spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10image_readEE
// CHECK: tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4
using namespace sycl::ext::oneapi::experimental;
class image_read;
int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<sycl::float4> dataIn1(width);
  for (int i = 0; i < width; i++) {
    dataIn1[i] = sycl::float4(i, i, i, i);
  }

  {
    image_descriptor desc({width}, sycl::image_channel_order::rgba,
                          sycl::image_channel_type::fp32);

    image_mem imgMem0(desc, dev, ctxt);
    unsampled_image_handle imgHandle1 = create_image(imgMem0, desc, dev, ctxt);

    q.ext_oneapi_copy(dataIn1.data(), imgMem0.get_handle(), desc);
    q.wait_and_throw();

    sycl::buffer<float, 1> buf((float *)out.data(), width);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<image_read>(width, [=](sycl::id<1> id) {
        sycl::float4 px1 = fetch_image<sycl::float4>(imgHandle1, int(id[0]));
        outAcc[id] = px1[0];
      });
    });

    q.wait_and_throw();
    destroy_image_handle(imgHandle1, dev, ctxt);
  }

  for (int i = 0; i < width; i++) {
    std::cout << "Actual: " << out[i] << std::endl;
  }
  return 0;
}
