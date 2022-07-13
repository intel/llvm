// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Unsupported hip call on AMD
// UNSUPPORTED: hip_amd

// Test image-specific printers of the Plugin Interace
//
// CHECK: ---> piMemImageCreate(
// CHECK:   image_desc w/h/d : 4 / 4 / 1  --  arrSz/row/slice : 0 / 64 / 256  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4337
// CHECK: ---> piEnqueueMemImageRead(
// CHECK:   pi_image_offset x/y/z : 0/0/0
// CHECK:   pi_image_region width/height/depth : 4/4/1

#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

int main() {
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> ImgSize(4, 4);

  std::vector<sycl::float4> ImgHostData(ImgSize.size(), {1, 2, 3, 4});

  {
    sycl::image<2> Img(ImgHostData.data(), ChanOrder, ChanType, ImgSize);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.single_task<class EmptyTask>([=]() {});
    });
  }
  return 0;
}
