// REQUIRES: level_zero, level_zero_dev_kit, aspect-ext_intel_legacy_image
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s

// spir-v gen for legacy images at O0 not working
// UNSUPPORTED: O0

// we use the interop to get the native image handle and then use that to make a
// new image and enumerate the pixels.

// CHECK:      (0 0) -- { 0 0 0 0 }
// CHECK-NEXT: (1 0) -- { 1 1 1 1 }
// CHECK-NEXT: (2 0) -- { 2 2 2 2 }
// CHECK-NEXT: (3 0) -- { 3 3 3 3 }
// CHECK-NEXT: (0 1) -- { 4 4 4 4 }
// CHECK-NEXT: (1 1) -- { 5 5 5 5 }
// CHECK-NEXT: (2 1) -- { 6 6 6 6 }
// CHECK-NEXT: (3 1) -- { 7 7 7 7 }

// clang++ -fsycl -o las.bin -I$SYCL_HOME/build/install/include/sycl -lze_loader
// interop-level-zero-image-get-native-mem.cpp

#include <level_zero/ze_api.h>
#include <sycl/accessor_image.hpp>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <sycl/stream.hpp>
using namespace sycl;

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  constexpr auto BE = sycl::backend::ext_oneapi_level_zero;
  sycl::device D =
      sycl::ext::oneapi::filter_selector("level_zero:gpu").select_device();

  sycl::context Ctx{D};
  sycl::queue Q(Ctx, D);
  auto ZeContext = sycl::get_native<BE>(Ctx);
  auto ZeDevice = sycl::get_native<BE>(D);

  // -----------  IMAGE STUFF
  using pixelT = sycl::uint4;        // accessor
  using ChannelDataT = std::uint8_t; // allocator
  constexpr long width = 4;
  constexpr long height = 2;
  constexpr long numPixels = width * height;
  ChannelDataT *sourceData =
      (ChannelDataT *)std::calloc(numPixels * 4, sizeof(ChannelDataT));
  // initialize data: [ (0 0 0 0)  (1 1 1 1) ...]
  for (size_t i = 0; i < numPixels; i++) {
    for (size_t chan = 0; chan < 4; chan++) {
      size_t idx = (i * 4) + chan;
      sourceData[idx] = (ChannelDataT)i;
    }
  }
  // 8 bits per channel, four per pixel.
  sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  sycl::image_channel_type ChanType = sycl::image_channel_type::unsigned_int8;

  const sycl::range<2> ImgRange_2D(width, height);
  { // closure
    // 1 - Create simple image.
    sycl::image<2> image_2D(sourceData, ChanOrder, ChanType, ImgRange_2D);

    // 2 - Grab it's image handle via the get_native_mem interop.
    using nativeH = sycl::backend_return_t<BE, sycl::image<2>>;
    sycl::buffer<nativeH, 1> passBack(range<1>{1});

    Q.submit([&](handler &cgh) {
       auto image_acc =
           image_2D.get_access<pixelT, sycl::access::mode::read>(cgh);
       auto passBackAcc = passBack.get_host_access(sycl::write_only);
       cgh.host_task([=](const interop_handle &IH) {
         // There is nothing with image handles in the L0 API except
         // create and destroy. So let's do that.
         auto ZeImageH = IH.get_native_mem<BE>(image_acc);
         passBackAcc[0] = ZeImageH;
       });
     }).wait();

    // Now we have the ZeImageH, so let's make a new SYCL image from it.
    auto passBackAcc = passBack.get_host_access(sycl::read_only);
    nativeH ZeImageH = passBackAcc[0];
    sycl::backend_input_t<BE, sycl::image<2>> imageData{
        ZeImageH, ChanOrder, ChanType, ImgRange_2D,
        sycl::ext::oneapi::level_zero::ownership::keep};
    sycl::image<2> NewImg = sycl::make_image<BE, 2>(imageData, Ctx);

    // Then use that image to read and stream out the data.
    Q.submit([&](handler &cgh) {
       auto read_acc = NewImg.get_access<pixelT, sycl::access::mode::read>(cgh);
       sycl::stream out(2024, 400, cgh);
       cgh.single_task([=]() {
         for (unsigned y = 0; y < height; y++) {
           for (unsigned x = 0; x < width; x++) {
             auto location = sycl::int2{x, y};
             pixelT somePixel = read_acc.read(location);
             out << "(" << x << " " << y << ") -- { " << somePixel[0] << " "
                 << somePixel[1] << " " << somePixel[2] << " " << somePixel[3]
                 << " }" << sycl::endl;
           }
         }
       });
     }).wait();
  } // ~image
  std::free(sourceData);
#else
  std::cout << "Missing  Level-Zero backend. Test skipped." << std::endl;
#endif
  return 0;
}
