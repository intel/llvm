// REQUIRES: level_zero, level_zero_dev_kit, aspect-ext_intel_legacy_image
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// spir-v gen for legacy images at O0 not working
// UNSUPPORTED: O0

// This test verifies that make_image is working for 1D, 2D and 3D images.
// We instantiate an image with L0, set its body, then use a host accessor to
// verify that the pixels are set correctly.

// clang++ -fsycl -o ilzi.bin -I$SYCL_HOME/build/install/include/sycl
// -lze_loader interop-level-zero-image.cpp

#include <level_zero/ze_api.h>
#include <sycl/accessor_image.hpp>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  constexpr auto BE = sycl::backend::ext_oneapi_level_zero;

  platform Plt{gpu_selector_v};

  auto Devices = Plt.get_devices();

  if (Devices.size() < 1) {
    std::cout << "Devices not found" << std::endl;
    return 0;
  }

  device Device = Devices[0];
  context Context{Device};
  queue Queue{Context, Device};

  // Get native Level Zero handles
  auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

  // -----------  Image Fundamentals
  using pixelT = sycl::uint4;        // accessor
  using ChannelDataT = std::uint8_t; // allocator
  sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  sycl::image_channel_type ChanType = sycl::image_channel_type::unsigned_int8;
  constexpr uint32_t numChannels = 4; // L0 only supports RGBA at this time.

  constexpr uint32_t width = 8;
  constexpr uint32_t height = 4;
  constexpr uint32_t depth = 2;

  const sycl::range<1> ImgRange_1D(width);
  const sycl::range<2> ImgRange_2D(width, height);
  const sycl::range<3> ImgRange_3D(width, height, depth);

  // ----------- Basic LevelZero Description
  ze_image_format_type_t ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
  size_t ZeImageFormatTypeSize = 8;
  ze_image_format_layout_t ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
  ze_image_format_t ZeFormatDesc = {
      ZeImageFormatLayout,       ZeImageFormatType,
      ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
      ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};

  ze_image_desc_t ZeImageDesc_base;
  ZeImageDesc_base.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
  ZeImageDesc_base.pNext = nullptr;
  ZeImageDesc_base.flags = ZE_IMAGE_FLAG_KERNEL_WRITE;
  // ZeImageDesc_base.flags = 0;  // <-- for read only
  ZeImageDesc_base.arraylevels = 0;
  ZeImageDesc_base.miplevels = 0;
  ZeImageDesc_base.format = ZeFormatDesc;

  // ------ 1D ------
  {
    std::cout << "glorious 1D" << std::endl;
    // 1D image
    ze_image_desc_t ZeImageDesc_1D = ZeImageDesc_base;
    ZeImageDesc_1D.type = ZE_IMAGE_TYPE_1D;
    ZeImageDesc_1D.width = width;
    ZeImageDesc_1D.height = 1;
    ZeImageDesc_1D.depth = 1;

    ze_image_handle_t ZeHImage_1D;
    zeImageCreate(ZeContext, ZeDevice, &ZeImageDesc_1D, &ZeHImage_1D);

    { // closure
      sycl::backend_input_t<BE, sycl::image<1>> ImageInteropInput_1D{
          ZeHImage_1D, ChanOrder, ChanType, ImgRange_1D,
          sycl::ext::oneapi::level_zero::ownership::keep};
      auto Image_1D = sycl::make_image<BE, 1>(ImageInteropInput_1D, Context);

      Queue.submit([&](sycl::handler &cgh) {
        auto write_acc =
            Image_1D.get_access<pixelT, sycl::access::mode::write>(cgh);

        cgh.parallel_for(ImgRange_1D, [=](sycl::item<1> Item) {
          int x = Item[0];
          const pixelT somePixel = {x, x, x, x};
          write_acc.write(x, somePixel);
        });
      });
      Queue.wait_and_throw();

      // now check with host accessor.
      auto read_acc = Image_1D.get_access<pixelT, access::mode::read>();
      for (int col = 0; col < width; col++) {
        const pixelT somePixel = read_acc.read(col);
        // const pixelT expectedPixel = {col,col,col,col};
        // assert(somePixel == expectedPixel);
        assert(somePixel[0] == col && somePixel[1] == col &&
               somePixel[2] == col && somePixel[3] == col);
      }

    } // ~image
  }   // closure

  {
    // ------ 2D ------
    std::cout << "glorious 2D" << std::endl;
    // 2D image
    ze_image_desc_t ZeImageDesc_2D = ZeImageDesc_base;
    ZeImageDesc_2D.type = ZE_IMAGE_TYPE_2D;
    ZeImageDesc_2D.width = width;
    ZeImageDesc_2D.height = height;
    ZeImageDesc_2D.depth = 1;

    ze_image_handle_t ZeHImage_2D;
    zeImageCreate(ZeContext, ZeDevice, &ZeImageDesc_2D, &ZeHImage_2D);

    { // closure
      sycl::backend_input_t<BE, sycl::image<2>> ImageInteropInput_2D{
          ZeHImage_2D, ChanOrder, ChanType, ImgRange_2D,
          sycl::ext::oneapi::level_zero::ownership::keep};
      auto Image_2D = sycl::make_image<BE, 2>(ImageInteropInput_2D, Context);

      Queue.submit([&](sycl::handler &cgh) {
        auto write_acc =
            Image_2D.get_access<pixelT, sycl::access::mode::write>(cgh);

        cgh.parallel_for(ImgRange_2D, [=](sycl::item<2> Item) {
          auto location = sycl::int2{Item[0], Item[1]};
          auto sum = Item[0] + Item[1];
          const pixelT somepixel = {sum, sum, sum, sum};
          write_acc.write(location, somepixel);
        });
      });
      Queue.wait_and_throw();

      // now check with host accessor.
      auto read_acc = Image_2D.get_access<pixelT, access::mode::read>();
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          auto location = sycl::int2{col, row};
          const pixelT somePixel = read_acc.read(location);
          auto sum = col + row;
          // const pixelT expectedPixel = {sum,sum,sum,sum};
          // assert(somePixel == expectedPixel);
          assert(somePixel[0] == sum && somePixel[1] == sum &&
                 somePixel[2] == sum && somePixel[3] == sum);
        }
      }

    } // ~image
  }   // closure

  {
    // ------ 3D ------
    std::cout << "glorious 3D" << std::endl;
    // 3D image
    ze_image_desc_t ZeImageDesc_3D = ZeImageDesc_base;
    ZeImageDesc_3D.type = ZE_IMAGE_TYPE_3D;
    ZeImageDesc_3D.width = width;
    ZeImageDesc_3D.height = height;
    ZeImageDesc_3D.depth = depth;

    ze_image_handle_t ZeHImage_3D;
    zeImageCreate(ZeContext, ZeDevice, &ZeImageDesc_3D, &ZeHImage_3D);

    { // closure
      sycl::backend_input_t<BE, sycl::image<3>> ImageInteropInput_3D{
          ZeHImage_3D, ChanOrder, ChanType, ImgRange_3D,
          sycl::ext::oneapi::level_zero::ownership::keep};
      auto Image_3D = sycl::make_image<BE, 3>(ImageInteropInput_3D, Context);

      Queue.submit([&](sycl::handler &cgh) {
        auto write_acc =
            Image_3D.get_access<pixelT, sycl::access::mode::write>(cgh);

        cgh.parallel_for(ImgRange_3D, [=](sycl::item<3> Item) {
          auto location = sycl::int4{Item[0], Item[1], Item[2], 0};
          auto sum = Item[0] + Item[1] + Item[2];
          const pixelT somepixel = {sum, sum, sum, sum};
          write_acc.write(location, somepixel);
        });
      });
      Queue.wait_and_throw();

      // now check with host accessor.
      auto read_acc = Image_3D.get_access<pixelT, access::mode::read>();
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          for (int z = 0; z < depth; z++) {
            auto location = sycl::int4{col, row, z, 0};
            const pixelT somePixel = read_acc.read(location);
            auto sum = col + row + z;
            // const pixelT expectedPixel = {sum,sum,sum,sum};
            // assert(somePixel == expectedPixel);
            assert(somePixel[0] == sum && somePixel[1] == sum &&
                   somePixel[2] == sum && somePixel[3] == sum);
          }
        }
      }

    } // ~image
  }   // closure

#else
  std::cout << "Missing  Level-Zero backend. Test skipped." << std::endl;
#endif
  return 0;
}
