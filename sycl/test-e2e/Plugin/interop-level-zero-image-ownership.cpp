// REQUIRES: level_zero, level_zero_dev_kit, aspect-ext_intel_legacy_image

// the ze_debug=4 memory check will fail on this test, since it itentionally
// makes an 'unbalanced' create/destroy situation for the test.
// UNSUPPORTED: ze_debug

// spir-v gen for legacy images at O0 not working
// UNSUPPORTED: O0

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// This test verifies that ownership is working correctly.
// If ownership is ::transfer then the ~image destructor will end up calling
// zeImageDestroy
// CHECK: test  ownership::transfer
// CHECK: ZE ---> zeImageDestroy

// With ownership ::keep it is must be called manually.
// CHECK: test  ownership::keep
// CHECK: zeImageDestroy MANUAL

// No other calls should appear.
// CHECK-NOT: zeImageDestroy

// clang++ -fsycl -o wfd.bin -I$SYCL_HOME/build/install/include/sycl -lze_loader
// interop-level-zero-image-ownership.cpp

#include <level_zero/ze_api.h>
#include <sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

void test(sycl::ext::oneapi::level_zero::ownership Ownership) {

  constexpr auto BE = sycl::backend::ext_oneapi_level_zero;

  platform Plt{gpu_selector_v};

  auto Devices = Plt.get_devices();

  if (Devices.size() < 1) {
    std::cout << "Devices not found" << std::endl;
    return;
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
  constexpr uint32_t depth = 1;

  const sycl::range<2> ImgRange_2D(width, height);

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
  // ZeImageDesc_base.flags = 0;
  ZeImageDesc_base.arraylevels = 0;
  ZeImageDesc_base.miplevels = 0;
  ZeImageDesc_base.format = ZeFormatDesc;

  {
    // ------ 2D ------
    ze_image_desc_t ZeImageDesc_2D = ZeImageDesc_base;
    ZeImageDesc_2D.type = ZE_IMAGE_TYPE_2D;
    ZeImageDesc_2D.width = width;
    ZeImageDesc_2D.height = height;
    ZeImageDesc_2D.depth = 1;

    ze_image_handle_t ZeHImage_2D;
    ze_result_t res =
        zeImageCreate(ZeContext, ZeDevice, &ZeImageDesc_2D, &ZeHImage_2D);
    if (res != ZE_RESULT_SUCCESS) {
      std::cout << "unable to create image " << res << std::endl;
      return;
    }

    { // closure
      sycl::backend_input_t<BE, sycl::image<2>> ImageInteropInput_2D{
          ZeHImage_2D, ChanOrder, ChanType, ImgRange_2D, Ownership};
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

    } // ~image
    // if ownership was transfer, then the ZeHImage_2D was destroyed as part of
    // the ~image destruction (or deferred)

    if (Ownership == sycl::ext::oneapi::level_zero::ownership::keep) {
      zeImageDestroy(ZeHImage_2D);
      std::cout << "zeImageDestroy MANUAL" << std::endl;
    }

  } // closure
}

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  std::cout << "test  ownership::transfer" << std::endl;
  test(sycl::ext::oneapi::level_zero::ownership::transfer);

  std::cout << "test  ownership::keep" << std::endl;
  test(sycl::ext::oneapi::level_zero::ownership::keep);
#else
  std::cout << "Missing  Level-Zero backend. Test skipped." << std::endl;
#endif
  std::cout << "chau" << std::endl;
  return 0;
}
