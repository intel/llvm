#include <gtest/gtest.h>

#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

TEST(ImageDescriptorSRGBTest, SRGBRequiresFourChannels) {
  EXPECT_THROW(
      syclexp::image_descriptor(
          {4, 4}, 3, sycl::image_channel_type::ext_oneapi_unorm_int8_srgb),
      sycl::exception);
}

TEST(ImageDescriptorSRGBTest, SRGBWithFourChannelsSucceeds) {
  EXPECT_NO_THROW(syclexp::image_descriptor(
      {4, 4}, 4, sycl::image_channel_type::ext_oneapi_unorm_int8_srgb));
}

TEST(ImageDescriptorSRGBTest, LinearDoesNotRequireFourChannels) {
  EXPECT_NO_THROW(syclexp::image_descriptor(
      {4, 4}, 3, sycl::image_channel_type::unorm_int8));
}

TEST(ImageDescriptorSRGBTest, SRGBMipLevelKeepsChannelType) {
  syclexp::image_descriptor Desc(
      {16, 16}, 4, sycl::image_channel_type::ext_oneapi_unorm_int8_srgb,
      syclexp::image_type::mipmap, 2);
  EXPECT_EQ(Desc.get_mip_level_desc(1).channel_type,
            sycl::image_channel_type::ext_oneapi_unorm_int8_srgb);
}
