#include <gtest/gtest.h>

#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES

TEST(ImageDescriptorSRGBTest, SRGBRequiresFourChannels) {
  EXPECT_THROW(syclexp::image_descriptor({4, 4}, 3,
                                         sycl::image_channel_type::unorm_int8,
                                         syclexp::image_color_space::srgb),
               sycl::exception);
}

TEST(ImageDescriptorSRGBTest, SRGBRequiresUnormInt8) {
  EXPECT_THROW(syclexp::image_descriptor({4, 4}, 4,
                                         sycl::image_channel_type::fp32,
                                         syclexp::image_color_space::srgb),
               sycl::exception);
}

TEST(ImageDescriptorSRGBTest, SRGBWithFourChannelsAndUnormInt8Succeeds) {
  EXPECT_NO_THROW(
      syclexp::image_descriptor({4, 4}, 4, sycl::image_channel_type::unorm_int8,
                                syclexp::image_color_space::srgb));
}

TEST(ImageDescriptorSRGBTest, LinearDoesNotRequireFourChannels) {
  EXPECT_NO_THROW(
      syclexp::image_descriptor({4, 4}, 3, sycl::image_channel_type::unorm_int8,
                                syclexp::image_color_space::linear));
}

#endif // __INTEL_PREVIEW_BREAKING_CHANGES