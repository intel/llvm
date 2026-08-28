// Tests that SYCL passes `numMipLevel == 0` to the UR/L0 backend for
// non-mipmap images, regardless of the user-supplied `num_levels`.
//
// The Level Zero spec requires `ze_image_desc_t::miplevels` to be 0
// (https://oneapi-src.github.io/level-zero-spec/). SYCL's image_descriptor
// defaults `num_levels` to 1 and used to forward that value verbatim, which
// caused regressions once the L0/LEO stack stopped ignoring the field
// (CMPLRLLVM-75426). Only `image_type::mipmap` should carry a non-zero count.

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

thread_local uint32_t LastNumMipLevel = 0xFFFFFFFFu;
thread_local int AllocateCallCount = 0;
thread_local int UnsampledCreateCallCount = 0;
thread_local int SampledCreateCallCount = 0;

ur_result_t mock_urBindlessImagesImageAllocateExp(void *pParams) {
  ++AllocateCallCount;
  auto &Params =
      *reinterpret_cast<ur_bindless_images_image_allocate_exp_params_t *>(
          pParams);
  LastNumMipLevel = (**Params.ppImageDesc).numMipLevel;
  // Hand back a real dummy handle so the default destroy path can release it.
  **Params.pphImageMem =
      mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urBindlessImagesUnsampledImageCreateExp(void *pParams) {
  ++UnsampledCreateCallCount;
  auto &Params = *reinterpret_cast<
      ur_bindless_images_unsampled_image_create_exp_params_t *>(pParams);
  LastNumMipLevel = (**Params.ppImageDesc).numMipLevel;
  **Params.pphImage = mock::createDummyHandle<ur_exp_image_native_handle_t>();
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urBindlessImagesSampledImageCreateExp(void *pParams) {
  ++SampledCreateCallCount;
  auto &Params =
      *reinterpret_cast<ur_bindless_images_sampled_image_create_exp_params_t *>(
          pParams);
  LastNumMipLevel = (**Params.ppImageDesc).numMipLevel;
  **Params.pphImage = mock::createDummyHandle<ur_exp_image_native_handle_t>();
  return UR_RESULT_SUCCESS;
}

void resetMockState() {
  LastNumMipLevel = 0xFFFFFFFFu;
  AllocateCallCount = 0;
  UnsampledCreateCallCount = 0;
  SampledCreateCallCount = 0;
}

void installMocks() {
  // Default mock-adapter implementations for free/destroy already do the right
  // thing (mock::releaseDummyHandle), so we only need to intercept the create
  // entry points to capture numMipLevel and hand back valid dummy handles.
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesImageAllocateExp",
      &mock_urBindlessImagesImageAllocateExp);
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesUnsampledImageCreateExp",
      &mock_urBindlessImagesUnsampledImageCreateExp);
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesSampledImageCreateExp",
      &mock_urBindlessImagesSampledImageCreateExp);
}

} // namespace

// A standard 2D image must reach UR/L0 with numMipLevel == 0, even though
// image_descriptor defaults num_levels to 1.
TEST(BindlessImagesMipLevels, StandardImageZerosMipLevel) {
  sycl::unittest::UrMock<> Mock;
  installMocks();
  resetMockState();

  sycl::queue Q;

  syclexp::image_descriptor Desc({16, 16}, 4, sycl::image_channel_type::fp32);
  ASSERT_EQ(Desc.num_levels, 1u);

  syclexp::image_mem ImgMem(Desc, Q);
  EXPECT_EQ(AllocateCallCount, 1);
  EXPECT_EQ(LastNumMipLevel, 0u)
      << "Standard image must pass numMipLevel == 0 to UR (was "
      << LastNumMipLevel << ")";

  syclexp::unsampled_image_handle Handle =
      syclexp::create_image(ImgMem, Desc, Q);
  EXPECT_EQ(UnsampledCreateCallCount, 1);
  EXPECT_EQ(LastNumMipLevel, 0u)
      << "Standard image create must pass numMipLevel == 0 to UR (was "
      << LastNumMipLevel << ")";

  syclexp::destroy_image_handle(Handle, Q);
}

// 1D and 3D standard images take the same path; sanity-check both.
TEST(BindlessImagesMipLevels, StandardImage1DAnd3DZeroMipLevel) {
  sycl::unittest::UrMock<> Mock;
  installMocks();
  resetMockState();

  sycl::queue Q;

  syclexp::image_descriptor Desc1D({64}, 1, sycl::image_channel_type::fp32);
  syclexp::image_mem ImgMem1D(Desc1D, Q);
  EXPECT_EQ(LastNumMipLevel, 0u);

  resetMockState();
  syclexp::image_descriptor Desc3D({8, 8, 8}, 4,
                                   sycl::image_channel_type::fp32);
  syclexp::image_mem ImgMem3D(Desc3D, Q);
  EXPECT_EQ(LastNumMipLevel, 0u);
}

// A true mipmap image must forward num_levels unchanged.
TEST(BindlessImagesMipLevels, MipmapImagePreservesMipLevel) {
  sycl::unittest::UrMock<> Mock;
  installMocks();
  resetMockState();

  sycl::queue Q;

  // Mipmaps require num_levels > 1.
  constexpr unsigned int NumLevels = 4;
  syclexp::image_descriptor Desc({16, 16}, 4, sycl::image_channel_type::fp32,
                                 syclexp::image_type::mipmap, NumLevels);

  syclexp::image_mem ImgMem(Desc, Q);
  EXPECT_EQ(AllocateCallCount, 1);
  EXPECT_EQ(LastNumMipLevel, NumLevels)
      << "Mipmap image must forward num_levels to UR (was " << LastNumMipLevel
      << ")";
}

// Sampled image creation goes through populate_ur_structs as well — verify
// the same translation applies on that path.
TEST(BindlessImagesMipLevels, SampledStandardImageZerosMipLevel) {
  sycl::unittest::UrMock<> Mock;
  installMocks();
  resetMockState();

  sycl::queue Q;

  syclexp::image_descriptor Desc({16, 16}, 4, sycl::image_channel_type::fp32);
  syclexp::image_mem ImgMem(Desc, Q);

  syclexp::bindless_image_sampler Sampler{
      sycl::addressing_mode::clamp_to_edge,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::linear};

  resetMockState();
  syclexp::sampled_image_handle Handle =
      syclexp::create_image(ImgMem, Sampler, Desc, Q);
  EXPECT_EQ(SampledCreateCallCount, 1);
  EXPECT_EQ(LastNumMipLevel, 0u)
      << "Sampled standard image must pass numMipLevel == 0 to UR (was "
      << LastNumMipLevel << ")";

  syclexp::destroy_image_handle(Handle, Q);
}
