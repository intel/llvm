#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/queue.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

thread_local size_t ExpectedRowPitch = 0;
thread_local size_t ExpectedSlicePitch = 0;
thread_local uint32_t ExpectedNumSamples = 0;
thread_local int MapExternalArrayCallCounter = 0;

thread_local size_t ImageCopySrcRowPitch = 0;
thread_local size_t ImageCopyDstRowPitch = 0;
thread_local size_t ImageCopyDstSlicePitch = 0;
thread_local int ImageCopyCallCounter = 0;

// -----
// Mocks
// -----
inline ur_result_t urBindlessImagesMapExternalArrayExp_replace(void *pParams) {
  ++MapExternalArrayCallCounter;
  auto Params =
      *reinterpret_cast<ur_bindless_images_map_external_array_exp_params_t *>(
          pParams);

  const ur_image_desc_t *urDesc = *Params.ppImageDesc;

  EXPECT_EQ(urDesc->rowPitch, ExpectedRowPitch);
  EXPECT_EQ(urDesc->slicePitch, ExpectedSlicePitch);
  EXPECT_EQ(urDesc->numSamples, ExpectedNumSamples);

  if (Params.pphImageMem && *Params.pphImageMem) {
    **Params.pphImageMem =
        mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t urBindlessImagesImageCopyExp_replace(void *pParams) {
  ++ImageCopyCallCounter;
  auto Params =
      *reinterpret_cast<ur_bindless_images_image_copy_exp_params_t *>(pParams);

  const ur_image_desc_t *urSrcDesc = *Params.ppSrcImageDesc;
  const ur_image_desc_t *urDstDesc = *Params.ppDstImageDesc;
  ImageCopySrcRowPitch = urSrcDesc->rowPitch;
  ImageCopyDstRowPitch = urDstDesc->rowPitch;
  ImageCopyDstSlicePitch = urDstDesc->slicePitch;

  return UR_RESULT_SUCCESS;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
TEST(BindlessImagesExtensionTests, ImageDescriptorPropagatesLayout) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesMapExternalArrayExp",
      &urBindlessImagesMapExternalArrayExp_replace);

  sycl::queue Q;

  MapExternalArrayCallCounter = 0;
  ExpectedRowPitch = 1024;
  ExpectedSlicePitch = 0;
  ExpectedNumSamples = 4;

  syclexp::image_descriptor Desc(sycl::range<2>{32, 32}, 4,
                                 sycl::image_channel_type::fp32,
                                 syclexp::image_type::standard, 1, 1,
                                 ExpectedNumSamples, ExpectedRowPitch);

  syclexp::external_mem_descriptor<syclexp::resource_fd> ExtMemDesc{
      {123}, syclexp::external_mem_handle_type::opaque_fd, 0};

  auto MemHandle = syclexp::import_external_memory(ExtMemDesc, Q.get_device(),
                                                   Q.get_context());

  try {
    auto ImgHandle = syclexp::map_external_image_memory(MemHandle, Desc, Q);
    (void)ImgHandle;
  } catch (const sycl::exception &e) {
    FAIL() << "Caught unexpected SYCL exception: " << e.what();
  }

  EXPECT_EQ(MapExternalArrayCallCounter, 1);
}

// Regression test: descriptor row_pitch must propagate through the
// ext_oneapi_copy path when the overload has no explicit pitch argument.
// Previously fill_copy_args unconditionally clobbered rowPitch with the
// (derived-from-extent) SrcPitch/DestPitch values, dropping the descriptor's
// pitch.
TEST(BindlessImagesExtensionTests, ImageDescriptorCopyPropagatesPitch) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesImageCopyExp", &urBindlessImagesImageCopyExp_replace);

  sycl::queue Q;

  ImageCopyCallCounter = 0;
  ImageCopySrcRowPitch = 0;
  ImageCopyDstRowPitch = 0;
  ImageCopyDstSlicePitch = 0;

  constexpr size_t DescRowPitch = 4096;
  // 32 * 4 channels * sizeof(float) = 512
  constexpr size_t TightHostPitch = 32 * 4 * sizeof(float);

  syclexp::image_descriptor Desc(
      sycl::range<2>{32, 32}, 4, sycl::image_channel_type::fp32,
      syclexp::image_type::standard, 1, 1, 0, DescRowPitch);

  syclexp::image_mem_handle DstHandle =
      syclexp::alloc_image_mem(Desc, Q.get_device(), Q.get_context());

  std::vector<float> HostSrc(32 * 32 * 4, 0.0f);

  try {
    Q.ext_oneapi_copy(HostSrc.data(), DstHandle, Desc);
    Q.wait();
  } catch (const sycl::exception &e) {
    syclexp::free_image_mem(DstHandle, syclexp::image_type::standard, Q);
    FAIL() << "Caught unexpected SYCL exception: " << e.what();
  }

  EXPECT_EQ(ImageCopyCallCounter, 1);
  // Image side (dst) gets the descriptor's row_pitch.
  EXPECT_EQ(ImageCopyDstRowPitch, DescRowPitch);
  // Memory side (src) must be the tight host stride, NOT the descriptor's
  // row_pitch. UR adapters use pSrcImageDesc->rowPitch as the host stride.
  EXPECT_EQ(ImageCopySrcRowPitch, TightHostPitch);

  syclexp::free_image_mem(DstHandle, syclexp::image_type::standard, Q);
}

TEST(BindlessImagesExtensionTests, ImageDescriptorPitchValidation) {
  // row_pitch on a 1D image is rejected. The 1D ctor doesn't take row_pitch,
  // so we default-construct and set the field directly.
  {
    syclexp::image_descriptor Desc;
    Desc.width = 32;
    Desc.num_channels = 4;
    Desc.channel_type = sycl::image_channel_type::fp32;
    Desc.row_pitch = 256;
    EXPECT_THROW(Desc.verify(), sycl::exception);
  }

  // slice_pitch on a plain 2D standard image is rejected.
  {
    syclexp::image_descriptor Desc;
    Desc.width = 32;
    Desc.height = 32;
    Desc.num_channels = 4;
    Desc.channel_type = sycl::image_channel_type::fp32;
    Desc.slice_pitch = 8192;
    EXPECT_THROW(Desc.verify(), sycl::exception);
  }

  // slice_pitch on a 2D image array is accepted.
  {
    syclexp::image_descriptor Desc;
    Desc.width = 32;
    Desc.height = 32;
    Desc.num_channels = 4;
    Desc.channel_type = sycl::image_channel_type::fp32;
    Desc.type = syclexp::image_type::array;
    Desc.array_size = 4;
    Desc.row_pitch = 512;
    Desc.slice_pitch = 512 * 32;
    EXPECT_NO_THROW(Desc.verify());
  }

  // slice_pitch smaller than row_pitch * height is rejected (3D case).
  {
    syclexp::image_descriptor Desc;
    Desc.width = 16;
    Desc.height = 16;
    Desc.depth = 4;
    Desc.num_channels = 1;
    Desc.channel_type = sycl::image_channel_type::fp32;
    Desc.row_pitch = 512;
    Desc.slice_pitch = 2048; // < 512 * 16 = 8192
    EXPECT_THROW(Desc.verify(), sycl::exception);
  }
}

TEST(BindlessImagesExtensionTests, ImageDescriptorPropagatesSlicePitch) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesMapExternalArrayExp",
      &urBindlessImagesMapExternalArrayExp_replace);

  sycl::queue Q;

  MapExternalArrayCallCounter = 0;
  ExpectedRowPitch = 512;
  // slice_pitch is the byte stride between successive slices; it must be at
  // least row_pitch * height (512 * 16 = 8192) to describe a non-overlapping
  // layout.
  ExpectedSlicePitch = 8192;
  ExpectedNumSamples = 1;

  syclexp::image_descriptor Desc(
      sycl::range<3>{16, 16, 4}, 1, sycl::image_channel_type::fp32,
      syclexp::image_type::standard, 1, 1, ExpectedNumSamples, ExpectedRowPitch,
      ExpectedSlicePitch);

  syclexp::external_mem_descriptor<syclexp::resource_fd> ExtMemDesc{
      {456}, syclexp::external_mem_handle_type::opaque_fd, 0};

  auto MemHandle = syclexp::import_external_memory(ExtMemDesc, Q.get_device(),
                                                   Q.get_context());

  try {
    auto ImgHandle = syclexp::map_external_image_memory(MemHandle, Desc, Q);
    (void)ImgHandle;
  } catch (const sycl::exception &e) {
    FAIL() << "Caught unexpected SYCL exception: " << e.what();
  }

  EXPECT_EQ(MapExternalArrayCallCounter, 1);
}
